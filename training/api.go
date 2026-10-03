package training

// 外部からの訓練データ登録 API。
//
// **学習データは壊れやすい資産**（手でラベル付けした正解であり、一度学習に
// 通すと MergeSamples が畳んだサンプルは履歴を消しても残る）なので、
// 書き込み口を開けるにあたっての約束を 3 つ置いてある。
//
//   - **ループバック以外から触れるのは /api/status と /api/register だけ。**
//     学習 (/api/trainhistory)・削除・セッション系は公開設定に関わらず
//     常にループバック限定。トークンが漏れても被害を「不正なデータが
//     1 件増える」に閉じ込めるため
//   - **ループバックは認証免除。** 同じサーバが配信している UI (static/index.html)
//     が 9 箇所で素の fetch を投げているので、ここを縛ると画面が全滅する。
//     守りたいのは LAN の他人であってローカルのプロセスではない
//   - **API 経由のデータは未確認 (Verified=false) として入る。** 画像と SFEN の
//     対応は機械には検証できないので、解析タブで人が一度見るまで学習に使わせない

import (
	"bytes"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	settingsFile = "data/settings.json"
	// maxHistory は履歴の上限。超えたら古いものを消すのではなく登録を断る
	// （手でラベル付けした局面が押し出されて消えるのを防ぐ）。
	//
	// **200 → 300（2026-09-13）。** 「100 局面あたりで頭打ち」という当初の
	// 見立ては**局面数の話**で、目標（どんな盤でも読む）に効くのは
	// **見た目の種類**のほう（「同じ出所を厚く」の節）。188 件の時点で
	// 残り 12 枚しか無く、**新しい見た目を 1 枚足すたびに枠を気にする**
	// 状態になっていた。上限は「消えると困る資産を守る歯止め」であって
	// 集める上限ではないので、種類が増える限り上げてよい。
	maxHistory = 300
	// maxUploadBytes は /api/register が受け取る画像の上限
	maxUploadBytes = 32 << 20
)

// APISettings は外部公開の設定。data/settings.json に保存する
type APISettings struct {
	// Enabled: /api/register で登録を受け付けるか
	Enabled bool `json:"enabled"`
	// External: ループバック以外からのアクセスを許すか
	External bool `json:"external"`
	// Token: Bearer トークン。空なら要求しない（ループバックには元々要求しない）
	Token string `json:"token"`
}

var (
	settings   APISettings
	settingsMu sync.RWMutex
)

func loadSettings() {
	f, err := os.Open(settingsFile)
	if err != nil {
		return
	}
	defer f.Close()
	var s APISettings
	if err := json.NewDecoder(f).Decode(&s); err != nil {
		logger().Warn("loadSettings", "err", err)
		return
	}
	settingsMu.Lock()
	settings = s
	settingsMu.Unlock()
}

func saveSettings(s APISettings) error {
	settingsMu.Lock()
	settings = s
	settingsMu.Unlock()
	os.MkdirAll(dataDir, 0755)
	return writeJSONFile(settingsFile, s)
}

func currentSettings() APISettings {
	settingsMu.RLock()
	defer settingsMu.RUnlock()
	return settings
}

// writeJSONFile は一時ファイルに書いてから rename する。
// 途中で落ちても既存の内容が消えないようにするため
func writeJSONFile(path string, v interface{}) error {
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if err := json.NewEncoder(f).Encode(v); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, path)
}

// isLoopback はリクエスト元がこのマシン自身かを返す
func isLoopback(r *http.Request) bool {
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		host = r.RemoteAddr
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// externalAllowed はループバック以外から叩いてよいパスかを返す
func externalAllowed(path string) bool {
	return path == "/api/status" || path == "/api/register"
}

// hasValidToken は Authorization: Bearer が設定値と一致するかを返す
func hasValidToken(r *http.Request, token string) bool {
	if token == "" {
		return true
	}
	h := r.Header.Get("Authorization")
	return strings.HasPrefix(h, "Bearer ") && strings.TrimPrefix(h, "Bearer ") == token
}

// withAccessControl はループバック以外からのアクセスを絞る。
// ループバックは素通し（同じサーバが配信している UI が認証を持たないため）
func withAccessControl(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if isLoopback(r) {
			next.ServeHTTP(w, r)
			return
		}
		s := currentSettings()
		// 公開していない・許可されていないパスは、存在自体を伏せる
		if !s.External || !externalAllowed(r.URL.Path) {
			http.NotFound(w, r)
			return
		}
		// status は「今受け付けているか」を知るための口なので、
		// Enabled でなくても・トークンが無くても応答する（内容は絞る）
		if r.URL.Path == "/api/status" {
			next.ServeHTTP(w, r)
			return
		}
		if !s.Enabled {
			httpJSONError(w, http.StatusServiceUnavailable, "API 登録は無効です")
			return
		}
		if !hasValidToken(r, s.Token) {
			w.Header().Set("WWW-Authenticate", "Bearer")
			httpJSONError(w, http.StatusUnauthorized, "トークンが不正です")
			return
		}
		next.ServeHTTP(w, r)
	})
}

func httpJSONError(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]string{"error": msg})
}

// handleStatus は登録を受け付けられる状態かを返す。
// **/api/health ではない。** 「サーバが生きているか」ではなく
// 「今このサーバに送ってよいか」を返すので、監視用の health とは別物
func handleStatus(w http.ResponseWriter, r *http.Request) {
	s := currentSettings()
	resp := map[string]interface{}{"enabled": s.Enabled}
	// 素性を明かすのは、ローカルか正しいトークンを持っている相手だけ
	if isLoopback(r) || hasValidToken(r, s.Token) {
		historyMu.RLock()
		n := len(loadHistory().Entries)
		historyMu.RUnlock()
		resp["external"] = s.External
		resp["auth"] = s.Token != ""
		resp["entries"] = n
		resp["capacity"] = maxHistory
		resp["data_version"] = dataFile
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// handleSettings は API タブの設定を読み書きする（ループバック限定）
func handleSettings(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(currentSettings())
	case http.MethodPost:
		var req struct {
			Enabled  bool `json:"enabled"`
			External bool `json:"external"`
			// UseToken が false ならトークンを消す。true でトークンが
			// 未発行なら発行する。Regenerate は発行し直し
			UseToken   bool `json:"use_token"`
			Regenerate bool `json:"regenerate"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			httpJSONError(w, http.StatusBadRequest, err.Error())
			return
		}
		s := currentSettings()
		s.Enabled = req.Enabled
		s.External = req.External
		switch {
		case !req.UseToken:
			s.Token = ""
		case req.Regenerate || s.Token == "":
			s.Token = newToken()
		}
		if err := saveSettings(s); err != nil {
			httpJSONError(w, http.StatusInternalServerError, err.Error())
			return
		}
		logger().Info("API settings", "enabled", s.Enabled, "external", s.External, "token", s.Token != "")
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(s)
	default:
		http.Error(w, "GET or POST", http.StatusMethodNotAllowed)
	}
}

func newToken() string {
	b := make([]byte, 24)
	if _, err := rand.Read(b); err != nil {
		// crypto/rand が失敗する状況ではトークンを作らせない
		return ""
	}
	return hex.EncodeToString(b)
}

// handleRegister は外部から訓練データ（画像 + 正解SFEN + 盤面座標）を受け取る。
//
// multipart/form-data:
//
//	image          画像ファイル（PNG / JPEG）
//	sfen           正解の SFEN（盤面部分だけでも可）
//	x1,y1,x2,y2    盤面の外枠座標（必須）
//	bounds_by      座標の出どころ（任意。"detect" / "manual"。既定は不明）
//
// **盤面座標は必須。** 座標なしのエントリは samplesFromHistory が
// DetectBoard に頼るうえ shiftAugment のずらしを作らないので、学習価値が
// 1/5 になる。しかもその事実に学習を押すまで気付けない。送り手（ikkyoku）は
// suteme の盤面認識を通した座標を持っているので、必須にして困らない。
//
// **`bounds_by` は名乗らなければ「不明」。自動検出とみなさない。**
// 送り手が人の手で枠を引ける作りなら "manual" を送ること。この値は
// 「その座標を突き合わせに使ってよいか」の判断に効く（自動検出をそのまま
// 送り返した座標は検出器自身の出力なので、突き合わせても自己充足になる）。
//
// **SFEN は検証しない。** 解析タブで人が確認する運びなので、
// ここで弾くと直す機会ごと失う。パースできない SFEN は学習時に
// その局面だけ失敗として報告される
func handleRegister(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxUploadBytes)
	if err := r.ParseMultipartForm(maxUploadBytes); err != nil {
		httpJSONError(w, http.StatusBadRequest, "リクエストを読めません: "+err.Error())
		return
	}

	file, _, err := r.FormFile("image")
	if err != nil {
		httpJSONError(w, http.StatusBadRequest, "image がありません")
		return
	}
	defer file.Close()
	raw, err := io.ReadAll(file)
	if err != nil {
		httpJSONError(w, http.StatusBadRequest, err.Error())
		return
	}
	img, _, err := image.Decode(bytes.NewReader(raw))
	if err != nil {
		httpJSONError(w, http.StatusBadRequest, "画像をデコードできません: "+err.Error())
		return
	}

	sfenStr := strings.TrimSpace(r.FormValue("sfen"))
	if sfenStr == "" {
		httpJSONError(w, http.StatusBadRequest, "sfen がありません")
		return
	}

	bounds, err := formBounds(r, img.Bounds())
	if err != nil {
		httpJSONError(w, http.StatusBadRequest, err.Error())
		return
	}

	// 同じ画像を送り直したとき（リトライ）に増やさない
	sum := sha256.Sum256(raw)
	hash := hex.EncodeToString(sum[:])

	historyMu.Lock()
	defer historyMu.Unlock()
	h := loadHistory()
	for _, e := range h.Entries {
		if e.Hash != "" && e.Hash == hash {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{
				"status": "ok", "id": e.ID, "duplicate": true,
			})
			return
		}
	}
	if len(h.Entries) >= maxHistory {
		// 古い局面を押し出すのではなく断る。手でラベル付けした資産を
		// 外部からの流し込みで失わせない
		httpJSONError(w, http.StatusInsufficientStorage,
			fmt.Sprintf("履歴が上限（%d 件）です。不要な局面を削除してください", maxHistory))
		return
	}

	id := newID()
	os.MkdirAll(dataDir, 0755)
	f, err := os.Create(filepath.Join(dataDir, id+".png"))
	if err != nil {
		httpJSONError(w, http.StatusInternalServerError, err.Error())
		return
	}
	// 履歴の画像は PNG に揃える（JPEG で来ても保存形式は変えない）
	if err := png.Encode(f, img); err != nil {
		f.Close()
		os.Remove(filepath.Join(dataDir, id+".png"))
		httpJSONError(w, http.StatusInternalServerError, err.Error())
		return
	}
	f.Close()

	entry := HistoryEntry{
		ID:          id,
		CreatedAt:   time.Now().Format("01/02 15:04"),
		SFEN:        sfenStr,
		BoardBounds: bounds,
		BoundsBy:    normalizeBoundsBy(r.FormValue("bounds_by")),
		Source:      SourceAPI,
		Verified:    false,
		Hash:        hash,
	}
	h.Entries = append([]HistoryEntry{entry}, h.Entries...)
	saveHistoryFile(h)

	logger().Info("registered", "id", id, "from", r.RemoteAddr, "entries", len(h.Entries))
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status": "ok", "id": id, "verified": false, "entries": len(h.Entries),
	})
}

// formBounds は x1,y1,x2,y2 を読んで検証する
func formBounds(r *http.Request, img image.Rectangle) (*BoardBounds, error) {
	v := make([]int, 4)
	for i, name := range []string{"x1", "y1", "x2", "y2"} {
		s := r.FormValue(name)
		if s == "" {
			return nil, fmt.Errorf("%s がありません（盤面座標は必須です）", name)
		}
		n, err := strconv.Atoi(s)
		if err != nil {
			return nil, fmt.Errorf("%s が数値ではありません", name)
		}
		v[i] = n
	}
	b := &BoardBounds{v[0], v[1], v[2], v[3]}
	// 1マスが 1px 未満になる指定は受け取っても意味が無い
	if b.X2-b.X1 < 9 || b.Y2-b.Y1 < 9 {
		return nil, fmt.Errorf("盤面座標が小さすぎます")
	}
	if b.X1 < img.Min.X || b.Y1 < img.Min.Y || b.X2 > img.Max.X || b.Y2 > img.Max.Y {
		return nil, fmt.Errorf("盤面座標が画像の範囲(%dx%d)からはみ出しています", img.Dx(), img.Dy())
	}
	return b, nil
}
