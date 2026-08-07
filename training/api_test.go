package training

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

// 外部からの訓練データ登録 API。
//
// 守りたいのは「学習データという壊れやすい資産に、検証されていない書き込みが
// 入らないこと」なので、次の 4 つを回帰テストにしてある。
//   - ループバック以外は /api/status と /api/register しか触れない
//   - 登録されたデータは未確認として入り、学習の対象にならない
//   - 上限に達したら古い局面を消さずに 507 で断る
//   - 盤面座標が無ければ 400（学習時に価値 1/5 のエントリを作らない）

func testImage(t *testing.T, w, h int, c color.RGBA) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, c)
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// registerRequest は /api/register への multipart リクエストを組む。
// bounds が nil なら座標を付けない
func registerRequest(t *testing.T, png []byte, sfen string, bounds *BoardBounds) *http.Request {
	t.Helper()
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	fw, err := mw.CreateFormFile("image", "board.png")
	if err != nil {
		t.Fatal(err)
	}
	fw.Write(png)
	mw.WriteField("sfen", sfen)
	if bounds != nil {
		for k, v := range map[string]int{"x1": bounds.X1, "y1": bounds.Y1, "x2": bounds.X2, "y2": bounds.Y2} {
			mw.WriteField(k, strconv.Itoa(v))
		}
	}
	mw.Close()

	r := httptest.NewRequest(http.MethodPost, "/api/register", &body)
	r.Header.Set("Content-Type", mw.FormDataContentType())
	return r
}

func doRegister(t *testing.T, png []byte, sfen string, bounds *BoardBounds) (*httptest.ResponseRecorder, map[string]interface{}) {
	t.Helper()
	w := httptest.NewRecorder()
	handleRegister(w, registerRequest(t, png, sfen, bounds))
	var resp map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &resp)
	return w, resp
}

// resetSettings はテスト間で設定が漏れないようにする
func resetSettings(t *testing.T) {
	t.Helper()
	settingsMu.Lock()
	settings = APISettings{}
	settingsMu.Unlock()
	t.Cleanup(func() {
		settingsMu.Lock()
		settings = APISettings{}
		settingsMu.Unlock()
	})
}

func TestRegisterCreatesUnverifiedEntry(t *testing.T) {
	chdirTemp(t)
	resetSettings(t)

	img := testImage(t, 100, 100, color.RGBA{200, 180, 120, 255})
	w, resp := doRegister(t, img, "9/9/9/9/9/9/9/9/9", &BoardBounds{5, 5, 95, 95})
	if w.Code != http.StatusOK {
		t.Fatalf("code = %d, body = %s", w.Code, w.Body)
	}
	if resp["verified"] != false {
		t.Errorf("verified = %v, want false", resp["verified"])
	}

	entries := loadHistory().Entries
	if len(entries) != 1 {
		t.Fatalf("entries = %d, want 1", len(entries))
	}
	e := entries[0]
	if e.Source != SourceAPI {
		t.Errorf("Source = %q, want %q", e.Source, SourceAPI)
	}
	if e.IsVerified() {
		t.Error("API 経由の登録が確認済みとして入っている")
	}
	if e.BoardBounds == nil || e.BoardBounds.X1 != 5 || e.BoardBounds.Y2 != 95 {
		t.Errorf("BoardBounds = %+v", e.BoardBounds)
	}
	if _, err := os.Stat(filepath.Join(dataDir, e.ID+".png")); err != nil {
		t.Errorf("画像が保存されていない: %v", err)
	}
}

// 未確認の局面は学習に使われない。
// 画像と SFEN の対応は機械には検証できないので、人が解析タブで見るまで通さない
func TestTrainSkipsUnverified(t *testing.T) {
	chdirTemp(t)
	resetSettings(t)

	img := testImage(t, 100, 100, color.RGBA{200, 180, 120, 255})
	_, resp := doRegister(t, img, "9/9/9/9/9/9/9/9/9", &BoardBounds{5, 5, 95, 95})
	id, _ := resp["id"].(string)

	body, _ := json.Marshal(map[string]interface{}{"ids": []string{id}})
	w := httptest.NewRecorder()
	handleTrainHistory(w, httptest.NewRequest(http.MethodPost, "/api/trainhistory", bytes.NewReader(body)))

	var got struct {
		Entries []struct {
			ID    string `json:"id"`
			Error string `json:"error"`
		} `json:"entries"`
	}
	json.Unmarshal(w.Body.Bytes(), &got)
	// 学習データが空なので全体は 400 になる。見たいのは「未確認で弾かれたこと」
	if len(got.Entries) == 1 && got.Entries[0].Error == "" {
		t.Error("未確認の局面が学習に使われている")
	}
	if w.Code == http.StatusOK && len(got.Entries) == 0 {
		t.Error("結果が返っていない")
	}
}

// 上限に達しても古い局面を消さない。**手でラベル付けした資産を守るのが主目的**
func TestRegisterFullReturns507(t *testing.T) {
	chdirTemp(t)
	resetSettings(t)

	h := &HistoryData{}
	for i := 0; i < maxHistory; i++ {
		h.Entries = append(h.Entries, HistoryEntry{ID: newID(), SFEN: "9/9/9/9/9/9/9/9/9"})
	}
	oldest := h.Entries[maxHistory-1].ID
	saveHistoryFile(h)

	img := testImage(t, 100, 100, color.RGBA{10, 20, 30, 255})
	w, _ := doRegister(t, img, "9/9/9/9/9/9/9/9/9", &BoardBounds{5, 5, 95, 95})
	if w.Code != http.StatusInsufficientStorage {
		t.Fatalf("code = %d, want %d", w.Code, http.StatusInsufficientStorage)
	}
	entries := loadHistory().Entries
	if len(entries) != maxHistory {
		t.Errorf("entries = %d, want %d", len(entries), maxHistory)
	}
	if entries[len(entries)-1].ID != oldest {
		t.Error("上限で古い局面が押し出されている")
	}
}

// 盤面座標なしのエントリは samplesFromHistory が shiftAugment を作れず
// 学習価値が 1/5 になる。しかもそれに学習時まで気付けないので登録時に断る
func TestRegisterRequiresBounds(t *testing.T) {
	chdirTemp(t)
	resetSettings(t)

	img := testImage(t, 100, 100, color.RGBA{200, 180, 120, 255})
	if w, _ := doRegister(t, img, "9/9/9/9/9/9/9/9/9", nil); w.Code != http.StatusBadRequest {
		t.Errorf("座標なし: code = %d, want 400", w.Code)
	}
	// 画像からはみ出す座標も断る
	if w, _ := doRegister(t, img, "9/9/9/9/9/9/9/9/9", &BoardBounds{5, 5, 500, 500}); w.Code != http.StatusBadRequest {
		t.Errorf("範囲外: code = %d, want 400", w.Code)
	}
	if n := len(loadHistory().Entries); n != 0 {
		t.Errorf("弾いたはずのリクエストで %d 件登録されている", n)
	}
}

// 同じ画像の再送（リトライ）で履歴が増えない
func TestRegisterDeduplicates(t *testing.T) {
	chdirTemp(t)
	resetSettings(t)

	img := testImage(t, 100, 100, color.RGBA{200, 180, 120, 255})
	_, first := doRegister(t, img, "9/9/9/9/9/9/9/9/9", &BoardBounds{5, 5, 95, 95})
	w, second := doRegister(t, img, "9/9/9/9/9/9/9/9/9", &BoardBounds{5, 5, 95, 95})

	if w.Code != http.StatusOK || second["duplicate"] != true {
		t.Errorf("再送が duplicate になっていない: code = %d, resp = %v", w.Code, second)
	}
	if second["id"] != first["id"] {
		t.Errorf("id = %v, want %v", second["id"], first["id"])
	}
	if n := len(loadHistory().Entries); n != 1 {
		t.Errorf("entries = %d, want 1", n)
	}
}

// 解析タブから保存し直すと確認済みになり、学習に使えるようになる
func TestSaveSessionVerifiesAPIEntry(t *testing.T) {
	chdirTemp(t)
	resetSettings(t)

	img := testImage(t, 100, 100, color.RGBA{200, 180, 120, 255})
	_, resp := doRegister(t, img, "9/9/9/9/9/9/9/9/9", &BoardBounds{5, 5, 95, 95})
	id, _ := resp["id"].(string)

	sess := newTestSession(t)
	postSave(t, sess, "9/9/9/9/9/9/9/9/9", id)

	entries := loadHistory().Entries
	if len(entries) != 1 {
		t.Fatalf("entries = %d, want 1（上書きされていない）", len(entries))
	}
	if !entries[0].IsVerified() {
		t.Error("保存し直しても未確認のまま")
	}
	if entries[0].Source != SourceAPI {
		t.Errorf("Source = %q, want %q（出所は保たれるべき）", entries[0].Source, SourceAPI)
	}
}

// 送られてきた手番・持ち駒が、解析タブでの確認（＝保存し直し）で消えない。
//
// 画面は盤面部分しか作れないので、以前はここで `b - 1` を作って上書きしていた。
// `-` は「持ち駒なし」という積極的な主張なので、落とすより質が悪い
func TestVerifyKeepsHandsAndTurn(t *testing.T) {
	chdirTemp(t)
	resetSettings(t)

	const full = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w 2G3P 42"
	img := testImage(t, 100, 100, color.RGBA{200, 180, 120, 255})
	_, resp := doRegister(t, img, full, &BoardBounds{5, 5, 95, 95})
	id, _ := resp["id"].(string)

	if got := loadHistory().Entries[0].SFEN; got != full {
		t.Fatalf("登録時点で欠けている\n got = %q\nwant = %q", got, full)
	}

	// 解析タブからの保存。画面が送るのは盤面部分だけ
	board := "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL"
	sess := newTestSession(t)
	saved := postSave(t, sess, board, id)

	if got := loadHistory().Entries[0].SFEN; got != full {
		t.Errorf("確認後に手番・持ち駒が失われた\n got = %q\nwant = %q", got, full)
	}
	// 表示と保存内容を食い違わせない
	if saved["sfen"] != full {
		t.Errorf("レスポンスの sfen = %v, want %q", saved["sfen"], full)
	}
}

func TestMergeSFEN(t *testing.T) {
	const board = "9/9/9/9/9/9/9/9/9"
	cases := []struct {
		name        string
		board, prev string
		want        string
	}{
		{"引き継ぐ元があれば引き継ぐ", board, board + " w 2G3P 42", board + " w 2G3P 42"},
		{"分からないものは書かない", board, "", board},
		{"引き継ぐ元が盤面だけなら盤面だけ", board, board, board},
		{"送られた側が持っていればそちらを優先", board + " b - 1", board + " w 2G3P 42", board + " b - 1"},
		{"盤面が空なら既存を残す", "", board + " w 2G3P 42", board + " w 2G3P 42"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := mergeSFEN(c.board, c.prev); got != c.want {
				t.Errorf("mergeSFEN(%q, %q) = %q, want %q", c.board, c.prev, got, c.want)
			}
		})
	}
}

// アクセス制御。ループバックは素通し、外部は公開設定と許可パスに縛られる
func TestAccessControl(t *testing.T) {
	resetSettings(t)

	ok := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTeapot) // 到達したことが分かる印
	})
	h := withAccessControl(ok)

	call := func(remote, path, token string, s APISettings) int {
		settingsMu.Lock()
		settings = s
		settingsMu.Unlock()
		r := httptest.NewRequest(http.MethodPost, path, nil)
		r.RemoteAddr = remote
		if token != "" {
			r.Header.Set("Authorization", "Bearer "+token)
		}
		w := httptest.NewRecorder()
		h.ServeHTTP(w, r)
		return w.Code
	}

	const local, remote = "127.0.0.1:5000", "192.168.1.50:5000"
	off := APISettings{}
	open := APISettings{Enabled: true, External: true}
	guarded := APISettings{Enabled: true, External: true, Token: "secret"}

	cases := []struct {
		name   string
		addr   string
		path   string
		token  string
		set    APISettings
		expect int
	}{
		// ループバックは常に素通し（この UI が認証を持たないため）
		{"ローカルは公開オフでも学習を叩ける", local, "/api/trainhistory", "", off, http.StatusTeapot},
		{"ローカルはトークン設定時も素通し", local, "/api/trainhistory", "", guarded, http.StatusTeapot},
		// 外部
		{"公開オフなら外部からは見えない", remote, "/api/status", "", off, http.StatusNotFound},
		{"公開オンでも学習は外部から叩けない", remote, "/api/trainhistory", "secret", guarded, http.StatusNotFound},
		{"公開オンでも削除は外部から叩けない", remote, "/api/history/abc", "secret", guarded, http.StatusNotFound},
		{"status は登録が無効でも応答する", remote, "/api/status", "", APISettings{External: true}, http.StatusTeapot},
		{"登録が無効なら 503", remote, "/api/register", "", APISettings{External: true}, http.StatusServiceUnavailable},
		{"トークン不要なら通る", remote, "/api/register", "", open, http.StatusTeapot},
		{"トークンが違えば 401", remote, "/api/register", "wrong", guarded, http.StatusUnauthorized},
		{"トークンが無ければ 401", remote, "/api/register", "", guarded, http.StatusUnauthorized},
		{"トークンが合えば通る", remote, "/api/register", "secret", guarded, http.StatusTeapot},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := call(c.addr, c.path, c.token, c.set); got != c.expect {
				t.Errorf("code = %d, want %d", got, c.expect)
			}
		})
	}
}

// /api/status は素性を外部にばらまかない
func TestStatusHidesDetailsWithoutToken(t *testing.T) {
	chdirTemp(t)
	resetSettings(t)

	settingsMu.Lock()
	settings = APISettings{Enabled: true, External: true, Token: "secret"}
	settingsMu.Unlock()

	get := func(token string) map[string]interface{} {
		r := httptest.NewRequest(http.MethodGet, "/api/status", nil)
		r.RemoteAddr = "192.168.1.50:5000"
		if token != "" {
			r.Header.Set("Authorization", "Bearer "+token)
		}
		w := httptest.NewRecorder()
		handleStatus(w, r)
		var resp map[string]interface{}
		json.Unmarshal(w.Body.Bytes(), &resp)
		return resp
	}

	anon := get("")
	if anon["enabled"] != true {
		t.Errorf("enabled = %v, want true", anon["enabled"])
	}
	for _, k := range []string{"entries", "capacity", "external", "data_version"} {
		if _, ok := anon[k]; ok {
			t.Errorf("トークン無しで %q が見えている", k)
		}
	}
	if _, ok := get("secret")["entries"]; !ok {
		t.Error("正しいトークンでも詳細が返らない")
	}
}

// 設定が data/settings.json に残り、読み直せる
func TestSettingsRoundTrip(t *testing.T) {
	chdirTemp(t)
	resetSettings(t)

	body, _ := json.Marshal(map[string]interface{}{
		"enabled": true, "external": true, "use_token": true,
	})
	w := httptest.NewRecorder()
	handleSettings(w, httptest.NewRequest(http.MethodPost, "/api/settings", bytes.NewReader(body)))
	if w.Code != http.StatusOK {
		t.Fatalf("code = %d, body = %s", w.Code, w.Body)
	}
	var saved APISettings
	json.Unmarshal(w.Body.Bytes(), &saved)
	if saved.Token == "" {
		t.Fatal("トークンが発行されていない")
	}

	settingsMu.Lock()
	settings = APISettings{}
	settingsMu.Unlock()
	loadSettings()
	if got := currentSettings(); got != saved {
		t.Errorf("読み直し = %+v, want %+v", got, saved)
	}

	// トークンを外す
	body, _ = json.Marshal(map[string]interface{}{"enabled": true, "external": true, "use_token": false})
	w = httptest.NewRecorder()
	handleSettings(w, httptest.NewRequest(http.MethodPost, "/api/settings", bytes.NewReader(body)))
	if currentSettings().Token != "" {
		t.Error("use_token=false でトークンが消えていない")
	}
}

// 履歴が上限のとき、UI からの新規保存も断る（古い局面を消さない）。
// 上書きは件数が増えないので通す
func TestSaveSessionFullReturns507(t *testing.T) {
	chdirTemp(t)
	resetSettings(t)

	h := &HistoryData{}
	for i := 0; i < maxHistory; i++ {
		h.Entries = append(h.Entries, HistoryEntry{ID: newID(), SFEN: fmt.Sprintf("entry%d", i)})
	}
	existing := h.Entries[0].ID
	saveHistoryFile(h)

	sess := newTestSession(t)
	body, _ := json.Marshal(map[string]string{"session": sess, "sfen": "9/9/9/9/9/9/9/9/9"})
	w := httptest.NewRecorder()
	handleSaveSession(w, httptest.NewRequest(http.MethodPost, "/api/savesession", bytes.NewReader(body)))
	if w.Code != http.StatusInsufficientStorage {
		t.Errorf("新規保存: code = %d, want %d", w.Code, http.StatusInsufficientStorage)
	}
	if n := len(loadHistory().Entries); n != maxHistory {
		t.Errorf("entries = %d, want %d", n, maxHistory)
	}

	// 上書きは通る
	body, _ = json.Marshal(map[string]string{"session": sess, "sfen": "上書き", "history_id": existing})
	w = httptest.NewRecorder()
	handleSaveSession(w, httptest.NewRequest(http.MethodPost, "/api/savesession", bytes.NewReader(body)))
	if w.Code != http.StatusOK {
		t.Errorf("上書き: code = %d, body = %s", w.Code, w.Body)
	}
}
