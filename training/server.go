package training

import (
	"crypto/rand"
	"embed"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"image"
	_ "image/jpeg"
	"image/png"
	_ "image/png"
	"io"
	"io/fs"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ShinteLab/core/sfen"
	shinteweb "github.com/ShinteLab/core/web"
	"github.com/ShinteLab/suteme"
)

//go:embed static
var static embed.FS

type session struct {
	Original image.Image
	Result   *suteme.AnalyzeResult
	BoardImg *image.RGBA
}

const (
	dataDir     = "data"
	historyFile = "data/history.json"
)

type BoardBounds struct {
	X1, Y1, X2, Y2 int
}

// 局面の出所。空文字（既存エントリ）は UI からの保存とみなす
const (
	SourceUI  = "ui"
	SourceAPI = "api"
)

type HistoryEntry struct {
	ID          string       `json:"id"`
	CreatedAt   string       `json:"created_at"`
	SFEN        string       `json:"sfen"`
	BoardBounds *BoardBounds `json:"board_bounds,omitempty"`
	// Source は登録経路（"ui" / "api"）。空は UI（既存エントリ）
	Source string `json:"source,omitempty"`
	// Verified は人が解析タブで内容を見たか。**画像と SFEN の対応は機械には
	// 検証できない**ので、API 経由で入ったものは false で入り、学習の対象に
	// しない。解析タブから保存し直すと true になる
	Verified bool `json:"verified,omitempty"`
	// Hash は登録された画像そのもののハッシュ。API のリトライで同じ局面が
	// 増えないようにするためだけに使う
	Hash string `json:"hash,omitempty"`
	// Look は**人が付けた「盤の見た目」の名前**（同じ中継・同じゲーム画面なら同じ名前）。
	//
	// 認識率は局面数より「同じ見た目が何枚あるか」で決まるので、集まり方を測るには
	// 見た目で束ねる必要がある。**機械には決められない**（画像サイズは撮るたびに
	// 変わるので当てにならず、同じ放送でも実盤・大盤・ゲーム画面は別の見た目）。
	// 空なら未判定で、`coverage.go` が署名から候補を出すだけ
	Look string `json:"look,omitempty"`
}

// IsVerified は学習に使ってよいエントリかを返す。
// Source を持たない既存エントリは UI で作られたものなので確認済みとみなす
func (e HistoryEntry) IsVerified() bool {
	return e.Source != SourceAPI || e.Verified
}

type HistoryData struct {
	Entries []HistoryEntry `json:"entries"`
}

func loadHistory() *HistoryData {
	f, err := os.Open(historyFile)
	if err != nil {
		return &HistoryData{Entries: []HistoryEntry{}}
	}
	defer f.Close()
	var h HistoryData
	json.NewDecoder(f).Decode(&h)
	if h.Entries == nil {
		h.Entries = []HistoryEntry{}
	}
	return &h
}

// saveHistoryFile は履歴を書き出す。
// 一時ファイル + rename にしてあるのは、書き込み中に落ちたときに
// 履歴が全損しないようにするため（API で書き込み頻度が上がる）
func saveHistoryFile(h *HistoryData) {
	os.MkdirAll(dataDir, 0755)
	if err := writeJSONFile(historyFile, h); err != nil {
		log.Printf("saveHistory: %v", err)
	}
}

var (
	sessions = make(map[string]*session)
	mu       sync.RWMutex
	model    *suteme.Model
	knn      *suteme.KNN
	// nnStale: 学習データを更新したのに gobrain NN を学習し直していない状態。
	// 比較用 SFEN（sfen_nn）が古いモデルの出力であることを画面に出すために持つ
	nnStale bool
	// 入力ベクトルの表現を変えたらファイル名の版を上げる。MergeSamples は
	// 入力の内容でマージするので、表現の違うサンプルを同じファイルに混ぜると
	// 古いものが消えずに残り続ける。
	// （v4 は float32 精度 + バイナリ、v3 は面積平均リサイズ + JSON、
	//   v2 は最近傍リサイズ。旧 training_data.json は空マスが歩として
	//   混入しているため使用しない）
	// **v3 の JSON は内容としては v4 と同じなので読み込みだけ受け付ける**
	// （loadExistingTrainingData）。書き出しは常に dataFile。
	dataFile  = suteme.DefaultDataFile
	modelFile = suteme.DefaultModelFile
	// stripFile は盤の縁の帯の教師データ。**駒種の学習データとは別**
	stripFile = suteme.DefaultStripFile
	modelLock sync.RWMutex
	historyMu sync.RWMutex
)

// Serve はラベリング・学習用のWebサーバを起動する
// 起動時にカレントディレクトリの学習データ（suteme.DefaultDataFile）と
// NN モデル（suteme.DefaultModelFile）を自動ロードする。
// **ファイル名を直に書かないこと**（版が上がると嘘になる）
func Serve(port string) error {
	loadSettings()
	// 学習データがあれば k-NN を構築（学習処理は不要）
	data, dataPath := loadExistingTrainingData()
	if data != nil {
		if kn := suteme.NewKNN(data.Samples); kn != nil {
			knn = kn
			log.Printf("Built k-NN from %d samples in %s", kn.Len(), dataPath)
		}
	}
	// 盤の縁の帯の判定器（1マス滑りの補正）。無ければ補正しないだけ
	if ss, err := suteme.LoadStripData(stripFile); err == nil {
		if j := suteme.NewStripJudge(ss); j != nil {
			suteme.SetStripJudge(j)
			log.Printf("Loaded strip judge from %s (%d strips)", stripFile, j.Samples())
		}
	}
	// 起動時に保存済みモデルを読み込む
	if m, err := suteme.LoadModel(modelFile); err == nil {
		model = m
		// NN は既定では学習し直さないので、前回の起動より前に置き去りになっている
		// ことがある。学習データより古ければ比較用 SFEN に断りを出す
		nnStale = olderThan(modelFile, dataPath)
		log.Printf("Loaded model from %s (stale=%v)", modelFile, nnStale)
	}

	mux := http.NewServeMux()
	staticFS, _ := fs.Sub(static, "static")
	mux.Handle("/", http.FileServer(http.FS(staticFS)))
	// 共有フロント資産(<shogi-board> と SFEN/USI ロジック)を配信する。
	// これにより UI 側の SFEN 処理を @shinte/web (core/web) に一本化する。
	mux.Handle("/shinte-web/", http.StripPrefix("/shinte-web/", http.FileServer(http.FS(shinteweb.Assets))))
	mux.HandleFunc("/api/analyze", handleAnalyze)
	mux.HandleFunc("/api/images/", handleImage)
	mux.HandleFunc("/api/cells/", handleCell)
	mux.HandleFunc("/api/history", handleHistory)
	mux.HandleFunc("/api/history/", handleHistoryItem)
	mux.HandleFunc("/api/savesession", handleSaveSession)
	mux.HandleFunc("/api/setboard", handleSetBoard)
	mux.HandleFunc("/api/trainhistory", handleTrainHistory)
	mux.HandleFunc("/api/recognize", handleRecognize)
	mux.HandleFunc("/api/handcheck", handleHandCheck)
	mux.HandleFunc("/api/evaluate", handleEvaluate)
	mux.HandleFunc("/api/evaluations", handleEvaluations)
	mux.HandleFunc("/api/evaluations/", handleEvaluations)
	mux.HandleFunc("/api/coverage", handleCoverage)
	mux.HandleFunc("/api/look", handleLook)
	mux.HandleFunc("/api/negative", handleNegatives)
	mux.HandleFunc("/api/negative/", handleNegativeItem)
	mux.HandleFunc("/api/status", handleStatus)
	mux.HandleFunc("/api/settings", handleSettings)
	mux.HandleFunc("/api/register", handleRegister)

	// リスナは常に全インターフェースで張り、外部公開のオン/オフは
	// リクエストごとに判定する（トグルが再起動なしで即時に効く）
	addr := ":" + port
	fmt.Printf("http://localhost%s\n", addr)
	if s := currentSettings(); s.External {
		log.Printf("外部公開: 有効 / 登録受付: %v / トークン: %v", s.Enabled, s.Token != "")
	}
	srv := &http.Server{
		Addr:              addr,
		Handler:           withAccessControl(mux),
		ReadHeaderTimeout: 10 * time.Second,
	}
	return srv.ListenAndServe()
}

// loadExistingTrainingData は既存の学習データを読む。dataFile（v4）が無ければ
// 旧版（v3 の JSON）を探す。**書き出しは常に dataFile なので、一度学習すれば移る。**
// 見つからなければ nil を返す。戻り値の 2 つ目は読み込み元のパス
func loadExistingTrainingData() (*suteme.TrainingData, string) {
	for _, path := range append([]string{dataFile}, suteme.LegacyDataFiles...) {
		data, err := suteme.LoadTrainingData(path)
		if err != nil {
			if !os.IsNotExist(err) {
				log.Printf("学習データが読めません (%s): %v", path, err)
			}
			continue
		}
		return data, path
	}
	return nil, ""
}

// olderThan は a の更新時刻が b より古いかを返す。どちらかが読めなければ false
func olderThan(a, b string) bool {
	fa, err := os.Stat(a)
	if err != nil {
		return false
	}
	fb, err := os.Stat(b)
	if err != nil {
		return false
	}
	return fa.ModTime().Before(fb.ModTime())
}

// currentPredictor は使用する推論器を返す（k-NN 優先、なければ gobrain NN）
// どちらも無い場合は nil
func currentPredictor() suteme.Predictor {
	modelLock.RLock()
	defer modelLock.RUnlock()
	if knn != nil {
		return knn
	}
	if model != nil {
		return model
	}
	return nil
}

// predictCells は全マスの駒種を推論し、SFENラベルの9x9グリッドを返す
// 推論器なし・盤面未検出時は nil を返す。空マスは "none"
func predictCells(s *session) *[9][9]map[string]interface{} {
	m := currentPredictor()
	if m == nil || s.Result == nil || s.Result.Board == nil {
		return nil
	}
	var grid [9][9]map[string]interface{}
	bc := suteme.BoardColor(s.Original, s.Result.Board)
	bo := suteme.NewBoardOrient(s.Original, s.Result.Board, bc)
	for row := 0; row < 9; row++ {
		for col := 0; col < 9; col++ {
			cell := s.Result.Board.ExtractCell(s.Original, row, col)
			if cell == nil {
				continue
			}
			cat, _ := bo.Classify(cell, bc, m)
			if cat == suteme.CellEmpty {
				grid[row][col] = map[string]interface{}{"label": suteme.EmptyLabel, "confidence": 95}
				continue
			}
			ncell := cell
			if cat == suteme.CellPieceDown {
				ncell = suteme.Rotate180(cell)
			}
			class, conf := m.Predict(ncell)
			// 推論器が空と言うなら分類より優先する（RecognizeBoard と同じ判断）
			if class == suteme.ClassEmpty {
				grid[row][col] = map[string]interface{}{
					"label":      suteme.EmptyLabel,
					"confidence": int(conf * 100),
				}
				continue
			}
			base := suteme.ClassToBaseLabel(class)
			label := base
			if cat == suteme.CellPieceDown {
				if len(base) > 1 && base[0] == '+' {
					label = "+" + strings.ToLower(base[1:])
				} else {
					label = strings.ToLower(base)
				}
			}
			grid[row][col] = map[string]interface{}{
				"label":      label,
				"confidence": int(conf * 100),
			}
		}
	}
	return &grid
}

func newID() string {
	b := make([]byte, 8)
	rand.Read(b)
	return hex.EncodeToString(b)
}

func handleAnalyze(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	r.ParseMultipartForm(32 << 20)
	file, _, err := r.FormFile("image")
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	result := suteme.Analyze(img)
	boardImg := result.DrawBoard(img)

	id := newID()
	mu.Lock()
	sessions[id] = &session{
		Original: img,
		Result:   result,
		BoardImg: boardImg,
	}
	mu.Unlock()

	hasBoard := result.Board != nil
	confidence := 0.0
	if hasBoard {
		confidence = suteme.ValidateBoard(img, result.Board)
	}

	log.Printf("session %s: board=%v confidence=%.0f%%", id, hasBoard, confidence*100)

	resp := map[string]interface{}{
		"id":         id,
		"board":      hasBoard && confidence >= 0.5,
		"confidence": int(confidence * 100),
	}
	if hasBoard && confidence >= 0.5 {
		b := result.Board.Bounds
		resp["bounds"] = map[string]int{
			"x": b.Min.X, "y": b.Min.Y,
			"w": b.Dx(), "h": b.Dy(),
		}
		cats := suteme.ClassifyBoard(img, result.Board)
		catGrid := [9][9]int{}
		for r := 0; r < 9; r++ {
			for c := 0; c < 9; c++ {
				catGrid[r][c] = int(cats[r][c])
			}
		}
		resp["categories"] = catGrid

		// モデルがあれば駒種の推論候補も返す（ラベリング支援用）
		mu.RLock()
		if sug := predictCells(sessions[id]); sug != nil {
			resp["suggestions"] = sug
		}
		mu.RUnlock()
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func handleImage(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/api/images/"), "/")
	if len(parts) != 2 {
		http.NotFound(w, r)
		return
	}

	id, typ := parts[0], parts[1]

	mu.RLock()
	s, ok := sessions[id]
	mu.RUnlock()
	if !ok {
		http.NotFound(w, r)
		return
	}

	var img image.Image
	switch typ {
	case "original":
		img = s.Original
	case "edges":
		img = s.Result.Edges
	case "board":
		img = s.BoardImg
	default:
		http.NotFound(w, r)
		return
	}

	w.Header().Set("Content-Type", "image/png")
	w.Header().Set("Cache-Control", "no-cache")
	png.Encode(w, img)
}

func handleCell(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/api/cells/"), "/")
	if len(parts) != 3 {
		http.NotFound(w, r)
		return
	}

	id := parts[0]
	row, err1 := strconv.Atoi(parts[1])
	col, err2 := strconv.Atoi(parts[2])
	if err1 != nil || err2 != nil || row < 0 || row > 8 || col < 0 || col > 8 {
		http.NotFound(w, r)
		return
	}

	mu.RLock()
	s, ok := sessions[id]
	mu.RUnlock()
	if !ok || s.Result.Board == nil {
		http.NotFound(w, r)
		return
	}

	cell := s.Result.Board.ExtractCell(s.Original, row, col)
	if cell == nil {
		http.NotFound(w, r)
		return
	}

	w.Header().Set("Content-Type", "image/png")
	w.Header().Set("Cache-Control", "no-cache")
	png.Encode(w, cell)
}

func handleHistory(w http.ResponseWriter, r *http.Request) {
	historyMu.RLock()
	h := loadHistory()
	historyMu.RUnlock()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(h)
}

// handleHistoryItem: /api/history/{id}/image で画像取得、
// /api/history/{id} への DELETE で履歴（画像 + エントリ）を削除する
func handleHistoryItem(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/api/history/")
	id = strings.TrimSuffix(id, "/image")
	if !validID(id) {
		http.NotFound(w, r)
		return
	}
	if r.Method == http.MethodDelete {
		handleHistoryDelete(w, r, id)
		return
	}
	f, err := os.Open(filepath.Join(dataDir, id+".png"))
	if err != nil {
		http.NotFound(w, r)
		return
	}
	defer f.Close()
	w.Header().Set("Content-Type", "image/png")
	// 上書き保存で同じ URL の画像が差し替わるので、長期キャッシュにはしない
	w.Header().Set("Cache-Control", "no-cache")
	io.Copy(w, f)
}

// validID は履歴 ID（newID の 16 桁 hex）としてパスに使ってよい文字列かを返す。
// ディレクトリを抜ける文字列で data/ 配下のファイルを触らせないための最低限の検査
func validID(id string) bool {
	if id == "" {
		return false
	}
	for _, ch := range id {
		if !((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f')) {
			return false
		}
	}
	return true
}

// handleHistoryDelete は履歴エントリと保存画像を削除する
func handleHistoryDelete(w http.ResponseWriter, r *http.Request, id string) {
	historyMu.Lock()
	h := loadHistory()
	found := false
	kept := h.Entries[:0]
	for _, e := range h.Entries {
		if e.ID == id {
			found = true
			continue
		}
		kept = append(kept, e)
	}
	if found {
		h.Entries = kept
		saveHistoryFile(h)
		os.Remove(filepath.Join(dataDir, id+".png"))
	}
	historyMu.Unlock()

	if !found {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok", "id": id})
}

func handleSaveSession(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		Session string `json:"session"`
		SFEN    string `json:"sfen"`
		// HistoryID があればその履歴を上書きする。保存し直すたびに
		// 同じ画像の局面が増えないようにするため（空なら新規保存）
		HistoryID string `json:"history_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	mu.RLock()
	s, ok := sessions[req.Session]
	mu.RUnlock()
	if !ok || s.Original == nil {
		http.NotFound(w, r)
		return
	}

	// 上書き対象があるかを先に確かめる。履歴に無い ID（既に削除された等）は
	// 黙って新規保存に倒す
	id := ""
	var prev HistoryEntry
	historyMu.RLock()
	cur := loadHistory()
	if validID(req.HistoryID) {
		for _, e := range cur.Entries {
			if e.ID == req.HistoryID {
				id, prev = e.ID, e
				break
			}
		}
	}
	full := len(cur.Entries) >= maxHistory
	historyMu.RUnlock()

	overwrite := id != ""
	if !overwrite {
		// **古い局面を押し出して消すのはやめた。** 手でラベル付けした正解が
		// 黙って失われる（実際に過去そうなった）ため、上限に達したら断る。
		// 上書きは件数が増えないので通す
		if full {
			http.Error(w,
				fmt.Sprintf("履歴が上限（%d 件）です。履歴タブで不要な局面を削除してください", maxHistory),
				http.StatusInsufficientStorage)
			return
		}
		id = newID()
	}
	os.MkdirAll(dataDir, 0755)
	f, err := os.Create(filepath.Join(dataDir, id+".png"))
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	png.Encode(f, s.Original)
	f.Close()

	entry := HistoryEntry{
		ID:        id,
		CreatedAt: time.Now().Format("01/02 15:04"),
		SFEN:      mergeSFEN(req.SFEN, prev.SFEN),
		Source:    SourceUI,
		// 画面から保存した＝人が解析タブで見た、ということ。
		// API 経由で入った未確認の局面はここを通ると確認済みになる
		Verified: true,
		Hash:     prev.Hash,
		// 見た目の判定は人が付けたもので、保存し直しても失わせない
		// （持ち駒・手番を引き継ぐのと同じ理由。`mergeSFEN` の項）
		Look: prev.Look,
	}
	if overwrite && prev.Source != "" {
		entry.Source = prev.Source
	}
	if s.Result != nil && s.Result.Board != nil {
		b := s.Result.Board.Bounds
		entry.BoardBounds = &BoardBounds{b.Min.X, b.Min.Y, b.Max.X, b.Max.Y}
	}
	historyMu.Lock()
	h := loadHistory()
	replaced := false
	if overwrite {
		for i := range h.Entries {
			if h.Entries[i].ID == id {
				// 一覧での位置は動かさない（直したものが先頭に飛ぶと追いにくい）
				h.Entries[i] = entry
				replaced = true
				break
			}
		}
	}
	if !replaced {
		h.Entries = append([]HistoryEntry{entry}, h.Entries...)
	}
	saveHistoryFile(h)
	historyMu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":    "ok",
		"id":        id,
		"overwrote": replaced,
		// 画面が送った盤面部分に手番・持ち駒が足されていることがあるので、
		// 実際に保存した文字列を返す（表示と保存内容を食い違わせない）
		"sfen": entry.SFEN,
	})
}

// handleHandCheck は「盤面 + 手入力した持ち駒」を駒種ごとに突き合わせる。
//
// **これは参考情報であってエラーではない。** 正しいのは画像であって、盤面が
// 全駒そろっている必要はない。**駒落ち・詰将棋では少ないほうが正常**なので、
// 不足は警告にしない（`missing` として返し、画面では注記に留める）。
// 多すぎる場合だけが認識か正解ラベルの誤りを示すので `excess` に入れる。
//
// 駒数の上限は `core/sfen` 由来（`suteme.PieceLimits`）。画面側に表を持たせない
// ためにサーバで計算する。
func handleHandCheck(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		Board string         `json:"board"`
		Hands map[string]int `json:"hands"` // 先後合計（駒台は先後を区別せず突き合わせる）
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httpJSONError(w, http.StatusBadRequest, err.Error())
		return
	}

	sente, gote := suteme.CountFromSFEN(req.Board)
	type diff struct {
		Piece string `json:"piece"`
		Total int    `json:"total"`
		Limit int    `json:"limit"`
	}
	resp := map[string]interface{}{
		"order":   suteme.HandOrder,
		"limits":  suteme.PieceLimits,
		"derived": map[string]int{},
		"excess":  []diff{},
		"missing": []diff{},
	}
	derived := map[string]int{}
	var excess, missing []diff
	for _, k := range suteme.HandOrder {
		limit := suteme.PieceLimits[k]
		onBoard := sente[k] + gote[k]
		// 盤面から逆算した駒台の枚数（「逆算」ボタンの下書き用）
		if n := limit - onBoard; n > 0 {
			derived[k] = n
		}
		total := onBoard + req.Hands[k]
		switch {
		case total > limit:
			excess = append(excess, diff{k, total, limit})
		case total < limit:
			missing = append(missing, diff{k, total, limit})
		}
	}
	resp["derived"] = derived
	if excess != nil {
		resp["excess"] = excess
	}
	if missing != nil {
		resp["missing"] = missing
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// mergeSFEN は画面が作り直した盤面部分に、既存エントリの手番・持ち駒・手数を引き継ぐ。
//
// **解析タブには手番や持ち駒を編集する UI が無い**ので、画面から送られてくるのは
// 盤面部分だけ。以前はここで `b - 1` を作って保存していたため、API 登録で受け取った
// 正確な持ち駒が、確認して保存し直した瞬間に「持ち駒なし・先手番」に化けていた。
// `-` は「無い」という積極的な主張なので、落とすより質が悪い。
//
// **分からないものは書かない。** 引き継ぐ元が無ければ盤面部分だけを保存する
// （学習は fields[0] しか見ないので、これで困ることはない）。
func mergeSFEN(board, prev string) string {
	f := strings.Fields(board)
	if len(f) == 0 {
		return prev
	}
	rest := f[1:]
	if len(rest) == 0 {
		// 画面からの保存。既存エントリが持っていれば引き継ぐ
		if p := strings.Fields(prev); len(p) > 1 {
			rest = p[1:]
		}
	}
	if len(rest) == 0 {
		return f[0]
	}
	return f[0] + " " + strings.Join(rest, " ")
}

// handleSetBoard: 手動指定の矩形で盤面を再設定する
func handleSetBoard(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		Session string `json:"session"`
		X1      int    `json:"x1"`
		Y1      int    `json:"y1"`
		X2      int    `json:"x2"`
		Y2      int    `json:"y2"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	mu.Lock()
	s, ok := sessions[req.Session]
	var cats [9][9]suteme.CellCategory
	if ok {
		br := suteme.BoardRegionFromRect(req.X1, req.Y1, req.X2, req.Y2)
		s.Result.Board = br
		s.BoardImg = s.Result.DrawBoard(s.Original)
		cats = suteme.ClassifyBoard(s.Original, br)
	}
	mu.Unlock()

	if !ok {
		http.NotFound(w, r)
		return
	}

	catGrid := [9][9]int{}
	for row := 0; row < 9; row++ {
		for col := 0; col < 9; col++ {
			catGrid[row][col] = int(cats[row][col])
		}
	}

	resp := map[string]interface{}{
		"status":     "ok",
		"categories": catGrid,
	}
	mu.RLock()
	if sug := predictCells(s); sug != nil {
		resp["suggestions"] = sug
	}
	mu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// handleTrainHistory: 選択した保存済み局面から学習する。
// 履歴には画像・盤面座標・正解SFENが揃っているので、解析タブに読み込む操作を
// 挟まずにサーバ側だけで学習データを組み立てられる。
func handleTrainHistory(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	defer func() {
		if rec := recover(); rec != nil {
			log.Printf("handleTrainHistory panic: %v", rec)
			http.Error(w, fmt.Sprintf("training failed: %v", rec), http.StatusInternalServerError)
		}
	}()

	var req struct {
		IDs []string `json:"ids"`
		// NN: gobrain NN も学習し直すか。**既定は false。**
		// 主認識器の k-NN は生データから即再構築されるので、通常は要らない。
		// 詳細は trainAndSave のコメント
		NN bool `json:"nn"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if len(req.IDs) == 0 {
		http.Error(w, "局面が選択されていません", http.StatusBadRequest)
		return
	}

	historyMu.RLock()
	h := loadHistory()
	historyMu.RUnlock()
	byID := make(map[string]HistoryEntry, len(h.Entries))
	for _, e := range h.Entries {
		byID[e.ID] = e
	}

	// 局面ごとの結果（どれが何枚取れたか・失敗したか）を返す
	type entryResult struct {
		ID     string `json:"id"`
		Pieces int    `json:"pieces"`
		Empty  int    `json:"empty"`
		Error  string `json:"error,omitempty"`
	}
	results := make([]entryResult, 0, len(req.IDs))

	var fresh []suteme.TrainingSample
	for _, id := range req.IDs {
		e, ok := byID[id]
		if !ok {
			results = append(results, entryResult{ID: id, Error: "履歴にありません"})
			continue
		}
		// **画像と SFEN の対応は機械には検証できない。** API 経由で入った
		// 局面は、解析タブで人が一度見て保存し直すまで学習に使わない
		if !e.IsVerified() {
			results = append(results, entryResult{ID: id, Error: "未確認です（解析タブで内容を確認してください）"})
			continue
		}
		samples, err := samplesFromHistory(e)
		if err != nil {
			results = append(results, entryResult{ID: id, Error: err.Error()})
			continue
		}
		fresh = append(fresh, samples...)
		empty := countEmpty(samples)
		results = append(results, entryResult{ID: id, Pieces: len(samples) - empty, Empty: empty})
	}

	log.Printf("Training from %d history entries (%d pieces, %d empty)",
		len(req.IDs), len(fresh)-countEmpty(fresh), countEmpty(fresh))
	res, err := trainAndSave(fresh, req.NN)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	res["entries"] = results

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(res)
}

// countEmpty は空マスのサンプル数を数える
func countEmpty(samples []suteme.TrainingSample) int {
	n := 0
	for _, s := range samples {
		if s.Label == suteme.ClassEmpty {
			n++
		}
	}
	return n
}

// shiftAugment は学習サンプルを作るときに盤面座標をずらす量(px)の組。
//
// **手でラベル付けした座標「ちょうど」でしか学習しないと k-NN がその
// クロップを丸暗記する。** DetectBoard は手動座標に対して 0.1〜0.5マス
// （実測で数px）ずれるので、推論時のクロップは学習時と必ず食い違う。
// 実測では、盤面座標を 1px ずらすだけで誤認識マスが 891 中 22 → 57 に増えた。
// ずらしたクロップも学習に入れることで、その食い違いを吸収する。
//
// 縦横の全組み合わせ（9通り）ではなく上下左右+原点の5通りにしてある。
// k-NN の推論コストはサンプル数に比例するため。
var shiftAugment = [][2]int{{0, 0}, {-2, -2}, {2, -2}, {-2, 2}, {2, 2}}

// samplesFromHistory は保存済み局面（画像 + 盤面座標 + 正解SFEN）から
// 学習サンプルを作る。空マスも ClassEmpty として含める
// （グリッド線・木目で駒に見える空マスを被覆率では分離できないため）。
// 盤面座標を shiftAugment の分だけずらしたクロップも一緒に作る。
func samplesFromHistory(e HistoryEntry) ([]suteme.TrainingSample, error) {
	f, err := os.Open(filepath.Join(dataDir, e.ID+".png"))
	if err != nil {
		return nil, fmt.Errorf("画像が読めません")
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("画像のデコードに失敗しました")
	}

	// 盤面座標。shiftAugment でずらした版も作る
	var regions []*suteme.BoardRegion
	if e.BoardBounds != nil {
		b := e.BoardBounds
		for _, s := range shiftAugment {
			regions = append(regions, suteme.BoardRegionFromRect(
				b.X1+s[0], b.Y1+s[1], b.X2+s[0], b.Y2+s[1]))
		}
	} else {
		// 盤面座標を持たない古いエントリは自動検出に頼る。
		// 検出座標自体が信用しきれないのでずらした版は作らない
		br := suteme.DetectBoard(img)
		if br == nil || suteme.ValidateBoard(img, br) < 0.5 {
			return nil, fmt.Errorf("盤面座標が保存されておらず、自動検出もできません")
		}
		regions = []*suteme.BoardRegion{br}
	}

	fields := strings.Fields(e.SFEN)
	if len(fields) == 0 {
		return nil, fmt.Errorf("SFEN が空です")
	}

	var samples []suteme.TrainingSample
	for _, br := range regions {
		s, err := samplesFromRegion(img, br, fields[0])
		if err != nil {
			return nil, err
		}
		samples = append(samples, s...)
	}
	if len(samples) == 0 {
		return nil, fmt.Errorf("駒が1枚も取れませんでした")
	}
	return samples, nil
}

// samplesFromRegion は1つの盤面座標から81マス分の学習サンプルを作る
func samplesFromRegion(img image.Image, br *suteme.BoardRegion, board string) ([]suteme.TrainingSample, error) {
	// SFEN の解析は core/sfen に委譲する（駒文字の対応表を持たない）。
	// ParseBoard は駒のあるマスだけを呼ぶので、埋まったマスを記録しておき
	// 残りを空マスとして補う
	var samples []suteme.TrainingSample
	var occupied [9][9]bool
	err := sfen.ParseBoard(board, func(rank, file, base int, black, promoted bool) {
		if rank < 0 || rank >= 9 || file < 0 || file >= 9 {
			return
		}
		label := sfen.Letter(base)
		if label == "None" {
			return
		}
		if promoted {
			label = "+" + label
		}
		class := suteme.LabelToClass(label)
		if class < 0 || class == suteme.ClassEmpty {
			return
		}
		cell := br.ExtractCell(img, rank, file)
		if cell == nil {
			return
		}
		occupied[rank][file] = true
		// 後手は180度回転して先手向きに正規化する（推論時と揃える）
		if !black {
			cell = suteme.Rotate180(cell)
		}
		samples = append(samples, suteme.TrainingSample{
			Input: suteme.CellToInput(cell),
			Label: class,
		})
	})
	if err != nil {
		return nil, fmt.Errorf("SFEN の解析に失敗しました")
	}

	// 空マス。向きが無いので回転はしない
	for rank := 0; rank < 9; rank++ {
		for file := 0; file < 9; file++ {
			if occupied[rank][file] {
				continue
			}
			cell := br.ExtractCell(img, rank, file)
			if cell == nil {
				continue
			}
			samples = append(samples, suteme.TrainingSample{
				Input: suteme.CellToInput(cell),
				Label: suteme.ClassEmpty,
			})
		}
	}
	return samples, nil
}

// trainAndSave は既存の学習データに fresh をマージして学習データ・k-NN を更新する。
// 戻り値はそのまま JSON で返せる形。
//
// **gobrain NN の学習は trainNN が真のときだけ行う（既定は行わない）。**
// 主認識器は k-NN（`currentPredictor` が優先）で、生データから即座に組み直せる。
// 一方 NN の学習は「選んだ局面数」ではなく**累積した全サンプル数**に比例し、
// しかも 300 エポック固定なので、1 局だけ選んでも全量を学習し直す。
// 実測で 1 パターン×1 エポック = 257µs、7000 サンプルで 9 分、
// 16605 サンプル（41局面）で 21 分。局面を足すほど線形に伸びる。
// その 9 分で作られる NN は /api/recognize の比較用 SFEN（sfen_nn）にしか
// 使われないので、既定から外して明示的に回す形にしてある。
func trainAndSave(fresh []suteme.TrainingSample, trainNN bool) (map[string]interface{}, error) {
	// 既存の学習データを読み込む（v4 が無ければ旧版から引き継ぐ）
	var existing []suteme.TrainingSample
	if e, path := loadExistingTrainingData(); e != nil {
		existing = e.Samples
		log.Printf("Loaded %d existing samples from %s", len(existing), path)
	}

	// 入力長が合わないサンプルを除外
	existing, skippedOld := filterByInputSize(existing)
	fresh, skippedNew := filterByInputSize(fresh)
	if skipped := skippedOld + skippedNew; skipped > 0 {
		log.Printf("Skipped %d samples with wrong input size", skipped)
	}

	// 既存分と今回分を入力の内容でマージして重複を除く。
	// これが無いと「学習」を押すたびに同じサンプルが二重に積まれる
	// （毎回「既存ファイルの全件 + 今回分」を保存し直すため）。
	beforeMerge := len(existing) + len(fresh)
	var data suteme.TrainingData
	data.Samples = MergeSamples(existing, fresh)
	duplicates := beforeMerge - len(data.Samples)
	if duplicates > 0 {
		log.Printf("Merged %d samples → %d (%d duplicates removed)", beforeMerge, len(data.Samples), duplicates)
	}

	if len(data.Samples) == 0 {
		return nil, fmt.Errorf("学習データがありません")
	}

	// 学習データを保存（生データ全件）。**これに失敗したら先へ進まない。**
	// ラベル付けした正解が消えたのに「学習完了」と出るのが一番まずい
	if err := suteme.SaveTrainingData(dataFile, &data); err != nil {
		return nil, fmt.Errorf("学習データを保存できませんでした: %w", err)
	}

	// k-NN は生データから即再構築（学習不要）
	if kn := suteme.NewKNN(data.Samples); kn != nil {
		modelLock.Lock()
		knn = kn
		modelLock.Unlock()
	}

	// 盤の縁の帯（1マス滑りの判定器）も作り直す。
	// **選んだ局面ではなく確認済みの全局面から毎回ゼロで作る**
	// （1 局面あたり 8 本しか無いので数秒。累積すると消した局面の帯が残る）。
	stripCount, err := rebuildStripData()
	if err != nil {
		// **ここで失敗しても学習は成功扱いにする。** 帯データは history から
		// いつでも作り直せるうえ、無ければ滑りの補正をしないだけで検出は動く
		log.Printf("Strip data rebuild failed: %v", err)
	}
	if stripCount > 0 {
		log.Printf("Rebuilt strip data: %d strips", stripCount)
	}

	res := map[string]interface{}{
		"status":     "ok",
		"samples":    len(data.Samples),
		"strips":     stripCount,
		"duplicates": duplicates,
		// 分布は生データから出す。NN を回さない場合も学習データの中身は見たい
		"distribution": ClassDistribution(data.Samples),
		"nn":           trainNN,
	}

	if !trainNN {
		// **既存の NN モデルは残るが、学習データより古くなる。**
		// 比較用 SFEN を「今の学習データの NN」だと誤読させないための印
		modelLock.Lock()
		nnStale = model != nil
		modelLock.Unlock()
		res["nn_stale"] = nnStale
		// 画面にファイル名を出すのはここから渡す。
		// **UI 側に名前を書かないこと**（版が上がると嘘になる。実際
		// v6 になっても「model_v3.json は古いままです」と出ていた）
		res["model_file"] = modelFile
		log.Printf("Training data updated (%d samples), NN training skipped", len(data.Samples))
		return res, nil
	}

	// クラスバランス調整してから学習
	balanced := BalanceData(data.Samples)
	res["balanced"] = len(balanced)
	res["distribution"] = ClassDistribution(balanced)
	log.Printf("Training: %d raw → %d balanced", len(data.Samples), len(balanced))

	m := Train(&suteme.TrainingData{Samples: balanced})
	SaveModel(modelFile, m)

	modelLock.Lock()
	model = m
	nnStale = false
	modelLock.Unlock()

	log.Printf("Training complete, model saved to %s", modelFile)

	return res, nil
}

// filterByInputSize は入力長が合わないサンプルを取り除き、除いた件数を返す
func filterByInputSize(samples []suteme.TrainingSample) ([]suteme.TrainingSample, int) {
	valid := samples[:0]
	for _, s := range samples {
		if len(s.Input) == suteme.InputSize {
			valid = append(valid, s)
		}
	}
	return valid, len(samples) - len(valid)
}

// handleRecognize: 学習済みモデルで盤面を認識する
func handleRecognize(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Session string `json:"session"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	m := currentPredictor()
	if m == nil {
		http.Error(w, "Model not trained yet", http.StatusBadRequest)
		return
	}

	mu.RLock()
	s, ok := sessions[req.Session]
	mu.RUnlock()
	if !ok || s.Result.Board == nil {
		http.NotFound(w, r)
		return
	}

	// 各マスを認識
	bc := suteme.BoardColor(s.Original, s.Result.Board)
	bo := suteme.NewBoardOrient(s.Original, s.Result.Board, bc)
	cells := make([][]map[string]interface{}, 9)
	for row := 0; row < 9; row++ {
		cells[row] = make([]map[string]interface{}, 9)
		for col := 0; col < 9; col++ {
			cell := s.Result.Board.ExtractCell(s.Original, row, col)
			if cell == nil {
				continue
			}
			// 空/向きは盤の地色を基準に判定（低確信のマスは推論器との回転照合）
			cat, _ := bo.Classify(cell, bc, m)
			if cat == suteme.CellEmpty {
				cells[row][col] = map[string]interface{}{
					"label": "none", "confidence": 95,
				}
				continue
			}
			// 駒種は gobrain で推論（後手は180度回転して先手向きに正規化）
			ncell := cell
			if cat == suteme.CellPieceDown {
				ncell = suteme.Rotate180(cell)
			}
			class, conf := m.Predict(ncell)
			base := suteme.ClassToBaseLabel(class)
			label := base
			if cat == suteme.CellPieceDown {
				if len(base) > 1 && base[0] == '+' {
					label = "+" + strings.ToLower(base[1:])
				} else {
					label = strings.ToLower(base)
				}
			}
			cells[row][col] = map[string]interface{}{
				"label":      label,
				"confidence": int(conf * 100),
			}
		}
	}

	sfen := suteme.RecognizeBoard(s.Original, s.Result.Board, m)
	validation := suteme.ValidatePieces(sfen)

	resp := map[string]interface{}{
		"sfen":       sfen,
		"cells":      cells,
		"validation": validation,
	}
	// k-NN 使用時、gobrain モデルもあれば比較用に NN の結果も返す
	modelLock.RLock()
	usingKNN := knn != nil
	nnModel := model
	stale := nnStale
	modelLock.RUnlock()
	if usingKNN {
		resp["engine"] = "knn"
		if nnModel != nil {
			resp["sfen_nn"] = suteme.RecognizeBoard(s.Original, s.Result.Board, nnModel)
			// 学習データを更新したあと NN を回していないなら、この結果は古い
			resp["nn_stale"] = stale
		}
	} else {
		resp["engine"] = "nn"
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}
