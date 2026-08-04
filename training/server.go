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

	shinteweb "github.com/ShinteLab/core/web"
	"github.com/ShinteLab/suteme"
)

//go:embed static
var static embed.FS

type session struct {
	Original image.Image
	Result   *suteme.AnalyzeResult
	BoardImg *image.RGBA
	Labels   map[string]string
}

const (
	dataDir     = "data"
	historyFile = "data/history.json"
)

type BoardBounds struct {
	X1, Y1, X2, Y2 int
}

type HistoryEntry struct {
	ID          string       `json:"id"`
	CreatedAt   string       `json:"created_at"`
	SFEN        string       `json:"sfen"`
	BoardBounds *BoardBounds `json:"board_bounds,omitempty"`
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

func saveHistoryFile(h *HistoryData) {
	os.MkdirAll(dataDir, 0755)
	f, err := os.Create(historyFile)
	if err != nil {
		log.Printf("saveHistory: %v", err)
		return
	}
	defer f.Close()
	json.NewEncoder(f).Encode(h)
}

var (
	sessions    = make(map[string]*session)
	mu          sync.RWMutex
	model       *suteme.Model
	knn         *suteme.KNN
	// v2: 入力の標準化・後手の回転正規化・空マス除外に対応した形式
	// （旧 training_data.json は空マスが歩として混入しているため使用しない）
	dataFile    = "training_data_v2.json"
	modelFile   = "model_v2.json"
	modelLock   sync.RWMutex
	historyMu   sync.RWMutex
)

// Serve はラベリング・学習用のWebサーバを起動する
// 起動時にカレントディレクトリの model_v2.json / training_data_v2.json を自動ロードする
func Serve(port string) error {
	// 起動時に保存済みモデルを読み込む
	if m, err := suteme.LoadModel(modelFile); err == nil {
		model = m
		log.Printf("Loaded model from %s", modelFile)
	}
	// 学習データがあれば k-NN を構築（学習処理は不要）
	if data, err := suteme.LoadTrainingData(dataFile); err == nil {
		if kn := suteme.NewKNN(data.Samples); kn != nil {
			knn = kn
			log.Printf("Built k-NN from %d samples in %s", kn.Len(), dataFile)
		}
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
	mux.HandleFunc("/api/history/", handleHistoryImage)
	mux.HandleFunc("/api/savesession", handleSaveSession)
	mux.HandleFunc("/api/label", handleLabel)
	mux.HandleFunc("/api/labelbulk", handleLabelBulk)
	mux.HandleFunc("/api/setboard", handleSetBoard)
	mux.HandleFunc("/api/train", handleTrain)
	mux.HandleFunc("/api/recognize", handleRecognize)

	addr := ":" + port
	fmt.Printf("http://localhost%s\n", addr)
	return http.ListenAndServe(addr, mux)
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
	for row := 0; row < 9; row++ {
		for col := 0; col < 9; col++ {
			cell := s.Result.Board.ExtractCell(s.Original, row, col)
			if cell == nil {
				continue
			}
			cat := suteme.ClassifyCellWith(cell, bc)
			if cat == suteme.CellEmpty {
				grid[row][col] = map[string]interface{}{"label": "none", "confidence": 95}
				continue
			}
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
			grid[row][col] = map[string]interface{}{
				"label":      label,
				"confidence": int(conf * 100),
			}
		}
	}
	return &grid
}

// isGoteLabel はSFENラベルが後手（小文字）かを返す
func isGoteLabel(label string) bool {
	s := strings.TrimPrefix(label, "+")
	if s == "" {
		return false
	}
	return s[0] >= 'a' && s[0] <= 'z'
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
		Labels:   make(map[string]string),
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

func handleLabel(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Session string `json:"session"`
		Row     int    `json:"row"`
		Col     int    `json:"col"`
		Piece   string `json:"piece"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	mu.Lock()
	s, ok := sessions[req.Session]
	if ok {
		key := strconv.Itoa(req.Row) + "-" + strconv.Itoa(req.Col)
		s.Labels[key] = req.Piece
	}
	mu.Unlock()

	if !ok {
		http.NotFound(w, r)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func handleHistory(w http.ResponseWriter, r *http.Request) {
	historyMu.RLock()
	h := loadHistory()
	historyMu.RUnlock()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(h)
}

func handleHistoryImage(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/api/history/")
	id = strings.TrimSuffix(id, "/image")
	for _, ch := range id {
		if !((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f')) {
			http.NotFound(w, r)
			return
		}
	}
	f, err := os.Open(filepath.Join(dataDir, id+".png"))
	if err != nil {
		http.NotFound(w, r)
		return
	}
	defer f.Close()
	w.Header().Set("Content-Type", "image/png")
	w.Header().Set("Cache-Control", "max-age=86400")
	io.Copy(w, f)
}

func handleSaveSession(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		Session string `json:"session"`
		SFEN    string `json:"sfen"`
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

	id := newID()
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
		SFEN:      req.SFEN,
	}
	if s.Result != nil && s.Result.Board != nil {
		b := s.Result.Board.Bounds
		entry.BoardBounds = &BoardBounds{b.Min.X, b.Min.Y, b.Max.X, b.Max.Y}
	}
	historyMu.Lock()
	h := loadHistory()
	h.Entries = append([]HistoryEntry{entry}, h.Entries...)
	if len(h.Entries) > 100 {
		for _, old := range h.Entries[100:] {
			os.Remove(filepath.Join(dataDir, old.ID+".png"))
		}
		h.Entries = h.Entries[:100]
	}
	saveHistoryFile(h)
	historyMu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok", "id": id})
}

// handleLabelBulk: 複数マスのラベルを一括登録する
func handleLabelBulk(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		Session string `json:"session"`
		Labels  []struct {
			Row   int    `json:"row"`
			Col   int    `json:"col"`
			Piece string `json:"piece"`
		} `json:"labels"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	mu.Lock()
	s, ok := sessions[req.Session]
	if ok {
		for _, l := range req.Labels {
			key := strconv.Itoa(l.Row) + "-" + strconv.Itoa(l.Col)
			s.Labels[key] = l.Piece
		}
	}
	mu.Unlock()
	if !ok {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
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
		s.Labels = make(map[string]string)
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

// handleTrain: ラベル済みデータで学習する
func handleTrain(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	defer func() {
		if rec := recover(); rec != nil {
			log.Printf("handleTrain panic: %v", rec)
			http.Error(w, fmt.Sprintf("training failed: %v", rec), http.StatusInternalServerError)
		}
	}()

	// 既存の学習データを読み込む
	var existing []suteme.TrainingSample
	if e, err := suteme.LoadTrainingData(dataFile); err == nil {
		existing = e.Samples
		log.Printf("Loaded %d existing samples from %s", len(existing), dataFile)
	}

	// 全セッションのラベルデータを集める
	mu.RLock()
	var fresh []suteme.TrainingSample
	newCount := 0
	for _, s := range sessions {
		if s.Result.Board == nil {
			continue
		}
		for key, label := range s.Labels {
			// 空マスは駒種学習の対象外（"none" を LabelToClass に通すと誤学習する）
			class := suteme.LabelToClass(label)
			if class < 0 {
				continue
			}
			parts := strings.Split(key, "-")
			if len(parts) != 2 {
				continue
			}
			row, _ := strconv.Atoi(parts[0])
			col, _ := strconv.Atoi(parts[1])
			cell := s.Result.Board.ExtractCell(s.Original, row, col)
			if cell == nil {
				continue
			}
			// 後手（小文字ラベル）は180度回転して先手向きに正規化
			if isGoteLabel(label) {
				cell = suteme.Rotate180(cell)
			}
			fresh = append(fresh, suteme.TrainingSample{
				Input: suteme.CellToInput(cell),
				Label: class,
			})
			newCount++
		}
	}
	mu.RUnlock()

	// 入力長が合わないサンプルを除外
	existing, skippedOld := filterByInputSize(existing)
	fresh, skippedNew := filterByInputSize(fresh)
	if skipped := skippedOld + skippedNew; skipped > 0 {
		log.Printf("Skipped %d samples with wrong input size", skipped)
	}

	// 既存分と今回分を入力の内容でマージして重複を除く。
	// これが無いと「学習」を押すたびに同じサンプルが二重に積まれる
	// （毎回「既存ファイルの全件 + 全セッションのラベル」を保存し直すため）。
	beforeMerge := len(existing) + len(fresh)
	var data suteme.TrainingData
	data.Samples = MergeSamples(existing, fresh)
	duplicates := beforeMerge - len(data.Samples)
	if duplicates > 0 {
		log.Printf("Merged %d samples → %d (%d duplicates removed)", beforeMerge, len(data.Samples), duplicates)
	}

	if len(data.Samples) == 0 {
		http.Error(w, "No labeled data", http.StatusBadRequest)
		return
	}

	// 学習データを保存（生データ全件）
	suteme.SaveTrainingData(dataFile, &data)

	// k-NN は生データから即再構築（学習不要）
	if kn := suteme.NewKNN(data.Samples); kn != nil {
		modelLock.Lock()
		knn = kn
		modelLock.Unlock()
	}

	// クラスバランス調整してから学習
	balanced := BalanceData(data.Samples)
	dist := ClassDistribution(balanced)
	log.Printf("Training: %d raw → %d balanced (%d labeled this run)", len(data.Samples), len(balanced), newCount)

	m := Train(&suteme.TrainingData{Samples: balanced})

	// モデルを保存
	SaveModel(modelFile, m)

	modelLock.Lock()
	model = m
	modelLock.Unlock()

	log.Printf("Training complete, model saved to %s", modelFile)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":       "ok",
		"samples":      len(data.Samples),
		"balanced":     len(balanced),
		"duplicates":   duplicates,
		"distribution": dist,
	})
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
	cells := make([][]map[string]interface{}, 9)
	for row := 0; row < 9; row++ {
		cells[row] = make([]map[string]interface{}, 9)
		for col := 0; col < 9; col++ {
			cell := s.Result.Board.ExtractCell(s.Original, row, col)
			if cell == nil {
				continue
			}
			// 空/向きは盤の地色を基準に判定
			cat := suteme.ClassifyCellWith(cell, bc)
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
	modelLock.RUnlock()
	if usingKNN {
		resp["engine"] = "knn"
		if nnModel != nil {
			resp["sfen_nn"] = suteme.RecognizeBoard(s.Original, s.Result.Board, nnModel)
		}
	} else {
		resp["engine"] = "nn"
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}
