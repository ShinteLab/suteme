package training

// 盤面が写っていない画像（負例）。
//
// **誤検出率だけは、正解データからは測れない。** 評価タブの 3 系統（検出のずれ /
// 手動座標での認識 / 自動座標での認識）はどれも「盤がある画像」が前提なので、
// 盤の無い画像に信頼度 1.00 を返しても数字に現れない。実際 CLAUDE.md の
// 「off がほぼ 0 の画面で gridAlignment が +1 に張り付く」は、一様な背景に線が
// 数本あるだけの画像が **信頼度 1.00 で通る**という穴で、今のところ
// この方向の劣化を検知する手段が無い。ikkyoku の実運用では盤が映っていない
// 時間（CM・インタビュー・対局者のアップ）のほうが長いので、ここが通ると
// 誤った SFEN を掴む。
//
// **ラベルは置き場所そのもの。** data/negative/{id}.png に画像が 1 枚ある
// ことが「この画像には盤面が無い」という宣言で、SFEN も座標も注記も要らない。
// メタデータのファイルを持たないので、**ディレクトリに画像をコピーするだけで
// 運用を始められる**（UI を通さなくてよい）。重複判定も、件数が二桁なので
// その場でファイルのハッシュを取れば足りる。
//
// **history.json には入れない。** 履歴は手でラベル付けした正解＝作り直せない
// 資産で、上限に達したら 507 で断る扱いにしてある。負例は測り直しがきく
// 消耗品なので、accuracy.json と同じく別枠にする。学習
// （samplesFromHistory）に紛れ込ませないためでもある。

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	"image/png"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/ShinteLab/suteme"
)

const (
	negativeDir = "data/negative"
	// maxNegatives は負例の枚数の上限。**多く集めても効かない。**
	// 効くのは枚数ではなく中身（盤に似た格子を持つ絵）なので、
	// 際限なく溜めるより上限で気付けるほうがよい
	maxNegatives = 200
)

var negativeMu sync.Mutex

// NegativeResult は負例 1 枚の判定結果。
//
// **棄却したものも信頼度を残す。** 「棄却できた」だけだと余裕を持って
// 落としているのか閾値ぎりぎりなのかが分からず、次の版で通るようになる
// 兆候を見逃す（EvalDetect が棄却された候補を残しているのと同じ理由）。
type NegativeResult struct {
	ID string `json:"id"`
	// Detected が真なら盤面として採用された ＝ 誤検出
	Detected bool `json:"detected"`
	// Source は領域の決め方（"detect" / "whole"）。候補が無ければ空
	Source string `json:"source,omitempty"`
	// Confidence は ValidateBoard の値
	Confidence float64 `json:"confidence"`
	// 盤とみなした矩形。候補が無ければすべて 0
	X1 int `json:"x1"`
	Y1 int `json:"y1"`
	X2 int `json:"x2"`
	Y2 int `json:"y2"`
}

// NegativeEval は負例に対する誤検出の記録。
type NegativeEval struct {
	Images   int              `json:"images"`
	Rejected int              `json:"rejected"`
	Results  []NegativeResult `json:"results"`
}

// negativeIDs は data/negative/ にある負例の ID を返す（ファイル名から拡張子を除いたもの）。
func negativeIDs() []string {
	ents, err := os.ReadDir(negativeDir)
	if err != nil {
		return nil
	}
	var ids []string
	for _, e := range ents {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".png") {
			continue
		}
		if id := strings.TrimSuffix(e.Name(), ".png"); validID(id) {
			ids = append(ids, id)
		}
	}
	sort.Strings(ids)
	return ids
}

func negativePath(id string) string {
	return filepath.Join(negativeDir, id+".png")
}

func loadNegativeImage(id string) (image.Image, error) {
	f, err := os.Open(negativePath(id))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return png.Decode(f)
}

// evaluateNegatives は負例それぞれについて「盤面として採用されてしまうか」を測る。
//
// **suteme.Recognize をそのまま通す。** DetectBoard + ValidateBoard を直接呼ぶと
// 「画像全体が盤面」フォールバック（detectBoardRegion）を通る誤検出を数え落とす。
// 認識まで走るのは誤検出したときだけなので、棄却できている限り検出のぶんしかかからない。
//
// **判定できなかった画像（推論器が無い等）は棄却として数えない。**
// 数えると誤検出率が実態より良く見える。
//
// 負例が 1 枚も無ければ nil を返す（記録に空の欄を作らない）。
func evaluateNegatives(p suteme.Predictor) *NegativeEval {
	ids := negativeIDs()
	if len(ids) == 0 {
		return nil
	}
	ev := &NegativeEval{}
	for _, id := range ids {
		img, err := loadNegativeImage(id)
		if err != nil {
			log.Printf("evaluate: 負例 %s を読めません (%v)", id, err)
			continue
		}
		res := NegativeResult{ID: id}
		// 既定のオプションは「全部調べるがエラーにはしない」なので、
		// 盤が採用されれば err なしで Result が返る
		r, err := suteme.Recognize(img, suteme.WithPredictor(p))
		switch {
		case r != nil && r.Debug != nil:
			res.Detected = true
			res.Source = string(r.Debug.RegionSource)
			res.Confidence = r.Debug.Confidence
			res.X1, res.Y1 = r.Debug.Region.Min.X, r.Debug.Region.Min.Y
			res.X2, res.Y2 = r.Debug.Region.Max.X, r.Debug.Region.Max.Y
		case errors.Is(err, suteme.ErrBoardNotFound):
			// 棄却された候補の信頼度も残す。**閾値ぎりぎりで落ちているのか
			// 余裕があるのかは、これでないと分からない**
			if br := suteme.DetectBoard(img); br != nil {
				res.Source = string(suteme.RegionFromDetect)
				res.Confidence = suteme.ValidateBoard(img, br)
				res.X1, res.Y1 = br.Bounds.Min.X, br.Bounds.Min.Y
				res.X2, res.Y2 = br.Bounds.Max.X, br.Bounds.Max.Y
			}
		default:
			// 推論器が無いなど、盤があったかどうかを判定できていない。
			// **棄却として数えない。** 数えると誤検出率が実態より良く見える
			log.Printf("evaluate: 負例 %s を判定できません (%v)", id, err)
			continue
		}
		ev.Images++
		if !res.Detected {
			ev.Rejected++
		}
		ev.Results = append(ev.Results, res)
		log.Printf("evaluate(負例) %s: %s conf=%.2f", id,
			map[bool]string{true: "誤検出", false: "棄却"}[res.Detected], res.Confidence)
	}
	if ev.Images == 0 {
		return nil
	}
	return ev
}

// handleNegatives は負例の一覧・登録（ループバック限定。externalAllowed に無い）。
//
//	GET  /api/negative   一覧（ID だけ。判定結果は評価の記録側にある）
//	POST /api/negative   負例を登録（multipart の image、または JSON の {"session": id}）
func handleNegatives(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		w.Header().Set("Content-Type", "application/json")
		ids := negativeIDs()
		if ids == nil {
			ids = []string{}
		}
		json.NewEncoder(w).Encode(map[string]interface{}{"ids": ids, "max": maxNegatives})
	case http.MethodPost:
		handleNegativeAdd(w, r)
	default:
		http.Error(w, "GET or POST only", http.StatusMethodNotAllowed)
	}
}

// negativeImageFromRequest は登録する画像を取り出す。
// 画面からは「今見ている画像」を指すだけで済むようセッション ID を受け付け、
// コマンドラインからは multipart で直接送れるようにしてある
func negativeImageFromRequest(w http.ResponseWriter, r *http.Request) (image.Image, error) {
	if strings.HasPrefix(r.Header.Get("Content-Type"), "multipart/") {
		r.Body = http.MaxBytesReader(w, r.Body, maxUploadBytes)
		if err := r.ParseMultipartForm(maxUploadBytes); err != nil {
			return nil, fmt.Errorf("リクエストを読めません: %w", err)
		}
		file, _, err := r.FormFile("image")
		if err != nil {
			return nil, fmt.Errorf("image がありません")
		}
		defer file.Close()
		raw, err := io.ReadAll(file)
		if err != nil {
			return nil, err
		}
		img, _, err := image.Decode(bytes.NewReader(raw))
		if err != nil {
			return nil, fmt.Errorf("画像をデコードできません: %w", err)
		}
		return img, nil
	}
	var req struct {
		Session string `json:"session"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		return nil, err
	}
	mu.RLock()
	s := sessions[req.Session]
	mu.RUnlock()
	if s == nil || s.Original == nil {
		return nil, fmt.Errorf("セッションが見つかりません")
	}
	return s.Original, nil
}

func handleNegativeAdd(w http.ResponseWriter, r *http.Request) {
	img, err := negativeImageFromRequest(w, r)
	if err != nil {
		httpJSONError(w, http.StatusBadRequest, err.Error())
		return
	}
	// 保存形式（PNG）で符号化してからハッシュを取る。同じ画像をセッション経由と
	// ファイル経由の両方から送っても同じバイト列になり、重複判定が効く
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		httpJSONError(w, http.StatusInternalServerError, err.Error())
		return
	}
	sum := sha256.Sum256(buf.Bytes())

	negativeMu.Lock()
	defer negativeMu.Unlock()
	ids := negativeIDs()
	for _, id := range ids {
		b, err := os.ReadFile(negativePath(id))
		if err != nil {
			continue
		}
		if s := sha256.Sum256(b); s == sum {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{
				"status": "ok", "id": id, "duplicate": true, "count": len(ids),
			})
			return
		}
	}
	if len(ids) >= maxNegatives {
		httpJSONError(w, http.StatusInsufficientStorage,
			fmt.Sprintf("負例が上限（%d 枚）です。不要なものを削除してください", maxNegatives))
		return
	}
	if err := os.MkdirAll(negativeDir, 0755); err != nil {
		httpJSONError(w, http.StatusInternalServerError, err.Error())
		return
	}
	id := newID()
	if err := os.WriteFile(negativePath(id), buf.Bytes(), 0644); err != nil {
		httpJSONError(w, http.StatusInternalServerError, err.Error())
		return
	}
	log.Printf("negative: %s を登録しました（%d 枚）", id, len(ids)+1)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status": "ok", "id": id, "count": len(ids) + 1,
	})
}

// handleNegativeItem は負例 1 件の画像取得・削除（ループバック限定）。
//
//	GET    /api/negative/{id}/image
//	DELETE /api/negative/{id}
func handleNegativeItem(w http.ResponseWriter, r *http.Request) {
	path := strings.Trim(strings.TrimPrefix(r.URL.Path, "/api/negative/"), "/")
	id := strings.TrimSuffix(path, "/image")
	if !validID(id) {
		http.NotFound(w, r)
		return
	}
	switch {
	case r.Method == http.MethodGet && strings.HasSuffix(path, "/image"):
		f, err := os.Open(negativePath(id))
		if err != nil {
			http.NotFound(w, r)
			return
		}
		defer f.Close()
		w.Header().Set("Content-Type", "image/png")
		io.Copy(w, f)
	case r.Method == http.MethodDelete:
		negativeMu.Lock()
		err := os.Remove(negativePath(id))
		negativeMu.Unlock()
		if err != nil {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok", "id": id})
	default:
		http.NotFound(w, r)
	}
}
