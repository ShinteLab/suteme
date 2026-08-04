package suteme

import (
	"errors"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"sync"
)

// 既定の認識器ファイル名（training パッケージが書き出すものと同じ）
const (
	DefaultDataFile  = "training_data_v2.json"
	DefaultModelFile = "model_v2.json"
)

// minBoardConfidence は盤面領域を採用する最低信頼度（ValidateBoard の値）。
// これを下回る検出結果は「盤面ではない」として棄却する。
const minBoardConfidence = 0.5

// ErrBoardNotFound は画像から盤面領域を特定できなかったことを表す
var ErrBoardNotFound = errors.New("盤面を検出できませんでした")

// ErrNoPredictor は駒種推論器（k-NN / NN）が用意できなかったことを表す
var ErrNoPredictor = errors.New("駒種推論器がありません")

var (
	predictorMu sync.RWMutex
	predictor   Predictor // SetPredictor で明示指定、または既定探索の結果をキャッシュ
)

// SetPredictor は LoadSFEN が使う駒種推論器を差し替える。
// nil を渡すと既定の探索（LoadPredictor によるファイル探索）に戻る。
func SetPredictor(p Predictor) {
	predictorMu.Lock()
	defer predictorMu.Unlock()
	predictor = p
}

// LoadPredictor は dir から駒種推論器を読み込む。
// training_data_v2.json があれば k-NN を優先する（学習処理が要らず、
// サンプルを足した瞬間に反映されるため）。無ければ model_v2.json の
// gobrain モデルを使う。
func LoadPredictor(dir string) (Predictor, error) {
	if data, err := LoadTrainingData(filepath.Join(dir, DefaultDataFile)); err == nil {
		if kn := NewKNN(data.Samples); kn != nil {
			return kn, nil
		}
	}
	if m, err := LoadModel(filepath.Join(dir, DefaultModelFile)); err == nil {
		return m, nil
	}
	return nil, fmt.Errorf("%w: %s に %s / %s がありません",
		ErrNoPredictor, dir, DefaultDataFile, DefaultModelFile)
}

// defaultPredictor は既定の推論器を返す。
// カレントディレクトリ → 実行ファイルのディレクトリの順に探し、
// 見つかった結果はキャッシュする（失敗はキャッシュしないので、
// あとからファイルを置けば次の呼び出しで拾える）。
func defaultPredictor() (Predictor, error) {
	predictorMu.RLock()
	p := predictor
	predictorMu.RUnlock()
	if p != nil {
		return p, nil
	}

	dirs := []string{"."}
	if exe, err := os.Executable(); err == nil {
		dirs = append(dirs, filepath.Dir(exe))
	}
	var lastErr error
	for _, dir := range dirs {
		found, err := LoadPredictor(dir)
		if err != nil {
			lastErr = err
			continue
		}
		predictorMu.Lock()
		if predictor == nil {
			predictor = found
		}
		p = predictor
		predictorMu.Unlock()
		return p, nil
	}
	return nil, lastErr
}

// LoadSFEN は画像から盤面を認識し、SFEN の盤面部分（'/' 区切りの9段）を返す。
// 手番・持ち駒・手数は付けない（suteme の責務は「画像 → 盤面」に閉じる）。
// 持ち駒の枚数が要るなら、戻り値を ValidatePieces に渡すこと。
//
// 駒種の推論器はカレントディレクトリ、次に実行ファイルのディレクトリから
// 自動で読み込む（training_data_v2.json → k-NN 優先、無ければ model_v2.json）。
// 明示的に指定する場合は SetPredictor / LoadSFENWith を使う。
func LoadSFEN(img image.Image) (string, error) {
	p, err := defaultPredictor()
	if err != nil {
		return "", err
	}
	return LoadSFENWith(img, p)
}

// LoadSFENWith は推論器を指定して画像から SFEN 盤面文字列を認識する。
func LoadSFENWith(img image.Image, p Predictor) (string, error) {
	if img == nil {
		return "", errors.New("画像がありません")
	}
	if p == nil {
		return "", ErrNoPredictor
	}
	br, conf := detectBoardRegion(img)
	if br == nil {
		return "", fmt.Errorf("%w（信頼度 %.0f%%）", ErrBoardNotFound, conf*100)
	}
	return RecognizeBoard(img, br, p), nil
}

// detectBoardRegion は認識に使う盤面領域とその信頼度を返す。
// グリッド検出（DetectBoard）を第一候補とし、信頼度が足りない場合のみ
// 「画像全体が盤面」の候補を試す。ikkyoku のガイド枠のように盤だけを
// 切り出した画像はグリッド線が画像端に来て検出が外れるため。
// どちらも minBoardConfidence に届かなければ nil を返す。
func detectBoardRegion(img image.Image) (*BoardRegion, float64) {
	best := DetectBoard(img)
	conf := ValidateBoard(img, best) // br が nil なら 0

	if conf < minBoardConfidence {
		b := img.Bounds()
		whole := BoardRegionFromRect(b.Min.X, b.Min.Y, b.Max.X, b.Max.Y)
		if c := ValidateBoard(img, whole); c > conf {
			best, conf = whole, c
		}
	}

	if conf < minBoardConfidence {
		return nil, conf
	}
	return best, conf
}

func ViewDebug(img image.Image) error {
	r := Analyze(img)
	out := r.DrawBoard(img)

	SaveImage("debug_board.png", out)
	SaveImage("debug_edges.png", r.Edges)

	if r.Board != nil {
		fmt.Printf("Board: %v\n", r.Board.Bounds)
	} else {
		fmt.Println("Board: not found")
	}
	fmt.Println("Saved: debug_board.png, debug_edges.png")
	return nil
}
