package suteme

import (
	"errors"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"sync"

	"github.com/ShinteLab/core/sfen"
)

// 既定の認識器ファイル名（training パッケージが書き出すものと同じ）。
//
// **入力ベクトルの作り方（CellToInput / resizeGray）を変えたら版を上げること。**
// MergeSamples は入力の内容でマージするので、表現の違うサンプルを同じファイルに
// 混ぜると古いものが消えずに残り続ける。
// v8: 空マス（ClassEmpty）の入力をマス全体で作るようにした（CellToInputFull 参照。
//
//	外接矩形は駒にしか定義できず、空マスでは木目を囲んだでたらめな矩形になる。
//	駒サンプルの表現は v7 と同じ）
//
// v7: 盤面座標を `SnapToGrid` で格子線へ寄せてから切り出すようにした
//
//	（学習データも検出も同じ一点で切り出す。全マスの入力が変わる）
//
// v6: ExtractCell の余白を盤の外枠でクリップした（外周 32 マスの切り出しが変わる）
// v5: 駒の外接矩形に切り揃えてから 24x24 に落とすようにした（CellToInput 参照）
// v4: 入力ベクトルを float32 の精度へ丸め、ファイルもバイナリにした
//
//	（JSON は float64 1 個が 20 文字前後になり 186MB / 読み込み 1.8s だった）
//
// v3: リサイズを最近傍法から面積平均に変更（v2 は標準化・回転正規化・空クラス対応）
const (
	DefaultDataFile  = "training_data_v8.bin"
	DefaultModelFile = "model_v8.json"
)

// LegacyDataFiles は既定のファイルが無いときに読みにいく旧版の学習データ。
//
// **v7 以前は入れない。** v8 で空マスの入力ベクトルが別物になったので、
// 混ぜると `MergeSamples` が畳めず、切り揃えた古い空サンプルが残り続ける
// （そちらは推論時に作る空の入力と一致しないので、ただの死荷重になる）。
// 学習データは `data/history.json` の画像と正解 SFEN から作り直せるので、
// 履歴タブで全件を選んで学習し直せば v8 のファイルができる。
var LegacyDataFiles []string

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
// 学習データ（DefaultDataFile、無ければ LegacyDataFiles）があれば k-NN を
// 優先する（学習処理が要らず、サンプルを足した瞬間に反映されるため）。
// 無ければ DefaultModelFile の gobrain モデルを使う。
func LoadPredictor(dir string) (Predictor, error) {
	for _, name := range append([]string{DefaultDataFile}, LegacyDataFiles...) {
		dataPath := filepath.Join(dir, name)
		data, err := LoadTrainingData(dataPath)
		if err != nil {
			continue
		}
		if kn := NewKNN(data.Samples); kn != nil {
			kn.source = dataPath
			return kn, nil
		}
	}
	modelPath := filepath.Join(dir, DefaultModelFile)
	if m, err := LoadModel(modelPath); err == nil {
		m.source = modelPath
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
// 持ち駒や検証結果も要るなら Recognize を使う。
//
// 駒種の推論器はカレントディレクトリ、次に実行ファイルのディレクトリから
// 自動で読み込む（training_data_v8.bin → k-NN 優先、無ければ model_v8.json）。
// 明示的に指定する場合は SetPredictor / WithPredictor を使う。
//
// 既定では盤面の検証は行うがエラーにはしない。おかしい盤面をエラーにしたい場合は
// WithErrorOn / WithStrict を渡す（違反の内容は BoardError から取れる）。
func LoadSFEN(img image.Image, opts ...Option) (string, error) {
	r, err := Recognize(img, opts...)
	if r == nil {
		return "", err
	}
	return r.Board, err
}

// LoadSFENWith は推論器を指定して画像から SFEN 盤面文字列を認識する。
// LoadSFEN(img, WithPredictor(p), ...) と同じだが、こちらは推論器の指定が必須
// （nil なら ErrNoPredictor）。
func LoadSFENWith(img image.Image, p Predictor, opts ...Option) (string, error) {
	if p == nil {
		return "", ErrNoPredictor
	}
	return LoadSFEN(img, append([]Option{WithPredictor(p)}, opts...)...)
}

// Recognize は画像から盤面を認識し、駒数の検証結果と持ち駒の推定まで含めて返す。
//
// 検証で違反が見つかっても既定ではエラーにしない（画像から出てきた盤面が
// おかしいこと自体は異常ではないため）。WithErrorOn / WithStrict を指定した
// 場合は ErrInvalidBoard を包んだ *BoardError を返すが、**そのときも Result は
// 返す**ので、呼び出し側は「どこがおかしいか」を UI に出せる。
//
//	// 駒数がおかしければエラー。足りない駒は先手の駒台に置く
//	r, err := suteme.Recognize(img,
//	    suteme.WithErrorOn(sfen.CheckPieceCount|sfen.CheckKing),
//	    suteme.WithHandTo(suteme.HandBlack))
func Recognize(img image.Image, opts ...Option) (*Result, error) {
	cfg := defaultConfig().apply(opts)

	if img == nil {
		return nil, errors.New("画像がありません")
	}
	p := cfg.predictor
	if p == nil {
		var err error
		if p, err = defaultPredictor(); err != nil {
			return nil, err
		}
	}
	br, conf, src := cfg.boardRegion(img)
	if br == nil {
		return nil, fmt.Errorf("%w（信頼度 %.0f%%）", ErrBoardNotFound, conf*100)
	}

	board, cells, bc := recognizeBoardDetail(img, br, p)
	info := sfen.Inspect(board, cfg.checks)
	r := newResult(board, conf, info, cfg)
	r.Debug = &Debug{
		ImageBounds:  img.Bounds(),
		Region:       br.Bounds,
		RegionSource: src,
		Confidence:   conf,
		BoardColor:   bc,
		Predictor:    predictorDebug(p),
		Cells:        cells,
	}

	if fatal := info.Filter(cfg.fatal); len(fatal) > 0 {
		return r, &BoardError{Board: board, Violations: fatal}
	}
	return r, nil
}

// detectBoardRegion は認識に使う盤面領域・信頼度・領域の決め方を返す。
// グリッド検出（DetectBoard）を第一候補とし、信頼度が足りない場合のみ
// 「画像全体が盤面」の候補を試す。ikkyoku のガイド枠のように盤だけを
// 切り出した画像はグリッド線が画像端に来て検出が外れるため。
// どちらも minBoardConfidence に届かなければ nil を返す。
func detectBoardRegion(img image.Image) (*BoardRegion, float64, RegionSource) {
	best := DetectBoard(img)
	conf := ValidateBoard(img, best) // br が nil なら 0
	src := RegionFromDetect

	if conf < minBoardConfidence {
		b := img.Bounds()
		whole := BoardRegionFromRect(b.Min.X, b.Min.Y, b.Max.X, b.Max.Y)
		if c := ValidateBoard(img, whole); c > conf {
			best, conf, src = whole, c, RegionFromWholeImage
		}
	}

	if conf < minBoardConfidence {
		return nil, conf, src
	}
	return best, conf, src
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
