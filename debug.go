package suteme

import (
	"fmt"
	"image"
	"strings"
)

// 認識が「どういう材料でその答えを出したか」を呼び出し側に見せるための情報。
//
// 認識が外れたとき、呼び出し側（ikkyoku など）が持っているのは画像と SFEN だけで、
// **盤面をどこだと思ったのか・どの推論器を使ったのかが分からない**ため、
// 座標の問題なのか認識器の問題なのかを切り分けられなかった。
// Result.Debug はその切り分けに要る材料だけを載せる。
//
// 認識の判断には使わない（あくまで観測用）。値は Recognize が常に埋める。

// RegionSource は盤面領域をどうやって決めたか。
type RegionSource string

const (
	// RegionFromOption は WithRegion / WithRect による明示指定。
	// **この場合 ValidateBoard の値でゲートしていない**ので、
	// Confidence が低くても領域はそのまま使われる。
	RegionFromOption RegionSource = "option"
	// RegionFromDetect は DetectBoard による検出。
	RegionFromDetect RegionSource = "detect"
	// RegionFromWholeImage は「画像全体が盤面」フォールバック
	// （盤だけを切り出した画像でグリッド線が画像端に来る場合）。
	RegionFromWholeImage RegionSource = "whole"
)

// PredictorDebug は駒種推論器の素性。
type PredictorDebug struct {
	// Kind は "knn" / "nn" / それ以外は Go の型名。
	Kind string `json:"kind"`
	// Source は読み込み元のファイルパス（SetPredictor / WithPredictor で
	// 直接渡された場合は空）。
	Source string `json:"source,omitempty"`
	// Detail は種別ごとの補足（k-NN なら "k=1, samples=4455"）。
	Detail string `json:"detail,omitempty"`
}

func (p PredictorDebug) String() string {
	s := p.Kind
	if p.Detail != "" {
		s += "(" + p.Detail + ")"
	}
	if p.Source != "" {
		s += " <- " + p.Source
	}
	return s
}

// DebugPredictor は自分の素性を説明できる Predictor。
// 実装していない推論器は型名だけが記録される。
type DebugPredictor interface {
	Predictor
	Debug() PredictorDebug
}

func predictorDebug(p Predictor) PredictorDebug {
	if d, ok := p.(DebugPredictor); ok {
		return d.Debug()
	}
	if p == nil {
		return PredictorDebug{Kind: "none"}
	}
	return PredictorDebug{Kind: fmt.Sprintf("%T", p)}
}

// CellDebug はマス1つ分の認識過程。
type CellDebug struct {
	Row int `json:"row"` // 0 = 上段（後手側）
	Col int `json:"col"` // 0 = 左（9筋）
	// Rect は切り出しに使った矩形（ExtractCell の 15% 拡張は含まない）。
	Rect image.Rectangle `json:"rect"`
	// Category は最終的に採用した 空/先手/後手（0=空 / 1=先手 / 2=後手）。
	Category CellCategory `json:"category"`
	// OrientBy は向きの決め方。回転照合で決めたなら "match"、
	// 照合できず分類器の幅プロファイルに落ちたときは ""。
	// 推論器が `OrientationMatcher` を実装していれば通常は "match" になる。
	OrientBy string `json:"orient_by,omitempty"`
	// PieceBy は駒があると決めた材料。分類器（被覆率）が空と言ったのを
	// 駒サンプルとの照合で駒に戻したなら "match"（`overturnEmpty`）、
	// 分類器が駒と言ったなら ""。
	PieceBy string `json:"piece_by,omitempty"`
	// EmptyBy は分類器が駒と言ったのを空にした材料。窓を ±1px ずらした照合で
	// 空サンプルと一致したなら "shift"（`emptyByShift`）。ずらさない窓で一致した
	// ときは従来どおり Class が ClassEmpty になる（こちらは ""）。
	EmptyBy string `json:"empty_by,omitempty"`
	// Class は Predictor が返した駒種クラス（推論しなかったら -1、
	// ClassEmpty なら分類を覆して空にしたということ）。
	Class int `json:"class"`
	// Confidence は Predictor が返した確信度。
	Confidence float64 `json:"confidence"`
	// Piece は最終的に採用した SFEN 表記（空マスなら ""）。
	Piece string `json:"piece"`
	// Hidden は「このマスは見えない」（手や頭などが盤に被っている）という印。
	// **Piece / Category はそのまま読んだ結果で、空きにはしていない**
	// （空きは「駒が無い」という読みなので、手の根拠になってしまう）。
	// 呼び出し側は Hidden のマスを盤面の根拠に使わないこと（判定は visible.go）。
	Hidden bool `json:"hidden,omitempty"`
	// Cover は分類器の一次マスクの被覆率（地色と異なる画素の割合）。
	// `Debug.EmptyCover` 未満なら分類器は空と判定している。**-1 は測れなかった**
	// （マスのほぼ全体が盤の地色と違う。直前の手の色付けなど）。
	Cover float64 `json:"cover"`
}

// Debug は認識1回分の観測情報。
type Debug struct {
	// ImageBounds は入力画像の範囲。
	ImageBounds image.Rectangle `json:"image_bounds"`
	// Region は盤面と判定した外枠の矩形。マス割りはこれを9等分したもの。
	Region image.Rectangle `json:"region"`
	// RegionSource は Region の決め方。
	RegionSource RegionSource `json:"region_source"`
	// Confidence は ValidateBoard の値（Result.Confidence と同じ）。
	// **RegionFromOption のときはゲートに使っていない**（手動指定は検証せず
	// そのまま使う）ので、低くても領域はその座標のまま。
	Confidence float64 `json:"confidence"`
	// BoardColor は盤の地色（分類の基準にした輝度の中央値）。
	BoardColor uint8 `json:"board_color"`
	// EmptyCover はこの盤の空判定の境目（被覆率。`BoardEmptyCover`）。
	EmptyCover float64 `json:"empty_cover"`
	// Predictor は使った駒種推論器。
	Predictor PredictorDebug `json:"predictor"`
	// Cells は81マスの認識過程（行優先）。
	Cells []CellDebug `json:"cells"`
}

// String は1行の要約を返す（ログ用）。
func (d *Debug) String() string {
	if d == nil {
		return "<no debug>"
	}
	r := d.Region
	return fmt.Sprintf("region=(%d,%d)-(%d,%d) cell=%dx%d src=%s conf=%.2f predictor=%s",
		r.Min.X, r.Min.Y, r.Max.X, r.Max.Y, r.Dx()/9, r.Dy()/9,
		d.RegionSource, d.Confidence, d.Predictor)
}

// Cell は指定マスの認識過程を返す（範囲外なら nil）。
func (d *Debug) Cell(row, col int) *CellDebug {
	if d == nil || row < 0 || row > 8 || col < 0 || col > 8 {
		return nil
	}
	i := row*9 + col
	if i >= len(d.Cells) {
		return nil
	}
	return &d.Cells[i]
}

// LowConfidenceCells は確信度が th 未満だった駒マスを返す。
// 認識が外れたときに「どのマスが怪しかったか」を絞り込むためのもの。
func (d *Debug) LowConfidenceCells(th float64) []CellDebug {
	if d == nil {
		return nil
	}
	var out []CellDebug
	for _, c := range d.Cells {
		if c.Category != CellEmpty && c.Confidence < th {
			out = append(out, c)
		}
	}
	return out
}

// HiddenCells は「見えない」（手や頭などが被っている）マスを返す。
func (d *Debug) HiddenCells() []CellDebug {
	if d == nil {
		return nil
	}
	var out []CellDebug
	for _, c := range d.Cells {
		if c.Hidden {
			out = append(out, c)
		}
	}
	return out
}

// Dump は盤の形に並べた文字列を返す（マスの表記と確信度。見えないマスは表記の前に `#`）。
func (d *Debug) Dump() string {
	if d == nil {
		return "<no debug>"
	}
	var b strings.Builder
	b.WriteString(d.String())
	b.WriteByte('\n')
	for r := 0; r < 9; r++ {
		for c := 0; c < 9; c++ {
			cell := d.Cell(r, c)
			if cell == nil {
				continue
			}
			p := cell.Piece
			if p == "" {
				p = "."
			}
			mark := " "
			if cell.Hidden {
				mark = "#"
			}
			fmt.Fprintf(&b, "%s%-3s%3.0f%%", mark, p, cell.Confidence*100)
		}
		b.WriteByte('\n')
	}
	return b.String()
}
