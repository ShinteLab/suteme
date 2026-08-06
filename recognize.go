package suteme

import (
	"encoding/json"
	"image"
	"image/color"
	"math"
	"os"
	"strings"

	"github.com/ShinteLab/core/sfen"
	"github.com/goml/gobrain"
)

const (
	cellSize  = 24
	inputSize = cellSize * cellSize
	// ClassEmpty は空マスのクラスID。駒種 14 クラスの次に置く。
	//
	// 空判定は ClassifyCellWith（地色との被覆率）が一次判定を担うが、
	// グリッド線・木目・隣のマスの駒の写り込みは「地色と異なる画素」を
	// 増やすため、被覆率が駒と重なって原理的に分離できないマスが出る
	// （実測で空マスの被覆率 0.38 に対し同一画像の駒の最小が 0.36）。
	// 空も学習対象にすることで、画素の並びで照合できるようにする。
	ClassEmpty = 14
	numClasses = ClassEmpty + 1 // 駒種14（向きなし）+ 空
)

// InputSize は外部パッケージから参照できる入力サイズ
const InputSize = inputSize

// NumClasses は推論器の出力クラス数（駒種 + 空）
const NumClasses = numClasses

// EmptyLabel は空マスを表すラベル文字列
const EmptyLabel = "none"

// BaseLabels はクラスIDと駒種（向きなし大文字）の対応
var BaseLabels = []string{
	"P",  // 0: 歩
	"L",  // 1: 香
	"N",  // 2: 桂
	"S",  // 3: 銀
	"G",  // 4: 金
	"B",  // 5: 角
	"R",  // 6: 飛
	"K",  // 7: 玉
	"+P", // 8: と
	"+L", // 9: 杏
	"+N", // 10: 圭
	"+S", // 11: 全
	"+B", // 12: 馬
	"+R", // 13: 龍
}

// LabelToClass はSFENラベルをクラスIDに変換
// 空マス（"none"）は ClassEmpty、それ以外の未知ラベルは -1 を返す
func LabelToClass(label string) int {
	upper := strings.ToUpper(label)
	if upper == strings.ToUpper(EmptyLabel) {
		return ClassEmpty
	}
	for i, b := range BaseLabels {
		if strings.ToUpper(b) == upper {
			return i
		}
	}
	return -1
}

// ClassToLabel はクラスIDをSFENラベルに変換（向きなし大文字）
// 空マスは EmptyLabel を返す
func ClassToLabel(class int) string {
	if class == ClassEmpty {
		return EmptyLabel
	}
	if class < 0 || class >= len(BaseLabels) {
		return "P"
	}
	return BaseLabels[class]
}

// ClassToBaseLabel は ClassToLabel の別名
func ClassToBaseLabel(class int) string {
	return ClassToLabel(class)
}

// TrainingData は学習データ
type TrainingData struct {
	Samples []TrainingSample `json:"samples"`
}

type TrainingSample struct {
	Input []float64 `json:"input"`
	Label int       `json:"label"`
}

// CellToInput はマス画像を入力ベクトルに変換 (24x24 グレースケール → 576 float64)
// 画像ごとの明るさ・コントラスト差を吸収するため、平均0・分散1に標準化する
func CellToInput(cell image.Image) []float64 {
	resized := resizeGray(cell, cellSize, cellSize)
	input := make([]float64, inputSize)
	bounds := resized.Bounds()
	var sum, sumSq float64
	for y := 0; y < cellSize; y++ {
		for x := 0; x < cellSize; x++ {
			v := float64(resized.GrayAt(x+bounds.Min.X, y+bounds.Min.Y).Y)
			input[y*cellSize+x] = v
			sum += v
			sumSq += v * v
		}
	}
	n := float64(inputSize)
	mean := sum / n
	std := math.Sqrt(sumSq/n - mean*mean)
	if std < 1 {
		std = 1
	}
	for i := range input {
		input[i] = (input[i] - mean) / std
	}
	return input
}

// SaveTrainingData は学習データをJSONファイルに保存
func SaveTrainingData(path string, data *TrainingData) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return json.NewEncoder(f).Encode(data)
}

// LoadTrainingData は学習データをJSONファイルから読み込み
func LoadTrainingData(path string) (*TrainingData, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var data TrainingData
	err = json.NewDecoder(f).Decode(&data)
	return &data, err
}

// Model は駒認識モデル
type Model struct {
	NN *gobrain.FeedForward

	source string // 読み込み元のファイルパス（LoadPredictor が設定。デバッグ表示用）
}

// Debug は認識結果に載せる推論器の素性を返す（DebugPredictor）。
func (m *Model) Debug() PredictorDebug {
	return PredictorDebug{Kind: "nn", Source: m.source}
}

// LoadModel はモデルをJSONファイルから読み込み
func LoadModel(path string) (*Model, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var nn gobrain.FeedForward
	if err := json.NewDecoder(f).Decode(&nn); err != nil {
		return nil, err
	}
	return &Model{NN: &nn}, nil
}

// Predict はマス画像から駒を推論する
func (m *Model) Predict(cell image.Image) (int, float64) {
	input := CellToInput(cell)
	output := m.NN.Update(input)

	bestClass := 0
	bestConf := 0.0
	for i, v := range output {
		if v > bestConf {
			bestConf = v
			bestClass = i
		}
	}
	return bestClass, bestConf
}

// OrientMarginMin はこの値未満の確信度（`ClassifyCellDetail`）のとき、
// 向きを回転照合で決め直す。
//
// **分類器の向き判定は上下半分の幅の差で決まるので、差が小さいマスだけが
// 危ない。** 実測（保存済み36局面 / 1202 駒マス）で、この値を境に
// 反転 47 件のうち 23 件が 85 マスに固まっている（その帯での誤り率 27%、
// 全体は 3.9%）。帯を広げると回転照合の弱いところまで任せることになり、
// 逆に悪化する:
//
//	しきい値 0.02 → 96.59% / 0.05 → 97.17% / **0.08 → 97.42%**
//	          0.12 → 97.00% / 0.20 → 95.75%（＝回転照合だけの成績）
//
// 回転照合単体は 95.75% で分類器（96.09%）より悪いので、
// **全面的に置き換えてはいけない。** 弱いところだけ補い合う関係にある。
const OrientMarginMin = 0.08

// ClassifyCellFor は推論器も使ってマスを 空/先手/後手 に分類する。
//
// **空/先手/後手 を出すところは必ずこれを通すこと。** `ClassifyCellWith` は
// 画像処理だけの一次判定で、確信度が足りないマスの向きは推論器との回転照合で
// 決め直される。素の `ClassifyCellWith` を別途呼ぶと `RecognizeBoard` が返す
// SFEN と食い違う。
//
// byMatch は回転照合で決め直したかどうか（観測用。`CellDebug.OrientBy`）。
func ClassifyCellFor(cell image.Image, boardColor uint8, m Predictor) (cat CellCategory, byMatch bool) {
	cat, margin := ClassifyCellDetail(cell, boardColor)
	if cat == CellEmpty || margin >= OrientMarginMin {
		return cat, false
	}
	om, ok := m.(OrientationMatcher)
	if !ok {
		return cat, false
	}
	// 学習データは後手の駒を Rotate180 して先手向きに揃えてあるので、
	// そのままのほうが近ければ先手、180度回した版のほうが近ければ後手
	up := om.PieceDistance(cell)
	down := om.PieceDistance(Rotate180(cell))
	if math.IsInf(up, 1) && math.IsInf(down, 1) {
		return cat, false
	}
	if up <= down {
		return CellPieceUp, true
	}
	return CellPieceDown, true
}

// RecognizeBoard は盤面全体を認識してSFENを返す
//
// 空/向きは ClassifyCellWith が一次判定し、駒種は Predictor（NN または k-NN）で
// 推論する。Predictor が空クラス（ClassEmpty）を返した場合は分類より優先して
// 空とする。グリッド線や木目で被覆率が上がったマスは、被覆率では駒と分離できないが
// 画素の並びは既知の空マスと一致するため、学習済みの空パターンのほうが確かなため。
// 逆方向（分類が空・推論が駒）は向きが決まらないので覆さない。
func RecognizeBoard(img image.Image, br *BoardRegion, m Predictor) string {
	board, _, _ := recognizeBoardDetail(img, br, m)
	return board
}

// recognizeBoardDetail は RecognizeBoard の本体で、盤面文字列に加えて
// マスごとの認識過程（Result.Debug 用）と盤の地色を返す。
func recognizeBoardDetail(img image.Image, br *BoardRegion, m Predictor) (string, []CellDebug, uint8) {
	// 各マスの SFEN 表記("" は空マス)を組み立て、盤面文字列化(空マスの
	// ランレングス圧縮・段区切り)は core/sfen に委譲する。
	var grid [9][9]string
	bc := BoardColor(img, br)
	cells := make([]CellDebug, 0, 81)
	for r := 0; r < 9; r++ {
		for c := 0; c < 9; c++ {
			d := CellDebug{Row: r, Col: c, Rect: br.Cells[r][c], Class: -1}
			// 先に登録して、以降は d を書き換えつつ continue できるようにする
			// （マスごとの記録を取りこぼさないため）。
			cells = append(cells, d)
			cur := &cells[len(cells)-1]

			cell := br.ExtractCell(img, r, c)
			if cell == nil {
				continue
			}

			cat, byMatch := ClassifyCellFor(cell, bc, m)
			cur.Category = cat
			if byMatch {
				cur.OrientBy = "match"
			}
			if cat == CellEmpty {
				continue
			}

			if m != nil {
				// gobrain で駒種を推論（後手は180度回転して先手向きに正規化）
				ncell := cell
				if cat == CellPieceDown {
					ncell = Rotate180(cell)
				}
				class, conf := m.Predict(ncell)
				cur.Class, cur.Confidence = class, conf
				if class == ClassEmpty {
					continue
				}
				base := ClassToBaseLabel(class)
				if cat == CellPieceDown {
					if len(base) > 0 {
						if base[0] == '+' {
							grid[r][c] = "+" + strings.ToLower(base[1:])
						} else {
							grid[r][c] = strings.ToLower(base)
						}
					}
				} else {
					grid[r][c] = base
				}
			} else {
				// モデルなし: 向き不問で駒があることだけを示す
				grid[r][c] = "?"
			}
			cur.Piece = grid[r][c]
		}
	}
	board := sfen.FormatBoard(func(rank, file int) string { return grid[rank][file] })
	return board, cells, bc
}

// resizeGray は画像を指定サイズのグレースケールにリサイズする（面積平均）。
//
// **最近傍法にしてはいけない。** 1マスは 40〜115px あり、これを 24x24 に
// 落とすので最近傍だと 3〜5px おきの点をつまむ形になる。盤面座標が 1px ずれると
// つまむ点が総入れ替えになり、576次元ベクトルが別物になってしまう。
// 実測で、盤面座標を 1px ずらすだけで誤認識マスが 22 → 112（全891マス）に
// 増えていた。面積平均なら 1px のずれは各画素の重みがわずかに動くだけになる。
func resizeGray(src image.Image, w, h int) *image.Gray {
	srcBounds := src.Bounds()
	srcW := srcBounds.Dx()
	srcH := srcBounds.Dy()

	dst := image.NewGray(image.Rect(0, 0, w, h))
	if srcW <= 0 || srcH <= 0 {
		return dst
	}
	for y := 0; y < h; y++ {
		y0 := srcBounds.Min.Y + y*srcH/h
		y1 := srcBounds.Min.Y + (y+1)*srcH/h
		if y1 <= y0 {
			y1 = y0 + 1
		}
		for x := 0; x < w; x++ {
			x0 := srcBounds.Min.X + x*srcW/w
			x1 := srcBounds.Min.X + (x+1)*srcW/w
			if x1 <= x0 {
				x1 = x0 + 1
			}
			sum, n := 0, 0
			for sy := y0; sy < y1; sy++ {
				for sx := x0; sx < x1; sx++ {
					r, g, b, _ := src.At(sx, sy).RGBA()
					sum += int((19595*r + 38470*g + 7471*b + 1<<15) >> 24)
					n++
				}
			}
			if n == 0 {
				continue
			}
			dst.SetGray(x, y, color.Gray{Y: uint8(sum / n)})
		}
	}
	return dst
}
