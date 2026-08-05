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

// RecognizeBoard は盤面全体を認識してSFENを返す
//
// 空/向きは ClassifyCellWith が一次判定し、駒種は Predictor（NN または k-NN）で
// 推論する。Predictor が空クラス（ClassEmpty）を返した場合は分類より優先して
// 空とする。グリッド線や木目で被覆率が上がったマスは、被覆率では駒と分離できないが
// 画素の並びは既知の空マスと一致するため、学習済みの空パターンのほうが確かなため。
// 逆方向（分類が空・推論が駒）は向きが決まらないので覆さない。
func RecognizeBoard(img image.Image, br *BoardRegion, m Predictor) string {
	// 各マスの SFEN 表記("" は空マス)を組み立て、盤面文字列化(空マスの
	// ランレングス圧縮・段区切り)は core/sfen に委譲する。
	var grid [9][9]string
	bc := BoardColor(img, br)
	for r := 0; r < 9; r++ {
		for c := 0; c < 9; c++ {
			cell := br.ExtractCell(img, r, c)
			if cell == nil {
				continue
			}

			cat := ClassifyCellWith(cell, bc)
			if cat == CellEmpty {
				continue
			}

			if m != nil {
				// gobrain で駒種を推論（後手は180度回転して先手向きに正規化）
				ncell := cell
				if cat == CellPieceDown {
					ncell = Rotate180(cell)
				}
				class, _ := m.Predict(ncell)
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
		}
	}
	return sfen.FormatBoard(func(rank, file int) string { return grid[rank][file] })
}

// resizeGray は画像を指定サイズのグレースケールにリサイズする（最近傍法）
func resizeGray(src image.Image, w, h int) *image.Gray {
	srcBounds := src.Bounds()
	srcW := srcBounds.Dx()
	srcH := srcBounds.Dy()

	dst := image.NewGray(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			srcX := srcBounds.Min.X + int(math.Round(float64(x)*float64(srcW)/float64(w)))
			srcY := srcBounds.Min.Y + int(math.Round(float64(y)*float64(srcH)/float64(h)))
			if srcX >= srcBounds.Max.X {
				srcX = srcBounds.Max.X - 1
			}
			if srcY >= srcBounds.Max.Y {
				srcY = srcBounds.Max.Y - 1
			}
			r, g, b, _ := src.At(srcX, srcY).RGBA()
			gray := uint8((19595*r + 38470*g + 7471*b + 1<<15) >> 24)
			dst.SetGray(x, y, color.Gray{Y: gray})
		}
	}
	return dst
}
