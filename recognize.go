package suteme

import (
	"encoding/json"
	"image"
	"image/color"
	"math"
	"os"
	"strings"

	"shinte/core/sfen"
	"github.com/goml/gobrain"
)

const (
	cellSize   = 24
	inputSize  = cellSize * cellSize
	numClasses = 14 // 駒種のみ（向きなし）
)

// InputSize は外部パッケージから参照できる入力サイズ
const InputSize = inputSize

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

// LabelToClass はSFENラベルを駒種クラスIDに変換
// 駒として認識できないラベル（"none" 等）は -1 を返す
func LabelToClass(label string) int {
	upper := strings.ToUpper(label)
	for i, b := range BaseLabels {
		if strings.ToUpper(b) == upper {
			return i
		}
	}
	return -1
}

// ClassToLabel はクラスIDをSFENラベルに変換（向きなし大文字）
func ClassToLabel(class int) string {
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
// 空/向きは ClassifyCell で判定し、駒種のみ Predictor（NN または k-NN）で推論する
func RecognizeBoard(img image.Image, br *BoardRegion, m Predictor) string {
	// 各マスの SFEN 表記("" は空マス)を組み立て、盤面文字列化(空マスの
	// ランレングス圧縮・段区切り)は core/sfen に委譲する。
	var grid [9][9]string
	for r := 0; r < 9; r++ {
		for c := 0; c < 9; c++ {
			cell := br.ExtractCell(img, r, c)
			if cell == nil {
				continue
			}

			cat := ClassifyCell(cell)
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
