package suteme

import (
	"encoding/json"
	"image"
	"image/color"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"

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

// SaveModel はモデルをJSONファイルに保存
func SaveModel(path string, m *Model) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return json.NewEncoder(f).Encode(m.NN)
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

// BalanceData はクラス偏りを軽減する
// 中央値の3倍を上限に多数クラスをダウンサンプルし、
// 中央値未満の少数クラスは複製で中央値までオーバーサンプルする。
// （最少クラス基準だと、1枚しかない駒種があるだけで全データが捨てられてしまう）
func BalanceData(samples []TrainingSample) []TrainingSample {
	byClass := make(map[int][]TrainingSample)
	for _, s := range samples {
		byClass[s.Label] = append(byClass[s.Label], s)
	}
	if len(byClass) == 0 {
		return samples
	}

	counts := make([]int, 0, len(byClass))
	for _, ss := range byClass {
		counts = append(counts, len(ss))
	}
	sort.Ints(counts)
	median := counts[len(counts)/2]

	maxCount := median * 3
	if maxCount < 20 {
		maxCount = 20
	}

	result := make([]TrainingSample, 0, len(samples))
	for _, ss := range byClass {
		switch {
		case len(ss) > maxCount:
			perm := rand.Perm(len(ss))
			for i := 0; i < maxCount; i++ {
				result = append(result, ss[perm[i]])
			}
		case len(ss) < median:
			result = append(result, ss...)
			for i := len(ss); i < median; i++ {
				result = append(result, ss[rand.Intn(len(ss))])
			}
		default:
			result = append(result, ss...)
		}
	}
	return result
}

// ClassDistribution はクラスごとのサンプル数を返す
func ClassDistribution(samples []TrainingSample) map[string]int {
	dist := map[string]int{}
	for _, s := range samples {
		dist[ClassToLabel(s.Label)]++
	}
	return dist
}

// Train は学習データからモデルを訓練する
func Train(data *TrainingData) *Model {
	if len(data.Samples) == 0 {
		return nil
	}

	ff := &gobrain.FeedForward{}
	ff.Init(inputSize, inputSize/4, numClasses)

	// クラス順に並んだまま渡すと逐次学習が最後のクラスに引きずられるためシャッフルする
	perm := rand.Perm(len(data.Samples))
	patterns := make([][][]float64, len(data.Samples))
	for i, pi := range perm {
		s := data.Samples[pi]
		output := make([]float64, numClasses)
		if s.Label >= 0 && s.Label < numClasses {
			output[s.Label] = 1.0
		}
		patterns[i] = [][]float64{s.Input, output}
	}

	ff.Train(patterns, 300, 0.2, 0.5, false)

	return &Model{NN: ff}
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
	sfen := ""
	for r := 0; r < 9; r++ {
		empty := 0
		for c := 0; c < 9; c++ {
			cell := br.ExtractCell(img, r, c)
			if cell == nil {
				empty++
				continue
			}

			cat := ClassifyCell(cell)
			if cat == CellEmpty {
				empty++
				continue
			}

			if empty > 0 {
				sfen += string(rune('0' + empty))
				empty = 0
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
							sfen += "+" + strings.ToLower(base[1:])
						} else {
							sfen += strings.ToLower(base)
						}
					}
				} else {
					sfen += base
				}
			} else {
				// モデルなし: 向きだけ
				if cat == CellPieceDown {
					sfen += "?"
				} else {
					sfen += "?"
				}
			}
		}
		if empty > 0 {
			sfen += string(rune('0' + empty))
		}
		if r < 8 {
			sfen += "/"
		}
	}
	return sfen
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
