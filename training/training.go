// Package training は駒種認識モデルの学習処理を提供する。
//
// suteme パッケージ本体は「画像データを元に棋譜データを作成する」推論側に徹し、
// モデルを作る側（NN訓練・クラスバランシング・モデル保存）は本パッケージが担う。
// 依存方向は training → suteme の一方向のみ。
package training

import (
	"encoding/json"
	"math/rand"
	"os"
	"sort"

	"github.com/goml/gobrain"

	"suteme"
)

// Train は学習データから gobrain FeedForward モデルを訓練する
func Train(data *suteme.TrainingData) *suteme.Model {
	if len(data.Samples) == 0 {
		return nil
	}

	numClasses := len(suteme.BaseLabels)

	ff := &gobrain.FeedForward{}
	ff.Init(suteme.InputSize, suteme.InputSize/4, numClasses)

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

	return &suteme.Model{NN: ff}
}

// BalanceData はクラス偏りを軽減する
// 中央値の3倍を上限に多数クラスをダウンサンプルし、
// 中央値未満の少数クラスは複製で中央値までオーバーサンプルする。
// （最少クラス基準だと、1枚しかない駒種があるだけで全データが捨てられてしまう）
func BalanceData(samples []suteme.TrainingSample) []suteme.TrainingSample {
	byClass := make(map[int][]suteme.TrainingSample)
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

	result := make([]suteme.TrainingSample, 0, len(samples))
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
func ClassDistribution(samples []suteme.TrainingSample) map[string]int {
	dist := map[string]int{}
	for _, s := range samples {
		dist[suteme.ClassToLabel(s.Label)]++
	}
	return dist
}

// SaveModel は学習済みモデルをJSONファイルに保存する
// （読み込みは推論側の suteme.LoadModel が担当）
func SaveModel(path string, m *suteme.Model) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return json.NewEncoder(f).Encode(m.NN)
}
