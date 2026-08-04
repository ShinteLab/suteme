// Package training は駒種認識モデルの学習処理を提供する。
//
// suteme パッケージ本体は「画像データを元に棋譜データを作成する」推論側に徹し、
// モデルを作る側（NN訓練・クラスバランシング・モデル保存）は本パッケージが担う。
// 依存方向は training → suteme の一方向のみ。
package training

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"sort"

	"github.com/goml/gobrain"

	"github.com/ShinteLab/suteme"
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

// MergeSamples は入力ベクトルの内容をキーにサンプルをマージし、重複を取り除く。
// 後から渡したものが勝つので、同じマスにラベルを付け直した場合は訂正が反映される。
// 並びは最初に現れた位置を保つ。
//
// /api/train は毎回「既存ファイルの全件 + メモリ上の全セッションのラベル」を
// 保存し直すため、これを通さないと学習を押すたびに同じサンプルが積み上がる。
// 同じ画像の同じマスからは同じ入力ベクトルが得られるので、内容が重複判定になる。
func MergeSamples(groups ...[]suteme.TrainingSample) []suteme.TrainingSample {
	total := 0
	for _, g := range groups {
		total += len(g)
	}
	buckets := make(map[uint64][]int, total)
	result := make([]suteme.TrainingSample, 0, total)

	for _, g := range groups {
		for _, s := range g {
			key := sampleKey(s.Input)
			found := -1
			// ハッシュ衝突で別サンプルを取り違えないよう中身も突き合わせる
			for _, i := range buckets[key] {
				if equalInput(result[i].Input, s.Input) {
					found = i
					break
				}
			}
			if found >= 0 {
				result[found] = s // 後勝ち
				continue
			}
			buckets[key] = append(buckets[key], len(result))
			result = append(result, s)
		}
	}
	return result
}

// sampleKey は入力ベクトルの内容ハッシュ（FNV-1a 64bit）
func sampleKey(input []float64) uint64 {
	const (
		offset64 = 14695981039346656037
		prime64  = 1099511628211
	)
	h := uint64(offset64)
	var buf [8]byte
	for _, v := range input {
		binary.LittleEndian.PutUint64(buf[:], math.Float64bits(v))
		for _, b := range buf {
			h ^= uint64(b)
			h *= prime64
		}
	}
	return h
}

func equalInput(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
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
