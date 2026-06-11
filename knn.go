package suteme

import (
	"image"
	"sort"
)

// Predictor は駒種推論器の共通インターフェース（gobrain NN / k-NN）
type Predictor interface {
	Predict(cell image.Image) (int, float64)
}

// KNN は k近傍法による駒種認識器
// 学習処理が不要で、サンプルを追加した瞬間に反映される。
// ゲーム画面のように同じ駒がほぼ同一ピクセルで描画される画像では
// NN より安定して高精度になる
type KNN struct {
	samples []TrainingSample
	k       int
}

// NewKNN はサンプルから k-NN 認識器を作る。サンプルが空なら nil
func NewKNN(samples []TrainingSample) *KNN {
	valid := make([]TrainingSample, 0, len(samples))
	for _, s := range samples {
		if len(s.Input) == inputSize && s.Label >= 0 && s.Label < numClasses {
			valid = append(valid, s)
		}
	}
	if len(valid) == 0 {
		return nil
	}
	k := 5
	if len(valid) < k {
		k = len(valid)
	}
	return &KNN{samples: valid, k: k}
}

// Len は保持しているサンプル数を返す
func (kn *KNN) Len() int {
	return len(kn.samples)
}

// Predict はマス画像から駒種を推論する
// 距離の逆数で重み付けした上位k件の投票で決定し、
// 信頼度は勝者クラスの重み比率を返す
func (kn *KNN) Predict(cell image.Image) (int, float64) {
	input := CellToInput(cell)

	type neighbor struct {
		dist  float64
		label int
	}
	nbrs := make([]neighbor, len(kn.samples))
	for i, s := range kn.samples {
		d := 0.0
		for j, v := range input {
			diff := v - s.Input[j]
			d += diff * diff
		}
		nbrs[i] = neighbor{dist: d, label: s.Label}
	}
	sort.Slice(nbrs, func(i, j int) bool { return nbrs[i].dist < nbrs[j].dist })

	const eps = 1e-6
	votes := make(map[int]float64)
	total := 0.0
	for i := 0; i < kn.k; i++ {
		w := 1.0 / (nbrs[i].dist + eps)
		votes[nbrs[i].label] += w
		total += w
	}

	bestClass := 0
	bestWeight := 0.0
	for label, w := range votes {
		if w > bestWeight {
			bestWeight = w
			bestClass = label
		}
	}
	if total == 0 {
		return bestClass, 0
	}
	return bestClass, bestWeight / total
}
