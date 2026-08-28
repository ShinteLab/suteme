package suteme

import (
	"fmt"
	"image"
	"math"
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
	source  string // 読み込み元のファイルパス（LoadPredictor が設定。デバッグ表示用）
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

// OrientationMatcher は「このマス画像が、先手向きに正規化された学習データから
// どれだけ離れているか」を返せる推論器。`recognize.go` が向きの決定に使う。
//
// 学習データは後手の駒を Rotate180 して先手向きに揃えてあるので、
// そのままと 180度回した版のどちらが近いかで向きが決まる。
type OrientationMatcher interface {
	// PieceDistance は駒サンプルへの最近傍距離を返す（小さいほど近い）。
	// 比較にしか使わないので尺度は実装依存でよい。
	PieceDistance(cell image.Image) float64
}

// PieceDistance は駒サンプル（空を除く）への最近傍の2乗距離を返す。
// 一致するサンプルが無ければ +Inf。
//
// **nil でも落ちない。** `NewKNN` はサンプルが無ければ nil を返すので、
// `SetOrientMatcher(NewKNN(...))` のように**型付きの nil が
// インタフェースに入る**ことがある（呼ぶ側では nil 判定にかからない）。
// 距離を +Inf で返せば「照合できない」として扱われ、向きは分類器に落ちる。
func (kn *KNN) PieceDistance(cell image.Image) float64 {
	if kn == nil {
		return math.Inf(1)
	}
	input := CellToInput(cell)
	best := math.Inf(1)
	for _, s := range kn.samples {
		if s.Label == ClassEmpty {
			continue
		}
		d := 0.0
		for j, v := range input {
			diff := v - s.Input[j]
			d += diff * diff
			if d >= best {
				break
			}
		}
		if d < best {
			best = d
		}
	}
	return best
}

// Debug は認識結果に載せる推論器の素性を返す（DebugPredictor）。
func (kn *KNN) Debug() PredictorDebug {
	return PredictorDebug{
		Kind:   "knn",
		Source: kn.source,
		Detail: fmt.Sprintf("k=%d, samples=%d", kn.k, len(kn.samples)),
	}
}

// emptyMatchMax は空マスと判定する最近傍距離の上限（emptyDistNorm で正規化後）。
//
// 空を通常の距離競争に参加させてはいけない。CellToInput は平均0・分散1に
// 標準化するが、平坦な空マスは分散が小さく std のクランプがかかるため
// ほぼ原点のベクトルに写る。原点は空間の中心でどの駒ベクトルからも
// 等距離に近く、未知の盤の駒にとっては「他の盤の同じ駒」より「空」のほうが
// 近くなる。実測で、多数決に混ぜると駒→空の誤りが 1 件から 152 件、
// 最近傍1件の相対比較でも 115 件に増えた。
//
// そこで「既知の空パターンとほぼ一致したときだけ空」という絶対しきい値にする。
// 実測の最近傍距離は、駒マス→空サンプルが最小 0.120（同じ盤）/ 0.161（他の盤）
// なのに対し、空マスは 1割が 0.039 以下。一致した場合だけ拾えば駒を巻き込まない。
// 一致しなければ ClassifyCellWith の判定がそのまま残るだけなので、
// この判定は精度を下げる方向には働かない。
const emptyMatchMax = 0.10

// emptyDistNorm は距離の正規化係数。無相関な標準化ベクトル同士の
// 2乗距離の期待値（2 * 次元数）で割り、次元数に依存しない値にする
const emptyDistNorm = 2 * inputSize

// Predict はマス画像から駒種を推論する
//
// 空マス（ClassEmpty）は既知の空パターンとの一致で判定し、駒種の多数決には
// 混ぜない（理由は emptyMatchMax のコメント）。
// 駒種は空を除いたサンプルで、距離の逆数で重み付けした上位k件の投票で決める。
// 信頼度は勝者クラスの重み比率。
func (kn *KNN) Predict(cell image.Image) (int, float64) {
	input := CellToInput(cell)
	// 空サンプルはマス全体で作ってあるので、こちらも切り揃えない版で測る
	// （CellToInputFull 参照。外接矩形は駒にしか定義できない）
	full := [2][]float64{CellToInputFull(cell), CellToInputFull(Rotate180(cell))}

	type neighbor struct {
		dist  float64
		label int
	}
	nbrs := make([]neighbor, 0, len(kn.samples))
	nearestEmpty := math.Inf(1)
	for _, s := range kn.samples {
		if s.Label == ClassEmpty {
			// 空には向きが無い。呼び出し側が後手と見て 180 度回したマスも
			// 空の可能性があるので、両向きの近いほうで測る
			// （空サンプルは回転させずに 1 通りしか持っていない）
			for _, in := range full {
				d := 0.0
				for j, v := range in {
					diff := v - s.Input[j]
					d += diff * diff
					if d >= nearestEmpty {
						break
					}
				}
				if d < nearestEmpty {
					nearestEmpty = d
				}
			}
			continue
		}
		d := 0.0
		for j, v := range input {
			diff := v - s.Input[j]
			d += diff * diff
		}
		nbrs = append(nbrs, neighbor{dist: d, label: s.Label})
	}
	sort.Slice(nbrs, func(i, j int) bool { return nbrs[i].dist < nbrs[j].dist })

	nearestPiece := math.Inf(1)
	if len(nbrs) > 0 {
		nearestPiece = nbrs[0].dist
	}
	if nearestEmpty/emptyDistNorm < emptyMatchMax && nearestEmpty < nearestPiece {
		return ClassEmpty, confFromDist(nearestPiece, nearestEmpty)
	}
	if len(nbrs) == 0 {
		return 0, 0
	}

	const eps = 1e-6
	k := kn.k
	if k > len(nbrs) {
		k = len(nbrs)
	}
	votes := make(map[int]float64)
	total := 0.0
	for i := 0; i < k; i++ {
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

// confFromDist は勝った側の距離と負けた側の距離から信頼度を作る。
// 差が開くほど 1 に近づき、拮抗していれば 0.5 に近づく
func confFromDist(loser, winner float64) float64 {
	if math.IsInf(loser, 1) {
		return 1
	}
	if sum := loser + winner; sum > 0 {
		return loser / sum
	}
	return 0.5
}
