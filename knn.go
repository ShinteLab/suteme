package suteme

import (
	"fmt"
	"image"
	"math"
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
	// **空サンプルは別に持つ。** 駒種の投票にも向きの照合にも使わず、
	// 「既知の空パターンと一致するか」だけに使う別物なので、同じ配列に
	// 混ぜておくと走査のたびにラベルを見て弾くことになる。
	// しかも**学習データの 2/3 が空**（実測 65226 件中 44220 件）なので、
	// 分けておかないと駒サンプルの走査が 3 倍の距離を歩く。
	pieces  []TrainingSample // 駒サンプル（空を除く）
	empties []TrainingSample // 空サンプル（ClassEmpty）
	k       int
	source  string // 読み込み元のファイルパス（LoadPredictor が設定。デバッグ表示用）
}

// NewKNN はサンプルから k-NN 認識器を作る。サンプルが空なら nil
func NewKNN(samples []TrainingSample) *KNN {
	kn := &KNN{k: 5}
	for _, s := range samples {
		if len(s.Input) != inputSize || s.Label < 0 || s.Label >= numClasses {
			continue
		}
		if s.Label == ClassEmpty {
			kn.empties = append(kn.empties, s)
		} else {
			kn.pieces = append(kn.pieces, s)
		}
	}
	n := len(kn.pieces) + len(kn.empties)
	if n == 0 {
		return nil
	}
	if n < kn.k {
		kn.k = n
	}
	return kn
}

// Len は保持しているサンプル数を返す
func (kn *KNN) Len() int {
	return len(kn.pieces) + len(kn.empties)
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
	for _, s := range kn.pieces {
		if d, ok := dist2(input, s.Input, best); ok {
			best = d
		}
	}
	return best
}

// ConcurrentPredict は「複数のゴルーチンから同時に Predict を呼んでよい」の表明
// （`ConcurrentPredictor`）。k-NN は学習データを読むだけで何も書き換えない。
func (kn *KNN) ConcurrentPredict() {}

// Debug は認識結果に載せる推論器の素性を返す（DebugPredictor）。
func (kn *KNN) Debug() PredictorDebug {
	return PredictorDebug{
		Kind:   "knn",
		Source: kn.source,
		Detail: fmt.Sprintf("k=%d, samples=%d", kn.k, kn.Len()),
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

// emptyDistCap は空サンプルへの距離を測るときの打ち切り上限。
//
// **最近傍距離そのものは要らない。要るのは「emptyMatchMax より近いか」だけ。**
// 戻り値が使われるのは `nearestEmpty/emptyDistNorm < emptyMatchMax` が
// 成り立つときだけなので、最初からこの値を上限にしてよい（超えた時点で
// 結果に影響しないことが確定する）。学習データの 2/3 は空サンプルで、
// **駒マスではそのどれとも一致しない**＝上限が下がらないまま全件を
// 最後まで足していた。上限を置くと数次元で抜けられる。
const emptyDistCap = emptyMatchMax * emptyDistNorm

// dist2 は 2乗距離を返す。limit 以上になった時点で打ち切り、false を返す。
//
// **足す順序は打ち切りの有無に依らない**（先頭から順に足す）ので、
// true で返った距離は最後まで足した値とビット単位で一致する。
// 打ち切りが変えるのは「捨てる相手にどれだけ時間をかけるか」だけで、
// 残った相手の距離も順位も動かない。
//
// 判定を 1 次元ごとに入れないのは、比較のほうが加算より高くつくため
// （`knnDistCheck` 次元ごとに見る）。
func dist2(a, b []float64, limit float64) (float64, bool) {
	a, b = a[:inputSize], b[:inputSize]
	d := 0.0
	for j := 0; j < inputSize; j++ {
		diff := a[j] - b[j]
		d += diff * diff
		if j&(knnDistCheck-1) == knnDistCheck-1 && d >= limit {
			return 0, false
		}
	}
	return d, d < limit
}

// knnDistCheck は打ち切りを確かめる間隔（次元数）。2 の冪にすること
const knnDistCheck = 32

// knnMaxK は topK が保てる件数の上限（k の既定値 5 に合わせてある）
const knnMaxK = 5

// topK は上位 k 件を距離の昇順で保つ。
//
// **全件を並べてはいけない。** 以前は 6 万件ぶんの距離を配列に積んで
// `sort.Slice` で並べ、先頭 5 件だけ使っていた。上位 k 件だけなら挿入で足りる。
// **速くなる本体はソートを省くことではなく、k 番目の距離を `dist2` の
// 打ち切りに渡せること**（負ける相手の距離を最後まで足さなくなる）。
type topK struct {
	dist  [knnMaxK]float64
	label [knnMaxK]int
	n     int
	k     int
}

func (t *topK) init(k int) {
	if k > knnMaxK {
		k = knnMaxK
	}
	t.k, t.n = k, 0
}

// limit は「これ以上なら採らない」距離。k 件たまるまでは上限なし
func (t *topK) limit() float64 {
	if t.k == 0 || t.n < t.k {
		return math.Inf(1)
	}
	return t.dist[t.k-1]
}

func (t *topK) push(d float64, label int) {
	i := t.n
	if i > t.k-1 {
		i = t.k - 1
	}
	for i > 0 && t.dist[i-1] > d {
		t.dist[i], t.label[i] = t.dist[i-1], t.label[i-1]
		i--
	}
	t.dist[i], t.label[i] = d, label
	if t.n < t.k {
		t.n++
	}
}

// vote は距離の逆数で重み付けした多数決。信頼度は勝者の重み比率。
//
// クラスごとの重みは**クラス数ぶんの配列**に積む（15 個しかない）。
// 以前の map は、同じ重みで並んだときに**どちらが勝つかが反復順まかせ**で、
// 同じ入力に同じ答えを返す保証が無かった。
func (t *topK) vote() (int, float64) {
	if t.n == 0 {
		return 0, 0
	}
	const eps = 1e-6
	var votes [numClasses]float64
	total := 0.0
	for i := 0; i < t.n; i++ {
		w := 1.0 / (t.dist[i] + eps)
		votes[t.label[i]] += w
		total += w
	}
	bestClass, bestWeight := 0, 0.0
	for label, w := range votes {
		if w > bestWeight {
			bestWeight, bestClass = w, label
		}
	}
	if total == 0 {
		return bestClass, 0
	}
	return bestClass, bestWeight / total
}

// nearestEmpty は空サンプルへの最近傍距離を返す（`emptyDistCap` で頭打ち）。
//
// 空には向きが無いので、そのままと 180 度回した版の近いほうで測る
// （空サンプルは回転させずに 1 通りしか持っていない）。
func (kn *KNN) nearestEmpty(full [2][]float64) float64 {
	best := emptyDistCap
	for _, s := range kn.empties {
		for _, in := range full {
			if d, ok := dist2(in, s.Input, best); ok {
				best = d
			}
		}
	}
	return best
}

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
	nearestEmpty := kn.nearestEmpty(full)

	var top topK
	k := kn.k
	if k > len(kn.pieces) {
		k = len(kn.pieces)
	}
	top.init(k)
	for _, s := range kn.pieces {
		if d, ok := dist2(input, s.Input, top.limit()); ok {
			top.push(d, s.Label)
		}
	}

	nearestPiece := math.Inf(1)
	if top.n > 0 {
		nearestPiece = top.dist[0]
	}
	if nearestEmpty/emptyDistNorm < emptyMatchMax && nearestEmpty < nearestPiece {
		return ClassEmpty, confFromDist(nearestPiece, nearestEmpty)
	}
	return top.vote()
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
