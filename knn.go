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

// KNN は近傍法による駒種認識器
// 学習処理が不要で、サンプルを追加した瞬間に反映される。
// ゲーム画面のように同じ駒がほぼ同一ピクセルで描画される画像では
// NN より安定して高精度になる。
//
// **駒種は最も近い 1 件で決める（k=1）。上位 5 件の多数決に戻さないこと。**
// 多数決では、近い正解 1 件が「少し遠いが数の多い別の駒種」に負ける。
// 学習データは歩が最も多いので、小さく薄い駒（墨の字しか拾えない実盤）が
// 歩に寄っていた。実測（233 局面、手動座標）:
//
//	                       全マス   駒種    駒種違い
//	局面 LOO   多数決(k=5)  97.4%   93.8%    449
//	           最近傍(k=1)  97.8%   94.9%    365
//	見た目 LOO 多数決(k=5)  91.4%   79.6%    519
//	           最近傍(k=1)  92.3%   81.7%    459
//
// 日付順（その日より前だけで学習して読む）でも 4 条件すべてで最近傍が上回る。
// k=3 / k=9 はどちらも k=5 と同じか悪い。票を件数で割る補正は歩への誤読を
// 減らすが、別の駒種違いが同じだけ増える（AGENTS.md「駒種認識 k-NN」の節）。
type KNN struct {
	// **空サンプルは別に持つ。** 駒種の比較にも向きの照合にも使わず、
	// 「既知の空パターンと一致するか」だけに使う別物なので、同じ配列に
	// 混ぜておくと走査のたびにラベルを見て弾くことになる。
	// しかも**学習データの 2/3 が空**（実測 65226 件中 44220 件）なので、
	// 分けておかないと駒サンプルの走査が 3 倍の距離を歩く。
	pieces  []TrainingSample // 駒サンプル（空を除く）
	empties []TrainingSample // 空サンプル（ClassEmpty）
	source  string           // 読み込み元のファイルパス（LoadPredictor が設定。デバッグ表示用）
}

// NewKNN はサンプルから k-NN 認識器を作る。サンプルが空なら nil
func NewKNN(samples []TrainingSample) *KNN {
	kn := &KNN{}
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
	if len(kn.pieces)+len(kn.empties) == 0 {
		return nil
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
		Detail: fmt.Sprintf("k=1, samples=%d", kn.Len()),
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

// nearestTwo は「最も近い駒種」と「それ以外で最も近い駒種」を保つ。
//
// 2 位の距離は確信度にだけ使う（`pieceConf`）。**2 位の距離を打ち切りに渡せる**
// ので、全件を最後まで足す必要は無い（1 位も 2 位も動かせない相手は、
// 2 位の距離を超えた時点で捨ててよい）。
type nearestTwo struct {
	d1, d2 float64 // 1 位の距離・1 位と違う駒種で最も近い距離
	l1     int
}

func newNearestTwo() nearestTwo {
	return nearestTwo{d1: math.Inf(1), d2: math.Inf(1), l1: -1}
}

func (t *nearestTwo) push(d float64, label int) {
	switch {
	case d < t.d1:
		if label != t.l1 {
			t.d2 = t.d1
		}
		t.d1, t.l1 = d, label
	case d < t.d2 && label != t.l1:
		t.d2 = d
	}
}

// pieceConfSpread は確信度の伸ばし方（`pieceConf`）。
const pieceConfSpread = 0.12

// pieceConf は 1 位と 2 位の駒種の距離から確信度を作る。拮抗していれば 0.5、
// 差が開くほど 1 に近づく（`confFromDist` と同じ意味）。
//
// **ikkyoku はこの値をマスごとの費用にそのまま使う**（修復の予算・下限 0.15）ので、
// 多数決の頃の重み比率と同じ尺度に寄せてある。`confFromDist` の素の値は
// 0.5 付近に固まる（2 乗距離どうしの比なので差が出にくい）ため、
// 0.5 からの差を `pieceConfSpread` で割って伸ばし、1 で頭打ちにする。
// 実測（見た目 LOO の駒マス）で、正しいマスの平均 0.94・誤ったマス 0.68・
// 0.6 未満 10.8%（多数決の頃は 0.93・0.68・11.7%）。
// **誤りの見分けは多数決の頃より良い**（AUC 0.834 → 0.886。局面 LOO では 0.887 → 0.911）。
func pieceConf(t nearestTwo) float64 {
	c := 0.5 + (confFromDist(t.d2, t.d1)-0.5)/pieceConfSpread
	if c > 1 {
		return 1
	}
	return c
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
// 空マス（ClassEmpty）は既知の空パターンとの一致で判定し、駒種の比較には
// 混ぜない（理由は emptyMatchMax のコメント）。
// 駒種は空を除いたサンプルの最近傍 1 件で決める（`KNN` のコメント）。
// 信頼度は 1 位と 2 位の駒種の距離の差（`pieceConf`）。
func (kn *KNN) Predict(cell image.Image) (int, float64) {
	input := CellToInput(cell)
	// 空サンプルはマス全体で作ってあるので、こちらも切り揃えない版で測る
	// （CellToInputFull 参照。外接矩形は駒にしか定義できない）
	full := [2][]float64{CellToInputFull(cell), CellToInputFull(Rotate180(cell))}
	nearestEmpty := kn.nearestEmpty(full)

	top := newNearestTwo()
	for _, s := range kn.pieces {
		if d, ok := dist2(input, s.Input, top.d2); ok {
			top.push(d, s.Label)
		}
	}

	if nearestEmpty/emptyDistNorm < emptyMatchMax && nearestEmpty < top.d1 {
		return ClassEmpty, confFromDist(top.d1, nearestEmpty)
	}
	if top.l1 < 0 {
		return 0, 0
	}
	return top.l1, pieceConf(top)
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
