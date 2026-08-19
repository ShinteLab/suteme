package suteme

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"sync"
)

// 盤の縁の帯を見て「1マス滑っているか」を決める（`StripJudge`）。
//
// **1マスの滑りは、正しい窓との違いが「9マスの帯 1 本」しかない。**
// 窓がちょうど 1マス動くので、各マスの中身は 1 つ隣のマスに入れ替わるだけで
// **駒はマスの中央に写ったまま**。残り 72 マスは完全に同じものを見ている。
// だから盤全体で測る指標（`gridAlignment`・駒数の違反・確信度の平均）は
// **信号を 9:81 に薄める**ので原理的に鈍い。実測でも
// 違反数 10/40・確信度 25/40・最近傍距離 9/40 でどれも使えなかった
// （CLAUDE.md「今後の課題」）。
//
// **その帯を手作りの指標で測る道も閉じている。** 縁の帯と 1マス外の帯を
// 「盤の地色に近いマスの数」で比べると 560 対中 505 で正しく順位が付くのに、
// 検出結果に当てはめると 0.5マス以内が 136 → 128 と悪化する。
// 乗り換えるのは正しかった窓ばかりで、**本当に滑っている局面は
// 4 方向とも同点で動かない**（盤の外が盤と同じ色だから滑る、という因果）。
//
// **そこで帯が盤か否かを学習で決める。** 判断そのものは変わらないが、
// 色・線・格子のどれか 1 つではなく**帯の見え方をそのまま**照合するので、
// 駒台・対局時計・筋の数字・畳といった文脈がまとめて効く。
// leave-one-out の実測（140 局面 × 4 方向 = 560 対）:
//
//	                    盤の勝ち  同点  負け
//	色（地色±40）          505     47     8
//	学習した帯判定         517      8     3
//
// 検出に当てはめた実測（leave-one-out、140 局面）:
//
//	margin  0.5マス以内   直った  壊れた
//	0.0     136 → 136      2       2
//	0.2     136 → 138      2       0
//	0.3     136 → 138      2       0
//
// 直るのは `11f95abb`（0.85 → 0.14マス）と `e0302032`（0.72 → 0.47マス）。
// **どちらも手作りの指標が全部失敗していた局面**（前者は「盤の上の UI の罫線が
// 格子線と同じだけ強い」大盤、後者は `snapToOuterFrame` が正しい窓を動かす例）。

// 帯の入力表現。9マスが横に並んだ細長い画像として 72x8 に落とす。
// **列の帯は転置して行の帯と同じ向きに揃える**（縦と横で別扱いにすると
// サンプルが半分になるうえ、盤の縁という同じものを二度学ぶことになる）。
const (
	stripLen  = 72
	stripThk  = 8
	stripSize = stripLen * stripThk
)

// StripInput は帯（9マス分の細長い矩形）を判定器の入力ベクトルにする。
// vertical は「列の帯」＝縦長のとき true。転置して向きを揃える。
//
// 駒の入力（`CellToInput`）と同じく平均0・分散1に標準化する。
// 明るさや配色は盤ごとに違うが、**盤の縁という構造は共通**なので
// そこだけを残したい。float32 へ丸めるのも同じ理由（ファイル表現と揃える）。
func StripInput(img image.Image, r image.Rectangle, vertical bool) []float64 {
	if img == nil {
		return nil
	}
	r = r.Intersect(img.Bounds())
	if r.Dx() < 4 || r.Dy() < 4 {
		return nil
	}
	sub := image.NewRGBA(image.Rect(0, 0, r.Dx(), r.Dy()))
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			sub.Set(x-r.Min.X, y-r.Min.Y, img.At(x, y))
		}
	}
	g := ConvertGray(sub)
	if vertical {
		g = transposeGray(g)
	}
	res := resizeGray(g, stripLen, stripThk)

	in := make([]float64, stripSize)
	var sum, sumSq float64
	for y := 0; y < stripThk; y++ {
		for x := 0; x < stripLen; x++ {
			v := float64(res.GrayAt(x, y).Y)
			in[y*stripLen+x] = v
			sum += v
			sumSq += v * v
		}
	}
	n := float64(stripSize)
	mean := sum / n
	std := math.Sqrt(sumSq/n - mean*mean)
	if std < 1 {
		std = 1
	}
	for i := range in {
		in[i] = quantize((in[i] - mean) / std)
	}
	return in
}

// StripSample は帯 1 本の教師データ。Board が true なら盤の縁の帯。
type StripSample struct {
	Input []float64
	Board bool
}

// StripJudge は「その帯が盤の中か外か」を返す k-NN。
//
// **駒種の推論器（`KNN`）とは別に持つ。** 目的も入力の意味も違ううえ、
// 駒のサンプルに混ぜると `MergeSamples` の重複除去が意味を失う。
type StripJudge struct {
	samples []StripSample
	path    string
}

// stripK は投票に使う近傍の数。駒種の k-NN と揃えてある
const stripK = 5

// NewStripJudge はサンプルから判定器を作る。空なら nil。
func NewStripJudge(samples []StripSample) *StripJudge {
	if len(samples) == 0 {
		return nil
	}
	return &StripJudge{samples: samples}
}

// Samples は保持しているサンプル数を返す
func (j *StripJudge) Samples() int {
	if j == nil {
		return 0
	}
	return len(j.samples)
}

// Path は読み込み元のファイル（判定器の素性を画面に出すため）
func (j *StripJudge) Path() string {
	if j == nil {
		return ""
	}
	return j.path
}

// BoardScore は帯が盤の縁である度合いを 0.0-1.0 で返す。
// 距離の逆数で重み付けした k 近傍の投票。
func (j *StripJudge) BoardScore(in []float64) float64 {
	if j == nil || len(in) != stripSize {
		return 0
	}
	type nb struct {
		d     float64
		board bool
	}
	best := make([]nb, 0, stripK+1)
	for _, s := range j.samples {
		if len(s.Input) != stripSize {
			continue
		}
		cut := math.Inf(1)
		if len(best) == stripK {
			cut = best[stripK-1].d
		}
		d := 0.0
		for i, v := range in {
			t := v - s.Input[i]
			d += t * t
			if d >= cut {
				break
			}
		}
		if d >= cut {
			continue
		}
		best = append(best, nb{d, s.Board})
		sort.Slice(best, func(a, b int) bool { return best[a].d < best[b].d })
		if len(best) > stripK {
			best = best[:stripK]
		}
	}
	var wb, wt float64
	for _, n := range best {
		w := 1 / (n.d + 1e-6)
		wt += w
		if n.board {
			wb += w
		}
	}
	if wt == 0 {
		return 0
	}
	return wb / wt
}

// unslipMargin は 1マス乗り換えに要求する「盤らしさ」の勝ち幅。
//
// **0 にしてはいけない。** 実測（leave-one-out、140 局面）で
// 0.0 は直った 2 件・壊れた 2 件で差し引きゼロ、0.2 で壊れる件数が 0 になる。
// 0.3 でも結果は同じなので、間を採って 0.2。
const unslipMargin = 0.2

// unslipByJudge は ±1マスずらした窓と見比べ、
// 「失う帯」より「得る帯」のほうが盤らしければ乗り換える。
//
// **絶対スコア（四辺の合計など）にしてはいけない。** 窓ごとに測る帯が
// 変わるので候補間の比較にならない。実測で 0.5マス以内が 136 → 113 まで落ちる。
// 比べるのは「失う帯」と「得る帯」の 2 本だけにすること。
func unslipByJudge(img image.Image, br *BoardRegion, j *StripJudge) *BoardRegion {
	if br == nil || j == nil {
		return br
	}
	b := br.Bounds
	cw, ch := b.Dx()/9, b.Dy()/9
	if cw < 8 || ch < 8 {
		return br
	}

	col := func(cx int) image.Rectangle { return image.Rect(cx, b.Min.Y, cx+cw, b.Max.Y) }
	row := func(cy int) image.Rectangle { return image.Rect(b.Min.X, cy, b.Max.X, cy+ch) }
	cands := []struct {
		dx, dy     int
		lost, gain image.Rectangle
		vertical   bool
	}{
		{1, 0, col(b.Min.X), col(b.Max.X), true},             // 右へ滑る
		{-1, 0, col(b.Max.X - cw), col(b.Min.X - cw), true},  // 左へ
		{0, 1, row(b.Min.Y), row(b.Max.Y), false},            // 下へ
		{0, -1, row(b.Max.Y - ch), row(b.Min.Y - ch), false}, // 上へ
	}

	bestGain, best := unslipMargin, br
	for _, c := range cands {
		lost := StripInput(img, c.lost, c.vertical)
		gain := StripInput(img, c.gain, c.vertical)
		if lost == nil || gain == nil {
			continue
		}
		if d := j.BoardScore(gain) - j.BoardScore(lost); d > bestGain {
			bestGain = d
			best = BoardRegionFromRect(b.Min.X+c.dx*cw, b.Min.Y+c.dy*ch,
				b.Max.X+c.dx*cw, b.Max.Y+c.dy*ch)
		}
	}
	return best
}

// DefaultStripFile は帯の教師データの既定のファイル名。
// **駒種の学習データとは別ファイル**（入力の意味が違う）。
const DefaultStripFile = "strip_data_v1.bin"

var (
	stripJudgeOnce sync.Once
	stripJudgeVal  *StripJudge
	stripJudgeMu   sync.RWMutex
	// stripJudgeSet は SetStripJudge が呼ばれたか。
	// 呼ばれるまでの間だけファイルを自動で探す
	stripJudgeSet bool
)

// SetStripJudge は使う帯判定器を明示指定する。
//
// **nil は「判定器を使わない」。自動探索に戻すのは `ResetStripJudge`。**
// `SetPredictor(nil)` が自動探索に戻るのとは逆なので注意すること。
// こちらは leave-one-out の評価とテストで**判定器を外して測る**用途が主で、
// nil を自動探索にすると「外したつもりがファイルを拾っていた」が起きる。
func SetStripJudge(j *StripJudge) {
	stripJudgeMu.Lock()
	stripJudgeVal, stripJudgeSet = j, true
	stripJudgeMu.Unlock()
}

// ResetStripJudge はファイルからの自動探索に戻す。
func ResetStripJudge() {
	stripJudgeMu.Lock()
	stripJudgeVal, stripJudgeSet = nil, false
	stripJudgeMu.Unlock()
}

// defaultStripJudge はカレント→実行ファイルのディレクトリの順に探す。
// **見つからなければ nil**（判定器が無ければ滑りの補正をしないだけで、
// 検出は従来どおり動く）。
func defaultStripJudge() *StripJudge {
	stripJudgeMu.RLock()
	if stripJudgeSet {
		j := stripJudgeVal
		stripJudgeMu.RUnlock()
		return j
	}
	stripJudgeMu.RUnlock()

	stripJudgeOnce.Do(func() {
		dirs := []string{"."}
		if exe, err := os.Executable(); err == nil {
			dirs = append(dirs, filepath.Dir(exe))
		}
		for _, d := range dirs {
			p := filepath.Join(d, DefaultStripFile)
			s, err := LoadStripData(p)
			if err != nil {
				continue
			}
			if j := NewStripJudge(s); j != nil {
				j.path = p
				stripJudgeMu.Lock()
				stripJudgeVal = j
				stripJudgeMu.Unlock()
				return
			}
		}
	})
	stripJudgeMu.RLock()
	defer stripJudgeMu.RUnlock()
	return stripJudgeVal
}

// 帯データのファイル形式。学習データ（"SUTEMETD"）と同じ並びだが
// **magic を分けてある**。stripSize と inputSize がどちらも 576 なので、
// magic まで同じにすると取り違えても読めてしまう。
//
//	ヘッダ 16 バイト
//	  magic     [8]byte  "SUTEMESJ"
//	  version   uint16   = 1
//	  inputSize uint16   = 576
//	  count     uint32
//	以降 count 件: board int32(0/1) + input [576]float32（リトルエンディアン）
const (
	stripDataVersion = 1
	stripDataHeader  = 16
)

var stripDataMagic = [8]byte{'S', 'U', 'T', 'E', 'M', 'E', 'S', 'J'}

// SaveStripData は帯の教師データを保存する（一時ファイル + rename）。
func SaveStripData(path string, samples []StripSample) error {
	tmp, err := os.CreateTemp(filepath.Dir(path), filepath.Base(path)+".tmp*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)

	w := bufio.NewWriterSize(tmp, 1<<20)
	var head [stripDataHeader]byte
	copy(head[:8], stripDataMagic[:])
	binary.LittleEndian.PutUint16(head[8:], stripDataVersion)
	binary.LittleEndian.PutUint16(head[10:], uint16(stripSize))
	binary.LittleEndian.PutUint32(head[12:], uint32(len(samples)))
	if _, err := w.Write(head[:]); err != nil {
		tmp.Close()
		return err
	}
	buf := make([]byte, 4+stripSize*4)
	for _, s := range samples {
		if len(s.Input) != stripSize {
			tmp.Close()
			return fmt.Errorf("帯の入力長が %d ではありません: %d", stripSize, len(s.Input))
		}
		var b uint32
		if s.Board {
			b = 1
		}
		binary.LittleEndian.PutUint32(buf, b)
		for i, v := range s.Input {
			binary.LittleEndian.PutUint32(buf[4+i*4:], math.Float32bits(float32(v)))
		}
		if _, err := w.Write(buf); err != nil {
			tmp.Close()
			return err
		}
	}
	if err := w.Flush(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, path)
}

// LoadStripData は帯の教師データを読む。
// **壊れたファイルはエラーにする**（空として読むと、次の保存で消える）。
func LoadStripData(path string) ([]StripSample, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := bufio.NewReaderSize(f, 1<<20)

	var head [stripDataHeader]byte
	if _, err := io.ReadFull(r, head[:]); err != nil {
		return nil, fmt.Errorf("帯データのヘッダが読めません: %w", err)
	}
	if !bytes.Equal(head[:8], stripDataMagic[:]) {
		return nil, fmt.Errorf("帯データファイルではありません")
	}
	if v := binary.LittleEndian.Uint16(head[8:]); v != stripDataVersion {
		return nil, fmt.Errorf("帯データの版が違います: %d", v)
	}
	if n := int(binary.LittleEndian.Uint16(head[10:])); n != stripSize {
		return nil, fmt.Errorf("帯データの入力長が違います: %d", n)
	}
	count := int(binary.LittleEndian.Uint32(head[12:]))
	if count > 1<<24 {
		return nil, fmt.Errorf("帯データの件数が不正です: %d", count)
	}

	out := make([]StripSample, 0, count)
	buf := make([]byte, 4+stripSize*4)
	for i := 0; i < count; i++ {
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, fmt.Errorf("帯データが途中で終わっています (%d/%d): %w", i, count, err)
		}
		in := make([]float64, stripSize)
		for j := 0; j < stripSize; j++ {
			in[j] = float64(math.Float32frombits(binary.LittleEndian.Uint32(buf[4+j*4:])))
		}
		out = append(out, StripSample{Input: in, Board: binary.LittleEndian.Uint32(buf) != 0})
	}
	return out, nil
}
