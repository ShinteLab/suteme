package suteme

import (
	"image"
	"math"
	"sort"
)

// ValidateBoard は検出した盤面が妥当かチェックする。
// 戻り値は 0.0-1.0 の信頼度で、次の 2 つの積。
//
//   - 背景色の均一性 … 81マスの中央値輝度が揃っているか（「そもそも盤か」）
//   - グリッド整合度 … 10x10 の境界線が実際の格子線に乗っているか（「マス割りが正しいか」）
//
// **均一性だけでは検出のズレを検知できない。** 盤の内側はどこを切り取っても
// 木目が均一なので、1〜3マスずれた領域でも均一性は 1.00 を返す。実測で
// 保存済み 11 局面すべてが誤検出なのに 10 件が 1.00、残り 1 件が 0.94 だった。
// その結果 detectBoardRegion の「画像全体が盤面」フォールバックが一度も
// 発動しないという状態になっていた。マス割りの正しさは別に測る必要がある。
func ValidateBoard(img image.Image, br *BoardRegion) float64 {
	if br == nil {
		return 0
	}
	u := cellUniformity(img, br)
	if u == 0 {
		return 0
	}
	return u * gridConfidence(img, br)
}

// cellUniformity は81マスの背景色（中央値）のうち、
// 盤の基準色に近いものの割合を返す。
//
// **マスの切り出しは `extractCellUnclipped`（盤の外枠より外も含める版）を使う。**
// ここは「この領域が本当に盤か」を測る場所で、領域が間違っている前提で呼ばれる。
// 外枠で切ると外れた候補も自分の箱の中しか見なくなり、均一に見えてしまう。
func cellUniformity(img image.Image, br *BoardRegion) float64 {
	return cellUniformitySkip(img, br, nil)
}

// cellUniformitySkip は cellUniformity のうち、skip が立ったマスを数えないもの
// （割合の分母からも外す）。遮られたマスを外して比べるのに使う（`unslipRegion`）。
func cellUniformitySkip(img image.Image, br *BoardRegion, skip *[9][9]bool) float64 {
	medians := make([]uint8, 0, 81)
	for r := 0; r < 9; r++ {
		for c := 0; c < 9; c++ {
			if skip != nil && skip[r][c] {
				continue
			}
			cell := br.extractCellUnclipped(img, r, c)
			if cell == nil {
				return 0
			}
			medians = append(medians, medianBrightness(cell))
		}
	}

	// 全マスの中央値の中央値 = 盤面の基準色
	sorted := make([]int, len(medians))
	for i, v := range medians {
		sorted[i] = int(v)
	}
	sort.Ints(sorted)
	boardColor := sorted[len(sorted)/2]

	// 基準色に近いマスを数える (±40 の範囲)
	const tolerance = 40
	ok := 0
	for _, v := range medians {
		diff := int(v) - boardColor
		if diff < 0 {
			diff = -diff
		}
		if diff <= tolerance {
			ok++
		}
	}

	if len(medians) == 0 {
		return 0
	}
	return float64(ok) / float64(len(medians))
}

// gridAlignFull は gridAlignment の値をこれ以上なら整合度 1.0 とみなす基準。
//
// 実測（保存済み33局面・悪いほうの軸）: 正解座標 +0.31〜+0.97 に対し、
// 半マスずらすと -1.00〜-0.41、周期を半分にすると -0.99〜+0.01。
// 0.50 を上限に線形にすると、既定の棄却しきい値 minBoardConfidence(0.5) が
// gridAlignment = 0.25 に対応する。歪みの最大 +0.01 と正解の最小 +0.31 の
// 間なので、正解を全件通しつつ壊れた検出を落とせる。
//
// **lineProjections や axisAlignment を変えたらこの値も測り直すこと。**
// 合計投影だった頃は正解が +0.13〜+0.56 で 0.20 が妥当、
// on をパリティで割る前（10本の平均）は正解 +0.38〜 で 0.60 が妥当だった。
const gridAlignFull = 0.50

// gridConfidence は gridAlignment を 0.0-1.0 に写す
func gridConfidence(img image.Image, br *BoardRegion) float64 {
	a := gridAlignment(img, br) / gridAlignFull
	if a <= 0 {
		return 0
	}
	if a > 1 {
		return 1
	}
	return a
}

// gridAlignment はマス割りの境界線が実際の格子線に乗っている度合いを
// -1.0〜1.0 で返す。1 に近いほど正しく乗っている。
//
// 盤面領域内のエッジ投影について、10本の境界線位置のピーク値（on）と、
// 線が無いはずの位置のピーク値（off）を比べ、(on-off)/(on+off) とする。
// **同じ画像の同じ領域どうしの比なので自己校正になる**（絶対量で測ると
// 駒の多い局面・ボケた画像・解像度で桁が変わり、しきい値が置けない）。
//
// **縦横は平均ではなく悪いほうを採る。** 半マスずらすと、ずらした軸だけが
// 壊れてもう一方は正解のまま高い値を出す。平均にすると横半マスずれが
// +0.05〜+0.13、縦半マスずれが -0.03〜+0.16 まで上がり、しきい値 0.10 を
// 超えるものが出て弁別できなくなる（実測）。min なら -0.04〜-0.38 に落ちる。
//
// 実物の盤を撮った画像では、片方の軸の格子線が投影にほとんど出ないことがある
// （木目・照明・駒の重なり）。その場合この値は 0 近辺までしか上がらず、
// 手動指定の正しい領域でも低く出る。**手動指定はこの値でゲートしていない**
// （Recognize の WithRegion も /api/setboard も検証せずそのまま使う）ので
// 実害は「信頼度の表示が低い」ことに留まる。
func gridAlignment(img image.Image, br *BoardRegion) float64 {
	b := br.Bounds.Intersect(img.Bounds())
	// 1マスが数pxしかない領域では投影に意味が無い
	if b.Dx() < 9*3 || b.Dy() < 9*3 {
		return 0
	}

	sub := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			sub.Set(x-b.Min.X, y-b.Min.Y, img.At(x, y))
		}
	}
	rowProj, colProj := lineProjections(BoxBlur(ConvertGray(sub), 2))

	// br.Bounds は画像からはみ出しうるので、投影の添字は b.Min からの相対にする
	v, vOn := axisAlignmentOn(rowProj, float64(br.Bounds.Min.Y-b.Min.Y), float64(br.Bounds.Dy())/9)
	h, hOn := axisAlignmentOn(colProj, float64(br.Bounds.Min.X-b.Min.X), float64(br.Bounds.Dx())/9)

	// **線が無い軸は比では落ちない。絶対量で落とす。**
	if vOn < minLineStrength || hOn < minLineStrength {
		return 0
	}

	if v < h {
		return v
	}
	return h
}

// minLineStrength は「その軸に格子線がある」と認めるための on の下限。
// 単位は lineProjections の値＝行/列ごとの |勾配| の中央値（0..255）。
//
// **`axisAlignment` の比 (on-off)/(on+off) は線の有無を測れない。**
// 自己校正のために off で割っているので、**線が 1 本も無い軸は
// off も 0 になり、比が +1 に張り付く**。実測（9本の縞模様だけの画像
// `d112c258`）: 縞に直交する軸が on=0.8 / off=0.0 で整合度 +1.00、
// 均一性 0.78 と掛けて**信頼度 0.78 で盤として採用**されていた。
// 縞と平行な軸（on=27.2）だけが本物で、直交する軸には何も無い。
//
// 比は候補どうしの順位付けには効くが、「そもそも格子か」は測れない。
// **絶対量で測るのはここだけ**（AGENTS.md の「絶対量で測ってはいけない」は
// 領域どうしの比較の話で、この下限は解像度・駒数では動かない
// ＝中央値なので面積に比例せず、コントラストだけで決まる）。
//
// 実測（保存済み 138 局面の手動指定座標、悪いほうの軸の on）:
// 最小 12.6 / 下位 5% 26.8 / 中央値 68.2。負例側は 0.2〜0.8 なので
// 桁で離れている。8 は最も弱い正解の 1.6 分の 1。
//
// **`lineProjections` を変えたらこの値も測り直すこと。**
const minLineStrength = 8

// lineProjections は「その行/列にどれだけ線が通っているか」の投影を返す。
//
// **合計ではなく中央値を採る。方向別の勾配を使う。** どちらも実測で効いている。
//
//   - **中央値**: 格子線は盤の端から端まで通るので、その行（列）の**大半の画素**が
//     エッジになる。合計だと、駒の輪郭や漢字の画のような**一部分に集中した強いエッジ**が
//     線と同じだけの量を稼いでしまう。中央値なら局所的なものは無視される
//   - **方向別**: 横線は |gy|、縦線は |gx| が強い。勾配強度（Sobel）のままだと
//     木目や駒の輪郭のような向きを持たないエッジが両軸に等しく乗る
//
// 実測（保存済み15局面、`gridAlignment` の生値。悪いほうの軸）:
//
//	                   正解の最小   正解−歪みの最小差
//	Sobel強度・合計      +0.02        -0.04（正解が歪みに負ける画像がある）
//	方向別・合計         +0.14        +0.15
//	方向別・エッジ被覆率  +0.30        +0.21
//	**方向別・中央値**   **+0.38**    **+0.26**
//
// **実物の盤を撮った中継画像（`26e06136`）の列方向が +0.02 → +0.38 になる。**
// 木目・照明・駒の重なりで格子線が合計投影にほとんど出ない画像であり、
// この値が上がらないことが「正しい手動座標でも信頼度 0.12 止まり」の原因だった。
//
// 中央値は 0..255 にクランプしたヒストグラムで取る（クランプ無しの厳密な
// 中央値と実測値は一致する）。ValidateBoard は候補ごとに呼ぶので、
// 列ごとのソート O(n log n) を避ける意味がある。
func lineProjections(blurred *image.Gray) (row, col []float64) {
	return lineProjectionsAxes(blurred, true, true)
}

// lineProjectionsAxes は要る軸だけを計算する `lineProjections`。
//
// **片方しか使わない呼び出しがある**（`axisProjection`＝`SnapToGrid` は
// 縦線か横線のどちらか一方しか見ない）のに両方返していたので、
// **Sobel と中央値の半分が捨てられていた**。`SnapToGrid` は収束するまで
// 繰り返し、そのたびに 2 つの範囲で投影を取るので効きが大きい。
// 要らない側は勾配の計算ごと省く。返り値の中身は `lineProjections` と同じで、
// 求めなかった側は 0 埋めのまま返る。
func lineProjectionsAxes(blurred *image.Gray, wantRow, wantCol bool) (row, col []float64) {
	b := blurred.Bounds()
	w, h := b.Dx(), b.Dy()
	row = make([]float64, h)
	col = make([]float64, w)
	if w < 3 || h < 3 || (!wantRow && !wantCol) {
		return row, col
	}

	at := func(x, y int) int { return int(blurred.GrayAt(x+b.Min.X, y+b.Min.Y).Y) }
	var gx, gy []int
	if wantCol {
		gx = make([]int, w*h)
	}
	if wantRow {
		gy = make([]int, w*h)
	}
	for y := 1; y < h-1; y++ {
		for x := 1; x < w-1; x++ {
			tl, tc, tr := at(x-1, y-1), at(x, y-1), at(x+1, y-1)
			bl, bc, br := at(x-1, y+1), at(x, y+1), at(x+1, y+1)
			if wantCol {
				gx[y*w+x] = absInt(-tl - 2*at(x-1, y) - bl + tr + 2*at(x+1, y) + br)
			}
			if wantRow {
				gy[y*w+x] = absInt(-tl - 2*tc - tr + bl + 2*bc + br)
			}
		}
	}

	var hist [256]int
	median := func(n int) float64 {
		acc := 0
		for i, c := range hist {
			acc += c
			if acc > n/2 {
				return float64(i)
			}
		}
		return 0
	}
	if wantRow {
		for y := 0; y < h; y++ {
			hist = [256]int{}
			for x := 0; x < w; x++ {
				hist[minInt(gy[y*w+x], 255)]++
			}
			row[y] = median(w)
		}
	}
	if wantCol {
		for x := 0; x < w; x++ {
			hist = [256]int{}
			for y := 0; y < h; y++ {
				hist[minInt(gx[y*w+x], 255)]++
			}
			col[x] = median(h)
		}
	}
	return row, col
}

// gridPeakTol は境界線位置のずれを許す範囲(px)。
// BoxBlur(r=2) とグリッド線自体の太さで数px はずれる
const gridPeakTol = 3

// offFractions は「線が無いはずの位置」を取る、マス内の相対位置。
//
// **マス中央（0.5）だけで測ってはいけない。** 駒の漢字の画はマスの中央に
// 集中するので、中央だけを off にすると駒のある盤で off が過大になり、
// 整合度が不当に下がる。実物の盤を撮った画像では列方向がこれで潰れていた。
// 1/4・1/2・3/4 に散らすと、正解の値が上がり歪みの値は 0 に寄る
// （実測: 正解 +0.13〜+0.54 → +0.16〜+0.51、縮小 +0.17 → +0.04 など）。
var offFractions = []float64{0.25, 0.5, 0.75}

// axisAlignment は 1 軸分の整合度を返す。
// origin から span 間隔で並ぶ10本の境界線と、線が無いはずの位置を比べる。
//
// **on は10本の平均ではなく、偶数番と奇数番の弱いほうを採る。**
// 平均だと**周期が半分の当たり方**を弾けない。マス2つぶんを1マスとみなすと
// 境界線が1本おきに本物の格子線に乗るので、乗っていない残り半分が
// マスの内側でも on の平均は本物の半分ほどまでしか下がらず、
// off（同じくマスの内側）を上回ったままになる。実測（保存済み33局面で
// 盤の1/4を切り出したもの、悪いほうの軸）: **平均だと +0.94 まで出る**のに対し、
// パリティの弱いほうなら最大 +0.01。正解座標のほうは +0.38〜+0.97 が
// +0.31〜+0.97 とほぼ変わらないので、この一点で両者が分離する。
//
// 「1本おきに合っている」は本物の格子には起こらない（10本とも線に乗る）ので、
// パリティで割っても正解側は失うものが無い、というのがこの測り方の根拠。
//
// **パリティごとに中央値を採るのは行き過ぎ**（「10本すべてが乗っている」を
// もっと厳しく測ることになる）。off がほぼ 0 の画面（一様な背景に線が数本）で
// 比が +1 に張り付く穴は塞がるが、格子線が薄い実盤の中継画像で正解座標が
// **−0.33** まで落ちて使えなくなる（実測。AGENTS.md「今後の課題」参照）。
func axisAlignment(proj []float64, origin, span float64) float64 {
	a, _ := axisAlignmentOn(proj, origin, span)
	return a
}

// axisAlignmentOn は axisAlignment の値と、その元になった on を返す。
// on は「境界線の位置にどれだけ線があるか」の絶対量で、
// 線の無い軸を落とすのに使う（minLineStrength）。
func axisAlignmentOn(proj []float64, origin, span float64) (float64, float64) {
	peak := func(pos float64) float64 {
		best := 0.0
		p := int(pos)
		for d := -gridPeakTol; d <= gridPeakTol; d++ {
			if i := p + d; i >= 0 && i < len(proj) && proj[i] > best {
				best = proj[i]
			}
		}
		return best
	}

	even, odd, off := 0.0, 0.0, 0.0
	for i := 0; i <= 9; i++ {
		if i%2 == 0 {
			even += peak(origin + span*float64(i))
		} else {
			odd += peak(origin + span*float64(i))
		}
	}
	for i := 0; i < 9; i++ {
		for _, fr := range offFractions {
			off += peak(origin + span*(float64(i)+fr))
		}
	}
	on := math.Min(even, odd) / 5
	off /= float64(9 * len(offFractions))
	if on+off == 0 {
		return 0, on
	}
	return (on - off) / (on + off), on
}

// weakestLineRatio は10本の境界線のうち**いちばん弱い線**と off レベルの比を
// -1.0〜1.0 で返す。`axisAlignment` と同じ (on-off)/(on+off) だが、
// on を「最小」で採る。
//
// **これは信頼度に使ってはいけない。候補どうしの比較専用。**
// 「10本すべてが線に乗っていること」を最も厳しく測る形なので、格子線が薄い
// 実盤の中継画像では正しい座標でも値が落ちる（AGENTS.md「今後の課題」に
// パリティごとの中央値でさえ正解が -0.33 まで落ちた記録がある）。
// 絶対値にしきい値を置く用途には耐えない。
//
// **一方で、同じ画像・同じ軸の候補どうしを比べるなら効く。** 1マス滑った窓は
// 9 本が本物の格子線に乗る代わりに外側の 1 本が線の無い所に来るので、
// 最弱の線だけが off に並ぶ。`axisAlignment` の和ではこの 1 本が
// 埋もれてしまう（実測 `2d7bfedf` の列方向: 整合度は 0.88 対 0.90 と
// ほぼ同じなのに、最弱の線の比は -0.12 対 0.87）。
// 使い手は `snapToOuterFrame`。
func weakestLineRatio(proj []float64, origin, span float64) float64 {
	peak := func(pos float64) float64 {
		best := 0.0
		p := int(pos)
		for d := -gridPeakTol; d <= gridPeakTol; d++ {
			if i := p + d; i >= 0 && i < len(proj) && proj[i] > best {
				best = proj[i]
			}
		}
		return best
	}

	on := math.Inf(1)
	for i := 0; i <= 9; i++ {
		if v := peak(origin + span*float64(i)); v < on {
			on = v
		}
	}
	off := 0.0
	for i := 0; i < 9; i++ {
		for _, fr := range offFractions {
			off += peak(origin + span*(float64(i)+fr))
		}
	}
	off /= float64(9 * len(offFractions))
	if on+off == 0 {
		return 0
	}
	return (on - off) / (on + off)
}

func medianBrightness(img image.Image) uint8 {
	bounds := img.Bounds()
	pixels := make([]int, 0, bounds.Dx()*bounds.Dy())

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			pixels = append(pixels, int(grayValue(img.At(x, y))))
		}
	}

	sort.Ints(pixels)
	if len(pixels) == 0 {
		return 0
	}
	return uint8(pixels[len(pixels)/2])
}
