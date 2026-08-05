package suteme

import (
	"image"
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
// 盤の基準色に近いものの割合を返す
func cellUniformity(img image.Image, br *BoardRegion) float64 {
	medians := make([]uint8, 0, 81)
	for r := 0; r < 9; r++ {
		for c := 0; c < 9; c++ {
			cell := br.ExtractCell(img, r, c)
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

	return float64(ok) / 81.0
}

// gridAlignFull は gridAlignment の値をこれ以上なら整合度 1.0 とみなす基準。
//
// 実測（保存済み11局面）: 正解座標 +0.13〜+0.56 に対し、
// 半マスずらすと -0.61〜-0.14、周期を 8/9 に縮めると -0.09〜+0.17、
// 壊れた DetectBoard の結果は -0.25〜+0.02。
// 0.20 を上限に線形にすると、既定の棄却しきい値 minBoardConfidence(0.5) が
// gridAlignment ≒ 0.10 に対応し、正解を全件通しつつ誤検出を全件落とせる。
const gridAlignFull = 0.20

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
	edges := Sobel(BoxBlur(ConvertGray(sub), 2))

	rowProj := make([]float64, b.Dy())
	colProj := make([]float64, b.Dx())
	for y := 0; y < b.Dy(); y++ {
		for x := 0; x < b.Dx(); x++ {
			v := float64(edges.GrayAt(x, y).Y)
			rowProj[y] += v
			colProj[x] += v
		}
	}

	// br.Bounds は画像からはみ出しうるので、投影の添字は b.Min からの相対にする
	v := axisAlignment(rowProj, float64(br.Bounds.Min.Y-b.Min.Y), float64(br.Bounds.Dy())/9)
	h := axisAlignment(colProj, float64(br.Bounds.Min.X-b.Min.X), float64(br.Bounds.Dx())/9)
	if v < h {
		return v
	}
	return h
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
// origin から span 間隔で並ぶ10本の境界線と、線が無いはずの位置を比べる
func axisAlignment(proj []float64, origin, span float64) float64 {
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

	on, off := 0.0, 0.0
	for i := 0; i <= 9; i++ {
		on += peak(origin + span*float64(i))
	}
	for i := 0; i < 9; i++ {
		for _, fr := range offFractions {
			off += peak(origin + span*(float64(i)+fr))
		}
	}
	on /= 10
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
