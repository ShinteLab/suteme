package suteme

import (
	"image"
	"math"
)

type BoardRegion struct {
	Bounds image.Rectangle
	Cells  [9][9]image.Rectangle
}

// boardAspect は将棋盤のマスの縦横比（高さ / 幅）。
//
// **将棋盤は正方形ではない。** 規格は 3.03寸 x 3.33寸 で 1.099、
// 保存済み局面の手動指定座標での実測は 1.045〜1.138（平均 1.09）。
// 正方形とみなすと横方向に 8〜9% 広い盤を探すことになる。
const boardAspect = 1.09

func DetectBoard(img image.Image) *BoardRegion {
	gray := ConvertGray(img)
	blurred := BoxBlur(gray, 2)
	edges := Sobel(blurred)

	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	minDist := minInt(w, h) / 40
	if minDist < 3 {
		minDist = 3
	}

	// Pass 1: 横方向の投影で水平線10本（外枠2本 + 内部8本）を検出。
	//
	// **上限は min(w,h)/9 ではなく h/9。** 行の間隔はマスの「高さ」なので
	// 縦の長さで抑えるのが正しい。マスは縦長（boardAspect）なうえ、盤だけを
	// 切り出した画像は縦長になるので min(w,h)=幅 とすると
	// 上限 < 真のマス高 となり、**正しい間隔が必ず弾かれていた**
	// （保存済み 11 件中 9 件が該当。ピーク自体は全て拾えているのに
	// 偽の並びが選ばれていた）。
	rowProj := make([]int, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			rowProj[y] += int(edges.GrayAt(x+bounds.Min.X, y+bounds.Min.Y).Y)
		}
	}
	rowPeaks := findPeaks(rowProj, minDist)
	hPos, hSpacing := pickWithSpacing(rowPeaks, rowProj, 10, 0, float64(h)/9)
	if hPos == nil {
		// 外枠が写っていない画像向け: 内部8本を拾って上下に1マス延長する
		var hInner []int
		hInner, hSpacing = pickWithSpacing(rowPeaks, rowProj, 8, 0, float64(h)/9)
		if hInner == nil {
			return nil
		}
		hPos = extendToBorders(hInner, 0, h, rowProj)
	}

	// Pass 2: 行範囲内の縦投影から垂直線10本を検出。
	// 期待するマス幅は hSpacing/boardAspect（正方形ではない）
	yTop := hPos[0]
	yBot := hPos[len(hPos)-1]
	colProj := make([]int, w)
	for x := 0; x < w; x++ {
		for y := maxInt(yTop, 0); y <= minInt(yBot, h-1); y++ {
			colProj[x] += int(edges.GrayAt(x+bounds.Min.X, y+bounds.Min.Y).Y)
		}
	}

	colPeaks := findPeaks(colProj, minDist)
	cellW := hSpacing / boardAspect
	vPos, _ := pickWithSpacing(colPeaks, colProj, 10, cellW, float64(w)/9)
	if vPos == nil {
		vPos = findBoardColumns(colProj, colPeaks, cellW, w)
		if vPos == nil {
			return nil
		}
	}

	// **検出したピーク位置をそのままマスの境界に使わない。** 外枠だけを採り、
	// 内側は等分する。ピークは線の太さやボケで数px 揺れるので、そのまま使うと
	// 1マスの高さが 47〜58px のようにばらつき、グリッド線がマス内に入り込む。
	// 手動指定（BoardRegionFromRect）と同じ割り方になるので、
	// 「手動座標なら合うのに検出だと合わない」の切り分けもしやすい。
	// 実測での改善は誤認識マス 266 → 258（保存済み10局面・810マス）と小さい。
	return BoardRegionFromRect(
		vPos[0]+bounds.Min.X,
		hPos[0]+bounds.Min.Y,
		vPos[9]+bounds.Min.X,
		hPos[9]+bounds.Min.Y,
	)
}

// findBoardColumns は等間隔10本を取れなかったときのフォールバック。
// 期待幅 cellW*9 に収まる左右の境界ペアを探し、見つからなければ
// スライディングウィンドウで最もエッジ密度の高い位置を採る
func findBoardColumns(colProj []int, colPeaks []int, cellW float64, w int) []int {
	expectedW := int(cellW * 9)
	if expectedW <= 0 || expectedW > w {
		return nil
	}
	tolerance := int(cellW)

	bestScore := 0
	bestLeft, bestRight := 0, 0
	for _, l := range colPeaks {
		for _, r := range colPeaks {
			span := r - l
			if span < expectedW-tolerance || span > expectedW+tolerance {
				continue
			}
			if score := colProj[l] + colProj[r]; score > bestScore {
				bestScore, bestLeft, bestRight = score, l, r
			}
		}
	}

	if bestRight <= bestLeft {
		for x := 0; x <= w-expectedW; x++ {
			sum := 0
			for dx := 0; dx < expectedW; dx++ {
				sum += colProj[x+dx]
			}
			if sum > bestScore {
				bestScore, bestLeft, bestRight = sum, x, x+expectedW
			}
		}
		if bestRight <= bestLeft {
			return nil
		}
	}

	boardW := bestRight - bestLeft
	vPos := make([]int, 10)
	for i := 0; i < 10; i++ {
		vPos[i] = bestLeft + boardW*i/9
	}
	return vPos
}

// findPeaks は投影の局所最大を返す。
//
// **端も走査対象にする（比較窓を配列内に切り詰める）。** かつては
// `for i := minDist; i < len(signal)-minDist` として端を捨てていたが、
// 盤だけを切り出した画像では盤の外枠線が画像端（実測で x=2 / y=4 など）に
// 来るため、外枠が原理的にピークになれなかった。保存済み局面は 11 件中 9 件が
// この形（盤が画像の 92〜98% を占める）で、横方向が常に1マスずれる原因だった。
func findPeaks(signal []int, minDist int) []int {
	maxVal := 0
	for _, v := range signal {
		if v > maxVal {
			maxVal = v
		}
	}
	thresh := maxVal / 5

	peaks := make([]int, 0)
	for i := 0; i < len(signal); i++ {
		if signal[i] < thresh {
			continue
		}
		isPeak := true
		for d := 1; d <= minDist && isPeak; d++ {
			if i-d >= 0 && signal[i] < signal[i-d] {
				isPeak = false
			}
			if i+d < len(signal) && signal[i] < signal[i+d] {
				isPeak = false
			}
		}
		if isPeak {
			peaks = append(peaks, i)
		}
	}
	return peaks
}

func pickWithSpacing(peaks []int, signal []int, count int, targetSpacing float64, maxSpacing float64) ([]int, float64) {
	n := len(peaks)
	minFound := count * 6 / 10
	if minFound < 5 {
		minFound = 5
	}
	if n < minFound {
		return nil, 0
	}

	bestFound := 0
	bestScore := 0
	bestSpacing := 0.0
	bestOrigin := 0.0
	bestSlots := make([]int, count)

	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			diff := float64(peaks[j] - peaks[i])
			for gap := 1; gap <= count-1; gap++ {
				spacing := diff / float64(gap)
				if spacing < 10 {
					continue
				}

				if maxSpacing > 0 && spacing > maxSpacing {
					continue
				}
				if targetSpacing > 0 {
					ratio := spacing / targetSpacing
					if ratio < 0.7 || ratio > 1.4 {
						continue
					}
				}

				tolerance := spacing * 0.3
				origin := float64(peaks[i])
				found := 0
				score := 0
				slots := make([]int, count)

				for k := 0; k < count; k++ {
					expected := origin + float64(k)*spacing
					bestDist := tolerance
					bestIdx := -1
					for idx, p := range peaks {
						d := math.Abs(float64(p) - expected)
						if d < bestDist {
							bestDist = d
							bestIdx = idx
						}
					}
					if bestIdx >= 0 {
						slots[k] = peaks[bestIdx]
						found++
						score += signal[peaks[bestIdx]]
					} else {
						slots[k] = -1
					}
				}

				if found < minFound {
					continue
				}
				if found > bestFound || (found == bestFound && score > bestScore) {
					bestFound = found
					bestScore = score
					bestSpacing = spacing
					bestOrigin = origin
					copy(bestSlots, slots)
				}
			}
		}
	}

	if bestFound < minFound {
		return nil, 0
	}

	result := make([]int, count)
	for k := 0; k < count; k++ {
		if bestSlots[k] >= 0 {
			result[k] = bestSlots[k]
		} else {
			result[k] = int(bestOrigin + float64(k)*bestSpacing)
		}
	}

	return result, bestSpacing
}

func extendToBorders(inner []int, minBound, maxBound int, proj []int) []int {
	n := len(inner)
	spacing := float64(inner[n-1]-inner[0]) / float64(n-1)

	result := make([]int, n+2)

	// 上端: 1マス分延長するが、画像端に近すぎる場合は半マス
	top := int(float64(inner[0]) - spacing)
	if top < minBound {
		top = int(float64(inner[0]) - spacing/2)
		if top < minBound {
			top = minBound
		}
	}
	result[0] = top

	for i, v := range inner {
		result[i+1] = v
	}

	// 下端: 同様
	bot := int(float64(inner[n-1]) + spacing)
	if bot > maxBound {
		bot = int(float64(inner[n-1]) + spacing/2)
		if bot > maxBound {
			bot = maxBound
		}
	}
	result[n+1] = bot

	return result
}

// BoardRegionFromRect は手動指定の2点からBoardRegionを作成する
func BoardRegionFromRect(x1, y1, x2, y2 int) *BoardRegion {
	left, right := x1, x2
	if left > right {
		left, right = right, left
	}
	top, bottom := y1, y2
	if top > bottom {
		top, bottom = bottom, top
	}
	w := right - left
	h := bottom - top
	br := &BoardRegion{Bounds: image.Rect(left, top, right, bottom)}
	for r := 0; r < 9; r++ {
		for c := 0; c < 9; c++ {
			br.Cells[r][c] = image.Rect(
				left+w*c/9, top+h*r/9,
				left+w*(c+1)/9, top+h*(r+1)/9,
			)
		}
	}
	return br
}

// cellPadding はセル抽出時に各辺に加える余白の割合（0.15 = 15%）
const cellPadding = 0.15

func (br *BoardRegion) ExtractCell(src image.Image, row, col int) image.Image {
	cell := br.Cells[row][col]
	pad := int(float64(cell.Dx()) * cellPadding)
	expanded := image.Rect(
		cell.Min.X-pad, cell.Min.Y-pad,
		cell.Max.X+pad, cell.Max.Y+pad,
	)
	rect := expanded.Intersect(src.Bounds())
	if rect.Empty() {
		return nil
	}
	dst := image.NewRGBA(image.Rect(0, 0, rect.Dx(), rect.Dy()))
	for y := rect.Min.Y; y < rect.Max.Y; y++ {
		for x := rect.Min.X; x < rect.Max.X; x++ {
			dst.Set(x-rect.Min.X, y-rect.Min.Y, src.At(x, y))
		}
	}
	return dst
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
