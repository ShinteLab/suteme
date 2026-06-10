package suteme

import (
	"image"
	"math"
)

type BoardRegion struct {
	Bounds image.Rectangle
	Cells  [9][9]image.Rectangle
}

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

	// Pass 1: 横方向の投影で行を検出
	rowProj := make([]int, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			rowProj[y] += int(edges.GrayAt(x+bounds.Min.X, y+bounds.Min.Y).Y)
		}
	}
	rowPeaks := findPeaks(rowProj, minDist)
	maxSpacing := float64(minInt(w, h)) / 9.0
	hInner, hSpacing := pickWithSpacing(rowPeaks, rowProj, 8, 0, maxSpacing)
	if hInner == nil {
		hInner, hSpacing = pickWithSpacing(rowPeaks, rowProj, 6, 0, maxSpacing)
		if hInner == nil {
			return nil
		}
	}

	// Pass 2: 行範囲内の縦投影を計算
	yTop := hInner[0]
	yBot := hInner[len(hInner)-1]
	colProj := make([]int, w)
	for x := 0; x < w; x++ {
		for y := yTop; y <= yBot; y++ {
			colProj[x] += int(edges.GrayAt(x+bounds.Min.X, y+bounds.Min.Y).Y)
		}
	}

	// 盤面左右の境界ペアを探す（正方形制約付き）
	colPeaks := findPeaks(colProj, minDist)
	expectedW := int(hSpacing * 9)
	tolerance := int(hSpacing * 2)

	bestScore := 0
	bestLeft, bestRight := 0, 0
	for _, l := range colPeaks {
		for _, r := range colPeaks {
			span := r - l
			if span < expectedW-tolerance || span > expectedW+tolerance {
				continue
			}
			score := colProj[l] + colProj[r]
			if score > bestScore {
				bestScore = score
				bestLeft = l
				bestRight = r
			}
		}
	}

	if bestRight <= bestLeft {
		// フォールバック: スライディングウィンドウで密度最大の位置を使う
		for x := 0; x <= w-expectedW; x++ {
			sum := 0
			for dx := 0; dx < expectedW; dx++ {
				sum += colProj[x+dx]
			}
			if sum > bestScore {
				bestScore = sum
				bestLeft = x
				bestRight = x + expectedW
			}
		}
		if bestRight <= bestLeft {
			return nil
		}
	}

	hPos := extendToBorders(hInner, 0, h, rowProj)

	// 縦は検出した左右境界から9等分
	boardW := bestRight - bestLeft
	vPos := make([]int, 10)
	for i := 0; i < 10; i++ {
		vPos[i] = bestLeft + boardW*i/9
	}

	boardRect := image.Rect(
		vPos[0]+bounds.Min.X,
		hPos[0]+bounds.Min.Y,
		vPos[9]+bounds.Min.X,
		hPos[9]+bounds.Min.Y,
	)

	br := &BoardRegion{Bounds: boardRect}
	for r := 0; r < 9; r++ {
		for c := 0; c < 9; c++ {
			br.Cells[r][c] = image.Rect(
				vPos[c]+bounds.Min.X,
				hPos[r]+bounds.Min.Y,
				vPos[c+1]+bounds.Min.X,
				hPos[r+1]+bounds.Min.Y,
			)
		}
	}

	return br
}

func findPeaks(signal []int, minDist int) []int {
	maxVal := 0
	for _, v := range signal {
		if v > maxVal {
			maxVal = v
		}
	}
	thresh := maxVal / 5

	peaks := make([]int, 0)
	for i := minDist; i < len(signal)-minDist; i++ {
		if signal[i] < thresh {
			continue
		}
		isPeak := true
		for d := 1; d <= minDist; d++ {
			if signal[i] < signal[i-d] || signal[i] < signal[i+d] {
				isPeak = false
				break
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
