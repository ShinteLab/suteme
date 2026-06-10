package suteme

import (
	"image"
	"image/color"
)

// CellCategory はマスの大分類
type CellCategory int

const (
	CellEmpty    CellCategory = 0 // 空
	CellPieceUp  CellCategory = 1 // 先手（上向き）
	CellPieceDown CellCategory = 2 // 後手（下向き）
)

func (c CellCategory) String() string {
	switch c {
	case CellEmpty:
		return "空"
	case CellPieceUp:
		return "☗"
	case CellPieceDown:
		return "☖"
	}
	return "?"
}

// ClassifyCell はマス画像を 空/先手/後手 に分類する
func ClassifyCell(cell image.Image) CellCategory {
	gray := ConvertGray(cell)
	bounds := gray.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	if w == 0 || h == 0 {
		return CellEmpty
	}

	// 1. 分散チェック：低分散なら空マス
	var sum, sumSq float64
	count := float64(w * h)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			v := float64(gray.GrayAt(x, y).Y)
			sum += v
			sumSq += v * v
		}
	}
	mean := sum / count
	variance := sumSq/count - mean*mean

	if variance < 1500 {
		return CellEmpty
	}

	// 2. 向き判定：上端の行 vs 下端の行で「盤面と異なるピクセル」を比較
	//    駒の五角形は尖り側が狭く、底辺側が広い
	//    → 幅広い側はエッジ行に盤面色でないピクセルが多い
	diffThresh := mean * 0.15
	edgeH := h / 5

	topDiff := 0
	for y := bounds.Min.Y; y < bounds.Min.Y+edgeH; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			v := float64(gray.GrayAt(x, y).Y)
			d := v - mean
			if d < 0 {
				d = -d
			}
			if d > diffThresh {
				topDiff++
			}
		}
	}

	botDiff := 0
	for y := bounds.Max.Y - edgeH; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			v := float64(gray.GrayAt(x, y).Y)
			d := v - mean
			if d < 0 {
				d = -d
			}
			if d > diffThresh {
				botDiff++
			}
		}
	}

	// 盤面と異なるピクセルが多い側 = 駒の幅広い側（底辺）
	// 先手（上向き）: 底辺が下 → botDiff > topDiff
	// 後手（下向き）: 底辺が上 → topDiff > botDiff
	if topDiff > botDiff {
		return CellPieceDown
	}
	return CellPieceUp
}

// ClassifyBoard は盤面の全81マスを分類する
func ClassifyBoard(img image.Image, br *BoardRegion) [9][9]CellCategory {
	var result [9][9]CellCategory
	for r := 0; r < 9; r++ {
		for c := 0; c < 9; c++ {
			cell := br.ExtractCell(img, r, c)
			if cell != nil {
				result[r][c] = ClassifyCell(cell)
			}
		}
	}
	return result
}

func regionAvg(gray *image.Gray, x0, y0, w, h int) float64 {
	sum := 0.0
	count := 0
	for y := y0; y < y0+h; y++ {
		for x := x0; x < x0+w; x++ {
			c := gray.At(x, y).(color.Gray)
			sum += float64(c.Y)
			count++
		}
	}
	if count == 0 {
		return 0
	}
	return sum / float64(count)
}
