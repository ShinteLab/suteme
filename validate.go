package suteme

import (
	"image"
	"sort"
)

// ValidateBoard は検出した盤面が妥当かチェックする。
// 全81マスの背景色（中央値）を比較し、大半が似た色なら盤面と判定。
// 戻り値: 妥当なマスの割合 (0.0-1.0)
func ValidateBoard(img image.Image, br *BoardRegion) float64 {
	if br == nil {
		return 0
	}

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

func medianBrightness(img image.Image) uint8 {
	bounds := img.Bounds()
	pixels := make([]int, 0, bounds.Dx()*bounds.Dy())

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			gray := (19595*r + 38470*g + 7471*b + 1<<15) >> 24
			pixels = append(pixels, int(gray))
		}
	}

	sort.Ints(pixels)
	if len(pixels) == 0 {
		return 0
	}
	return uint8(pixels[len(pixels)/2])
}
