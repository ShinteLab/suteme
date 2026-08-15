package suteme

import (
	"image"
	"image/color"
	"testing"
)

// 盤ごとの空/駒の境目（BoardEmptyCover の中身）。
func TestAdaptiveEmptyCover(t *testing.T) {
	// 空 0.00〜0.14 / 駒 0.40〜0.76 の盤（模様のある盤の実測値の形）。
	// 0.10 固定だと 0.10〜0.14 の空マスが駒に化けるので、境目が上がってほしい。
	patterned := []float64{
		0.00, 0.01, 0.02, 0.03, 0.06, 0.08, 0.10, 0.11, 0.12, 0.14,
		0.40, 0.45, 0.49, 0.56, 0.61, 0.72, 0.76,
	}
	// 駒の色が盤の地色とほぼ同じ木目の盤。駒が 0.13 から始まるので
	// 境目を動かすと駒が空に化ける。0.10 のままでなければならない。
	woodgrain := []float64{
		0.00, 0.01, 0.02, 0.03, 0.05,
		0.13, 0.16, 0.19, 0.22, 0.25, 0.27,
	}

	if got := adaptiveEmptyCover(patterned); got <= 0.14 || got >= 0.40 {
		t.Errorf("分かれている盤: 境目 %.3f は空(max 0.14)と駒(min 0.40)の間に来るべき", got)
	}
	if got := adaptiveEmptyCover(woodgrain); got != emptyCoverMax {
		t.Errorf("木目の盤: 境目 %.3f、既定の %.3f のままであるべき", got, emptyCoverMax)
	}
	// 実写のように空と駒が重なる盤も既定のまま
	overlap := []float64{0.02, 0.05, 0.11, 0.18, 0.24, 0.29, 0.36, 0.38, 0.42}
	if got := adaptiveEmptyCover(overlap); got != emptyCoverMax {
		t.Errorf("重なる盤: 境目 %.3f、既定の %.3f のままであるべき", got, emptyCoverMax)
	}
	// 境目は emptyCoverHigh を超えない（超えると木目の盤の駒を飲み込む）
	if got := adaptiveEmptyCover([]float64{0.01, 0.02, 0.55, 0.60}); got > emptyCoverHigh {
		t.Errorf("境目 %.3f が上限 %.3f を超えている", got, emptyCoverHigh)
	}
}

// ExtractCell の余白が盤の外枠より外へ出ないこと。
// 外周マスの局所地色が枠の色に引きずられると、盤の木地まで「駒」に見える。
func TestExtractCellStaysInsideBoard(t *testing.T) {
	const size = 200
	img := image.NewRGBA(image.Rect(0, 0, size, size))
	// 盤の外は真っ黒、盤の中は一様な明るい地色
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			img.Set(x, y, color.RGBA{0, 0, 0, 255})
		}
	}
	board := image.Rect(20, 20, 200-20, 200-20)
	for y := board.Min.Y; y < board.Max.Y; y++ {
		for x := board.Min.X; x < board.Max.X; x++ {
			img.Set(x, y, color.RGBA{200, 180, 140, 255})
		}
	}
	br := BoardRegionFromRect(board.Min.X, board.Min.Y, board.Max.X, board.Max.Y)

	for _, rc := range [][2]int{{0, 0}, {0, 8}, {8, 0}, {8, 8}, {0, 4}, {4, 0}} {
		cell := br.ExtractCell(img, rc[0], rc[1])
		if cell == nil {
			t.Fatalf("r%dc%d: セルが取れない", rc[0], rc[1])
		}
		b := cell.Bounds()
		for y := b.Min.Y; y < b.Max.Y; y++ {
			for x := b.Min.X; x < b.Max.X; x++ {
				if grayValue(cell.At(x, y)) < 100 {
					t.Fatalf("r%dc%d: 盤の外（暗い画素）が入っている", rc[0], rc[1])
				}
			}
		}
	}
}
