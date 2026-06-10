package suteme

import (
	"image"
	"image/color"
)

type AnalyzeResult struct {
	Edges *image.Gray
	Board *BoardRegion
}

func Analyze(img image.Image) *AnalyzeResult {
	gray := ConvertGray(img)
	blurred := BoxBlur(gray, 2)
	edges := Sobel(blurred)
	board := DetectBoard(img)

	return &AnalyzeResult{
		Edges: edges,
		Board: board,
	}
}

func (r *AnalyzeResult) DrawBoard(src image.Image) *image.RGBA {
	out := CloneRGBA(src)
	if r.Board == nil {
		return out
	}

	bounds := src.Bounds()
	green := color.RGBA{R: 0, G: 255, B: 0, A: 255}
	yellow := color.RGBA{R: 255, G: 255, B: 0, A: 255}

	b := r.Board.Bounds
	drawRectOutline(out, b, green)

	for row := 0; row < 9; row++ {
		for col := 0; col < 9; col++ {
			cell := r.Board.Cells[row][col]
			drawRectOutline(out, cell, yellow)
		}
	}
	_ = bounds

	return out
}

func drawRectOutline(img *image.RGBA, rect image.Rectangle, c color.RGBA) {
	for x := rect.Min.X; x < rect.Max.X; x++ {
		if image.Pt(x, rect.Min.Y).In(img.Bounds()) {
			img.SetRGBA(x, rect.Min.Y, c)
		}
		if image.Pt(x, rect.Max.Y-1).In(img.Bounds()) {
			img.SetRGBA(x, rect.Max.Y-1, c)
		}
	}
	for y := rect.Min.Y; y < rect.Max.Y; y++ {
		if image.Pt(rect.Min.X, y).In(img.Bounds()) {
			img.SetRGBA(rect.Min.X, y, c)
		}
		if image.Pt(rect.Max.X-1, y).In(img.Bounds()) {
			img.SetRGBA(rect.Max.X-1, y, c)
		}
	}
}
