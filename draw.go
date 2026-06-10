package suteme

import (
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"os"
)

func DrawLine(img *image.RGBA, p1, p2 image.Point, c color.RGBA) {
	dx := p2.X - p1.X
	dy := p2.Y - p1.Y
	if dx < 0 {
		dx = -dx
	}
	if dy < 0 {
		dy = -dy
	}

	sx, sy := 1, 1
	if p1.X > p2.X {
		sx = -1
	}
	if p1.Y > p2.Y {
		sy = -1
	}

	err := dx - dy
	bounds := img.Bounds()

	for {
		if p1.In(bounds) {
			img.SetRGBA(p1.X, p1.Y, c)
		}
		if p1.X == p2.X && p1.Y == p2.Y {
			break
		}
		e2 := 2 * err
		if e2 > -dy {
			err -= dy
			p1.X += sx
		}
		if e2 < dx {
			err += dx
			p1.Y += sy
		}
	}
}

func CloneRGBA(src image.Image) *image.RGBA {
	bounds := src.Bounds()
	dst := image.NewRGBA(bounds)
	draw.Draw(dst, bounds, src, bounds.Min, draw.Src)
	return dst
}

func SaveImage(path string, img image.Image) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}
