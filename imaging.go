package suteme

import (
	"image"
	"image/color"
	"image/draw"
	"math"
)

func ConvertGray(src image.Image) *image.Gray {
	bounds := src.Bounds()
	gray := image.NewGray(bounds)
	draw.Draw(gray, bounds, src, bounds.Min, draw.Src)
	return gray
}

// grayValue は色を ITU-R BT.601 の輝度に変換する
func grayValue(c color.Color) uint8 {
	r, g, b, _ := c.RGBA()
	return uint8((19595*r + 38470*g + 7471*b + 1<<15) >> 24)
}

func Threshold(src *image.Gray, thresh uint8) *image.Gray {
	bounds := src.Bounds()
	dst := image.NewGray(bounds)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			if src.GrayAt(x, y).Y < thresh {
				dst.SetGray(x, y, color.Gray{Y: 255})
			}
		}
	}
	return dst
}

func BoxBlur(src *image.Gray, radius int) *image.Gray {
	bounds := src.Bounds()
	dst := image.NewGray(bounds)
	size := 2*radius + 1
	div := size * size

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			sum := 0
			for dy := -radius; dy <= radius; dy++ {
				for dx := -radius; dx <= radius; dx++ {
					ny, nx := y+dy, x+dx
					if ny < bounds.Min.Y {
						ny = bounds.Min.Y
					}
					if ny >= bounds.Max.Y {
						ny = bounds.Max.Y - 1
					}
					if nx < bounds.Min.X {
						nx = bounds.Min.X
					}
					if nx >= bounds.Max.X {
						nx = bounds.Max.X - 1
					}
					sum += int(src.GrayAt(nx, ny).Y)
				}
			}
			dst.SetGray(x, y, color.Gray{Y: uint8(sum / div)})
		}
	}
	return dst
}

// Rotate180 は画像を180度回転する
// 後手の駒を先手向きに正規化してNNに入力するために使う
func Rotate180(src image.Image) image.Image {
	bounds := src.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	dst := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			dst.Set(w-1-x, h-1-y, src.At(bounds.Min.X+x, bounds.Min.Y+y))
		}
	}
	return dst
}

// transposeGray は縦横を入れ替えたグレースケール画像を返す。
// 横線を探す処理（segment.go）をそのまま縦線に使うためのもの
func transposeGray(src *image.Gray) *image.Gray {
	b := src.Bounds()
	dst := image.NewGray(image.Rect(0, 0, b.Dy(), b.Dx()))
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			dst.SetGray(y-b.Min.Y, x-b.Min.X, src.GrayAt(x, y))
		}
	}
	return dst
}

func Sobel(src *image.Gray) *image.Gray {
	bounds := src.Bounds()
	dst := image.NewGray(bounds)

	for y := bounds.Min.Y + 1; y < bounds.Max.Y-1; y++ {
		for x := bounds.Min.X + 1; x < bounds.Max.X-1; x++ {
			gx := -int(src.GrayAt(x-1, y-1).Y) - 2*int(src.GrayAt(x-1, y).Y) - int(src.GrayAt(x-1, y+1).Y) +
				int(src.GrayAt(x+1, y-1).Y) + 2*int(src.GrayAt(x+1, y).Y) + int(src.GrayAt(x+1, y+1).Y)
			gy := -int(src.GrayAt(x-1, y-1).Y) - 2*int(src.GrayAt(x, y-1).Y) - int(src.GrayAt(x+1, y-1).Y) +
				int(src.GrayAt(x-1, y+1).Y) + 2*int(src.GrayAt(x, y+1).Y) + int(src.GrayAt(x+1, y+1).Y)

			mag := math.Sqrt(float64(gx*gx + gy*gy))
			if mag > 255 {
				mag = 255
			}
			dst.SetGray(x, y, color.Gray{Y: uint8(mag)})
		}
	}
	return dst
}
