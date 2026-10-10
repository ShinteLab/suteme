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

// BoxBlur は半径 radius の箱ぼかしをかける。端は最も近い画素で埋める。
//
// **分離して計算する（横の窓和 → 縦の窓和 → 最後に 1 回だけ割る）。**
// 素直に (2r+1)^2 の窓を毎回舐める版は 1 画素あたり 25 回の `GrayAt`
// （境界チェック + `PixOffset`）を回すので、これが検出まわりの CPU の
// 半分を占めていた。**割り算を最後の 1 回に寄せてあるので出力は
// 総当たり版とビット単位で一致する**（横の和を int のまま持ち回るため
// 途中の丸めが入らない）。実測 1280x900 / radius 2 で 59ms → 8ms。
// 回帰テストは `TestBoxBlurMatchesNaive`。
func BoxBlur(src *image.Gray, radius int) *image.Gray {
	bounds := src.Bounds()
	dst := image.NewGray(bounds)
	w, h := bounds.Dx(), bounds.Dy()
	if w <= 0 || h <= 0 {
		return dst
	}
	size := 2*radius + 1
	div := size * size

	// 横方向の窓和。割らずに整数のまま持つ（最大 (2r+1)*255。int32 で足りる）
	rowSum := make([]int32, w*h)
	for y := 0; y < h; y++ {
		base := src.PixOffset(bounds.Min.X, bounds.Min.Y+y)
		pix := src.Pix[base : base+w]
		out := rowSum[y*w : (y+1)*w]
		sum := int32(0)
		for dx := -radius; dx <= radius; dx++ {
			sum += int32(pix[clampIndex(dx, w)])
		}
		out[0] = sum
		for x := 1; x < w; x++ {
			sum -= int32(pix[clampIndex(x-radius-1, w)])
			sum += int32(pix[clampIndex(x+radius, w)])
			out[x] = sum
		}
	}

	// 縦方向の窓和。ここで初めて div で割る。
	// **行ごとに進める**（列ごとに縦へ舐めると rowSum を w 飛びに読むことになる）。
	// 列ごとの窓和を 1 行ずつずらすだけなので、足す値も結果も列ごとに進めた場合と同じ
	colSum := make([]int32, w)
	for dy := -radius; dy <= radius; dy++ {
		r := rowSum[clampIndex(dy, h)*w : clampIndex(dy, h)*w+w]
		for x, v := range r {
			colSum[x] += v
		}
	}
	d := int32(div)
	for y := 0; y < h; y++ {
		if y > 0 {
			sub := rowSum[clampIndex(y-radius-1, h)*w:][:w]
			add := rowSum[clampIndex(y+radius, h)*w:][:w]
			for x := range colSum {
				colSum[x] += add[x] - sub[x]
			}
		}
		out := dst.Pix[dst.PixOffset(bounds.Min.X, bounds.Min.Y+y):][:w]
		for x, s := range colSum {
			out[x] = uint8(s / d)
		}
	}
	return dst
}

// clampIndex は添字を [0, n) に丸める（画像の端は最も近い画素で埋める）
func clampIndex(i, n int) int {
	if i < 0 {
		return 0
	}
	if i >= n {
		return n - 1
	}
	return i
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

	// 画素は Pix から直接読み書きする（GrayAt / SetGray は 1 回ごとに範囲の確認と
	// PixOffset を回す）。計算は同じ（`TestSobelMatchesNaive`）
	for y := bounds.Min.Y + 1; y < bounds.Max.Y-1; y++ {
		up := src.Pix[src.PixOffset(bounds.Min.X, y-1):]
		md := src.Pix[src.PixOffset(bounds.Min.X, y):]
		dn := src.Pix[src.PixOffset(bounds.Min.X, y+1):]
		out := dst.Pix[dst.PixOffset(bounds.Min.X, y):]
		for i := 1; i < bounds.Dx()-1; i++ {
			gx := -int(up[i-1]) - 2*int(md[i-1]) - int(dn[i-1]) +
				int(up[i+1]) + 2*int(md[i+1]) + int(dn[i+1])
			gy := -int(up[i-1]) - 2*int(up[i]) - int(up[i+1]) +
				int(dn[i-1]) + 2*int(dn[i]) + int(dn[i+1])

			mag := math.Sqrt(float64(gx*gx + gy*gy))
			if mag > 255 {
				mag = 255
			}
			out[i] = uint8(mag)
		}
	}
	return dst
}
