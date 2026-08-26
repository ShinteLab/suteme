package suteme

import (
	"image"
	"math/rand"
	"testing"
)

// boxBlurNaive は BoxBlur の素直な実装（(2r+1)^2 の窓を毎回舐める）。
// **高速化した BoxBlur の答えが変わっていないことを測るためだけに置いてある。**
// 検出・分類・帯判定はすべてこの値の上に乗っているので、
// 「速いが少し違う」は許容できない（学習データの版が上がることになる）。
func boxBlurNaive(src *image.Gray, radius int) *image.Gray {
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
			dst.Pix[dst.PixOffset(x, y)] = uint8(sum / div)
		}
	}
	return dst
}

// TestBoxBlurMatchesNaive は分離版がビット単位で同じ答えを返すことを確かめる。
// 窓より小さい画像・非ゼロ原点・部分画像（Stride > 幅）も含める。
func TestBoxBlurMatchesNaive(t *testing.T) {
	rnd := rand.New(rand.NewSource(1))
	fill := func(g *image.Gray) *image.Gray {
		for i := range g.Pix {
			g.Pix[i] = uint8(rnd.Intn(256))
		}
		return g
	}

	cases := []struct {
		name   string
		img    *image.Gray
		radius int
	}{
		{"1x1", fill(image.NewGray(image.Rect(0, 0, 1, 1))), 2},
		{"窓より小さい", fill(image.NewGray(image.Rect(0, 0, 3, 2))), 2},
		{"radius0", fill(image.NewGray(image.Rect(0, 0, 20, 12))), 0},
		{"radius1", fill(image.NewGray(image.Rect(0, 0, 41, 37))), 1},
		{"通常", fill(image.NewGray(image.Rect(0, 0, 120, 90))), 2},
		{"非ゼロ原点", fill(image.NewGray(image.Rect(13, 7, 13+64, 7+48))), 2},
		{"radius5", fill(image.NewGray(image.Rect(0, 0, 64, 48))), 5},
	}
	for _, c := range cases {
		got, want := BoxBlur(c.img, c.radius), boxBlurNaive(c.img, c.radius)
		if got.Bounds() != want.Bounds() {
			t.Errorf("%s: bounds %v want %v", c.name, got.Bounds(), want.Bounds())
			continue
		}
		b := got.Bounds()
		for y := b.Min.Y; y < b.Max.Y; y++ {
			for x := b.Min.X; x < b.Max.X; x++ {
				if g, w := got.GrayAt(x, y).Y, want.GrayAt(x, y).Y; g != w {
					t.Fatalf("%s: (%d,%d) = %d, want %d", c.name, x, y, g, w)
				}
			}
		}
	}

	// 部分画像（Stride が幅より大きい）
	full := fill(image.NewGray(image.Rect(0, 0, 100, 80)))
	sub := full.SubImage(image.Rect(20, 10, 20+50, 10+40)).(*image.Gray)
	if sub.Stride == sub.Bounds().Dx() {
		t.Fatal("部分画像になっていない（Stride == 幅）")
	}
	got, want := BoxBlur(sub, 2), boxBlurNaive(sub, 2)
	b := got.Bounds()
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			if g, w := got.GrayAt(x, y).Y, want.GrayAt(x, y).Y; g != w {
				t.Fatalf("部分画像: (%d,%d) = %d, want %d", x, y, g, w)
			}
		}
	}
}
