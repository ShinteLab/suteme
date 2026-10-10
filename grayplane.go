package suteme

import (
	"image"
	"image/color"
	"runtime"
	"sync"
	"sync/atomic"
)

// 盤面検出の輝度を「1 枚につき 1 回だけ」作るための仕組み。
//
// 検出と検証（`ValidateBoard` / `refineRegion` / `snapToOuterFrame` / `SnapToGrid`）は
// 候補の窓ごとに同じ画像の同じあたりを何度も読む。読むたびに `image.Image.At` で
// 1 画素ずつ色を取り出し、`*image.RGBA` へ写してから輝度に落としていたので、
// **プロファイルの 4 分の 1 以上がインタフェース経由の画素アクセスと、その色の
// 詰め替え**だった（`image.(*RGBA).At` / `runtime.convTnoptr` / `(*RGBA).Set` / `draw.Draw`）。
//
// **輝度の値は変えない。** 以前の経路は「`*image.RGBA` へ写す（`color.RGBAModel` で
// 8bit に丸める）→ 輝度（`grayValue` と同じ式）」なので、ここでも同じ 2 段を通す
// （`rgbaLuma`）。JPEG のように 8bit より細かい色を返す画像では、丸めを省くと
// 輝度が 1 違う画素が出る。回帰テストは `TestRGBAGrayMatchesCopy`。

// grayPlaned は画像と、その一部（area）の輝度をまとめて持つ。
// `image.Image` を埋め込むので、At などはそのまま元の画像に届く。
type grayPlaned struct {
	image.Image
	area image.Rectangle
	gray *image.Gray // bounds = area

	// confs は `ValidateBoard` の結果の控え（同じ窓を何度も測るので）。
	// キーは BoardRegion の値そのもの（外枠とマス割り）。`ValidateBoard` は
	// 画像と窓だけで決まるので、控えを返しても値は変わらない
	confs sync.Map // BoardRegion → float64
}

// withGrayPlane は img の area の輝度を先に作っておいた画像を返す。
// 既に area を覆う輝度を持っていればそのまま返す。
func withGrayPlane(img image.Image, area image.Rectangle) image.Image {
	if img == nil {
		return nil
	}
	area = area.Intersect(img.Bounds())
	if p, ok := img.(*grayPlaned); ok {
		if area.In(p.area) {
			return p
		}
		img = p.Image
	}
	if area.Empty() {
		return img
	}
	return &grayPlaned{Image: img, area: area, gray: rgbaGrayOf(img, area)}
}

// rgbaGray は img の範囲 r の輝度を、範囲 r をそのまま bounds に持つ `*image.Gray` で返す。
// 中身は「r を `*image.RGBA` へ写して `ConvertGray` したもの」と画素単位で一致する。
// **返した画像に書き込まないこと**（先に作った輝度を共有していることがある）。
func rgbaGray(img image.Image, r image.Rectangle) *image.Gray {
	if p, ok := img.(*grayPlaned); ok {
		if r.In(p.area) {
			return p.gray.SubImage(r).(*image.Gray)
		}
		img = p.Image
	}
	return rgbaGrayOf(img, r)
}

func rgbaGrayOf(img image.Image, r image.Rectangle) *image.Gray {
	g := image.NewGray(r)
	w := r.Dx()
	for y := r.Min.Y; y < r.Max.Y; y++ {
		out := g.Pix[g.PixOffset(r.Min.X, y) : g.PixOffset(r.Min.X, y)+w]
		switch src := img.(type) {
		case *image.RGBA:
			// color.RGBAModel は color.RGBA をそのまま返す
			p := src.Pix[src.PixOffset(r.Min.X, y):]
			for x := range out {
				out[x] = rgbaLuma(p[4*x], p[4*x+1], p[4*x+2])
			}
		case *image.NRGBA:
			p := src.Pix[src.PixOffset(r.Min.X, y):]
			for x := range out {
				c := color.NRGBA{R: p[4*x], G: p[4*x+1], B: p[4*x+2], A: p[4*x+3]}
				out[x] = nrgbaLuma(c)
			}
		default:
			for x := range out {
				c := color.RGBAModel.Convert(img.At(r.Min.X+x, y)).(color.RGBA)
				out[x] = rgbaLuma(c.R, c.G, c.B)
			}
		}
	}
	return g
}

// rgbaLuma は 8bit の RGB の輝度（`grayValue` を color.RGBA に掛けたのと同じ）。
func rgbaLuma(r8, g8, b8 uint8) uint8 {
	r := uint32(r8) * 0x101
	g := uint32(g8) * 0x101
	b := uint32(b8) * 0x101
	return uint8((19595*r + 38470*g + 7471*b + 1<<15) >> 24)
}

// nrgbaLuma は color.NRGBA を color.RGBAModel で丸めてから輝度にしたもの
// （`color.NRGBA.RGBA` と `color.RGBAModel` の計算をそのまま写してある）。
func nrgbaLuma(c color.NRGBA) uint8 {
	if c.A == 0xff {
		return rgbaLuma(c.R, c.G, c.B)
	}
	r := uint32(c.R)
	r |= r << 8
	r *= uint32(c.A)
	r /= 0xff
	g := uint32(c.G)
	g |= g << 8
	g *= uint32(c.A)
	g /= 0xff
	b := uint32(c.B)
	b |= b << 8
	b *= uint32(c.A)
	b /= 0xff
	return rgbaLuma(uint8(r>>8), uint8(g>>8), uint8(b>>8))
}

// parallelFor は fn(0..n-1) を並列に走らせて、すべて終わるまで待つ。
// fn は i 番目の置き場所にしか書かないこと（`recognizeCells` と同じ約束）。
func parallelFor(n int, fn func(i int)) {
	workers := runtime.GOMAXPROCS(0)
	if workers > n {
		workers = n
	}
	if workers < 2 {
		for i := 0; i < n; i++ {
			fn(i)
		}
		return
	}
	var next atomic.Int32
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				i := int(next.Add(1)) - 1
				if i >= n {
					return
				}
				fn(i)
			}
		}()
	}
	wg.Wait()
}

// medianGrayRect は輝度画像の範囲 r の中央値（小さい順に並べて len/2 番目）。
// 並べて真ん中を取るのと同じ定義をヒストグラムで取る（`TestCellUniformityMatchesCopy`）。
func medianGrayRect(g *image.Gray, r image.Rectangle) uint8 {
	if r.Empty() {
		return 0
	}
	var hist [256]int
	w := r.Dx()
	for y := r.Min.Y; y < r.Max.Y; y++ {
		i := g.PixOffset(r.Min.X, y)
		for _, v := range g.Pix[i : i+w] {
			hist[v]++
		}
	}
	half := r.Dx() * r.Dy() / 2
	sum := 0
	for v, n := range hist {
		sum += n
		if sum > half {
			return uint8(v)
		}
	}
	return 255
}
