package suteme

import (
	"image"
	"image/color"
	"image/draw"
	"math"
	"math/rand"
	"sort"
	"testing"
)

// 盤面検出を速くした部分（grayplane.go ほか）が、**総当たりの旧実装と同じ答えを返す**こと。
// 旧実装はここに写してある（`boxBlurNaive` と同じ流儀）。

// copyToRGBA は旧実装の「範囲 r を *image.RGBA へ写す」（原点は 0）。
func copyToRGBA(img image.Image, r image.Rectangle) *image.RGBA {
	sub := image.NewRGBA(image.Rect(0, 0, r.Dx(), r.Dy()))
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			sub.Set(x-r.Min.X, y-r.Min.Y, img.At(x, y))
		}
	}
	return sub
}

// medianBrightnessNaive は旧 `medianBrightness`（全画素を並べて真ん中）。
func medianBrightnessNaive(img image.Image) uint8 {
	b := img.Bounds()
	pixels := make([]int, 0, b.Dx()*b.Dy())
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			pixels = append(pixels, int(grayValue(img.At(x, y))))
		}
	}
	sort.Ints(pixels)
	if len(pixels) == 0 {
		return 0
	}
	return uint8(pixels[len(pixels)/2])
}

// cellUniformityNaive は旧 `cellUniformitySkip`（マスを *image.RGBA に写して中央値を並べ替えで取る）。
func cellUniformityNaive(img image.Image, br *BoardRegion, skip *[9][9]bool) float64 {
	medians := make([]uint8, 0, 81)
	for r := 0; r < 9; r++ {
		for c := 0; c < 9; c++ {
			if skip != nil && skip[r][c] {
				continue
			}
			rect := unclippedCellRect(img, br, r, c)
			if rect.Empty() {
				return 0
			}
			medians = append(medians, medianBrightnessNaive(copyToRGBA(img, rect)))
		}
	}
	sorted := make([]int, len(medians))
	for i, v := range medians {
		sorted[i] = int(v)
	}
	sort.Ints(sorted)
	boardColor := sorted[len(sorted)/2]
	ok := 0
	for _, v := range medians {
		d := int(v) - boardColor
		if d < 0 {
			d = -d
		}
		if d <= 40 {
			ok++
		}
	}
	return float64(ok) / float64(len(medians))
}

// lineProjectionsNaive は旧 `lineProjections`（GrayAt で読み、int の勾配を持つ）。
func lineProjectionsNaive(blurred *image.Gray) (row, col []float64) {
	b := blurred.Bounds()
	w, h := b.Dx(), b.Dy()
	row = make([]float64, h)
	col = make([]float64, w)
	if w < 3 || h < 3 {
		return row, col
	}
	at := func(x, y int) int { return int(blurred.GrayAt(x+b.Min.X, y+b.Min.Y).Y) }
	gx := make([]int, w*h)
	gy := make([]int, w*h)
	for y := 1; y < h-1; y++ {
		for x := 1; x < w-1; x++ {
			tl, tc, tr := at(x-1, y-1), at(x, y-1), at(x+1, y-1)
			bl, bc, br := at(x-1, y+1), at(x, y+1), at(x+1, y+1)
			gx[y*w+x] = absInt(-tl - 2*at(x-1, y) - bl + tr + 2*at(x+1, y) + br)
			gy[y*w+x] = absInt(-tl - 2*tc - tr + bl + 2*bc + br)
		}
	}
	median := func(v []int) float64 {
		s := append([]int(nil), v...)
		for i := range s {
			s[i] = minInt(s[i], 255)
		}
		sort.Ints(s)
		return float64(s[len(s)/2])
	}
	for y := 0; y < h; y++ {
		row[y] = median(gy[y*w : (y+1)*w])
	}
	for x := 0; x < w; x++ {
		v := make([]int, h)
		for y := 0; y < h; y++ {
			v[y] = gx[y*w+x]
		}
		col[x] = median(v)
	}
	return row, col
}

// gridAlignmentNaive は旧 `gridAlignment`（範囲を *image.RGBA に写して ConvertGray）。
func gridAlignmentNaive(img image.Image, br *BoardRegion) float64 {
	b := br.Bounds.Intersect(img.Bounds())
	if b.Dx() < 9*3 || b.Dy() < 9*3 {
		return 0
	}
	rowProj, colProj := lineProjectionsNaive(boxBlurNaive(ConvertGray(copyToRGBA(img, b)), 2))
	v, vOn := axisAlignmentOn(rowProj, float64(br.Bounds.Min.Y-b.Min.Y), float64(br.Bounds.Dy())/9)
	h, hOn := axisAlignmentOn(colProj, float64(br.Bounds.Min.X-b.Min.X), float64(br.Bounds.Dx())/9)
	if vOn < minLineStrength || hOn < minLineStrength {
		return 0
	}
	return math.Min(v, h)
}

func validateBoardNaive(img image.Image, br *BoardRegion) float64 {
	u := cellUniformityNaive(img, br, nil)
	if u == 0 {
		return 0
	}
	a := gridAlignmentNaive(img, br) / gridAlignFull
	if a <= 0 {
		return 0
	}
	if a > 1 {
		a = 1
	}
	return u * a
}

// sobelNaive は旧 `Sobel`（GrayAt / SetGray）。
func sobelNaive(src *image.Gray) *image.Gray {
	b := src.Bounds()
	dst := image.NewGray(b)
	for y := b.Min.Y + 1; y < b.Max.Y-1; y++ {
		for x := b.Min.X + 1; x < b.Max.X-1; x++ {
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

// testImages は色の持ち方が違う画像（同じ盤の絵）。JPEG（YCbCr）と半透明の NRGBA は
// `*image.RGBA` へ写すと 8bit に丸められるので、輝度の取り方の違いが出やすい。
func testImages(t *testing.T) map[string]image.Image {
	t.Helper()
	board := makeBoardImage([]testPiece{{row: 2, col: 3, gote: true}, {row: 6, col: 5}, {row: 0, col: 0}})
	rng := rand.New(rand.NewSource(1))
	b := board.Bounds().Inset(-7) // 盤の外にも少し余白
	rgba := image.NewRGBA(b)
	nrgba := image.NewNRGBA(b)
	ycc := image.NewYCbCr(b, image.YCbCrSubsampleRatio420)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			v := 230
			if (image.Point{x, y}).In(board.Bounds()) {
				v = int(board.GrayAt(x, y).Y)
			}
			v += rng.Intn(21) - 10
			v = minInt(maxInt(v, 0), 255)
			c := color.RGBA{uint8(v), uint8(minInt(v+rng.Intn(40), 255)), uint8(v / 2), 255}
			rgba.SetRGBA(x, y, c)
			nrgba.SetNRGBA(x, y, color.NRGBA{c.R, c.G, c.B, uint8(128 + rng.Intn(128))})
			yy, cb, cr := color.RGBToYCbCr(c.R, c.G, c.B)
			ycc.Y[ycc.YOffset(x, y)] = yy
			ycc.Cb[ycc.COffset(x, y)] = cb
			ycc.Cr[ycc.COffset(x, y)] = cr
		}
	}
	return map[string]image.Image{"rgba": rgba, "nrgba": nrgba, "ycbcr": ycc, "gray": board}
}

func TestRGBAGrayMatchesCopy(t *testing.T) {
	for name, img := range testImages(t) {
		b := img.Bounds()
		for _, r := range []image.Rectangle{b, image.Rect(b.Min.X+3, b.Min.Y+5, b.Max.X-11, b.Max.Y-2), image.Rect(40, 41, 97, 130)} {
			want := ConvertGray(copyToRGBA(img, r))
			for _, src := range []image.Image{img, withGrayPlane(img, b), withGrayPlane(img, r.Inset(4))} {
				got := rgbaGray(src, r)
				if got.Bounds() != r {
					t.Fatalf("%s: bounds %v, want %v", name, got.Bounds(), r)
				}
				for y := 0; y < r.Dy(); y++ {
					for x := 0; x < r.Dx(); x++ {
						if g, w := got.GrayAt(r.Min.X+x, r.Min.Y+y).Y, want.GrayAt(x, y).Y; g != w {
							t.Fatalf("%s %v (%d,%d): %d, want %d", name, r, x, y, g, w)
						}
					}
				}
			}
		}
		// DetectBoard の近道: *image.RGBA なら ConvertGray(img) とも同じ
		if _, ok := img.(*image.RGBA); ok {
			want := ConvertGray(img)
			got := rgbaGray(img, b)
			for i := range want.Pix {
				if want.Pix[i] != got.Pix[i] {
					t.Fatalf("%s: ConvertGray と違う (%d)", name, i)
				}
			}
		}
	}
}

func TestCellUniformityMatchesCopy(t *testing.T) {
	var skip [9][9]bool
	skip[0][0], skip[4][5] = true, true
	for name, img := range testImages(t) {
		b := img.Bounds()
		for _, br := range []*BoardRegion{
			BoardRegionFromRect(0, 0, testCell*9, testCell*9),
			BoardRegionFromRect(-20, 10, testCell*9-15, testCell*9+30), // はみ出す窓
			BoardRegionFromRect(b.Min.X, b.Min.Y, b.Max.X, b.Max.Y),
		} {
			for _, sk := range []*[9][9]bool{nil, &skip} {
				want := cellUniformityNaive(img, br, sk)
				for _, src := range []image.Image{img, withGrayPlane(img, b)} {
					if got := cellUniformitySkip(src, br, sk); got != want {
						t.Errorf("%s %v: %v, want %v", name, br.Bounds, got, want)
					}
				}
			}
		}
	}
}

func TestValidateBoardMatchesNaive(t *testing.T) {
	for name, img := range testImages(t) {
		b := img.Bounds()
		for _, br := range []*BoardRegion{
			BoardRegionFromRect(0, 0, testCell*9, testCell*9),
			BoardRegionFromRect(testCell/2, 3, testCell*9+testCell/2, testCell*9+3), // 半マスずれ
			BoardRegionFromRect(-20, 10, testCell*9-15, testCell*9+30),
		} {
			want := validateBoardNaive(img, br)
			if got := ValidateBoard(img, br); got != want {
				t.Errorf("%s %v: %v, want %v", name, br.Bounds, got, want)
			}
			// 検出の中で使う控え（同じ値を返すこと）
			p := withGrayPlane(img, b)
			for i := 0; i < 2; i++ {
				if got := ValidateBoard(p, br); got != want {
					t.Errorf("%s %v（控え %d 回目）: %v, want %v", name, br.Bounds, i, got, want)
				}
			}
		}
	}
}

func TestLineProjectionsMatchesNaive(t *testing.T) {
	for name, img := range testImages(t) {
		g := BoxBlur(ConvertGray(img), 2)
		for _, sub := range []*image.Gray{g, g.SubImage(image.Rect(10, 12, 200, 170)).(*image.Gray)} {
			wr, wc := lineProjectionsNaive(sub)
			gr, gc := lineProjections(sub)
			for i := range wr {
				if wr[i] != gr[i] {
					t.Fatalf("%s row[%d] = %v, want %v", name, i, gr[i], wr[i])
				}
			}
			for i := range wc {
				if wc[i] != gc[i] {
					t.Fatalf("%s col[%d] = %v, want %v", name, i, gc[i], wc[i])
				}
			}
		}
	}
}

func TestSobelMatchesNaive(t *testing.T) {
	for name, img := range testImages(t) {
		g := BoxBlur(ConvertGray(img), 2)
		for _, sub := range []*image.Gray{g, g.SubImage(image.Rect(10, 12, 200, 170)).(*image.Gray)} {
			want, got := sobelNaive(sub), Sobel(sub)
			r := sub.Bounds()
			for y := r.Min.Y; y < r.Max.Y; y++ {
				for x := r.Min.X; x < r.Max.X; x++ {
					if want.GrayAt(x, y) != got.GrayAt(x, y) {
						t.Fatalf("%s (%d,%d): %v, want %v", name, x, y, got.GrayAt(x, y), want.GrayAt(x, y))
					}
				}
			}
		}
	}
}

// horizontalLinesNaive の閾値は旧実装（全値を並べて分位を取る）。
func horizontalLinesThreshNaive(blurred *image.Gray) int {
	b := blurred.Bounds()
	w, h := b.Dx(), b.Dy()
	var vals []int
	for y := 1; y < h-1; y++ {
		for x := 1; x < w-1; x++ {
			X, Y := x+b.Min.X, y+b.Min.Y
			v := -int(blurred.GrayAt(X-1, Y-1).Y) - 2*int(blurred.GrayAt(X, Y-1).Y) - int(blurred.GrayAt(X+1, Y-1).Y) +
				int(blurred.GrayAt(X-1, Y+1).Y) + 2*int(blurred.GrayAt(X, Y+1).Y) + int(blurred.GrayAt(X+1, Y+1).Y)
			vals = append(vals, absInt(v))
		}
	}
	sort.Ints(vals)
	return vals[int(float64(len(vals)-1)*segEdgePercentile)]
}

// 度数で取る分位が並べ替えと同じになること、線分の答えが変わらないこと
// （線分そのものは閾値と |gy| だけで決まる）。
func TestHorizontalLinesMatchesNaive(t *testing.T) {
	for name, img := range testImages(t) {
		g := BoxBlur(ConvertGray(img), 2)
		for _, sub := range []*image.Gray{g, g.SubImage(image.Rect(10, 12, 200, 170)).(*image.Gray), transposeGray(g)} {
			want := horizontalLinesThreshNaive(sub)
			// 閾値を直接は返さないので、同じ閾値で線分を作る旧実装と突き合わせる
			got := horizontalLines(sub)
			ref := horizontalLinesWithThresh(sub, maxInt(want, 1))
			if len(got) != len(ref) {
				t.Fatalf("%s: %d 本, want %d 本", name, len(got), len(ref))
			}
			for i := range got {
				if got[i] != ref[i] {
					t.Fatalf("%s [%d]: %v, want %v", name, i, got[i], ref[i])
				}
			}
		}
	}
}

// horizontalLinesWithThresh は旧 `horizontalLines` の閾値より後ろ（閾値を外から与える）。
func horizontalLinesWithThresh(blurred *image.Gray, thresh int) []lineSegment {
	b := blurred.Bounds()
	w, h := b.Dx(), b.Dy()
	gy := make([]int, w*h)
	for y := 1; y < h-1; y++ {
		for x := 1; x < w-1; x++ {
			X, Y := x+b.Min.X, y+b.Min.Y
			v := -int(blurred.GrayAt(X-1, Y-1).Y) - 2*int(blurred.GrayAt(X, Y-1).Y) - int(blurred.GrayAt(X+1, Y-1).Y) +
				int(blurred.GrayAt(X-1, Y+1).Y) + 2*int(blurred.GrayAt(X, Y+1).Y) + int(blurred.GrayAt(X+1, Y+1).Y)
			gy[y*w+x] = absInt(v)
		}
	}
	minLen := maxInt(int(float64(w)*segMinLenRatio), 10)
	on := func(x, y int) bool {
		for dy := -1; dy <= 1; dy++ {
			if ny := y + dy; ny >= 0 && ny < h && gy[ny*w+x] >= thresh {
				return true
			}
		}
		return false
	}
	var rows []lineSegment
	for y := 1; y < h-1; y++ {
		best := lineSegment{pos: y, start: 0, end: -1}
		for x := 0; x < w; {
			if !on(x, y) {
				x++
				continue
			}
			start, last := x, x
			for x < w {
				if on(x, y) {
					last = x
				} else if x-last > segGapTol {
					break
				}
				x++
			}
			if last-start > best.end-best.start {
				best = lineSegment{pos: y, start: start, end: last}
			}
		}
		if best.end-best.start >= minLen {
			rows = append(rows, lineSegment{pos: y, start: best.start + b.Min.X, end: best.end + b.Min.X})
		}
	}
	tol := maxInt(w/50, 10)
	var lines []lineSegment
	for _, r := range rows {
		if n := len(lines); n > 0 && r.pos-lines[n-1].pos <= segGapTol &&
			absInt(r.start-lines[n-1].start) < tol && absInt(r.end-lines[n-1].end) < tol {
			continue
		}
		lines = append(lines, r)
	}
	return lines
}

// peakTable が axisAlignment とビット単位で同じ値を返すこと（refineAxis が使う）。
func TestPeakTableMatchesAxisAlignment(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for trial := 0; trial < 20; trial++ {
		proj := make([]float64, 50+rng.Intn(400))
		for i := range proj {
			proj[i] = float64(rng.Intn(120))
		}
		pt := newPeakTable(proj)
		for k := 0; k < 300; k++ {
			origin := rng.Float64()*float64(len(proj)) - 30
			span := 3 + rng.Float64()*float64(len(proj))/8
			if a, b := axisAlignment(proj, origin, span), pt.alignment(origin, span); math.Float64bits(a) != math.Float64bits(b) {
				t.Fatalf("origin=%v span=%v: %v, want %v", origin, span, b, a)
			}
		}
	}
}

// 検出の答えが並列化・近道で変わらないこと（同じ画像を何度検出しても同じ、
// *image.RGBA と同じ絵の *image.NRGBA（不透明）でも同じ）。
func TestDetectBoardDeterministic(t *testing.T) {
	imgs := testImages(t)
	rgba := imgs["rgba"].(*image.RGBA)
	opaque := image.NewNRGBA(rgba.Bounds())
	draw.Draw(opaque, opaque.Bounds(), rgba, rgba.Bounds().Min, draw.Src)
	first := DetectBoard(rgba)
	for i := 0; i < 5; i++ {
		for _, img := range []image.Image{rgba, opaque} {
			got := DetectBoard(img)
			if (got == nil) != (first == nil) || (got != nil && *got != *first) {
				t.Fatalf("%d 回目で答えが変わった: %v / %v", i, got, first)
			}
		}
	}
}
