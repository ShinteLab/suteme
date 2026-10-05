package suteme

import (
	"image"
	"image/color"
	"testing"
)

// syntheticBoard は罫線と駒を描いた 9x9 の盤（1 マス 40x44px、余白 30px）を返す。
// 駒は**マスより大きく辺の中ほどの罫線に被るもの**も置く（ゲーム画面・木の盤で
// 普通に起きる。四辺の罫線で判定していた頃はこれを「見えない」と読んでいた）。
func syntheticBoard() (*image.RGBA, *BoardRegion) {
	const cw, ch, m = 40, 44, 30
	img := image.NewRGBA(image.Rect(0, 0, 9*cw+2*m, 9*ch+2*m))
	fill := func(r image.Rectangle, c color.RGBA) {
		r = r.Intersect(img.Bounds())
		for y := r.Min.Y; y < r.Max.Y; y++ {
			for x := r.Min.X; x < r.Max.X; x++ {
				img.SetRGBA(x, y, c)
			}
		}
	}
	fill(img.Bounds(), color.RGBA{150, 110, 60, 255})                                  // 盤の外
	fill(image.Rect(m-10, m-10, m+9*cw+10, m+9*ch+10), color.RGBA{220, 180, 110, 255}) // 盤
	line := color.RGBA{90, 60, 30, 255}
	for i := 0; i <= 9; i++ {
		fill(image.Rect(m+i*cw, m, m+i*cw+1, m+9*ch+1), line)
		fill(image.Rect(m, m+i*ch, m+9*cw+1, m+i*ch+1), line)
	}
	piece := func(r, c int, big bool) {
		x0, y0 := m+c*cw, m+r*ch
		box := image.Rect(x0+6, y0+5, x0+cw-6, y0+ch-4)
		if big { // 上下の辺の中ほどに被る
			box = image.Rect(x0+8, y0-5, x0+cw-8, y0+ch+5)
		}
		fill(box, color.RGBA{240, 215, 160, 255})
		ink := color.RGBA{30, 20, 10, 255}
		cx, cy := (box.Min.X+box.Max.X)/2, (box.Min.Y+box.Max.Y)/2
		fill(image.Rect(cx-8, cy-1, cx+8, cy+2), ink)
		fill(image.Rect(cx-1, cy-10, cx+2, cy+10), ink)
	}
	for c := 0; c < 9; c++ {
		piece(2, c, false)
		piece(6, c, c%2 == 0)
	}
	piece(4, 4, true)
	piece(0, 4, false)
	return img, BoardRegionFromRect(m, m, m+9*cw, m+9*ch)
}

func countHidden(h [9][9]bool) (n int) {
	for r := 0; r < 9; r++ {
		for c := 0; c < 9; c++ {
			if h[r][c] {
				n++
			}
		}
	}
	return n
}

// 何も被っていない盤（辺に被る大きな駒を含む）では、見えないマスを返さないこと。
func TestHiddenCellsCleanSynthetic(t *testing.T) {
	img, br := syntheticBoard()
	if h := HiddenCells(img, br); countHidden(h) > 0 {
		t.Errorf("何も被っていないのに見えないマスがある: %v", h)
	}
}

// 肌色（盤の地色に近い）の塊が被ったマスを「見えない」と言うこと。
// 色では見分けられないので、罫線の交点が消えていることで拾う。
func TestHiddenCellsSkinBlob(t *testing.T) {
	img, br := syntheticBoard()
	// 6七〜3九あたり（段 6..8、列 3..6）に手を置く。盤の地色とほぼ同じ明るさ
	hand := image.Rect(br.Cells[6][3].Min.X+10, br.Cells[6][3].Min.Y+8, br.Cells[8][6].Max.X-10, br.Bounds.Max.Y+20)
	for y := hand.Min.Y; y < hand.Max.Y; y++ {
		for x := hand.Min.X; x < hand.Max.X; x++ {
			img.SetRGBA(x, y, color.RGBA{230, 175, 140, 255})
		}
	}
	h := HiddenCells(img, br)
	for r := 7; r <= 8; r++ {
		for c := 4; c <= 5; c++ {
			if !h[r][c] {
				t.Errorf("手の下の %s を見えないと言っていない", squareName(r, c))
			}
		}
	}
	for r := 0; r <= 4; r++ {
		for c := 0; c < 9; c++ {
			if h[r][c] {
				t.Errorf("手から離れた %s を見えないと言った", squareName(r, c))
			}
		}
	}
}

// 髪のような暗い塊が上辺に被ったマスを「見えない」と言うこと。
//
// **塊の縁が外枠のすぐ下で切れていると、縁が外枠の罫線に見える**（外枠は盤の内側と
// だけ比べるので、暗い髪と明るい盤の境目は「暗い線」と区別が付かない）。
// 実例（`143758-001-P@3a`）がこの形で、太い暗い塊の篩（solidDark）が無いと 3一 を落とす。
func TestHiddenCellsDarkBlobOnFrame(t *testing.T) {
	img, br := syntheticBoard()
	top := br.Bounds.Min.Y
	x6 := br.Cells[0][6].Min.X // 4一と 3一の境の縦線
	paint := func(x0, x1, y1 int) {
		for y := 0; y < y1; y++ {
			for x := x0; x < x1; x++ {
				img.SetRGBA(x, y, color.RGBA{25, 20, 18, 255})
			}
		}
	}
	paint(br.Cells[0][4].Min.X-12, x6+2, top+20) // 5一〜4一の上半分まで深く
	paint(x6+2, x6+14, top+4)                    // 3一の上は外枠をかすめるだけ
	h := HiddenCells(img, br)
	for _, c := range []int{4, 5, 6} {
		if !h[0][c] {
			t.Errorf("髪の下の %s を見えないと言っていない", squareName(0, c))
		}
	}
	for r := 2; r < 9; r++ {
		for c := 0; c < 9; c++ {
			if h[r][c] {
				t.Errorf("髪から離れた %s を見えないと言った", squareName(r, c))
			}
		}
	}
}

// Recognize の Debug にマスごとの印が載ること。読んだ結果（Piece / Category）は変えない。
func TestRecognizeMarksHiddenCells(t *testing.T) {
	img, br := syntheticBoard()
	clean, err := Recognize(img, WithRegion(br), WithPredictor(constPredictor{}))
	if err != nil {
		t.Fatal(err)
	}
	if n := len(clean.Debug.HiddenCells()); n != 0 {
		t.Fatalf("何も被っていないのに見えないマスが %d", n)
	}
	hand := image.Rect(br.Cells[6][3].Min.X+10, br.Cells[6][3].Min.Y+8, br.Cells[8][6].Max.X-10, br.Bounds.Max.Y+20)
	for y := hand.Min.Y; y < hand.Max.Y; y++ {
		for x := hand.Min.X; x < hand.Max.X; x++ {
			img.SetRGBA(x, y, color.RGBA{230, 175, 140, 255})
		}
	}
	r, err := Recognize(img, WithRegion(br), WithPredictor(constPredictor{}))
	if err != nil {
		t.Fatal(err)
	}
	hid := r.Debug.HiddenCells()
	if len(hid) == 0 {
		t.Fatal("手が被っているのに見えないマスが無い")
	}
	for _, c := range hid {
		if got := r.Debug.Cell(c.Row, c.Col); got == nil || !got.Hidden {
			t.Errorf("Cell(%d,%d).Hidden が立っていない", c.Row, c.Col)
		}
	}
}

// constPredictor は常に歩を返す推論器（認識の流れだけを通すため）。
type constPredictor struct{}

func (constPredictor) Predict(image.Image) (int, float64) { return 0, 1 }
