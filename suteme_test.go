package suteme

import (
	"errors"
	"image"
	"image/color"
	"testing"
)

// stubPredictor は常に同じ駒種を返す推論器（LoadSFEN の配線確認用）
type stubPredictor struct {
	class int
	calls int
}

func (s *stubPredictor) Predict(cell image.Image) (int, float64) {
	s.calls++
	return s.class, 1.0
}

const (
	testCell   = 40  // 1マスの辺
	testBoardV = 200 // 盤面の明るさ
	testPieceV = 30  // 駒（暗いブロブ）の明るさ
)

// testPiece は合成画像に置く駒。gote=true なら駒をマスの上端側に置く
// （ClassifyCell は「盤面と異なるピクセルが多い側 = 駒の底辺」で向きを決める）
type testPiece struct {
	row, col int
	gote     bool
}

// makeBoardImage は 9x9 の合成盤面画像を作る。
// 盤面は一様なので DetectBoard は失敗し、detectBoardRegion の
// 「画像全体が盤面」フォールバックが使われる。
func makeBoardImage(pieces []testPiece) *image.Gray {
	size := testCell * 9
	img := image.NewGray(image.Rect(0, 0, size, size))
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			img.SetGray(x, y, color.Gray{Y: testBoardV})
		}
	}
	// マス面積の約8%を占めるブロブ。これより大きいと平均が引っ張られ、
	// 盤面色まで「異なるピクセル」に数えられて向き判定が壊れる
	const blobW, blobH = 16, 8
	for _, p := range pieces {
		x0 := p.col*testCell + (testCell-blobW)/2
		y0 := p.row*testCell + testCell - blobH // 先手: 下端
		if p.gote {
			y0 = p.row * testCell // 後手: 上端
		}
		for y := y0; y < y0+blobH; y++ {
			for x := x0; x < x0+blobW; x++ {
				img.SetGray(x, y, color.Gray{Y: testPieceV})
			}
		}
	}
	return img
}

func TestLoadSFENWithEmptyBoard(t *testing.T) {
	img := makeBoardImage(nil)
	p := &stubPredictor{class: LabelToClass("K")}

	got, err := LoadSFENWith(img, p)
	if err != nil {
		t.Fatalf("LoadSFENWith: %v", err)
	}
	if want := "9/9/9/9/9/9/9/9/9"; got != want {
		t.Errorf("LoadSFENWith = %q, want %q", got, want)
	}
	if p.calls != 0 {
		t.Errorf("空マスで推論器が %d 回呼ばれた", p.calls)
	}
}

func TestLoadSFENWithPieces(t *testing.T) {
	// 端のマスは ExtractCell のパディングが画像外で切られるため内側に置く
	img := makeBoardImage([]testPiece{
		{row: 2, col: 3, gote: true},
		{row: 6, col: 5},
	})
	p := &stubPredictor{class: LabelToClass("K")}

	got, err := LoadSFENWith(img, p)
	if err != nil {
		t.Fatalf("LoadSFENWith: %v", err)
	}
	if want := "9/9/3k5/9/9/9/5K3/9/9"; got != want {
		t.Errorf("LoadSFENWith = %q, want %q", got, want)
	}
	if p.calls != 2 {
		t.Errorf("推論器の呼び出し回数 = %d, want 2", p.calls)
	}
}

// 成駒は後手でも '+' を先頭に保ったまま小文字化されること
func TestLoadSFENWithPromoted(t *testing.T) {
	img := makeBoardImage([]testPiece{
		{row: 4, col: 4, gote: true},
		{row: 5, col: 4},
	})
	p := &stubPredictor{class: LabelToClass("+P")}

	got, err := LoadSFENWith(img, p)
	if err != nil {
		t.Fatalf("LoadSFENWith: %v", err)
	}
	if want := "9/9/9/9/4+p4/4+P4/9/9/9"; got != want {
		t.Errorf("LoadSFENWith = %q, want %q", got, want)
	}
}

func TestLoadSFENWithBoardNotFound(t *testing.T) {
	// 横方向グラデーション: マスごとの背景色がばらつくので ValidateBoard が低い
	img := image.NewGray(image.Rect(0, 0, 20, 20))
	for y := 0; y < 20; y++ {
		for x := 0; x < 20; x++ {
			img.SetGray(x, y, color.Gray{Y: uint8(x * 13)})
		}
	}

	_, err := LoadSFENWith(img, &stubPredictor{})
	if !errors.Is(err, ErrBoardNotFound) {
		t.Fatalf("err = %v, want ErrBoardNotFound", err)
	}
}

func TestLoadSFENWithInvalidArgs(t *testing.T) {
	if _, err := LoadSFENWith(nil, &stubPredictor{}); err == nil {
		t.Error("画像 nil でエラーにならない")
	}
	if _, err := LoadSFENWith(makeBoardImage(nil), nil); !errors.Is(err, ErrNoPredictor) {
		t.Error("推論器 nil で ErrNoPredictor にならない")
	}
}

// SetPredictor で差し替えた推論器が LoadSFEN から使われること
func TestLoadSFENUsesSetPredictor(t *testing.T) {
	p := &stubPredictor{class: LabelToClass("K")}
	SetPredictor(p)
	defer SetPredictor(nil)

	got, err := LoadSFEN(makeBoardImage([]testPiece{{row: 4, col: 4}}))
	if err != nil {
		t.Fatalf("LoadSFEN: %v", err)
	}
	if want := "9/9/9/9/4K4/9/9/9/9"; got != want {
		t.Errorf("LoadSFEN = %q, want %q", got, want)
	}
}

// 推論器が見つからないディレクトリでは ErrNoPredictor
func TestLoadPredictorMissing(t *testing.T) {
	if _, err := LoadPredictor(t.TempDir()); !errors.Is(err, ErrNoPredictor) {
		t.Fatalf("err = %v, want ErrNoPredictor", err)
	}
}
