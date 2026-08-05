package suteme

import (
	"errors"
	"strings"
	"testing"

	"github.com/ShinteLab/core/sfen"
)

// 空の盤面（駒が1枚も無い）は玉が両方いないので CheckKing に引っかかる。
// 既定ではエラーにせず、違反だけを結果に載せる。
func TestRecognizeLenientByDefault(t *testing.T) {
	img := makeBoardImage(nil)
	r, err := Recognize(img, WithPredictor(&stubPredictor{}))
	if err != nil {
		t.Fatalf("既定でエラーになった: %v", err)
	}
	if len(r.Violations) == 0 {
		t.Fatal("玉なしが違反として載っていない")
	}
	if got := strings.Join(r.Warnings(), " / "); !strings.Contains(got, "玉") {
		t.Errorf("警告に玉の話が無い: %q", got)
	}
}

// WithStrict / WithErrorOn を指定すると ErrInvalidBoard になる。
// エラーでも Result は返るので、どこがおかしいかは呼び出し側で出せる。
func TestRecognizeStrict(t *testing.T) {
	img := makeBoardImage(nil)
	r, err := Recognize(img, WithPredictor(&stubPredictor{}), WithStrict())
	if !errors.Is(err, ErrInvalidBoard) {
		t.Fatalf("err = %v, want ErrInvalidBoard", err)
	}
	if r == nil {
		t.Fatal("エラー時も Result を返すこと")
	}
	var be *BoardError
	if !errors.As(err, &be) || len(be.Violations) == 0 {
		t.Fatalf("BoardError を取り出せない: %v", err)
	}
	var v sfen.Violation
	if !errors.As(err, &v) || v.Check != sfen.CheckKing {
		t.Errorf("個々の Violation を取り出せない: %v", err)
	}
}

// エラーにするチェックを絞れば、他の違反があってもエラーにならない
func TestRecognizeErrorOnSelected(t *testing.T) {
	img := makeBoardImage(nil)
	r, err := Recognize(img, WithPredictor(&stubPredictor{}),
		WithErrorOn(sfen.CheckPieceCount))
	if err != nil {
		t.Fatalf("駒数の違反は無いのにエラー: %v", err)
	}
	if len(r.Filter(sfen.CheckKing)) == 0 {
		t.Error("エラーにしないチェックの違反も結果には載ること")
	}
}

// 実施しないチェックの違反は結果にも載らない
func TestRecognizeChecksOff(t *testing.T) {
	img := makeBoardImage(nil)
	r, err := Recognize(img, WithPredictor(&stubPredictor{}),
		WithChecks(sfen.CheckPieceCount))
	if err != nil {
		t.Fatalf("Recognize: %v", err)
	}
	if !r.OK() {
		t.Errorf("チェックを外したのに違反が載った: %v", r.Warnings())
	}
}

// 盤上に無い駒を先手/後手の駒台に寄せられる
func TestRecognizeHandTo(t *testing.T) {
	img := makeBoardImage(nil)

	none, err := Recognize(img, WithPredictor(&stubPredictor{}))
	if err != nil {
		t.Fatalf("Recognize: %v", err)
	}
	if len(none.BlackHand) != 0 || len(none.WhiteHand) != 0 {
		t.Errorf("既定では割り振らないこと: %v %v", none.BlackHand, none.WhiteHand)
	}
	if none.HandTotal["P"] != 18 {
		t.Errorf("駒台合計の歩 = %d, want 18", none.HandTotal["P"])
	}
	if !strings.HasSuffix(none.SFEN(), " b - 1") {
		t.Errorf("持ち駒を割り振らないなら '-': %q", none.SFEN())
	}

	black, err := Recognize(img, WithPredictor(&stubPredictor{}),
		WithHandTo(HandBlack), WithTurn(false), WithMoveNumber(7))
	if err != nil {
		t.Fatalf("Recognize: %v", err)
	}
	if black.BlackHand["P"] != 18 || len(black.WhiteHand) != 0 {
		t.Errorf("先手に寄せられていない: %v / %v", black.BlackHand, black.WhiteHand)
	}
	want := "9/9/9/9/9/9/9/9/9 w 2R2B4G4S4N4L18P 7"
	if got := black.SFEN(); got != want {
		t.Errorf("SFEN() = %q, want %q", got, want)
	}

	white, err := Recognize(img, WithPredictor(&stubPredictor{}), WithHandTo(HandWhite))
	if err != nil {
		t.Fatalf("Recognize: %v", err)
	}
	if white.WhiteHand["P"] != 18 || len(white.BlackHand) != 0 {
		t.Errorf("後手に寄せられていない: %v / %v", white.BlackHand, white.WhiteHand)
	}
}

// WithRegion で盤面領域を明示指定できる（検出を挟まない）
func TestRecognizeWithRegion(t *testing.T) {
	img := makeBoardImage([]testPiece{{row: 4, col: 4}})
	b := img.Bounds()
	r, err := Recognize(img,
		WithPredictor(&stubPredictor{}),
		WithRect(b.Min.X, b.Min.Y, b.Max.X, b.Max.Y))
	if err != nil {
		t.Fatalf("Recognize: %v", err)
	}
	if want := "9/9/9/9/4P4/9/9/9/9"; r.Board != want {
		t.Errorf("Board = %q, want %q", r.Board, want)
	}
}

// LoadSFEN もオプションを受け取り、エラー時も盤面文字列は返す
func TestLoadSFENWithOptions(t *testing.T) {
	img := makeBoardImage(nil)
	board, err := LoadSFEN(img, WithPredictor(&stubPredictor{}), WithStrict())
	if !errors.Is(err, ErrInvalidBoard) {
		t.Fatalf("err = %v, want ErrInvalidBoard", err)
	}
	if board != "9/9/9/9/9/9/9/9/9" {
		t.Errorf("エラーでも認識できた盤面は返すこと: %q", board)
	}
}
