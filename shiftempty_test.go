package suteme

import (
	"image"
	"testing"
)

// 窓を ±1px ずらして空サンプルと照合し直す条件（`emptyByShift`）。
//
// 合成盤の空マスを 1px ずらした窓を空サンプルとして持たせ、
// 「分類器が駒と言った」状態（cellClass）を与えて確かめる。
func TestEmptyByShift(t *testing.T) {
	img := makeBoardImage(nil)
	br := BoardRegionFromRect(0, 0, testCell*9, testCell*9)
	const row, col = 4, 4
	bo := &BoardOrient{emptyMax: emptyCoverMax, img: img, br: br}

	shifted := br.ExtractCellShifted(img, row, col, 1, 0)
	far := make([]float64, inputSize)
	for i := range far {
		far[i] = float64(i%5) - 2
	}
	// 少しだけ離しておく（距離 0 だと「駒サンプルのほうが近い」が作れない）
	emptyIn := CellToInputFull(shifted)
	for i := range emptyIn {
		emptyIn[i] += 0.01
	}
	kn := NewKNN([]TrainingSample{
		{Input: emptyIn, Label: ClassEmpty},
		{Input: far, Label: LabelToClass("P")},
	})
	piece := func(cover, pieceDist float64) cellClass {
		return cellClass{cat: CellPieceUp, byMatch: true, cover: cover, pieceDist: pieceDist}
	}
	farPiece := float64(emptyDistNorm) // 駒サンプルは遠い

	cases := []struct {
		name  string
		c     cellClass
		m     Predictor
		empty bool
	}{
		{"ずらした窓が空サンプルと一致", piece(emptyCoverMax*1.2, farPiece), kn, true},
		// 被覆率が境目の 2 倍以上（駒の大半）は照合しない
		{"被覆率が高い", piece(emptyCoverMax*shiftCoverMax, farPiece), kn, false},
		// 駒サンプルのほうが近ければ駒のまま（ずらさない窓の判定と同じ条件）
		{"駒サンプルのほうが近い", piece(emptyCoverMax*1.2, 1e-9), kn, false},
		{"k-NN でない", piece(emptyCoverMax*1.2, farPiece), &stubPredictor{class: 0}, false},
		// 照合で駒に戻したマスは空に戻さない（行ったり来たりさせない）
		{"照合で駒に戻したマス", cellClass{cat: CellPieceUp, cover: coverUnmeasured, overturned: true, pieceDist: farPiece}, kn, false},
	}
	for _, c := range cases {
		got := bo.emptyByShift(c.c, row, col, c.m)
		if (got.cat == CellEmpty) != c.empty || got.shifted != c.empty {
			t.Errorf("%s: cat=%v shifted=%v（期待 空=%v）", c.name, got.cat, got.shifted, c.empty)
		}
	}
}

// ExtractCellShifted は ExtractCell と同じ規則で切り出すこと
// （ずらさなければ同じ画像、ずらした窓は盤の外枠で切られる）。
func TestExtractCellShifted(t *testing.T) {
	img := makeBoardImage([]testPiece{{row: 0, col: 0}})
	br := BoardRegionFromRect(0, 0, testCell*9, testCell*9)
	a := br.ExtractCell(img, 3, 3)
	b := br.ExtractCellShifted(img, 3, 3, 0, 0)
	if a.Bounds() != b.Bounds() {
		t.Fatalf("ずらさない窓の大きさが違う: %v / %v", a.Bounds(), b.Bounds())
	}
	for y := 0; y < a.Bounds().Dy(); y++ {
		for x := 0; x < a.Bounds().Dx(); x++ {
			if a.At(x, y) != b.At(x, y) {
				t.Fatalf("ずらさない窓の画素が違う (%d,%d)", x, y)
			}
		}
	}
	// 左上の角のマスを左上へずらすと、盤の外枠（= 画像の端）で切られて小さくなる
	c := br.ExtractCell(img, 0, 0)
	d := br.ExtractCellShifted(img, 0, 0, -1, -1)
	if d.Bounds().Dx() != c.Bounds().Dx()-1 || d.Bounds().Dy() != c.Bounds().Dy()-1 {
		t.Errorf("外枠で切られていない: %v（元 %v）", d.Bounds(), c.Bounds())
	}
	if !d.Bounds().Eq(image.Rect(0, 0, d.Bounds().Dx(), d.Bounds().Dy())) {
		t.Errorf("原点が 0 でない: %v", d.Bounds())
	}
}
