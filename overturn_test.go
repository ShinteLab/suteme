package suteme

import "testing"

// 分類器が空と言ったマスを、駒サンプルとの照合で駒に戻す条件（`overturnEmpty`）。
//
// 合成盤の後手の駒 1 枚を使い、空判定の境目（emptyMax）を動かして
// 「分類器は空と言う」状態を作る。
func TestOverturnEmpty(t *testing.T) {
	img := makeBoardImage([]testPiece{{row: 2, col: 3, gote: true}})
	br := BoardRegionFromRect(0, 0, testCell*9, testCell*9)
	cell := br.ExtractCell(img, 2, 3)
	_, _, cover := classifyCellCover(cell, testBoardV, emptyCoverMax)
	if cover <= 0 {
		t.Fatalf("合成の駒の被覆率が測れない: %v", cover)
	}

	// 学習データは後手の駒を 180 度回して先手向きに揃えてある
	match := NewKNN([]TrainingSample{{Input: CellToInput(Rotate180(cell)), Label: LabelToClass("P")}})
	far := make([]float64, inputSize)
	for i := range far {
		far[i] = float64(i%7) - 3
	}
	farKNN := NewKNN([]TrainingSample{{Input: far, Label: LabelToClass("P")}})

	cases := []struct {
		name       string
		emptyMax   float64
		m          Predictor
		want       CellCategory
		overturned bool
	}{
		// 被覆率が境目のすぐ下（半分以上）で、駒サンプルと一致する → 駒に戻す。向きは回転照合
		{"境目のすぐ下・一致", cover * 1.2, match, CellPieceDown, true},
		// 境目の半分に届かない（真っさらな空マスに近い）→ 照合しない
		{"境目の半分未満", cover * 2.5, match, CellEmpty, false},
		// 駒サンプルが遠い → 空のまま
		{"駒サンプルが遠い", cover * 1.2, farKNN, CellEmpty, false},
		// k-NN でない推論器（距離の尺度が無い）→ 今までどおり覆さない
		{"k-NN でない", cover * 1.2, &stubPredictor{class: 0}, CellEmpty, false},
		{"推論器なし", cover * 1.2, nil, CellEmpty, false},
	}
	for _, c := range cases {
		got := classifyCellInfo(cell, testBoardV, c.emptyMax, c.m)
		if got.cat != c.want || got.overturned != c.overturned {
			t.Errorf("%s: cat=%v overturned=%v（期待 %v / %v。被覆率 %.3f・境目 %.3f）",
				c.name, got.cat, got.overturned, c.want, c.overturned, cover, c.emptyMax)
		}
		if got.overturned && !got.byMatch {
			t.Errorf("%s: 駒に戻したのに向きを照合で決めた印が無い", c.name)
		}
	}
}

// 被覆率を測れなかったマス（`coverUnmeasured`）も照合に回すこと。
// ゲーム画面の直前の手の色付けのように、マス全体が盤の地色と違うと
// 全列がグリッド線として落ちてマスクが作れない。
func TestOverturnEmptyUnmeasured(t *testing.T) {
	img := makeBoardImage([]testPiece{{row: 4, col: 4}})
	br := BoardRegionFromRect(0, 0, testCell*9, testCell*9)
	cell := br.ExtractCell(img, 4, 4)
	kn := NewKNN([]TrainingSample{{Input: CellToInput(cell), Label: LabelToClass("P")}})

	res := overturnEmpty(cell, cellClass{cat: CellEmpty, cover: coverUnmeasured}, emptyCoverMax, kn)
	if res.cat != CellPieceUp || !res.overturned {
		t.Errorf("被覆率を測れないマスを照合に回していない: cat=%v overturned=%v", res.cat, res.overturned)
	}
	res = overturnEmpty(cell, cellClass{cat: CellEmpty, cover: 0}, emptyCoverMax, kn)
	if res.cat != CellEmpty {
		t.Errorf("被覆率 0 のマスを照合に回した: cat=%v", res.cat)
	}
}
