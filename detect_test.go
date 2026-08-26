package suteme

import (
	"image"
	"image/color"
	"math"
	"testing"
)

const (
	slipCellW  = 40
	slipCellH  = 44 // ≒ boardAspect
	slipBoardX = 120
	slipBoardY = 100
)

// makeSlipScene は「盤の1マス外に、格子線と紛れる罫線が 1 本ある」画像を作る。
//
// この形が **窓が1マス滑る**（周期は正しいまま外枠が内部の格子線に乗り、
// 外側の1本だけが盤ではない線に当たる）誤検出の元になる。盤の外周には
// 余白（背景）を置く: 盤が画像いっぱいだと滑った窓が画像からはみ出して
// `unslipRegion` の担当になってしまう。
func makeSlipScene() *image.Gray {
	w := slipBoardX*2 + slipCellW*9
	h := slipBoardY*2 + slipCellH*9
	img := image.NewGray(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetGray(x, y, color.Gray{Y: 150}) // 背景
		}
	}
	for y := slipBoardY; y < slipBoardY+slipCellH*9; y++ {
		for x := slipBoardX; x < slipBoardX+slipCellW*9; x++ {
			img.SetGray(x, y, color.Gray{Y: testBoardV})
		}
	}
	for i := 0; i <= 9; i++ {
		y := slipBoardY + i*slipCellH
		for x := slipBoardX; x < slipBoardX+slipCellW*9; x++ {
			img.SetGray(x, y, color.Gray{Y: testLineV})
		}
		x := slipBoardX + i*slipCellW
		for y := slipBoardY; y < slipBoardY+slipCellH*9; y++ {
			img.SetGray(x, y, color.Gray{Y: testLineV})
		}
	}
	// 盤の 1マス上にある、盤とは無関係な罫線（UI パネルの縁の類）。
	// **短いが濃い**のが要点で、`detectBoardIn` の合計投影では格子線に勝てても
	// `lineProjections` の中央値には出ない＝滑った窓の外側の1本は
	// 「線の無い所」になる
	for i := 0; i < 3; i++ {
		y := slipBoardY - slipCellH + i
		for x := slipBoardX; x < slipBoardX+slipCellW*4; x++ {
			img.SetGray(x, y, color.Gray{Y: 0})
		}
	}
	return img
}

// 1マス滑った窓を外枠へ寄せ直せること。
//
// **`gridAlignment` では見分けられない**（全マスシフトに不変）ので、
// `snapToOuterFrame` は「10本の境界線のうち最弱の線」で判定する。
// 滑った窓は外側の1本が線の無い所（または盤と無関係な短い罫線）に来る。
func TestSnapToOuterFrameFixesOneCellSlip(t *testing.T) {
	img := makeSlipScene()
	truth := image.Rect(slipBoardX, slipBoardY,
		slipBoardX+slipCellW*9, slipBoardY+slipCellH*9)

	slipped := truth.Sub(image.Pt(0, slipCellH))
	got := snapToOuterFrame(img, BoardRegionFromRect(
		slipped.Min.X, slipped.Min.Y, slipped.Max.X, slipped.Max.Y))

	dy := float64(got.Bounds.Min.Y-truth.Min.Y) / slipCellH
	dx := float64(got.Bounds.Min.X-truth.Min.X) / slipCellW
	t.Logf("滑り %v -> %v (dx=%+.2f dy=%+.2f)", slipped, got.Bounds, dx, dy)
	if dy < -0.5 || dy > 0.5 || dx < -0.5 || dx > 0.5 {
		t.Errorf("1マス滑った窓が戻らない: %v（正解 %v）", got.Bounds, truth)
	}
}

// 合っている窓は動かさないこと。**乗り換えには snapMargin の勝ち幅を要求する**
// （小差で動かすと、もともと合っていた検出を壊す）
func TestSnapToOuterFrameKeepsCorrectRegion(t *testing.T) {
	img := makeSlipScene()
	truth := image.Rect(slipBoardX, slipBoardY,
		slipBoardX+slipCellW*9, slipBoardY+slipCellH*9)
	br := BoardRegionFromRect(truth.Min.X, truth.Min.Y, truth.Max.X, truth.Max.Y)
	if got := snapToOuterFrame(img, br); got.Bounds != truth {
		t.Errorf("合っている窓が動いた: %v -> %v", truth, got.Bounds)
	}
}

// 最弱の線は「10本すべてが線に乗っているか」を測る。1本でも外れれば落ちる
func TestWeakestLineRatio(t *testing.T) {
	// 40px 間隔で 10 本の線がある投影
	proj := make([]float64, 500)
	for i := 0; i <= 9; i++ {
		proj[i*40] = 100
	}
	if v := weakestLineRatio(proj, 0, 40); v < 0.9 {
		t.Errorf("10本すべて乗っているのに %.2f", v)
	}
	// 1本だけ欠けている（＝滑った窓の外側の1本）
	proj[0] = 0
	if v := weakestLineRatio(proj, 0, 40); v > 0.1 {
		t.Errorf("1本欠けているのに %.2f", v)
	}
}

// 信頼度が同点で並んだ候補は「別々の ROI から同じ窓が出た数」（票）で割る。
//
// **`gridAlignment` は全マスシフトに対して不変**なので、正しい窓と1マス滑った
// 窓がどちらも 1.00 で並ぶ。以前は先に評価された候補がそのまま残っていた。
// 実測（`26a22911`）: 正 (573,196)-(1146,830) 票2 に対し
// 誤 (578,267)-(1139,890) 票1 で、先に来る誤ったほうが採用されていた。
func TestCandidateSetBreaksTieByVotes(t *testing.T) {
	slipped := BoardRegionFromRect(578, 267, 1139, 890)
	correct := BoardRegionFromRect(573, 196, 1146, 830)

	s := &candidateSet{}
	s.add(slipped, 1.0) // 先に来る誤った窓
	s.add(correct, 1.0)
	s.add(correct, 1.0) // 別の ROI からも同じ窓
	if got := s.best(); got.br.Bounds != correct.Bounds {
		t.Errorf("同点で票の少ない窓が残った: %v（期待 %v）", got.br.Bounds, correct.Bounds)
	}

	// 信頼度が上なら票は関係ない。**票は同点のときだけの材料**
	s = &candidateSet{}
	s.add(slipped, 1.0)
	s.add(slipped, 1.0)
	s.add(correct, 0.9)
	if got := s.best(); got.br.Bounds != slipped.Bounds {
		t.Errorf("信頼度の低い窓が票で勝った: %v", got.br.Bounds)
	}

	// 数px の違いは同じ窓として畳み、位置は信頼度が高いほうを残す。
	// **先勝ちにすると数px の詰めを捨てることになる**（実測で 3 局面が
	// 0.00マスから 0.91〜1.27マスへ壊れた）
	s = &candidateSet{}
	s.add(BoardRegionFromRect(573, 196, 1146, 830), 0.95)
	s.add(BoardRegionFromRect(575, 198, 1148, 832), 1.0)
	if len(s.cands) != 1 {
		t.Fatalf("ほぼ同じ窓が %d 個に分かれた", len(s.cands))
	}
	if got := s.best(); got.br.Bounds.Min.X != 575 || got.votes != 2 {
		t.Errorf("畳んだ結果が %v 票=%d（信頼度の高い位置・票2 のはず）", got.br.Bounds, got.votes)
	}
}

// TestIsOneCellSlipRejectsWrongSize は「1マス滑り」の見分けを縛る。
//
// **この判定は `TestDetectBoardMatchesManual` の見逃し口**なので、
// 緩めると壊れた検出まで黙って通ることになる。滑りは
// 「大きさは合っているのに位置だけ 1マスぶん動いた」ものに限る。
func TestIsOneCellSlipRejectsWrongSize(t *testing.T) {
	cases := []struct {
		name           string
		dx, dy, dw, dh float64
		want           bool
	}{
		{"縦に1マス滑り", 0.00, -0.99, 0.00, -0.01, true},
		{"横に1マス滑り", 1.01, 0.02, -0.05, 0.03, true},
		{"滑りぎみ（0.64マス）", 0.17, -0.64, -0.35, -0.45, true},
		{"合っている", 0.03, 0.05, -0.08, -0.14, false},
		{"半分の周期（大きさが違う）", 0.10, 0.20, -4.50, -4.50, false},
		{"位置も大きさも大外し", 3.57, 1.20, 3.90, 2.10, false},
		{"2マス滑り（滑りとして見逃さない）", 0.00, -2.00, 0.00, 0.00, false},
		{"大きさだけ 1マス違う（滑りではない）", 0.00, 0.00, -1.00, 0.00, false},
	}
	for _, c := range cases {
		if got := isOneCellSlip(c.dx, c.dy, c.dw, c.dh); got != c.want {
			t.Errorf("%s: isOneCellSlip(%.2f,%.2f,%.2f,%.2f) = %v, want %v",
				c.name, c.dx, c.dy, c.dw, c.dh, got, c.want)
		}
	}
}

// TestOverflowCells ははみ出し量がマス単位で出ることを確かめる
func TestOverflowCells(t *testing.T) {
	img := image.Rect(0, 0, 664, 701)
	cw, ch := 71.7, 76.3

	if got := overflowCells(image.Rect(10, 10, 600, 600), img, cw, ch); got != 0 {
		t.Errorf("画像内なのに %.2f マスはみ出し扱い", got)
	}
	// 実測 9d2c92ae: 下端が 6px 外（SnapToGrid の当てはめ由来）
	if got := overflowCells(image.Rect(2, 2, 647, 707), img, cw, ch); math.Abs(got-6/ch) > 1e-9 {
		t.Errorf("下へ 6px = %.3f マス, got %.3f", 6/ch, got)
	}
	// 1マス滑って右へ出た場合は見逃さない
	if got := overflowCells(image.Rect(72, 0, 736, 700), img, cw, ch); got < 1 {
		t.Errorf("右へ 72px（1マス超）が %.2f マス扱い", got)
	}
	// 左・上へのはみ出しも数える
	if got := overflowCells(image.Rect(-80, -10, 600, 600), img, cw, ch); got < 1 {
		t.Errorf("左へ 80px（1マス超）が %.2f マス扱い", got)
	}
}
