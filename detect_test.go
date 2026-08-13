package suteme

import (
	"image"
	"image/color"
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
