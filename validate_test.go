package suteme

import (
	"image"
	"image/color"
	"testing"
)

// makeStripeImage は「9本の縞模様だけ」の画像を作る。
// 縞の境目は盤の格子線と同じ周期で並ぶが、直交する方向には何も無い。
//
// 直交する方向に**ごく薄い線**（4 階調）を置いてあるのは、実物の負例
// （角丸のカードが背景に載ったスクリーンショット）でカードの縁がそう見えるため。
// **これが無いと投影が完全に 0 になり、`axisAlignment` の `on+off == 0` の
// ガードで落ちてしまう**＝ここで塞ぎたい「off が 0 なので比が +1 に張り付く」
// 経路を通らない。
//
// 縞の階調を近づけてあるのも実物に合わせたもの（色は派手でも輝度は近い）。
// 散らすと `cellUniformity` のほうで落ちてしまい、やはり経路が変わる。
func makeStripeImage(w, h int, horizontal bool) *image.Gray {
	img := image.NewGray(image.Rect(0, 0, w, h))
	shades := []uint8{120, 140, 150, 160, 145, 135, 155, 125, 165}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := x * 9 / w
			if horizontal {
				i = y * 9 / h
			}
			v := int(shades[i])
			// 直交方向の「縁」。盤の格子線と同じ位置に来るが 4 階調しかない
			if horizontal {
				if x%(w/9) < 2 {
					v += 4
				}
			} else if y%(h/9) < 2 {
				v += 4
			}
			img.SetGray(x, y, color.Gray{Y: uint8(v)})
		}
	}
	return img
}

// 縞模様だけの画像を盤面として採用してはいけない。
//
// **`axisAlignment` の比 (on-off)/(on+off) では落ちない。** 縞に直交する軸は
// 線が無いに等しいので on も off も 0 付近になり、比が +1 に張り付く。
// 実測（負例 `d112c258`、横縞）: 直交軸が on=0.8 / off=0.0 で整合度 +1.00、
// 均一性 0.78 と掛けて**信頼度 0.78 で採用**されていた。
// 落とすのは絶対量の下限（`minLineStrength`）。
func TestValidateBoardRejectsStripes(t *testing.T) {
	for _, tc := range []struct {
		name       string
		horizontal bool
	}{
		{"横縞", true},
		{"縦縞", false},
	} {
		img := makeStripeImage(450, 495, tc.horizontal)
		b := img.Bounds()
		whole := BoardRegionFromRect(b.Min.X, b.Min.Y, b.Max.X, b.Max.Y)
		if got := ValidateBoard(img, whole); got >= minBoardConfidence {
			t.Errorf("%s: 画像全体を盤面として採用してしまう conf=%.2f", tc.name, got)
		}
		if br := DetectBoard(img); br != nil {
			if got := ValidateBoard(img, br); got >= minBoardConfidence {
				t.Errorf("%s: 検出結果を採用してしまう region=%v conf=%.2f", tc.name, br.Bounds, got)
			}
		}
	}
}

// TestLineProjectionsAxesMatchesBoth は「要る軸だけ計算する」版が
// 両方求めた場合と同じ値を返すことを確かめる。
// `SnapToGrid` はこの投影の上でサブピクセルの重心を取るので、
// わずかでも違えば切り出しが動く（＝学習データの版が上がる）。
func TestLineProjectionsAxesMatchesBoth(t *testing.T) {
	img := image.NewGray(image.Rect(0, 0, 120, 96))
	for y := 0; y < 96; y++ {
		for x := 0; x < 120; x++ {
			v := 180
			if x%13 == 0 || y%11 == 0 {
				v = 40
			}
			img.SetGray(x, y, color.Gray{Y: uint8(v)})
		}
	}
	blurred := BoxBlur(img, 2)
	wantRow, wantCol := lineProjections(blurred)

	gotRow, _ := lineProjectionsAxes(blurred, true, false)
	for i := range wantRow {
		if gotRow[i] != wantRow[i] {
			t.Fatalf("row[%d] = %v, want %v", i, gotRow[i], wantRow[i])
		}
	}
	_, gotCol := lineProjectionsAxes(blurred, false, true)
	for i := range wantCol {
		if gotCol[i] != wantCol[i] {
			t.Fatalf("col[%d] = %v, want %v", i, gotCol[i], wantCol[i])
		}
	}
}
