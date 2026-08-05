package suteme

import (
	"encoding/json"
	"image"
	_ "image/png"
	"os"
	"testing"
)

// 保存済み局面（data/）の手動指定座標を正解として ValidateBoard の
// 弁別力を測る。正しいマス割りと、ずらした／周期を変えたマス割りを
// 比べ、**正解が必ず最高スコアになる**ことを確認する。
//
// これが崩れると detectBoardRegion のフォールバックも
// minBoardConfidence による棄却も効かなくなる（旧実装の均一性のみでは
// どの候補も 1.00 を返していた）。
//
// data/ は .gitignore 対象なので、無ければスキップする。
func TestValidateBoardDiscriminates(t *testing.T) {
	dir := findDataDir()
	if dir == "" {
		t.Skip("data/history.json が無いのでスキップ")
	}
	f, err := os.Open(dir + "/history.json")
	if err != nil {
		t.Skip(err)
	}
	var h historyFile
	err = json.NewDecoder(f).Decode(&h)
	f.Close()
	if err != nil {
		t.Fatalf("history.json: %v", err)
	}

	n, clear := 0, 0
	for _, e := range h.Entries {
		imgF, err := os.Open(dir + "/" + e.ID + ".png")
		if err != nil {
			continue
		}
		img, _, err := image.Decode(imgF)
		imgF.Close()
		if err != nil {
			continue
		}
		n++

		x1, y1, x2, y2 := e.Bounds.X1, e.Bounds.Y1, e.Bounds.X2, e.Bounds.Y2
		cw, ch := (x2-x1)/9, (y2-y1)/9

		want := ValidateBoard(img, BoardRegionFromRect(x1, y1, x2, y2))
		if want >= minBoardConfidence {
			clear++
		}

		// 位置ずれ（周期は正しい）と周期ずれ。いずれも正解を上回ってはいけない
		bad := []struct {
			name string
			br   *BoardRegion
		}{
			{"横半マスずれ", BoardRegionFromRect(x1+cw/2, y1, x2+cw/2, y2)},
			{"縦半マスずれ", BoardRegionFromRect(x1, y1+ch/2, x2, y2+ch/2)},
			{"上下左右半マス縮小", BoardRegionFromRect(x1+cw/2, y1+ch/2, x2-cw/2, y2-ch/2)},
		}
		got := make([]float64, len(bad))
		for i, b := range bad {
			got[i] = ValidateBoard(img, b.br)
			if got[i] > want {
				t.Errorf("%s: %s のスコア %.2f が正解 %.2f を上回った", e.ID, b.name, got[i], want)
			}
		}

		// 正解が採用される画像では、半マスずれは棄却されなければならない。
		// 正解自体が閾値に届かない画像（後述）では比較する意味が無い
		if want >= minBoardConfidence && got[0] >= minBoardConfidence {
			t.Errorf("%s: 横半マスずれが棄却されない conf=%.2f", e.ID, got[0])
		}

		t.Logf("%s: 正解=%.2f 横半マス=%.2f 縦半マス=%.2f 縮小=%.2f",
			e.ID, want, got[0], got[1], got[2])
	}
	if n == 0 {
		t.Skip("画像が無いのでスキップ")
	}

	// **全件が閾値を超えることは要求しない。** 実物の盤を撮った画像では
	// 片方の軸の格子線が投影にほとんど出ず（木目・照明・駒の重なり）、
	// 正しい領域でも 0 近辺までしか上がらないことがある。
	// 手動指定はこの値でゲートしていないので実害は表示だけ。
	// ただし大半で効いていないと自動検出のフォールバックが働かないので、
	// 8 割は超えることを要求する。
	t.Logf("正解座標が %s を超えたのは %d/%d", "minBoardConfidence", clear, n)
	if clear*10 < n*8 {
		t.Errorf("正解座標が採用されたのが %d/%d しかない", clear, n)
	}
}
