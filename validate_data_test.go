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

	n := 0
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
		if want < minBoardConfidence {
			t.Errorf("%s: 正解座標が棄却された conf=%.2f (< %.2f)", e.ID, want, minBoardConfidence)
		}

		// 位置ずれ（周期は正しい）と周期ずれ。いずれも正解より低くなるべき
		bad := []struct {
			name string
			br   *BoardRegion
		}{
			{"横半マスずれ", BoardRegionFromRect(x1+cw/2, y1, x2+cw/2, y2)},
			{"縦半マスずれ", BoardRegionFromRect(x1, y1+ch/2, x2, y2+ch/2)},
			{"上下左右半マス縮小", BoardRegionFromRect(x1+cw/2, y1+ch/2, x2-cw/2, y2-ch/2)},
		}
		for _, b := range bad {
			if got := ValidateBoard(img, b.br); got >= want {
				t.Errorf("%s: %s のスコア %.2f が正解 %.2f 以上", e.ID, b.name, got, want)
			}
		}

		// 半マスずれは「盤ではない」と言い切れる水準まで落ちるべき
		if got := ValidateBoard(img, bad[0].br); got >= minBoardConfidence {
			t.Errorf("%s: 横半マスずれが棄却されない conf=%.2f", e.ID, got)
		}

		t.Logf("%s: 正解=%.2f 横半マス=%.2f 縦半マス=%.2f 縮小=%.2f",
			e.ID, want,
			ValidateBoard(img, bad[0].br), ValidateBoard(img, bad[1].br), ValidateBoard(img, bad[2].br))
	}
	if n == 0 {
		t.Skip("画像が無いのでスキップ")
	}
}
