package training

import (
	"image"
	_ "image/png"
	"os"
	"path/filepath"
	"testing"

	"github.com/ShinteLab/suteme"
)

// 盤面座標を数px ずらしたときに認識がどれだけ崩れるかを測る。
//
// **これが「手動座標ちょうど」でしか測れていなかったのが長らく見えていなかった。**
// DetectBoard は手動座標に対して数px ずれるので、推論時のクロップは学習時と
// 必ず食い違う。手動座標での 22/891 は k-NN がそのクロップを丸暗記した結果で、
// 1px ずらすと 112/891 まで崩れていた（最近傍リサイズ・増強なしの頃）。
//
// data/ は .gitignore 対象なので、無ければスキップする。
// `go test -run TestShiftRobustness -v ./training` で数字を見る。
func TestShiftRobustness(t *testing.T) {
	if !chdirToData(t) {
		t.Skip("data/history.json が無いのでスキップ")
	}
	h := loadHistory()
	if h == nil || len(h.Entries) < 2 {
		t.Skip("局面が足りないのでスキップ")
	}

	byEntry := map[string][]suteme.TrainingSample{}
	for _, e := range h.Entries {
		s, err := samplesFromHistory(e)
		if err != nil {
			continue
		}
		byEntry[e.ID] = s
	}

	shifts := []int{0, 1, 2, 4}
	missAll := make([]int, len(shifts))  // 全局面で学習（同じ中継を繰り返し読む想定）
	missHold := make([]int, len(shifts)) // 対象局面を学習から外す（未知の盤）
	cells := 0

	for _, e := range h.Entries {
		if e.BoardBounds == nil {
			continue
		}
		if _, ok := byEntry[e.ID]; !ok {
			continue
		}
		f, err := os.Open(filepath.Join(dataDir, e.ID+".png"))
		if err != nil {
			continue
		}
		img, _, err := image.Decode(f)
		f.Close()
		if err != nil {
			continue
		}
		want, err := wantGrid(e.SFEN)
		if err != nil {
			continue
		}

		var all, others []suteme.TrainingSample
		for id, s := range byEntry {
			all = append(all, s...)
			if id != e.ID {
				others = append(others, s...)
			}
		}
		knAll := suteme.NewKNN(MergeSamples(all))
		knOut := suteme.NewKNN(MergeSamples(others))
		if knAll == nil || knOut == nil {
			continue
		}

		b := e.BoardBounds
		for si, d := range shifts {
			br := suteme.BoardRegionFromRect(b.X1+d, b.Y1+d, b.X2+d, b.Y2+d)
			bc := suteme.BoardColor(img, br)
			for r := 0; r < 9; r++ {
				for c := 0; c < 9; c++ {
					cell := br.ExtractCell(img, r, c)
					if cell == nil {
						continue
					}
					if si == 0 {
						cells++
					}
					if predictCell(cell, bc, knAll) != want[r][c] {
						missAll[si]++
					}
					if predictCell(cell, bc, knOut) != want[r][c] {
						missHold[si]++
					}
				}
			}
		}
	}

	if cells == 0 {
		t.Skip("評価できるマスが無いのでスキップ")
	}
	for si, d := range shifts {
		t.Logf("+%dpx: 誤り 全データ %d/%d  局面ホールドアウト %d/%d",
			d, missAll[si], cells, missHold[si], cells)
	}

	// 1px ずらしただけで崩れないこと。最近傍リサイズ・増強なしの頃は
	// 22 → 112 と 5 倍になっていた
	if missAll[0] > 0 && missAll[1] > missAll[0]*3 {
		t.Errorf("1px のずれで誤りが %d → %d に増えた（3倍以内であること）", missAll[0], missAll[1])
	}
}
