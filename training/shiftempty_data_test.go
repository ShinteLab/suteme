package training

import (
	"image"
	"strings"
	"testing"

	"github.com/ShinteLab/suteme"
)

// 木目の強い実盤で、同じ対局を学習済みなら窓が 1px 違っても空マスを空と読むこと
// （`emptyByShift`）。履歴 `72af86de` の 9八 は、同じ対局の 3 枚を学習していても
// 窓の大きさが 1px 違うせいで空サンプルに届かず駒に化けていた（TODO.md）。
//
// 学習に使うのは同じ対局の 3 枚だけ（72af86de 自身は外す）。data/ に無ければスキップ。
func TestShiftEmptyWoodGrain(t *testing.T) {
	if !chdirToData(t) {
		t.Skip("data/history.json が無いのでスキップ")
	}
	const target = "72af86de"
	siblings := []string{"635b14144662d14f", "ffe304a0a725ffc5", "1d4c22bc17ded5c6"}
	var tg *evalTarget
	var train []suteme.TrainingSample
	for _, e := range loadHistory().Entries {
		if strings.HasPrefix(e.ID, target) {
			tg, _ = loadEvalTarget(e)
			continue
		}
		for _, id := range siblings {
			if e.ID == id {
				s, err := samplesFromHistory(e)
				if err != nil {
					t.Fatalf("%s: %v", id, err)
				}
				train = append(train, s...)
			}
		}
	}
	if tg == nil || len(train) == 0 {
		t.Skip("72af86de か同じ対局の局面が無いのでスキップ")
	}
	kn := suteme.NewKNN(MergeSamples(train))
	b := tg.rect
	for _, w := range []struct {
		name string
		r    image.Rectangle
	}{{"寄せた座標", tg.snap}, {"保存座標", image.Rect(b.X1, b.Y1, b.X2, b.Y2)}} {
		r, err := suteme.Recognize(tg.img, suteme.WithPredictor(kn), suteme.WithRect(w.r.Min.X, w.r.Min.Y, w.r.Max.X, w.r.Max.Y))
		if r == nil {
			t.Fatalf("%s: %v", w.name, err)
		}
		for _, sq := range []struct {
			name     string
			row, col int
			must     bool
		}{{"9七", 6, 0, false}, {"9八", 7, 0, true}} {
			cd := r.Debug.Cell(sq.row, sq.col)
			t.Logf("%s %s: 読み %q empty_by=%q cover=%.3f / 境目 %.3f", w.name, sq.name, cd.Piece, cd.EmptyBy, cd.Cover, r.Debug.EmptyCover)
			if sq.must && cd.Piece != "" {
				t.Errorf("%s: %s（空マス）を %q と読んだ", w.name, sq.name, cd.Piece)
			}
		}
	}
}
