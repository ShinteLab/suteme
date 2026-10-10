package training

import (
	"fmt"
	"image"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"

	"github.com/ShinteLab/suteme"
)

// TestShiftEmptyDump は「分類器が駒と言ったマス」を、窓を ±1px ずらして空サンプルと
// 照合したらどうなるかの材料を書き出す（調査用）。
//
// 空の一致判定（`emptyMatchMax`）は距離の絶対しきい値なので、同じ盤を学習済みでも
// **窓が 1px 違うだけで空マスどうしの距離が開いて届かない**ことがある
// （TODO.md「木目の強い実盤の空マスが駒に化ける」）。推論時に窓をずらして照合すれば
// 拾えるかを、真の空（いま 空→駒 のもの）と真の駒で比べるための CSV:
//
//	cover / emptymax  被覆率と、その盤の空判定の境目
//	class             いまの推論結果（14 = 空サンプルと一致して空にした）
//	dpiece            駒サンプルへの最近傍（そのまま / 180 度の近いほう）
//	d0                ずらさない窓での空サンプルへの最近傍（打ち切りは shiftEmptyCap）
//	dbest / bdx / bdy ±1px（縦横・斜めの 8 通り + そのまま）で最良の距離とずらし量
//	dcross            縦横 4 通り + そのままでの最良
//
// 距離はすべて emptyDistNorm で正規化。
//
//	SUTEME_SHIFTEMPTY=out.csv go test -run TestShiftEmptyDump -timeout 120m -v ./training
//
// `SUTEME_SHIFTEMPTY_MODE=look` で見た目ごと、`SUTEME_SHIFTEMPTY_RECT=raw` で保存された
// 座標そのもの（既定は格子線へ寄せた座標）。`SUTEME_SHIFTEMPTY_WATCH=72af86de:6:0,...` の
// マスは 9 通りの距離をログに出す。対象は確認済みの局面だけ。
func TestShiftEmptyDump(t *testing.T) {
	out := os.Getenv("SUTEME_SHIFTEMPTY")
	if out == "" {
		t.Skip("SUTEME_SHIFTEMPTY が空なのでスキップ（調査用）")
	}
	if !filepath.IsAbs(out) {
		if abs, err := filepath.Abs(out); err == nil {
			out = abs
		}
	}
	byLook := os.Getenv("SUTEME_SHIFTEMPTY_MODE") == "look"
	rawRect := os.Getenv("SUTEME_SHIFTEMPTY_RECT") == "raw"
	watch := map[string]bool{}
	for _, w := range strings.Split(os.Getenv("SUTEME_SHIFTEMPTY_WATCH"), ",") {
		if w != "" {
			watch[w] = true
		}
	}
	if !chdirToData(t) {
		t.Fatal("data/history.json が見つかりません")
	}
	h := loadHistory()

	keyOf := func(e HistoryEntry) string { return e.ID }
	mode := "board"
	if byLook {
		keyOf = func(e HistoryEntry) string { return strings.TrimSpace(e.Look) }
		mode = "look"
	}
	var targets []*evalTarget
	for _, e := range h.Entries {
		if !e.IsVerified() || keyOf(e) == "" {
			continue
		}
		tg, err := loadEvalTarget(e)
		if err != nil {
			continue
		}
		targets = append(targets, tg)
	}
	byKey := map[string][]suteme.TrainingSample{}
	for _, tg := range targets {
		s, err := samplesFromHistory(tg.entry)
		if err != nil {
			continue
		}
		k := keyOf(tg.entry)
		byKey[k] = append(byKey[k], s...)
	}
	for k, s := range byKey {
		byKey[k] = MergeSamples(s)
	}

	rows := make([]string, len(targets))
	logs := make([]string, len(targets))
	var wg sync.WaitGroup
	sem := make(chan struct{}, max(1, runtime.GOMAXPROCS(0)/4))
	for i, tg := range targets {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int, tg *evalTarget) {
			defer wg.Done()
			defer func() { <-sem }()
			k := keyOf(tg.entry)
			var train []suteme.TrainingSample
			for other, s := range byKey {
				if other != k {
					train = append(train, s...)
				}
			}
			kn := suteme.NewKNN(train)
			if kn == nil {
				return
			}
			var empties [][]float64
			for _, s := range train {
				if s.Label == suteme.ClassEmpty {
					empties = append(empties, s.Input)
				}
			}
			rect := tg.snap
			if rawRect {
				b := tg.rect
				rect = image.Rect(b.X1, b.Y1, b.X2, b.Y2)
			}
			rows[i], logs[i] = dumpShiftEmptyBoard(mode, k, tg, rect, kn, empties, watch)
		}(i, tg)
	}
	wg.Wait()

	var sb strings.Builder
	sb.WriteString("mode,key,id,row,col,want,got,cover,emptymax,class,dpiece,d0,dbest,bdx,bdy,dcross\n")
	for i, r := range rows {
		sb.WriteString(r)
		if logs[i] != "" {
			t.Log(logs[i])
		}
	}
	if err := os.WriteFile(out, []byte(sb.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Logf("%d 局面を書き出し: %s", len(targets), out)
}

// shiftEmptyCap は調査用の打ち切り（正規化後）。これより遠い距離は区別しない
// （空の一致判定は 0.10 なので、0.5 より遠いものは何をしても空にならない）。
const shiftEmptyCap = 0.5

func dumpShiftEmptyBoard(mode, key string, tg *evalTarget, rect image.Rectangle, kn *suteme.KNN,
	empties [][]float64, watch map[string]bool) (string, string) {
	r, _ := suteme.Recognize(tg.img, suteme.WithPredictor(kn), suteme.WithRect(rect.Min.X, rect.Min.Y, rect.Max.X, rect.Max.Y))
	if r == nil {
		return "", ""
	}
	br := suteme.BoardRegionFromRect(rect.Min.X, rect.Min.Y, rect.Max.X, rect.Max.Y)
	norm := float64(2 * suteme.InputSize)
	var sb, lg strings.Builder
	for _, cd := range r.Debug.Cells {
		// 分類器が駒と言ったマスだけ（照合で駒に戻したマスは分類器は空と言っている）
		if cd.Category == suteme.CellEmpty || cd.PieceBy != "" {
			continue
		}
		cell := br.ExtractCell(tg.img, cd.Row, cd.Col)
		if cell == nil {
			continue
		}
		want := tg.want[cd.Row][cd.Col]
		got := suteme.EmptyLabel
		if cd.Piece != "" {
			got = sfenToGridLabel(cd.Piece)
		}
		dpiece := math.Min(kn.PieceDistance(cell), kn.PieceDistance(suteme.Rotate180(cell))) / norm

		var ds [3][3]float64
		d0, dbest, dcross := math.Inf(1), math.Inf(1), math.Inf(1)
		bdx, bdy := 0, 0
		for dy := -1; dy <= 1; dy++ {
			for dx := -1; dx <= 1; dx++ {
				c := br.ExtractCellShifted(tg.img, cd.Row, cd.Col, dx, dy)
				d := math.Inf(1)
				if c != nil {
					d = nearestEmptyUncapped(c, empties, shiftEmptyCap*norm) / norm
				}
				ds[dy+1][dx+1] = d
				if dx == 0 && dy == 0 {
					d0 = d
				}
				if d < dbest {
					dbest, bdx, bdy = d, dx, dy
				}
				if (dx == 0 || dy == 0) && d < dcross {
					dcross = d
				}
			}
		}
		fmt.Fprintf(&sb, "%s,%s,%s,%d,%d,%s,%s,%.4f,%.4f,%d,%.4f,%.4f,%.4f,%d,%d,%.4f\n",
			mode, csvSafe(key), tg.entry.ID, cd.Row, cd.Col, want, got, cd.Cover, r.Debug.EmptyCover,
			cd.Class, dpiece, d0, dbest, bdx, bdy, dcross)
		if watch[fmt.Sprintf("%s:%d:%d", tg.entry.ID[:8], cd.Row, cd.Col)] {
			fmt.Fprintf(&lg, "%s r%d c%d want=%s got=%s cover=%.3f/%.3f dpiece=%.3f 空サンプルへの距離（行 dy=-1..1、列 dx=-1..1）:\n",
				tg.entry.ID[:8], cd.Row, cd.Col, want, got, cd.Cover, r.Debug.EmptyCover, dpiece)
			for _, row := range ds {
				fmt.Fprintf(&lg, "    %.3f %.3f %.3f\n", row[0], row[1], row[2])
			}
		}
	}
	return sb.String(), lg.String()
}

// nearestEmptyUncapped はマス全体（そのまま / 180 度）での空サンプルへの最近傍の 2 乗距離
// （`KNN.Predict` の空の照合と同じ測り方。limit で打ち切り、届かなければ +Inf）。
func nearestEmptyUncapped(cell image.Image, empties [][]float64, limit float64) float64 {
	full := [2][]float64{suteme.CellToInputFull(cell), suteme.CellToInputFull(suteme.Rotate180(cell))}
	best := limit
	found := false
	for _, e := range empties {
		for _, in := range full {
			if d := dist2Limit(in, e, best); d < best {
				best, found = d, true
			}
		}
	}
	if !found {
		return math.Inf(1)
	}
	return best
}
