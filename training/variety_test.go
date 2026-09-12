package training

import (
	"fmt"
	"image"
	_ "image/png"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/ShinteLab/suteme"
)

// TestVarietyVsThickness は「未知の盤に効くのはどちらか」を測る（調査用）。
//
//	SUTEME_VARIETY=1 go test -run TestVarietyVsThickness -timeout 90m -v ./training
//
// **「同じ見た目を厚くする」はその見た目にしか効かない**のではないか、という
// 当然の疑問に答えるためのもの。**学習に使う枚数を揃えて**、
//
//	厚み … 1 つの見た目から N 枚
//	多様 … N 通りの見た目から 1 枚ずつ
//
// のどちらが「その見た目を一度も見ていない盤」を読めるかを比べる。
// 対象の見た目は**必ず学習から丸ごと外す**ので、どちらの条件でも
// 「初めて見る盤」を読んでいることになる。
func TestVarietyVsThickness(t *testing.T) {
	requireSlow(t, "SUTEME_VARIETY")
	if !chdirToData(t) {
		t.Fatal("data/history.json が見つかりません")
	}
	h := loadHistory()

	type board struct {
		id      string
		look    string
		img     image.Image
		br      *suteme.BoardRegion
		want    *[9][9]string
		samples []suteme.TrainingSample
	}
	var boards []board
	for _, e := range h.Entries {
		if e.BoardBounds == nil || strings.TrimSpace(e.Look) == "" {
			continue // 見た目が判定済みのものだけで測る
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
		s, err := samplesFromHistory(e)
		if err != nil {
			continue
		}
		b := e.BoardBounds
		boards = append(boards, board{
			id: e.ID, look: e.Look, img: img,
			br:      suteme.BoardRegionFromRect(b.X1, b.Y1, b.X2, b.Y2),
			want:    want,
			samples: MergeSamples(s),
		})
	}
	byLook := map[string][]int{}
	for i, b := range boards {
		byLook[b.look] = append(byLook[b.look], i)
	}
	looks := make([]string, 0, len(byLook))
	for l := range byLook {
		looks = append(looks, l)
	}
	sort.Strings(looks)
	t.Logf("判定済み %d 局面 / %d 通りの見た目", len(boards), len(looks))

	read := func(target int, train []int) (ok, total int) {
		var samples []suteme.TrainingSample
		for _, i := range train {
			samples = append(samples, boards[i].samples...)
		}
		kn := suteme.NewKNN(samples)
		if kn == nil {
			return 0, 0
		}
		tb := &boards[target]
		bc := suteme.BoardColor(tb.img, tb.br)
		bo := suteme.NewBoardOrient(tb.img, tb.br, bc)
		for r := 0; r < 9; r++ {
			for c := 0; c < 9; c++ {
				cell := tb.br.ExtractCell(tb.img, r, c)
				if cell == nil {
					continue
				}
				total++
				if predictCellBO(cell, bc, bo, kn) == tb.want[r][c] {
					ok++
				}
			}
		}
		return ok, total
	}

	sizes := []int{1, 2, 4, 8}
	type agg struct{ ok, total int }
	variety := map[int]*agg{}
	thick := map[int]*agg{}
	for _, n := range sizes {
		variety[n], thick[n] = &agg{}, &agg{}
	}

	for ti := range boards {
		// **対象の見た目は丸ごと除外**（初めて見る盤という条件）
		var otherLooks []string
		for _, l := range looks {
			if l != boards[ti].look {
				otherLooks = append(otherLooks, l)
			}
		}
		if len(otherLooks) == 0 {
			continue
		}
		for _, n := range sizes {
			// 多様: n 通りの見た目から 1 枚ずつ（対象ごとに開始位置をずらす）
			if len(otherLooks) >= n {
				var train []int
				for k := 0; k < n; k++ {
					l := otherLooks[(ti+k)%len(otherLooks)]
					train = append(train, byLook[l][ti%len(byLook[l])])
				}
				ok, total := read(ti, train)
				variety[n].ok += ok
				variety[n].total += total
			}
			// 厚み: 1 つの見た目から n 枚（n 枚以上ある見た目だけ）
			var pick []int
			for k := 0; k < len(otherLooks); k++ {
				l := otherLooks[(ti+k)%len(otherLooks)]
				if len(byLook[l]) >= n {
					pick = byLook[l][:n]
					break
				}
			}
			if len(pick) == n {
				ok, total := read(ti, pick)
				thick[n].ok += ok
				thick[n].total += total
			}
		}
	}

	// **同じ予算（枚数）をどう割るか。** 種類を増やすのが良いとして、
	// 1 種類あたり何枚までなら無駄にならないかを見る
	const budget = 8
	splits := [][2]int{{8, 1}, {4, 2}, {2, 4}, {1, 8}} // {種類, 1種類あたりの枚数}
	split := map[[2]int]*agg{}
	for _, sp := range splits {
		split[sp] = &agg{}
	}
	for ti := range boards {
		var otherLooks []string
		for _, l := range looks {
			if l != boards[ti].look {
				otherLooks = append(otherLooks, l)
			}
		}
		for _, sp := range splits {
			nLooks, per := sp[0], sp[1]
			var train []int
			for k := 0; k < len(otherLooks) && len(train) < budget; k++ {
				l := otherLooks[(ti+k)%len(otherLooks)]
				if len(byLook[l]) < per {
					continue
				}
				train = append(train, byLook[l][:per]...)
				if len(train) >= nLooks*per {
					break
				}
			}
			if len(train) != budget {
				continue // その割り方ができるだけの見た目が無い
			}
			ok, total := read(ti, train)
			split[sp].ok += ok
			split[sp].total += total
		}
	}

	pct := func(a *agg) string {
		if a.total == 0 {
			return "—"
		}
		return fmt.Sprintf("%.1f%% (%d/%d)", 100*float64(a.ok)/float64(a.total), a.ok, a.total)
	}
	t.Logf("=== 未知の見た目を読む（対象の見た目は学習から丸ごと除外）===")
	t.Logf("%6s  %-22s %-22s", "枚数", "多様（N通り×1枚）", "厚み（1通り×N枚）")
	for _, n := range sizes {
		t.Logf("%6d  %-22s %-22s", n, pct(variety[n]), pct(thick[n]))
	}
	t.Logf("=== 同じ %d 枚をどう割るか（対象の見た目は丸ごと除外）===", budget)
	for _, sp := range splits {
		t.Logf("%2d 種類 × %d 枚: %s", sp[0], sp[1], pct(split[sp]))
	}
}
