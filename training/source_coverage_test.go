package training

import (
	"fmt"
	"image"
	_ "image/png"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/ShinteLab/suteme"
)

// TestSourceCoverage は「次にどの盤の局面を集めるべきか」を出す（調査用）。
//
// **認識率は局面数より「出所ごとに何枚あるか」で決まる**（`TestSameSourceEffect`）。
// ところが集めた結果がその方針に沿っているかは、これまで測る手段が無かった。
// 出所ごとに枚数と成績を並べて、**薄くて読めていない出所**を上に出す。
//
//	SUTEME_COVERAGE=1 go test -run TestSourceCoverage -timeout 60m -v ./training
//
// 出所は画像サイズで代用する（同じ画面をキャプチャすれば一致する。
// `TestSameSourceEffect` と同じ割り方）。**同じサイズの別物・同じ出所の別サイズは
// 分けられない**ので、枚数は目安として読むこと。
//
// 2 通りで測る。**1 枚しかない出所ではこの 2 つは同じ値**になる。
//
//   - 局面ホールドアウト … その局面だけを学習から外す（同じ出所の他の枚は残る）
//   - 出所ホールドアウト … 同じサイズの局面を丸ごと外す＝**その盤を初めて見る条件**
func TestSourceCoverage(t *testing.T) {
	if os.Getenv("SUTEME_COVERAGE") == "" {
		t.Skip("SUTEME_COVERAGE が空なのでスキップ（調査用）")
	}
	if !chdirToData(t) {
		t.Fatal("data/history.json が見つかりません")
	}
	h := loadHistory()
	if h == nil || len(h.Entries) < 2 {
		t.Fatal("局面が足りません")
	}

	type board struct {
		id     string
		img    image.Image
		br     *suteme.BoardRegion
		want   *[9][9]string
		source string
	}
	byEntry := make(map[string][]suteme.TrainingSample, len(h.Entries))
	boards := make([]board, 0, len(h.Entries))
	for _, e := range h.Entries {
		if e.BoardBounds == nil {
			continue
		}
		s, err := samplesFromHistory(e)
		if err != nil {
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
		byEntry[e.ID] = MergeSamples(s)
		b := e.BoardBounds
		boards = append(boards, board{
			id:     e.ID,
			img:    img,
			br:     suteme.BoardRegionFromRect(b.X1, b.Y1, b.X2, b.Y2),
			want:   want,
			source: fmt.Sprintf("%dx%d", img.Bounds().Dx(), img.Bounds().Dy()),
		})
	}

	source := make(map[string]string, len(boards))
	count := make(map[string]int, len(boards))
	for _, b := range boards {
		source[b.id] = b.source
		count[b.source]++
	}

	// score は指定した局面を、除外条件つきの k-NN で読んだときの一致マス数
	score := func(b board, bySource bool) (ok, total int) {
		var train []suteme.TrainingSample
		for id, s := range byEntry {
			if id == b.id || (bySource && source[id] == b.source) {
				continue
			}
			train = append(train, s...)
		}
		kn := suteme.NewKNN(train)
		if kn == nil {
			return 0, 0
		}
		bc := suteme.BoardColor(b.img, b.br)
		bo := suteme.NewBoardOrient(b.img, b.br, bc)
		for r := 0; r < 9; r++ {
			for c := 0; c < 9; c++ {
				cell := b.br.ExtractCell(b.img, r, c)
				if cell == nil {
					continue
				}
				total++
				if predictCellBO(cell, bc, bo, kn) == b.want[r][c] {
					ok++
				}
			}
		}
		return ok, total
	}

	type stat struct {
		n                   int
		entryOK, entryAll   int
		sourceOK, sourceAll int
	}
	stats := make(map[string]*stat, len(count))
	for _, b := range boards {
		st := stats[b.source]
		if st == nil {
			st = &stat{}
			stats[b.source] = st
		}
		st.n++
		ok, all := score(b, false)
		st.entryOK, st.entryAll = st.entryOK+ok, st.entryAll+all
		// 1 枚しかない出所は、局面を外した時点で出所ごと消えている
		if count[b.source] > 1 {
			ok, all = score(b, true)
		}
		st.sourceOK, st.sourceAll = st.sourceOK+ok, st.sourceAll+all
	}

	pct := func(ok, all int) float64 {
		if all == 0 {
			return 0
		}
		return 100 * float64(ok) / float64(all)
	}
	keys := make([]string, 0, len(stats))
	for s := range stats {
		keys = append(keys, s)
	}
	// 薄くて読めていない出所を上に（＝次に集める価値が高い順）
	sort.Slice(keys, func(i, j int) bool {
		a, b := stats[keys[i]], stats[keys[j]]
		if a.n != b.n {
			return a.n < b.n
		}
		return pct(a.sourceOK, a.sourceAll) < pct(b.sourceOK, b.sourceAll)
	})

	t.Logf("局面 %d / 出所 %d", len(boards), len(stats))
	t.Logf("%-12s %4s %10s %10s", "出所", "枚数", "局面LOO", "出所LOO")
	for _, s := range keys {
		st := stats[s]
		t.Logf("%-12s %4d %9.1f%% %9.1f%%", s, st.n,
			pct(st.entryOK, st.entryAll), pct(st.sourceOK, st.sourceAll))
	}

	// 枚数ごとの平均。**「何枚あれば足りるか」を測るのはここ。**
	byCount := map[int]*stat{}
	for _, st := range stats {
		agg := byCount[st.n]
		if agg == nil {
			agg = &stat{}
			byCount[st.n] = agg
		}
		agg.n++
		agg.entryOK, agg.entryAll = agg.entryOK+st.entryOK, agg.entryAll+st.entryAll
		agg.sourceOK, agg.sourceAll = agg.sourceOK+st.sourceOK, agg.sourceAll+st.sourceAll
	}
	sizes := make([]int, 0, len(byCount))
	for n := range byCount {
		sizes = append(sizes, n)
	}
	sort.Ints(sizes)
	t.Logf("--- 枚数ごとの平均 ---")
	t.Logf("%4s %6s %10s %10s", "枚数", "出所数", "局面LOO", "出所LOO")
	for _, n := range sizes {
		agg := byCount[n]
		t.Logf("%4d %6d %9.1f%% %9.1f%%", n, agg.n,
			pct(agg.entryOK, agg.entryAll), pct(agg.sourceOK, agg.sourceAll))
	}
}
