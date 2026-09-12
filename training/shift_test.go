package training

import (
	"image"
	_ "image/png"
	"os"
	"path/filepath"
	"runtime"
	"sync"
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
	// **`-short` で外せるようにする（`TestUnslipHoldout` と揃える）。**
	// 判定は持っているので既定では走らせるが、188局面 × 81マス の推論なので
	// 「いま書いたコードが通るか」を見たいだけのときは邪魔になる
	if testing.Short() {
		t.Skip("短縮モードではスキップ")
	}
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

	// **対象局面を外した k-NN（未知の盤の数字）は判定に使っていない。**
	// 下の合否は missAll の +0px と +1px しか見ないのに、局面ごとに
	// 6 万件の k-NN を組み直していたので 188 局面で **45 分**かかっていた。
	// 数字が要るときだけ作る（`requireSlow` と同じ SUTEME_SLOW）
	holdout := os.Getenv("SUTEME_SLOW") != ""

	// **全データの k-NN はループの外で 1 回だけ作る。**
	// 中身は局面によらず同じなのに、毎回 MergeSamples から組み直していた。
	// 1 回にすると map の反復順にも依らなくなる（同じ答えが安定して出る）
	var all []suteme.TrainingSample
	for _, s := range byEntry {
		all = append(all, s...)
	}
	knAll := suteme.NewKNN(MergeSamples(all))
	if knAll == nil {
		t.Skip("k-NN を構築できないのでスキップ")
	}

	// **判定が見るのは +0px と +1px だけ。** +2/+4px は「どこまで崩れるか」を
	// 知るための計測で、1 段ごとに 188局面 × 81マス の推論が増える
	shifts := []int{0, 1}
	if holdout {
		shifts = []int{0, 1, 2, 4}
	}

	// 1 局面ぶんの結果。局面どうしは独立なので並列に測る
	type res struct {
		cells    int
		all, out []int
	}
	var targets []HistoryEntry
	for _, e := range h.Entries {
		if e.BoardBounds == nil {
			continue
		}
		if _, ok := byEntry[e.ID]; !ok {
			continue
		}
		targets = append(targets, e)
	}
	results := make([]res, len(targets))
	var next int
	var mu sync.Mutex
	var wg sync.WaitGroup
	workers := runtime.GOMAXPROCS(0)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				mu.Lock()
				i := next
				next++
				mu.Unlock()
				if i >= len(targets) {
					return
				}
				e := targets[i]
				r := res{all: make([]int, len(shifts)), out: make([]int, len(shifts))}
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
				var knOut *suteme.KNN
				if holdout {
					var others []suteme.TrainingSample
					for id, s := range byEntry {
						if id != e.ID {
							others = append(others, s...)
						}
					}
					if knOut = suteme.NewKNN(MergeSamples(others)); knOut == nil {
						continue
					}
				}
				b := e.BoardBounds
				for si, d := range shifts {
					br := suteme.BoardRegionFromRect(b.X1+d, b.Y1+d, b.X2+d, b.Y2+d)
					bc := suteme.BoardColor(img, br)
					for rr := 0; rr < 9; rr++ {
						for c := 0; c < 9; c++ {
							cell := br.ExtractCell(img, rr, c)
							if cell == nil {
								continue
							}
							if si == 0 {
								r.cells++
							}
							if predictCell(cell, bc, knAll) != want[rr][c] {
								r.all[si]++
							}
							if holdout && predictCell(cell, bc, knOut) != want[rr][c] {
								r.out[si]++
							}
						}
					}
				}
				results[i] = r
			}
		}()
	}
	wg.Wait()

	missAll := make([]int, len(shifts))  // 全局面で学習（同じ中継を繰り返し読む想定）
	missHold := make([]int, len(shifts)) // 対象局面を学習から外す（未知の盤）
	cells := 0
	for _, r := range results {
		cells += r.cells
		for si := range shifts {
			missAll[si] += r.all[si]
			missHold[si] += r.out[si]
		}
	}

	if cells == 0 {
		t.Skip("評価できるマスが無いのでスキップ")
	}
	for si, d := range shifts {
		if holdout {
			t.Logf("+%dpx: 誤り 全データ %d/%d  局面ホールドアウト %d/%d",
				d, missAll[si], cells, missHold[si], cells)
			continue
		}
		t.Logf("+%dpx: 誤り 全データ %d/%d", d, missAll[si], cells)
	}
	if !holdout {
		t.Log("+2px/+4px と局面ホールドアウト（未知の盤）の数字は SUTEME_SLOW=1 で出る")
	}

	// 1px ずらしただけで崩れないこと。最近傍リサイズ・増強なしの頃は
	// 22 → 112 と 5 倍になっていた
	if missAll[0] > 0 && missAll[1] > missAll[0]*3 {
		t.Errorf("1px のずれで誤りが %d → %d に増えた（3倍以内であること）", missAll[0], missAll[1])
	}
}
