package training

import (
	"image"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"

	"github.com/ShinteLab/suteme"
)

// 盤の縁の帯による滑りの補正を leave-one-out で測る。
//
// **その局面の帯を学習から外して判定器を作り直す。** 全件で作った判定器で
// 測ると自分の帯が最近傍に来るので必ず当たる（駒種の k-NN と同じ事情）。
//
// 期待するのは「直る件数 ≧ 壊れる件数」。実測（140 局面、`unslipMargin`=0.2）は
// 直った 2（`11f95abb` 0.85 → 0.14マス / `e0302032` 0.72 → 0.47マス）・
// 壊れた 0 で、0.5マス以内が 136 → 138。
//
// data/ は .gitignore 対象なので、無ければスキップする。
func TestUnslipHoldout(t *testing.T) {
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

	type ent struct {
		id     string
		img    image.Image
		man    image.Rectangle
		strips []suteme.StripSample
	}
	var ents []ent
	for _, e := range h.Entries {
		if e.BoardBounds == nil {
			continue
		}
		f, err := os.Open(filepath.Join(dataDir, e.ID+".png"))
		if err != nil {
			continue
		}
		img, err := png.Decode(f)
		f.Close()
		if err != nil {
			continue
		}
		b := e.BoardBounds
		s := stripSamplesFromRegion(img, b.X1, b.Y1, b.X2, b.Y2)
		if len(s) == 0 {
			continue
		}
		ents = append(ents, ent{e.ID, img,
			image.Rect(b.X1, b.Y1, b.X2, b.Y2), s})
	}
	if len(ents) < 2 {
		t.Skip("帯を作れる局面が足りないのでスキップ")
	}

	// **元に戻す（自動探索）。** 他のテストが既定の判定器を見る
	t.Cleanup(suteme.ResetStripJudge)

	// ずれ（マス単位）。局面ごとに手動座標と比べる
	offOf := func(e ent, br *suteme.BoardRegion) float64 {
		if br == nil {
			return math.Inf(1)
		}
		mcw, mch := float64(e.man.Dx())/9, float64(e.man.Dy())/9
		r := br.Bounds
		return math.Max(
			math.Max(math.Abs(float64(r.Min.X-e.man.Min.X))/mcw,
				math.Abs(float64(r.Min.Y-e.man.Min.Y))/mch),
			math.Max(math.Abs(float64(r.Dx()-e.man.Dx()))/mcw,
				math.Abs(float64(r.Dy()-e.man.Dy()))/mch))
	}

	// **判定器を外した検出（before）だけ先に並列で済ませる。**
	// 全局面が同じ「判定器なし」を使うので、グローバルを 1 回 nil にすれば
	// 同時に走らせられる。leave-one-out の after は局面ごとに別の判定器が
	// 要り、`SetStripJudge` はプロセス全体の状態なので逐次のまま
	// （並列にするには DetectBoard が判定器を引数で受ける必要がある）。
	// 実測ではこの前半が全体のほぼ半分
	suteme.SetStripJudge(nil)
	befores := make([]float64, len(ents))
	var next int
	var mu sync.Mutex
	var wg sync.WaitGroup
	for w := 0; w < runtime.GOMAXPROCS(0); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				mu.Lock()
				i := next
				next++
				mu.Unlock()
				if i >= len(ents) {
					return
				}
				befores[i] = offOf(ents[i], suteme.DetectBoard(ents[i].img))
			}
		}()
	}
	wg.Wait()

	okWith, okWithout, fixed, broke := 0, 0, 0, 0
	for i, e := range ents {
		var train []suteme.StripSample
		for j, o := range ents {
			if i != j {
				train = append(train, o.strips...)
			}
		}
		before := befores[i]
		suteme.SetStripJudge(suteme.NewStripJudge(train))
		after := offOf(e, suteme.DetectBoard(e.img))

		if before <= 0.5 {
			okWithout++
		}
		if after <= 0.5 {
			okWith++
		}
		switch {
		case before > 0.5 && after <= 0.5:
			fixed++
			t.Logf("%s 直った %.2f → %.2f マス", e.id[:8], before, after)
		case before <= 0.5 && after > 0.5:
			broke++
			t.Logf("%s 壊れた %.2f → %.2f マス", e.id[:8], before, after)
		}
	}

	t.Logf("=== %d 局面: 0.5マス以内 %d → %d（直った %d / 壊れた %d）",
		len(ents), okWithout, okWith, fixed, broke)
	if broke > fixed {
		t.Errorf("帯の補正で悪化している（直った %d / 壊れた %d）", fixed, broke)
	}
}
