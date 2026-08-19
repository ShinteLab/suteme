package training

import (
	"fmt"
	"image"
	_ "image/png"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/ShinteLab/suteme"
)

// TestSameSourceEffect は「同じ出所（同じゲーム画面・同じ中継）の局面が
// 学習データにあると何が変わるか」を測る。
//
// **同じ出所 1 枚が他所 15 枚より効く**というのが実測の結論で、
// データ収集の方針（種類を増やすより出所ごとに 3〜5 枚）はこの数字に基づく。
// 合否判定はせず数字を出すだけ:
// `go test -run TestSameSourceEffect -v ./training`
//
// 出所は画像サイズで代用する（同じ画面をキャプチャすれば一致する）。
// 最も枚数の多いサイズをグループとみなすので、data/ に同一出所が
// 複数入っていなければ意味のある数字にならない。
func TestSameSourceEffect(t *testing.T) {
	if !chdirToData(t) {
		t.Skip("no data")
	}
	h := loadHistory()
	type bd struct {
		id   string
		img  image.Image
		br   *suteme.BoardRegion
		bc   uint8
		want *[9][9]string
		size string
	}
	byEntry := map[string][]suteme.TrainingSample{}
	boards := []bd{}
	for _, e := range h.Entries {
		s, err := samplesFromHistory(e)
		if err != nil || e.BoardBounds == nil {
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
		w, err := wantGrid(e.SFEN)
		if err != nil {
			continue
		}
		byEntry[e.ID] = s
		b := e.BoardBounds
		br := suteme.BoardRegionFromRect(b.X1, b.Y1, b.X2, b.Y2)
		boards = append(boards, bd{e.ID, img, br, suteme.BoardColor(img, br), w,
			fmt.Sprintf("%dx%d", img.Bounds().Dx(), img.Bounds().Dy())})
	}
	// 同じ出所のグループは `lookGroups` で束ねる（人が付けた見た目が優先、
	// 無ければ署名の候補）。**画像サイズで代用してはいけない** ——
	// 撮るたびにサイズが変わるので同じ出所が割れる（`coverage.go` の項）
	look := map[string]string{}
	cnt := map[string]int{}
	named, proposed := lookGroups(h.Entries)
	for name, ids := range named {
		for _, id := range ids {
			look[id] = name
			cnt[name]++
		}
	}
	for i, ids := range proposed {
		name := fmt.Sprintf("候補%d", i+1)
		for _, id := range ids {
			look[id] = name
			cnt[name]++
		}
	}
	for i := range boards {
		boards[i].size = look[boards[i].id]
	}
	group := ""
	for s, n := range cnt {
		if n > cnt[group] {
			group = s
		}
	}
	t.Logf("同じ見た目とみなすグループ: %s (%d 枚)", group, cnt[group])

	eval := func(target bd, train []suteme.TrainingSample) (int, int) {
		kn := suteme.NewKNN(MergeSamples(train))
		if kn == nil {
			return 0, 0
		}
		ok, all := 0, 0
		for r := 0; r < 9; r++ {
			for c := 0; c < 9; c++ {
				cell := target.br.ExtractCell(target.img, r, c)
				if cell == nil || target.want[r][c] == suteme.EmptyLabel {
					continue
				}
				all++
				if predictCell(cell, target.bc, kn) == target.want[r][c] {
					ok++
				}
			}
		}
		return ok, all
	}

	rnd := rand.New(rand.NewSource(3))
	var sibOK, sibAll, othOK, othAll, allOK, allAll int
	for _, target := range boards {
		if target.size != group {
			continue
		}
		var sibs, others []string
		for _, b := range boards {
			if b.id == target.id {
				continue
			}
			if b.size == group {
				sibs = append(sibs, b.id)
			} else {
				others = append(others, b.id)
			}
		}
		gather := func(ids []string) []suteme.TrainingSample {
			var s []suteme.TrainingSample
			for _, id := range ids {
				s = append(s, byEntry[id]...)
			}
			return s
		}
		// (a) 同じ出所だけ（6枚）
		o, n := eval(target, gather(sibs))
		sibOK, sibAll = sibOK+o, sibAll+n
		// (b) 他所だけ 6枚（5回平均）
		for rep := 0; rep < 3; rep++ {
			rnd.Shuffle(len(others), func(i, j int) { others[i], others[j] = others[j], others[i] })
			o, n = eval(target, gather(others[:len(sibs)]))
			othOK, othAll = othOK+o, othAll+n
		}
		// (c) 他所 全部（15枚）
		o, n = eval(target, gather(others))
		allOK, allAll = allOK+o, allAll+n
	}
	fmt.Printf("同じ出所 %d 枚だけで学習   : 駒種 %.1f%% (%d/%d)\n", cnt[group]-1, 100*float64(sibOK)/float64(sibAll), sibOK, sibAll)
	fmt.Printf("他所 %d 枚だけで学習       : 駒種 %.1f%% (%d/%d)\n", cnt[group]-1, 100*float64(othOK)/float64(othAll), othOK, othAll)
	fmt.Printf("他所 全部(%d枚)で学習      : 駒種 %.1f%% (%d/%d)\n", len(boards)-cnt[group], 100*float64(allOK)/float64(allAll), allOK, allAll)

	fmt.Println()
	for n := 1; n <= cnt[group]-1; n++ {
		var ok, all int
		for _, target := range boards {
			if target.size != group {
				continue
			}
			var sibs []string
			for _, b := range boards {
				if b.id != target.id && b.size == group {
					sibs = append(sibs, b.id)
				}
			}
			for rep := 0; rep < 3; rep++ {
				rnd.Shuffle(len(sibs), func(i, j int) { sibs[i], sibs[j] = sibs[j], sibs[i] })
				var s []suteme.TrainingSample
				for _, id := range sibs[:n] {
					s = append(s, byEntry[id]...)
				}
				o, m := eval(target, s)
				ok, all = ok+o, all+m
				if n == len(sibs) {
					break
				}
			}
		}
		fmt.Printf("同じ出所 %d 枚だけ: 駒種 %.1f%%\n", n, 100*float64(ok)/float64(all))
	}
}
