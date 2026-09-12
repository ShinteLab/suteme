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

// TestLearningCurve は「学習局面を何件集めれば精度がいくらになるか」を測る。
//
// N 件をランダムに選んで k-NN を作り、選ばれなかった局面で認識する
// （= 未知の盤に対する実力）。合否判定はせず数字を出すだけ:
// `go test -run TestLearningCurve -v ./training`
//
// **保存済みの局面は 15 件すべて見た目が別**（盤の色・木目・駒の字体・
// 一字彫か二字か）なので、これは「N 種類の見た目で学習して N+1 種類目を読む」
// という一番厳しい条件を測っていることになる。同じ中継・同じゲーム画面を
// 繰り返し読む運用ではもっと有利になる（`TestKNNCellHoldout` が同じ盤での値）。
func TestLearningCurve(t *testing.T) {
	requireSlow(t)
	if !chdirToData(t) {
		t.Skip("no data")
	}
	h := loadHistory()
	byEntry := map[string][]suteme.TrainingSample{}
	ids := []string{}
	for _, e := range h.Entries {
		s, err := samplesFromHistory(e)
		if err != nil {
			continue
		}
		byEntry[e.ID] = s
		ids = append(ids, e.ID)
	}
	t.Logf("局面数 %d", len(ids))

	type board struct {
		id   string
		img  image.Image
		br   *suteme.BoardRegion
		bc   uint8
		want *[9][9]string
	}
	boards := []board{}
	for _, e := range h.Entries {
		if _, ok := byEntry[e.ID]; !ok || e.BoardBounds == nil {
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
		b := e.BoardBounds
		br := suteme.BoardRegionFromRect(b.X1, b.Y1, b.X2, b.Y2)
		boards = append(boards, board{e.ID, img, br, suteme.BoardColor(img, br), w})
	}

	rnd := rand.New(rand.NewSource(1))
	const reps = 5
	fmt.Printf("%-6s %-14s %-14s\n", "N", "全マス一致", "駒マスの駒種")
	for _, n := range []int{1, 2, 4, 8, 14} {
		if n > len(ids)-1 {
			continue
		}
		var cellOK, cellAll, pieceOK, pieceAll int
		for _, bd := range boards {
			others := []string{}
			for _, id := range ids {
				if id != bd.id {
					others = append(others, id)
				}
			}
			for rep := 0; rep < reps; rep++ {
				rnd.Shuffle(len(others), func(i, j int) { others[i], others[j] = others[j], others[i] })
				var train []suteme.TrainingSample
				for _, id := range others[:n] {
					train = append(train, byEntry[id]...)
				}
				kn := suteme.NewKNN(MergeSamples(train))
				if kn == nil {
					continue
				}
				for r := 0; r < 9; r++ {
					for c := 0; c < 9; c++ {
						cell := bd.br.ExtractCell(bd.img, r, c)
						if cell == nil {
							continue
						}
						got := predictCell(cell, bd.bc, kn)
						cellAll++
						if got == bd.want[r][c] {
							cellOK++
						}
						if bd.want[r][c] != suteme.EmptyLabel {
							pieceAll++
							if got == bd.want[r][c] {
								pieceOK++
							}
						}
					}
				}
				if n == len(ids)-1 {
					break // 全件はランダム性が無い
				}
			}
		}
		fmt.Printf("%-6d %5.1f%%         %5.1f%%\n", n,
			100*float64(cellOK)/float64(cellAll), 100*float64(pieceOK)/float64(pieceAll))
	}
}
