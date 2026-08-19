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

// TestLookSignatureDump は「盤の見た目」の署名を書き出す（調査用）。
//
// **画像サイズで見た目を代用するのは駄目だった**（撮るたびに大きさが変わる）。
// 代わりに画像の中身で束ねられるかを測るための材料。
//
//	SUTEME_LOOKSIG=out.csv go test -run TestLookSignatureDump -timeout 30m ./training
//
// 署名は「空マスの平均ベクトル」。空マスは盤の木目・格子線・地色をそのまま
// 写していて、**どの駒が乗っているか（局面の中身）に左右されない**。
// 駒のあるマスで作ると、同じ盤でも並びが違えば別物になってしまう。
func TestLookSignatureDump(t *testing.T) {
	out := os.Getenv("SUTEME_LOOKSIG")
	if out == "" {
		t.Skip("SUTEME_LOOKSIG が空なのでスキップ（調査用）")
	}
	if !filepath.IsAbs(out) {
		if abs, err := filepath.Abs(out); err == nil {
			out = abs
		}
	}
	if !chdirToData(t) {
		t.Fatal("data/history.json が見つかりません")
	}
	h := loadHistory()

	var sb strings.Builder
	for _, e := range h.Entries {
		if e.BoardBounds == nil {
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
		b := e.BoardBounds
		br := suteme.BoardRegionFromRect(b.X1, b.Y1, b.X2, b.Y2)
		bc := suteme.BoardColor(img, br)

		sum := make([]float64, suteme.InputSize)
		n := 0
		for r := 0; r < 9; r++ {
			for c := 0; c < 9; c++ {
				if want[r][c] != suteme.EmptyLabel {
					continue
				}
				cell := br.ExtractCell(img, r, c)
				if cell == nil {
					continue
				}
				for i, v := range suteme.CellToInput(cell) {
					sum[i] += v
				}
				n++
			}
		}
		if n == 0 {
			continue
		}
		cw := float64(br.Cells[0][0].Dx())
		ch := float64(br.Cells[0][0].Dy())
		fmt.Fprintf(&sb, "%s,%dx%d,%d,%d,%.2f,%.2f", e.ID,
			img.Bounds().Dx(), img.Bounds().Dy(), n, bc, cw, ch)
		for _, v := range sum {
			fmt.Fprintf(&sb, ",%.5f", v/float64(n))
		}
		sb.WriteString("\n")
	}
	if err := os.WriteFile(out, []byte(sb.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Logf("書き出し: %s", out)
}

// TestLookHelpVsDistance は「署名が近い相手のデータは本当に役に立つか」を測る（調査用）。
//
//	SUTEME_LOOKHELP=out.csv go test -run TestLookHelpVsDistance -timeout 60m ./training
//
// **見た目の束ね方の正解は「その相手のサンプルがあると読めるようになるか」**
// なので、目で見て似ているかではなく、そこを直接測る。
// 1 局面だけを学習データにして別の局面を読み、署名の距離と一致率の関係を見る
// （「同じ出所 1 枚 対 他所 1 枚」を全組み合わせに広げたもの）。
func TestLookHelpVsDistance(t *testing.T) {
	out := os.Getenv("SUTEME_LOOKHELP")
	if out == "" {
		t.Skip("SUTEME_LOOKHELP が空なのでスキップ（調査用）")
	}
	if !filepath.IsAbs(out) {
		if abs, err := filepath.Abs(out); err == nil {
			out = abs
		}
	}
	if !chdirToData(t) {
		t.Fatal("data/history.json が見つかりません")
	}
	h := loadHistory()

	type board struct {
		id      string
		img     image.Image
		br      *suteme.BoardRegion
		want    *[9][9]string
		sig     []float64
		samples []suteme.TrainingSample
		size    string
		at      string
	}
	boards := []board{}
	for _, e := range h.Entries {
		if e.BoardBounds == nil {
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
		s, err := samplesFromHistory(e)
		if err != nil {
			continue
		}
		b := e.BoardBounds
		br := suteme.BoardRegionFromRect(b.X1, b.Y1, b.X2, b.Y2)
		sig, n := make([]float64, suteme.InputSize), 0
		for r := 0; r < 9; r++ {
			for c := 0; c < 9; c++ {
				if want[r][c] != suteme.EmptyLabel {
					continue
				}
				if cell := br.ExtractCell(img, r, c); cell != nil {
					for i, v := range suteme.CellToInput(cell) {
						sig[i] += v
					}
					n++
				}
			}
		}
		if n == 0 {
			continue
		}
		for i := range sig {
			sig[i] /= float64(n)
		}
		boards = append(boards, board{
			id: e.ID, img: img, br: br, want: want, sig: sig,
			samples: MergeSamples(s),
			size:    fmt.Sprintf("%dx%d", img.Bounds().Dx(), img.Bounds().Dy()),
			at:      e.CreatedAt,
		})
	}
	t.Logf("局面 %d", len(boards))

	sigDist := func(a, b []float64) float64 {
		s := 0.0
		for i := range a {
			d := a[i] - b[i]
			s += d * d
		}
		return s / float64(len(a))
	}
	// read は「相手 1 局面だけを学習データにして読む」
	read := func(target, partner *board) (ok, total int) {
		kn := suteme.NewKNN(partner.samples)
		if kn == nil {
			return 0, 0
		}
		bc := suteme.BoardColor(target.img, target.br)
		bo := suteme.NewBoardOrient(target.img, target.br, bc)
		for r := 0; r < 9; r++ {
			for c := 0; c < 9; c++ {
				cell := target.br.ExtractCell(target.img, r, c)
				if cell == nil {
					continue
				}
				total++
				if predictCellBO(cell, bc, bo, kn) == target.want[r][c] {
					ok++
				}
			}
		}
		return ok, total
	}

	var sb strings.Builder
	sb.WriteString("id,partner,same_size,sig_dist,ok,total\n")
	for i := range boards {
		// 相手は署名の近い順に並べ、1番目・2番目・中央・最遠を測る
		type cand struct {
			j int
			d float64
		}
		cands := make([]cand, 0, len(boards))
		for j := range boards {
			if j != i {
				cands = append(cands, cand{j, sigDist(boards[i].sig, boards[j].sig)})
			}
		}
		sort.Slice(cands, func(a, b int) bool { return cands[a].d < cands[b].d })
		pick := []int{0, 1, len(cands) / 2, len(cands) - 1}
		for _, p := range pick {
			c := cands[p]
			ok, total := read(&boards[i], &boards[c.j])
			fmt.Fprintf(&sb, "%s,%s,%d,%.5f,%d,%d\n", boards[i].id, boards[c.j].id,
				btoi(boards[i].size == boards[c.j].size), c.d, ok, total)
		}
	}
	if err := os.WriteFile(out, []byte(sb.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Logf("書き出し: %s", out)
}
