package suteme

import (
	"encoding/json"
	"image"
	_ "image/png"
	"os"
	"strings"
	"testing"
)

// 学習用Webサーバが保存した局面（data/）を正解データとして
// ClassifyCellWith の 空/先手/後手 の分類精度を測る。
//
// data/ は .gitignore 対象なので、無ければスキップする。
// 保存された SFEN が画像と食い違っている局面が混ざりうる（履歴を読み込んで
// 別画像を貼り直すと前の SFEN が残る）ため、閾値での合否判定はせず
// 数字を出すだけにしてある。`go test -run TestClassifyAccuracy -v` で見る。
type historyFile struct {
	Entries []historyEntry `json:"entries"`
}

type historyEntry struct {
	ID     string `json:"id"`
	SFEN   string `json:"sfen"`
	Bounds struct {
		X1, Y1, X2, Y2 int
	} `json:"board_bounds"`
}

func findDataDir() string {
	// カレント（通常）と worktree からの相対（.claude/worktrees/<name>/）
	for _, d := range []string{"data", "../../../data"} {
		if _, err := os.Stat(d + "/history.json"); err == nil {
			return d
		}
	}
	return ""
}

// sfenBoardToCategory は SFEN の盤面部分をマスの大分類に変換する
func sfenBoardToCategory(s string) [9][9]CellCategory {
	var g [9][9]CellCategory
	fields := strings.Fields(s)
	if len(fields) == 0 {
		return g
	}
	for r, rank := range strings.Split(fields[0], "/") {
		if r >= 9 {
			break
		}
		c := 0
		for i := 0; i < len(rank) && c < 9; i++ {
			switch ch := rank[i]; {
			case ch >= '1' && ch <= '9':
				c += int(ch - '0')
			case ch == '+':
				// 次の文字が駒
			case ch >= 'a' && ch <= 'z':
				g[r][c] = CellPieceDown
				c++
			default:
				g[r][c] = CellPieceUp
				c++
			}
		}
	}
	return g
}

func TestClassifyAccuracy(t *testing.T) {
	dir := findDataDir()
	if dir == "" {
		t.Skip("data/history.json が無いのでスキップ")
	}
	f, err := os.Open(dir + "/history.json")
	if err != nil {
		t.Skip(err)
	}
	var h historyFile
	err = json.NewDecoder(f).Decode(&h)
	f.Close()
	if err != nil {
		t.Fatalf("history.json: %v", err)
	}

	// [正解][判定] の混同行列
	var confusion [3][3]int
	totalOK, total := 0, 0

	for _, e := range h.Entries {
		imgF, err := os.Open(dir + "/" + e.ID + ".png")
		if err != nil {
			t.Logf("%s: 画像なし (%v)", e.ID, err)
			continue
		}
		img, _, err := image.Decode(imgF)
		imgF.Close()
		if err != nil {
			t.Logf("%s: デコード失敗 (%v)", e.ID, err)
			continue
		}

		br := BoardRegionFromRect(e.Bounds.X1, e.Bounds.Y1, e.Bounds.X2, e.Bounds.Y2)
		want := sfenBoardToCategory(e.SFEN)
		// 空判定の境目は盤ごとに決まる（BoardEmptyCover）ので、
		// マス単位の ClassifyCellWith ではなく盤単位で通す
		cats := ClassifyBoard(img, br)

		var view strings.Builder
		ok := 0
		for r := 0; r < 9; r++ {
			for c := 0; c < 9; c++ {
				got := cats[r][c]
				confusion[want[r][c]][got]++
				total++
				if got == want[r][c] {
					ok++
					view.WriteString(" .  ")
					continue
				}
				view.WriteString(want[r][c].String() + ">" + got.String() + " ")
			}
			view.WriteString("\n")
		}
		totalOK += ok
		t.Logf("%s: %d/81\n%s", e.ID, ok, view.String())
	}

	if total == 0 {
		t.Skip("正解データが無いのでスキップ")
	}
	t.Logf("合計: %d/%d = %.1f%%", totalOK, total, 100*float64(totalOK)/float64(total))
	for _, w := range []CellCategory{CellEmpty, CellPieceUp, CellPieceDown} {
		sum := confusion[w][0] + confusion[w][1] + confusion[w][2]
		if sum == 0 {
			continue
		}
		t.Logf("  %s: %d/%d (%.1f%%)  空=%d ☗=%d ☖=%d",
			w.String(), confusion[w][w], sum, 100*float64(confusion[w][w])/float64(sum),
			confusion[w][CellEmpty], confusion[w][CellPieceUp], confusion[w][CellPieceDown])
	}
}
