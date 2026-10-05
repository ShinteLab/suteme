package suteme

import (
	"encoding/json"
	"fmt"
	"image"
	_ "image/png"
	"os"
	"path/filepath"
	"testing"
)

// hiddenFalseMaxRatio は、隠れていない保存済み局面のうち「見えないマスがある」と
// 言ってよい盤の割合。
//
// **0 にはできない。** 判定は外枠の上の交点も見るが、外枠の外は盤の外なので
// 内側とだけ比べることになり、盤が画像の端で切れているゲーム画面や、
// 中継の UI が盤の角に被っている画像では測れない（実測 206 局面で 2 枚:
// `bb67a1a1` は最後に動いた駒のマスの色付けが右の外枠に接している、
// `5e1b56df` は UI の飾りが盤の左下の角に実際に被っている）。
// **増えたら気付けるように件数を縛る。**
const hiddenFalseMaxRatio = 0.02

// 保存済み局面（隠れていない盤）で「見えない」と言うマスが出ないこと。
//
// 誤って「見えない」と言うと、呼び出し側（ikkyoku）はそのマスに関わる手を
// 待ち続ける。盤面座標は人が引いたものを `SnapToGrid` に通して使う（学習と同じ切り出し）。
func TestHiddenCellsCleanBoards(t *testing.T) {
	dir, entries := loadSavedBoards(t)
	type res struct {
		id    string
		cells []string
	}
	rs := mapSavedBoards(dir, entries, func(e historyEntry, img image.Image) res {
		r := res{id: e.ID}
		if e.Bounds.X2 == 0 {
			return r
		}
		br := SnapToGrid(img, BoardRegionFromRect(e.Bounds.X1, e.Bounds.Y1, e.Bounds.X2, e.Bounds.Y2))
		h := HiddenCells(img, br)
		for row := 0; row < 9; row++ {
			for col := 0; col < 9; col++ {
				if h[row][col] {
					r.cells = append(r.cells, squareName(row, col))
				}
			}
		}
		return r
	})
	boards, cells := 0, 0
	for _, r := range rs {
		if len(r.cells) > 0 {
			boards++
			cells += len(r.cells)
			t.Logf("%s: 見えない %v", r.id[:8], r.cells)
		}
	}
	t.Logf("見えないマスのある盤 %d/%d（マス %d）", boards, len(rs), cells)
	if max := int(float64(len(rs)) * hiddenFalseMaxRatio); boards > max {
		t.Errorf("隠れていない盤で見えないマスを返した盤が %d 枚（上限 %d）", boards, max)
	}
}

// squareName はマス (段, 左からの列) を「3一」の形にする。
func squareName(row, col int) string {
	return fmt.Sprintf("%d%s", 9-col, []string{"一", "二", "三", "四", "五", "六", "七", "八", "九"}[row])
}

// occludedCase は手や頭が被った実例（data/occluded/cases.json）。
type occludedCase struct {
	File string `json:"file"`
	// Hidden は見えないと言わなければならないマス（「3一」の形）。
	// 誤った手の元になったマスだけを書く（被っているマスを全部は書かない）。
	Hidden []string `json:"hidden"`
	// Clean は隠れていない画像（見えないマスが 1 つも無いこと）。
	Clean bool `json:"clean"`
	// Note は何が写っているか。
	Note string `json:"note,omitempty"`
}

// 手や頭が被った実例で、誤った手の元になったマスを「見えない」と言うこと。
//
// 画像は ikkyoku の追従の録画（`%APPDATA%\ikkyoku\captures\follow\`）から
// `data/occluded/` に写したもの。中継の画像なので data/ と同じく gitignore で、
// 無ければスキップする。盤面は自動検出（ikkyoku と同じく `Recognize` の経路）。
func TestHiddenCellsOccluded(t *testing.T) {
	dir := findDataDir()
	if dir == "" {
		t.Skip("data/ が無いのでスキップ")
	}
	dir = filepath.Join(dir, "occluded")
	f, err := os.Open(filepath.Join(dir, "cases.json"))
	if err != nil {
		t.Skip("data/occluded/cases.json が無いのでスキップ")
	}
	var cases []occludedCase
	err = json.NewDecoder(f).Decode(&cases)
	f.Close()
	if err != nil {
		t.Fatalf("cases.json: %v", err)
	}
	for _, c := range cases {
		fp, err := os.Open(filepath.Join(dir, c.File))
		if err != nil {
			t.Errorf("%s: %v", c.File, err)
			continue
		}
		img, _, err := image.Decode(fp)
		fp.Close()
		if err != nil {
			t.Errorf("%s: %v", c.File, err)
			continue
		}
		br, _, _ := detectBoardRegion(img)
		if br == nil {
			t.Errorf("%s: 盤を検出できない", c.File)
			continue
		}
		h := HiddenCells(img, br)
		var got []string
		set := map[string]bool{}
		for row := 0; row < 9; row++ {
			for col := 0; col < 9; col++ {
				if h[row][col] {
					got = append(got, squareName(row, col))
					set[squareName(row, col)] = true
				}
			}
		}
		t.Logf("%s（%s）: 見えない %v", c.File, c.Note, got)
		for _, want := range c.Hidden {
			if !set[want] {
				t.Errorf("%s: %s を見えないと言っていない（%s）", c.File, want, c.Note)
			}
		}
		if c.Clean && len(got) > 0 {
			t.Errorf("%s: 隠れていないのに見えない %v", c.File, got)
		}
	}
}
