package suteme

import (
	"encoding/json"
	"image"
	_ "image/png"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

// emptiedCase は「駒があるのに分類器が空と言う」実例（data/emptied/cases.json）。
type emptiedCase struct {
	File string `json:"file"`
	// Region は盤の外枠（x1, y1, x2, y2）。ikkyoku が認識に使った矩形そのもの
	// （`WithRect` で渡す。検出の話と切り離すため）。
	Region []int `json:"region"`
	// Pieces は読めなければならないマス（「3七」の形）と正解の SFEN 表記。
	Pieces map[string]string `json:"pieces"`
	// Unread は駒があるのに**まだ読めない**と分かっているマス（落とさずに記録だけする）。
	// 読めるようになったらログに出るので、Pieces へ移すこと。
	Unread map[string]string `json:"unread,omitempty"`
	// Note は何が写っているか。
	Note string `json:"note,omitempty"`
}

// 分類器が空と言った駒を、推論器の照合で駒に戻せること（`overturnEmpty`）。
//
// 画像は ikkyoku の録画（`%APPDATA%\ikkyoku\captures\`）から `data/emptied/` に
// 写したもの。中継・ゲーム画面の画像なので data/ と同じく gitignore で、
// 無ければスキップする。推論器は全件の学習データ（`training_data_v8.bin`。
// 無ければスキップ）。
//
//   - 字画の細い「と」が空きと読まれる中継（同じ局面で矩形が 1px 違う 1 枚は読めている）
//   - ゲーム画面で、直前の手のマスの赤い色付けの上の歩が空きと読まれる
func TestEmptiedPiecesRead(t *testing.T) {
	dir := findDataDir()
	if dir == "" {
		t.Skip("data/ が無いのでスキップ")
	}
	cases := loadEmptiedCases(t, dir)
	p, err := LoadPredictor(filepath.Dir(dir))
	if err != nil {
		t.Skipf("学習データが無いのでスキップ（%v）", err)
	}
	kn, _ := p.(*KNN)
	for _, c := range cases {
		img := loadEmptiedImage(t, dir, c.File)
		if img == nil {
			continue
		}
		r, err := Recognize(img, WithPredictor(p), WithRect(c.Region[0], c.Region[1], c.Region[2], c.Region[3]))
		if r == nil {
			t.Errorf("%s: %v", c.File, err)
			continue
		}
		t.Logf("%s（%s）\n%s", c.File, c.Note, r.Debug.Dump())
		for _, sq := range sortedKeys(c.Unread) {
			row, col, ok := parseSquareName(sq)
			if !ok {
				t.Errorf("%s: マスの名前が読めない %q", c.File, sq)
				continue
			}
			cd := r.Debug.Cell(row, col)
			state := "まだ読めない"
			if cd.Piece == c.Unread[sq] {
				state = "**読めるようになった（pieces へ移すこと）**"
			}
			t.Logf("  %s: %s 読み %q（正解 %q） cover=%.3f / 境目 %.3f %s",
				sq, state, cd.Piece, c.Unread[sq], cd.Cover, r.Debug.EmptyCover, emptiedDistances(kn, img, r, row, col))
		}
		for _, sq := range sortedKeys(c.Pieces) {
			row, col, ok := parseSquareName(sq)
			if !ok {
				t.Errorf("%s: マスの名前が読めない %q", c.File, sq)
				continue
			}
			cd := r.Debug.Cell(row, col)
			t.Logf("  %s: 読み %q（正解 %q） category=%v class=%d conf=%.2f cover=%.3f / 境目 %.3f orient=%q piece_by=%q %s",
				sq, cd.Piece, c.Pieces[sq], cd.Category, cd.Class, cd.Confidence,
				cd.Cover, r.Debug.EmptyCover, cd.OrientBy, cd.PieceBy, emptiedDistances(kn, img, r, row, col))
			if cd.Piece != c.Pieces[sq] {
				t.Errorf("%s: %s を %q と読んだ（正解 %q。%s）", c.File, sq, cd.Piece, c.Pieces[sq], c.Note)
			}
		}
	}
}

// emptiedDistances は調査用に照合の距離を文字列にする（emptyDistNorm で正規化）。
func emptiedDistances(kn *KNN, img image.Image, r *Result, row, col int) string {
	if kn == nil {
		return ""
	}
	br := BoardRegionFromRect(r.Debug.Region.Min.X, r.Debug.Region.Min.Y, r.Debug.Region.Max.X, r.Debug.Region.Max.Y)
	cell := br.ExtractCell(img, row, col)
	if cell == nil {
		return ""
	}
	up := kn.PieceDistance(cell) / emptyDistNorm
	down := kn.PieceDistance(Rotate180(cell)) / emptyDistNorm
	full := [2][]float64{CellToInputFull(cell), CellToInputFull(Rotate180(cell))}
	best := math.Inf(1)
	for _, s := range kn.empties {
		for _, in := range full {
			if d, ok := dist2(in, s.Input, best); ok {
				best = d
			}
		}
	}
	return "piece(up/down)=" + ftoa(up) + "/" + ftoa(down) + " empty=" + ftoa(best/emptyDistNorm)
}

func ftoa(v float64) string {
	b, _ := json.Marshal(math.Round(v*1000) / 1000)
	return string(b)
}

func sortedKeys(m map[string]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func loadEmptiedCases(t *testing.T, dir string) []emptiedCase {
	t.Helper()
	f, err := os.Open(filepath.Join(dir, "emptied", "cases.json"))
	if err != nil {
		t.Skip("data/emptied/cases.json が無いのでスキップ")
	}
	defer f.Close()
	var cases []emptiedCase
	if err := json.NewDecoder(f).Decode(&cases); err != nil {
		t.Fatalf("cases.json: %v", err)
	}
	return cases
}

func loadEmptiedImage(t *testing.T, dir, file string) image.Image {
	t.Helper()
	f, err := os.Open(filepath.Join(dir, "emptied", file))
	if err != nil {
		t.Errorf("%s: %v", file, err)
		return nil
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		t.Errorf("%s: %v", file, err)
		return nil
	}
	return img
}

// parseSquareName は「3一」を (段, 左からの列) にする（squareName の逆）。
func parseSquareName(s string) (row, col int, ok bool) {
	rs := []rune(s)
	if len(rs) != 2 || rs[0] < '1' || rs[0] > '9' {
		return 0, 0, false
	}
	for i, k := range []rune("一二三四五六七八九") {
		if rs[1] == k {
			return i, 9 - int(rs[0]-'0'), true
		}
	}
	return 0, 0, false
}
