package training

import (
	"fmt"
	"image"
	_ "image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ShinteLab/suteme"
)

// TestOrientDump は向き判定の材料をホールドアウトで書き出す（調査用）。
//
// 分類器の確信度（`BoardOrient.Detail`）と回転照合の距離（`PieceDistance` の
// そのまま / 180度回した版）を全駒マスについて取り、CSV に落とす。
// **決め方（しきい値・重み・組み合わせ方）を変えるたびに全局面を計算し直さず、
// この CSV の上で数え直せるようにするためのもの。** 1 回 90 秒。
// 向きを回転照合に一本化した判断もこの材料で出した（`suteme.classifyCellOrient`）。
//
//	SUTEME_ORIENT=out.csv go test -run TestOrientDump -timeout 60m -v ./training
//
// `SUTEME_ORIENT_MODE=source` を付けると、局面ではなく**出所ごと**学習から外す
// （出所は画像サイズで代用。`TestSameSourceEffect` と同じ）。回転照合は
// 同じ出所の局面が学習データにあるかで成績が大きく変わるので、
// **両方測らないと判断を誤る。**
func TestOrientDump(t *testing.T) {
	out := os.Getenv("SUTEME_ORIENT")
	if out == "" {
		t.Skip("SUTEME_ORIENT が空なのでスキップ（調査用）")
	}
	if !filepath.IsAbs(out) {
		if abs, err := filepath.Abs(out); err == nil {
			out = abs // chdirToData で作業ディレクトリが動くため
		}
	}
	bySource := os.Getenv("SUTEME_ORIENT_MODE") == "source"
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
	// 局面ごとに一度だけ畳んでおく。局面をまたいだ重複は同じ画像を
	// 二重登録したときしか出ないので、以降は連結だけにして時間を稼ぐ。
	byEntry := make(map[string][]suteme.TrainingSample, len(h.Entries))
	boards := make([]board, 0, len(h.Entries))
	for _, e := range h.Entries {
		if e.BoardBounds == nil {
			continue
		}
		s, err := samplesFromHistory(e)
		if err != nil {
			t.Logf("%s: サンプル化に失敗 (%v)", e.ID, err)
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
			t.Logf("%s: SFEN 解析に失敗 (%v)", e.ID, err)
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

	source := make(map[string]string, len(boards)) // ID → 出所
	for _, b := range boards {
		source[b.id] = b.source
	}

	var sb strings.Builder
	sb.WriteString("id,source,row,col,want_down,cat_down,margin,dist_up,dist_down\n")
	for _, b := range boards {
		var train []suteme.TrainingSample
		for id, s := range byEntry {
			if id == b.id || (bySource && source[id] == b.source) {
				continue
			}
			train = append(train, s...)
		}
		kn := suteme.NewKNN(train)
		if kn == nil {
			continue
		}
		bc := suteme.BoardColor(b.img, b.br)
		bo := suteme.NewBoardOrient(b.img, b.br, bc)

		n := 0
		for r := 0; r < 9; r++ {
			for c := 0; c < 9; c++ {
				if b.want[r][c] == suteme.EmptyLabel {
					continue
				}
				cell := b.br.ExtractCell(b.img, r, c)
				if cell == nil {
					continue
				}
				cat, margin := bo.Detail(cell, bc)
				if cat == suteme.CellEmpty {
					continue // 駒→空。向きの話ではない
				}
				up := kn.PieceDistance(cell)
				down := kn.PieceDistance(suteme.Rotate180(cell))
				fmt.Fprintf(&sb, "%s,%s,%d,%d,%d,%d,%.6f,%.6f,%.6f\n",
					b.id, b.source, r, c,
					btoi(strings.HasPrefix(b.want[r][c], "-")),
					btoi(cat == suteme.CellPieceDown),
					margin, up, down)
				n++
			}
		}
		t.Logf("%s (%s): %d マス", b.id, b.source, n)
	}
	if err := os.WriteFile(out, []byte(sb.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Logf("書き出し: %s", out)
}

func btoi(b bool) int {
	if b {
		return 1
	}
	return 0
}
