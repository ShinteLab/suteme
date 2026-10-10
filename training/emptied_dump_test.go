package training

import (
	"fmt"
	"image"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"

	"github.com/ShinteLab/suteme"
)

// TestEmptiedDump は「分類器が空と言ったマス」の材料をホールドアウトで書き出す（調査用）。
//
// 分類器（被覆率としきい値）が空と言ったマスは推論器に回らない。そこに
// 「駒なのに空と言われたマス」（字画の細い「と」・直前の手の色付けの上の駒など）が
// 混ざるので、**真の空と分けられる材料があるか**をこの CSV の上で数える。
// 書き出すのは全マスで、分類器が空と言ったマスにだけ照合の距離を付ける:
//
//	cover / emptymax   被覆率と、その盤の空判定の境目（`Debug.EmptyCover`）
//	dup / ddown        駒サンプルへの最近傍（そのまま / 180度回した版。emptyDistNorm で正規化）
//	dempty             空サンプルへの最近傍（打ち切らない。同上）
//	pclass / pconf     近いほうの向きに揃えて `Predict` した駒種と確信度
//	                   （空サンプルと一致すれば ClassEmpty が返る）
//
//	SUTEME_EMPTIED=out.csv go test -run TestEmptiedDump -timeout 60m -v ./training
//
// `SUTEME_EMPTIED_MODE=look` で**見た目ごと**学習から外す（`holdoutPredictors` を
// 見た目の名前で束ねる。評価タブの「見た目LOO」と同じ）。既定は局面ごと。
// 対象は確認済みの局面だけ。盤面座標は格子線へ寄せたもの（評価タブと同じ切り出し）。
// `SUTEME_EMPTIED_RECT=raw` で保存された座標そのもの（`TestKNNHoldout` と同じ切り出し）。
// **両方測ること。** 窓が数 px 動くだけで、空マスに隣の駒の切れ端が入って駒サンプルに
// 近づくことがある（寄せた座標では出ず、寄せない座標でだけ出た誤りがある）。
func TestEmptiedDump(t *testing.T) {
	out := os.Getenv("SUTEME_EMPTIED")
	if out == "" {
		t.Skip("SUTEME_EMPTIED が空なのでスキップ（調査用）")
	}
	if !filepath.IsAbs(out) {
		if abs, err := filepath.Abs(out); err == nil {
			out = abs // chdirToData で作業ディレクトリが動くため
		}
	}
	byLook := os.Getenv("SUTEME_EMPTIED_MODE") == "look"
	rawRect := os.Getenv("SUTEME_EMPTIED_RECT") == "raw"
	if !chdirToData(t) {
		t.Fatal("data/history.json が見つかりません")
	}
	h := loadHistory()

	keyOf := func(e HistoryEntry) string { return e.ID }
	mode := "board"
	if byLook {
		keyOf = func(e HistoryEntry) string { return strings.TrimSpace(e.Look) }
		mode = "look"
	}

	var targets []*evalTarget
	for _, e := range h.Entries {
		if !e.IsVerified() || keyOf(e) == "" {
			continue
		}
		tg, err := loadEvalTarget(e)
		if err != nil {
			t.Logf("%s: %v", e.ID, err)
			continue
		}
		targets = append(targets, tg)
	}

	// 学習データは key ごとに畳んでおき、外す key 以外を連結する
	byKey := map[string][]suteme.TrainingSample{}
	for _, tg := range targets {
		s, err := samplesFromHistory(tg.entry)
		if err != nil {
			t.Logf("%s: サンプル化に失敗 (%v)", tg.entry.ID, err)
			continue
		}
		k := keyOf(tg.entry)
		byKey[k] = append(byKey[k], s...)
	}
	for k, s := range byKey {
		byKey[k] = MergeSamples(s)
	}

	rows := make([]string, len(targets))
	var wg sync.WaitGroup
	sem := make(chan struct{}, max(1, runtime.GOMAXPROCS(0)/4))
	for i, tg := range targets {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int, tg *evalTarget) {
			defer wg.Done()
			defer func() { <-sem }()
			k := keyOf(tg.entry)
			var train []suteme.TrainingSample
			for other, s := range byKey {
				if other != k {
					train = append(train, s...)
				}
			}
			kn := suteme.NewKNN(train)
			if kn == nil {
				return
			}
			var empties []suteme.TrainingSample
			for _, s := range train {
				if s.Label == suteme.ClassEmpty {
					empties = append(empties, s)
				}
			}
			rows[i] = dumpEmptiedBoard(mode, k, tg, kn, empties, rawRect)
		}(i, tg)
	}
	wg.Wait()

	var sb strings.Builder
	sb.WriteString("mode,key,id,row,col,want,got,cat,cover,emptymax,dup,ddown,dempty,pclass,pconf,plabel\n")
	for _, r := range rows {
		sb.WriteString(r)
	}
	if err := os.WriteFile(out, []byte(sb.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Logf("%d 局面を書き出し: %s", len(targets), out)
}

func dumpEmptiedBoard(mode, key string, tg *evalTarget, kn *suteme.KNN, empties []suteme.TrainingSample, rawRect bool) string {
	s := tg.snap
	if rawRect {
		b := tg.rect
		s = image.Rect(b.X1, b.Y1, b.X2, b.Y2)
	}
	r, err := suteme.Recognize(tg.img, suteme.WithPredictor(kn), suteme.WithRect(s.Min.X, s.Min.Y, s.Max.X, s.Max.Y))
	if r == nil {
		return ""
	}
	_ = err
	br := suteme.BoardRegionFromRect(s.Min.X, s.Min.Y, s.Max.X, s.Max.Y)
	norm := float64(2 * suteme.InputSize)
	var sb strings.Builder
	for _, cd := range r.Debug.Cells {
		want := tg.want[cd.Row][cd.Col]
		got := suteme.EmptyLabel
		if cd.Piece != "" {
			got = sfenToGridLabel(cd.Piece)
		}
		// cat は**分類器の**判定（照合で駒に戻したマスも分類器は空と言っている）
		cat := cd.Category
		if cd.PieceBy != "" {
			cat = suteme.CellEmpty
		}
		fmt.Fprintf(&sb, "%s,%s,%s,%d,%d,%s,%s,%d,%.4f,%.4f", mode, csvSafe(key), tg.entry.ID, cd.Row, cd.Col,
			want, got, int(cat), cd.Cover, r.Debug.EmptyCover)
		cell := br.ExtractCell(tg.img, cd.Row, cd.Col)
		if cat != suteme.CellEmpty || cell == nil {
			sb.WriteString(",,,,,,\n")
			continue
		}
		rot := suteme.Rotate180(cell)
		up := kn.PieceDistance(cell) / norm
		down := kn.PieceDistance(rot) / norm
		full := [2][]float64{suteme.CellToInputFull(cell), suteme.CellToInputFull(rot)}
		best := math.Inf(1)
		for _, e := range empties {
			for _, in := range full {
				if d := dist2Limit(in, e.Input, best); d < best {
					best = d
				}
			}
		}
		ncell, pre := cell, ""
		if down < up {
			ncell, pre = rot, "-"
		}
		class, conf := kn.Predict(ncell)
		label := suteme.EmptyLabel
		if class != suteme.ClassEmpty {
			label = pre + suteme.ClassToBaseLabel(class)
		}
		fmt.Fprintf(&sb, ",%.4f,%.4f,%.4f,%d,%.4f,%s\n", up, down, best/norm, class, conf, label)
	}
	return sb.String()
}

// sfenToGridLabel は SFEN の駒表記（"p" / "+P"）を boardGrid の表記（"-P" / "+P"）にする。
func sfenToGridLabel(p string) string {
	base := strings.TrimPrefix(p, "+")
	pre := ""
	if strings.HasPrefix(p, "+") {
		pre = "+"
	}
	if base != strings.ToUpper(base) {
		return "-" + pre + strings.ToUpper(base)
	}
	return pre + base
}

func csvSafe(s string) string { return strings.NewReplacer(",", "_", "\n", " ").Replace(s) }

// dist2Limit は 2 乗距離（limit を超えたら打ち切って +Inf）。
func dist2Limit(a, b []float64, limit float64) float64 {
	d := 0.0
	for j := range a {
		x := a[j] - b[j]
		d += x * x
		if j&31 == 31 && d >= limit {
			return math.Inf(1)
		}
	}
	return d
}
