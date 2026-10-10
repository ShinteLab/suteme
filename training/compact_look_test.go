package training

import (
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/ShinteLab/suteme"
)

// TestCompactLookHoldout は**配布用に間引いた k-NN**（`BuildCompactData`）の、
// 初めて見る盤に対する成績を全件と並べて測る。
//
// 「認識器を配る」の節の表は**その局面を学習済み**の条件なので、間引きが
// 目標（どんな盤でも読む）に効くかは分からない。ここでは見た目ホールドアウト
// （`TestLookHoldout` と同じ。その見た目を丸ごと学習から外す）で、
// 外した残りを全件のまま使う場合と、クラスごとに間引いた場合を比べる。
//
// ⚠️ **学習に使うのは見た目の判定済みの局面だけ**（`holdoutPredictors` と同じ）。
// 判定していない局面には外した見た目と同じ盤が混ざっているかもしれないため。
// そのため母集団が配る学習データ（全局面から作る）より小さく、**間引く割合が実際の
// 配布より緩い**（実測で 1000/class は 27239 → 7065 件＝26%。配布は 66679 → 9233 件＝14%）。
// 実際の配布での差がこの測り方より大きいか小さいかは分からない。
//
// 間引きの段階は `SUTEME_COMPACT_LEVELS`（例 "250,1000"。既定 250,500,1000,2000）。
// 合否判定はしない。
//
//	$env:SUTEME_SLOW=1; go test -run TestCompactLookHoldout -timeout 60m -v ./training
func TestCompactLookHoldout(t *testing.T) {
	requireSlow(t)
	if !chdirToData(t) {
		t.Skip("data/history.json が無いのでスキップ")
	}
	levels := []int{250, 500, 1000, 2000}
	if v := os.Getenv("SUTEME_COMPACT_LEVELS"); v != "" {
		levels = levels[:0]
		for _, s := range strings.Split(v, ",") {
			n, err := strconv.Atoi(strings.TrimSpace(s))
			if err != nil || n <= 0 {
				t.Fatalf("SUTEME_COMPACT_LEVELS: %q", s)
			}
			levels = append(levels, n)
		}
	}

	h := loadHistory()
	lookOf := func(e HistoryEntry) string { return strings.TrimSpace(e.Look) }

	// 見た目ごとのサンプル（holdoutPredictors と同じ集め方）
	byLook := map[string][]suteme.TrainingSample{}
	for _, e := range h.Entries {
		look := lookOf(e)
		if look == "" || !e.IsVerified() {
			continue
		}
		s, err := samplesFromHistory(e)
		if err != nil {
			t.Logf("%s: サンプル化に失敗 (%v)", e.ID, err)
			continue
		}
		byLook[look] = append(byLook[look], s...)
	}
	if len(byLook) < 2 {
		t.Skip("見た目の判定済みの局面が足りないのでスキップ")
	}

	// 対象の局面を見た目ごとに読み込んでおく
	targets := map[string][]*evalTarget{}
	for _, e := range h.Entries {
		look := lookOf(e)
		if _, ok := byLook[look]; !ok || !e.IsVerified() {
			continue
		}
		tg, err := loadEvalTarget(e)
		if err != nil {
			continue
		}
		targets[look] = append(targets[look], tg)
	}

	looks := make([]string, 0, len(targets))
	for k := range targets {
		looks = append(looks, k)
	}
	sort.Strings(looks)

	// 列 0 が全件、以降が間引き
	names := []string{"全件"}
	for _, n := range levels {
		names = append(names, fmt.Sprintf("%d/class", n))
	}
	total := make([]EvalMetrics, len(names))
	samples := make([]int, len(names)) // 見た目をまたいだ平均を出すための合計

	for _, look := range looks {
		var train []suteme.TrainingSample
		for other, s := range byLook {
			if other != look {
				train = append(train, s...)
			}
		}
		full := MergeSamples(train)
		sets := [][]suteme.TrainingSample{full}
		for _, n := range levels {
			sets = append(sets, BuildCompactData(full, n))
		}

		line := fmt.Sprintf("[%s] %d 局面:", look, len(targets[look]))
		for i, set := range sets {
			kn := suteme.NewKNN(set)
			if kn == nil {
				continue
			}
			samples[i] += kn.Len()
			var m EvalMetrics
			for _, tg := range targets[look] {
				s := tg.snap
				bm := EvalMetrics{}
				if g, _, err := recognizeGrid(tg.img, suteme.Predictor(kn), suteme.WithRect(s.Min.X, s.Min.Y, s.Max.X, s.Max.Y)); err == nil {
					bm = gradeBoard(tg.want, g)
				}
				m.add(bm)
			}
			total[i].add(m)
			line += fmt.Sprintf("  %s %d/%d", names[i], m.OK, m.Cells)
		}
		t.Log(line)
	}

	t.Logf("見た目LOO %d 見た目（学習は判定済みの局面だけ）", len(looks))
	for i, m := range total {
		if m.Cells == 0 {
			continue
		}
		t.Logf("  %-10s 平均 %6d サンプル  全マス %d/%d = %.2f%%  完全一致 %d  空→駒 %d  駒→空 %d  向き %d  駒種違い %d  駒種 %.2f%%",
			names[i], samples[i]/len(looks), m.OK, m.Cells, 100*float64(m.OK)/float64(m.Cells), m.BoardsPerfect,
			m.EmptyAsPiece, m.PieceAsEmpty, m.OrientFlip, m.TypeMiss, 100*float64(m.PieceOK)/float64(m.PieceCells))
	}
}
