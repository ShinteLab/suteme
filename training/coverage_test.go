package training

import (
	"encoding/json"
	"image/color"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

// 盤の見た目ごとの厚み（coverage.go）。
//
// 守りたいのは 3 つ。
//   - 同じ画像サイズの局面が 1 つの見た目に束ねられること
//   - **薄いものが先、その中で読めていない順**（＝次に集める価値が高い順）に並ぶこと
//   - 一致率は最新の評価実行から取り、leave-one-out があればそちらを使うこと

// saveLook は指定サイズの画像を持つ履歴エントリを作る
func saveLook(t *testing.T, id string, w, h int) {
	t.Helper()
	os.MkdirAll(dataDir, 0755)
	raw := testImage(t, w, h, color.RGBA{30, 30, 40, 255})
	if err := os.WriteFile(filepath.Join(dataDir, id+".png"), raw, 0644); err != nil {
		t.Fatal(err)
	}
}

func TestCoverageGroupsByLook(t *testing.T) {
	chdirTemp(t)
	saveLook(t, "1111111111111111", 40, 30)
	saveLook(t, "2222222222222222", 40, 30)
	saveLook(t, "3333333333333333", 50, 30)
	h := &HistoryData{Entries: []HistoryEntry{
		{ID: "1111111111111111"}, {ID: "2222222222222222"}, {ID: "3333333333333333"},
	}}

	rep := coverageReport(h, nil)
	if len(rep.Groups) != 2 {
		t.Fatalf("見た目は 2 通りのはず: %+v", rep.Groups)
	}
	byLook := map[string]LookGroup{}
	for _, g := range rep.Groups {
		byLook[g.Look] = g
	}
	if g := byLook["40x30"]; g.Count != 2 || len(g.IDs) != 2 {
		t.Errorf("40x30 は 2 枚のはず: %+v", g)
	}
	if g := byLook["50x30"]; g.Count != 1 {
		t.Errorf("50x30 は 1 枚のはず: %+v", g)
	}
	// 評価がまだ無いので一致率は「未評価」
	for _, g := range rep.Groups {
		if g.Rate != -1 {
			t.Errorf("%s: 評価が無いのに一致率が入っている: %v", g.Look, g.Rate)
		}
		if !g.Thin {
			t.Errorf("%s: %d 枚なら薄いはず", g.Look, g.Count)
		}
	}
}

// **並び順がこの機能の中身。** 薄いものを先に、その中で読めていない順。
// 1 枚しかなくてもよく読めている見た目は下へ回る（足す価値が薄いので）。
func TestCoverageOrdersThinAndWeakFirst(t *testing.T) {
	chdirTemp(t)
	// 40x30: 1枚・成績が悪い / 50x30: 1枚・成績が良い / 60x30: 3枚（厚い）・成績が悪い
	saveLook(t, "1111111111111111", 40, 30)
	saveLook(t, "2222222222222222", 50, 30)
	saveLook(t, "3333333333333333", 60, 30)
	saveLook(t, "4444444444444444", 60, 30)
	saveLook(t, "5555555555555555", 60, 30)
	h := &HistoryData{Entries: []HistoryEntry{
		{ID: "1111111111111111"}, {ID: "2222222222222222"},
		{ID: "3333333333333333"}, {ID: "4444444444444444"}, {ID: "5555555555555555"},
	}}
	metrics := func(ok int) EvalMetrics { return EvalMetrics{Cells: 81, OK: ok} }
	runs := []EvalRun{{ID: "aaaa", Entries: []EvalEntry{
		{ID: "1111111111111111", Manual: metrics(60)},
		{ID: "2222222222222222", Manual: metrics(81)},
		{ID: "3333333333333333", Manual: metrics(50)},
		{ID: "4444444444444444", Manual: metrics(50)},
		{ID: "5555555555555555", Manual: metrics(50)},
	}}}

	rep := coverageReport(h, runs)
	got := []string{rep.Groups[0].Look, rep.Groups[1].Look, rep.Groups[2].Look}
	want := []string{"40x30", "50x30", "60x30"}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("並び順が違う: %v (期待 %v)", got, want)
		}
	}
	if rep.Groups[2].Thin {
		t.Error("3 枚あるものを薄い扱いにしている")
	}
	if r := rep.Groups[0].Rate; r < 0.73 || r > 0.75 {
		t.Errorf("一致率が合わない: %v", r)
	}
}

// **leave-one-out があるならそちらを使う。** 学習済みの局面をそのまま測った
// 数字は高く出るので、薄い見た目でも良い成績に見えてしまう
func TestCoveragePrefersHoldout(t *testing.T) {
	chdirTemp(t)
	saveLook(t, "1111111111111111", 40, 30)
	h := &HistoryData{Entries: []HistoryEntry{{ID: "1111111111111111"}}}
	loo := EvalMetrics{Cells: 81, OK: 40}
	runs := []EvalRun{{ID: "aaaa", Holdout: true, Entries: []EvalEntry{{
		ID:            "1111111111111111",
		Manual:        EvalMetrics{Cells: 81, OK: 81},
		ManualHoldout: &loo,
	}}}}

	rep := coverageReport(h, runs)
	if r := rep.Groups[0].Rate; r > 0.5 {
		t.Errorf("leave-one-out ではなく学習済みの数字を使っている: %v", r)
	}
	if rep.Run == nil || !rep.Run.Holdout {
		t.Error("成績の出どころ（実行）が付いていない")
	}
}

// 画像が消えている履歴エントリは数えない（見た目が分からないため）
func TestCoverageSkipsMissingImage(t *testing.T) {
	chdirTemp(t)
	h := &HistoryData{Entries: []HistoryEntry{{ID: "1111111111111111"}}}
	if rep := coverageReport(h, nil); len(rep.Groups) != 0 {
		t.Errorf("画像が無いのに数えている: %+v", rep.Groups)
	}
}

func TestCoverageHandlerReturnsJSON(t *testing.T) {
	chdirTemp(t)
	saveLook(t, "1111111111111111", 40, 30)
	saveHistoryFile(&HistoryData{Entries: []HistoryEntry{{ID: "1111111111111111"}}})

	w := httptest.NewRecorder()
	handleCoverage(w, httptest.NewRequest(http.MethodGet, "/api/coverage", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("status %d: %s", w.Code, w.Body.String())
	}
	var rep CoverageReport
	if err := json.Unmarshal(w.Body.Bytes(), &rep); err != nil {
		t.Fatal(err)
	}
	if rep.Target != lookTarget || len(rep.Groups) != 1 || rep.Groups[0].Look != "40x30" {
		t.Fatalf("中身が違う: %+v", rep)
	}
}
