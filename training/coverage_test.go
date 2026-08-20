package training

import (
	"bytes"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// 盤の見た目ごとの厚み（coverage.go）。
//
// 守りたいのは 4 つ。
//   - **人が付けた見た目（`HistoryEntry.Look`）が最優先**。機械の候補で上書きしない
//   - 未判定のものだけ署名で候補を出し、似ていないものを混ぜないこと
//   - **枚数の多い順**（＝もう撮らなくていい見た目が上）に並ぶこと
//   - 一致率は最新の評価実行から取り、leave-one-out があればそちらを使うこと

// lookImage は「見た目」の違う盤の画像を作る。
//
// 署名は空マスの見た目（木目・格子線）から作るので、**一様な板では駄目**
// （`CellToInput` が標準化するので、色だけ違う平坦な画像は同じ署名になる）。
// pattern でマスの中の模様を変えて別の見た目にする。
func lookImage(t *testing.T, pattern int) []byte {
	t.Helper()
	const size = 180 // 9 マス × 20px
	img := image.NewRGBA(image.Rect(0, 0, size, size))
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			v := 200
			cx, cy := x%20, y%20 // マスの中での位置
			switch pattern {
			case 0: // マスの上のほうが暗い盤
				if cy < 7 {
					v = 120
				}
			case 1: // 同じ形（少しだけ濃さが違う＝同じ見た目のつもり）
				if cy < 7 {
					v = 110
				}
			case 2: // マスの左のほうが暗い盤（別の見た目）
				if cx < 7 {
					v = 120
				}
			}
			if x%20 == 0 || y%20 == 0 { // 格子線
				v = 40
			}
			img.Set(x, y, color.RGBA{uint8(v), uint8(v), uint8(v), 255})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// saveLook は空の盤（9x9 すべて空マス）の履歴エントリを作る
func saveLook(t *testing.T, id string, pattern int) HistoryEntry {
	t.Helper()
	os.MkdirAll(dataDir, 0755)
	if err := os.WriteFile(filepath.Join(dataDir, id+".png"), lookImage(t, pattern), 0644); err != nil {
		t.Fatal(err)
	}
	return HistoryEntry{
		ID:          id,
		SFEN:        "9/9/9/9/9/9/9/9/9 b - 1",
		BoardBounds: &BoardBounds{0, 0, 180, 180},
	}
}

// **人が付けた見た目が正解。** 機械の候補はそれを上書きしない
func TestCoverageUsesHumanLook(t *testing.T) {
	chdirTemp(t)
	a := saveLook(t, "1111111111111111", 0)
	b := saveLook(t, "2222222222222222", 1) // 署名は a に近い
	a.Look, b.Look = "中継A", "ゲーム画面B"        // だが人は別物だと判定した
	c := saveLook(t, "3333333333333333", 2)
	c.Look = "中継A"

	rep := coverageReport(&HistoryData{Entries: []HistoryEntry{a, b, c}}, nil)
	if len(rep.Groups) != 2 {
		t.Fatalf("人の判定どおり 2 つに分かれるはず: %+v", rep.Groups)
	}
	for _, g := range rep.Groups {
		if g.Proposed {
			t.Errorf("%s: 人が付けた見た目を候補扱いにしている", g.Look)
		}
		if g.Look == "中継A" && g.Count != 2 {
			t.Errorf("中継A は 2 枚のはず: %+v", g)
		}
	}
	if rep.Named != 3 || rep.Total != 3 {
		t.Errorf("判定済みの数が合わない: named=%d total=%d", rep.Named, rep.Total)
	}
}

// 未判定のものは署名で候補を出す。**似ていないものを混ぜない**
func TestCoverageProposesBySignature(t *testing.T) {
	chdirTemp(t)
	entries := []HistoryEntry{
		saveLook(t, "1111111111111111", 0),
		saveLook(t, "2222222222222222", 1),
		saveLook(t, "3333333333333333", 2),
	}
	rep := coverageReport(&HistoryData{Entries: entries}, nil)
	if rep.Named != 0 {
		t.Fatalf("まだ誰も判定していない: %+v", rep)
	}
	var pair, alone int
	for _, g := range rep.Groups {
		if !g.Proposed {
			t.Errorf("未判定なのに候補になっていない: %+v", g)
		}
		switch g.Count {
		case 2:
			pair++
		case 1:
			alone++
		}
	}
	if pair != 1 || alone != 1 {
		t.Fatalf("似た 2 枚が候補にまとまり、違う 1 枚は別になるはず: %+v", rep.Groups)
	}
}

// **並び順がこの機能の中身。** 枚数の多い順＝「もう撮らなくていい見た目」が上。
// 目標は種類を増やすことなので、薄い見た目を厚くする案内はしない
func TestCoverageOrdersByCount(t *testing.T) {
	chdirTemp(t)
	mk := func(id string, look string) HistoryEntry {
		e := saveLook(t, id, 0)
		e.Look = look
		return e
	}
	entries := []HistoryEntry{
		mk("1111111111111111", "薄くて弱い"),
		mk("2222222222222222", "薄いがよく読めている"),
		mk("3333333333333333", "厚いが弱い"),
		mk("4444444444444444", "厚いが弱い"),
		mk("5555555555555555", "厚いが弱い"),
	}
	metrics := func(ok int) EvalMetrics { return EvalMetrics{Cells: 81, OK: ok} }
	runs := []EvalRun{{ID: "aaaa", Entries: []EvalEntry{
		{ID: "1111111111111111", Manual: metrics(60)},
		{ID: "2222222222222222", Manual: metrics(81)},
		{ID: "3333333333333333", Manual: metrics(50)},
		{ID: "4444444444444444", Manual: metrics(50)},
		{ID: "5555555555555555", Manual: metrics(50)},
	}}}

	rep := coverageReport(&HistoryData{Entries: entries}, runs)
	// **枚数の多い順**（＝もう撮らなくていい見た目が上）。
	// 薄い見た目を厚くしても「どんな盤でも読む」目標には効かないので、
	// 「薄い順に潰す」並びにはしない
	want := []string{"厚いが弱い", "薄くて弱い", "薄いがよく読めている"}
	for i, w := range want {
		if rep.Groups[i].Look != w {
			t.Fatalf("並び順が違う: %d 番目が %q（期待 %q）", i, rep.Groups[i].Look, w)
		}
	}
	if !rep.Groups[0].Excess {
		t.Error("3 枚あるものを「積みすぎ」にしていない")
	}
}

// **leave-one-out があるならそちらを使う。** 学習済みの局面をそのまま測った
// 数字は高く出るので、薄い見た目でも良い成績に見えてしまう
func TestCoveragePrefersHoldout(t *testing.T) {
	chdirTemp(t)
	e := saveLook(t, "1111111111111111", 0)
	e.Look = "中継A"
	loo := EvalMetrics{Cells: 81, OK: 40}
	runs := []EvalRun{{ID: "aaaa", Holdout: true, Entries: []EvalEntry{{
		ID:            "1111111111111111",
		Manual:        EvalMetrics{Cells: 81, OK: 81},
		ManualHoldout: &loo,
	}}}}

	rep := coverageReport(&HistoryData{Entries: []HistoryEntry{e}}, runs)
	if r := rep.Groups[0].Rate; r > 0.5 {
		t.Errorf("leave-one-out ではなく学習済みの数字を使っている: %v", r)
	}
	if rep.Run == nil || !rep.Run.Holdout {
		t.Error("成績の出どころ（実行）が付いていない")
	}
}

// 画像が消えている履歴エントリは候補に出せない（見た目が分からないため）。
// **人が名前を付けてあれば数える**（判定はもう済んでいる）
func TestCoverageSkipsMissingImage(t *testing.T) {
	chdirTemp(t)
	entries := []HistoryEntry{{ID: "1111111111111111"}}
	if rep := coverageReport(&HistoryData{Entries: entries}, nil); len(rep.Groups) != 0 {
		t.Errorf("画像が無いのに候補にしている: %+v", rep.Groups)
	}
	entries[0].Look = "中継A"
	if rep := coverageReport(&HistoryData{Entries: entries}, nil); len(rep.Groups) != 1 {
		t.Errorf("人が付けた見た目は画像が無くても数えるべき: %+v", rep.Groups)
	}
}

func postLook(t *testing.T, body string) *httptest.ResponseRecorder {
	t.Helper()
	w := httptest.NewRecorder()
	handleLook(w, httptest.NewRequest(http.MethodPost, "/api/look", strings.NewReader(body)))
	return w
}

// 見た目の付け外し。**空文字で未判定に戻せる**（間違えたときに直せないと困る）
func TestLookAssignAndClear(t *testing.T) {
	chdirTemp(t)
	saveLook(t, "1111111111111111", 0)
	saveLook(t, "2222222222222222", 1)
	saveHistoryFile(&HistoryData{Entries: []HistoryEntry{
		{ID: "1111111111111111"}, {ID: "2222222222222222"},
	}})

	w := postLook(t, `{"ids":["1111111111111111","2222222222222222"],"look":"中継A"}`)
	if w.Code != http.StatusOK {
		t.Fatalf("status %d: %s", w.Code, w.Body.String())
	}
	for _, e := range loadHistory().Entries {
		if e.Look != "中継A" {
			t.Fatalf("%s に見た目が付いていない: %+v", e.ID, e)
		}
	}
	if w := postLook(t, `{"ids":["2222222222222222"],"look":""}`); w.Code != http.StatusOK {
		t.Fatalf("status %d", w.Code)
	}
	for _, e := range loadHistory().Entries {
		want := "中継A"
		if e.ID == "2222222222222222" {
			want = ""
		}
		if e.Look != want {
			t.Errorf("%s: 見た目が %q（期待 %q）", e.ID, e.Look, want)
		}
	}
	if w := postLook(t, `{"ids":[],"look":"x"}`); w.Code != http.StatusBadRequest {
		t.Errorf("ids が空なら 400 のはず: %d", w.Code)
	}
}

func TestCoverageHandlerReturnsJSON(t *testing.T) {
	chdirTemp(t)
	e := saveLook(t, "1111111111111111", 0)
	e.Look = "中継A"
	saveHistoryFile(&HistoryData{Entries: []HistoryEntry{e}})

	w := httptest.NewRecorder()
	handleCoverage(w, httptest.NewRequest(http.MethodGet, "/api/coverage", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("status %d: %s", w.Code, w.Body.String())
	}
	var rep CoverageReport
	if err := json.Unmarshal(w.Body.Bytes(), &rep); err != nil {
		t.Fatal(err)
	}
	if rep.Target != lookEnough || len(rep.Groups) != 1 || rep.Groups[0].Look != "中継A" {
		t.Fatalf("中身が違う: %+v", rep)
	}
}
