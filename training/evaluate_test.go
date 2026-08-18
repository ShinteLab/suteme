package training

import (
	"encoding/json"
	"image"
	"image/color"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ShinteLab/suteme"
)

// gradeBoard の内訳。誤りの種類ごとに 1 件ずつ作って数え方を確かめる。
//
// **「空→駒」「駒→空」「向きの反転」「駒種違い」を混ぜない**のがこの関数の要で、
// ここが崩れると評価タブの数字を見ても何が悪くなったのか分からなくなる。
func TestGradeBoardCounts(t *testing.T) {
	var want, got [9][9]string
	for r := range want {
		for c := range want[r] {
			want[r][c] = suteme.EmptyLabel
			got[r][c] = suteme.EmptyLabel
		}
	}
	// 正解: 歩(先手) / 香(後手) / 銀(先手) / 金(先手) / 空
	want[0][0], got[0][0] = "P", "P"    // 一致
	want[0][1], got[0][1] = "-L", "L"   // 向きの反転
	want[0][2], got[0][2] = "S", "G"    // 駒種違い
	want[0][3], got[0][3] = "N", "none" // 駒→空
	want[0][4], got[0][4] = "none", "B" // 空→駒

	m := gradeBoard(&want, &got)
	if m.Boards != 1 || m.Cells != 81 {
		t.Fatalf("Boards=%d Cells=%d", m.Boards, m.Cells)
	}
	if m.PieceCells != 4 || m.PieceOK != 1 {
		t.Errorf("駒マス %d（期待 4） / 正解 %d（期待 1）", m.PieceCells, m.PieceOK)
	}
	if m.OrientFlip != 1 || m.TypeMiss != 1 || m.PieceAsEmpty != 1 || m.EmptyAsPiece != 1 {
		t.Errorf("反転=%d 駒種違い=%d 駒→空=%d 空→駒=%d（すべて 1 のはず）",
			m.OrientFlip, m.TypeMiss, m.PieceAsEmpty, m.EmptyAsPiece)
	}
	// 一致したのは 81 - 5 の空マスと P の 1 件
	if m.OK != 77 {
		t.Errorf("一致 %d（期待 77）", m.OK)
	}
	if m.BoardsPerfect != 0 {
		t.Errorf("誤りがあるのに完全一致に数えている")
	}

	// 全部合っていれば完全一致
	if m := gradeBoard(&want, &want); m.OK != 81 || m.BoardsPerfect != 1 {
		t.Errorf("完全一致のはずが OK=%d perfect=%d", m.OK, m.BoardsPerfect)
	}
}

// **盤面を検出できなかった局面は 0/81 として数える。**
// 「認識できた局面だけの平均」にすると、盤を見つけられない画像が増えるほど
// 数字が良くなるという逆転が起きる。手入力の座標なしで動くかを測るのが目的なので
// 検出の失敗も失点として残す。
func TestMissedBoardCountsAsZero(t *testing.T) {
	want, err := boardGrid("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL")
	if err != nil {
		t.Fatal(err)
	}
	m := missedBoard(want)
	if m.Cells != 81 || m.OK != 0 || m.NoBoard != 1 || m.Boards != 1 {
		t.Fatalf("%+v", m)
	}
	if m.PieceCells != 40 {
		t.Errorf("駒マス %d（平手なので 40）", m.PieceCells)
	}
	// 認識器を呼んでいないので、誤りの内訳には数えない
	if m.EmptyAsPiece != 0 || m.PieceAsEmpty != 0 || m.OrientFlip != 0 || m.TypeMiss != 0 {
		t.Errorf("検出失敗を認識器の誤りとして数えている: %+v", m)
	}
}

// 検出のずれは px ではなくマス単位で残す（画像の解像度に依らず比べられるように）。
func TestGradeDetectShiftInCells(t *testing.T) {
	b := &BoardBounds{X1: 100, Y1: 200, X2: 100 + 90, Y2: 200 + 180} // 1マス 10x20
	region := image.Rect(105, 200, 105+90, 200+180)
	got := gradeDetect(region, suteme.RegionFromDetect, 0.9, true, b)
	if !got.Found || got.Source != "detect" {
		t.Fatalf("%+v", got)
	}
	if got.DX != 0.5 || got.DY != 0 || got.DW != 0 || got.DH != 0 {
		t.Errorf("dx=%.2f dy=%.2f dw=%.2f dh=%.2f（dx=0.5 のみのはず）", got.DX, got.DY, got.DW, got.DH)
	}
	if got.MaxShift != 0.5 || !got.Aligned {
		t.Errorf("MaxShift=%.2f aligned=%v（境界は合っている扱い）", got.MaxShift, got.Aligned)
	}

	// 1マスずれたら合っていない扱い
	shifted := image.Rect(100, 220, 100+90, 220+180)
	if got := gradeDetect(shifted, suteme.RegionFromDetect, 0.9, true, b); got.DY != 1 || got.Aligned {
		t.Errorf("dy=%.2f aligned=%v（1マスずれ）", got.DY, got.Aligned)
	}

	// **棄却された候補も位置と信頼度を残す。** 「見つからなかった」だけだと
	// 惜しかったのか全然違う場所を見ていたのかが分からず、次の版で直ったかを追えない
	rejected := gradeDetect(shifted, suteme.RegionFromDetect, 0.12, false, b)
	if rejected.Found || !rejected.Candidate {
		t.Errorf("棄却の記録が残っていない: %+v", rejected)
	}
	if rejected.DY != 1 || rejected.Confidence != 0.12 {
		t.Errorf("棄却された候補のずれ・信頼度が落ちている: %+v", rejected)
	}
	// 候補すら出なかった場合
	if none := gradeDetect(image.Rectangle{}, "", 0, false, b); none.Candidate {
		t.Errorf("候補が無いのに Candidate=true: %+v", none)
	}
}

// 評価結果は data/accuracy.json に積む。新しいものが先頭に来ること、
// GET / DELETE が通ることを確かめる。
func TestEvalHistoryRoundTrip(t *testing.T) {
	chdirTemp(t)

	for _, label := range []string{"古い版", "新しい版"} {
		run := &EvalRun{ID: newID(), Label: label, Manual: EvalMetrics{Cells: 81, OK: 80}}
		if err := appendEvalRun(run); err != nil {
			t.Fatal(err)
		}
	}

	req := httptest.NewRequest(http.MethodGet, "/api/evaluations", nil)
	w := httptest.NewRecorder()
	handleEvaluations(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("GET: %d", w.Code)
	}
	var h EvalHistory
	if err := json.NewDecoder(w.Body).Decode(&h); err != nil {
		t.Fatal(err)
	}
	if len(h.Runs) != 2 || h.Runs[0].Label != "新しい版" {
		t.Fatalf("新しい実行が先頭に来ていない: %+v", h.Runs)
	}

	id := h.Runs[0].ID
	req = httptest.NewRequest(http.MethodDelete, "/api/evaluations/"+id, nil)
	w = httptest.NewRecorder()
	handleEvaluations(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("DELETE: %d %s", w.Code, w.Body.String())
	}
	h2 := loadEvalHistory()
	if len(h2.Runs) != 1 || h2.Runs[0].Label != "古い版" {
		t.Fatalf("削除後: %+v", h2.Runs)
	}
}

// 保存済み局面（data/）に対して評価を通す。合否は見ず、
// 評価タブに出るのと同じ数字をログに出すだけ（data/ が無ければスキップ）。
//
//	go test -run TestEvaluateSavedBoards -v ./training
//
// **accuracy.json には書かない。** 記録を残すのは画面からの実行だけにする
// （テストを回すたびに実行履歴が増えると版の比較が読みにくくなる）。
func TestEvaluateSavedBoards(t *testing.T) {
	if testing.Short() {
		t.Skip("-short のためスキップ")
	}
	if !chdirToData(t) {
		t.Skip("data/history.json が無いのでスキップ")
	}
	data, path := loadExistingTrainingData()
	if data == nil {
		t.Skip("学習データが無いのでスキップ")
	}
	kn := suteme.NewKNN(data.Samples)
	if kn == nil {
		t.Skip("k-NN を構築できないのでスキップ")
	}
	modelLock.Lock()
	saved := knn
	knn = kn
	modelLock.Unlock()
	t.Cleanup(func() {
		modelLock.Lock()
		knn = saved
		modelLock.Unlock()
	})

	// leave-one-out はここでは回さない（TestKNNHoldout が測っている）
	run, err := runEvaluation(nil, "test", false)
	if err != nil {
		t.Skipf("評価できません: %v", err)
	}
	d := run.DetectSummary
	t.Logf("学習データ: %s (%d サンプル)", path, kn.Len())
	t.Logf("盤面検出: %d/%d が 0.5マス以内（採用 %d / 画像全体 %d）",
		d.Aligned, d.Boards, d.Found, d.Whole)
	for _, m := range []struct {
		name string
		v    EvalMetrics
	}{{"手動座標", run.Manual}, {"自動座標", run.Auto}} {
		t.Logf("%s: %d/%d = %.1f%% / 完全一致 %d局面 / 検出不可 %d局面 / "+
			"空→駒 %d 駒→空 %d 反転 %d 駒種違い %d",
			m.name, m.v.OK, m.v.Cells, 100*float64(m.v.OK)/float64(m.v.Cells),
			m.v.BoardsPerfect, m.v.NoBoard,
			m.v.EmptyAsPiece, m.v.PieceAsEmpty, m.v.OrientFlip, m.v.TypeMiss)
	}
	for _, e := range run.Entries {
		if e.Error != "" {
			t.Logf("%s: %s", e.ID, e.Error)
			continue
		}
		t.Logf("%s: 手動 %d/81 自動 %d/81 検出 %.2fマス conf=%.2f src=%s",
			e.ID, e.Manual.OK, e.Auto.OK, e.Detect.MaxShift, e.Detect.Confidence, e.Detect.Source)
	}
}

// 評価できない局面（画像が無い・座標が無い）は run に理由つきで残し、
// 全体が失敗にはならないこと。
func TestRunEvaluationSkipsIncomplete(t *testing.T) {
	chdirTemp(t)
	historyMu.Lock()
	saveHistoryFile(&HistoryData{Entries: []HistoryEntry{
		{ID: "aaaa", SFEN: "9/9/9/9/9/9/9/9/9"}, // 盤面座標なし
	}})
	historyMu.Unlock()

	// 推論器が無ければそもそも測れない
	modelLock.Lock()
	savedKNN, savedModel := knn, model
	knn, model = nil, nil
	modelLock.Unlock()
	t.Cleanup(func() {
		modelLock.Lock()
		knn, model = savedKNN, savedModel
		modelLock.Unlock()
	})
	if _, err := runEvaluation(nil, "", false); err == nil ||
		!strings.Contains(err.Error(), "推論器") {
		t.Fatalf("推論器なしでエラーにならない: %v", err)
	}

	modelLock.Lock()
	knn = suteme.NewKNN([]suteme.TrainingSample{
		{Input: make([]float64, suteme.InputSize), Label: suteme.ClassEmpty},
	})
	modelLock.Unlock()

	_, err := runEvaluation(nil, "", false)
	if err == nil || !strings.Contains(err.Error(), "評価できる局面がありません") {
		t.Fatalf("座標の無い局面だけで評価が通った: %v", err)
	}
}

// 保存されている座標が検出結果そのものだったら印を付ける。
//
// **一致は「検出が当たった」証拠ではない。** ikkyoku は盤の座標が無いと
// 登録できないので、API 経由の局面は構造的に「検出できた局面」ばかりになり、
// その座標の多くは検出器の出力そのもの。混ぜて平均すると実力より良く見える
func TestGradeDetectMarksExactAsSelfFulfilling(t *testing.T) {
	b := &BoardBounds{X1: 100, Y1: 200, X2: 190, Y2: 380}
	same := image.Rect(100, 200, 190, 380)
	if got := gradeDetect(same, suteme.RegionFromDetect, 1.0, true, b); !got.Exact {
		t.Errorf("座標が検出結果と同じなのに Exact=false: %+v", got)
	}
	// 1px でも違えば独立した正解として扱う
	off := image.Rect(101, 200, 191, 380)
	if got := gradeDetect(off, suteme.RegionFromDetect, 1.0, true, b); got.Exact {
		t.Errorf("1px ずれているのに Exact=true: %+v", got)
	}
}

// 盤面検出の集計を局面の出所ごとに分ける。
//
// **全件をまとめた比率は実力ではない**（`EvalSourceDetect`）。
// 人が引いた座標（ui）を先頭に置き、版を上げたときに同じ母集団どうしで
// 比べられるようにする
func TestEvalGroupsDetectBySource(t *testing.T) {
	chdirTemp(t)
	entries := []HistoryEntry{
		{ID: "1111111111111111", SFEN: "9/9/9/9/9/9/9/9/9", Source: SourceAPI,
			BoardBounds: &BoardBounds{X1: 10, Y1: 10, X2: 100, Y2: 110}},
		{ID: "2222222222222222", SFEN: "9/9/9/9/9/9/9/9/9", Source: SourceUI,
			BoardBounds: &BoardBounds{X1: 10, Y1: 10, X2: 100, Y2: 110}},
	}
	os.MkdirAll(dataDir, 0755)
	for _, e := range entries {
		if err := os.WriteFile(filepath.Join(dataDir, e.ID+".png"), testImage(t, 120, 130, color.RGBA{180, 150, 100, 255}), 0644); err != nil {
			t.Fatal(err)
		}
	}
	historyMu.Lock()
	saveHistoryFile(&HistoryData{Entries: entries})
	historyMu.Unlock()

	modelLock.Lock()
	saved := knn
	knn = suteme.NewKNN([]suteme.TrainingSample{
		{Input: make([]float64, suteme.InputSize), Label: suteme.ClassEmpty},
	})
	modelLock.Unlock()
	t.Cleanup(func() {
		modelLock.Lock()
		knn = saved
		modelLock.Unlock()
	})

	run, err := runEvaluation(nil, "", false)
	if err != nil {
		t.Fatal(err)
	}
	if len(run.DetectBySource) != 2 {
		t.Fatalf("出所の数が %d（ui と api の 2 つのはず）: %+v", len(run.DetectBySource), run.DetectBySource)
	}
	// 人が引いた座標を先頭に置く（検出の実力に近いのはこちら）
	if run.DetectBySource[0].Source != SourceUI || run.DetectBySource[1].Source != SourceAPI {
		t.Errorf("並び順が違う: %+v", run.DetectBySource)
	}
	for _, d := range run.DetectBySource {
		if d.Boards != 1 {
			t.Errorf("%s の局面数が %d（1 のはず）", d.Source, d.Boards)
		}
	}
	if run.DetectSummary.Boards != 2 {
		t.Errorf("全体の局面数が %d（2 のはず）", run.DetectSummary.Boards)
	}
	// 局面ごとの内訳にも出所を残す（画面のバッジがここを見る）
	for _, e := range run.Entries {
		if e.Source == "" {
			t.Errorf("%s に出所が入っていない", e.ID)
		}
	}
}
