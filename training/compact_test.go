package training

import (
	"encoding/json"
	"image"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"github.com/ShinteLab/suteme"
)

func writeSamples(t *testing.T, path string, samples []suteme.TrainingSample) {
	t.Helper()
	if err := suteme.SaveTrainingData(path, &suteme.TrainingData{Samples: samples}); err != nil {
		t.Fatal(err)
	}
}

// TestBuildCompactKeepsEmpty は配布用の間引きが**空マスを残す**ことを確かめる。
//
// **向き照合用（`BuildOrientData`）と逆。** あちらは `PieceDistance` が
// 空を読み飛ばすので落とすが、こちらは k-NN 本体なので
// `ClassEmpty` との一致で空を確定する（落とすと空判定が分類器頼みになる）。
func TestBuildCompactKeepsEmpty(t *testing.T) {
	var samples []suteme.TrainingSample
	for i := 0; i < 40; i++ {
		samples = append(samples, sample(0, float64(i)))
	}
	for i := 0; i < 40; i++ {
		samples = append(samples, sample(suteme.ClassEmpty, float64(100+i)))
	}

	got := BuildCompactData(samples, 10)
	counts := map[int]int{}
	for _, s := range got {
		counts[s.Label]++
	}
	if counts[0] != 10 {
		t.Errorf("歩が %d 件（上限 10）", counts[0])
	}
	if counts[suteme.ClassEmpty] != 10 {
		t.Errorf("空マスが %d 件（10 件残るはず）", counts[suteme.ClassEmpty])
	}
	if got := BuildOrientData(samples, 10); len(got) != 10 {
		t.Errorf("向き照合用は空を落として 10 件のはずが %d 件", len(got))
	}
}

// TestExportCompactWritesSet は dist/ に配布セットが出ること、
// **手元の学習データを上書きしない**ことを確かめる。
func TestExportCompactWritesSet(t *testing.T) {
	chdirTemp(t)

	var samples []suteme.TrainingSample
	for i := 0; i < 30; i++ {
		samples = append(samples, sample(i%3, float64(i)))
	}
	writeSamples(t, suteme.DefaultDataFile, samples)
	// 帯の入力は StripInput で作る（長さが決まっているので手で組まない）
	strip := image.NewGray(image.Rect(0, 0, 90, 10))
	for i := range strip.Pix {
		strip.Pix[i] = uint8(i % 251)
	}
	in := suteme.StripInput(strip, strip.Bounds(), false)
	if in == nil {
		t.Fatal("帯の入力が作れない")
	}
	if err := suteme.SaveStripData(stripFile, []suteme.StripSample{
		{Input: in, Board: true},
		{Input: in, Board: false},
	}); err != nil {
		t.Fatal(err)
	}

	files, err := ExportCompact(2)
	if err != nil {
		t.Fatal(err)
	}
	if len(files) != 2 {
		t.Fatalf("書き出したファイルが %d 個（学習データ + 帯の 2 つのはず）", len(files))
	}

	// 間引いたものが dist/ に、正式名で入っている
	dst := filepath.Join(distDir, suteme.DefaultDataFile)
	got, err := suteme.LoadTrainingData(dst)
	if err != nil {
		t.Fatalf("%s が読めない: %v", dst, err)
	}
	if len(got.Samples) != 6 { // 3クラス × 2
		t.Errorf("%s に %d 件（3クラス × 2 = 6 のはず）", dst, len(got.Samples))
	}

	// **手元の全件はそのまま**
	orig, err := suteme.LoadTrainingData(suteme.DefaultDataFile)
	if err != nil {
		t.Fatal(err)
	}
	if len(orig.Samples) != 30 {
		t.Errorf("手元の学習データが %d 件に変わっている（30 件のままのはず）", len(orig.Samples))
	}

	// 帯はそのまま複製されている
	strips, err := suteme.LoadStripData(filepath.Join(distDir, stripFile))
	if err != nil {
		t.Fatalf("帯が読めない: %v", err)
	}
	if len(strips) != 2 {
		t.Errorf("帯が %d 本（2 本のはず）", len(strips))
	}
}

// TestExportCompactWithoutStrip は帯が無くても書き出せることを確かめる
// （判定器が無くても検出は動くので、そこで止める理由が無い）。
func TestExportCompactWithoutStrip(t *testing.T) {
	chdirTemp(t)
	writeSamples(t, suteme.DefaultDataFile, []suteme.TrainingSample{sample(0, 1), sample(1, 2)})

	files, err := ExportCompact(1)
	if err != nil {
		t.Fatal(err)
	}
	if len(files) != 1 {
		t.Errorf("帯が無いのに %d ファイル書き出した", len(files))
	}
}

// TestExportCompactRejectsBadInput は入口の断りを確かめる
func TestExportCompactRejectsBadInput(t *testing.T) {
	chdirTemp(t)

	// 学習データが無い
	if _, err := ExportCompact(10); err == nil {
		t.Error("学習データが無いのにエラーにならない")
	}
	writeSamples(t, suteme.DefaultDataFile, []suteme.TrainingSample{sample(0, 1)})
	if _, err := ExportCompact(0); err == nil {
		t.Error("1クラス 0 件を断っていない")
	}
}

// TestExportHandler は API の入口を確かめる（本文なしで既定値が効くこと）
func TestExportHandler(t *testing.T) {
	chdirTemp(t)
	writeSamples(t, suteme.DefaultDataFile, []suteme.TrainingSample{sample(0, 1), sample(1, 2)})

	req := httptest.NewRequest(http.MethodPost, "/api/export", nil)
	w := httptest.NewRecorder()
	handleExport(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("status %d: %s", w.Code, w.Body.String())
	}
	var res struct {
		Dir      string `json:"dir"`
		PerClass int    `json:"per_class"`
		Files    []struct {
			Name    string `json:"name"`
			Samples int    `json:"samples"`
		} `json:"files"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &res); err != nil {
		t.Fatal(err)
	}
	if res.PerClass != CompactPerClass {
		t.Errorf("既定の件数が使われていない: %d", res.PerClass)
	}
	if res.Dir != distDir || len(res.Files) == 0 {
		t.Errorf("応答が変: %+v", res)
	}
	// **受け取る側がそのまま使える名前で出ること**（LoadPredictor は決め打ち）
	if res.Files[0].Name != suteme.DefaultDataFile {
		t.Errorf("ファイル名が %s（%s のはず）", res.Files[0].Name, suteme.DefaultDataFile)
	}

	// GET は受けない
	w = httptest.NewRecorder()
	handleExport(w, httptest.NewRequest(http.MethodGet, "/api/export", nil))
	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("GET が %d で通った", w.Code)
	}
}
