package suteme

import (
	"strings"
	"testing"
)

// Debug が「盤面と判定した矩形・その決め方・使った推論器・マスごとの結果」を
// 揃えて返すこと。呼び出し側はこれだけで座標の問題か認識器の問題かを切り分ける。
func TestRecognizeDebug(t *testing.T) {
	img := makeBoardImage([]testPiece{
		{row: 2, col: 3, gote: true},
		{row: 6, col: 5},
	})
	p := &stubPredictor{class: LabelToClass("K")}

	r, err := Recognize(img, WithPredictor(p))
	if err != nil {
		t.Fatalf("Recognize: %v", err)
	}
	d := r.Debug
	if d == nil {
		t.Fatal("Debug が nil")
	}
	if d.Region.Empty() {
		t.Error("Region が空")
	}
	if d.ImageBounds != img.Bounds() {
		t.Errorf("ImageBounds = %v, want %v", d.ImageBounds, img.Bounds())
	}
	if d.Confidence != r.Confidence {
		t.Errorf("Confidence = %v, want %v", d.Confidence, r.Confidence)
	}
	if d.RegionSource != RegionFromDetect && d.RegionSource != RegionFromWholeImage {
		t.Errorf("RegionSource = %q", d.RegionSource)
	}
	if d.BoardColor != testBoardV {
		t.Errorf("BoardColor = %d, want %d", d.BoardColor, testBoardV)
	}
	if len(d.Cells) != 81 {
		t.Fatalf("len(Cells) = %d, want 81", len(d.Cells))
	}

	// 駒のあるマスは分類・駒種・確信度が入り、空マスは Class = -1 のまま
	gote := d.Cell(2, 3)
	if gote.Category != CellPieceDown || gote.Piece != "k" {
		t.Errorf("(2,3) = %+v, want 後手の k", *gote)
	}
	if gote.Class != LabelToClass("K") || gote.Confidence != 1.0 {
		t.Errorf("(2,3) の推論結果 = class %d conf %v", gote.Class, gote.Confidence)
	}
	if sente := d.Cell(6, 5); sente.Category != CellPieceUp || sente.Piece != "K" {
		t.Errorf("(6,5) = %+v, want 先手の K", *sente)
	}
	empty := d.Cell(0, 0)
	if empty.Category != CellEmpty || empty.Piece != "" || empty.Class != -1 {
		t.Errorf("(0,0) = %+v, want 空マス", *empty)
	}
	if empty.Rect != d.Cells[0].Rect || empty.Rect.Empty() {
		t.Errorf("(0,0) の Rect = %v", empty.Rect)
	}
	if d.Cell(9, 0) != nil || d.Cell(-1, 0) != nil {
		t.Error("範囲外の Cell が nil でない")
	}

	// 推論器の素性。WithPredictor で直接渡した場合は型名だけ
	if !strings.Contains(d.Predictor.Kind, "stubPredictor") {
		t.Errorf("Predictor.Kind = %q", d.Predictor.Kind)
	}
	if !strings.Contains(d.String(), "region=") {
		t.Errorf("String() = %q", d.String())
	}
	if lines := strings.Count(d.Dump(), "\n"); lines != 10 {
		t.Errorf("Dump() の行数 = %d, want 10（要約 + 9段）", lines)
	}
}

// WithRect で座標を渡した場合は RegionSource がその旨を示し、
// 矩形もそのまま載ること（信頼度でゲートしないので低くても採用される）。
func TestRecognizeDebugRegionFromOption(t *testing.T) {
	img := makeBoardImage(nil)
	size := testCell * 9

	r, err := Recognize(img, WithPredictor(&stubPredictor{}), WithRect(0, 0, size, size))
	if err != nil {
		t.Fatalf("Recognize: %v", err)
	}
	if r.Debug.RegionSource != RegionFromOption {
		t.Errorf("RegionSource = %q, want %q", r.Debug.RegionSource, RegionFromOption)
	}
	if got := r.Debug.Region; got.Dx() != size || got.Dy() != size {
		t.Errorf("Region = %v", got)
	}
}

// k-NN は読み込み元とサンプル数を返す（どのデータで認識したかの切り分け用）。
func TestKNNDebug(t *testing.T) {
	kn := NewKNN([]TrainingSample{{Input: make([]float64, inputSize), Label: 0}})
	if kn == nil {
		t.Fatal("NewKNN = nil")
	}
	kn.source = "training_data_v3.json"

	d := kn.Debug()
	if d.Kind != "knn" || d.Source != "training_data_v3.json" {
		t.Errorf("Debug = %+v", d)
	}
	if !strings.Contains(d.Detail, "samples=1") {
		t.Errorf("Detail = %q", d.Detail)
	}
	if !strings.Contains(d.String(), "knn(") {
		t.Errorf("String = %q", d.String())
	}
}

// 低確信度のマスだけを拾えること（怪しいマスの絞り込み用）
func TestDebugLowConfidenceCells(t *testing.T) {
	d := &Debug{Cells: []CellDebug{
		{Row: 0, Col: 0, Category: CellEmpty, Confidence: 0},
		{Row: 0, Col: 1, Category: CellPieceUp, Confidence: 0.9},
		{Row: 0, Col: 2, Category: CellPieceDown, Confidence: 0.3},
	}}
	got := d.LowConfidenceCells(0.5)
	if len(got) != 1 || got[0].Col != 2 {
		t.Errorf("LowConfidenceCells = %+v, want (0,2) のみ", got)
	}
}
