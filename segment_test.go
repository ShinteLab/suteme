package suteme

import (
	"image"
	"image/color"
	"math"
	"testing"
)

const (
	sceneCellW = 40
	sceneCellH = 44 // ≒ boardAspect
	sceneX     = 200
	sceneY     = 120
	sceneW     = 900
	sceneH     = 700
)

// makeSceneImage は「盤の周りに盤以外のものが写っている」画像を作る。
// 中継画像の構図（画像の全幅にわたる罫線・駒台）を模したもので、
// 投影を全幅で取ると盤以外の横線に負ける。
func makeSceneImage() *image.Gray {
	img := image.NewGray(image.Rect(0, 0, sceneW, sceneH))
	for y := 0; y < sceneH; y++ {
		for x := 0; x < sceneW; x++ {
			img.SetGray(x, y, color.Gray{Y: 150}) // 畳
		}
	}
	hline := func(y, x1, x2 int, v uint8) {
		for i := 0; i < 3; i++ {
			for x := x1; x < x2; x++ {
				img.SetGray(x, y+i, color.Gray{Y: v})
			}
		}
	}
	// 盤以外: 画像の全幅にわたる罫線（対局時計バー・タイトルパネルの類）
	hline(40, 0, sceneW, 40)
	hline(80, 0, sceneW, 40)
	hline(640, 0, sceneW, 40)
	hline(670, 0, sceneW, 40)
	// 盤以外: 駒台（盤とは違う幅の矩形）
	for y := 300; y < 420; y++ {
		for x := 760; x < 880; x++ {
			img.SetGray(x, y, color.Gray{Y: 200})
		}
	}
	hline(300, 760, 880, 60)
	hline(417, 760, 880, 60)

	// 盤
	for y := sceneY; y < sceneY+sceneCellH*9; y++ {
		for x := sceneX; x < sceneX+sceneCellW*9; x++ {
			img.SetGray(x, y, color.Gray{Y: testBoardV})
		}
	}
	for i := 0; i <= 9; i++ {
		y := sceneY + i*sceneCellH
		for x := sceneX; x < sceneX+sceneCellW*9; x++ {
			img.SetGray(x, y, color.Gray{Y: testLineV})
		}
		x := sceneX + i*sceneCellW
		for y := sceneY; y < sceneY+sceneCellH*9; y++ {
			img.SetGray(x, y, color.Gray{Y: testLineV})
		}
	}
	return img
}

// 盤の格子線は同じ x 範囲で始まり終わるので、最上位のクラスタが盤の幅になる。
// 盤以外の線（全幅の罫線・駒台）のほうが本数は多くても構わない
func TestSegmentSpansFindsBoardWidth(t *testing.T) {
	img := makeSceneImage()
	spans := segmentSpans(BoxBlur(img, 2), segMaxCandidates)
	if len(spans) == 0 {
		t.Fatal("候補が無い")
	}
	want1, want2 := sceneX, sceneX+sceneCellW*9
	found := false
	for i, s := range spans {
		t.Logf("span%d = %d..%d", i, s[0], s[1])
		if absInt(s[0]-want1) <= sceneCellW/2 && absInt(s[1]-want2) <= sceneCellW/2 {
			found = true
		}
	}
	if !found {
		t.Errorf("盤の幅 %d..%d に近い候補が無い", want1, want2)
	}
}

// 縦線は transposeGray で同じ処理に通す
func TestSegmentSpansFindsBoardHeight(t *testing.T) {
	img := makeSceneImage()
	spans := segmentSpans(transposeGray(BoxBlur(img, 2)), segMaxCandidates)
	want1, want2 := sceneY, sceneY+sceneCellH*9
	found := false
	for i, s := range spans {
		t.Logf("span%d = %d..%d", i, s[0], s[1])
		if absInt(s[0]-want1) <= sceneCellH/2 && absInt(s[1]-want2) <= sceneCellH/2 {
			found = true
		}
	}
	if !found {
		t.Errorf("盤の高さ %d..%d に近い候補が無い", want1, want2)
	}
}

// 盤以外が写っていても DetectBoard が盤を捉えること。
// 画像全体で投影を取ると盤外の罫線に引かれるので、
// 線分で領域を絞る処理が効いているかがここで分かる
func TestDetectBoardInScene(t *testing.T) {
	img := makeSceneImage()
	br := DetectBoard(img)
	if br == nil {
		t.Fatal("検出できなかった")
	}
	dx := float64(br.Bounds.Min.X-sceneX) / sceneCellW
	dy := float64(br.Bounds.Min.Y-sceneY) / sceneCellH
	dw := float64(br.Bounds.Dx()-sceneCellW*9) / sceneCellW
	dh := float64(br.Bounds.Dy()-sceneCellH*9) / sceneCellH
	worst := math.Max(math.Max(math.Abs(dx), math.Abs(dy)), math.Max(math.Abs(dw), math.Abs(dh)))
	conf := ValidateBoard(img, br)
	t.Logf("dx=%+.2f dy=%+.2f dw=%+.2f dh=%+.2f conf=%.2f", dx, dy, dw, dh, conf)
	if worst > 0.5 {
		t.Errorf("盤面から %.2fマスずれている: %v", worst, br.Bounds)
	}
	if conf < minBoardConfidence {
		t.Errorf("座標は合っているのに conf=%.2f で棄却される", conf)
	}
}

// 縦横比が将棋盤と違う候補は落とす（ROI の中で UI の格子を拾ったとき用）
func TestPlausibleAspect(t *testing.T) {
	tests := []struct {
		name string
		w, h int
		want bool
	}{
		{"規格どおり", 360, 396, true},
		{"実測の下限あたり", 360, 376, true},
		{"正方形", 360, 360, true},
		{"横長すぎ", 360, 260, false},
		{"縦長すぎ", 360, 520, false},
	}
	for _, tt := range tests {
		got := plausibleAspect(BoardRegionFromRect(0, 0, tt.w, tt.h))
		if got != tt.want {
			t.Errorf("%s: %dx%d = %v, want %v", tt.name, tt.w, tt.h, got, tt.want)
		}
	}
}
