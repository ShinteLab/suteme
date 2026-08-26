package suteme

import (
	"image"
	"math"
	"testing"
)

// cropTo は画像の一部を原点 (0,0) に寄せて切り出す（キャプチャの枠を変える代用）
func cropTo(src image.Image, r image.Rectangle) image.Image {
	r = r.Intersect(src.Bounds())
	dst := image.NewGray(image.Rect(0, 0, r.Dx(), r.Dy()))
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			dst.Set(x-r.Min.X, y-r.Min.Y, src.At(x, y))
		}
	}
	return dst
}

// **同じ盤なら、余白の取り方を変えても同じ位置に切り出せること。**
//
// これは精度の話ではなく再現性の話で、suteme が最も守りたい性質
// （「こう撮ると認識率が高い」を無くす）。実測では同一画素の 2 枚で
// 外枠が 5px 違い、それだけで認識が 2マス変わっていた（`SnapToGrid` 参照）。
func TestDetectBoardIsCropInvariant(t *testing.T) {
	SetStripJudge(nil) // 環境にファイルがあるかで結果が変わらないように
	t.Cleanup(ResetStripJudge)

	img := makeSceneImage()
	board := image.Rect(sceneX, sceneY, sceneX+sceneCellW*9, sceneY+sceneCellH*9)

	var first image.Rectangle
	for i, m := range []int{15, 45, 90, 160} {
		crop := board.Inset(-m).Intersect(img.Bounds())
		br := DetectBoard(cropTo(img, crop))
		if br == nil {
			t.Fatalf("余白 %dpx で検出できず", m)
		}
		got := br.Bounds.Add(crop.Min) // 元画像の座標へ戻す
		t.Logf("余白 %3dpx → %v", m, got)
		if i == 0 {
			first = got
			continue
		}
		if got != first {
			t.Errorf("余白 %dpx で外枠が変わった: %v（余白 15px では %v）", m, got, first)
		}
	}
}

// fitGridAxis は等間隔に並ぶピークにサブピクセルで当てはまること。
// 開始位置が数px ずれていても同じ答えに収束するのが要点。
func TestFitGridAxisSubpixel(t *testing.T) {
	const (
		origin = 40.5
		span   = 37.25
	)
	proj := make([]float64, 500)
	for i := range proj {
		proj[i] = 1 // 台座
	}
	for i := 0; i <= 9; i++ {
		p := origin + span*float64(i)
		// 線は太さを持つので山として置く（重心が真の位置に来る）
		for d := -2; d <= 2; d++ {
			x := int(math.Round(p)) + d
			if x < 0 || x >= len(proj) {
				continue
			}
			w := 1 - math.Abs(p-float64(x))/3
			if w > 0 {
				proj[x] += 20 * w
			}
		}
	}
	for _, start := range []float64{origin - 5, origin, origin + 5} {
		o, s, ok := fitGridAxis(proj, start, span)
		if !ok {
			t.Fatalf("start=%.1f: 当てはめられず", start)
		}
		if math.Abs(o-origin) > 0.5 || math.Abs(s-span) > 0.1 {
			t.Errorf("start=%.1f: origin=%.2f span=%.3f（正解 %.2f / %.3f）", start, o, s, origin, span)
		}
	}
}

// 格子が読めない画像では何もしない（元の窓をそのまま返す）
func TestSnapToGridKeepsRegionWithoutLines(t *testing.T) {
	img := image.NewGray(image.Rect(0, 0, 400, 400))
	for i := range img.Pix {
		img.Pix[i] = 180
	}
	br := BoardRegionFromRect(50, 50, 320, 344)
	if got := SnapToGrid(img, br); got.Bounds != br.Bounds {
		t.Errorf("線の無い画像で窓が動いた: %v → %v", br.Bounds, got.Bounds)
	}
}

// 1マス動かす余地は無い（探索は ±gridSnapWindow マス）
func TestSnapToGridDoesNotSlipACell(t *testing.T) {
	SetStripJudge(nil)
	t.Cleanup(ResetStripJudge)
	img := makeSceneImage()
	// わざと 1マス上へずらした窓を渡す
	br := BoardRegionFromRect(sceneX, sceneY-sceneCellH, sceneX+sceneCellW*9, sceneY+sceneCellH*8)
	got := SnapToGrid(img, br)
	if d := absInt(got.Bounds.Min.Y - br.Bounds.Min.Y); d > sceneCellH/2 {
		t.Errorf("1マス滑った: %v → %v", br.Bounds, got.Bounds)
	}
}
