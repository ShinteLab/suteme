package suteme

import (
	"image"
	"image/color"
	"math"
	"testing"
)

// woodCell は木目の縞が数本入った空マスを作る。
// extra が真なら縁に短いかけらを足す（撮り直しで縞の端が出入りする状況）。
func woodCell(w, h int, extra bool) image.Image {
	img := image.NewGray(image.Rect(0, 0, w, h))
	lines := []float64{14, 20, 27}
	for y := 0; y < h; y++ {
		// 実盤の木目はまっすぐではない。蛇行させないと「列のほぼ全高が
		// 埋まっている列」としてグリッド線と一緒に落ちてしまう
		wob := 2 * math.Sin(float64(y)/7)
		for x := 0; x < w; x++ {
			v := uint8(170)
			for _, cx := range lines {
				if math.Abs(float64(x)-(cx+wob)) < 1.2 {
					v = 120
				}
			}
			// 縁に短い縞のかけら。**マスの見た目はほとんど変わらないのに
			// 外接矩形の左端はここまで広がる**
			if extra && y >= 12 && y < 22 && (x == 8 || x == 9) {
				v = 120
			}
			img.SetGray(x, y, color.Gray{Y: v})
		}
	}
	return img
}

func inputDist(x, y []float64) float64 {
	d := 0.0
	for i := range x {
		d += (x[i] - y[i]) * (x[i] - y[i])
	}
	return d / emptyDistNorm
}

// 空マスの照合を外接矩形で切ってはいけない（CellToInputFull）。
//
// `pieceBox` は「地色と異なる画素」の外接矩形なので、駒の無いマスでは
// 木目の縞をどこまで囲むかで矩形が変わる。縁に数px のかけらが出入りしただけで
// 切り出す幅が変わり、24x24 に引き伸ばすと**ほとんど同じマスなのに別のベクトル**
// になる。実データでも同じ盤の同じ空マスが 0.936 まで離れていた。
func TestEmptyMatchIgnoresPieceBox(t *testing.T) {
	a := woodCell(40, 46, false)
	b := woodCell(40, 46, true)

	ba, _ := pieceBox(a)
	bb, _ := pieceBox(b)
	if ba == bb {
		t.Fatalf("外接矩形が同じでは何も測れない: %v", ba)
	}

	crop := inputDist(CellToInput(a), CellToInput(b))
	full := inputDist(CellToInputFull(a), CellToInputFull(b))
	if full >= crop {
		t.Errorf("マス全体 %.3f が外接矩形 %.3f より遠い（切らないほうが安定するはず）", full, crop)
	}
	if full >= emptyMatchMax {
		t.Errorf("マス全体の距離 %.3f が %.2f 以上。空の一致判定が働かない", full, emptyMatchMax)
	}

	// 学習側（SampleInput）も同じ表現でなければ意味が無い
	kn := NewKNN([]TrainingSample{{Input: SampleInput(a, ClassEmpty), Label: ClassEmpty}})
	if class, _ := kn.Predict(b); class != ClassEmpty {
		t.Errorf("class = %d, want ClassEmpty（縁に数px 増えただけの同じ空マス）", class)
	}
	// 分類器が後手と誤って 180 度回して渡してきても拾えること（空に向きは無い）
	if class, _ := kn.Predict(Rotate180(b)); class != ClassEmpty {
		t.Error("180度回した空マスが ClassEmpty にならない")
	}
}

// 駒サンプルの表現は v7 のまま（外接矩形で切る）
func TestSampleInputCropsPieces(t *testing.T) {
	cell := woodCell(40, 46, true)
	got, want := SampleInput(cell, 0), CellToInput(cell)
	for i := range want {
		if got[i] != want[i] {
			t.Fatal("駒サンプルは CellToInput のままであること")
		}
	}
}
