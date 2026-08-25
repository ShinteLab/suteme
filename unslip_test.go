package suteme

import (
	"image"
	"image/color"
	"os"
	"path/filepath"
	"testing"
)

// makeStripBoardImage は盤と、その外に別のもの（駒台らしき明るい矩形）が
// 写った画像を作る。盤は (bx,by) から 9x9 マス。
func makeStripBoardImage(bx, by, cw, ch int) *image.Gray {
	w, h := bx+cw*9+cw*2, by+ch*9+ch*2
	img := image.NewGray(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetGray(x, y, color.Gray{Y: 60}) // 盤の外（暗い背景）
		}
	}
	for y := by; y < by+ch*9; y++ {
		for x := bx; x < bx+cw*9; x++ {
			img.SetGray(x, y, color.Gray{Y: 180}) // 盤の地
		}
	}
	for i := 0; i <= 9; i++ {
		for y := by; y < by+ch*9; y++ {
			for d := 0; d < 2; d++ {
				img.SetGray(bx+i*cw+d, y, color.Gray{Y: 30})
			}
		}
		for x := bx; x < bx+cw*9; x++ {
			for d := 0; d < 2; d++ {
				img.SetGray(x, by+i*ch+d, color.Gray{Y: 30})
			}
		}
	}
	return img
}

func TestStripInputShape(t *testing.T) {
	img := makeStripBoardImage(40, 40, 30, 33)
	in := StripInput(img, image.Rect(40, 40, 70, 40+33*9), true)
	if len(in) != stripSize {
		t.Fatalf("入力長が %d ではない: %d", stripSize, len(in))
	}
	// 縦の帯は転置して横の帯と同じ向きに揃える。長さが同じことだけ確認する
	in2 := StripInput(img, image.Rect(40, 40, 40+30*9, 73), false)
	if len(in2) != stripSize {
		t.Fatalf("横の帯の入力長が %d ではない: %d", stripSize, len(in2))
	}
	// 小さすぎる矩形は nil
	if StripInput(img, image.Rect(0, 0, 2, 2), false) != nil {
		t.Error("小さすぎる矩形で nil が返らない")
	}
	if StripInput(nil, image.Rect(0, 0, 30, 30), false) != nil {
		t.Error("nil 画像で nil が返らない")
	}
}

// 帯データの保存と読み込みで内容が変わらないこと。
// **float32 へ丸めてあるので往復しても一致する**（駒の入力と同じ理由）。
func TestStripDataSurvivesSaveLoad(t *testing.T) {
	img := makeStripBoardImage(40, 40, 30, 33)
	want := []StripSample{
		{Input: StripInput(img, image.Rect(40, 40, 70, 40+33*9), true), Board: true},
		{Input: StripInput(img, image.Rect(10, 40, 40, 40+33*9), true), Board: false},
	}
	path := filepath.Join(t.TempDir(), "strip.bin")
	if err := SaveStripData(path, want); err != nil {
		t.Fatalf("保存に失敗: %v", err)
	}
	got, err := LoadStripData(path)
	if err != nil {
		t.Fatalf("読み込みに失敗: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("件数が違う: %d != %d", len(got), len(want))
	}
	for i := range got {
		if got[i].Board != want[i].Board {
			t.Errorf("%d: Board が違う", i)
		}
		for j := range got[i].Input {
			if got[i].Input[j] != want[i].Input[j] {
				t.Fatalf("%d: 入力の %d 番目が変わった %v != %v",
					i, j, got[i].Input[j], want[i].Input[j])
			}
		}
	}
}

// 壊れたファイルは「空」ではなくエラーにすること。
// 空として読むと、次の保存で正解データが消える。
func TestLoadStripDataRejectsGarbage(t *testing.T) {
	dir := t.TempDir()
	for name, body := range map[string][]byte{
		"empty.bin": {},
		"other.bin": []byte("SUTEMETD\x04\x00\x40\x02\x00\x00\x00\x00"), // 駒の学習データ
		"short.bin": append([]byte("SUTEMESJ\x01\x00\x40\x02\x02\x00\x00\x00"), 1, 2, 3),
	} {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, body, 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := LoadStripData(p); err == nil {
			t.Errorf("%s: エラーにならない", name)
		}
	}
}

// 判定器が無ければ滑りの補正はしない（従来どおりの検出結果になる）。
func TestUnslipByJudgeNoopWithoutJudge(t *testing.T) {
	img := makeStripBoardImage(40, 40, 30, 33)
	br := BoardRegionFromRect(40, 40, 40+30*9, 40+33*9)
	if got := unslipByJudge(img, br, nil); got != br {
		t.Error("判定器が nil なのに窓が動いた")
	}
	if got := unslipByJudge(img, nil, nil); got != nil {
		t.Error("窓が nil なのに nil が返らない")
	}
}

// 1マス滑った窓を、盤の縁の帯で元に戻せること。
//
// 学習データは同じ画像の縁の帯（盤）と 1マス外の帯（盤の外）。
// **実データでの成績は `training` の leave-one-out で測る**ので、
// ここで見るのは「仕組みが繋がっていること」だけ。
func TestUnslipByJudgeFixesOneCellSlip(t *testing.T) {
	const cw, ch, bx, by = 30, 33, 60, 66
	img := makeStripBoardImage(bx, by, cw, ch)

	var samples []StripSample
	add := func(r image.Rectangle, vertical, board bool) {
		if in := StripInput(img, r, vertical); in != nil {
			samples = append(samples, StripSample{Input: in, Board: board})
		}
	}
	x2, y2 := bx+cw*9, by+ch*9
	add(image.Rect(bx, by, bx+cw, y2), true, true)
	add(image.Rect(x2, by, x2+cw, y2), true, false)
	add(image.Rect(x2-cw, by, x2, y2), true, true)
	add(image.Rect(bx-cw, by, bx, y2), true, false)
	add(image.Rect(bx, by, x2, by+ch), false, true)
	add(image.Rect(bx, y2, x2, y2+ch), false, false)
	j := NewStripJudge(samples)
	if j == nil {
		t.Fatal("判定器を作れなかった")
	}

	// 右へ 1マス滑った窓を渡すと、左へ戻ること
	slipped := BoardRegionFromRect(bx+cw, by, x2+cw, y2)
	got := unslipByJudge(img, slipped, j)
	if got.Bounds.Min.X != bx {
		t.Errorf("滑りが直らない: %v (正しくは x=%d)", got.Bounds, bx)
	}

	// 正しい窓は動かさないこと
	correct := BoardRegionFromRect(bx, by, x2, y2)
	if got := unslipByJudge(img, correct, j); got.Bounds != correct.Bounds {
		t.Errorf("正しい窓が動いた: %v → %v", correct.Bounds, got.Bounds)
	}
}

// 得る帯がほとんど画像の外にあるときは乗り換えないこと（`stripGainCover`）。
//
// `StripInput` は画像の範囲で切ってから 72x8 に潰すので、外へ出た帯は
// 「9マス分の帯」ではなく別物になるのに、判定器はそれでもスコアを返す。
// 実データ（`6850d800`）では、上の帯が 65px 中 49px しか無いのに
// 盤らしさ 1.00 を返し、**正しかった窓（0.07マス）を 0.98マスへ動かして**いた。
func TestUnslipByJudgeIgnoresClippedStrip(t *testing.T) {
	const cw, ch, bx, by = 30, 33, 60, 66
	img := makeStripBoardImage(bx, by, cw, ch)
	x2, y2 := bx+cw*9, by+ch*9

	// 盤の 1マス上に、盤とそっくりな帯（同じ地色・格子線）を描く。
	// 中継の UI パネルのように「盤らしく見える盤の外」に相当する
	for y := by - ch; y < by; y++ {
		for x := bx; x < x2; x++ {
			img.SetGray(x, y, color.Gray{Y: 180})
		}
	}
	for i := 0; i <= 9; i++ {
		for y := by - ch; y < by; y++ {
			for d := 0; d < 2; d++ {
				img.SetGray(bx+i*cw+d, y, color.Gray{Y: 30})
			}
		}
	}
	for x := bx; x < x2; x++ {
		for d := 0; d < 2; d++ {
			img.SetGray(x, by-ch+d, color.Gray{Y: 30})
		}
	}

	// 盤の下端の 1 行は、影や UI に覆われて盤らしく見えない状態にする。
	// これが無いと「失う帯」も盤そのものなので差が付かず、前提が作れない
	for y := y2 - ch; y < y2; y++ {
		for x := bx; x < x2; x++ {
			img.SetGray(x, y, color.Gray{Y: 60})
		}
	}

	var samples []StripSample
	add := func(r image.Rectangle, vertical, board bool) {
		if in := StripInput(img, r, vertical); in != nil {
			samples = append(samples, StripSample{Input: in, Board: board})
		}
	}
	add(image.Rect(bx, by, x2, by+ch), false, true)  // 盤の上端
	add(image.Rect(bx, y2, x2, y2+ch), false, false) // 盤の下の外
	add(image.Rect(bx, by, bx+cw, y2), true, true)   // 盤の左端
	add(image.Rect(bx-cw, by, bx, y2), true, false)  // 盤の左の外
	j := NewStripJudge(samples)
	if j == nil {
		t.Fatal("判定器を作れなかった")
	}

	correct := BoardRegionFromRect(bx, by, x2, y2)

	// 帯が全部写っていれば、判定器は盤そっくりの帯に釣られて上へ動く
	// （この局面自体は直せない。ここで確かめたいのは判定器が働いていること）
	if got := unslipByJudge(img, correct, j); got.Bounds == correct.Bounds {
		t.Fatal("前提が崩れている: 盤そっくりの帯があるのに窓が動かない")
	}

	// 同じ帯が 1/4 しか写っていなければ、比べずに窓を残すこと
	clipped := img.SubImage(image.Rect(0, by-ch/4, img.Bounds().Max.X, img.Bounds().Max.Y))
	if got := unslipByJudge(clipped, correct, j); got.Bounds != correct.Bounds {
		t.Errorf("切れた帯で窓が動いた: %v → %v", correct.Bounds, got.Bounds)
	}
}
