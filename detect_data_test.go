package suteme

import (
	"encoding/json"
	"image"
	_ "image/png"
	"math"
	"os"
	"testing"
)

// detectTolCells は DetectBoard が手動指定座標からずれてよい範囲（マス単位）。
// 修正前は 1〜3マスずれていた（詳細は CLAUDE.md「盤面検出」）
const detectTolCells = 0.5

// 保存済み局面（data/）の手動指定座標を正解として DetectBoard のズレを測る。
// 位置（左上）と大きさの両方を1マスを単位とした比で見る。
//
// **帯の判定器（`unslipByJudge`）は外して測る。** カレントに
// `strip_data_v1.bin` があるかどうかで結果が変わると、同じテストが
// 環境によって違う数字を出すことになる。判定器を入れた効果は
// `training` の `TestUnslipHoldout` が leave-one-out で測る
// （全件で作った判定器では自分の帯が最近傍に来るので、ここで測っても意味が無い）。
//
// data/ は .gitignore 対象なので、無ければスキップする。
func TestDetectBoardMatchesManual(t *testing.T) {
	SetStripJudge(nil) // nil = 判定器を使わない
	t.Cleanup(ResetStripJudge)

	dir := findDataDir()
	if dir == "" {
		t.Skip("data/history.json が無いのでスキップ")
	}
	f, err := os.Open(dir + "/history.json")
	if err != nil {
		t.Skip(err)
	}
	var h historyFile
	err = json.NewDecoder(f).Decode(&h)
	f.Close()
	if err != nil {
		t.Fatalf("history.json: %v", err)
	}

	ok, total := 0, 0
	for _, e := range h.Entries {
		imgF, err := os.Open(dir + "/" + e.ID + ".png")
		if err != nil {
			continue
		}
		img, _, err := image.Decode(imgF)
		imgF.Close()
		if err != nil {
			continue
		}
		total++

		man := image.Rect(e.Bounds.X1, e.Bounds.Y1, e.Bounds.X2, e.Bounds.Y2)
		cw, ch := float64(man.Dx())/9, float64(man.Dy())/9

		det := DetectBoard(img)
		if det == nil {
			t.Logf("%s: 検出できず", e.ID)
			continue
		}
		// 画像からはみ出した領域を返さない（`unslipRegion`）。
		// はみ出しは「窓が1マス滑った」印であると同時に、
		// そのままでは ExtractCell が画像外を読む
		if !det.Bounds.In(img.Bounds()) {
			t.Errorf("%s: 検出 %v が画像 %v からはみ出している", e.ID, det.Bounds, img.Bounds())
		}
		dx := float64(det.Bounds.Min.X-man.Min.X) / cw
		dy := float64(det.Bounds.Min.Y-man.Min.Y) / ch
		dw := float64(det.Bounds.Dx()-man.Dx()) / cw
		dh := float64(det.Bounds.Dy()-man.Dy()) / ch
		worst := math.Max(math.Max(math.Abs(dx), math.Abs(dy)), math.Max(math.Abs(dw), math.Abs(dh)))
		conf := ValidateBoard(img, det)

		t.Logf("%s: dx=%+.2f dy=%+.2f dw=%+.2f dh=%+.2f conf=%.2f", e.ID, dx, dy, dw, dh, conf)

		if worst <= detectTolCells {
			ok++
			if conf < minBoardConfidence {
				t.Errorf("%s: 座標は合っている(最大 %.2fマス)のに conf=%.2f で棄却される", e.ID, worst, conf)
			}
			continue
		}
		// 外した場合、黙って通してはいけない（ValidateBoard が気付くこと）
		if conf >= minBoardConfidence {
			t.Errorf("%s: %.2fマスずれているのに conf=%.2f で採用される", e.ID, worst, conf)
		}
	}

	if total == 0 {
		t.Skip("画像が無いのでスキップ")
	}
	t.Logf("合計: %d/%d が %.1fマス以内", ok, total, detectTolCells)
	// 修正前は 11 件中 1 件しか合っていなかった
	if ok*10 < total*8 {
		t.Errorf("%.1fマス以内に収まったのが %d/%d しかない", detectTolCells, ok, total)
	}
}

// cropInvarianceMargins は同じ盤を切り出す余白（px）。
// 詰めた枠と広い枠で答えが変わらないことを見る
var cropInvarianceMargins = []int{30, 150}

// **同じ盤なら、キャプチャの余白が違っても同じ位置に切り出せること。**
//
// suteme が最も守りたい性質（「こう撮ると認識率が高い」を無くす）で、
// 精度とは別に測る。実測（同一画素の 2 枚 `8ee60d34` / `8b737554` は
// 平行移動しただけの関係）では外枠が 5px 違い、それだけで認識が 2マス変わった。
//
// **見るのは 1マス未満のぶれ。** 余白でマス 1 つぶん滑るもの（実測 6 件）は
// **帯の判定器（`unslipByJudge`）が引き受ける領分**で、このテストは環境に
// `strip_data_v1.bin` があるかで数字が変わらないよう判定器を外して測るため、
// ここでは件数を数えるだけにする（判定器込みの成績は `TestUnslipHoldout` が
// leave-one-out で測っていて 157/157）。
//
// 実測（148局面 × 余白 2 通り）: 完全一致 137 / 1マス滑り 8 /
// 残り 3 件のぶれが平均 3.0px・最大 4px。判定器を入れた 3 通りの測り方では
// 完全一致 140・平均 0.41px（`SnapToGrid` の前は 97・2.32px）。
func TestDetectBoardCropInvariance(t *testing.T) {
	SetStripJudge(nil) // 環境にファイルがあるかで結果が変わらないように
	t.Cleanup(ResetStripJudge)

	dir := findDataDir()
	if dir == "" {
		t.Skip("data/history.json が無いのでスキップ")
	}
	f, err := os.Open(dir + "/history.json")
	if err != nil {
		t.Skip(err)
	}
	var h historyFile
	err = json.NewDecoder(f).Decode(&h)
	f.Close()
	if err != nil {
		t.Fatalf("history.json: %v", err)
	}

	n, same, slip, wobble, worstWobble := 0, 0, 0, 0, 0
	for _, e := range h.Entries {
		imgF, err := os.Open(dir + "/" + e.ID + ".png")
		if err != nil {
			continue
		}
		img, _, err := image.Decode(imgF)
		imgF.Close()
		if err != nil {
			continue
		}
		man := image.Rect(e.Bounds.X1, e.Bounds.Y1, e.Bounds.X2, e.Bounds.Y2)
		if man.Dx() < 90 || man.Dy() < 90 {
			continue
		}
		cell := math.Min(float64(man.Dx()), float64(man.Dy())) / 9

		var rects []image.Rectangle
		for _, m := range cropInvarianceMargins {
			crop := man.Inset(-m).Intersect(img.Bounds())
			// 余白が取れない画像（盤が画像いっぱい）は比べようが無い
			if crop.Dx() < man.Dx()+10 || crop.Dy() < man.Dy()+10 {
				rects = nil
				break
			}
			br := DetectBoard(cropSubImage(img, crop))
			if br == nil {
				rects = nil
				break
			}
			rects = append(rects, br.Bounds.Add(crop.Min)) // 元画像の座標へ戻す
		}
		if len(rects) < len(cropInvarianceMargins) {
			continue
		}

		n++
		worst := 0
		for _, r := range rects[1:] {
			for _, d := range []int{
				r.Min.X - rects[0].Min.X, r.Min.Y - rects[0].Min.Y,
				r.Max.X - rects[0].Max.X, r.Max.Y - rects[0].Max.Y,
			} {
				if d = absInt(d); d > worst {
					worst = d
				}
			}
		}
		switch {
		case worst == 0:
			same++
		case float64(worst) >= cell/2:
			slip++
			t.Logf("%s: 余白でマスがずれる（%dpx = %.2fマス）%v",
				e.ID, worst, float64(worst)/cell, rects)
		default:
			wobble += worst
			if worst > worstWobble {
				worstWobble = worst
			}
		}
	}
	if n == 0 {
		t.Skip("比べられる画像が無いのでスキップ")
	}
	avg := float64(wobble) / math.Max(1, float64(n-same-slip))
	t.Logf("余白違いで完全一致 %d/%d（1マス滑り %d / 残りの平均ぶれ %.2fpx・最大 %dpx）",
		same, n, slip, avg, worstWobble)

	// 実測 137/148・残りは最大 4px。**下回ったら「撮り方で切り出しが動く」に
	// 戻っているということ**なので、余裕を持たせつつ歯止めを置く
	if same*4 < n*3 {
		t.Errorf("余白違いで完全一致したのが %d/%d しかない", same, n)
	}
	if worstWobble > 8 {
		t.Errorf("完全一致しなかった局面のぶれが最大 %dpx ある", worstWobble)
	}
}

// cropSubImage は画像の一部を原点 (0,0) に寄せて切り出す（キャプチャの枠の代用）
func cropSubImage(src image.Image, r image.Rectangle) image.Image {
	r = r.Intersect(src.Bounds())
	dst := image.NewRGBA(image.Rect(0, 0, r.Dx(), r.Dy()))
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			dst.Set(x-r.Min.X, y-r.Min.Y, src.At(x, y))
		}
	}
	return dst
}
