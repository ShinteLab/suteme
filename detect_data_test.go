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
