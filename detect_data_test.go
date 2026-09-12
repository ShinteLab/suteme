package suteme

import (
	"image"
	_ "image/png"
	"math"
	"testing"
)

// detectTolCells は DetectBoard が手動指定座標からずれてよい範囲（マス単位）。
// 修正前は 1〜3マスずれていた（詳細は CLAUDE.md「盤面検出」）
const detectTolCells = 0.5

// detectSlipMaxRatio は**判定器を外した素の検出**に残ってよい 1マス滑りの割合。
//
// **1マスの滑りは `ValidateBoard` では原理的に見分けられない。**
// `gridAlignment` は全マスシフトに不変なので、周期が正しいまま窓が 1マス
// 滑った検出は**信頼度 1.00 のまま通る**（CLAUDE.md「盤面検出の検証」）。
// この見分けは帯の判定器（`unslipByJudge`）の領分で、このテストは
// 環境に依らない数字を出すためにその判定器を**外して**測っている。
// つまり「滑ったのに黙って通る」ことをここで責めるのは筋が違う。
//
// 代わりに**件数を縛って**、増えたら気付けるようにする。実測（157局面）:
// 滑りは 5 件（3.2%）で、**`training` の `TestUnslipHoldout` が
// leave-one-out でその 5 件すべてを直す**ことを確かめている（157/157）。
const detectSlipMaxRatio = 0.05

// detectOverflowTolCells は検出領域が画像からはみ出してよい範囲（マス単位）。
//
// **0 にはできない。** 仕上げの `SnapToGrid` は外枠を格子線そのものへ
// サブピクセルで合わせ直すので、**盤が画像の端で切れている画像では
// 格子線の推定位置が画像の外に来る**（実測 `9d2c92ae`: 画像の下端 701 に対し
// 707＝0.24マス）。これは滑りではなく、**人が引いた枠を同じ工程に通した
// 座標と px 単位で一致する**（学習側の `samplesFromHistory` も同じ座標を使う）。
// `ExtractCell` は画像の範囲で切るので、はみ出した側のマスが少し狭くなるだけ。
//
// 1マス以上のはみ出しは話が別で、そちらは窓が滑った印（`unslipRegion` の領分）。
const detectOverflowTolCells = 0.5

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

	dir, entries := loadSavedBoards(t)

	// 1 局面ぶんの測定結果。**判定とログは並列の外で行う**（mapSavedBoards の説明）
	type res struct {
		id             string
		found          bool
		over           float64
		detB, imgB     image.Rectangle
		dx, dy, dw, dh float64
		worst, conf    float64
	}
	rs := mapSavedBoards(dir, entries, func(e historyEntry, img image.Image) res {
		r := res{id: e.ID, imgB: img.Bounds()}
		man := image.Rect(e.Bounds.X1, e.Bounds.Y1, e.Bounds.X2, e.Bounds.Y2)
		cw, ch := float64(man.Dx())/9, float64(man.Dy())/9
		det := DetectBoard(img)
		if det == nil {
			return r
		}
		r.found = true
		r.detB = det.Bounds
		r.over = overflowCells(det.Bounds, img.Bounds(), cw, ch)
		r.dx = float64(det.Bounds.Min.X-man.Min.X) / cw
		r.dy = float64(det.Bounds.Min.Y-man.Min.Y) / ch
		r.dw = float64(det.Bounds.Dx()-man.Dx()) / cw
		r.dh = float64(det.Bounds.Dy()-man.Dy()) / ch
		r.worst = math.Max(math.Max(math.Abs(r.dx), math.Abs(r.dy)), math.Max(math.Abs(r.dw), math.Abs(r.dh)))
		r.conf = ValidateBoard(img, det)
		return r
	})

	ok, total, slipped := 0, 0, 0
	for _, r := range rs {
		total++
		if !r.found {
			t.Logf("%s: 検出できず", r.id)
			continue
		}
		// **1マス以上はみ出した領域を返さない（`unslipRegion`）。**
		// はみ出しは「窓が1マス滑った」印であると同時に、
		// そのままでは ExtractCell が画像の外を読むことになる。
		// 数px は `SnapToGrid` の当てはめ由来なので許す（`detectOverflowTolCells`）
		if r.over > detectOverflowTolCells {
			t.Errorf("%s: 検出 %v が画像 %v から %.2fマスはみ出している",
				r.id, r.detB, r.imgB, r.over)
		}
		t.Logf("%s: dx=%+.2f dy=%+.2f dw=%+.2f dh=%+.2f conf=%.2f", r.id, r.dx, r.dy, r.dw, r.dh, r.conf)

		if r.worst <= detectTolCells {
			ok++
			if r.conf < minBoardConfidence {
				t.Errorf("%s: 座標は合っている(最大 %.2fマス)のに conf=%.2f で棄却される", r.id, r.worst, r.conf)
			}
			continue
		}
		// **1マスの滑りだけは別扱い。** 大きさは合っているのに位置だけが
		// 1マスぶん動いた検出は、判定器を外している以上ここでは落とせない
		// （`detectSlipMaxRatio`）。件数だけ数えておく
		if isOneCellSlip(r.dx, r.dy, r.dw, r.dh) {
			slipped++
			t.Logf("%s: 1マス滑り（判定器の領分。TestUnslipHoldout が直すこと）", r.id)
			continue
		}
		// それ以外で外した場合は、黙って通してはいけない（ValidateBoard が気付くこと）
		if r.conf >= minBoardConfidence {
			t.Errorf("%s: %.2fマスずれているのに conf=%.2f で採用される", r.id, r.worst, r.conf)
		}
	}

	if total == 0 {
		t.Skip("画像が無いのでスキップ")
	}
	t.Logf("合計: %d/%d が %.1fマス以内（1マス滑り %d 件 = %.1f%%）",
		ok, total, detectTolCells, slipped, float64(slipped)/float64(total)*100)
	// 修正前は 11 件中 1 件しか合っていなかった
	if ok*10 < total*8 {
		t.Errorf("%.1fマス以内に収まったのが %d/%d しかない", detectTolCells, ok, total)
	}
	if float64(slipped) > float64(total)*detectSlipMaxRatio {
		t.Errorf("1マス滑りが %d/%d 件ある（上限 %.0f%%）。帯の判定器を外した素の検出の話なので、"+
			"増えたら滑りを作っている前段（unslipRegion / snapToOuterFrame / pickTaper）を疑うこと",
			slipped, total, detectSlipMaxRatio*100)
	}
}

// overflowCells は矩形 r が枠 outer から何マスはみ出しているかを返す
func overflowCells(r, outer image.Rectangle, cw, ch float64) float64 {
	over := 0.0
	for _, v := range []struct {
		px   int
		cell float64
	}{
		{outer.Min.X - r.Min.X, cw}, {outer.Min.Y - r.Min.Y, ch},
		{r.Max.X - outer.Max.X, cw}, {r.Max.Y - outer.Max.Y, ch},
	} {
		if v.px > 0 {
			over = math.Max(over, float64(v.px)/v.cell)
		}
	}
	return over
}

// isOneCellSlip は「大きさは合っているのに位置だけ 1マスぶん動いた」検出かを返す。
//
// **周期が違う検出（半分の周期・盤の一部）と区別するために大きさも見る。**
// あちらは `pickTaper` や `gridAlignment` で落とせるので、
// 見逃してよい理由が無い（CLAUDE.md「半分の周期」）。
func isOneCellSlip(dx, dy, dw, dh float64) bool {
	if math.Abs(dw) > detectTolCells || math.Abs(dh) > detectTolCells {
		return false
	}
	slid := func(d float64) bool { return math.Abs(d) > detectTolCells && math.Abs(d) <= 1.5 }
	return slid(dx) || slid(dy)
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

	dir, entries := loadSavedBoards(t)

	type res struct {
		id    string
		use   bool
		cell  float64
		worst int
		rects []image.Rectangle
	}
	rs := mapSavedBoards(dir, entries, func(e historyEntry, img image.Image) res {
		r := res{id: e.ID}
		man := image.Rect(e.Bounds.X1, e.Bounds.Y1, e.Bounds.X2, e.Bounds.Y2)
		if man.Dx() < 90 || man.Dy() < 90 {
			return r
		}
		r.cell = math.Min(float64(man.Dx()), float64(man.Dy())) / 9

		var rects []image.Rectangle
		for _, m := range cropInvarianceMargins {
			crop := man.Inset(-m).Intersect(img.Bounds())
			// 余白が取れない画像（盤が画像いっぱい）は比べようが無い
			if crop.Dx() < man.Dx()+10 || crop.Dy() < man.Dy()+10 {
				return r
			}
			br := DetectBoard(cropSubImage(img, crop))
			if br == nil {
				return r
			}
			rects = append(rects, br.Bounds.Add(crop.Min)) // 元画像の座標へ戻す
		}
		r.use, r.rects = true, rects
		for _, x := range rects[1:] {
			for _, d := range []int{
				x.Min.X - rects[0].Min.X, x.Min.Y - rects[0].Min.Y,
				x.Max.X - rects[0].Max.X, x.Max.Y - rects[0].Max.Y,
			} {
				if d = absInt(d); d > r.worst {
					r.worst = d
				}
			}
		}
		return r
	})

	n, same, slip, wobble, worstWobble := 0, 0, 0, 0, 0
	for _, r := range rs {
		if !r.use {
			continue
		}
		n++
		switch {
		case r.worst == 0:
			same++
		case float64(r.worst) >= r.cell/2:
			slip++
			t.Logf("%s: 余白でマスがずれる（%dpx = %.2fマス）%v",
				r.id, r.worst, float64(r.worst)/r.cell, r.rects)
		default:
			wobble += r.worst
			if r.worst > worstWobble {
				worstWobble = r.worst
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
