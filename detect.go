package suteme

import (
	"image"
	"math"
)

type BoardRegion struct {
	Bounds image.Rectangle
	Cells  [9][9]image.Rectangle
}

// boardAspect は将棋盤のマスの縦横比（高さ / 幅）。
//
// **将棋盤は正方形ではない。** 規格は 3.03寸 x 3.33寸 で 1.099、
// 保存済み局面の手動指定座標での実測は 1.045〜1.138（平均 1.09）。
// 正方形とみなすと横方向に 8〜9% 広い盤を探すことになる。
const boardAspect = 1.09

// segSkipConfidence は画像全体での検出がこの信頼度に達したら
// 線分による絞り込みを試さない基準。
//
// 盤だけを切り出した画像（保存済み 14 件中 11 件）は全体で 1.00 が出るので、
// 候補を増やす意味が無いうえ、候補が同点で割り込む余地も与えたくない。
const segSkipConfidence = 0.9

// DetectBoard は画像から盤面領域を検出する。
//
// まず画像全体で検出し、信頼度が足りなければ **線分抽出で盤らしい x 範囲を
// 絞ってから**同じ検出をやり直す（`segment.go`）。キャプチャは駒台まで含めて
// 撮る想定なので、盤以外が写るのは例外ではなく常態であり、投影を全幅で取ると
// 盤外の罫線に負ける。候補は ValidateBoard が最も高いものを採り、
// **同点なら「別々の ROI から同じ窓が出た数」（票）が多いほう**を採り、
// それも同じなら先に見つけたものを残す（`candidateSet`）。
func DetectBoard(img image.Image) *BoardRegion {
	gray := ConvertGray(img)
	blurred := BoxBlur(gray, 2)
	edges := Sobel(blurred)

	whole := detectBoardIn(edges, img.Bounds())
	wholeConf := ValidateBoard(img, whole)
	if wholeConf >= segSkipConfidence {
		return finishRegion(img, whole)
	}

	set := &candidateSet{}
	set.add(whole, wholeConf)
	for _, roi := range boardROIs(blurred) {
		cand := detectBoardIn(edges, roi)
		if cand == nil || !plausibleAspect(cand) {
			continue
		}
		set.add(refineRegion(img, cand))
	}
	best := set.best()
	if best == nil {
		return nil
	}
	// **票で選ばれた窓でも `snapToOuterFrame` は通す。** 「複数の ROI が合意
	// しているなら 1マス動かさない」も試したが、`snapToOuterFrame` が本来
	// 直すはずの局面（`2d7bfedf` `9eb72770`）が 0.91〜1.09マスへ戻り、
	// 135 局面で 131 → 130 と差し引き悪くなる。
	// なお `e0302032` は逆に、票で選んだ正しい窓 (473,166)-(1082,854)（0.44マス）を
	// snap が (465,101)-(1081,756)（0.72マス）へ動かしてしまう例（後述の課題）
	return finishRegion(img, best.br)
}

// finishRegion は検出した窓の仕上げ。3 つとも「1マスの滑り」を直すもので、
// **効く場面が別々なので順に通す**。
//
//  1. `unslipRegion` … 画像からはみ出していたら 1マス内側へ戻す（幾何）
//  2. `snapToOuterFrame` … 10本のうち最弱の線がはっきり強くなるなら乗り換える（線）
//  3. `unslipByJudge` … 盤の縁の帯が盤らしいほうへ乗り換える（学習）
//
// 3 は帯の教師データ（`strip_data_v1.bin`）が無ければ**何もしない**ので、
// 判定器を置いていない環境の挙動は 1・2 までと変わらない。
func finishRegion(img image.Image, br *BoardRegion) *BoardRegion {
	return unslipByJudge(img, snapToOuterFrame(img, unslipRegion(img, br)), defaultStripJudge())
}

// candTolCells は 2 つの候補を「同じ窓」とみなす差（マス単位）。
// ROI が違えば投影に載る画素も少し変わるので、数px の差は同じ窓とみなす
const candTolCells = 1.0 / 3.0

// candidate は検出候補 1 つと、その窓を出した ROI の数。
type candidate struct {
	br    *BoardRegion
	conf  float64
	votes int
}

// candidateSet は検出候補を集めて 1 つ選ぶ。
//
// **信頼度が同点で並ぶ候補があり、そこで負けていた。** `gridAlignment` は
// 全マスシフトに対して不変なので（CLAUDE.md「盤面検出の検証」）、正しい窓と
// 1マス滑った窓が **どちらも 1.00** で並ぶ。以前の実装は `conf > bestConf` の
// 狭義比較だったため、**先に評価された候補がそのまま残っていた**。
// 必要なのは信頼度の改善ではなく**同点を割る材料**（CLAUDE.md「今後の課題」）。
//
// 材料は「**別々の ROI から同じ窓が出たか**」。盤の格子線は複数の絞り込み方
// （横線のクラスタ / 縦線のクラスタ / その組み合わせ）から同じ位置に出るのに対し、
// 滑った窓は特定の ROI でしか成立しないことが多い。実測（人が座標を引いた
// 11 局面の候補、conf 0.9 以上）:
//
//	26a22911  正 (573,196)-(1146,830) 票2  対  誤 (578,267)-(1139,890) 票1
//	c611e647  正 (330,111)-(999,847)  票4  対  誤 (330,38)-(997,764)   票1
//	e0302032  正 (473,166)-(1082,854) 票2  対  誤 (465,257)-(1085,894) 票1
//	11f95abb  正 (84,70)-(517,538)    票2  対  誤 (81,15)-(521,490)    票2  ← 割れない
//
// **他に試して駄目だったもの**（同じ道を再訪しないための記録。上と同じ 11 局面）:
//
//	窓のすぐ外側と内側の色差（四辺の最小）  26a22911 が 1.0 対 1.0 で同点、
//	                                        fa37bb49 は誤ったほうが高い（3.0 対 1.0）
//	`weakestLineRatio`                      26a22911 は誤 +0.17 対 正 +0.14 で逆
//	駒の外接矩形のマス内での偏り            全マスシフトでは駒はマス中央に残るので原理的に無効
//	窓の 1マス外側に格子線が続いていないか  正しい窓の外にも駒台・UI の線があり分離しない
//
// **票が同じなら順序を変えない**（画像全体 → `boardROIs` の順）。同点で
// 動かすと、これまで通っていた局面が理由なく別の窓に移る。
// 画像全体から得た候補を同点で特別扱いする案も試したが、135 局面で成績は
// 変わらなかったので**特別扱いは置いていない**（概念を増やさない）。
type candidateSet struct {
	cands []candidate
}

// add は候補を足す。既にある窓とほぼ同じなら票を足し、
// **位置は信頼度が高いほうを残す**。
//
// **ここで先に入ったほうの位置を残してはいけない。** ほぼ同じ窓でも数px の差で
// 整合度は変わり、以前の実装（`conf > bestConf` なら丸ごと差し替え）は
// 高いほうの位置を採っていた。先勝ちにすると、その数px を捨てることになる。
// 実測でこれをやると `22a8f6e2` `f8ddbeee` `4deb69f4` が 0.00マスから
// 0.91〜1.27マスへ壊れる。
func (s *candidateSet) add(br *BoardRegion, conf float64) {
	if br == nil {
		return
	}
	for i := range s.cands {
		if sameWindow(s.cands[i].br.Bounds, br.Bounds) {
			s.cands[i].votes++
			if conf > s.cands[i].conf {
				s.cands[i].br, s.cands[i].conf = br, conf
			}
			return
		}
	}
	s.cands = append(s.cands, candidate{br: br, conf: conf, votes: 1})
}

// best は信頼度 → 票 の順で最も良い候補を返す。
func (s *candidateSet) best() *candidate {
	var best *candidate
	for i := range s.cands {
		c := &s.cands[i]
		switch {
		case best == nil, c.conf > best.conf:
		case c.conf < best.conf:
			continue
		case c.votes > best.votes:
		default:
			continue
		}
		best = c
	}
	return best
}

// sameWindow は 2 つの窓を同じとみなしてよいかを返す（`candTolCells`）。
func sameWindow(a, b image.Rectangle) bool {
	cw := float64(a.Dx()) / 9
	ch := float64(a.Dy()) / 9
	if cw <= 0 || ch <= 0 {
		return false
	}
	tolW, tolH := cw*candTolCells, ch*candTolCells
	return math.Abs(float64(a.Min.X-b.Min.X)) < tolW && math.Abs(float64(a.Min.Y-b.Min.Y)) < tolH &&
		math.Abs(float64(a.Dx()-b.Dx())) < tolW && math.Abs(float64(a.Dy()-b.Dy())) < tolH
}

// snapMargin は snapToOuterFrame が窓を1マス動かすのに要求する
// `weakestLineRatio` の勝ち幅。
//
// **同点や小差で動かしてはいけない。** 保存済み 103 局面で 0.1 にすると
// `baeac294`（もともと 0.34マスで合っていた）が 0.87マスへ動いて壊れる。
// 0.3 なら**滑った 1 件だけが動く**（103 件中 1 件。他は候補が margin を
// 越えないのでそのまま）。
const snapMargin = 0.3

// snapToOuterFrame は1マス滑った窓を外枠へ寄せ直す。
//
// **`gridAlignment` は全マスシフトに対して不変**なので（CLAUDE.md
// 「盤面検出の検証」）、周期が正しいまま窓が1マス滑った検出は
// 信頼度 1.00 のまま通る。`unslipRegion` が拾えるのは画像からはみ出した
// 場合だけで、**収まったまま滑る**ものは見分けられなかった。
//
// 見分けるのは「10本の境界線のうち**いちばん弱い線**」
// （`weakestLineRatio`）。滑った窓は 9 本が本物の格子線に乗る代わりに、
// 外側の 1 本が**線の無い所**に来る。合計やパリティの和ではこの 1 本が
// 埋もれるが、最小を採れば表に出る。実測（`2d7bfedf`、列方向）:
// 滑った窓は最弱の線が off と同じ（比 -0.12）なのに対し、
// 1マス戻した窓は 0.87。
//
// 候補は ±1マスの平行移動 9 通りで、それぞれ `refineAxis` で
// 原点±半マス・間隔±10% を詰め直してから測る（滑った窓は間隔も
// 誤った線に合わせてあるので、平行移動だけでは正解に届かない）。
// **`snapMargin` を越える候補が無ければ動かさない**うえ、
// 乗り換えは `ValidateBoard` が下がらない場合に限る。
//
// 実測（保存済み 103 局面、手動座標との最大ずれ）:
//
//	2d7bfedf  0.91マス（conf 0.90 で採用）→ **0.06マス**（conf 0.96）
//	他 102 件  変化なし
//
// **これで直らない滑りが 2 件残る**（`c611e647` `11f95abb`）。どちらも
// 正しい位置のほうが最弱の線が弱い（盤の下端の線が薄い / 盤の上の UI の
// 罫線が格子線と同じだけ強い）ため、この指標では順位が付かない。
func snapToOuterFrame(img image.Image, br *BoardRegion) *BoardRegion {
	if br == nil {
		return nil
	}
	cw, ch := br.Bounds.Dx()/9, br.Bounds.Dy()/9
	if cw < 3 || ch < 3 {
		return br
	}

	// 候補（±1マス）とその微調整の余地（±半マス）が収まる範囲で投影を取る。
	// **候補どうしを同じ投影の上で比べる**（別々に取ると中央値がずれる）
	pad := image.Rect(
		br.Bounds.Min.X-cw*3/2, br.Bounds.Min.Y-ch*3/2,
		br.Bounds.Max.X+cw*3/2, br.Bounds.Max.Y+ch*3/2,
	).Intersect(img.Bounds())
	if pad.Dx() < 9*3 || pad.Dy() < 9*3 {
		return br
	}
	sub := image.NewRGBA(image.Rect(0, 0, pad.Dx(), pad.Dy()))
	for y := pad.Min.Y; y < pad.Max.Y; y++ {
		for x := pad.Min.X; x < pad.Max.X; x++ {
			sub.Set(x-pad.Min.X, y-pad.Min.Y, img.At(x, y))
		}
	}
	row, col := lineProjections(BoxBlur(ConvertGray(sub), 2))

	score := func(r image.Rectangle) float64 {
		v := weakestLineRatio(row, float64(r.Min.Y-pad.Min.Y), float64(r.Dy())/9)
		h := weakestLineRatio(col, float64(r.Min.X-pad.Min.X), float64(r.Dx())/9)
		if v < h {
			return v
		}
		return h
	}

	best := br
	bestScore := score(br.Bounds)
	for dy := -1; dy <= 1; dy++ {
		for dx := -1; dx <= 1; dx++ {
			if dx == 0 && dy == 0 {
				continue
			}
			r := br.Bounds.Add(image.Pt(dx*cw, dy*ch))
			if !r.In(img.Bounds()) {
				continue
			}
			oy, sy := refineAxis(row, float64(r.Min.Y-pad.Min.Y), float64(r.Dy())/9)
			ox, sx := refineAxis(col, float64(r.Min.X-pad.Min.X), float64(r.Dx())/9)
			r = image.Rect(
				pad.Min.X+int(math.Round(ox)), pad.Min.Y+int(math.Round(oy)),
				pad.Min.X+int(math.Round(ox+sx*9)), pad.Min.Y+int(math.Round(oy+sy*9)),
			)
			if !r.In(img.Bounds()) {
				continue
			}
			cand := BoardRegionFromRect(r.Min.X, r.Min.Y, r.Max.X, r.Max.Y)
			if !plausibleAspect(cand) {
				continue
			}
			if s := score(r); s > bestScore+snapMargin {
				best, bestScore = cand, s
			}
		}
	}
	if best == br {
		return br
	}
	// 乗り換えるのは信頼度を下げないときだけ（滑りを直すなら普通は上がる。
	// 実測 `2d7bfedf` は 0.90 → 0.96）
	if ValidateBoard(img, best) < ValidateBoard(img, br) {
		return br
	}
	return best
}

// unslipRegion は画像からはみ出した検出領域を1マス内側へ戻す。
//
// **盤の外枠線を切り落とした画像では、検出器から見て一番外の縦線が
// 1本目の内側の線になり、周期は正しいまま窓が1マス滑る**
// （CLAUDE.md「盤面検出」の `pickTaper` の項）。滑った窓は格子線には
// 乗っているので `gridAlignment` では見分けられず、**信頼度 1.00 のまま
// 通る**という質の悪い壊れ方をする。
//
// 見分けが付くのは**画像からはみ出すこと**。保存済み 89 局面で調べると、
// はみ出したのは滑った 2 件（`576b34e8` `c4a7e592`）だけで、
// 正しく検出できている 87 件は 0 件だった。しかも 1マス戻すと
// **両方とも信頼度 1.00 のまま正解に収まる**:
//
//	576b34e8  dx=+0.97 / はみ出し 52px  →  0.09マス
//	c4a7e592  dx=+0.94 / はみ出し 41px  →  0.14マス
//
// **信頼度が下がらない限り戻したほうを採る**（同点でも戻す）。滑りは
// `gridAlignment` に不変なので同点になるのが普通で、勝たないと採れない
// 条件にすると一度も発動しない。盤が本当に画面外へ続いている中継なら
// 戻した窓は格子線から外れて信頼度が落ちるので、そちらは元のまま残る。
// **はみ出したままの領域はどのみち `ExtractCell` が画像外を読む**ので、
// 収まる候補が同点なら収まるほうが良い。
func unslipRegion(img image.Image, br *BoardRegion) *BoardRegion {
	if br == nil {
		return nil
	}
	b := img.Bounds()
	if br.Bounds.In(b) {
		return br
	}

	cw, ch := br.Bounds.Dx()/9, br.Bounds.Dy()/9
	var dx, dy int
	switch {
	case br.Bounds.Max.X > b.Max.X:
		dx = -cw
	case br.Bounds.Min.X < b.Min.X:
		dx = cw
	}
	switch {
	case br.Bounds.Max.Y > b.Max.Y:
		dy = -ch
	case br.Bounds.Min.Y < b.Min.Y:
		dy = ch
	}
	if dx == 0 && dy == 0 {
		return br
	}

	// 戻しても収まらない（＝盤が画像より大きい）なら手の出しようが無い
	r := br.Bounds.Add(image.Pt(dx, dy))
	if !r.In(b) {
		return br
	}
	cand := BoardRegionFromRect(r.Min.X, r.Min.Y, r.Max.X, r.Max.Y)
	if ValidateBoard(img, cand) < ValidateBoard(img, br) {
		return br
	}
	return cand
}

// refineRegion は外枠を整合度が最大になる位置に寄せ、その領域と信頼度を返す。
//
// **投影のピークを拾って等間隔に割るところまでで 0.3〜0.5マスの誤差が残る。**
// 1マスに直すと数px でも、`gridAlignment` は境界線が格子線に ±3px で
// 乗っているかを見るので、この程度のずれで整合度が 0 付近まで落ちる。
// 実測（中継画像 `26e06136`）: 正しい ROI から得た領域が
// dw=+0.29 dh=+0.45 で整合度 -0.06、微調整すると 0.32マス・信頼度 1.00 になる。
//
// 探索は投影の上だけで行う（原点 ±半マス・間隔 ±10%）ので、
// `lineProjections` を 1 回足すだけで済む。
// **整合度が上がらなければ元の領域を返す**（縦横比が崩れる場合も同様）。
func refineRegion(img image.Image, br *BoardRegion) (*BoardRegion, float64) {
	base := ValidateBoard(img, br)

	// 探索の余地として上下左右に半マス広げた範囲で投影を取る
	padX := br.Bounds.Dx() / 18
	padY := br.Bounds.Dy() / 18
	b := image.Rect(
		br.Bounds.Min.X-padX, br.Bounds.Min.Y-padY,
		br.Bounds.Max.X+padX, br.Bounds.Max.Y+padY,
	).Intersect(img.Bounds())
	if b.Dx() < 9*3 || b.Dy() < 9*3 {
		return br, base
	}

	sub := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			sub.Set(x-b.Min.X, y-b.Min.Y, img.At(x, y))
		}
	}
	row, col := lineProjections(BoxBlur(ConvertGray(sub), 2))

	oy, sy := refineAxis(row, float64(br.Bounds.Min.Y-b.Min.Y), float64(br.Bounds.Dy())/9)
	ox, sx := refineAxis(col, float64(br.Bounds.Min.X-b.Min.X), float64(br.Bounds.Dx())/9)

	refined := BoardRegionFromRect(
		b.Min.X+int(math.Round(ox)), b.Min.Y+int(math.Round(oy)),
		b.Min.X+int(math.Round(ox+sx*9)), b.Min.Y+int(math.Round(oy+sy*9)),
	)
	if !plausibleAspect(refined) {
		return br, base
	}
	if conf := ValidateBoard(img, refined); conf > base {
		return refined, conf
	}
	return br, base
}

// refineAxis は 1 軸分の (原点, 間隔) を整合度が最大になるように動かす
func refineAxis(proj []float64, origin, span float64) (float64, float64) {
	bestO, bestS := origin, span
	best := axisAlignment(proj, origin, span)
	for do := -span / 2; do <= span/2; do++ {
		for ds := -span * 0.1; ds <= span*0.1; ds += 0.25 {
			s := span + ds
			if s < 3 {
				continue
			}
			if v := axisAlignment(proj, origin+do, s); v > best {
				best, bestO, bestS = v, origin+do, s
			}
		}
	}
	return bestO, bestS
}

// aspectTolerance はマスの縦横比が boardAspect からずれてよい割合。
// 保存済み局面の手動指定座標の実測が 1.045〜1.138（boardAspect の -4%〜+4%）なので
// 余裕を持って ±20% とする
const aspectTolerance = 0.2

// plausibleAspect は候補のマスが将棋盤の縦横比に近いかを返す。
// ROI を絞った先で盤以外（UI パネルなど）の格子が拾われたときに落とす
func plausibleAspect(br *BoardRegion) bool {
	if br == nil || br.Bounds.Dx() <= 0 || br.Bounds.Dy() <= 0 {
		return false
	}
	a := (float64(br.Bounds.Dy()) / 9) / (float64(br.Bounds.Dx()) / 9)
	return a >= boardAspect*(1-aspectTolerance) && a <= boardAspect*(1+aspectTolerance)
}

// detectBoardIn は指定範囲の中だけで盤面領域を検出する
func detectBoardIn(edges *image.Gray, bounds image.Rectangle) *BoardRegion {
	bounds = bounds.Intersect(edges.Bounds())
	w, h := bounds.Dx(), bounds.Dy()
	if w < 9*3 || h < 9*3 {
		return nil
	}

	minDist := minInt(w, h) / 40
	if minDist < 3 {
		minDist = 3
	}

	// Pass 1: 横方向の投影で水平線10本（外枠2本 + 内部8本）を検出。
	//
	// **上限は min(w,h)/9 ではなく h/9。** 行の間隔はマスの「高さ」なので
	// 縦の長さで抑えるのが正しい。マスは縦長（boardAspect）なうえ、盤だけを
	// 切り出した画像は縦長になるので min(w,h)=幅 とすると
	// 上限 < 真のマス高 となり、**正しい間隔が必ず弾かれていた**
	// （保存済み 11 件中 9 件が該当。ピーク自体は全て拾えているのに
	// 偽の並びが選ばれていた）。
	rowProj := make([]int, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			rowProj[y] += int(edges.GrayAt(x+bounds.Min.X, y+bounds.Min.Y).Y)
		}
	}
	rowPeaks := findPeaks(rowProj, minDist)
	hPos, hSpacing := pickWithSpacing(rowPeaks, rowProj, 10, 0, float64(h)/9)
	if hPos == nil {
		// 外枠が写っていない画像向け: 内部8本を拾って上下に1マス延長する
		var hInner []int
		hInner, hSpacing = pickWithSpacing(rowPeaks, rowProj, 8, 0, float64(h)/9)
		if hInner == nil {
			return nil
		}
		hPos = extendToBorders(hInner, 0, h, rowProj)
	}

	// Pass 2: 行範囲内の縦投影から垂直線10本を検出。
	// 期待するマス幅は hSpacing/boardAspect（正方形ではない）
	yTop := hPos[0]
	yBot := hPos[len(hPos)-1]
	colProj := make([]int, w)
	for x := 0; x < w; x++ {
		for y := maxInt(yTop, 0); y <= minInt(yBot, h-1); y++ {
			colProj[x] += int(edges.GrayAt(x+bounds.Min.X, y+bounds.Min.Y).Y)
		}
	}

	colPeaks := findPeaks(colProj, minDist)
	cellW := hSpacing / boardAspect
	vPos, _ := pickWithSpacing(colPeaks, colProj, 10, cellW, float64(w)/9)
	if vPos == nil {
		vPos = findBoardColumns(colProj, colPeaks, cellW, w)
		if vPos == nil {
			return nil
		}
	}

	// **検出したピーク位置をそのままマスの境界に使わない。** 外枠だけを採り、
	// 内側は等分する。ピークは線の太さやボケで数px 揺れるので、そのまま使うと
	// 1マスの高さが 47〜58px のようにばらつき、グリッド線がマス内に入り込む。
	// 手動指定（BoardRegionFromRect）と同じ割り方になるので、
	// 「手動座標なら合うのに検出だと合わない」の切り分けもしやすい。
	// 実測での改善は誤認識マス 266 → 258（保存済み10局面・810マス）と小さい。
	return BoardRegionFromRect(
		vPos[0]+bounds.Min.X,
		hPos[0]+bounds.Min.Y,
		vPos[9]+bounds.Min.X,
		hPos[9]+bounds.Min.Y,
	)
}

// findBoardColumns は等間隔10本を取れなかったときのフォールバック。
// 期待幅 cellW*9 に収まる左右の境界ペアを探し、見つからなければ
// スライディングウィンドウで最もエッジ密度の高い位置を採る
func findBoardColumns(colProj []int, colPeaks []int, cellW float64, w int) []int {
	expectedW := int(cellW * 9)
	if expectedW <= 0 || expectedW > w {
		return nil
	}
	tolerance := int(cellW)

	bestScore := 0
	bestLeft, bestRight := 0, 0
	for _, l := range colPeaks {
		for _, r := range colPeaks {
			span := r - l
			if span < expectedW-tolerance || span > expectedW+tolerance {
				continue
			}
			if score := colProj[l] + colProj[r]; score > bestScore {
				bestScore, bestLeft, bestRight = score, l, r
			}
		}
	}

	if bestRight <= bestLeft {
		for x := 0; x <= w-expectedW; x++ {
			sum := 0
			for dx := 0; dx < expectedW; dx++ {
				sum += colProj[x+dx]
			}
			if sum > bestScore {
				bestScore, bestLeft, bestRight = sum, x, x+expectedW
			}
		}
		if bestRight <= bestLeft {
			return nil
		}
	}

	boardW := bestRight - bestLeft
	vPos := make([]int, 10)
	for i := 0; i < 10; i++ {
		vPos[i] = bestLeft + boardW*i/9
	}
	return vPos
}

// findPeaks は投影の局所最大を返す。
//
// **端も走査対象にする（比較窓を配列内に切り詰める）。** かつては
// `for i := minDist; i < len(signal)-minDist` として端を捨てていたが、
// 盤だけを切り出した画像では盤の外枠線が画像端（実測で x=2 / y=4 など）に
// 来るため、外枠が原理的にピークになれなかった。保存済み局面は 11 件中 9 件が
// この形（盤が画像の 92〜98% を占める）で、横方向が常に1マスずれる原因だった。
func findPeaks(signal []int, minDist int) []int {
	maxVal := 0
	for _, v := range signal {
		if v > maxVal {
			maxVal = v
		}
	}
	thresh := maxVal / 5

	peaks := make([]int, 0)
	for i := 0; i < len(signal); i++ {
		if signal[i] < thresh {
			continue
		}
		isPeak := true
		for d := 1; d <= minDist && isPeak; d++ {
			if i-d >= 0 && signal[i] < signal[i-d] {
				isPeak = false
			}
			if i+d < len(signal) && signal[i] < signal[i+d] {
				isPeak = false
			}
		}
		if isPeak {
			peaks = append(peaks, i)
		}
	}
	return peaks
}

// pickWithSpacing は投影のピーク列から等間隔 count 本の並びを選ぶ。
//
// **候補の順位は「何本合ったか」ではなく「どれだけぴったり合ったか」で決める。**
// 許容ずれは間隔の 30% と広いので、ピークが密な画像では**でたらめな間隔でも
// 10本すべてが何かに当たる**。本数を先に見ると、この水増しした並びが
// 「9本だが誤差ほぼ 0」の本物の格子に勝ってしまう。実測（`70f42d2474d2c171`）:
// 本物は間隔 75.0・平均ずれ 0.04マス・強度合計 519771 なのに、
// 間隔 60.2（平均ずれ 0.14マス・合計 500913）に本数で負けて、
// **盤の 1/4 が信頼度 1.00 で返っていた**。
//
// そこで各ピークの強度に「ずれ」で決まる重み（`pickTaper`）を掛けて足す。
// ぴったり合ったピークほど満額に近くなるので、**本数を稼ぐだけの並びは自然に沈む**。
// pickTaper はピークの強度を「期待位置からのずれ」で割り引く強さ。
// 重みは 1 - pickTaper*(ずれ/許容) で、0 なら割り引かない（＝強度の合計そのまま）、
// 1 なら許容の端で 0 になる。
//
// 実測（保存済み33局面・手動座標との比較）:
//
//	taper   0.5マス以内   黙って通した誤検出
//	0.00     31/33          1
//	**0.25**  **32/33**      **0**
//	0.50     31/33          2
//	1.00     29/33          4
//
// **強くしすぎると1マスずれが増える。** 盤が画像いっぱいに写っていると、
// 端の1本を捨てて9本にぴったり合わせた並びのほうが 10本の並びより高くなり、
// 格子の周期は正しいまま窓が1マス横に滑る。この滑りは周期が正しいので
// `axisAlignment` でも見分けられず（格子線には乗っている）、
// **信頼度 1.00 のまま間違える**という半周期より質の悪い壊れ方になる。
const pickTaper = 0.25

func pickWithSpacing(peaks []int, signal []int, count int, targetSpacing float64, maxSpacing float64) ([]int, float64) {
	n := len(peaks)
	minFound := count * 6 / 10
	if minFound < 5 {
		minFound = 5
	}
	if n < minFound {
		return nil, 0
	}

	bestScore := 0.0
	bestSpacing := 0.0
	bestOrigin := 0.0
	bestSlots := make([]int, count)

	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			diff := float64(peaks[j] - peaks[i])
			for gap := 1; gap <= count-1; gap++ {
				spacing := diff / float64(gap)
				if spacing < 10 {
					continue
				}

				if maxSpacing > 0 && spacing > maxSpacing {
					continue
				}
				if targetSpacing > 0 {
					ratio := spacing / targetSpacing
					if ratio < 0.7 || ratio > 1.4 {
						continue
					}
				}

				tolerance := spacing * 0.3
				origin := float64(peaks[i])
				found := 0
				score := 0.0
				slots := make([]int, count)

				for k := 0; k < count; k++ {
					expected := origin + float64(k)*spacing
					bestDist := tolerance
					bestIdx := -1
					for idx, p := range peaks {
						d := math.Abs(float64(p) - expected)
						if d < bestDist {
							bestDist = d
							bestIdx = idx
						}
					}
					if bestIdx >= 0 {
						slots[k] = peaks[bestIdx]
						found++
						score += float64(signal[peaks[bestIdx]]) * (1 - pickTaper*bestDist/tolerance)
					} else {
						slots[k] = -1
					}
				}

				if found < minFound {
					continue
				}
				if score > bestScore {
					bestScore = score
					bestSpacing = spacing
					bestOrigin = origin
					copy(bestSlots, slots)
				}
			}
		}
	}

	if bestSpacing == 0 {
		return nil, 0
	}

	result := make([]int, count)
	for k := 0; k < count; k++ {
		if bestSlots[k] >= 0 {
			result[k] = bestSlots[k]
		} else {
			result[k] = int(bestOrigin + float64(k)*bestSpacing)
		}
	}

	return result, bestSpacing
}

func extendToBorders(inner []int, minBound, maxBound int, proj []int) []int {
	n := len(inner)
	spacing := float64(inner[n-1]-inner[0]) / float64(n-1)

	result := make([]int, n+2)

	// 上端: 1マス分延長するが、画像端に近すぎる場合は半マス
	top := int(float64(inner[0]) - spacing)
	if top < minBound {
		top = int(float64(inner[0]) - spacing/2)
		if top < minBound {
			top = minBound
		}
	}
	result[0] = top

	for i, v := range inner {
		result[i+1] = v
	}

	// 下端: 同様
	bot := int(float64(inner[n-1]) + spacing)
	if bot > maxBound {
		bot = int(float64(inner[n-1]) + spacing/2)
		if bot > maxBound {
			bot = maxBound
		}
	}
	result[n+1] = bot

	return result
}

// BoardRegionFromRect は手動指定の2点からBoardRegionを作成する
func BoardRegionFromRect(x1, y1, x2, y2 int) *BoardRegion {
	left, right := x1, x2
	if left > right {
		left, right = right, left
	}
	top, bottom := y1, y2
	if top > bottom {
		top, bottom = bottom, top
	}
	w := right - left
	h := bottom - top
	br := &BoardRegion{Bounds: image.Rect(left, top, right, bottom)}
	for r := 0; r < 9; r++ {
		for c := 0; c < 9; c++ {
			br.Cells[r][c] = image.Rect(
				left+w*c/9, top+h*r/9,
				left+w*(c+1)/9, top+h*(r+1)/9,
			)
		}
	}
	return br
}

// cellPadding はセル抽出時に各辺に加える余白の割合（0.15 = 15%）
const cellPadding = 0.15

// ExtractCell はマス画像を切り出す。cellPadding 分だけ外側まで含めるが、
// **盤の外枠より外へは出さない。**
//
// 余白を含めるのは `cellBoardColor`（マス画像の中央値＝局所地色）のためで、
// 駒がマスの大半を覆っていても地色の画素が中央値を取れる程度に残るようにしている。
// ところが**外周のマスでは、その余白が盤の外（枠線・盤外の背景）を読んでいた。**
// 枠が暗ければ局所地色ごと暗いほうへ引きずられ、許容帯（±`pieceDiffFrac`）の
// 中心が盤の地色から外れる。すると**盤の木地も駒も両方「地色と異なる画素」に
// なり**、被覆率が跳ね上がって `pieceSideOf` のマスクが濁る。
//
// 実測（桜の絵柄の盤 `ad2394e4`、盤の地色 165）: 1 段目のマスの局所地色が
// 127〜147 まで落ち、被覆率が 0.73（正常な駒は 0.45〜0.65）になって
// **後手の駒 7 枚が先手に反転**していた。空→駒 の誤りも外周に偏る。
//
// 盤の外枠で切ると、外周マスは内側にしか余白が付かなくなる。
// 実測（保存済み 118 局面 / 9558 マス、手動座標・分類器のみ）:
// 空→駒 26 → 16 件、向きの反転 242 → 233 件。
// `BoardEmptyCover` と合わせて 97.11% → 97.39%。
//
// **なお `cellBoardColor` 自体を「余白を除いた範囲」で取るのは駄目**（試した）。
// 駒がマスの大半を覆うマスで中央値が駒の色になり、マスクが全反転する。
// 向きの反転が 242 → 623 件に増える。
//
// **外周マスの入力ベクトルが変わるので、学習データの版を上げること**
// （v5 → v6。CLAUDE.md「シフト不変性」の末尾）。
func (br *BoardRegion) ExtractCell(src image.Image, row, col int) image.Image {
	return br.extractCell(src, row, col, true)
}

// extractCellUnclipped は盤の外枠より外も含めてマスを切り出す。
//
// **`ValidateBoard` の `cellUniformity` はこちらを使う。安易に `ExtractCell` へ
// 寄せないこと。** そちらは「この領域が本当に盤か」を測るもので、**領域が
// 間違っている前提**で呼ばれる。外枠で切ると外れた候補も自分の箱の中しか
// 見なくなり、盤の外の畳や UI が写り込んでいても均一に見えてしまう。
// 実測（`5795bf16`）: クリップした版で測ると、正しい候補（dx=0.00）ではなく
// 1.01マスずれた候補が信頼度 1.00 で選ばれるようになる。
func (br *BoardRegion) extractCellUnclipped(src image.Image, row, col int) image.Image {
	return br.extractCell(src, row, col, false)
}

func (br *BoardRegion) extractCell(src image.Image, row, col int, clip bool) image.Image {
	cell := br.Cells[row][col]
	pad := int(float64(cell.Dx()) * cellPadding)
	expanded := image.Rect(
		cell.Min.X-pad, cell.Min.Y-pad,
		cell.Max.X+pad, cell.Max.Y+pad,
	)
	rect := expanded.Intersect(src.Bounds())
	if clip {
		rect = rect.Intersect(br.Bounds)
	}
	if rect.Empty() {
		return nil
	}
	dst := image.NewRGBA(image.Rect(0, 0, rect.Dx(), rect.Dy()))
	for y := rect.Min.Y; y < rect.Max.Y; y++ {
		for x := rect.Min.X; x < rect.Max.X; x++ {
			dst.Set(x-rect.Min.X, y-rect.Min.Y, src.At(x, y))
		}
	}
	return dst
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
