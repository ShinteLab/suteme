package suteme

import (
	"image"
	"sort"
)

// 盤領域の絞り込み（線分抽出）
//
// **投影を画像の全幅で取ると、盤以外の横線が投影を支配する。** 実物の盤を撮った
// 中継画像（`26e06136`）では対局時計バー・畳の横目・駒台・タイトルパネルの罫線が
// 画像の全幅にわたって並ぶため、盤の格子線（画像の幅の半分しかない）は埋もれる。
// 実測で DetectBoard は dx=+3.57マス と外していた。
//
// **手掛かりは幾何。盤の格子線は同じ x 範囲で始まり終わる。** 色・輝度・エッジ密度は
// 盤と畳で分離しない（同じ木・同じ照明。実測で 32px ブロックの明るさ 6 対 5、
// 彩度 9 対 8、エッジ密度 1〜3 対 1〜2）が、線の端点は盤の幅で揃う。
// そこで横方向の線分を抜き出し、(左端, 右端) が近いものをまとめて
// 「同じ幅で何本も並んでいる範囲」を盤の候補とする。
//
// **候補は複数出して ValidateBoard で選ぶ。盤以外の線のほうが本数が多いので
// 最大クラスタを無条件に採ってはいけない。** ただしこの選択は候補が
// **x 範囲で異なる**ので成立する。位相（全マスシフト）で異なる候補を
// gridAlignment で選び直す試みは、整合度が全マスシフトに対して不変なため
// 失敗している（AGENTS.md「今後の課題」参照）。

// **`horizontalLines` が盤の格子線を拾えているかが全体の成否を決める。**
// 拾えなければ盤の x/y 範囲がクラスタに出ず、候補にすら入らないので
// `ValidateBoard` で選び直す機会が無い。閾値まわりを触るときは
// 「線が何本拾えたか」ではなく「**盤の範囲の線が拾えたか**」で確かめること
// （`segEdgePercentile` のコメントに実測がある）。

// segMinLenRatio は線分として拾う最小の長さ（画像幅に対する比）。
// 盤が画像の 4 割弱しか占めない中継画像でも拾えるように小さめに取る
const segMinLenRatio = 0.2

// segGapTol は線分の途中で途切れてよい画素数。
// 駒が格子線を隠す・木目で途切れる分を繋ぐ
const segGapTol = 8

// segEdgePercentile は水平エッジとみなす強さの分位。
//
// **絶対値のしきい値にしてはいけない。** 画像ごとにコントラストが違う。
// 分位なら「上位 30% が線」という相対的な意味になる。
//
// **0.90（上位 10%）では盤の格子線が一本も拾えない画像がある。安易に戻さないこと。**
// 分位は画像全体で取るので、UI パネル・画面枠・対局時計バーのような**強い罫線が
// 閾値を吊り上げる**。盤の格子線は木目やアンチエイリアスで弱いため、そこで切られる。
// 実測（`horizontalLines` が返した線分の位置）:
//
//	baeac294  盤は y 83..485 なのに、拾えた横線は y 12〜56（盤の上の UI）と y 484,493 のみ
//	b3d5ce3c  盤は y 38..754 なのに、y 8〜31（上部 UI）・y 263〜516（右パネル）・y 773〜775 のみ
//
// 線が無いのではなく分位の問題で、`detectBoardIn` の Sobel 投影は同じ格子線を
// ちゃんと拾える（正しい x 範囲を ROI に与えれば両方とも 0.2マス・信頼度 1.00）。
// 拾えなければ盤の x/y 範囲がクラスタに出ず、**候補にすら入らない**。
//
// 保存済み 89 局面での実測（0.5マス以内 / 黙って通した誤検出）:
//
//	| 分位 | 0.5マス以内 | 誤検出 |
//	|---|---|---|
//	| 0.90（旧）| 84/89 | 1 |
//	| 0.85 | 86/89 | 1 |
//	| **0.80 〜 0.55** | **88/89** | **1** |
//	| 0.50 | b3d5ce3c が崩れる（conf 0.11 で棄却されるので実害は無い）|
//
// 0.80〜0.55 は成績が完全に同じなので中央付近の 0.70 を採る。
// 下げても実行時間はほとんど変わらず（89 局面で 27s → 32s）、負例 2 枚の
// 誤検出も変わらない（どの分位でも 2/2 通る。別軸の既知の穴）。
const segEdgePercentile = 0.70

// segMaxCandidates は返す候補の数。
// 中継画像では盤（3〜5本）より画面枠（全幅・複数本）のほうが上位に来ることがある
const segMaxCandidates = 3

// segROIMargin は候補の x 範囲を左右に広げる割合。
// クラスタは格子線の端であって外枠ではないので、少し余裕を持たせる
const segROIMargin = 0.03

// boardROIs は線分から盤らしい領域の候補を返す（良い順）。
//
// 横線の x 範囲と縦線の y 範囲をそれぞれクラスタにして組み合わせる。
// **片方だけでは足りない。** 実測では x を絞ると横方向は合うようになるが、
// 盤の上下に UI や駒台があると行の投影がそちらの罫線を拾って 1〜1.5マスずれる
// （`b858741` / `baeac294`）。y も絞ると `b858741` が信頼度 1.00 で通る。
//
// **同じクラスタの線が「どこからどこまで並んでいるか」もそのまま候補にする**
// （`segSpan.from/to`）。横線のクラスタは盤の x 範囲を与えるが、その線が
// 並んでいる y の範囲は**盤の上下端そのもの**なので、縦線のクラスタが
// y 範囲を出せなくても盤を四方から囲める。
//
// これが無いと、x だけ絞った ROI（＝画像の高さいっぱい）で
// **`findPeaks` が盤の格子線を 1 本も返さない**ことがある。ピークの判定は
// 投影の最大値に対する相対値なので、盤の外にある強い罫線（画面下部の UI など）が
// 最大値を吊り上げると、盤の格子線がしきい値を下回って消える。
// 実測（`525c544f`、盤が画像の右 36%）:
//
//	ROI (635,0)-(1139,745)  ピーク6本 [3 612 658 659 704 743]  → 検出できず
//	ROI (635,34)-(1139,568) ピーク15本（盤の格子線が出る）     → conf 1.00
//
// 上の y 範囲 34..568 が、x=635..1139 に揃った横線の並びそのもの。
func boardROIs(blurred *image.Gray) []image.Rectangle {
	b := blurred.Bounds()
	xs := segmentSpans(blurred, segMaxCandidates)
	ys := segmentSpans(transposeGray(blurred), segMaxCandidates)

	rois := make([]image.Rectangle, 0, len(xs)*(len(ys)+2)+2*len(ys))
	for _, xr := range xs {
		// 線分の並びから四方を囲んだ候補。最も的を絞れているので先に置く
		rois = append(rois, image.Rect(xr.lo, xr.from, xr.hi, xr.to))
		rois = append(rois, image.Rect(xr.lo, b.Min.Y, xr.hi, b.Max.Y))
		for _, yr := range ys {
			rois = append(rois, image.Rect(xr.lo, yr.lo, xr.hi, yr.hi))
		}
	}
	for _, yr := range ys {
		// 縦線のクラスタ側も同じ。lo/hi が y 範囲、from/to が x 範囲
		rois = append(rois, image.Rect(yr.from, yr.lo, yr.to, yr.hi))
		rois = append(rois, image.Rect(b.Min.X, yr.lo, b.Max.X, yr.hi))
	}
	return rois
}

// segSpan は線分のクラスタ 1 つ。横線に使うと lo/hi が x 範囲、
// from/to がその線が並んでいる y 範囲（縦線では入れ替わる）。
type segSpan struct {
	lo, hi   int
	from, to int
}

// segmentSpans は横方向の線分の x 範囲をクラスタにして返す（良い順）。
// あわせて、その線分が並んでいる y の範囲も返す（`segSpan.from/to`）。
// 縦線に使うときは transposeGray した画像を渡す
func segmentSpans(blurred *image.Gray, max int) []segSpan {
	b := blurred.Bounds()
	lines := horizontalLines(blurred)
	if len(lines) == 0 {
		return nil
	}

	tol := b.Dx() / 50
	if tol < 10 {
		tol = 10
	}

	type cluster struct {
		x1, x2 int
		y1, y2 int
		count  int
	}
	clusters := make([]cluster, 0, len(lines))
	used := make([]bool, len(lines))
	for i, l := range lines {
		if used[i] {
			continue
		}
		used[i] = true
		c := cluster{x1: l.start, x2: l.end, y1: l.pos, y2: l.pos, count: 1}
		for j := i + 1; j < len(lines); j++ {
			if used[j] {
				continue
			}
			if absInt(lines[j].start-l.start) <= tol && absInt(lines[j].end-l.end) <= tol {
				used[j] = true
				c.count++
				if lines[j].pos < c.y1 {
					c.y1 = lines[j].pos
				}
				if lines[j].pos > c.y2 {
					c.y2 = lines[j].pos
				}
			}
		}
		clusters = append(clusters, c)
	}

	// 本数が多い順。同数なら幅の広いほうを先に見る
	sort.SliceStable(clusters, func(i, j int) bool {
		if clusters[i].count != clusters[j].count {
			return clusters[i].count > clusters[j].count
		}
		return clusters[i].x2-clusters[i].x1 > clusters[j].x2-clusters[j].x1
	})

	spans := make([]segSpan, 0, max)
	for _, c := range clusters {
		if len(spans) >= max {
			break
		}
		// 1 本しか無い範囲は「同じ幅で並んでいる」の根拠にならない
		if c.count < 2 {
			continue
		}
		margin := int(float64(c.x2-c.x1) * segROIMargin)
		x1 := maxInt(c.x1-margin, b.Min.X)
		x2 := minInt(c.x2+margin+1, b.Max.X)
		// 9 マスに割れない幅は候補にしない
		if x2-x1 < 9*3 {
			continue
		}
		// 線の並びの側も同じだけ広げる。クラスタが拾うのは格子線であって
		// 外枠とは限らないため（実測では余白 0〜10px のどちらでも通る）
		pm := int(float64(c.y2-c.y1) * segROIMargin)
		spans = append(spans, segSpan{
			lo: x1, hi: x2,
			from: maxInt(c.y1-pm, b.Min.Y),
			to:   minInt(c.y2+pm+1, b.Max.Y),
		})
	}
	return spans
}

// lineSegment は 1 本の横線。pos が y、start/end が x の範囲
type lineSegment struct {
	pos   int
	start int
	end   int
}

// horizontalLines は横方向に伸びる線分を返す。
//
// 縦方向の勾配 |gy| が強い画素を線の候補とし、行ごとに連続する範囲（run）を取る。
// **上下 1px の揺れは許す**（線は完全な水平ではないし、ボケで太る）。
// 近接した y の run は 1 本の線としてまとめる。
func horizontalLines(blurred *image.Gray) []lineSegment {
	b := blurred.Bounds()
	w, h := b.Dx(), b.Dy()
	if w < 3 || h < 3 {
		return nil
	}

	gy := make([]int, w*h)
	vals := make([]int, 0, w*h)
	for y := 1; y < h-1; y++ {
		for x := 1; x < w-1; x++ {
			X, Y := x+b.Min.X, y+b.Min.Y
			v := -int(blurred.GrayAt(X-1, Y-1).Y) - 2*int(blurred.GrayAt(X, Y-1).Y) - int(blurred.GrayAt(X+1, Y-1).Y) +
				int(blurred.GrayAt(X-1, Y+1).Y) + 2*int(blurred.GrayAt(X, Y+1).Y) + int(blurred.GrayAt(X+1, Y+1).Y)
			if v < 0 {
				v = -v
			}
			gy[y*w+x] = v
			vals = append(vals, v)
		}
	}
	if len(vals) == 0 {
		return nil
	}
	sort.Ints(vals)
	thresh := vals[int(float64(len(vals)-1)*segEdgePercentile)]
	if thresh < 1 {
		thresh = 1
	}

	minLen := int(float64(w) * segMinLenRatio)
	if minLen < 10 {
		minLen = 10
	}

	on := func(x, y int) bool {
		for dy := -1; dy <= 1; dy++ {
			if ny := y + dy; ny >= 0 && ny < h && gy[ny*w+x] >= thresh {
				return true
			}
		}
		return false
	}

	// 行ごとに最長の run だけを採る（盤の格子線は行の中で最も長いはず）
	rows := make([]lineSegment, 0, h)
	for y := 1; y < h-1; y++ {
		best := lineSegment{pos: y, start: 0, end: -1}
		for x := 0; x < w; {
			if !on(x, y) {
				x++
				continue
			}
			start, last := x, x
			for x < w {
				if on(x, y) {
					last = x
				} else if x-last > segGapTol {
					break
				}
				x++
			}
			if last-start > best.end-best.start {
				best = lineSegment{pos: y, start: start, end: last}
			}
		}
		if best.end-best.start >= minLen {
			rows = append(rows, lineSegment{pos: y, start: best.start + b.Min.X, end: best.end + b.Min.X})
		}
	}

	// 線の太さ・ボケで同じ線が数行にわたって出るのでまとめる
	tol := w / 50
	if tol < 10 {
		tol = 10
	}
	lines := make([]lineSegment, 0, len(rows))
	for _, r := range rows {
		if n := len(lines); n > 0 && r.pos-lines[n-1].pos <= segGapTol &&
			absInt(r.start-lines[n-1].start) < tol && absInt(r.end-lines[n-1].end) < tol {
			continue
		}
		lines = append(lines, r)
	}
	return lines
}

func absInt(a int) int {
	if a < 0 {
		return -a
	}
	return a
}
