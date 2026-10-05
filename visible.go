package suteme

import (
	"image"
	"image/draw"
	"math"
	"sort"
)

// 「このマスは見えない」の判定。
//
// 中継では、指している手や解説者の頭が盤に被った 1 枚が撮れる。認識はどのマスも
// 「空き・先手・後手」のどれかに決めるので、**手や頭は空きか、どちらかの駒に化ける**
// （手で隠れた飛車が空き、指先が先手の金、髪が後手の歩）。呼び出し側（ikkyoku）は
// それをそのまま手の根拠にしてしまうので、**被っているマスに印を付けて返す**。
//
// **色では見分けられない**（髪は黒いが、禿頭や手は肌色で盤の地色に近い）。
// 手掛かりは**罫線が見えているか**で、それも**罫線の交点の近く**で測る:
//
//   - 駒はマスの中央に置かれる。**ゲーム画面や木の盤では駒がマスより大きく、
//     辺の中ほどの罫線は駒が覆う**ので、「マスの四辺の罫線」では駒と手を見分けられない
//     （実測: 保存済み 206 局面で、隠れていないのに四辺の最小が 0.05 未満のマスが 841）
//   - 交点は 4 つのマスの角が集まる所で、**駒から最も遠い**。駒が交点まで覆うのは
//     まれだが、手や頭は交点ごと覆う
//
// 交点から伸びる 4 本の腕（`armFrom`〜`armTo`）で、罫線が見えている点の割合を測り、
// **どの腕もろくに見えていなければその交点は隠れている**。マスは 4 隅のどれかの交点が
// 隠れていれば「見えない」。

const (
	// visScanWin は罫線の中心を探す範囲（マス間隔に対する割合。推定位置の ±）。
	// マス割りは外枠を 9 等分したものなので、実際の罫線とは数px ずれる。
	visScanWin = 0.06
	// visLineHalf は罫線の半幅（同上）。その外側を罫線の両脇として測る。
	visLineHalf = 0.025
	// visSideWidth は罫線の両脇を測る幅（同上）。
	visSideWidth = 0.08
	// armFrom / armTo は交点から腕を測る範囲（同上）。交点のすぐ近くは
	// 直交する罫線が乗るので測らない。
	armFrom = 0.08
	armTo   = 0.30
	// visLineFrac は「罫線が見えている」とみなすコントラスト
	// （その盤の罫線のコントラストの中央値に対する割合）。
	visLineFrac = 0.4
	// visLineMin はその下限（輝度の差）。
	visLineMin = 4
	// armSeenMin は腕が「見えている」とみなす割合。交点の腕がどれもこれ未満なら隠れている。
	armSeenMin = 0.5
	// frameArmSeenMin は外枠の上の交点に使う割合。外枠の外は盤の外なので
	// 内側だけと比べることになり、測りにくい（盤が画像の端で切れている・外が暗い）。
	frameArmSeenMin = 0.3
	// solidDarkFrac は「太い暗い塊」とみなす輝度（盤の地色に対する割合）。
	solidDarkFrac = 0.5
	// solidRadius は塊の太さの篩（マス間隔に対する割合。この半径の正方形が
	// 丸ごと暗い画素だけを残す）。
	solidRadius = 0.08
)

// lineProber は罫線のコントラストを点ごとに測る道具。
type lineProber struct {
	g      *image.Gray
	xs, ys [10]int     // 縦線の x / 横線の y
	pw, ph float64     // マスの幅 / 高さ
	solid  *image.Gray // 太い暗い塊（髪など）。0 以外 = 塊の中
}

func newLineProber(g *image.Gray, br *BoardRegion) *lineProber {
	p := &lineProber{g: g}
	for i := 0; i < 9; i++ {
		p.xs[i] = br.Cells[0][i].Min.X
		p.ys[i] = br.Cells[i][0].Min.Y
	}
	p.xs[9] = br.Cells[0][8].Max.X
	p.ys[9] = br.Cells[8][0].Max.Y
	p.pw = float64(p.xs[9]-p.xs[0]) / 9
	p.ph = float64(p.ys[9]-p.ys[0]) / 9
	return p
}

func (p *lineProber) at(x, y int) (int, bool) {
	b := p.g.Bounds()
	if !(image.Point{x, y}.In(b)) {
		return 0, false
	}
	return int(p.g.Pix[(y-b.Min.Y)*p.g.Stride+(x-b.Min.X)]), true
}

// sample は罫線 line（横線なら horiz=true で ys[line]、縦線なら xs[line]）の、
// 線に沿った座標 t における「線の暗さ」（両脇の明るさ − 線の明るさ）を返す。
//
// **線の中心を ±visScanWin の中で走査し、その中心から見た両脇で比べる。**
// 推定位置から固定の範囲で「最も暗い値」と「その外側」を比べると、
// 線のすぐ脇にある駒の輪郭を線と取り違え、外側の明るさも輪郭に引っ張られて、
// 見えている罫線を「見えない」と読む。
//
// **両脇は平均ではなく最も明るい値を取る**（同じ理由。脇の輪郭や木目で暗く引っ張られない）。
// **外枠は盤の内側だけと比べる**（外は盤の外で、何があるか分からない）。
func (p *lineProber) sample(horiz bool, line, t int) (float64, bool) {
	pos, pitch := p.xs[line], p.pw
	if horiz {
		pos, pitch = p.ys[line], p.ph
	}
	win := maxInt(2, int(math.Round(pitch*visScanWin)))
	k := maxInt(2, int(math.Round(pitch*visLineHalf)))
	side := maxInt(3, int(math.Round(pitch*visSideWidth)))
	at := func(d int) (int, bool) {
		if horiz {
			return p.at(t, pos+d)
		}
		return p.at(pos+d, t)
	}
	brightest := func(d0, d1 int) (int, bool) {
		mx := -1
		for d := d0; d <= d1; d++ {
			if v, ok := at(d); ok && v > mx {
				mx = v
			}
		}
		return mx, mx >= 0
	}
	best, found := math.Inf(-1), false
	for d := -win; d <= win; d++ {
		v, ok := at(d)
		if !ok {
			continue
		}
		low, okL := brightest(d-k-side, d-k)
		high, okH := brightest(d+k, d+k+side)
		var bg int
		switch {
		case line == 0 && okH:
			bg = high
		case line == 9 && okL:
			bg = low
		case line != 0 && line != 9 && okL && okH:
			bg = minInt(low, high)
		default:
			continue
		}
		if c := float64(bg - v); c > best {
			best, found = c, true
		}
	}
	return best, found
}

// lineThreshold は盤全体の罫線のコントラストの中央値から「見えている」のしきい値を決める。
// 同じ画像の中での比なので、解像度・ボケ・盤の配色に依らない。
func (p *lineProber) lineThreshold() float64 {
	var all []float64
	for line := 0; line < 10; line++ {
		for t := p.xs[0]; t < p.xs[9]; t++ {
			if c, ok := p.sample(true, line, t); ok {
				all = append(all, c)
			}
		}
		for t := p.ys[0]; t < p.ys[9]; t++ {
			if c, ok := p.sample(false, line, t); ok {
				all = append(all, c)
			}
		}
	}
	if len(all) == 0 {
		return visLineMin
	}
	sort.Float64s(all)
	return math.Max(all[len(all)/2]*visLineFrac, visLineMin)
}

// buildSolidDark は「盤の地色よりずっと暗い画素」を太さで篩い、太い塊だけを残す。
//
// **髪のような暗い塊が外枠に被ると、塊の縁が罫線に見える**（盤の内側と比べると
// 暗いので）。駒の字・罫線は細いので篩で消え、髪や袖は残る。塊の上の点は
// 罫線が見えていないものとして数える。
func (p *lineProber) buildSolidDark(level uint8, r int) {
	g := p.g
	b := g.Bounds()
	w, h := b.Dx(), b.Dy()
	ii := make([]int32, (w+1)*(h+1)) // 暗い画素の数の積分画像
	for y := 0; y < h; y++ {
		var row int32
		for x := 0; x < w; x++ {
			if g.Pix[y*g.Stride+x] < level {
				row++
			}
			ii[(y+1)*(w+1)+x+1] = ii[y*(w+1)+x+1] + row
		}
	}
	out := image.NewGray(b)
	area := int32((2*r + 1) * (2*r + 1))
	for y := r; y < h-r; y++ {
		for x := r; x < w-r; x++ {
			x0, y0, x1, y1 := x-r, y-r, x+r+1, y+r+1
			if ii[y1*(w+1)+x1]-ii[y0*(w+1)+x1]-ii[y1*(w+1)+x0]+ii[y0*(w+1)+x0] == area {
				out.Pix[y*out.Stride+x] = 1
			}
		}
	}
	p.solid = out
}

func (p *lineProber) solidAt(x, y int) bool {
	if p.solid == nil {
		return false
	}
	b := p.solid.Bounds()
	if !(image.Point{x, y}.In(b)) {
		return false
	}
	return p.solid.Pix[(y-b.Min.Y)*p.solid.Stride+(x-b.Min.X)] != 0
}

// armSeen は罫線 line の t0..t1 の区間で、罫線が見えている点の割合を返す（測れなければ -1）。
func (p *lineProber) armSeen(horiz bool, line, t0, t1 int, th float64) float64 {
	if t0 > t1 {
		t0, t1 = t1, t0
	}
	n, seen := 0, 0
	for t := t0; t <= t1; t++ {
		c, ok := p.sample(horiz, line, t)
		if !ok {
			continue
		}
		n++
		x, y := p.xs[line], t
		if horiz {
			x, y = t, p.ys[line]
		}
		if c >= th && !p.solidAt(x, y) {
			seen++
		}
	}
	if n == 0 {
		return -1
	}
	return float64(seen) / float64(n)
}

// intersectionArms は交点 (i=縦線, j=横線) から伸びる腕（左・右・上・下）それぞれで
// 罫線が見えている割合を返す。盤の外へ出る腕と測れない腕は -1。
func (p *lineProber) intersectionArms(i, j int, th float64) [4]float64 {
	out := [4]float64{-1, -1, -1, -1}
	x, y := p.xs[i], p.ys[j]
	ax0, ax1 := int(math.Round(p.pw*armFrom)), int(math.Round(p.pw*armTo))
	ay0, ay1 := int(math.Round(p.ph*armFrom)), int(math.Round(p.ph*armTo))
	if i > 0 {
		out[0] = p.armSeen(true, j, x-ax1, x-ax0, th)
	}
	if i < 9 {
		out[1] = p.armSeen(true, j, x+ax0, x+ax1, th)
	}
	if j > 0 {
		out[2] = p.armSeen(false, i, y-ay1, y-ay0, th)
	}
	if j < 9 {
		out[3] = p.armSeen(false, i, y+ay0, y+ay1, th)
	}
	return out
}

// intersectionSeen は 10x10 の交点それぞれについて、最もよく見えている腕の割合を返す
// （添字は [横線][縦線]。腕が 1 本も測れなければ -1）。
func intersectionSeen(img image.Image, br *BoardRegion, boardColor uint8) [10][10]float64 {
	// 盤の外枠から 1 マス外までを灰色にする（画像全体は要らない）
	pad := image.Pt(int(math.Ceil(float64(br.Bounds.Dx())/9)), int(math.Ceil(float64(br.Bounds.Dy())/9)))
	rect := image.Rectangle{br.Bounds.Min.Sub(pad), br.Bounds.Max.Add(pad)}.Intersect(img.Bounds())
	g := image.NewGray(rect)
	draw.Draw(g, rect, img, rect.Min, draw.Src)

	p := newLineProber(g, br)
	th := p.lineThreshold()
	p.buildSolidDark(uint8(float64(boardColor)*solidDarkFrac),
		maxInt(2, int(math.Round(math.Min(p.pw, p.ph)*solidRadius))))
	var out [10][10]float64
	for j := 0; j < 10; j++ {
		for i := 0; i < 10; i++ {
			best := -1.0
			for _, v := range p.intersectionArms(i, j, th) {
				best = math.Max(best, v)
			}
			out[j][i] = best
		}
	}
	return out
}

// hiddenCells は 81マスそれぞれが「見えない」（4 隅のどれかの交点が隠れている）かを返す。
func hiddenCells(img image.Image, br *BoardRegion, boardColor uint8) [9][9]bool {
	seen := intersectionSeen(img, br, boardColor)
	var hid [10][10]bool
	for j := 0; j < 10; j++ {
		for i := 0; i < 10; i++ {
			min := armSeenMin
			if i == 0 || i == 9 || j == 0 || j == 9 {
				min = frameArmSeenMin
			}
			// 腕が 1 本も測れない交点（画像の外）は隠れているとはみなさない
			hid[j][i] = seen[j][i] >= 0 && seen[j][i] < min
		}
	}
	var out [9][9]bool
	for r := 0; r < 9; r++ {
		for c := 0; c < 9; c++ {
			out[r][c] = hid[r][c] || hid[r][c+1] || hid[r+1][c] || hid[r+1][c+1]
		}
	}
	return out
}

// HiddenCells は 81マスそれぞれが「見えない」（手や頭などが被っている）かを返す。
// 添字は [段][筋の左から]（[0][0] が 9一）。`Recognize` の `Debug.Cells[].Hidden` と同じもの。
func HiddenCells(img image.Image, br *BoardRegion) [9][9]bool {
	return hiddenCells(img, br, BoardColor(img, br))
}

// occludedCells は「罫線はあるのに、途中を何かが覆っている」マスを返す。
//
// hiddenCells との違いは**罫線そのものがあるか**を見ること。窓が盤からずれて
// 盤の外を含むと、その辺の罫線は丸ごと見えない（隠れているのではなく無い）。
// 線ごとに、測れる交点の半分以上で見えていれば「その線はある」とみなし、
// ある線どうしの交点が隠れているときだけ遮られているとする。
func occludedCells(img image.Image, br *BoardRegion, boardColor uint8) [9][9]bool {
	seen := intersectionSeen(img, br, boardColor)
	var hid, ok [10][10]bool // ok = 測れて見えている
	for j := 0; j < 10; j++ {
		for i := 0; i < 10; i++ {
			min := armSeenMin
			if i == 0 || i == 9 || j == 0 || j == 9 {
				min = frameArmSeenMin
			}
			hid[j][i] = seen[j][i] >= 0 && seen[j][i] < min
			ok[j][i] = seen[j][i] >= min
		}
	}
	exists := func(n, seenN int) bool { return n > 0 && seenN*2 >= n }
	var hx, vx [10]bool
	for k := 0; k < 10; k++ {
		hn, hs, vn, vs := 0, 0, 0, 0
		for m := 0; m < 10; m++ {
			if hid[k][m] || ok[k][m] {
				hn++
			}
			if ok[k][m] {
				hs++
			}
			if hid[m][k] || ok[m][k] {
				vn++
			}
			if ok[m][k] {
				vs++
			}
		}
		hx[k], vx[k] = exists(hn, hs), exists(vn, vs)
	}
	occ := func(i, j int) bool { return hid[j][i] && hx[j] && vx[i] }
	var out [9][9]bool
	for r := 0; r < 9; r++ {
		for c := 0; c < 9; c++ {
			out[r][c] = occ(c, r) || occ(c+1, r) || occ(c, r+1) || occ(c+1, r+1)
		}
	}
	return out
}
