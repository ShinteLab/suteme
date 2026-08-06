package suteme

import (
	"image"
	"sort"
)

// CellCategory はマスの大分類
type CellCategory int

const (
	CellEmpty     CellCategory = 0 // 空
	CellPieceUp   CellCategory = 1 // 先手（上向き）
	CellPieceDown CellCategory = 2 // 後手（下向き）
)

func (c CellCategory) String() string {
	switch c {
	case CellEmpty:
		return "空"
	case CellPieceUp:
		return "☗"
	case CellPieceDown:
		return "☖"
	}
	return "?"
}

const (
	// pieceDiffFrac は「盤の地色と異なる」と見なす明度差（地色に対する割合）。
	// 絶対値ではなく地色基準にすることで、画像ごとの明るさ差を吸収する
	pieceDiffFrac = 0.12
	// pieceDiffMin は明度差の下限（暗い盤で割合が小さくなりすぎるのを防ぐ）
	pieceDiffMin = 12.0
	// emptyCoverMax はこの被覆率未満を空マスと見なす。
	// 駒はマスの 40〜70% を覆うのに対し、空マスは盤の地色そのものなので
	// ほぼ 0 になる。間を広く取っている
	emptyCoverMax = 0.15
	// gridLineFrac はこの割合以上を占める行・列をグリッド線／盤外と見なして除外する。
	// グリッド線は行または列の全体を貫くが、駒はマスの端まで届かない
	gridLineFrac = 0.90
	// pieceExtentFrac は駒の上下端と見なす行の幅（最大幅に対する割合）
	pieceExtentFrac = 0.3
)

// ClassifyCellWith は盤の地色を与えてマス画像を 空/先手/後手 に分類する。
//
// 空判定は「地色と異なる画素の被覆率」で行う（分散の絶対値では画像の
// コントラストやスケールに依存して破綻する）。向き判定は駒マスクの
// 幅プロファイルの重心が駒の外接範囲の上下どちらに寄っているかで決める。
// 外接範囲を基準にするので、盤面領域が数 px ずれてマスと駒の位置が
// 合っていなくても結果が変わらない。
//
// **基準色はマスごとの局所地色（`cellBoardColor`）、許容幅は引数の盤全体の
// 地色から決める**（`newCellMaskSign`）。向き判定のマスクはさらに
// 「地色より暗い画素だけ」「明るい画素だけ」の 2 通りを作り、
// 被覆率の大きいほうを採る（`pieceSideOf`）。
func ClassifyCellWith(cell image.Image, boardColor uint8) CellCategory {
	local := cellBoardColor(cell)
	m := newCellMaskSign(cell, local, boardColor, signBoth)
	if m == nil || m.cover < emptyCoverMax {
		return CellEmpty
	}
	if m = pieceSideOf(cell, local, boardColor); m == nil {
		return CellEmpty
	}
	top, bottom, ok := m.span()
	if !ok {
		return CellEmpty
	}
	// 駒の五角形は尖り側が狭く底辺側が広いので、駒の外接範囲を上下に割ると
	// 底辺のある側の幅が広い。広いほうが下 = 底辺が下 = 先手
	if m.lowerIsWider(top, bottom) {
		return CellPieceUp
	}
	return CellPieceDown
}

// pieceSideOf は向き判定に使うマスクを返す。
//
// **駒の五角形の輪郭がどちら側に出るかは盤によって逆になる。**
// 木目の明るい盤（普通のゲーム画面）では駒のほうが地色より明るく、
// 背景に絵や光源を敷いた盤では駒のほうが暗い。両側をまとめてマスク化すると
// 五角形の輪郭に、反対側にしか出ないもの（墨の字画・駒の影・盤の模様）が
// 重なって幅プロファイルが濁る。
//
// どちら側かは「被覆率の大きいほう＝駒の本体が写っているほう」で決まる。
// 反対側は字画と影だけなので被覆率がはっきり小さい（実測で盤ごとに
// 0.30〜0.37 対 0.04〜0.16 と 2 倍以上開く）。
func pieceSideOf(cell image.Image, center, diffBase uint8) *cellMask {
	dark := newCellMaskSign(cell, center, diffBase, signDark)
	bright := newCellMaskSign(cell, center, diffBase, signBright)
	switch {
	case dark == nil:
		return bright
	case bright == nil:
		return dark
	case dark.cover > bright.cover:
		return dark
	}
	return bright
}

// ClassifyCell はマス画像を 空/先手/後手 に分類する。
// 盤の地色をマス画像単体から推定するため、盤面全体を扱えるなら
// BoardColor で求めた地色を ClassifyCellWith に渡すほうが安定する。
func ClassifyCell(cell image.Image) CellCategory {
	return ClassifyCellWith(cell, cellBoardColor(cell))
}

// ClassifyBoard は盤面の全81マスを分類する
func ClassifyBoard(img image.Image, br *BoardRegion) [9][9]CellCategory {
	var result [9][9]CellCategory
	bc := BoardColor(img, br)
	for r := 0; r < 9; r++ {
		for c := 0; c < 9; c++ {
			cell := br.ExtractCell(img, r, c)
			if cell != nil {
				result[r][c] = ClassifyCellWith(cell, bc)
			}
		}
	}
	return result
}

// BoardColor は盤の地色（グレースケール明度）を推定する。
// 盤面領域の全画素の中央値を採る。駒が置かれていても盤の地色が
// 最も広い領域を占めるため、中央値は安定して地色を指す
// （マスごとの中央値の中央値だと、駒を持つマスが半数を超える序盤で崩れる）。
func BoardColor(img image.Image, br *BoardRegion) uint8 {
	if img == nil || br == nil {
		return 0
	}
	return medianBrightnessRect(img, br.Bounds.Intersect(img.Bounds()))
}

// cellBoardColor はマス画像単体から盤の地色を推定する。
// ExtractCell は cellPadding 分外側まで含むので、駒がマスを覆っていても
// 地色の占める割合が中央値を取れる程度には残る。
//
// **空判定はこの局所地色を使う。盤にグラデーションのある画像では
// 盤全体の中央値では空マスを拾えない。安易に戻さないこと。**
// 実測（保存済み36局面）で、空マスの局所地色が盤の中で 65 も開く画像がある
// （全体地色 187 に対し 156〜221）。pieceDiffFrac は ±12% なので許容帯は
// 165〜209 にしかならず、**空マスの地色そのものが帯の外**に出て被覆率が
// 0.2〜0.6 まで上がり、駒として扱われていた。誤りは明るさに沿って偏り、
// 暗い側は 空→☖、明るい側は 空→☗ になる。
// 空マスの誤りが 208 → 2 件、全体が 88.9% → 95.7% になった。
func cellBoardColor(cell image.Image) uint8 {
	if cell == nil {
		return 0
	}
	return medianBrightnessRect(cell, cell.Bounds())
}

func medianBrightnessRect(img image.Image, rect image.Rectangle) uint8 {
	if rect.Empty() {
		return 0
	}
	var hist [256]int
	for y := rect.Min.Y; y < rect.Max.Y; y++ {
		for x := rect.Min.X; x < rect.Max.X; x++ {
			hist[grayValue(img.At(x, y))]++
		}
	}
	half := rect.Dx() * rect.Dy() / 2
	sum := 0
	for v, n := range hist {
		sum += n
		if sum > half {
			return uint8(v)
		}
	}
	return 255
}

// cellMask はマス画像から「盤の地色と異なる画素」のマスクを作り、
// 行ごとの駒の幅を保持する
type cellMask struct {
	extent []int   // 行ごとの駒の幅（除外した行は 0）
	cover  float64 // 有効領域に対する被覆率
}

// maskSign はマスクに含める画素の向き（地色より暗い/明るい）
type maskSign int

const (
	signBoth   maskSign = 0 // 地色と異なる画素すべて
	signDark   maskSign = 1 // 地色より暗い画素だけ
	signBright maskSign = 2 // 地色より明るい画素だけ
)

// newCellMaskSign は center を基準色、diffBase を許容幅の基準にして
// 分類用のマスクを作る。判定できない小さすぎるマスでは nil。
//
// **中心は局所地色、幅は盤全体の地色から決める。** グラデーションのある盤では
// マスごとに基準色が動くが、許容幅までマスごとに動かすと暗いマスだけ帯が狭くなり
// 木目を拾う。幅は盤の性質なので盤全体の地色から一度決めるほうが素直
// （実測でも局所から決めるより良い。CLAUDE.md の「駒分類」の節）。
func newCellMaskSign(cell image.Image, center, diffBase uint8, sign maskSign) *cellMask {
	if cell == nil {
		return nil
	}
	gray := ConvertGray(cell)
	b := trimCellPadding(gray.Bounds())
	w, h := b.Dx(), b.Dy()
	if w < 8 || h < 8 {
		return nil
	}

	diff := float64(diffBase) * pieceDiffFrac
	if diff < pieceDiffMin {
		diff = pieceDiffMin
	}
	lo := float64(center) - diff
	hi := float64(center) + diff

	marked := make([]bool, w*h)
	colCount := make([]int, w)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := float64(gray.GrayAt(b.Min.X+x, b.Min.Y+y).Y)
			var on bool
			switch sign {
			case signDark:
				on = v < lo
			case signBright:
				on = v > hi
			default:
				on = v < lo || v > hi
			}
			if on {
				marked[y*w+x] = true
				colCount[x]++
			}
		}
	}

	// 縦のグリッド線・盤外の帯を落とす（列のほぼ全高が埋まっている列）。
	// 先に列を処理しないと、1本の縦線で全行が「グリッド線」に見えてしまう
	colLimit := int(float64(h) * gridLineFrac)
	validW := 0
	for x := 0; x < w; x++ {
		if colCount[x] >= colLimit {
			for y := 0; y < h; y++ {
				marked[y*w+x] = false
			}
			continue
		}
		validW++
	}
	if validW < 8 {
		return nil
	}

	// 横のグリッド線を落としつつ、行ごとの幅を測る
	rowLimit := int(float64(validW) * gridLineFrac)
	extent := make([]int, h)
	count, validH := 0, 0
	for y := 0; y < h; y++ {
		row := marked[y*w : (y+1)*w]
		n := 0
		for _, v := range row {
			if v {
				n++
			}
		}
		if n >= rowLimit {
			continue
		}
		if left, right := runEdges(row); right > left {
			extent[y] = right - left + 1
		}
		count += n
		validH++
	}
	if validH < 8 {
		return nil
	}

	return &cellMask{extent: extent, cover: float64(count) / float64(validH*validW)}
}

// span は駒の上端・下端の行を返す。最大幅の pieceExtentFrac 未満の行は
// 駒の外（ノイズ）と見なす
func (m *cellMask) span() (top, bottom int, ok bool) {
	maxExtent := 0
	for _, e := range m.extent {
		if e > maxExtent {
			maxExtent = e
		}
	}
	if maxExtent == 0 {
		return 0, 0, false
	}
	limit := int(float64(maxExtent) * pieceExtentFrac)
	top, bottom = -1, -1
	for y, e := range m.extent {
		if e < limit {
			continue
		}
		if top < 0 {
			top = y
		}
		bottom = y
	}
	if top < 0 || bottom <= top {
		return 0, 0, false
	}
	return top, bottom, true
}

// lowerIsWider は駒の外接範囲を上下に割り、下半分のほうが広いかを返す。
//
// 幅の**中央値**で比べる。平均や重心だと、グリッド線や隣のマスから
// 入り込んだ駒の断片が 1〜2 行あるだけで結果が反転する。中央値なら
// そうした外れ行を無視できる（実測で向き判定 87.3% → 93.5%）。
func (m *cellMask) lowerIsWider(top, bottom int) bool {
	half := (top + bottom) / 2
	return m.medianExtent(half+1, bottom) >= m.medianExtent(top, half)
}

// medianExtent は lo..bottom の行の幅の中央値を返す
func (m *cellMask) medianExtent(lo, hi int) int {
	if hi < lo {
		return 0
	}
	v := make([]int, 0, hi-lo+1)
	for y := lo; y <= hi; y++ {
		v = append(v, m.extent[y])
	}
	sort.Ints(v)
	return v[len(v)/2]
}

// runEdges は 2 画素以上連続してマークされた最初と最後の位置を返す。
// 孤立したノイズ画素で幅が過大にならないようにする
func runEdges(row []bool) (left, right int) {
	left, right = -1, -1
	for x := 0; x+1 < len(row); x++ {
		if row[x] && row[x+1] {
			left = x
			break
		}
	}
	for x := len(row) - 1; x-1 >= 0; x-- {
		if row[x] && row[x-1] {
			right = x
			break
		}
	}
	return left, right
}

// trimCellPadding は ExtractCell が付けた cellPadding 相当の縁を除いた
// 範囲（＝元のマスの範囲）を返す
func trimCellPadding(outer image.Rectangle) image.Rectangle {
	// padding込みサイズに対する縁の割合: pad / (cell + 2*pad)
	trimFrac := cellPadding / (1 + 2*cellPadding)
	trimX := int(float64(outer.Dx()) * trimFrac)
	trimY := int(float64(outer.Dy()) * trimFrac)
	return image.Rect(
		outer.Min.X+trimX, outer.Min.Y+trimY,
		outer.Max.X-trimX, outer.Max.Y-trimY,
	)
}
