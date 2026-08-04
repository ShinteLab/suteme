package suteme

import (
	"image"
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
func ClassifyCellWith(cell image.Image, boardColor uint8) CellCategory {
	m := newCellMask(cell, boardColor)
	if m == nil || m.cover < emptyCoverMax {
		return CellEmpty
	}
	top, bottom, ok := m.span()
	if !ok {
		return CellEmpty
	}
	// 駒の五角形は尖り側が狭く底辺側が広いので、幅で重み付けした重心は
	// 底辺の側に寄る。重心が上寄り = 底辺が上 = 後手
	if m.centroid(top, bottom) < 0.5 {
		return CellPieceDown
	}
	return CellPieceUp
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
// 地色の占める割合が中央値を取れる程度には残る
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

// newCellMask は分類用のマスクを作る。判定できない小さすぎるマスでは nil
func newCellMask(cell image.Image, boardColor uint8) *cellMask {
	if cell == nil {
		return nil
	}
	gray := ConvertGray(cell)
	b := trimCellPadding(gray.Bounds())
	w, h := b.Dx(), b.Dy()
	if w < 8 || h < 8 {
		return nil
	}

	diff := float64(boardColor) * pieceDiffFrac
	if diff < pieceDiffMin {
		diff = pieceDiffMin
	}
	lo := float64(boardColor) - diff
	hi := float64(boardColor) + diff

	marked := make([]bool, w*h)
	colCount := make([]int, w)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := float64(gray.GrayAt(b.Min.X+x, b.Min.Y+y).Y)
			if v < lo || v > hi {
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

// centroid は top..bottom を 0.0〜1.0 に正規化した幅重心を返す
func (m *cellMask) centroid(top, bottom int) float64 {
	sum, weighted := 0.0, 0.0
	for y := top; y <= bottom; y++ {
		e := float64(m.extent[y])
		sum += e
		weighted += e * float64(y)
	}
	if sum == 0 {
		return 0.5
	}
	return (weighted/sum - float64(top)) / float64(bottom-top)
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
