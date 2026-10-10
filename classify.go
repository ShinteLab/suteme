package suteme

import (
	"image"
	"sort"
)

// CellCategory はマスの大分類。
//
// **「手前（上向き）＝先手」は規約であり、固定する。反転する設定は作らない。**
// 駒の向きで分かるのは「どちらの対局者の駒か」だけで、その人が先手か後手かは
// 盤の絵に写っていない（中継・紙面は「手前が先手」だが、ゲーム画面では
// 自分が後手番のとき手前が後手になる）。**画像認識では原理的に解けないので
// 解かない。** 呼び出し側が手前を先手として表現した SFEN を渡す責任を持つ。
// 詳細は AGENTS.md「先後の割り当ては『手前が先手』で固定（規約）」。
type CellCategory int

const (
	CellEmpty     CellCategory = 0 // 空
	CellPieceUp   CellCategory = 1 // 上向き。手前側の対局者の駒 → 先手として扱う
	CellPieceDown CellCategory = 2 // 下向き。奥側の対局者の駒 → 後手として扱う
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
	// 空マスは盤の地色そのものなので被覆率はほぼ 0 になる（実測 51 局面 2428 マスで
	// 中央値 0.007・9割が 0.029 以下）。駒との間は本来かなり開いている。
	//
	// **駒の被覆率は盤によって大きく変わるので、駒の側に寄せてはいけない。**
	// 駒の地に色が付いた盤（普通のゲーム画面・橙の中継）では駒の本体ごと
	// マスクに入って 0.4〜0.6 になるが、**駒の色が盤の地色とほぼ同じ木目の盤**では
	// 墨の字画と駒の輪郭しかマスクに残らず 0.13〜0.27 まで落ちる。
	// 0.15 にしていたときは後者の盤で駒 102 マス中 22 マスが空に化けていた。
	//
	//	しきい値  空→駒 / 駒→空（51局面 4131 マス）
	//	0.15         3 / 31
	//	0.12         3 /  5
	//	**0.10       4 /  5**
	//	0.08        14 /  5
	//	0.06        34 /  4
	//
	// 0.12 でも同等だが、木目の盤の駒の下限が 0.130 なので余裕が 1 割も無い。
	// 空マス側は 0.10 まで上げても 1 件しか増えないので、下限から離せる 0.10 を採る。
	emptyCoverMax = 0.10
	// emptyCoverGap / emptyCoverHigh は盤ごとに空/駒の境目を引き直すための帯。
	// 詳細は adaptiveEmptyCover。
	emptyCoverGap  = 0.10
	emptyCoverHigh = 0.30
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
	cat, _ := ClassifyCellDetail(cell, boardColor)
	return cat
}

// ClassifyCellDetail は ClassifyCellWith の判定に加えて、向き判定の
// **確信の度合い**（0.0〜1.0）を返す。空マスでは 0。
//
// 値は上下半分の幅の中央値の差を駒の最大幅で割ったもので、
// 「五角形がどれだけはっきり片側に広がっているか」を表す。
// 0 に近いほど上下が同じ幅＝向きが決まっていないということ。
//
// **向きを外したマスはこの値が小さいほうに偏る。** 実測（保存済み36局面 /
// 1202 駒マス）で、値が 0.08 未満の 85 マスに反転 47 件のうち 23 件が入る。
//
// **かつてはこの値で「回転照合に回すマス」を選んでいたが、いまは使っていない。**
// 向きは常に回転照合で決める（`classifyCellOrient`）。この確信度は
// 観測用として残してある。
func ClassifyCellDetail(cell image.Image, boardColor uint8) (CellCategory, float64) {
	return classifyCellDetail(cell, boardColor, emptyCoverMax)
}

// classifyCellDetail は空判定のしきい値を差し替えられる ClassifyCellDetail。
// 盤全体を扱えるなら adaptiveEmptyCover で求めた値を渡す。
func classifyCellDetail(cell image.Image, boardColor uint8, emptyMax float64) (CellCategory, float64) {
	cat, margin, _ := classifyCellCover(cell, boardColor, emptyMax)
	return cat, margin
}

// classifyCellCover は classifyCellDetail に加えて一次マスクの被覆率を返す。
// **マスクが作れなかったマスでは -1**（`coverUnmeasured`。マスのほぼ全体が
// 地色と違って全列がグリッド線として落ちた、など）。被覆率は空判定の材料そのもので、
// 分類器が空と言ったマスを推論器に回すかどうかと観測（`CellDebug.Cover`）に使う。
func classifyCellCover(cell image.Image, boardColor uint8, emptyMax float64) (CellCategory, float64, float64) {
	local := cellBoardColor(cell)
	m := newCellMaskSign(cell, local, boardColor, signBoth)
	cover := float64(coverUnmeasured)
	if m != nil {
		cover = m.cover
	}
	cat, margin := classifyCellMask(cell, boardColor, local, m, emptyMax)
	return cat, margin, cover
}

// coverUnmeasured は被覆率を測れなかったマスの印（`classifyCellCover`）。
const coverUnmeasured = -1

// classifyCellMask は一次マスク（signBoth）を作り終えたところから先。
// 盤ごとのしきい値を決めるために 81 マスの被覆率を先に取る呼び出し側
// （`ClassifyBoard` / `boardCells`）が、同じマスクを作り直さずに済むように
// 分けてある。
func classifyCellMask(cell image.Image, boardColor, local uint8, m *cellMask, emptyMax float64) (CellCategory, float64) {
	if m == nil || m.cover < emptyMax {
		return CellEmpty, 0
	}
	if m = pieceSideOf(cell, local, boardColor); m == nil {
		return CellEmpty, 0
	}
	top, bottom, ok := m.span()
	if !ok {
		return CellEmpty, 0
	}
	// 駒の五角形は尖り側が狭く底辺側が広いので、駒の外接範囲を上下に割ると
	// 底辺のある側の幅が広い。広いほうが下 = 底辺が下 = 先手
	half := (top + bottom) / 2
	lower, upper := m.medianExtent(half+1, bottom), m.medianExtent(top, half)
	cat := CellPieceDown
	if lower >= upper {
		cat = CellPieceUp
	}
	maxExtent := 0
	for y := top; y <= bottom; y++ {
		if m.extent[y] > maxExtent {
			maxExtent = m.extent[y]
		}
	}
	if maxExtent == 0 {
		return cat, 0
	}
	diff := lower - upper
	if diff < 0 {
		diff = -diff
	}
	return cat, float64(diff) / float64(maxExtent)
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
	cells, emptyMax := boardCells(img, br, bc)
	for _, w := range cells {
		result[w.row][w.col], _ = classifyCellMask(w.img, bc, w.local, w.mask, emptyMax)
	}
	return result
}

// cellWork は 1 マス分の「切り出した画像・局所地色・一次マスク」。
// 空判定の境目を決めるには全マスの被覆率が要るので、
// 先に 81 マスぶんを作ってから 2 周目で分類する。
type cellWork struct {
	row, col int
	img      image.Image
	local    uint8
	mask     *cellMask
}

// boardCells は盤の全マスを 1 回だけ切り出し、あわせて
// 空判定の境目（`BoardEmptyCover` と同じもの）を返す。
func boardCells(img image.Image, br *BoardRegion, boardColor uint8) ([]cellWork, float64) {
	if img == nil || br == nil {
		return nil, emptyCoverMax
	}
	works := make([]cellWork, 0, 81)
	cov := make([]float64, 0, 81)
	for r := 0; r < 9; r++ {
		for c := 0; c < 9; c++ {
			cell := br.ExtractCell(img, r, c)
			if cell == nil {
				continue
			}
			local := cellBoardColor(cell)
			m := newCellMaskSign(cell, local, boardColor, signBoth)
			works = append(works, cellWork{r, c, cell, local, m})
			if m != nil {
				cov = append(cov, m.cover)
			}
		}
	}
	return works, adaptiveEmptyCover(cov)
}

// BoardEmptyCover は盤ごとの「空マスと見なす被覆率の上限」を求める。
//
// **定数 `emptyCoverMax`(0.10) は空マスの側に近すぎる。** 空マスの被覆率は
// どの盤でもほぼ 0 だが、木目・グリッド線・盤に描かれた模様があると 0.10〜0.14 まで
// 上がる。実測（桜の絵柄の盤 3 局面）で、空マスの最大 0.126〜0.142 に対し
// 駒マスの最小は 0.404〜0.407 と**大きく開いている**のに、境目が 0.10 に
// 固定されているせいで空 15 マスが駒に化けていた。
//
// **かといって定数を上げてはいけない。** 駒の色が盤の地色とほぼ同じ木目の盤では
// 駒の被覆率が 0.13〜0.27 まで落ちる（`emptyCoverMax` の項）。0.15 にすると
// 保存済み 118 局面で駒→空が 8 → 71 件になる。
//
// そこで**その盤の 81 マスの被覆率を並べ、`emptyCoverMax` から
// `emptyCoverHigh`(0.30) までの帯の中で最も広い空隙**を探す。
// 空隙が `emptyCoverGap`(0.10) 以上あればその中点を境目にし、
// 無ければ `emptyCoverMax` のまま。空と駒が分かれている盤だけ境目が動く。
//
// 実測（保存済み 118 局面 / 9558 マス、手動座標・分類器のみ）:
//
//	                        全体     空→駒  駒→空  向き
//	現行                   97.11%     26     8    242
//	**適応しきい値**       97.30%    **6**   10   242
//	余白のクリップ         97.29%     16     10   233
//	**両方**             **97.39%**  **6**   10  **233**
//
// **上限 `emptyCoverHigh` を 0.35 に緩めてはいけない。** 木目の盤の駒が
// 境目の下に入り、駒→空が 10 → 114 件に崩れる。0.30 と 0.35 の間が崖。
func BoardEmptyCover(img image.Image, br *BoardRegion, boardColor uint8) float64 {
	_, emptyMax := boardCells(img, br, boardColor)
	return emptyMax
}

// adaptiveEmptyCover は被覆率の並びから空/駒の境目を返す（`BoardEmptyCover`）。
func adaptiveEmptyCover(cov []float64) float64 {
	v := make([]float64, 0, len(cov))
	for _, x := range cov {
		if x > emptyCoverMax {
			v = append(v, x)
		}
	}
	sort.Float64s(v)

	best, bestGap := emptyCoverMax, 0.0
	prev := emptyCoverMax
	for _, x := range v {
		hi := x
		if hi > emptyCoverHigh {
			hi = emptyCoverHigh
		}
		if g := hi - prev; g > bestGap {
			bestGap, best = g, (prev+hi)/2
		}
		if x >= emptyCoverHigh {
			break
		}
		prev = x
	}
	// 帯の中に値が 1 つも無ければ、帯全体がそのまま空隙になる
	if g := emptyCoverHigh - prev; g > bestGap {
		bestGap, best = g, (prev+emptyCoverHigh)/2
	}
	if bestGap < emptyCoverGap {
		return emptyCoverMax
	}
	return best
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
	extent []int    // 行ごとの駒の幅（除外した行は 0）
	edges  [][2]int // 行ごとの駒の左右端（除外した行は {-1,-1}）
	cover  float64  // 有効領域に対する被覆率
	origin image.Point
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
// （実測でも局所から決めるより良い。AGENTS.md の「駒分類」の節）。
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
	edges := make([][2]int, h)
	for i := range edges {
		edges[i] = [2]int{-1, -1}
	}
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
			edges[y] = [2]int{left, right}
		}
		count += n
		validH++
	}
	if validH < 8 {
		return nil
	}

	return &cellMask{
		extent: extent,
		edges:  edges,
		cover:  float64(count) / float64(validH*validW),
		origin: b.Min,
	}
}

// pieceBox は駒の外接矩形（cell の座標系）を返す。駒が見つからなければ false。
//
// **駒種の照合はこの矩形に切り揃えてから行う（`CellToInput`）。**
// マスをそのまま潰すと、盤ごとに違う「駒がマスのどこにどの大きさで描かれるか」が
// そのまま特徴量に乗ってしまい、未知の盤の駒が既知の同じ駒と一致しなくなる。
func pieceBox(cell image.Image) (image.Rectangle, bool) {
	if cell == nil {
		return image.Rectangle{}, false
	}
	local := cellBoardColor(cell)
	m := pieceSideOf(cell, local, local)
	if m == nil {
		return image.Rectangle{}, false
	}
	top, bottom, ok := m.span()
	if !ok {
		return image.Rectangle{}, false
	}
	left, right := -1, -1
	for y := top; y <= bottom; y++ {
		e := m.edges[y]
		if e[0] < 0 {
			continue
		}
		if left < 0 || e[0] < left {
			left = e[0]
		}
		if e[1] > right {
			right = e[1]
		}
	}
	if left < 0 || right <= left {
		return image.Rectangle{}, false
	}
	return image.Rect(
		m.origin.X+left, m.origin.Y+top,
		m.origin.X+right+1, m.origin.Y+bottom+1,
	), true
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

// medianExtent は lo..hi の行の幅の中央値を返す。
//
// 向きは上下半分の幅の**中央値**で比べる（`ClassifyCellDetail`）。平均や重心だと、
// グリッド線や隣のマスから入り込んだ駒の断片が 1〜2 行あるだけで結果が反転する。
// 中央値ならそうした外れ行を無視できる（実測で向き判定 87.3% → 93.5%）。
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
