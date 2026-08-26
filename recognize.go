package suteme

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/ShinteLab/core/sfen"
	"github.com/goml/gobrain"
)

const (
	cellSize  = 24
	inputSize = cellSize * cellSize
	// ClassEmpty は空マスのクラスID。駒種 14 クラスの次に置く。
	//
	// 空判定は ClassifyCellWith（地色との被覆率）が一次判定を担うが、
	// グリッド線・木目・隣のマスの駒の写り込みは「地色と異なる画素」を
	// 増やすため、被覆率が駒と重なって原理的に分離できないマスが出る
	// （実測で空マスの被覆率 0.38 に対し同一画像の駒の最小が 0.36）。
	// 空も学習対象にすることで、画素の並びで照合できるようにする。
	ClassEmpty = 14
	numClasses = ClassEmpty + 1 // 駒種14（向きなし）+ 空
)

// InputSize は外部パッケージから参照できる入力サイズ
const InputSize = inputSize

// NumClasses は推論器の出力クラス数（駒種 + 空）
const NumClasses = numClasses

// EmptyLabel は空マスを表すラベル文字列
const EmptyLabel = "none"

// BaseLabels はクラスIDと駒種（向きなし大文字）の対応
var BaseLabels = []string{
	"P",  // 0: 歩
	"L",  // 1: 香
	"N",  // 2: 桂
	"S",  // 3: 銀
	"G",  // 4: 金
	"B",  // 5: 角
	"R",  // 6: 飛
	"K",  // 7: 玉
	"+P", // 8: と
	"+L", // 9: 杏
	"+N", // 10: 圭
	"+S", // 11: 全
	"+B", // 12: 馬
	"+R", // 13: 龍
}

// LabelToClass はSFENラベルをクラスIDに変換
// 空マス（"none"）は ClassEmpty、それ以外の未知ラベルは -1 を返す
func LabelToClass(label string) int {
	upper := strings.ToUpper(label)
	if upper == strings.ToUpper(EmptyLabel) {
		return ClassEmpty
	}
	for i, b := range BaseLabels {
		if strings.ToUpper(b) == upper {
			return i
		}
	}
	return -1
}

// ClassToLabel はクラスIDをSFENラベルに変換（向きなし大文字）
// 空マスは EmptyLabel を返す
func ClassToLabel(class int) string {
	if class == ClassEmpty {
		return EmptyLabel
	}
	if class < 0 || class >= len(BaseLabels) {
		return "P"
	}
	return BaseLabels[class]
}

// ClassToBaseLabel は ClassToLabel の別名
func ClassToBaseLabel(class int) string {
	return ClassToLabel(class)
}

// TrainingData は学習データ
type TrainingData struct {
	Samples []TrainingSample `json:"samples"`
}

type TrainingSample struct {
	Input []float64 `json:"input"`
	Label int       `json:"label"`
}

// CellToInput はマス画像を入力ベクトルに変換 (24x24 グレースケール → 576 float64)
// 画像ごとの明るさ・コントラスト差を吸収するため、平均0・分散1に標準化する。
//
// **まず駒の外接矩形に切り揃えてから 24x24 に落とす（v5）。**
// マスをそのまま潰していたときは、盤ごとに違う「駒がマスのどこに・どの大きさで
// 描かれるか」がそのまま 576 次元に乗っていた。同じ駒でも盤が変われば別のベクトルに
// なるので、k-NN は**その盤の駒を一度も見ていないと当たらない**。
// 外接矩形に揃えると駒の位置と大きさが消え、字の形だけが残る。
//
// 出所ごと丸ごと学習から外した実測（駒種のみ・向きは正解を使用）:
//
//	                      木目     橙    その他   全体
//	マスそのまま (v4)      41.2%  48.2%  76.9%  72.9%
//	**外接矩形 (v5)**     45.1%  65.5%  83.2%  79.8%
//	外接矩形 + 墨2値       36.3%  61.8%  83.1%  78.9%
//	外接矩形 + 1割の余白    42.2%  64.6%  79.9%  76.6%
//
// 余白を残すと切り揃えた意味が薄れ、墨だけの 2 値にすると駒の地の濃淡という
// 手掛かりまで捨ててしまう。**外接矩形ちょうどで切るのが一番良い。**
// 空マスなど駒が見つからないマスは、従来どおりマス全体を使う。
//
// **最後に float32 の精度へ丸める（v4）。学習データファイルが float32 で
// 持つので、そこに合わせないと `MergeSamples` の重複除去が効かなくなる。**
// MergeSamples は入力ベクトルの内容が一致するかで重複を判定するが、
// ファイルから読んだサンプル（float32 で丸め済み）と新しく作ったサンプル
// （素の float64）は同じマスから作っても値がわずかに食い違うため、
// 丸めないと**学習を押すたびに同じサンプルが積み上がる**
// （実装当初にこれで 1173 件中 838 件＝71% が重複していた）。
//
// 元の値は 0〜255 の輝度から作るので、float32 の仮数 24bit で
// 表現できる範囲を超える情報は元から無い（相対誤差 6e-8。
// k-NN の距離の順位には影響しない）。
func CellToInput(cell image.Image) []float64 {
	if box, ok := pieceBox(cell); ok {
		cell = cropImage(cell, box)
	}
	resized := resizeGray(cell, cellSize, cellSize)
	input := make([]float64, inputSize)
	bounds := resized.Bounds()
	var sum, sumSq float64
	for y := 0; y < cellSize; y++ {
		for x := 0; x < cellSize; x++ {
			v := float64(resized.GrayAt(x+bounds.Min.X, y+bounds.Min.Y).Y)
			input[y*cellSize+x] = v
			sum += v
			sumSq += v * v
		}
	}
	n := float64(inputSize)
	mean := sum / n
	std := math.Sqrt(sumSq/n - mean*mean)
	if std < 1 {
		std = 1
	}
	for i := range input {
		input[i] = quantize((input[i] - mean) / std)
	}
	return input
}

// quantize は float32 の精度へ丸める（ファイル表現と揃えるため。CellToInput 参照）
func quantize(v float64) float64 { return float64(float32(v)) }

// cropImage は画像の一部を切り出す。SubImage を持たない実装のために
// グレースケールへ落としてから切る道も用意しておく。
func cropImage(src image.Image, r image.Rectangle) image.Image {
	type subImager interface {
		SubImage(image.Rectangle) image.Image
	}
	r = r.Intersect(src.Bounds())
	if r.Empty() {
		return src
	}
	if s, ok := src.(subImager); ok {
		return s.SubImage(r)
	}
	return ConvertGray(src).SubImage(r)
}

// 学習データファイル（v4）のバイナリ表現。
//
// **v3 まではこれを JSON で持っていたが、float64 1 個が 20 文字前後の
// テキストになるので実測 16605 サンプルで 186MB あった**（576 × 16605 ≒
// 956 万個の数値）。読むたびに全部を strconv で数値に戻すため、
// 読み込み 1.8s・保存 0.8s かかっていた。同じ内容を float32 で並べれば
// 38MB・0.1s 程度で済む。
//
//	ヘッダ 16 バイト
//	  magic     [8]byte  "SUTEMETD"
//	  version   uint16   = 4
//	  inputSize uint16   = 576
//	  count     uint32   サンプル数
//	以降 count 件
//	  label     int32
//	  input     [inputSize]float32
//
// すべてリトルエンディアン。
const (
	trainingDataVersion = 4
	trainingDataHeader  = 16
)

var trainingDataMagic = [8]byte{'S', 'U', 'T', 'E', 'M', 'E', 'T', 'D'}

// SaveTrainingData は学習データをバイナリファイルに保存する。
//
// **一時ファイルに書いてから rename する。** 手でラベル付けした正解から
// 作られる壊れやすい資産で、しかも書き込みに数百 ms かかるので、
// 途中で落ちたときに全損させない（history.json と同じ扱い）。
func SaveTrainingData(path string, data *TrainingData) error {
	tmp, err := os.CreateTemp(filepath.Dir(path), filepath.Base(path)+".tmp*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName) // rename に成功していれば消えるものは無い

	w := bufio.NewWriterSize(tmp, 1<<20)
	if err := writeTrainingData(w, data); err != nil {
		tmp.Close()
		return err
	}
	if err := w.Flush(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, path)
}

func writeTrainingData(w io.Writer, data *TrainingData) error {
	var head [trainingDataHeader]byte
	copy(head[:8], trainingDataMagic[:])
	binary.LittleEndian.PutUint16(head[8:], trainingDataVersion)
	binary.LittleEndian.PutUint16(head[10:], uint16(inputSize))
	binary.LittleEndian.PutUint32(head[12:], uint32(len(data.Samples)))
	if _, err := w.Write(head[:]); err != nil {
		return err
	}

	buf := make([]byte, 4+inputSize*4)
	for _, s := range data.Samples {
		if len(s.Input) != inputSize {
			return fmt.Errorf("入力長が %d ではありません: %d", inputSize, len(s.Input))
		}
		binary.LittleEndian.PutUint32(buf, uint32(int32(s.Label)))
		for i, v := range s.Input {
			binary.LittleEndian.PutUint32(buf[4+i*4:], math.Float32bits(float32(v)))
		}
		if _, err := w.Write(buf); err != nil {
			return err
		}
	}
	return nil
}

// LoadTrainingData は学習データをファイルから読み込む。
// v4 のバイナリと、v3 までの JSON の**どちらも読める**
// （186MB の既存ファイルを捨てずに移行するため。書き出しは常に v4）。
func LoadTrainingData(path string) (*TrainingData, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := bufio.NewReaderSize(f, 1<<20)
	head, err := r.Peek(1)
	if err != nil {
		return nil, err
	}
	if head[0] == '{' {
		return readTrainingDataJSON(r)
	}
	return readTrainingData(r)
}

func readTrainingData(r io.Reader) (*TrainingData, error) {
	var head [trainingDataHeader]byte
	if _, err := io.ReadFull(r, head[:]); err != nil {
		return nil, fmt.Errorf("学習データのヘッダが読めません: %w", err)
	}
	if !bytes.Equal(head[:8], trainingDataMagic[:]) {
		return nil, fmt.Errorf("学習データファイルではありません")
	}
	if v := binary.LittleEndian.Uint16(head[8:]); v != trainingDataVersion {
		return nil, fmt.Errorf("学習データの版が違います: %d (対応 %d)", v, trainingDataVersion)
	}
	n := int(binary.LittleEndian.Uint16(head[10:]))
	if n != inputSize {
		return nil, fmt.Errorf("入力長が違います: %d (対応 %d)", n, inputSize)
	}
	count := int(binary.LittleEndian.Uint32(head[12:]))

	data := &TrainingData{Samples: make([]TrainingSample, 0, count)}
	buf := make([]byte, 4+n*4)
	for i := 0; i < count; i++ {
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, fmt.Errorf("%d 件目が読めません: %w", i, err)
		}
		input := make([]float64, n)
		for j := 0; j < n; j++ {
			input[j] = float64(math.Float32frombits(binary.LittleEndian.Uint32(buf[4+j*4:])))
		}
		data.Samples = append(data.Samples, TrainingSample{
			Input: input,
			Label: int(int32(binary.LittleEndian.Uint32(buf))),
		})
	}
	return data, nil
}

// readTrainingDataJSON は v3 までの JSON 形式を読む。
// **読んだ値は float32 に丸める。** v4 のファイルから読んだ場合と同じ値にしないと、
// MergeSamples の重複判定が形式をまたいだときだけ効かなくなる（CellToInput 参照）。
func readTrainingDataJSON(r io.Reader) (*TrainingData, error) {
	var data TrainingData
	if err := json.NewDecoder(r).Decode(&data); err != nil {
		return nil, err
	}
	for _, s := range data.Samples {
		for i := range s.Input {
			s.Input[i] = quantize(s.Input[i])
		}
	}
	return &data, nil
}

// Model は駒認識モデル
type Model struct {
	NN *gobrain.FeedForward

	source string // 読み込み元のファイルパス（LoadPredictor が設定。デバッグ表示用）
}

// Debug は認識結果に載せる推論器の素性を返す（DebugPredictor）。
func (m *Model) Debug() PredictorDebug {
	return PredictorDebug{Kind: "nn", Source: m.source}
}

// LoadModel はモデルをJSONファイルから読み込み
func LoadModel(path string) (*Model, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var nn gobrain.FeedForward
	if err := json.NewDecoder(f).Decode(&nn); err != nil {
		return nil, err
	}
	return &Model{NN: &nn}, nil
}

// Predict はマス画像から駒を推論する
func (m *Model) Predict(cell image.Image) (int, float64) {
	input := CellToInput(cell)
	output := m.NN.Update(input)

	bestClass := 0
	bestConf := 0.0
	for i, v := range output {
		if v > bestConf {
			bestConf = v
			bestClass = i
		}
	}
	return bestClass, bestConf
}

// BoardOrient は盤ごとの空判定の境目（`BoardEmptyCover`）を持つ。
//
// **向きは回転照合（`OrientationMatcher`）で決める。分類器の向き判定は
// 照合できないときの控えでしかない。** 以前はここに「確信度が盤の下位
// 25% のマスだけ回転照合に回す」という境目（`OrientMarginQuantile`）を
// 持っていたが、局面が増えて回転照合が単独で分類器を大きく上回ったので
// 不要になった（`classifyCellOrient` の実測表）。
type BoardOrient struct {
	emptyMax float64
}

// NewBoardOrient は盤ごとの空判定の境目を求める。
func NewBoardOrient(img image.Image, br *BoardRegion, boardColor uint8) *BoardOrient {
	if img == nil || br == nil {
		return &BoardOrient{emptyMax: emptyCoverMax}
	}
	return &BoardOrient{emptyMax: BoardEmptyCover(img, br, boardColor)}
}

// Classify はマスを 空/先手/後手 に分類する。
// byMatch は向きを回転照合で決めたかどうか（観測用。`CellDebug.OrientBy`）。
func (bo *BoardOrient) Classify(cell image.Image, boardColor uint8, m Predictor) (cat CellCategory, byMatch bool) {
	return classifyCellOrient(cell, boardColor, bo.emptyMax, m)
}

// ClassifyCellFor は推論器も使ってマスを 空/先手/後手 に分類する。
//
// **空/先手/後手 を出すところは必ずこれか `BoardOrient.Classify` を通すこと。**
// `ClassifyCellWith` は画像処理だけの一次判定で、向きは推論器との回転照合で
// 決め直される。素の `ClassifyCellWith` を別途呼ぶと `RecognizeBoard` が返す
// SFEN と食い違う。
//
// 盤全体を扱えるなら `NewBoardOrient` のほうが精度が高い
// （空判定の境目を盤ごとに引き直せる）。
func ClassifyCellFor(cell image.Image, boardColor uint8, m Predictor) (cat CellCategory, byMatch bool) {
	return classifyCellOrient(cell, boardColor, emptyCoverMax, m)
}

// classifyCellOrient は空/駒を分類器で決め、駒の向きは回転照合で決める。
//
// **向きは回転照合が主で、分類器（上下半分の幅の差）は照合できないときの控え。**
// 学習データは後手の駒を `Rotate180` して先手向きに揃えてあるので、
// そのままのほうが近ければ先手、180度回した版のほうが近ければ後手。
//
// **以前は逆で、分類器を主にして「確信度が低いマスだけ」照合に回していた**
// （`OrientMarginQuantile` = 盤ごとの下位 25%）。**局面が増えたので測り直したら
// 逆転していた。** 回転照合は同じ駒の絵を一度見ていれば当たるので、
// 出所ごとの枚数が増えるほど強くなる。一方で分類器は画像処理なので
// データが増えても変わらない。実測（141局面 / 4761 駒マス、向きの誤り件数。
// `TestOrientDump` が書き出した材料で数え直したもの）:
//
//	                        局面ホールドアウト   出所ホールドアウト
//	分類器のみ                    275                275
//	**回転照合のみ（現在）**       41                 60
//	分位点 0.25（旧）             147                156
//	どちらも外す（oracle）         22                 28
//
// 51局面の頃は「分類器のみ 92 / 回転照合のみ 74 / 分位点 65」で、
// 混ぜたほうが良かった。**同じ測り方でも局面数が違えば結論が変わる**ので、
// この表は局面数とセットで読むこと。
//
// **確信度を足し合わせる形（`mscore + w * 分類器の確信度`）は測って駄目だった。**
// 分類器の確信度を盤ごとの分位点に直して足しても、w を上げるほど単調に悪化する
// （局面ホールドアウトで w=0.05 → 41、0.1 → 44、0.2 → 60、0.5 → 89、1.0 → 118）。
// 分類器の誤りは回転照合の誤りをほぼ包含していて、足すぶんだけ濁る。
//
// **「照合が拮抗しているマスだけ分類器に返す」も効きが薄い。** 距離の差の比
// |(up-down)/(up+down)| が 0.02 以下のとき分類器を採ると 41 → 38 / 60 → 56 に
// なるが、4761 マス中 3〜4 件のために定数がもう 1 つ増えるので採らない
// （0.05 まで緩めると 43 / 58 で逆に悪くなる。最適が狭い）。
//
// **盤ごとに「その盤を見たことがあるか」で切り替えるのも駄目だった。**
// 最近傍距離の中央値が大きい盤＝未知の出所とみなして分類器に任せる形にしても、
// 回転照合が良い盤（0.42〜0.44）のほうが悪い盤（0.32）より距離が遠く、
// しきい値を置ける並びになっていない。
//
// **代償は推論時間。** 全駒マスで最近傍探索を 2 回（そのまま / 180度）走らせるので、
// 分位点で 25% だけ回していた頃より増える。実測は `TestKNNHoldout` で
// 219s → 243s（+11%。141局面 × 81マス、駒種の推論も含めた全体）。
func classifyCellOrient(cell image.Image, boardColor uint8, emptyMax float64, m Predictor) (CellCategory, bool) {
	cat, _ := classifyCellDetail(cell, boardColor, emptyMax)
	if cat == CellEmpty {
		return cat, false
	}
	om, ok := m.(OrientationMatcher)
	if !ok {
		// **推論器が照合できないなら、向き専用の照合データを探す**
		// （`orient_data_v1.bin`。NN を配る構成のための穴埋め。orient.go）。
		// それも無ければ分類器の判定がそのまま残る
		if om = defaultOrientMatcher(); om == nil {
			return cat, false
		}
	}
	up := om.PieceDistance(cell)
	down := om.PieceDistance(Rotate180(cell))
	if math.IsInf(up, 1) && math.IsInf(down, 1) {
		return cat, false
	}
	if up <= down {
		return CellPieceUp, true
	}
	return CellPieceDown, true
}

// RecognizeBoard は盤面全体を認識してSFENを返す
//
// 空/駒は ClassifyCellWith が判定し、向きは Predictor との回転照合
// （`classifyCellOrient`）で決める。駒種は Predictor（NN または k-NN）で
// 推論する。Predictor が空クラス（ClassEmpty）を返した場合は分類より優先して
// 空とする。グリッド線や木目で被覆率が上がったマスは、被覆率では駒と分離できないが
// 画素の並びは既知の空マスと一致するため、学習済みの空パターンのほうが確かなため。
// 逆方向（分類が空・推論が駒）は向きが決まらないので覆さない。
func RecognizeBoard(img image.Image, br *BoardRegion, m Predictor) string {
	board, _, _ := recognizeBoardDetail(img, br, m)
	return board
}

// recognizeBoardDetail は RecognizeBoard の本体で、盤面文字列に加えて
// マスごとの認識過程（Result.Debug 用）と盤の地色を返す。
func recognizeBoardDetail(img image.Image, br *BoardRegion, m Predictor) (string, []CellDebug, uint8) {
	// 各マスの SFEN 表記("" は空マス)を組み立て、盤面文字列化(空マスの
	// ランレングス圧縮・段区切り)は core/sfen に委譲する。
	var grid [9][9]string
	bc := BoardColor(img, br)
	bo := NewBoardOrient(img, br, bc)
	cells := make([]CellDebug, 0, 81)
	for r := 0; r < 9; r++ {
		for c := 0; c < 9; c++ {
			d := CellDebug{Row: r, Col: c, Rect: br.Cells[r][c], Class: -1}
			// 先に登録して、以降は d を書き換えつつ continue できるようにする
			// （マスごとの記録を取りこぼさないため）。
			cells = append(cells, d)
			cur := &cells[len(cells)-1]

			cell := br.ExtractCell(img, r, c)
			if cell == nil {
				continue
			}

			cat, byMatch := bo.Classify(cell, bc, m)
			cur.Category = cat
			if byMatch {
				cur.OrientBy = "match"
			}
			if cat == CellEmpty {
				continue
			}

			if m != nil {
				// gobrain で駒種を推論（後手は180度回転して先手向きに正規化）
				ncell := cell
				if cat == CellPieceDown {
					ncell = Rotate180(cell)
				}
				class, conf := m.Predict(ncell)
				cur.Class, cur.Confidence = class, conf
				if class == ClassEmpty {
					continue
				}
				base := ClassToBaseLabel(class)
				if cat == CellPieceDown {
					if len(base) > 0 {
						if base[0] == '+' {
							grid[r][c] = "+" + strings.ToLower(base[1:])
						} else {
							grid[r][c] = strings.ToLower(base)
						}
					}
				} else {
					grid[r][c] = base
				}
			} else {
				// モデルなし: 向き不問で駒があることだけを示す
				grid[r][c] = "?"
			}
			cur.Piece = grid[r][c]
		}
	}
	board := sfen.FormatBoard(func(rank, file int) string { return grid[rank][file] })
	return board, cells, bc
}

// resizeGray は画像を指定サイズのグレースケールにリサイズする（面積平均）。
//
// **最近傍法にしてはいけない。** 1マスは 40〜115px あり、これを 24x24 に
// 落とすので最近傍だと 3〜5px おきの点をつまむ形になる。盤面座標が 1px ずれると
// つまむ点が総入れ替えになり、576次元ベクトルが別物になってしまう。
// 実測で、盤面座標を 1px ずらすだけで誤認識マスが 22 → 112（全891マス）に
// 増えていた。面積平均なら 1px のずれは各画素の重みがわずかに動くだけになる。
func resizeGray(src image.Image, w, h int) *image.Gray {
	srcBounds := src.Bounds()
	srcW := srcBounds.Dx()
	srcH := srcBounds.Dy()

	dst := image.NewGray(image.Rect(0, 0, w, h))
	if srcW <= 0 || srcH <= 0 {
		return dst
	}
	for y := 0; y < h; y++ {
		y0 := srcBounds.Min.Y + y*srcH/h
		y1 := srcBounds.Min.Y + (y+1)*srcH/h
		if y1 <= y0 {
			y1 = y0 + 1
		}
		for x := 0; x < w; x++ {
			x0 := srcBounds.Min.X + x*srcW/w
			x1 := srcBounds.Min.X + (x+1)*srcW/w
			if x1 <= x0 {
				x1 = x0 + 1
			}
			sum, n := 0, 0
			for sy := y0; sy < y1; sy++ {
				for sx := x0; sx < x1; sx++ {
					r, g, b, _ := src.At(sx, sy).RGBA()
					sum += int((19595*r + 38470*g + 7471*b + 1<<15) >> 24)
					n++
				}
			}
			if n == 0 {
				continue
			}
			dst.SetGray(x, y, color.Gray{Y: uint8(sum / n)})
		}
	}
	return dst
}

// Detail は盤ごとの空判定の境目を使った `ClassifyCellDetail`。
// 分類器だけの判定（空/先手/後手 と向きの確信度）が要る呼び出し側のためのもので、
// 認識そのものは `Classify` を通すこと。
func (bo *BoardOrient) Detail(cell image.Image, boardColor uint8) (CellCategory, float64) {
	return classifyCellDetail(cell, boardColor, bo.emptyMax)
}
