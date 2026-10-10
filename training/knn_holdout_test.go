package training

import (
	"image"
	_ "image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ShinteLab/suteme"
)

// 保存済み局面（data/）を使った leave-one-out 評価。
//
// 1 局面を除いた残り全部で k-NN を構築し、除いた局面で認識精度を測る。
// 学習に使った局面でそのまま測ると当然一致するので、未知の局面に対する
// 実力を見るにはこの形にする必要がある。
//
// data/ は .gitignore 対象なので、無ければスキップする。
// 合否判定はせず数字を出すだけ: `go test -run TestKNNHoldout -v ./training`
//
// 主眼は「空を駒と誤る」件数。空を学習対象に含める前は分類器（被覆率）だけが
// 空を判定しており、学習で改善する余地がまったく無かった。

// chdirToData は data/history.json を持つディレクトリへ移動する。
// server.go の dataDir/historyFile が定数で相対パスのため。
func chdirToData(t *testing.T) bool {
	t.Helper()
	// テストの作業ディレクトリはパッケージのディレクトリ（training/）。
	// 通常は 1 つ上、worktree 内なら実体のリポジトリまで遡る
	for _, d := range []string{"..", "../../../.."} {
		if _, err := os.Stat(filepath.Join(d, historyFile)); err != nil {
			continue
		}
		wd, err := os.Getwd()
		if err != nil {
			return false
		}
		if err := os.Chdir(d); err != nil {
			return false
		}
		t.Cleanup(func() { os.Chdir(wd) })
		return true
	}
	return false
}

// wantGrid は正解 SFEN を「マスごとのラベル」に展開する。空マスは EmptyLabel。
// 実体は evaluate.go の boardGrid（評価タブと同じ展開を使う）
func wantGrid(s string) (*[9][9]string, error) { return boardGrid(s) }

// predictCell は RecognizeBoard と同じ判断でマスのラベルを返す
func predictCell(cell image.Image, bc uint8, p suteme.Predictor) string {
	return predictCellBO(-1, -1, cell, bc, nil, p)
}

// predictCellBO は盤ごとの向き判定（BoardOrient）を使う版。bo が nil なら定数版。
// row, col はマスの位置（`BoardOrient.ClassifyAt` が窓をずらして切り出し直すのに使う）
func predictCellBO(row, col int, cell image.Image, bc uint8, bo *suteme.BoardOrient, p suteme.Predictor) string {
	var cat suteme.CellCategory
	if bo != nil {
		cat, _ = bo.ClassifyAt(row, col, cell, bc, p)
	} else {
		cat, _ = suteme.ClassifyCellFor(cell, bc, p)
	}
	if cat == suteme.CellEmpty {
		return suteme.EmptyLabel
	}
	ncell := cell
	if cat == suteme.CellPieceDown {
		ncell = suteme.Rotate180(cell)
	}
	class, _ := p.Predict(ncell)
	if class == suteme.ClassEmpty {
		return suteme.EmptyLabel
	}
	label := suteme.ClassToBaseLabel(class)
	if cat == suteme.CellPieceDown {
		label = "-" + label
	}
	return label
}

func TestKNNHoldout(t *testing.T) {
	requireSlow(t)
	if !chdirToData(t) {
		t.Skip("data/history.json が無いのでスキップ")
	}
	h := loadHistory()
	if h == nil || len(h.Entries) < 2 {
		t.Skip("局面が足りないのでスキップ")
	}

	// 局面ごとにサンプルを作っておく
	byEntry := make(map[string][]suteme.TrainingSample, len(h.Entries))
	for _, e := range h.Entries {
		s, err := samplesFromHistory(e)
		if err != nil {
			t.Logf("%s: サンプル化に失敗 (%v)", e.ID, err)
			continue
		}
		byEntry[e.ID] = s
	}

	var totalCells, okCells int
	var emptyTotal, emptyAsPiece, emptyAsPieceByClassifier int
	var pieceTotal, pieceOK, pieceAsEmpty, pieceAsEmptyByClassifier int
	var orientFlips, orientFlipsByClassifier int

	for _, e := range h.Entries {
		if _, ok := byEntry[e.ID]; !ok {
			continue
		}
		// 自分以外の全局面で k-NN を作る
		var train []suteme.TrainingSample
		for id, s := range byEntry {
			if id != e.ID {
				train = append(train, s...)
			}
		}
		kn := suteme.NewKNN(MergeSamples(train))
		if kn == nil {
			continue
		}

		f, err := os.Open(filepath.Join(dataDir, e.ID+".png"))
		if err != nil {
			continue
		}
		img, _, err := image.Decode(f)
		f.Close()
		if err != nil {
			continue
		}
		if e.BoardBounds == nil {
			t.Logf("%s: 盤面座標が無いのでスキップ", e.ID)
			continue
		}
		b := e.BoardBounds
		br := suteme.BoardRegionFromRect(b.X1, b.Y1, b.X2, b.Y2)
		bc := suteme.BoardColor(img, br)
		bo := suteme.NewBoardOrient(img, br, bc)
		want, err := wantGrid(e.SFEN)
		if err != nil {
			t.Logf("%s: SFEN 解析に失敗 (%v)", e.ID, err)
			continue
		}

		var ok, missEmpty, missEmptyClassifier int
		for r := 0; r < 9; r++ {
			for c := 0; c < 9; c++ {
				cell := br.ExtractCell(img, r, c)
				if cell == nil {
					continue
				}
				got := predictCellBO(r, c, cell, bc, bo, kn)
				totalCells++
				if got == want[r][c] {
					ok++
					okCells++
				}
				if want[r][c] == suteme.EmptyLabel {
					emptyTotal++
					if got != suteme.EmptyLabel {
						missEmpty++
						emptyAsPiece++
					}
					// 推論器を使わず分類器（画像処理）だけで判定した場合。
					// 空判定の境目は盤ごとに決まる（BoardEmptyCover）ので、
					// マス単位の ClassifyCellWith ではなく bo に nil を渡す
					if cat, _ := bo.Classify(cell, bc, nil); cat != suteme.CellEmpty {
						missEmptyClassifier++
						emptyAsPieceByClassifier++
					}
					continue
				}
				pieceTotal++
				if got == want[r][c] {
					pieceOK++
				}
				if got == suteme.EmptyLabel {
					pieceAsEmpty++
				}
				if cat, _ := bo.Classify(cell, bc, nil); cat == suteme.CellEmpty {
					pieceAsEmptyByClassifier++
				}
				// 向きの反転。回転照合が効いているかを見る（分類器のみとの差）
				wantDown := strings.HasPrefix(want[r][c], "-")
				if got != suteme.EmptyLabel && strings.HasPrefix(got, "-") != wantDown {
					orientFlips++
				}
				if cat, _ := bo.Classify(cell, bc, nil); cat != suteme.CellEmpty &&
					(cat == suteme.CellPieceDown) != wantDown {
					orientFlipsByClassifier++
				}
			}
		}
		t.Logf("%s: %d/81  空→駒 %d件（分類器のみなら %d件）",
			e.ID, ok, missEmpty, missEmptyClassifier)
	}

	if totalCells == 0 {
		t.Skip("評価できる局面が無いのでスキップ")
	}
	t.Logf("全体（駒種まで一致）: %d/%d = %.1f%%",
		okCells, totalCells, 100*float64(okCells)/float64(totalCells))
	t.Logf("空 %d マス中 駒と誤り: %d件（分類器のみなら %d件）",
		emptyTotal, emptyAsPiece, emptyAsPieceByClassifier)
	t.Logf("駒 %d マス中 空と誤り: %d件（分類器のみなら %d件）",
		pieceTotal, pieceAsEmpty, pieceAsEmptyByClassifier)
	t.Logf("駒 %d マス中 向きの反転: %d件（分類器のみなら %d件）",
		pieceTotal, orientFlips, orientFlipsByClassifier)
	t.Logf("駒 %d マス中 駒種まで正解: %d件 = %.1f%%",
		pieceTotal, pieceOK, 100*float64(pieceOK)/float64(pieceTotal))
}

// TestKNNCellHoldout は「同じ見た目の盤を既にラベル付けしてある」状況を測る。
//
// TestKNNHoldout は局面ごと学習から外すため、その盤のグリッド線や木目を持つ
// 空マスが訓練データに 1 つも無い状態を測ることになる。実運用では同じ中継・
// 同じゲーム画面を繰り返し読むので、そちらの条件も見ておく必要がある。
//
// 評価するマス自身のサンプルだけを除外し（完全一致の照合になるのを防ぐ）、
// 同じ画像の他のマスは訓練データに残す。
func TestKNNCellHoldout(t *testing.T) {
	requireSlow(t)
	if !chdirToData(t) {
		t.Skip("data/history.json が無いのでスキップ")
	}
	h := loadHistory()
	if h == nil || len(h.Entries) == 0 {
		t.Skip("局面が無いのでスキップ")
	}

	var all []suteme.TrainingSample
	for _, e := range h.Entries {
		s, err := samplesFromHistory(e)
		if err != nil {
			continue
		}
		all = append(all, s...)
	}
	all = MergeSamples(all)
	if len(all) == 0 {
		t.Skip("サンプルが無いのでスキップ")
	}

	var emptyTotal, emptyAsPiece, emptyAsPieceByClassifier int
	var pieceTotal, pieceOK int

	for _, e := range h.Entries {
		if e.BoardBounds == nil {
			continue
		}
		f, err := os.Open(filepath.Join(dataDir, e.ID+".png"))
		if err != nil {
			continue
		}
		img, _, err := image.Decode(f)
		f.Close()
		if err != nil {
			continue
		}
		b := e.BoardBounds
		br := suteme.BoardRegionFromRect(b.X1, b.Y1, b.X2, b.Y2)
		bc := suteme.BoardColor(img, br)
		bo := suteme.NewBoardOrient(img, br, bc)
		want, err := wantGrid(e.SFEN)
		if err != nil {
			continue
		}

		var missEmpty, missClassifier int
		for r := 0; r < 9; r++ {
			for c := 0; c < 9; c++ {
				cell := br.ExtractCell(img, r, c)
				if cell == nil {
					continue
				}
				// このマス自身のサンプルを訓練データから除く
				self := suteme.CellToInput(cell)
				selfRot := suteme.CellToInput(suteme.Rotate180(cell))
				// 空サンプルはマス全体で作ってあるので、そちらも除く
				selfFull := suteme.CellToInputFull(cell)
				train := make([]suteme.TrainingSample, 0, len(all))
				for _, s := range all {
					if equalInput(s.Input, self) || equalInput(s.Input, selfRot) ||
						equalInput(s.Input, selfFull) {
						continue
					}
					train = append(train, s)
				}
				kn := suteme.NewKNN(train)
				if kn == nil {
					continue
				}
				got := predictCellBO(r, c, cell, bc, bo, kn)

				if want[r][c] == suteme.EmptyLabel {
					emptyTotal++
					if got != suteme.EmptyLabel {
						missEmpty++
						emptyAsPiece++
					}
					if cat, _ := bo.Classify(cell, bc, nil); cat != suteme.CellEmpty {
						missClassifier++
						emptyAsPieceByClassifier++
					}
					continue
				}
				pieceTotal++
				if got == want[r][c] {
					pieceOK++
				}
			}
		}
		t.Logf("%s: 空→駒 %d件（分類器のみなら %d件）", e.ID, missEmpty, missClassifier)
	}

	if emptyTotal == 0 {
		t.Skip("評価できるマスが無いのでスキップ")
	}
	t.Logf("空 %d マス中 駒と誤り: %d件（分類器のみなら %d件）",
		emptyTotal, emptyAsPiece, emptyAsPieceByClassifier)
	t.Logf("駒 %d マス中 駒種まで正解: %d件 = %.1f%%",
		pieceTotal, pieceOK, 100*float64(pieceOK)/float64(pieceTotal))
}
