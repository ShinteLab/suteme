package training

// 認識率の記録（評価タブ）。
//
// **モデルを更新したときに新旧を比べたい**が、その場の数字だけを見ても
// 良くなったのか分からない。実行 1 回を 1 レコードとして
// data/accuracy.json に積み、あとから並べて見られるようにする。
//
// 測るのは 3 つ。
//
//   - **盤面検出**: 自動検出した外枠と、履歴に保存してある手動指定座標との差
//     （マス単位）。**最終目標は手入力の座標なしで認識が動くこと**なので、
//     外した局面もずれ幅つきで残す。「あの中継画像の盤が拾えるようになったか」を
//     後から確かめられるようにするため、成功件数だけに畳まない
//   - **手動座標での認識率**: 座標を与えたときの成績＝認識器そのものの実力
//   - **自動座標での認識率**: 検出から通した端から端まで。**盤面を検出できなかった
//     局面は 0/81 として数える**（それが手入力なしで動かしたときの実力）
//   - **負例（盤面が写っていない画像）に対する誤検出**: data/negative/ に置いた
//     画像を盤として採用してしまわないか。**これだけは正解データからは測れない**
//     （上の 3 つはどれも「盤がある画像」が前提）。詳細は negative.go
//
// それぞれを 2 通りの推論器で測る。
//
//   - **full**: 起動中の推論器そのまま。**評価する局面が学習済みなら高く出る**ので
//     絶対値は信用できない（丸暗記）。版どうしの相対比較に使う
//   - **holdout**: その局面を学習から外して作り直した k-NN（leave-one-out）。
//     未知の局面に対する実力。局面ごとに k-NN を組み直すので時間がかかる
//
// 数字の意味は TestKNNHoldout と揃えてある（同じ predictCellBO 相当の判断を
// suteme.Recognize 経由で通す）。違いは、こちらが盤面検出も測ることと、
// 結果をファイルに残すこと。

import (
	"encoding/json"
	"fmt"
	"image"
	_ "image/jpeg"
	"image/png"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/ShinteLab/core/sfen"
	"github.com/ShinteLab/suteme"
)

const (
	evalFile = "data/accuracy.json"
	// maxEvalRuns は残す実行の件数。版の比較が目的なので、
	// 履歴（正解データ）と違って古いものは押し出してよい
	maxEvalRuns = 100
	// alignedMaxShift は盤面検出が「合っている」とみなす手動座標との差（マス）。
	// detect_data_test.go の判定と揃えてある
	alignedMaxShift = 0.5
)

var evalMu sync.Mutex

// EvalMetrics は認識の成績。実行全体の集計にも、局面 1 件の内訳にも使う。
type EvalMetrics struct {
	Boards        int `json:"boards"`         // 評価した局面数
	BoardsPerfect int `json:"boards_perfect"` // 81マスすべて一致した局面数
	NoBoard       int `json:"no_board"`       // 盤面を検出できず 0/81 とした局面数
	Cells         int `json:"cells"`          // 比較したマス数
	OK            int `json:"ok"`             // 空/向き/駒種まで一致
	PieceCells    int `json:"piece_cells"`    // 正解が駒のマス
	PieceOK       int `json:"piece_ok"`       // うち向き・駒種まで一致
	EmptyAsPiece  int `json:"empty_as_piece"` // 空を駒と誤った
	PieceAsEmpty  int `json:"piece_as_empty"` // 駒を空と誤った
	OrientFlip    int `json:"orient_flip"`    // 駒だが向きが逆
	TypeMiss      int `json:"type_miss"`      // 向きは合っているが駒種が違う
}

func (m *EvalMetrics) add(o EvalMetrics) {
	m.Boards += o.Boards
	m.BoardsPerfect += o.BoardsPerfect
	m.NoBoard += o.NoBoard
	m.Cells += o.Cells
	m.OK += o.OK
	m.PieceCells += o.PieceCells
	m.PieceOK += o.PieceOK
	m.EmptyAsPiece += o.EmptyAsPiece
	m.PieceAsEmpty += o.PieceAsEmpty
	m.OrientFlip += o.OrientFlip
	m.TypeMiss += o.TypeMiss
}

// EvalDetect は盤面検出の結果を手動指定座標と突き合わせたもの。
// ずれはマス単位（画像の解像度に依らず比較できるようにするため）。
//
// **棄却された候補もずれと信頼度を残す。** 「盤を見つけられなかった」だけだと
// 惜しかったのか全然違う場所を見ていたのかが分からず、次の版で直ったかを
// 追えない（CLAUDE.md の「dx=+3.57 を conf 0.00 で棄却」のような記録が要る）。
type EvalDetect struct {
	// Found は信頼度を満たして採用されたか。false でも Candidate が真なら
	// 棄却された候補の位置・信頼度が入っている
	Found bool `json:"found"`
	// Candidate は DetectBoard が領域を返したか（採用されたかは別）
	Candidate bool `json:"candidate"`
	// Source は領域の決め方（"detect" / "whole"）
	Source string `json:"source"`
	// Confidence は ValidateBoard の値
	Confidence float64 `json:"confidence"`
	// DX/DY は左上の、DW/DH は幅・高さの手動座標との差（マス単位）
	DX float64 `json:"dx"`
	DY float64 `json:"dy"`
	DW float64 `json:"dw"`
	DH float64 `json:"dh"`
	// MaxShift は上の 4 つの絶対値の最大
	MaxShift float64 `json:"max_shift"`
	// Aligned は MaxShift が alignedMaxShift 以内か
	Aligned bool `json:"aligned"`
	// Exact は保存座標が検出結果と px 単位で一致したか。
	//
	// **一致した局面は「検出が当たった」証拠にならない。** 座標そのものが
	// この検出器の出力だったという意味なので、突き合わせても同じ結果が
	// 出るだけ（自己充足）。ikkyoku は盤の座標が無いと `/api/register` に
	// 送れないので、API 経由の局面は構造的にここに寄る。
	// 実測（133 局面）: api 83 件のうち 73 件、出所なしの 39 件のうち 24 件が
	// px 完全一致で、外した局面は 0 件だった
	Exact bool `json:"exact"`
}

// EvalDetectSummary は盤面検出の集計。
type EvalDetectSummary struct {
	Boards  int `json:"boards"`
	Found   int `json:"found"`   // 信頼度を満たして領域が採用された
	Aligned int `json:"aligned"` // うち手動座標と 0.5マス以内
	Whole   int `json:"whole"`   // 「画像全体が盤面」フォールバックで通った
	Exact   int `json:"exact"`   // 保存座標が検出結果と px 単位で一致（自己充足）
}

// EvalSourceDetect は盤面検出の集計を局面の出所ごとに分けたもの。
//
// **全件をまとめた数字だけを見てはいけない。** 検出の評価は「保存されている
// 座標」を正解として突き合わせるが、その座標がどこから来たかで意味が変わる。
//
//	api      ikkyoku が送ってきた座標。**盤を検出できた時しか送れない**ので、
//	         成功した局面ばかりが集まる。しかも多くは検出器の出力そのもの（Exact）
//	ui       画面で人がドラッグして引いた座標。**検出が外れた時に人が引く**ので、
//	         難しい局面ばかりが集まる
//
// **どちらも偏った標本**であって、混ぜた比率は「実力」ではない。
// 出所ごとに分けて、版を上げたときに同じ母集団どうしで比べられるようにする。
type EvalSourceDetect struct {
	Source string `json:"source"` // "ui" / "api" / ""（旧エントリ）
	EvalDetectSummary
}

// EvalEntry は局面 1 件の評価結果。
type EvalEntry struct {
	ID string `json:"id"`
	// Source は履歴エントリの出所（`EvalSourceDetect` の説明を見ること）
	Source string `json:"source,omitempty"`
	// Error があればその局面は評価できていない（画像が無い・SFEN が壊れている等）
	Error  string     `json:"error,omitempty"`
	Detect EvalDetect `json:"detect"`
	// Manual は手動座標、Auto は自動検出座標での成績（起動中の推論器）
	Manual EvalMetrics `json:"manual"`
	Auto   EvalMetrics `json:"auto"`
	// *Holdout はこの局面を学習から外した k-NN での成績（holdout=false なら nil）
	ManualHoldout *EvalMetrics `json:"manual_holdout,omitempty"`
	AutoHoldout   *EvalMetrics `json:"auto_holdout,omitempty"`
	// ManualLook は**その見た目を丸ごと**学習から外した k-NN での成績（手動座標）。
	// 見た目が未判定の局面では nil
	ManualLook *EvalMetrics `json:"manual_look,omitempty"`
}

// EvalRun は評価の実行 1 回。
type EvalRun struct {
	ID        string `json:"id"`
	CreatedAt string `json:"created_at"`
	// Label は実行につける覚え書き（「v5 外接矩形」など）。空なら自動生成
	Label string `json:"label"`
	// Engine / Samples / DataFile は「何を測ったか」の素性。
	// **これが無いと後から見て何と何を比べているのか分からない**
	Engine   string `json:"engine"`
	Samples  int    `json:"samples"`
	DataFile string `json:"data_file"`
	// Seconds は所要時間（leave-one-out を付けると桁が変わるので残す）
	Seconds float64 `json:"seconds"`
	Holdout bool    `json:"holdout"`

	DetectSummary EvalDetectSummary `json:"detect_summary"`
	// DetectBySource は同じ集計を局面の出所ごとに分けたもの。
	// **混ぜた比率は実力ではない**（`EvalSourceDetect`）
	DetectBySource []EvalSourceDetect `json:"detect_by_source,omitempty"`
	// Negative は data/negative/ の「盤面が写っていない画像」に対する誤検出。
	// **正解データからは測れない唯一の系統**（negative.go）。負例が無ければ nil
	Negative      *NegativeEval `json:"negative,omitempty"`
	Manual        EvalMetrics   `json:"manual"`
	Auto          EvalMetrics   `json:"auto"`
	ManualHoldout *EvalMetrics  `json:"manual_holdout,omitempty"`
	AutoHoldout   *EvalMetrics  `json:"auto_holdout,omitempty"`
	// ManualLook は**見た目ホールドアウト**（その盤を丸ごと学習から外す）の集計。
	//
	// **「どんな盤でも読む」という目標に対する実力はこの数字。**
	// 局面 leave-one-out（`ManualHoldout`）は同じ見た目の別局面が学習データに
	// 残るので高く出る。対象は**見た目が判定済みの局面だけ**なので、
	// `Manual` とは母集団が違う（`ManualLook.Boards` で分かる）
	ManualLook *EvalMetrics `json:"manual_look,omitempty"`

	Entries []EvalEntry `json:"entries"`
	// Skipped は評価に入れられなかった局面数（座標が無い・未確認など）
	Skipped int `json:"skipped"`
}

type EvalHistory struct {
	Runs []EvalRun `json:"runs"`
}

func loadEvalHistory() *EvalHistory {
	f, err := os.Open(evalFile)
	if err != nil {
		return &EvalHistory{Runs: []EvalRun{}}
	}
	defer f.Close()
	var h EvalHistory
	if err := json.NewDecoder(f).Decode(&h); err != nil {
		log.Printf("loadEvalHistory: %v", err)
	}
	if h.Runs == nil {
		h.Runs = []EvalRun{}
	}
	return &h
}

func saveEvalHistory(h *EvalHistory) error {
	os.MkdirAll(dataDir, 0755)
	return writeJSONFile(evalFile, h)
}

// boardGrid は SFEN の盤面部分をマスごとのラベルに展開する。
// 空マスは EmptyLabel、後手は "-" を前置（TestKNNHoldout の表記と同じ）。
func boardGrid(s string) (*[9][9]string, error) {
	var g [9][9]string
	for r := range g {
		for c := range g[r] {
			g[r][c] = suteme.EmptyLabel
		}
	}
	fields := strings.Fields(s)
	if len(fields) == 0 {
		return nil, fmt.Errorf("SFEN が空です")
	}
	err := sfen.ParseBoard(fields[0], func(rank, file, base int, black, promoted bool) {
		// 保存済み SFEN が壊れている場合に備える
		if rank < 0 || rank >= 9 || file < 0 || file >= 9 {
			return
		}
		label := sfen.Letter(base)
		if label == "None" {
			return
		}
		if promoted {
			label = "+" + label
		}
		if !black {
			label = "-" + label
		}
		g[rank][file] = label
	})
	return &g, err
}

// gradeBoard は正解と認識結果を突き合わせて成績を出す。
func gradeBoard(want, got *[9][9]string) EvalMetrics {
	m := EvalMetrics{Boards: 1}
	for r := 0; r < 9; r++ {
		for c := 0; c < 9; c++ {
			w, g := want[r][c], got[r][c]
			m.Cells++
			if w == g {
				m.OK++
			}
			if w == suteme.EmptyLabel {
				if g != suteme.EmptyLabel {
					m.EmptyAsPiece++
				}
				continue
			}
			m.PieceCells++
			switch {
			case w == g:
				m.PieceOK++
			case g == suteme.EmptyLabel:
				m.PieceAsEmpty++
			case strings.HasPrefix(w, "-") != strings.HasPrefix(g, "-"):
				m.OrientFlip++
			default:
				m.TypeMiss++
			}
		}
	}
	if m.OK == m.Cells {
		m.BoardsPerfect = 1
	}
	return m
}

// missedBoard は盤面を検出できなかった局面を 0/81 として数える。
//
// **端から端までの成績なので、検出できなかったことも失点として数える。**
// 認識器の内訳（空→駒など）には入れない（認識を呼んでいないため）。
func missedBoard(want *[9][9]string) EvalMetrics {
	m := EvalMetrics{Boards: 1, NoBoard: 1, Cells: 81}
	for r := 0; r < 9; r++ {
		for c := 0; c < 9; c++ {
			if want[r][c] != suteme.EmptyLabel {
				m.PieceCells++
			}
		}
	}
	return m
}

// recognizeGrid は指定の推論器・オプションで認識してマスのラベルに展開する。
// 盤面を検出できなければ (nil, debug, err) を返す。
func recognizeGrid(img image.Image, p suteme.Predictor, opts ...suteme.Option) (*[9][9]string, *suteme.Debug, error) {
	r, err := suteme.Recognize(img, append(opts, suteme.WithPredictor(p))...)
	if r == nil {
		return nil, nil, err
	}
	g, gerr := boardGrid(r.Board)
	if gerr != nil {
		return nil, r.Debug, gerr
	}
	return g, r.Debug, nil
}

// evalTarget は評価できる局面（画像 + 正解 SFEN + 手動座標が揃っているもの）。
type evalTarget struct {
	entry HistoryEntry
	img   image.Image
	want  *[9][9]string
	// rect は保存されている座標そのもの。**検出のずれを測る基準**なので
	// 寄せない（人が記録した値と比べるための物差し）。
	rect BoardBounds
	// snap は rect を格子線へ寄せたもの。**認識に使うのはこちら。**
	// 学習データも `SnapToGrid` を通した切り出しで作るので、寄せない座標で
	// 測ると「誰も使っていない切り出し」の成績を見ることになる。
	// 実測（157局面、寄せた学習データ）: 手動座標を寄せずに測ると
	// 98.81%・完全一致 95局面まで落ちる（自動座標は 99.91%・147局面）。
	snap image.Rectangle
}

// loadEvalTarget は履歴 1 件を評価できる形に読む。
// **手動座標が無いものは評価に入れない。** 検出のずれを測る基準が無く、
// 自動検出の結果と比べようが無いため
func loadEvalTarget(e HistoryEntry) (*evalTarget, error) {
	if e.BoardBounds == nil {
		return nil, fmt.Errorf("盤面座標が保存されていません")
	}
	f, err := os.Open(filepath.Join(dataDir, e.ID+".png"))
	if err != nil {
		return nil, fmt.Errorf("画像が読めません")
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("画像のデコードに失敗しました")
	}
	want, err := boardGrid(e.SFEN)
	if err != nil {
		return nil, fmt.Errorf("SFEN の解析に失敗しました")
	}
	b := e.BoardBounds
	snap := suteme.SnapToGrid(img, suteme.BoardRegionFromRect(b.X1, b.Y1, b.X2, b.Y2)).Bounds
	return &evalTarget{entry: e, img: img, want: want, rect: *b, snap: snap}, nil
}

// gradeDetect は自動検出の結果を手動指定座標と突き合わせる。
// found は信頼度を満たして採用されたか（棄却された候補も位置は記録する）。
func gradeDetect(r image.Rectangle, src suteme.RegionSource, conf float64, found bool, b *BoardBounds) EvalDetect {
	cw := float64(b.X2-b.X1) / 9
	ch := float64(b.Y2-b.Y1) / 9
	if cw <= 0 || ch <= 0 || r.Empty() {
		return EvalDetect{Confidence: conf}
	}
	res := EvalDetect{
		Found:      found,
		Candidate:  true,
		Source:     string(src),
		Confidence: conf,
		DX:         float64(r.Min.X-b.X1) / cw,
		DY:         float64(r.Min.Y-b.Y1) / ch,
		DW:         float64(r.Dx()-(b.X2-b.X1)) / cw,
		DH:         float64(r.Dy()-(b.Y2-b.Y1)) / ch,
	}
	for _, v := range []float64{res.DX, res.DY, res.DW, res.DH} {
		if a := math.Abs(v); a > res.MaxShift {
			res.MaxShift = a
		}
	}
	res.Aligned = res.MaxShift <= alignedMaxShift
	// 保存座標が検出結果そのものなら、突き合わせても何も測れていない
	res.Exact = r.Min.X == b.X1 && r.Min.Y == b.Y1 && r.Max.X == b.X2 && r.Max.Y == b.Y2
	return res
}

// holdoutPredictors は「何を学習から外した k-NN か」を key ごとに作る。
// 学習に使うのは確認済みの局面だけ（/api/trainhistory と同じ条件）。
//
// key の取り方で意味が変わる。**この 2 つは別の質問に答えている。**
//
//	局面ホールドアウト … key = 局面ID。その局面だけを外す。
//	                     同じ見た目の別局面は学習データに残るので**高く出る**
//	見た目ホールドアウト … key = 見た目の名前。その盤を丸ごと外す＝**初めて見る盤**。
//	                     「どんな盤でも読む」という目標に対する実力はこちら
func holdoutPredictors(entries []HistoryEntry, keyOf func(HistoryEntry) string) map[string]*suteme.KNN {
	byKey := map[string][]suteme.TrainingSample{}
	keys := map[string]bool{}
	for _, e := range entries {
		if !e.IsVerified() {
			continue
		}
		s, err := samplesFromHistory(e)
		if err != nil {
			log.Printf("evaluate: %s のサンプル化に失敗 (%v)", e.ID, err)
			continue
		}
		k := keyOf(e)
		if k == "" {
			continue
		}
		byKey[k] = append(byKey[k], s...)
		keys[k] = true
	}
	out := make(map[string]*suteme.KNN, len(keys))
	for k := range keys {
		var train []suteme.TrainingSample
		for other, s := range byKey {
			if other != k {
				train = append(train, s...)
			}
		}
		if kn := suteme.NewKNN(MergeSamples(train)); kn != nil {
			out[k] = kn
		}
	}
	return out
}

// runEvaluation は選択した局面の認識率を測って 1 レコードにまとめる。
// ids が空なら評価できる局面すべてを対象にする。
func runEvaluation(ids []string, label string, holdout bool) (*EvalRun, error) {
	p := currentPredictor()
	if p == nil {
		return nil, fmt.Errorf("推論器がありません（先に履歴タブで学習してください）")
	}
	historyMu.RLock()
	h := loadHistory()
	historyMu.RUnlock()

	selected := map[string]bool{}
	for _, id := range ids {
		selected[id] = true
	}

	start := time.Now()
	run := &EvalRun{
		ID:        newID(),
		CreatedAt: time.Now().Format("2006-01-02 15:04"),
		Label:     strings.TrimSpace(label),
		DataFile:  dataFile,
		Holdout:   holdout,
	}
	if d, ok := p.(suteme.DebugPredictor); ok {
		run.Engine = d.Debug().Kind
	}
	modelLock.RLock()
	if knn != nil {
		run.Samples = knn.Len()
	}
	modelLock.RUnlock()

	// 対象を先に確定する（画像・SFEN・手動座標が揃っているものだけ）
	var targets []*evalTarget
	for _, e := range h.Entries {
		if len(selected) > 0 && !selected[e.ID] {
			continue
		}
		t, err := loadEvalTarget(e)
		if err != nil {
			run.Entries = append(run.Entries, EvalEntry{ID: e.ID, Error: err.Error()})
			run.Skipped++
			continue
		}
		targets = append(targets, t)
	}
	if len(targets) == 0 {
		return nil, fmt.Errorf("評価できる局面がありません（画像・正解 SFEN・盤面座標が揃っている必要があります）")
	}

	var holdoutKNN, lookKNN map[string]*suteme.KNN
	if holdout {
		log.Printf("evaluate: leave-one-out 用の学習データを作成中（%d 局面）", len(h.Entries))
		holdoutKNN = holdoutPredictors(h.Entries, func(e HistoryEntry) string { return e.ID })
		run.ManualHoldout = &EvalMetrics{}
		run.AutoHoldout = &EvalMetrics{}

		// **見た目ホールドアウト＝目標（どんな盤でも読む）に対する実力。**
		// 束ねる数は見た目の数なので、局面ごとに組み直すより速い
		lookKNN = holdoutPredictors(h.Entries, func(e HistoryEntry) string {
			return strings.TrimSpace(e.Look)
		})
		if len(lookKNN) > 0 {
			run.ManualLook = &EvalMetrics{}
		}
	}

	bySource := map[string]*EvalDetectSummary{}
	for i, t := range targets {
		res := EvalEntry{ID: t.entry.ID, Source: t.entry.Source}

		// 手動座標（格子線へ寄せたもの）。認識器そのものの成績
		if g, _, err := recognizeGrid(t.img, p, suteme.WithRect(t.snap.Min.X, t.snap.Min.Y, t.snap.Max.X, t.snap.Max.Y)); err == nil {
			res.Manual = gradeBoard(t.want, g)
		} else {
			res.Error = err.Error()
		}

		// 自動検出。検出できなければ 0/81
		g, dbg, err := recognizeGrid(t.img, p)
		if err == nil && dbg != nil {
			res.Detect = gradeDetect(dbg.Region, dbg.RegionSource, dbg.Confidence, true, t.entry.BoardBounds)
			res.Auto = gradeBoard(t.want, g)
		} else {
			// 棄却された候補の位置と信頼度も残す。次の版で拾えるように
			// なったかを追えるようにするため（0/81 という結果だけでは分からない）
			if br := suteme.DetectBoard(t.img); br != nil {
				res.Detect = gradeDetect(br.Bounds, suteme.RegionFromDetect,
					suteme.ValidateBoard(t.img, br), false, t.entry.BoardBounds)
			}
			res.Auto = missedBoard(t.want)
		}

		// 見た目ホールドアウトは**手動座標だけ**で測る。盤面検出は学習データに
		// 依存しないので、自動で測っても局面ホールドアウトと同じ数字になる
		if look := strings.TrimSpace(t.entry.Look); look != "" {
			if kn, ok := lookKNN[look]; ok {
				ml := EvalMetrics{}
				if g, _, err := recognizeGrid(t.img, kn, suteme.WithRect(t.snap.Min.X, t.snap.Min.Y, t.snap.Max.X, t.snap.Max.Y)); err == nil {
					ml = gradeBoard(t.want, g)
				}
				res.ManualLook = &ml
				run.ManualLook.add(ml)
			}
		}

		if kn, ok := holdoutKNN[t.entry.ID]; ok {
			mh := EvalMetrics{}
			if g, _, err := recognizeGrid(t.img, kn, suteme.WithRect(t.snap.Min.X, t.snap.Min.Y, t.snap.Max.X, t.snap.Max.Y)); err == nil {
				mh = gradeBoard(t.want, g)
			}
			ah := missedBoard(t.want)
			if g, _, err := recognizeGrid(t.img, kn); err == nil {
				ah = gradeBoard(t.want, g)
			}
			res.ManualHoldout, res.AutoHoldout = &mh, &ah
			run.ManualHoldout.add(mh)
			run.AutoHoldout.add(ah)
		}

		run.Manual.add(res.Manual)
		run.Auto.add(res.Auto)
		src := bySource[res.Source]
		if src == nil {
			src = &EvalDetectSummary{}
			bySource[res.Source] = src
		}
		for _, d := range []*EvalDetectSummary{&run.DetectSummary, src} {
			d.Boards++
			if res.Detect.Exact {
				d.Exact++
			}
			if !res.Detect.Found {
				continue
			}
			d.Found++
			if res.Detect.Aligned {
				d.Aligned++
			}
			if res.Detect.Source == string(suteme.RegionFromWholeImage) {
				d.Whole++
			}
		}
		run.Entries = append(run.Entries, res)
		det := fmt.Sprintf("%.2fマス conf=%.2f", res.Detect.MaxShift, res.Detect.Confidence)
		switch {
		case !res.Detect.Candidate:
			det = "候補なし"
		case !res.Detect.Found:
			det += "（棄却）"
		}
		log.Printf("evaluate [%d/%d] %s: 手動 %d/81 自動 %d/81 検出 %s",
			i+1, len(targets), t.entry.ID, res.Manual.OK, res.Auto.OK, det)
	}
	// **人が引いた座標（ui）を先頭に置く。** 検出の実力に近いのはこちらで、
	// api は「検出できた局面だけ」が集まる標本なので後ろでよい
	for _, src := range []string{SourceUI, SourceAPI, ""} {
		if d := bySource[src]; d != nil {
			run.DetectBySource = append(run.DetectBySource, EvalSourceDetect{Source: src, EvalDetectSummary: *d})
			delete(bySource, src)
		}
	}
	for src, d := range bySource { // 将来増えた出所も落とさない
		run.DetectBySource = append(run.DetectBySource, EvalSourceDetect{Source: src, EvalDetectSummary: *d})
	}

	// 盤面が写っていない画像。**誤検出はここでしか測れない**
	run.Negative = evaluateNegatives(p)
	run.Seconds = time.Since(start).Seconds()
	if run.Label == "" {
		run.Label = fmt.Sprintf("%s / %d サンプル", run.DataFile, run.Samples)
	}
	return run, nil
}

// appendEvalRun は実行結果を data/accuracy.json の先頭に足す。
func appendEvalRun(run *EvalRun) error {
	evalMu.Lock()
	defer evalMu.Unlock()
	h := loadEvalHistory()
	h.Runs = append([]EvalRun{*run}, h.Runs...)
	if len(h.Runs) > maxEvalRuns {
		h.Runs = h.Runs[:maxEvalRuns]
	}
	return saveEvalHistory(h)
}

// handleEvaluate は評価を実行して結果を保存する（ループバック限定）。
//
// **同期で返す。** leave-one-out を付けると局面数に比例して数分かかるが、
// 学習（NN 込みで数十分）と同じ扱いにしてある。進捗はサーバのログに出る。
func handleEvaluate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		// IDs が空なら評価できる局面すべて
		IDs   []string `json:"ids"`
		Label string   `json:"label"`
		// Holdout: その局面を学習から外した k-NN でも測るか
		Holdout bool `json:"holdout"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httpJSONError(w, http.StatusBadRequest, err.Error())
		return
	}
	run, err := runEvaluation(req.IDs, req.Label, req.Holdout)
	if err != nil {
		httpJSONError(w, http.StatusBadRequest, err.Error())
		return
	}
	if err := appendEvalRun(run); err != nil {
		httpJSONError(w, http.StatusInternalServerError, "評価結果を保存できませんでした: "+err.Error())
		return
	}
	neg := ""
	if run.Negative != nil {
		neg = fmt.Sprintf(" 負例 %d/%d 棄却", run.Negative.Rejected, run.Negative.Images)
	}
	log.Printf("evaluate: %s 手動 %d/%d 自動 %d/%d 検出 %d/%d%s (%.1fs)",
		run.Label, run.Manual.OK, run.Manual.Cells, run.Auto.OK, run.Auto.Cells,
		run.DetectSummary.Aligned, run.DetectSummary.Boards, neg, run.Seconds)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(run)
}

// handleEvaluations は保存済みの評価結果を返す / 削除する（ループバック限定）。
//
//	GET    /api/evaluations       実行の一覧（新しい順）
//	DELETE /api/evaluations/{id}  実行 1 件を削除
func handleEvaluations(w http.ResponseWriter, r *http.Request) {
	id := strings.Trim(strings.TrimPrefix(r.URL.Path, "/api/evaluations"), "/")
	switch {
	case r.Method == http.MethodGet && id == "":
		evalMu.Lock()
		h := loadEvalHistory()
		evalMu.Unlock()
		// ?latest=1 は最新の実行 1 件だけを返す。履歴タブの行に出す認識率は
		// 先頭の実行しか見ないのに、全文は局面ごとの内訳を抱えた実行が
		// 100 件まで載っていて実測 1.05MB ある（評価タブは全件を引く）
		if r.URL.Query().Get("latest") != "" && len(h.Runs) > 0 {
			h = &EvalHistory{Runs: h.Runs[:1]}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(h)
	case r.Method == http.MethodDelete && validID(id):
		evalMu.Lock()
		h := loadEvalHistory()
		kept := h.Runs[:0]
		found := false
		for _, run := range h.Runs {
			if run.ID == id {
				found = true
				continue
			}
			kept = append(kept, run)
		}
		if found {
			h.Runs = kept
			saveEvalHistory(h)
		}
		evalMu.Unlock()
		if !found {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok", "id": id})
	default:
		http.NotFound(w, r)
	}
}
