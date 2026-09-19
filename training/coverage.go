package training

import (
	"encoding/json"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/ShinteLab/suteme"
)

// 盤の「見た目」ごとの厚み。
//
// **認識率は局面数より「同じ見た目の盤が何枚あるか」で決まる。**
// 実測（`TestSameSourceEffect`）で同じ出所 1 枚が他所 111 枚に匹敵し、
// 3 枚で 95.0%・5 枚で 97.9% に届く。ところが**集まり方がその方針に
// 沿っているかを知る手段が無かった**ので、キャプチャする人は
// 「次にどの盤を撮ればいいか」を判断できなかった。
//
// **見た目は人が判定する（`HistoryEntry.Look`）。機械には決められない。**
//
//   - **画像サイズは当てにならない。** 中継を見ながら都度サイズを変えて
//     撮るので、同じ出所でも毎回違うサイズになる。最初この実装は
//     サイズで束ねていて、「103 通りのうち 82 が 1 枚だけ」という
//     **誤った結論**を出した
//   - **時刻の近さも決め手にならない。** 同じ放送の中に実盤・大盤・
//     ゲーム画面という別の見た目が混ざる。実測でも、時刻が 10 分以内でも
//     署名が遠いペアは役に立ち方が落ちる（88.6% → 76.0%）
//
// **ここでいう「見た目」は `HistoryEntry.Source`（api / ui）とは別物。**
// あちらは「誰が登録したか」で、こちらは「どの画面を写したものか」。
// 同じ語を使うと評価タブの出所別集計（`EvalSourceDetect`）と混ざるので、
// 画面でも「見た目」と呼ぶ。

// lookEnough は 1 つの見た目について「これ以上は目標に効かない」枚数。
//
// **suteme の目標は「どんな盤でも読む」こと**（AGENTS.md 冒頭）なので、
// 同じ見た目を積んでも目標には効かない。同じ枚数をどう割るかの実測
// （`TestVarietyVsThickness`。対象の見た目は丸ごと学習から外す＝初見の盤）:
//
//	8 種類 × 1 枚  79.3%
//	4 種類 × 2 枚  77.7%
//	2 種類 × 4 枚  76.3%
//	1 種類 × 8 枚  70.6%
//
// **1 枚ずつ種類を増やすのが最良。** 2 枚目でわずかに保険が利く程度なので、
// 2 枚を「足りている」の線にして、それ以上は積みすぎとして下へ回す。
// （特定の中継だけ読み切りたいなら 3〜5 枚積むと 95〜98% に届くが、
// それは特化であって目標ではない）
const lookEnough = 2

// lookProposeMax は「同じ見た目かもしれない」と候補に出す署名距離の上限。
//
// 署名（`lookSignature`）の距離と「その 1 局面だけを学習データにして
// 相手を読んだときの一致率」の実測（`TestLookHelpVsDistance`、568 ペア）:
//
//	署名距離     ペア数  平均一致率  うちサイズも同じ
//	〜0.02        28     91.8%      92.9%
//	0.02〜0.04    69     87.2%      20.3%   ← ほとんどがサイズ違い
//	0.04〜0.06    97     78.5%       6.2%
//	0.06〜0.10    65     75.7%       0.0%
//	0.20〜       212     68.6%       0.0%
//
// 0.04 を境に効きが落ちる。**候補は人が確認して確定させるので、
// 拾い漏らすより多めに出すほうがよい**として 0.05 にしてある
// （0.02〜0.04 の帯は 8 割がサイズ違い＝サイズで束ねると必ず割れる）。
const lookProposeMax = 0.05

// LookGroup は同じ見た目の盤の枚数と、最新の評価での成績。
type LookGroup struct {
	// Look は見た目の名前。人が付けたものがあればそれ、
	// まだ判定されていなければ空（Proposed が真になる）
	Look string `json:"look"`
	// Proposed は署名から機械が出した候補（＝人がまだ判定していない）
	Proposed bool     `json:"proposed"`
	Count    int      `json:"count"`
	IDs      []string `json:"ids"`
	// Cells / OK は最新の評価実行での一致マス数。評価がまだ無い、または
	// その局面が評価に含まれていなければ 0
	Cells int `json:"cells"`
	OK    int `json:"ok"`
	// Rate は OK/Cells（0.0〜1.0）。Cells が 0 なら -1（＝未評価）
	Rate float64 `json:"rate"`
	// Excess は lookEnough を超えて積んである（＝これ以上撮っても目標には効かない）
	Excess bool `json:"excess"`
}

// CoverageReport は「次にどの盤を集めるか」を決めるための一覧。
type CoverageReport struct {
	// Target は「1 つの見た目にこれ以上積んでも目標には効かない」枚数（lookEnough）
	Target int         `json:"target"`
	Groups []LookGroup `json:"groups"`
	// Named は人が見た目を判定済みの局面数
	Named int `json:"named"`
	Total int `json:"total"`
	// Run は成績の出どころ（最新の評価実行）。評価がまだ無ければ nil
	Run *CoverageRun `json:"run,omitempty"`
}

// CoverageRun は成績を取った評価実行の素性。
//
// **どの実行の数字かを必ず添えること。** leave-one-out を外した実行は
// 学習済みの局面で高く出るので、薄い見た目でも良い成績に見えてしまう。
type CoverageRun struct {
	ID        string `json:"id"`
	Label     string `json:"label"`
	CreatedAt string `json:"created_at"`
	Holdout   bool   `json:"holdout"`
}

// lookSignature は「盤の見た目」の署名を返す。
//
// **空マスの平均ベクトル。** 空マスは盤の木目・格子線・地色をそのまま写していて、
// **どの駒が乗っているか（局面の中身）に左右されない**。駒のあるマスで作ると、
// 同じ盤でも駒の並びが違えば別物になってしまう。
// `CellToInput` は 24x24 に落として標準化するので、撮影サイズの違いは吸収される。
func lookSignature(e HistoryEntry) ([]float64, bool) {
	if e.BoardBounds == nil {
		return nil, false
	}
	f, err := os.Open(filepath.Join(dataDir, e.ID+".png"))
	if err != nil {
		return nil, false
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, false
	}
	want, err := boardGrid(e.SFEN)
	if err != nil {
		return nil, false
	}
	b := e.BoardBounds
	br := suteme.BoardRegionFromRect(b.X1, b.Y1, b.X2, b.Y2)
	sig := make([]float64, suteme.InputSize)
	n := 0
	for r := 0; r < 9; r++ {
		for c := 0; c < 9; c++ {
			if want[r][c] != suteme.EmptyLabel {
				continue
			}
			cell := br.ExtractCell(img, r, c)
			if cell == nil {
				continue
			}
			for i, v := range suteme.CellToInput(cell) {
				sig[i] += v
			}
			n++
		}
	}
	if n == 0 {
		return nil, false
	}
	for i := range sig {
		sig[i] /= float64(n)
	}
	return sig, true
}

// 署名は画像を 1 枚ずつ読む（141 枚で 7 秒）ので、画面から叩かれるたびに
// 作り直さないよう覚えておく。画像は上書き保存されうるので更新時刻で確かめる。
var (
	sigMu    sync.Mutex
	sigCache = map[string]cachedSig{}
)

type cachedSig struct {
	vec []float64
	mod time.Time
	ok  bool
}

func cachedSignature(e HistoryEntry) ([]float64, bool) {
	st, err := os.Stat(filepath.Join(dataDir, e.ID+".png"))
	if err != nil {
		return nil, false
	}
	sigMu.Lock()
	defer sigMu.Unlock()
	if c, hit := sigCache[e.ID]; hit && c.mod.Equal(st.ModTime()) {
		return c.vec, c.ok
	}
	vec, ok := lookSignature(e)
	sigCache[e.ID] = cachedSig{vec: vec, mod: st.ModTime(), ok: ok}
	return vec, ok
}

func sigDistance(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return s / float64(len(a))
}

// proposeLooks は見た目が未判定の局面を署名で束ねた候補を返す。
//
// **完全連結**（塊の中のいちばん遠い 2 枚が `lookProposeMax` 以内）で束ねる。
// 単連結だと「A と B が近い、B と C が近い」の連鎖で無関係な盤まで
// 1 つの塊に流れ込む。人が確認する前提なので、繋げすぎるより細かく出す。
func proposeLooks(ids []string, sigs map[string][]float64) [][]string {
	groups := make([][]string, 0, len(ids))
	for _, id := range ids {
		if _, ok := sigs[id]; ok {
			groups = append(groups, []string{id})
		}
	}
	for {
		bestDist, bestA, bestB := lookProposeMax, -1, -1
		for a := 0; a < len(groups); a++ {
			for b := a + 1; b < len(groups); b++ {
				far := 0.0
				for _, x := range groups[a] {
					for _, y := range groups[b] {
						if d := sigDistance(sigs[x], sigs[y]); d > far {
							far = d
						}
					}
				}
				if far <= bestDist {
					bestDist, bestA, bestB = far, a, b
				}
			}
		}
		if bestA < 0 {
			break
		}
		groups[bestA] = append(groups[bestA], groups[bestB]...)
		groups = append(groups[:bestB], groups[bestB+1:]...)
	}
	return groups
}

// lookGroups は履歴を見た目ごとに束ねる。
//
// **人が付けた名前が優先。** 付いていないものだけ署名で候補にまとめる。
// 評価の集計（`coverageReport`）と調査用テスト（`TestSourceCoverage`）で
// **同じ束ね方を使う**ためにここに置いてある。
func lookGroups(entries []HistoryEntry) (named map[string][]string, proposed [][]string) {
	named = map[string][]string{}
	var unnamed []string
	sigs := map[string][]float64{}
	for _, e := range entries {
		if look := strings.TrimSpace(e.Look); look != "" {
			named[look] = append(named[look], e.ID)
			continue
		}
		if vec, ok := cachedSignature(e); ok {
			sigs[e.ID] = vec
			unnamed = append(unnamed, e.ID)
		}
	}
	return named, proposeLooks(unnamed, sigs)
}

// coverageReport は履歴と最新の評価実行から見た目ごとの厚みを組み立てる。
func coverageReport(h *HistoryData, runs []EvalRun) *CoverageReport {
	rep := &CoverageReport{Target: lookEnough, Groups: []LookGroup{}}
	if h == nil {
		return rep
	}
	// 成績は最新の実行から取る。局面ごとの内訳を持つ実行だけが対象
	var run *EvalRun
	for i := range runs {
		if len(runs[i].Entries) > 0 {
			run = &runs[i]
			break
		}
	}
	// ID → 一致マス数。**手動座標での成績を使う**（見たいのは「この盤の駒を
	// 読めているか」であって、盤面検出が当たるかではない）。
	// leave-one-out があればそちらを優先する
	type cell struct{ cells, ok int }
	score := map[string]cell{}
	if run != nil {
		rep.Run = &CoverageRun{
			ID: run.ID, Label: run.Label, CreatedAt: run.CreatedAt, Holdout: run.Holdout,
		}
		for _, e := range run.Entries {
			m := e.Manual
			if e.ManualHoldout != nil {
				m = *e.ManualHoldout
			}
			score[e.ID] = cell{cells: m.Cells, ok: m.OK}
		}
	}

	named, proposed := lookGroups(h.Entries)
	for _, e := range h.Entries {
		rep.Total++
		if strings.TrimSpace(e.Look) != "" {
			rep.Named++
		}
	}

	add := func(look string, proposed bool, ids []string) {
		g := LookGroup{Look: look, Proposed: proposed, Count: len(ids), IDs: ids, Rate: -1}
		g.Excess = g.Count > lookEnough
		for _, id := range ids {
			if s, ok := score[id]; ok {
				g.Cells += s.cells
				g.OK += s.ok
			}
		}
		if g.Cells > 0 {
			g.Rate = float64(g.OK) / float64(g.Cells)
		}
		rep.Groups = append(rep.Groups, g)
	}
	for look, ids := range named {
		add(look, false, ids)
	}
	for _, ids := range proposed {
		add("", true, ids)
	}

	// **枚数の多い順。** 目標は種類を増やすことなので、この表で分かるのは
	// 「もうこれ以上撮らなくていい見た目」のほう（＝上に出るもの）。
	// **「薄い順に潰す」ではない。** 薄い見た目を厚くしても目標には効かない
	// （`lookEnough`）。次に何を撮るかは「まだ無い種類」なので画面には出せない
	sort.Slice(rep.Groups, func(i, j int) bool {
		a, b := rep.Groups[i], rep.Groups[j]
		if a.Count != b.Count {
			return a.Count > b.Count
		}
		if (a.Rate < 0) != (b.Rate < 0) {
			return b.Rate < 0
		}
		if a.Rate != b.Rate {
			return a.Rate < b.Rate
		}
		return a.Look < b.Look
	})
	return rep
}

// handleCoverage は見た目ごとの厚みを返す（ループバック限定）。
func handleCoverage(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "GET only", http.StatusMethodNotAllowed)
		return
	}
	historyMu.RLock()
	h := loadHistory()
	historyMu.RUnlock()
	evalMu.Lock()
	eh := loadEvalHistory()
	evalMu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(coverageReport(h, eh.Runs))
}

// handleLook は見た目の名前を付ける／外す（ループバック限定）。
//
// **人の判定が正解データ。** 署名から出した候補はあくまで並べ替えの材料で、
// 「同じ見た目か」は画像を見た人にしか決められない。
func handleLook(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		IDs []string `json:"ids"`
		// Look が空なら判定を外す（未判定に戻す）
		Look string `json:"look"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httpJSONError(w, http.StatusBadRequest, err.Error())
		return
	}
	if len(req.IDs) == 0 {
		httpJSONError(w, http.StatusBadRequest, "ids が空です")
		return
	}
	look := strings.TrimSpace(req.Look)
	if len(look) > 60 {
		httpJSONError(w, http.StatusBadRequest, "見た目の名前が長すぎます")
		return
	}
	want := map[string]bool{}
	for _, id := range req.IDs {
		want[id] = true
	}

	historyMu.Lock()
	h := loadHistory()
	n := 0
	for i := range h.Entries {
		if want[h.Entries[i].ID] {
			h.Entries[i].Look = look
			n++
		}
	}
	if n > 0 {
		saveHistoryFile(h)
	}
	historyMu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status": "ok", "updated": n, "look": look,
	})
}
