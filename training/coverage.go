package training

import (
	"encoding/json"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"net/http"
	"os"
	"path/filepath"
	"sort"
)

// 盤の「見た目」ごとの厚み。
//
// **認識率は局面数より「同じ見た目の盤が何枚あるか」で決まる。**
// 実測（`TestSameSourceEffect`）で同じ出所 1 枚が他所 111 枚に匹敵し、
// 3 枚で 95.0%・5 枚で 97.9% に届く。ところが**集めた結果がその方針に
// 沿っているかを画面から知る手段が無かった**ので、キャプチャする人は
// 「次にどの盤を撮ればいいか」を判断できなかった。
// 実測（141局面）: 103 通りの見た目のうち **82 が 1 枚だけ**だった。
//
// **ここでいう「見た目」は `HistoryEntry.Source`（api / ui）とは別物。**
// あちらは「誰が登録したか」で、こちらは「どの画面を写したものか」。
// 同じ語を使うと評価タブの出所別集計（`EvalSourceDetect`）と混ざるので、
// 画面でも「見た目」と呼ぶ。

// lookTarget は 1 つの見た目について集めたい枚数。
//
// 実測（`TestSameSourceEffect`、同じ出所 7 枚を対象に駒種の一致率）:
// 1 枚 87.8% / 2 枚 93.4% / **3 枚 95.0%** / 5 枚 97.9% / 6 枚 98.1%。
// 3 枚で伸びが緩むので、まず全部の見た目を 3 枚にするのを目標にする
// （5 枚まで積むより、薄い見た目を潰すほうが 1 枚あたりの効きが大きい）。
const lookTarget = 3

// LookGroup は同じ見た目の盤の枚数と、最新の評価での成績。
type LookGroup struct {
	// Look は見た目のキー。画像サイズで代用する（同じ画面をキャプチャすれば
	// 一致する）。**同じサイズの別物・同じ出所の別サイズは分けられない**ので
	// 枚数は目安。`TestSameSourceEffect` / `TestSourceCoverage` と同じ割り方
	Look  string   `json:"look"`
	Count int      `json:"count"`
	IDs   []string `json:"ids"`
	// Cells / OK は最新の評価実行での一致マス数。評価がまだ無い、または
	// その局面が評価に含まれていなければ 0
	Cells int `json:"cells"`
	OK    int `json:"ok"`
	// Rate は OK/Cells（0.0〜1.0）。Cells が 0 なら -1（＝未評価）
	Rate float64 `json:"rate"`
	// Thin は lookTarget に足りていないか
	Thin bool `json:"thin"`
}

// CoverageReport は「次にどの盤を集めるか」を決めるための一覧。
type CoverageReport struct {
	Target int         `json:"target"`
	Groups []LookGroup `json:"groups"`
	// Run は成績の出どころ（最新の評価実行）。評価がまだ無ければ nil
	Run *CoverageRun `json:"run,omitempty"`
}

// CoverageRun は成績を取った評価実行の素性。
//
// **どの実行の数字かを必ず添えること。** leave-one-out を外した実行は
// 学習済みの局面で高く出るので、薄い見た目でも良い数字が並んでしまう。
type CoverageRun struct {
	ID        string `json:"id"`
	Label     string `json:"label"`
	CreatedAt string `json:"created_at"`
	Holdout   bool   `json:"holdout"`
}

// lookOf は保存画像のサイズを見た目のキーとして返す。
// 画像ヘッダだけを読むので、141 枚でも数 ms で済む。
func lookOf(id string) (string, error) {
	f, err := os.Open(filepath.Join(dataDir, id+".png"))
	if err != nil {
		return "", err
	}
	defer f.Close()
	cfg, _, err := image.DecodeConfig(f)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%dx%d", cfg.Width, cfg.Height), nil
}

// coverageReport は履歴と最新の評価実行から見た目ごとの厚みを組み立てる。
func coverageReport(h *HistoryData, runs []EvalRun) *CoverageReport {
	rep := &CoverageReport{Target: lookTarget, Groups: []LookGroup{}}
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

	byLook := map[string]*LookGroup{}
	for _, e := range h.Entries {
		look, err := lookOf(e.ID)
		if err != nil {
			continue // 画像が無いものは数えない
		}
		g := byLook[look]
		if g == nil {
			g = &LookGroup{Look: look, Rate: -1}
			byLook[look] = g
		}
		g.Count++
		g.IDs = append(g.IDs, e.ID)
		if s, ok := score[e.ID]; ok {
			g.Cells += s.cells
			g.OK += s.ok
		}
	}

	for _, g := range byLook {
		g.Thin = g.Count < lookTarget
		if g.Cells > 0 {
			g.Rate = float64(g.OK) / float64(g.Cells)
		}
		rep.Groups = append(rep.Groups, *g)
	}
	// **薄いものを先に、その中で読めていない順。** 1 枚しかなくても
	// よく読めている見た目は足す価値が薄いので、下のほうへ回る。
	// 未評価（Rate < 0）は判断材料が無いので各群の末尾
	sort.Slice(rep.Groups, func(i, j int) bool {
		a, b := rep.Groups[i], rep.Groups[j]
		if a.Thin != b.Thin {
			return a.Thin
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
