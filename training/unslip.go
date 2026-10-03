package training

import (
	"fmt"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/ShinteLab/suteme"
)

// 盤の縁の帯の教師データを、保存済み局面から作る。
//
// **新しくラベルを付けてもらう必要は無い。** `history.json` の局面には
// 人が引いた盤の矩形が入っているので、そこから
// **「縁の帯＝盤」4 本と「1マス外の帯＝盤の外」4 本**が機械的に取れる。
// 局面 1 つにつき 8 本、138 局面で 1104 本。
//
// 駒種の学習（`samplesFromHistory`）とは**別のデータ**で、混ぜてはいけない。
// 入力の意味が違ううえ、`MergeSamples` は駒の入力ベクトル用。

// stripSamplesFromRegion は 1 局面から帯のサンプルを作る。
//
// **内と外を必ず対で作る。** 判定器は「この帯は盤か」を絶対値で答えるが、
// 使うのは常に「失う帯 対 得る帯」の比較なので、同じ画像から両方を
// 入れておかないと、盤ごとの明るさの癖を「盤らしさ」と取り違える。
func stripSamplesFromRegion(img image.Image, x1, y1, x2, y2 int) []suteme.StripSample {
	cw, ch := (x2-x1)/9, (y2-y1)/9
	if cw < 8 || ch < 8 {
		return nil
	}
	col := func(cx int) image.Rectangle { return image.Rect(cx, y1, cx+cw, y2) }
	row := func(cy int) image.Rectangle { return image.Rect(x1, cy, x2, cy+ch) }

	pairs := []struct {
		in, out  image.Rectangle
		vertical bool
	}{
		{col(x1), col(x2), true},            // 左端の列 / その右外
		{col(x2 - cw), col(x1 - cw), true},  // 右端の列 / その左外
		{row(y1), row(y2), false},           // 上端の行 / その下外
		{row(y2 - ch), row(y1 - ch), false}, // 下端の行 / その上外
	}

	out := make([]suteme.StripSample, 0, len(pairs)*2)
	for _, p := range pairs {
		a := suteme.StripInput(img, p.in, p.vertical)
		b := suteme.StripInput(img, p.out, p.vertical)
		if a == nil || b == nil {
			continue
		}
		out = append(out,
			suteme.StripSample{Input: a, Board: true},
			suteme.StripSample{Input: b, Board: false})
	}
	return out
}

// stripSamplesFromHistory は履歴エントリ 1 件から帯のサンプルを作る。
// **盤面座標が無いエントリは対象外**（縁がどこか分からない）。
func stripSamplesFromHistory(e HistoryEntry) ([]suteme.StripSample, error) {
	if e.BoardBounds == nil {
		return nil, fmt.Errorf("盤面座標がありません")
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
	// 駒の学習（`samplesFromHistory`）と同じく格子線へ寄せてから帯を取る。
	// 帯は「盤の縁の 1マス」なので、枠が数px ずれるとその分だけ別の画素を
	// 見ることになり、判定器と検出側で見ているものが食い違う
	b := e.BoardBounds
	r := suteme.SnapToGrid(img, suteme.BoardRegionFromRect(b.X1, b.Y1, b.X2, b.Y2)).Bounds
	s := stripSamplesFromRegion(img, r.Min.X, r.Min.Y, r.Max.X, r.Max.Y)
	if len(s) == 0 {
		return nil, fmt.Errorf("帯を作れませんでした")
	}
	return s, nil
}

// 帯は局面ごとに覚えておく（`SnapToGrid` が重い）。
//
// **作り直しをやめるのではなく、変わっていない局面の計算を飛ばす。**
// 帯データは毎回ゼロから組み直す約束（消した局面の帯を残さない）だが、
// **費用の実体は「確認済みの全局面ぶんの `SnapToGrid`」**で、
// 学習で選んだ局面数には関係なく効いてくる。実測（157局面）:
// 帯の作り直し 39.6s のうち **`SnapToGrid` が 34.6s**・PNG デコードが 3.7s。
// 1 局だけ選んでも 40 秒かかっていたのはここ。
//
// 画像は上書き保存されうるので**更新時刻と大きさで確かめる**。
// 盤面座標を引き直しても帯は変わるので**座標もキーに入れる**
// （サムネイル `thumb.go` や見た目の署名 `sigCache` と同じ流儀）。
var (
	stripMu    sync.Mutex
	stripCache = map[string]cachedStrips{}
)

type cachedStrips struct {
	samples []suteme.StripSample
	mod     time.Time
	size    int64
	bounds  BoardBounds
}

// cachedStripSamples は帯を覚えておいて返す。
// 画像が差し替わったか盤面座標が変わっていれば作り直す。
func cachedStripSamples(e HistoryEntry) []suteme.StripSample {
	if e.BoardBounds == nil {
		return nil
	}
	st, err := os.Stat(filepath.Join(dataDir, e.ID+".png"))
	if err != nil {
		return nil
	}

	// **計算の間もロックを持ったままにする。** 起動直後の温めと人が押した
	// 学習が重なりうるので、離すと同じ局面を 2 回計算することになる
	stripMu.Lock()
	defer stripMu.Unlock()
	if c, hit := stripCache[e.ID]; hit &&
		c.mod.Equal(st.ModTime()) && c.size == st.Size() && c.bounds == *e.BoardBounds {
		return c.samples
	}
	// 失敗（画像が読めない・帯が作れない）も覚える。
	// 覚えないと、壊れた 1 件のために毎回デコードを試すことになる
	s, err := stripSamplesFromHistory(e)
	if err != nil {
		s = nil
	}
	stripCache[e.ID] = cachedStrips{
		samples: s, mod: st.ModTime(), size: st.Size(), bounds: *e.BoardBounds,
	}
	return s
}

// BuildStripData は確認済みの全局面から帯の教師データを作り直す。
//
// **毎回ゼロから作る（累積しない）。** 駒の学習データと違って
// 1 局面あたり 8 本しか無く、全件でも 1000 本強。累積すると、
// 局面を消したときに古い帯が残り続ける。
// 実際の計算は局面ごとに覚えてある（`cachedStripSamples`）ので、
// 作り直しても変わった局面のぶんしか走らない。
func BuildStripData(entries []HistoryEntry) []suteme.StripSample {
	var out []suteme.StripSample
	live := make(map[string]bool, len(entries))
	for _, e := range entries {
		live[e.ID] = true
		if !e.IsVerified() {
			continue
		}
		out = append(out, cachedStripSamples(e)...)
	}

	// 消された局面のぶんは覚えたままにしない
	stripMu.Lock()
	for id := range stripCache {
		if !live[id] {
			delete(stripCache, id)
		}
	}
	stripMu.Unlock()
	return out
}

// warmStripCache は帯のキャッシュを裏で埋める（起動時に呼ぶ）。
// **判定器やファイルには触らない。** 計算を先にやっておくだけなので、
// 途中で失敗しても学習がその場で計算し直すだけで済む。
func warmStripCache() {
	historyMu.RLock()
	h := loadHistory()
	historyMu.RUnlock()
	if h == nil || len(h.Entries) == 0 {
		return
	}
	st := time.Now()
	n := len(BuildStripData(h.Entries))
	logger().Info("Warmed strip cache", "strips", n, "elapsed", time.Since(st))
}

// rebuildStripData は確認済みの全局面から帯データを作り直して保存し、
// 既定の判定器を差し替える。作れた帯の本数を返す。
func rebuildStripData() (int, error) {
	historyMu.RLock()
	h := loadHistory()
	historyMu.RUnlock()
	if h == nil {
		return 0, nil
	}

	samples := BuildStripData(h.Entries)
	if len(samples) == 0 {
		return 0, nil
	}
	if err := suteme.SaveStripData(stripFile, samples); err != nil {
		return 0, err
	}
	suteme.SetStripJudge(suteme.NewStripJudge(samples))
	return len(samples), nil
}
