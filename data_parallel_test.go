package suteme

import (
	"encoding/json"
	"image"
	_ "image/png"
	"os"
	"runtime"
	"sync"
	"testing"
)

// 保存済み局面（`data/`）を使うテストの土台。
//
// **局面が増えるほど費用が線形に伸びる。並列にして畳む。**
// 188 局面の時点でルートパッケージは 443 秒かかっていて、その 96% が
// `TestDetectBoardCropInvariance`(246s) / `TestDetectBoardMatchesManual`(139s) /
// `TestValidateBoardDiscriminates`(41s) の 3 本だった。
//
// **局面をサンプルして減らす道は採らない。** これらは合否を判定する
// 回帰テストで、AGENTS.md の数字（「0.5マス以内 152/157」など）の出所でもある。
// 間引くと、落ちなくなった代わりに**何件で測ったのかが環境ごとに変わる**。
// 一方で 1 局面の処理は互いに独立なので、**並列にすれば
// 中身を 1 つも削らずに壁時計だけ縮む**。
//
// **順序は保つ。** 結果は入力順に並べ直してから t.Log / t.Error に渡すので、
// ログの並びも合否も逐次版と同じになる（並列にした証拠がテストの出力に
// 出ないのが正しい）。

// loadSavedBoards は data/history.json を読む。無ければテストをスキップする。
func loadSavedBoards(t *testing.T) (string, []historyEntry) {
	t.Helper()
	dir := findDataDir()
	if dir == "" {
		t.Skip("data/history.json が無いのでスキップ")
	}
	f, err := os.Open(dir + "/history.json")
	if err != nil {
		t.Skip(err)
	}
	defer f.Close()
	var h historyFile
	if err := json.NewDecoder(f).Decode(&h); err != nil {
		t.Fatalf("history.json: %v", err)
	}
	return dir, h.Entries
}

// mapSavedBoards は各局面の画像を読んで fn に渡し、**入力順**の結果を返す。
// 画像が開けない・壊れている局面は結果に入らない（逐次版の `continue` と同じ）。
//
// fn は**複数のゴルーチンから同時に呼ばれる**ので、t.Log / t.Error を
// 呼んではいけない（並びが崩れるうえ、どの局面のログか分からなくなる）。
// 報告は戻り値に積んで、呼び出し側が順に処理する。
func mapSavedBoards[T any](dir string, entries []historyEntry, fn func(historyEntry, image.Image) T) []T {
	type slot struct {
		val T
		ok  bool
	}
	slots := make([]slot, len(entries))
	workers := runtime.GOMAXPROCS(0)
	if workers > len(entries) {
		workers = len(entries)
	}
	if workers < 1 {
		workers = 1
	}
	var next int
	var mu sync.Mutex
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				mu.Lock()
				i := next
				next++
				mu.Unlock()
				if i >= len(entries) {
					return
				}
				e := entries[i]
				f, err := os.Open(dir + "/" + e.ID + ".png")
				if err != nil {
					continue
				}
				img, _, err := image.Decode(f)
				f.Close()
				if err != nil {
					continue
				}
				slots[i] = slot{val: fn(e, img), ok: true}
			}
		}()
	}
	wg.Wait()
	out := make([]T, 0, len(entries))
	for _, s := range slots {
		if s.ok {
			out = append(out, s.val)
		}
	}
	return out
}
