package training

import (
	"fmt"
	"image"
	"image/png"
	"os"
	"path/filepath"

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
	b := e.BoardBounds
	s := stripSamplesFromRegion(img, b.X1, b.Y1, b.X2, b.Y2)
	if len(s) == 0 {
		return nil, fmt.Errorf("帯を作れませんでした")
	}
	return s, nil
}

// BuildStripData は確認済みの全局面から帯の教師データを作り直す。
//
// **毎回ゼロから作る（累積しない）。** 駒の学習データと違って
// 1 局面あたり 8 本しか無く、全件でも 1000 本強なので作り直しても数秒。
// 累積すると、局面を消したときに古い帯が残り続ける。
func BuildStripData(entries []HistoryEntry) []suteme.StripSample {
	var out []suteme.StripSample
	for _, e := range entries {
		if !e.IsVerified() {
			continue
		}
		s, err := stripSamplesFromHistory(e)
		if err != nil {
			continue
		}
		out = append(out, s...)
	}
	return out
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
