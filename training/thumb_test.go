package training

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// 履歴一覧のサムネイル。守りたいのは次の 3 つ。
//   - 一覧に流すのは縮小した PNG（元画像をそのまま返さない）
//   - 上書き保存したら作り直す（古い絵を返し続けない）
//   - 変わっていなければ 304（`no-cache` を素通しにしない）

func writeHistoryImage(t *testing.T, id string, w, h int, c color.RGBA) {
	t.Helper()
	if err := os.MkdirAll(dataDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dataDir, id+".png"), testImage(t, w, h, c), 0644); err != nil {
		t.Fatal(err)
	}
}

func getHistoryItem(path string, header http.Header) *httptest.ResponseRecorder {
	r := httptest.NewRequest(http.MethodGet, path, nil)
	for k, v := range header {
		r.Header[k] = v
	}
	w := httptest.NewRecorder()
	handleHistoryItem(w, r)
	return w
}

// **サムネイルは縮小されたものが返る。** 元画像をそのまま返していた頃は、
// 56px の枠に 155 枚 139MB を流し込んでいた
func TestHistoryThumbIsSmall(t *testing.T) {
	chdirTemp(t)
	const id = "00112233445566aa"
	writeHistoryImage(t, id, 1200, 800, color.RGBA{40, 90, 140, 255})

	w := getHistoryItem("/api/history/"+id+"/thumb", nil)
	if w.Code != http.StatusOK {
		t.Fatalf("サムネイルが取れない: %d", w.Code)
	}
	cfg, err := png.DecodeConfig(bytes.NewReader(w.Body.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Width != thumbSize || cfg.Height != thumbSize*800/1200 {
		t.Fatalf("縮小されていない: %dx%d", cfg.Width, cfg.Height)
	}

	full := getHistoryItem("/api/history/"+id+"/image", nil)
	if w.Body.Len() >= full.Body.Len() {
		t.Fatalf("サムネイル %d B が元画像 %d B より小さくない", w.Body.Len(), full.Body.Len())
	}
	if _, err := os.Stat(thumbPath(id)); err != nil {
		t.Fatalf("サムネイルが data/thumb/ に残っていない: %v", err)
	}
}

// **元画像が差し替わったら作り直す。** 履歴は上書き保存できるので、
// 一度作ったサムネイルを返し続けると別の局面の絵が出る
func TestHistoryThumbFollowsSource(t *testing.T) {
	chdirTemp(t)
	const id = "00112233445566bb"
	writeHistoryImage(t, id, 200, 200, color.RGBA{0, 0, 0, 255})
	first := getHistoryItem("/api/history/"+id+"/thumb", nil).Body.Bytes()

	// 更新時刻の粒度に負けないよう、はっきり進めてから書き換える
	writeHistoryImage(t, id, 200, 200, color.RGBA{255, 255, 255, 255})
	now := time.Now().Add(2 * time.Second)
	os.Chtimes(filepath.Join(dataDir, id+".png"), now, now)

	second := getHistoryItem("/api/history/"+id+"/thumb", nil).Body.Bytes()
	if bytes.Equal(first, second) {
		t.Fatal("上書きした元画像が反映されていない")
	}
}

// **ETag を付けること。** `no-cache` は検証子が無いと毎回まるごと取り直しになり、
// 一覧を開くたびに全枚数を再転送・再デコードすることになる
func TestHistoryImageRevalidates(t *testing.T) {
	chdirTemp(t)
	const id = "00112233445566cc"
	writeHistoryImage(t, id, 300, 200, color.RGBA{10, 20, 30, 255})

	for _, kind := range []string{"image", "thumb"} {
		w := getHistoryItem("/api/history/"+id+"/"+kind, nil)
		etag := w.Header().Get("ETag")
		if etag == "" {
			t.Fatalf("%s に ETag が無い", kind)
		}
		h := http.Header{"If-None-Match": []string{etag}}
		again := getHistoryItem("/api/history/"+id+"/"+kind, h)
		if again.Code != http.StatusNotModified {
			t.Fatalf("%s が 304 を返さない: %d", kind, again.Code)
		}
		if again.Body.Len() != 0 {
			t.Fatalf("%s が 304 なのに本文を返した: %d B", kind, again.Body.Len())
		}
	}
}

// 履歴を消したらサムネイルも消える（`data/` にゴミを残さない）
func TestHistoryDeleteRemovesThumb(t *testing.T) {
	chdirTemp(t)
	const id = "00112233445566dd"
	writeHistoryImage(t, id, 100, 100, color.RGBA{1, 2, 3, 255})
	saveHistoryFile(&HistoryData{Entries: []HistoryEntry{{ID: id, SFEN: "9/9/9/9/9/9/9/9/9"}}})
	getHistoryItem("/api/history/"+id+"/thumb", nil)

	w := httptest.NewRecorder()
	handleHistoryItem(w, httptest.NewRequest(http.MethodDelete, "/api/history/"+id, nil))
	if w.Code != http.StatusOK {
		t.Fatalf("削除できない: %d %s", w.Code, w.Body.String())
	}
	if _, err := os.Stat(thumbPath(id)); !os.IsNotExist(err) {
		t.Fatalf("サムネイルが残っている: %v", err)
	}
}

// 元画像が無ければ 404（サムネイルの生成に落ちて 200 を返さない）
func TestHistoryThumbMissing(t *testing.T) {
	chdirTemp(t)
	if w := getHistoryItem("/api/history/00112233445566ee/thumb", nil); w.Code != http.StatusNotFound {
		t.Fatalf("404 にならない: %d", w.Code)
	}
}

// 縮小は面積平均。**最近傍にすると点をつまむ形になり、駒が消えたり
// 格子線がモアレになったりして見た目の判定に使えない絵になる**
func TestShrinkAverages(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 4, 4))
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			c := color.RGBA{0, 0, 0, 255}
			if (x+y)%2 == 0 {
				c = color.RGBA{200, 200, 200, 255}
			}
			src.Set(x, y, c)
		}
	}
	got := shrink(src, 2)
	if b := got.Bounds(); b.Dx() != 2 || b.Dy() != 2 {
		t.Fatalf("大きさが違う: %v", b)
	}
	r, _, _, _ := got.At(0, 0).RGBA()
	if v := r >> 8; v < 90 || v > 110 {
		t.Fatalf("市松模様の平均が 100 付近にならない: %d", v)
	}
}

// 元画像より小さい枠は要求しない（拡大しない）
func TestShrinkKeepsSmallImages(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 30, 20))
	if b := shrink(src, thumbSize).Bounds(); b.Dx() != 30 || b.Dy() != 20 {
		t.Fatalf("拡大された: %v", b)
	}
}
