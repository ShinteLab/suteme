package training

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"sync"
)

// 履歴一覧のサムネイル。
//
// **一覧の 56px の枠に元画像をそのまま流し込まないこと。** 保存画像は中継の
// スクリーンショットそのもので、実測（155 局面）で **1 枚平均 895KB・合計 139MB**、
// 最大 2.9MB ある。`<img>` の表示サイズは CSS が決めるだけなので、履歴タブを
// 開くたびにブラウザは 139MB を受け取って 155 枚を実寸でデコードしていた
// （ローカルでも転送だけで 5.5 秒。これにデコードが乗る）。
//
// **縮小した PNG をディスクに持つ。** 元画像から作り直すのは 1 枚あたり
// デコード + 縮小ぶんかかるので、`data/thumb/{id}.png` に置いて 2 回目以降は
// ファイルをそのまま返す。元画像は上書き保存されうるので**更新時刻で確かめ、
// 古ければ作り直す**（`sigCache` が署名を覚えるのと同じ考え方）。
const (
	// thumbSize はサムネイルの長辺（px）。CSS の表示は 56px なので、
	// HiDPI で粗く見えないよう 2 倍を持つ
	thumbSize = 112
)

func thumbDir() string           { return filepath.Join(dataDir, "thumb") }
func thumbPath(id string) string { return filepath.Join(thumbDir(), id+".png") }

// thumbMu はサムネイルの生成を直列化する。履歴タブを開くとブラウザが
// 6 本並列で取りに来るので、同じ画像を何本も同時にデコードさせない
var thumbMu sync.Mutex

// historyThumb は履歴画像のサムネイルのパスを返す（無ければ作る）。
// 作れなければ空文字を返す（呼び出し側は元画像に落とす）。
func historyThumb(id string) string {
	src := filepath.Join(dataDir, id+".png")
	ss, err := os.Stat(src)
	if err != nil {
		return ""
	}
	dst := thumbPath(id)
	if ds, err := os.Stat(dst); err == nil && !ds.ModTime().Before(ss.ModTime()) {
		return dst
	}

	thumbMu.Lock()
	defer thumbMu.Unlock()
	// ロック待ちの間に別のリクエストが作っているかもしれない
	if ds, err := os.Stat(dst); err == nil && !ds.ModTime().Before(ss.ModTime()) {
		return dst
	}
	if err := makeThumb(src, dst); err != nil {
		return ""
	}
	return dst
}

func makeThumb(src, dst string) error {
	f, err := os.Open(src)
	if err != nil {
		return err
	}
	img, _, err := image.Decode(f)
	f.Close()
	if err != nil {
		return err
	}

	var buf bytes.Buffer
	if err := png.Encode(&buf, shrink(img, thumbSize)); err != nil {
		return err
	}
	if err := os.MkdirAll(thumbDir(), 0755); err != nil {
		return err
	}
	// 一時ファイル + rename。読みながら書き換わって半端な PNG を返さないため
	tmp := dst + ".tmp"
	if err := os.WriteFile(tmp, buf.Bytes(), 0644); err != nil {
		return err
	}
	return os.Rename(tmp, dst)
}

// removeThumb は履歴を消したときにサムネイルも片付ける。
// 残しても次の ID と衝突はしないが、`data/` にゴミが溜まる
func removeThumb(id string) { os.Remove(thumbPath(id)) }

// shrink は長辺が max px になるよう面積平均で縮小する。
// **最近傍にしないこと。** 1/20 まで落とすので、点をつまむ形だと駒だけが
// 消えたり格子線がモアレになったりして、見た目の判定に使えない絵になる
// （`resizeGray` を面積平均にしてあるのと同じ理由）。
func shrink(src image.Image, max int) image.Image {
	b := src.Bounds()
	sw, sh := b.Dx(), b.Dy()
	if sw <= 0 || sh <= 0 {
		return image.NewRGBA(image.Rect(0, 0, 1, 1))
	}
	w, h := sw, sh
	if sw > max || sh > max {
		if sw >= sh {
			w, h = max, sh*max/sw
		} else {
			w, h = sw*max/sh, max
		}
	}
	if w < 1 {
		w = 1
	}
	if h < 1 {
		h = 1
	}

	dst := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		y0 := b.Min.Y + y*sh/h
		y1 := b.Min.Y + (y+1)*sh/h
		if y1 <= y0 {
			y1 = y0 + 1
		}
		for x := 0; x < w; x++ {
			x0 := b.Min.X + x*sw/w
			x1 := b.Min.X + (x+1)*sw/w
			if x1 <= x0 {
				x1 = x0 + 1
			}
			var sr, sg, sb uint32
			n := uint32(0)
			for sy := y0; sy < y1; sy++ {
				for sx := x0; sx < x1; sx++ {
					r, g, bb, _ := src.At(sx, sy).RGBA()
					sr += r >> 8
					sg += g >> 8
					sb += bb >> 8
					n++
				}
			}
			if n == 0 {
				continue
			}
			dst.SetRGBA(x, y, color.RGBA{uint8(sr / n), uint8(sg / n), uint8(sb / n), 255})
		}
	}
	return dst
}
