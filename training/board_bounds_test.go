package training

import (
	"encoding/json"
	"strings"
	"testing"
)

// 履歴の盤面座標は**画面が読むキー**（小文字）で出す。
// 大文字で出していた頃は、履歴からの再入力で `bounds.x1` が undefined になり、
// JSON.stringify が null に落として (0,0)-(0,0) の盤面が設定されていた。
// 格子も 81 マスの画像も消えるのに、画面にはエラーが出ないという壊れ方をする。
func TestBoardBoundsUsesLowerCaseJSON(t *testing.T) {
	b, err := json.Marshal(BoardBounds{X1: 296, Y1: 171, X2: 678, Y2: 589})
	if err != nil {
		t.Fatal(err)
	}
	got := string(b)
	for _, k := range []string{`"x1":296`, `"y1":171`, `"x2":678`, `"y2":589`} {
		if !strings.Contains(got, k) {
			t.Errorf("%s が無い: %s", k, got)
		}
	}
}

// 既存の history.json は大文字キーで書かれている。読めなくなると
// 保存済みの座標が丸ごと失われる（学習価値が 1/5 になる）
func TestBoardBoundsReadsLegacyUpperCaseJSON(t *testing.T) {
	var bb BoardBounds
	if err := json.Unmarshal([]byte(`{"X1":296,"Y1":171,"X2":678,"Y2":589}`), &bb); err != nil {
		t.Fatal(err)
	}
	if bb != (BoardBounds{X1: 296, Y1: 171, X2: 678, Y2: 589}) {
		t.Errorf("大文字キーの履歴を読めていない: %+v", bb)
	}
}
