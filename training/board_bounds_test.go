package training

import (
	"encoding/json"
	"image"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/ShinteLab/suteme"
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

// 盤面座標を**誰が決めたか**を履歴に残す。
//
// 座標があることと人が引いたことは別で、自動検出をそのまま保存した局面の
// 座標は検出器自身の出力（突き合わせても自己充足になる）。持っていなかった頃は
// 再入力しても一律「保存済みの座標」としか出せなかった。
func TestSaveSessionRecordsBoundsBy(t *testing.T) {
	chdirTemp(t)

	for _, tc := range []struct {
		name string
		by   string
		want string
	}{
		{"人が引いた", BoundsByManual, BoundsByManual},
		{"自動検出のまま", BoundsByDetect, BoundsByDetect},
		{"名乗らない値は不明に倒す", "drag", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sess := newTestSession(t)
			// 画像は 4x4 なので、盤面として成立する大きさの矩形を別に用意する
			mu.Lock()
			sessions[sess].Original = image.NewRGBA(image.Rect(0, 0, 200, 200))
			sessions[sess].Result = &suteme.AnalyzeResult{}
			mu.Unlock()

			body, _ := json.Marshal(map[string]interface{}{
				"session": sess, "x1": 10, "y1": 10, "x2": 190, "y2": 190, "by": tc.by,
			})
			rec := httptest.NewRecorder()
			handleSetBoard(rec, httptest.NewRequest(http.MethodPost, "/api/setboard", strings.NewReader(string(body))))
			if rec.Code != http.StatusOK {
				t.Fatalf("setboard: status %d (%s)", rec.Code, rec.Body.String())
			}

			id, _ := postSave(t, sess, "9/9/9/9/9/9/9/9/9", "")["id"].(string)
			var h HistoryData
			b, err := os.ReadFile(historyFile)
			if err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(b, &h); err != nil {
				t.Fatal(err)
			}
			for _, e := range h.Entries {
				if e.ID != id {
					continue
				}
				if e.BoundsBy != tc.want {
					t.Errorf("bounds_by = %q, want %q", e.BoundsBy, tc.want)
				}
				if e.BoardBounds == nil {
					t.Error("座標が保存されていない")
				}
				return
			}
			t.Fatalf("保存した局面 %s が履歴に無い", id)
		})
	}
}

// 潰れた矩形は 400 で断る（黙って受けると (0,0)-(0,0) の盤面ができ、
// 格子も 81 マスの画像も消えるのに画面にはエラーが出ない）
func TestSetBoardRejectsTinyRect(t *testing.T) {
	chdirTemp(t)
	sess := newTestSession(t)
	body, _ := json.Marshal(map[string]interface{}{"session": sess, "x1": 0, "y1": 0, "x2": 0, "y2": 0})
	rec := httptest.NewRecorder()
	handleSetBoard(rec, httptest.NewRequest(http.MethodPost, "/api/setboard", strings.NewReader(string(body))))
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", rec.Code)
	}
}
