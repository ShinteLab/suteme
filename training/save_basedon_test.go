package training

import (
	"encoding/json"
	"image/color"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// **入力済みの盤面を、画面がそれを見ないまま上書きできてはいけない。**
//
// 2026-09-13〜09-18 に実際に起きた事故の回帰テスト。API で受け付けた局面を
// 「再入力」で開いたとき、画面が SFEN を 81マスに展開しなくなっていたため、
// 何も入力せずに保存すると未入力の 81マス全部が推論候補で埋められ、
// **ikkyoku が入力したデータが認識器の読みで置き換わっていた**。
// しかも Verified が立つので、その結果が正解データとして学習に入る。
//
// 画面側（reenterHistory が必ず applyFromSFEN を通す）で直してあるが、
// 同じ形の事故は画面のどの経路からでも起こせるのでサーバ側で断る。
// 画面は「読み込んだ盤面」を based_on で名乗り、食い違えば 409。
func TestSaveSessionRejectsOverwriteWithoutBasedOn(t *testing.T) {
	chdirTemp(t)
	resetSettings(t)

	const sent = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w 2G3P 42"
	img := testImage(t, 100, 100, color.RGBA{200, 180, 120, 255})
	_, resp := doRegister(t, img, sent, &BoardBounds{5, 5, 95, 95})
	id, _ := resp["id"].(string)
	if id == "" {
		t.Fatal("登録できていない")
	}

	// 認識器の読み（＝入力とは違う盤面）で、名乗らずに上書きしようとする
	const recognized = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSN1"
	sess := newTestSession(t)

	post := func(basedOn string) *httptest.ResponseRecorder {
		body, _ := json.Marshal(map[string]string{
			"session": sess, "sfen": recognized, "history_id": id, "based_on": basedOn})
		rec := httptest.NewRecorder()
		handleSaveSession(rec, httptest.NewRequest(http.MethodPost, "/api/savesession",
			strings.NewReader(string(body))))
		return rec
	}

	if rec := post(""); rec.Code != http.StatusConflict {
		t.Errorf("based_on なしの上書き: status %d (want 409)\n%s", rec.Code, rec.Body.String())
	}
	if got := loadHistory().Entries[0].SFEN; got != sent {
		t.Fatalf("断ったのに入力が書き換わっている\n got = %q\nwant = %q", got, sent)
	}
	if loadHistory().Entries[0].IsVerified() {
		t.Error("断ったのに確認済みになっている（学習に入ってしまう）")
	}

	// 別の盤面を読み込んだと名乗る場合も断る（画面が古い）
	if rec := post("9/9/9/9/9/9/9/9/9"); rec.Code != http.StatusConflict {
		t.Errorf("食い違う based_on: status %d (want 409)\n%s", rec.Code, rec.Body.String())
	}

	// 読み込んだ盤面を正しく名乗れば、直した内容で上書きできる
	rec := post(strings.Fields(sent)[0])
	if rec.Code != http.StatusOK {
		t.Fatalf("正しい based_on で断られた: status %d\n%s", rec.Code, rec.Body.String())
	}
	e := loadHistory().Entries[0]
	if strings.Fields(e.SFEN)[0] != recognized {
		t.Errorf("盤面が更新されていない: %q", e.SFEN)
	}
	// 手番・持ち駒は引き継ぐ（画面は盤面部分しか作れない）
	if e.SFEN != recognized+" w 2G3P 42" {
		t.Errorf("手番・持ち駒が失われた: %q", e.SFEN)
	}
	if !e.IsVerified() {
		t.Error("保存し直したのに未確認のまま")
	}
}

// 盤面を持たない局面（座標だけ・SFEN 未保存）の上書きは名乗らなくても通る。
// 失う入力が無いので、ここで断ると新規の局面が保存できなくなる
func TestSaveSessionAllowsOverwriteWhenNoBoardYet(t *testing.T) {
	chdirTemp(t)
	sess := newTestSession(t)

	id, _ := postSave(t, sess, "9/9/9/9/9/9/9/9/9", "")["id"].(string)
	// 直後の保存し直し（画面は保存した盤面を名乗る）
	if got := postSave(t, sess, "lnsgkgsnl/9/9/9/9/9/9/9/9", id, "9/9/9/9/9/9/9/9/9")["overwrote"]; got != true {
		t.Errorf("overwrote = %v, want true", got)
	}

	// SFEN が空のエントリは名乗らなくても上書きできる
	historyMu.Lock()
	h := loadHistory()
	h.Entries[0].SFEN = ""
	saveHistoryFile(h)
	historyMu.Unlock()
	if got := postSave(t, sess, "9/9/9/9/9/9/9/9/9", id)["overwrote"]; got != true {
		t.Errorf("盤面の無いエントリの上書きが断られた: overwrote = %v", got)
	}
}

// checkBasedOn は盤面の合法性を見ない。**認識器は変な盤面こそ記録したい**ので、
// 二歩・玉の枚数・駒数超過で保存を止めてはいけない（駒落ち・詰将棋・
// 認識の誤りをそのまま残す場面が正当に存在する）
func TestCheckBasedOnIgnoresBoardLegality(t *testing.T) {
	// 玉が 3 枚ある盤面（実際に事故で記録されたもの）でも、
	// 名乗りが合っていれば通す
	const weird = "l5knl/1p3ggg1/5p1p1/p2s1sp1p/2P6/PPNSKPP1P/5PN2/1BG4L1/p1K5L"
	if err := checkBasedOn(weird, weird+" w N6Pn 1"); err != nil {
		t.Errorf("合法性で断っている: %v", err)
	}
	if err := checkBasedOn("", ""); err != nil {
		t.Errorf("盤面の無い相手で断っている: %v", err)
	}
}
