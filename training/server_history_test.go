package training

import (
	"encoding/json"
	"image"
	"image/color"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// 履歴の保存（新規/上書き）と削除。
//
// 「保存 → 直して保存」で局面が二重に登録されていたのを上書きに変えた回帰テスト。
// dataDir / historyFile が相対パスなので、実データを壊さないよう
// 一時ディレクトリへ移ってから叩く。
func chdirTemp(t *testing.T) {
	t.Helper()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(t.TempDir()); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.Chdir(wd) })
}

// newTestSession は保存に必要な最小限のセッション（元画像だけ）を登録する
func newTestSession(t *testing.T) string {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 4, 4))
	img.Set(0, 0, color.RGBA{1, 2, 3, 255})
	id := newID()
	mu.Lock()
	sessions[id] = &session{Original: img}
	mu.Unlock()
	t.Cleanup(func() {
		mu.Lock()
		delete(sessions, id)
		mu.Unlock()
	})
	return id
}

func postSave(t *testing.T, sess, sfen, historyID string) map[string]interface{} {
	t.Helper()
	body, _ := json.Marshal(map[string]string{"session": sess, "sfen": sfen, "history_id": historyID})
	rec := httptest.NewRecorder()
	handleSaveSession(rec, httptest.NewRequest(http.MethodPost, "/api/savesession", strings.NewReader(string(body))))
	if rec.Code != http.StatusOK {
		t.Fatalf("savesession: status %d (%s)", rec.Code, rec.Body.String())
	}
	var resp map[string]interface{}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	return resp
}

func TestSaveSessionOverwritesHistory(t *testing.T) {
	chdirTemp(t)
	sess := newTestSession(t)

	first := postSave(t, sess, "9/9/9/9/9/9/9/9/9", "")
	id, _ := first["id"].(string)
	if id == "" {
		t.Fatal("id が返っていない")
	}
	if first["overwrote"] != false {
		t.Errorf("新規保存なのに overwrote=%v", first["overwrote"])
	}

	// 直して保存し直す: 同じ ID を上書きし、履歴は増えない
	second := postSave(t, sess, "lnsgkgsnl/9/9/9/9/9/9/9/9", id)
	if second["id"] != id {
		t.Errorf("上書きなのに別 ID になった: %v (want %s)", second["id"], id)
	}
	if second["overwrote"] != true {
		t.Errorf("overwrote=%v (want true)", second["overwrote"])
	}

	h := loadHistory()
	if len(h.Entries) != 1 {
		t.Fatalf("履歴が %d 件（1 件のはず）", len(h.Entries))
	}
	if h.Entries[0].SFEN != "lnsgkgsnl/9/9/9/9/9/9/9/9" {
		t.Errorf("SFEN が更新されていない: %q", h.Entries[0].SFEN)
	}
	if _, err := os.Stat(filepath.Join(dataDir, id+".png")); err != nil {
		t.Errorf("上書き後に画像が消えている: %v", err)
	}

	// 履歴に無い ID を指定されたら新規保存に倒す（削除済みの局面を直した場合）
	third := postSave(t, sess, "9/9/9/9/9/9/9/9/9", "deadbeefdeadbeef")
	if third["overwrote"] != false {
		t.Errorf("存在しない履歴 ID なのに overwrote=%v", third["overwrote"])
	}
	if len(loadHistory().Entries) != 2 {
		t.Errorf("履歴が %d 件（2 件のはず）", len(loadHistory().Entries))
	}
}

func TestHistoryDelete(t *testing.T) {
	chdirTemp(t)
	sess := newTestSession(t)
	id, _ := postSave(t, sess, "9/9/9/9/9/9/9/9/9", "")["id"].(string)

	rec := httptest.NewRecorder()
	handleHistoryItem(rec, httptest.NewRequest(http.MethodDelete, "/api/history/"+id, nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("delete: status %d (%s)", rec.Code, rec.Body.String())
	}
	if n := len(loadHistory().Entries); n != 0 {
		t.Errorf("削除後の履歴が %d 件", n)
	}
	if _, err := os.Stat(filepath.Join(dataDir, id+".png")); !os.IsNotExist(err) {
		t.Errorf("画像が残っている: %v", err)
	}

	// 二重削除・不正な ID は 404
	rec = httptest.NewRecorder()
	handleHistoryItem(rec, httptest.NewRequest(http.MethodDelete, "/api/history/"+id, nil))
	if rec.Code != http.StatusNotFound {
		t.Errorf("存在しない履歴の削除: status %d (want 404)", rec.Code)
	}
	rec = httptest.NewRecorder()
	handleHistoryItem(rec, httptest.NewRequest(http.MethodDelete, "/api/history/../../etc", nil))
	if rec.Code != http.StatusNotFound {
		t.Errorf("不正な ID: status %d (want 404)", rec.Code)
	}
}
