package training

import (
	"bytes"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
)

// 負例（盤面が写っていない画像）。
//
// 守りたいのは次の 3 つ。
//   - ラベルは置き場所そのもの（data/negative/ に置くだけで負例になる）
//   - 同じ画像を送り直しても増えない
//   - 盤として採用されたら「誤検出」として数える。棄却できたものも信頼度を残す

func negativeRequest(t *testing.T, raw []byte) *http.Request {
	t.Helper()
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	fw, err := mw.CreateFormFile("image", "shot.png")
	if err != nil {
		t.Fatal(err)
	}
	fw.Write(raw)
	mw.Close()
	r := httptest.NewRequest(http.MethodPost, "/api/negative", &body)
	r.Header.Set("Content-Type", mw.FormDataContentType())
	return r
}

func postNegative(t *testing.T, raw []byte) (*httptest.ResponseRecorder, map[string]interface{}) {
	t.Helper()
	w := httptest.NewRecorder()
	handleNegatives(w, negativeRequest(t, raw))
	var resp map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &resp)
	return w, resp
}

// 登録・重複・削除。**同じ画像で増えない**のは、気になった画面を何度も
// 送っても負例が水増しされないようにするため（リトライを安全にする）
func TestNegativeRegisterDedupDelete(t *testing.T) {
	chdirTemp(t)

	raw := testImage(t, 60, 40, color.RGBA{20, 20, 30, 255})
	w, resp := postNegative(t, raw)
	if w.Code != http.StatusOK {
		t.Fatalf("登録に失敗した: %d %s", w.Code, w.Body.String())
	}
	id, _ := resp["id"].(string)
	if !validID(id) {
		t.Fatalf("ID が不正: %q", id)
	}
	if _, err := os.Stat(negativePath(id)); err != nil {
		t.Fatalf("画像が保存されていない: %v", err)
	}

	// 同じ画像は増やさず、既存の ID を返す
	_, resp2 := postNegative(t, raw)
	if resp2["duplicate"] != true || resp2["id"] != id {
		t.Errorf("同じ画像で増えている: %v", resp2)
	}
	if ids := negativeIDs(); len(ids) != 1 {
		t.Errorf("負例が %d 枚（1 枚のはず）", len(ids))
	}

	// 削除
	w = httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodDelete, "/api/negative/"+id, nil)
	handleNegativeItem(w, r)
	if w.Code != http.StatusOK {
		t.Fatalf("削除に失敗した: %d", w.Code)
	}
	if ids := negativeIDs(); len(ids) != 0 {
		t.Errorf("削除後に %d 枚残っている", len(ids))
	}
}

// 負例が 1 枚も無ければ nil。記録に空の欄を作らない
func TestEvaluateNegativesEmpty(t *testing.T) {
	chdirTemp(t)
	if ev := evaluateNegatives(negativeStub{}); ev != nil {
		t.Errorf("負例が無いのに %+v を返した", ev)
	}
}

// negativeStub は駒種を返すだけの推論器。負例の判定に要るのは
// 「盤として採用されるか」だけなので、駒種は何でもよい
type negativeStub struct{}

func (negativeStub) Predict(cell image.Image) (int, float64) { return 0, 1.0 }

// 判定できなかった画像を棄却として数えない。**数えると誤検出率が
// 実態より良く見える**（推論器を入れ忘れた実行が満点になる）
func TestEvaluateNegativesSkipsUnjudgeable(t *testing.T) {
	chdirTemp(t)
	writeNegativeFile(t, gridPNG(t)) // 盤として採用される絵

	if ev := evaluateNegatives(nil); ev != nil {
		t.Errorf("推論器なしで判定できたことにしている: %+v", ev)
	}
}

// 一様な画像は盤面ではないので棄却される（＝負例として正しく扱われる）。
// **格子線の無い一様な板は detectBoardRegion が nil を返す**（AGENTS.md）
func TestEvaluateNegativesRejectsFlatImage(t *testing.T) {
	chdirTemp(t)
	writeNegativeFile(t, testImage(t, 200, 200, color.RGBA{180, 150, 100, 255}))

	ev := evaluateNegatives(negativeStub{})
	if ev == nil || ev.Images != 1 {
		t.Fatalf("負例が数えられていない: %+v", ev)
	}
	if ev.Rejected != 1 || ev.Results[0].Detected {
		t.Errorf("一様な画像を盤面として採用した: %+v", ev.Results[0])
	}
}

// 盤に見える格子は誤検出として数える。**「棄却できた」だけを数えていると
// 数字が動かないので、採用された側が記録に残ることを確かめる**
func TestEvaluateNegativesCountsDetected(t *testing.T) {
	chdirTemp(t)
	writeNegativeFile(t, gridPNG(t))

	ev := evaluateNegatives(negativeStub{})
	if ev == nil || len(ev.Results) != 1 {
		t.Fatalf("負例が数えられていない: %+v", ev)
	}
	got := ev.Results[0]
	if !got.Detected {
		t.Skipf("この合成格子は検出されなかった（conf %.2f）。誤検出の計上は実画像で確認すること", got.Confidence)
	}
	if ev.Rejected != 0 {
		t.Errorf("採用されたのに棄却として数えている: %+v", ev)
	}
	if got.X2 <= got.X1 || got.Confidence <= 0 {
		t.Errorf("盤とみなした範囲・信頼度が残っていない: %+v", got)
	}
}

func writeNegativeFile(t *testing.T, raw []byte) {
	t.Helper()
	if err := os.MkdirAll(negativeDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(negativePath(newID()), raw, 0644); err != nil {
		t.Fatal(err)
	}
}

// gridPNG は 9x9 の格子だけを描いた画像。盤の絵ではないが、
// 検出は「等間隔の線が 10 本」を探すので候補になりうる
func gridPNG(t *testing.T) []byte {
	t.Helper()
	const cw, ch = 70, 76
	img := image.NewRGBA(image.Rect(0, 0, cw*9+80, ch*9+80))
	for y := img.Bounds().Min.Y; y < img.Bounds().Max.Y; y++ {
		for x := img.Bounds().Min.X; x < img.Bounds().Max.X; x++ {
			img.Set(x, y, color.RGBA{200, 175, 120, 255})
		}
	}
	line := color.RGBA{40, 30, 20, 255}
	for i := 0; i <= 9; i++ {
		for y := 40; y <= 40+ch*9; y++ {
			img.Set(40+i*cw, y, line)
		}
		for x := 40; x <= 40+cw*9; x++ {
			img.Set(x, 40+i*ch, line)
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}
