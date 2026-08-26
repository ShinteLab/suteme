package suteme

import (
	"bytes"
	"image"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// makeSamples はテスト用の学習サンプルを作る
func makeSamples(n int) []TrainingSample {
	out := make([]TrainingSample, 0, n)
	for i := 0; i < n; i++ {
		in := make([]float64, InputSize)
		for j := range in {
			in[j] = float64((i*7+j)%251) / 251
		}
		out = append(out, TrainingSample{Input: in, Label: i % 3})
	}
	return out
}

// bytesOf はデータをファイルに書いてから読み戻す（配布物と同じ中身を得る）
func bytesOf(t *testing.T, write func(path string) error) []byte {
	t.Helper()
	p := filepath.Join(t.TempDir(), "data.bin")
	if err := write(p); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// TestPredictorFromBytes は**ファイルを置かずに**推論器を渡せることを確かめる。
//
// suteme はライブラリなので、使う側は `go:embed` で自分のバイナリに
// 焼き込みたい。ここが無いと「モデルはファイルで添える」形に固定される。
func TestPredictorFromBytes(t *testing.T) {
	samples := makeSamples(30)
	raw := bytesOf(t, func(p string) error {
		return SaveTrainingData(p, &TrainingData{Samples: samples})
	})

	p, err := PredictorFrom(bytes.NewReader(raw), "embed:test")
	if err != nil {
		t.Fatal(err)
	}
	kn, ok := p.(*KNN)
	if !ok {
		t.Fatalf("k-NN のはずが %T", p)
	}
	if kn.Len() != len(samples) {
		t.Errorf("%d 件（%d 件のはず）", kn.Len(), len(samples))
	}
	// **どの版を焼き込んだのかを名乗れること**（Debug の観測用）
	if d := p.(DebugPredictor).Debug(); d.Source != "embed:test" || d.Kind != "knn" {
		t.Errorf("Debug が %+v", d)
	}
	// 回転照合もできる（k-NN なので向き照合データは要らない）
	if _, ok := p.(OrientationMatcher); !ok {
		t.Error("埋め込んだ k-NN が OrientationMatcher を実装していない")
	}
}

// TestPredictorFromDetectsNN は同じ入口で NN も読めることを確かめる
// （埋め込む側がどちらを焼いても呼び出しが変わらない）。
func TestPredictorFromDetectsNN(t *testing.T) {
	// gobrain の JSON はここでは作れないので、形だけの JSON で分岐を見る
	raw := []byte(`{"NHiddens":1}`)
	p, err := PredictorFrom(bytes.NewReader(raw), "embed:nn")
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := p.(*Model); !ok {
		t.Fatalf("NN のはずが %T", p)
	}
	if d := p.(DebugPredictor).Debug(); d.Kind != "nn" || d.Source != "embed:nn" {
		t.Errorf("Debug が %+v", d)
	}
}

// TestStripJudgeFromBytes は帯の判定器も埋め込みから渡せることを確かめる。
// **帯が無いと盤面検出が別物になる**ので、こちらも渡せないと意味が無い。
func TestStripJudgeFromBytes(t *testing.T) {
	strip := image.NewGray(image.Rect(0, 0, 90, 10))
	for i := range strip.Pix {
		strip.Pix[i] = uint8(i % 251)
	}
	in := StripInput(strip, strip.Bounds(), false)
	if in == nil {
		t.Fatal("帯の入力が作れない")
	}
	raw := bytesOf(t, func(p string) error {
		return SaveStripData(p, []StripSample{{Input: in, Board: true}, {Input: in, Board: false}})
	})

	j, err := StripJudgeFrom(bytes.NewReader(raw), "embed:strip")
	if err != nil {
		t.Fatal(err)
	}
	if j.Samples() != 2 {
		t.Errorf("%d 本（2 本のはず）", j.Samples())
	}
	if j.Path() != "embed:strip" {
		t.Errorf("名乗りが %q", j.Path())
	}
}

// TestOrientMatcherFromBytes は向き照合器も埋め込みから渡せることを確かめる
func TestOrientMatcherFromBytes(t *testing.T) {
	raw := bytesOf(t, func(p string) error {
		return SaveTrainingData(p, &TrainingData{Samples: makeSamples(10)})
	})
	om, err := OrientMatcherFrom(bytes.NewReader(raw), "embed:orient")
	if err != nil {
		t.Fatal(err)
	}
	cell := image.NewGray(image.Rect(0, 0, 24, 24))
	if d := om.PieceDistance(cell); d < 0 {
		t.Errorf("距離が負: %v", d)
	}
}

// TestReadRejectsGarbage は壊れた入力を黙って受けないことを確かめる。
// **空データとして通すと「認識器はあるのに何も当たらない」になる**ので、
// ファイル版と同じくエラーにする。
func TestReadRejectsGarbage(t *testing.T) {
	for _, c := range []struct {
		name string
		fn   func() error
	}{
		{"学習データ", func() error { _, err := ReadTrainingData(bytes.NewReader([]byte("garbage!!"))); return err }},
		{"帯データ", func() error { _, err := ReadStripData(bytes.NewReader([]byte("garbage!!"))); return err }},
		{"推論器", func() error { _, err := PredictorFrom(bytes.NewReader([]byte("garbage!!")), ""); return err }},
		{"空の入力", func() error { _, err := PredictorFrom(bytes.NewReader(nil), ""); return err }},
	} {
		if err := c.fn(); err == nil {
			t.Errorf("%s: 壊れた入力を受け入れた", c.name)
		}
	}
	// 帯データを推論器として渡した場合（取り違え）も落とす
	strip := image.NewGray(image.Rect(0, 0, 90, 10))
	in := StripInput(strip, strip.Bounds(), false)
	raw := bytesOf(t, func(p string) error {
		return SaveStripData(p, []StripSample{{Input: in, Board: true}})
	})
	if _, err := PredictorFrom(bytes.NewReader(raw), ""); err == nil {
		t.Error("帯データを推論器として受け入れた")
	} else if !strings.Contains(err.Error(), "学習データ") {
		t.Logf("（参考）取り違えのエラー文: %v", err)
	}
}

// TestLoadPredictorMatchesEmbedded はファイル経由と埋め込み経由で
// **同じものが得られる**ことを確かめる（配布形態で答えが変わらないこと）。
func TestLoadPredictorMatchesEmbedded(t *testing.T) {
	dir := t.TempDir()
	samples := makeSamples(20)
	if err := SaveTrainingData(filepath.Join(dir, DefaultDataFile), &TrainingData{Samples: samples}); err != nil {
		t.Fatal(err)
	}
	fromFile, err := LoadPredictor(dir)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(filepath.Join(dir, DefaultDataFile))
	if err != nil {
		t.Fatal(err)
	}
	fromBytes, err := PredictorFrom(bytes.NewReader(raw), "embed")
	if err != nil {
		t.Fatal(err)
	}

	a, b := fromFile.(*KNN), fromBytes.(*KNN)
	if a.Len() != b.Len() {
		t.Fatalf("件数が違う: %d vs %d", a.Len(), b.Len())
	}
	cell := image.NewGray(image.Rect(0, 0, 24, 24))
	for i := range cell.Pix {
		cell.Pix[i] = uint8(i % 251)
	}
	ca, fa := a.Predict(cell)
	cb, fb := b.Predict(cell)
	if ca != cb || fa != fb {
		t.Errorf("推論が違う: (%d,%v) vs (%d,%v)", ca, fa, cb, fb)
	}
}
