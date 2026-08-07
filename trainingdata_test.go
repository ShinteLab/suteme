package suteme

import (
	"encoding/json"
	"image"
	"image/color"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

func sampleData(n int) *TrainingData {
	rng := rand.New(rand.NewSource(7))
	d := &TrainingData{Samples: make([]TrainingSample, n)}
	for i := range d.Samples {
		in := make([]float64, InputSize)
		for j := range in {
			in[j] = quantize(rng.NormFloat64() * 3)
		}
		d.Samples[i] = TrainingSample{Input: in, Label: i % NumClasses}
	}
	return d
}

// v4 のバイナリを書いて読み戻すと同じ内容になること
func TestTrainingDataRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "training_data_v4.bin")
	want := sampleData(5)
	if err := SaveTrainingData(path, want); err != nil {
		t.Fatal(err)
	}

	got, err := LoadTrainingData(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Samples) != len(want.Samples) {
		t.Fatalf("samples = %d, want %d", len(got.Samples), len(want.Samples))
	}
	for i := range want.Samples {
		if got.Samples[i].Label != want.Samples[i].Label {
			t.Fatalf("[%d] label = %d, want %d", i, got.Samples[i].Label, want.Samples[i].Label)
		}
		if !equalFloats(got.Samples[i].Input, want.Samples[i].Input) {
			t.Fatalf("[%d] input が一致しない", i)
		}
	}

	// ヘッダ 16 バイト + (ラベル4 + 576*4) * 件数
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if wantSize := int64(16 + (4+InputSize*4)*5); fi.Size() != wantSize {
		t.Errorf("size = %d, want %d", fi.Size(), wantSize)
	}
}

// **これが本命の回帰テスト。**
// 学習データは float32 で保存されるので、CellToInput が素の float64 を返すと
// 「ファイルから読んだサンプル」と「同じマスから作り直したサンプル」が
// 一致しなくなり、MergeSamples の重複除去が効かなくなる（＝学習を押すたびに
// 同じサンプルが積み上がる）。CellToInput 側でも float32 に丸めて防ぐ。
func TestCellToInputSurvivesSaveLoad(t *testing.T) {
	cell := image.NewRGBA(image.Rect(0, 0, 40, 44))
	rng := rand.New(rand.NewSource(3))
	for y := 0; y < 44; y++ {
		for x := 0; x < 40; x++ {
			v := uint8(rng.Intn(256))
			cell.Set(x, y, color.RGBA{v, v, v, 255})
		}
	}
	want := CellToInput(cell)

	path := filepath.Join(t.TempDir(), "training_data_v4.bin")
	if err := SaveTrainingData(path, &TrainingData{
		Samples: []TrainingSample{{Input: want, Label: 3}},
	}); err != nil {
		t.Fatal(err)
	}
	got, err := LoadTrainingData(path)
	if err != nil {
		t.Fatal(err)
	}
	if !equalFloats(got.Samples[0].Input, want) {
		t.Error("保存して読み直した入力が CellToInput の出力と一致しない" +
			"（MergeSamples の重複除去が効かなくなる）")
	}
}

// v3 までの JSON も読めること（186MB の既存ファイルを捨てずに移行するため）。
// 読んだ値は v4 と同じく float32 に丸まる
func TestLoadTrainingDataAcceptsLegacyJSON(t *testing.T) {
	path := filepath.Join(t.TempDir(), "training_data_v3.json")
	want := sampleData(3)
	// v3 の書き出しと同じ形（生の JSON）
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	// float32 では表せない値を入れて、丸めが効いていることも見る
	want.Samples[0].Input[0] = 0.1234567890123456
	if err := json.NewEncoder(f).Encode(want); err != nil {
		t.Fatal(err)
	}
	f.Close()

	got, err := LoadTrainingData(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Samples) != 3 {
		t.Fatalf("samples = %d, want 3", len(got.Samples))
	}
	if got.Samples[0].Input[0] != quantize(0.1234567890123456) {
		t.Errorf("JSON から読んだ値が float32 に丸まっていない: %v", got.Samples[0].Input[0])
	}
	if got.Samples[2].Label != want.Samples[2].Label {
		t.Errorf("label = %d, want %d", got.Samples[2].Label, want.Samples[2].Label)
	}
}

// 学習データは手でラベル付けした資産なので、壊れたファイルを黙って
// 空データとして読まない（新規扱いになって上書きで消える）
func TestLoadTrainingDataRejectsGarbage(t *testing.T) {
	path := filepath.Join(t.TempDir(), "broken.bin")
	if err := os.WriteFile(path, []byte("not a training data file at all"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadTrainingData(path); err == nil {
		t.Error("壊れたファイルがエラーにならない")
	}
}

func equalFloats(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
