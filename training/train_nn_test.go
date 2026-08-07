package training

import (
	"math/rand"
	"os"
	"testing"

	"github.com/ShinteLab/suteme"
)

// testSamples はダミーの学習サンプルを作る（中身は問わない。件数だけが要る）
func testSamples(n int) []suteme.TrainingSample {
	rng := rand.New(rand.NewSource(1))
	samples := make([]suteme.TrainingSample, n)
	for i := range samples {
		in := make([]float64, suteme.InputSize)
		for j := range in {
			in[j] = rng.NormFloat64()
		}
		samples[i] = suteme.TrainingSample{Input: in, Label: i % suteme.NumClasses}
	}
	return samples
}

// gobrain NN の学習は「選んだ局面数」ではなく**累積した全サンプル数**に比例し、
// 300 エポック固定なので 1 局選んだだけで数分〜数十分かかる。
// 主認識器の k-NN は生データから即再構築できるので、既定では NN を学習しない。
func TestTrainAndSaveSkipsNNByDefault(t *testing.T) {
	chdirTemp(t)
	modelLock.Lock()
	knn, model, nnStale = nil, nil, false
	modelLock.Unlock()

	res, err := trainAndSave(testSamples(4), false)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := os.Stat(modelFile); !os.IsNotExist(err) {
		t.Errorf("既定で %s が作られている（NN を学習してしまっている）", modelFile)
	}
	if _, err := os.Stat(dataFile); err != nil {
		t.Errorf("学習データが保存されていない: %v", err)
	}
	if res["nn"] != false {
		t.Errorf("nn = %v, want false", res["nn"])
	}
	// 精度に効くのは k-NN。こちらは必ず組み直されていること
	modelLock.RLock()
	built := knn != nil
	modelLock.RUnlock()
	if !built {
		t.Error("k-NN が再構築されていない")
	}
	if res["balanced"] != nil {
		t.Errorf("NN を回していないのに balanced が入っている: %v", res["balanced"])
	}
	// 分布は生データから出す（NN を回さなくても中身は見たい）
	if dist, ok := res["distribution"].(map[string]int); !ok || len(dist) == 0 {
		t.Errorf("distribution = %v", res["distribution"])
	}
}

// 明示的に指定したときは従来どおり NN も学習する（比較用モデルを作る道は残す）
func TestTrainAndSaveTrainsNNWhenRequested(t *testing.T) {
	chdirTemp(t)
	modelLock.Lock()
	knn, model, nnStale = nil, nil, true
	modelLock.Unlock()

	res, err := trainAndSave(testSamples(4), true)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := os.Stat(modelFile); err != nil {
		t.Errorf("%s が作られていない: %v", modelFile, err)
	}
	if res["nn"] != true {
		t.Errorf("nn = %v, want true", res["nn"])
	}
	modelLock.RLock()
	stale := nnStale
	trained := model != nil
	modelLock.RUnlock()
	if !trained {
		t.Error("モデルが差し替わっていない")
	}
	if stale {
		t.Error("学習したのに nnStale が立ったまま")
	}
}
