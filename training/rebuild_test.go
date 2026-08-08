package training

import (
	"os"
	"testing"

	"github.com/ShinteLab/suteme"
)

// TestRebuildTrainingData は保存済み局面（data/）から学習データを作り直す。
//
// **入力ベクトルの作り方（`suteme.CellToInput`）を変えたときに使う。**
// 版を上げると古いファイルのベクトルは使えなくなるが、学習データは
// 履歴の画像と正解 SFEN から全部作り直せる。UI の履歴タブで全件を選んで
// 学習するのと同じことを、コマンドラインからやるためのもの。
//
//	go test -run TestRebuildTrainingData -timeout 30m -v ./training
//
// 誤って走らせるとファイルを上書きするので、環境変数を付けたときだけ動く。
func TestRebuildTrainingData(t *testing.T) {
	if os.Getenv("SUTEME_REBUILD") == "" {
		t.Skip("SUTEME_REBUILD が空なのでスキップ（学習データを上書きする）")
	}
	if !chdirToData(t) {
		t.Fatal("data/history.json が見つかりません")
	}
	h := loadHistory()
	if h == nil || len(h.Entries) == 0 {
		t.Fatal("履歴が空です")
	}

	var all []suteme.TrainingSample
	for _, e := range h.Entries {
		s, err := samplesFromHistory(e)
		if err != nil {
			t.Logf("%s: スキップ (%v)", e.ID, err)
			continue
		}
		all = append(all, s...)
	}
	all = MergeSamples(all)
	if len(all) == 0 {
		t.Fatal("サンプルが作れませんでした")
	}

	data := suteme.TrainingData{Samples: all}
	if err := suteme.SaveTrainingData(dataFile, &data); err != nil {
		t.Fatalf("保存に失敗: %v", err)
	}
	t.Logf("%s に %d サンプルを書き出しました（%d 局面）",
		dataFile, len(all), len(h.Entries))
}
