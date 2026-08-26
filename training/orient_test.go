package training

import (
	"image"
	"math"
	"os"
	"testing"

	"github.com/ShinteLab/suteme"
)

// TestMain は向き照合データの自動探索を止める。
//
// **カレントにファイルがあるかで結果が変わってはいけない。**
// データ用テストはリポジトリの根（`orient_data_v1.bin` が置かれる場所）へ
// 移動して走るので、止めておかないと環境で数字が動く。
// 評価は k-NN で測っており、k-NN は自分で `OrientationMatcher` を実装しているので
// この既定は評価の数字には影響しない。
func TestMain(m *testing.M) {
	suteme.SetOrientMatcher(nil)
	os.Exit(m.Run())
}

// TestBuildOrientDataThins は間引きの約束を確かめる。
//
//   - クラスごとに上限まで
//   - **空マスは入れない**（`PieceDistance` が読み飛ばすので太るだけ）
//   - **等間隔に抜く**（先頭から詰めると古い局面の盤しか残らない）
func TestBuildOrientDataThins(t *testing.T) {
	var samples []suteme.TrainingSample
	// 歩を 100 件（0..99 を先頭の値に入れて、どれが選ばれたか分かるようにする）
	for i := 0; i < 100; i++ {
		samples = append(samples, sample(0, float64(i)))
	}
	// 香は 3 件だけ
	for i := 0; i < 3; i++ {
		samples = append(samples, sample(1, float64(1000+i)))
	}
	// 空は 50 件
	for i := 0; i < 50; i++ {
		samples = append(samples, sample(suteme.ClassEmpty, float64(2000+i)))
	}

	got := BuildOrientData(samples, 10)

	counts := map[int]int{}
	for _, s := range got {
		counts[s.Label]++
	}
	if counts[0] != 10 {
		t.Errorf("歩は上限の 10 件のはずが %d 件", counts[0])
	}
	if counts[1] != 3 {
		t.Errorf("香は全 3 件残るはずが %d 件", counts[1])
	}
	if counts[suteme.ClassEmpty] != 0 {
		t.Errorf("空マスが %d 件入っている（入れない約束）", counts[suteme.ClassEmpty])
	}

	// 等間隔＝先頭 10 件に固まっていないこと
	var first float64 = -1
	var last float64
	for _, s := range got {
		if s.Label != 0 {
			continue
		}
		if first < 0 {
			first = s.Input[0]
		}
		last = s.Input[0]
	}
	if first != 0 {
		t.Errorf("先頭のサンプルが入っていない（first=%v）", first)
	}
	if last < 80 {
		t.Errorf("後ろのサンプルまで散っていない（last=%v）。先頭から詰めていないか", last)
	}

	// 0 件・上限 0
	if got := BuildOrientData(samples, 0); got != nil {
		t.Errorf("perClass=0 で %d 件返った", len(got))
	}
	if got := BuildOrientData(nil, 10); got != nil {
		t.Errorf("空の入力で %d 件返った", len(got))
	}
}

// TestOrientMatcherFillsInForNN は「照合できない推論器のときだけ」
// 向き照合データが使われることを確かめる。
//
// **k-NN のときに読みにいってはいけない**（自前で照合できるので、
// 間引いたデータで上書きすると精度が落ちる）。
func TestOrientMatcherFillsInForNN(t *testing.T) {
	var samples []suteme.TrainingSample
	for i := 0; i < 4; i++ {
		samples = append(samples, sample(i%2, float64(i)))
	}
	kn := suteme.NewKNN(samples)
	if _, ok := interface{}(kn).(suteme.OrientationMatcher); !ok {
		t.Fatal("k-NN が OrientationMatcher を実装していない")
	}
	// gobrain の Model は実装していない＝穴埋めが要る側
	var m *suteme.Model
	if _, ok := interface{}(m).(suteme.OrientationMatcher); ok {
		t.Error("Model が OrientationMatcher を実装している（前提が変わった）")
	}

	// **型付きの nil を渡しても落ちない。**
	// NewKNN は空なら nil を返すので、SetOrientMatcher に nil の *KNN が
	// 入りうる（インタフェース越しでは nil 判定にかからない）
	var empty *suteme.KNN
	var om suteme.OrientationMatcher = empty
	if om == nil {
		t.Fatal("型付き nil がインタフェースの nil になっている（前提が変わった）")
	}
	img := image.NewGray(image.Rect(0, 0, 24, 24))
	if d := om.PieceDistance(img); !math.IsInf(d, 1) {
		t.Errorf("nil の照合器が %v を返した（+Inf のはず）", d)
	}
}
