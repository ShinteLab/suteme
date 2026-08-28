package suteme

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"

	"github.com/goml/gobrain"
)

// 認識器を**ファイル以外から**渡すための口。
//
// **suteme はライブラリで、配布物はアプリのバイナリ。** 既定の探索
// （`LoadPredictor` / `defaultStripJudge` / `defaultOrientMatcher`）は
// カレントか実行ファイルの横を見るので、**そのままだと「モデルはファイルで
// 添える」形に固定される**。使う側が `go:embed` で自分のバイナリに
// 埋め込みたい（ikkyoku のような Wails3 アプリでは実行ファイル 1 つで
// 配りたい）ときに、置き場所の約束を持ち込まずに渡せるようにする。
//
//	//go:embed model/training_data_v8.bin
//	var modelData []byte
//
//	//go:embed model/strip_data_v1.bin
//	var stripData []byte
//
//	func init() {
//		p, err := suteme.PredictorFrom(bytes.NewReader(modelData), "embed:v8")
//		if err != nil { log.Fatal(err) }
//		suteme.SetPredictor(p)
//		if j, err := suteme.StripJudgeFrom(bytes.NewReader(stripData), "embed:v1"); err == nil {
//			suteme.SetStripJudge(j)
//		}
//	}
//
// `embed.FS` から渡す場合も同じで、`fsys.Open(name)` が返す `io.Reader` を
// そのまま渡せばよい（`fs.FS` 専用の口は用意しない）。
//
// **name は観測用のラベル**（`Debug` の "<- ここ" に出る。`Result.Debug`）。
// 埋め込むと元のファイル名が残らないので、**どの版を焼き込んだのかを
// 名乗らせる**ための引数。空文字でもよい。

// ReadTrainingData は学習データを io.Reader から読む。
// バイナリ（現行）と旧 JSON のどちらでも読める（`LoadTrainingData` と同じ）。
func ReadTrainingData(r io.Reader) (*TrainingData, error) {
	br := asPeeker(r)
	head, err := br.Peek(1)
	if err != nil {
		return nil, err
	}
	if head[0] == '{' {
		return readTrainingDataJSON(br)
	}
	return readTrainingData(br)
}

// ReadStripData は帯データ（1マス滑りの判定器）を io.Reader から読む。
func ReadStripData(r io.Reader) ([]StripSample, error) {
	return readStripData(asPeeker(r))
}

// ReadModel は gobrain の NN モデル（JSON）を io.Reader から読む。
func ReadModel(r io.Reader) (*Model, error) {
	var nn gobrain.FeedForward
	if err := json.NewDecoder(r).Decode(&nn); err != nil {
		return nil, err
	}
	return &Model{NN: &nn}, nil
}

// PredictorFrom は駒種推論器を io.Reader から作る。
//
// **中身を見て形式を決める**（学習データなら k-NN、JSON なら gobrain の NN）ので、
// 埋め込む側はどちらを焼き込んでも同じ呼び出しで済む。
// **旧 JSON 形式の学習データはここでは読めない**（NN モデルと区別が付かないため）。
// v8 では旧形式を読まない方針なので実害は無い（`LegacyDataFiles` は空）。
func PredictorFrom(r io.Reader, name string) (Predictor, error) {
	br := asPeeker(r)
	head, err := br.Peek(1)
	if err != nil {
		return nil, fmt.Errorf("認識器のデータが読めません: %w", err)
	}
	if head[0] == '{' {
		m, err := ReadModel(br)
		if err != nil {
			return nil, err
		}
		m.source = name
		return m, nil
	}
	data, err := readTrainingData(br)
	if err != nil {
		return nil, err
	}
	kn := NewKNN(data.Samples)
	if kn == nil {
		return nil, fmt.Errorf("学習データが空です")
	}
	kn.source = name
	return kn, nil
}

// StripJudgeFrom は帯の判定器を io.Reader から作る（`SetStripJudge` に渡す）。
func StripJudgeFrom(r io.Reader, name string) (*StripJudge, error) {
	samples, err := ReadStripData(r)
	if err != nil {
		return nil, err
	}
	j := NewStripJudge(samples)
	if j == nil {
		return nil, fmt.Errorf("帯データが空です")
	}
	j.path = name
	return j, nil
}

// OrientMatcherFrom は向きの回転照合器を io.Reader から作る
// （`SetOrientMatcher` に渡す）。**要るのは NN を推論器にする構成だけ**で、
// k-NN は自分で照合できる（orient.go）。
func OrientMatcherFrom(r io.Reader, name string) (OrientationMatcher, error) {
	data, err := ReadTrainingData(r)
	if err != nil {
		return nil, err
	}
	kn := NewKNN(data.Samples)
	if kn == nil {
		return nil, fmt.Errorf("向き照合データが空です")
	}
	kn.source = name
	return kn, nil
}

// asPeeker は Peek できる Reader にする。
// 既に bufio.Reader ならそのまま使う（二重にバッファしない）。
func asPeeker(r io.Reader) *bufio.Reader {
	if br, ok := r.(*bufio.Reader); ok {
		return br
	}
	return bufio.NewReaderSize(r, 1<<20)
}
