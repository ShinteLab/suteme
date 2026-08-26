package training

import (
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"

	"github.com/ShinteLab/suteme"
)

// 配布用の認識器を書き出す。
//
// **学習データ全件（114MB）は配れない。** GitHub は 100MB がハード上限で、
// 受け取る側の推論も 1 局面 1.2 秒かかる。しかも中身は 24x24 の輝度そのもので、
// **並べ直せば盤の絵になる**（CLAUDE.md「認識器を配る」）。
//
// クラスごとに間引くと、同じサイズ帯では NN より強い。実測（30局面・
// 手動座標・**その局面を学習済みの条件**なので絶対値は高く出る）:
//
//	model_v7.json + orient_data_v1.bin   8.7MB   90.8%
//	間引き  250/class ( 2793件)          5.7MB   93.8%
//	間引き 1000/class ( 8828件)           19MB   97.5%
//	全件      (51932件)                  114MB   98.8%
//
// **向き照合データ（`orient_data_v1.bin`）は要らない。** 書き出すのは
// 学習データと同じ形式で、読む側は k-NN として構築する＝自分で照合できる。

// CompactPerClass は配布用に残すクラスごとのサンプル数の既定値。
// 上の表の 1000（19MB・97.5%）を採る。**空マスも残す**
// （k-NN は `ClassEmpty` との一致で空を確定するので、抜くと空判定が分類器頼みになる）。
const CompactPerClass = 1000

// distDir は書き出し先。**カレントに直接書かない。**
// 名前は受け取る側でそのまま使える正式名（`training_data_v7.bin`）にするので、
// 同じディレクトリに置くと**手元の学習データ全件を上書きしてしまう**。
const distDir = "dist"

// BuildCompactData は配布用に学習データを間引く（空マスも残す）。
func BuildCompactData(samples []suteme.TrainingSample, perClass int) []suteme.TrainingSample {
	return thinByClass(samples, perClass, true)
}

// ExportedFile は書き出したファイル 1 つ
type ExportedFile struct {
	Name    string `json:"name"`
	Path    string `json:"path"`
	Bytes   int64  `json:"bytes"`
	Samples int    `json:"samples,omitempty"`
	Note    string `json:"note"`
}

// ExportCompact は配布用の認識器を `dist/` に書き出す。
//
// **盤の縁の帯（`strip_data_v1.bin`）も一緒に置く。** 無いと盤面検出が
// 157/157 → 152/157 になる（1マス滑りが直らない）ので、
// 学習データだけ渡すと**受け取った側だけ検出が別物**になる。
// 中身は変えずにそのまま複製する（間引く意味が無いほど小さい）。
func ExportCompact(perClass int) ([]ExportedFile, error) {
	if perClass <= 0 {
		return nil, fmt.Errorf("1クラスあたりの件数は 1 以上にしてください")
	}
	data, path := loadExistingTrainingData()
	if data == nil || len(data.Samples) == 0 {
		return nil, fmt.Errorf("学習データがありません")
	}
	if err := os.MkdirAll(distDir, 0o755); err != nil {
		return nil, err
	}

	compact := BuildCompactData(data.Samples, perClass)
	if len(compact) == 0 {
		return nil, fmt.Errorf("間引いた結果が空になりました")
	}
	// **受け取る側がそのまま使える名前で書く**（`LoadPredictor` は
	// ファイル名を決め打ちで探す）。`dist/` に置くのは手元の全件を守るため
	dst := filepath.Join(distDir, suteme.DefaultDataFile)
	if err := suteme.SaveTrainingData(dst, &suteme.TrainingData{Samples: compact}); err != nil {
		return nil, err
	}
	out := []ExportedFile{{
		Name: suteme.DefaultDataFile, Path: dst, Bytes: fileSize(dst), Samples: len(compact),
		Note: fmt.Sprintf("駒種の認識（k-NN）。%s の %d 件から 1クラス %d 件まで間引いたもの",
			path, len(data.Samples), perClass),
	}}

	// 帯はそのまま複製。無ければ黙って飛ばす（判定器が無くても検出は動く）
	if n, err := copyFile(stripFile, filepath.Join(distDir, stripFile)); err == nil {
		out = append(out, ExportedFile{
			Name: stripFile, Path: filepath.Join(distDir, stripFile), Bytes: n,
			Note: "盤面検出の1マス滑りの補正。無いと 157/157 → 152/157 になる",
		})
	} else if !os.IsNotExist(err) {
		return nil, err
	}

	log.Printf("Exported distribution set to %s/ (%d samples, <= %d per class)",
		distDir, len(compact), perClass)
	return out, nil
}

func fileSize(p string) int64 {
	fi, err := os.Stat(p)
	if err != nil {
		return 0
	}
	return fi.Size()
}

// copyFile は src を dst へそのまま複製し、書いた大きさを返す。
// **一時ファイル + rename** にしてあるのは、途中で落ちたときに
// 中途半端なファイルを配ってしまわないため（学習データの保存と同じ理由）。
func copyFile(src, dst string) (int64, error) {
	in, err := os.Open(src)
	if err != nil {
		return 0, err
	}
	defer in.Close()

	tmp, err := os.CreateTemp(filepath.Dir(dst), filepath.Base(dst)+".tmp*")
	if err != nil {
		return 0, err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)

	n, err := io.Copy(tmp, in)
	if err != nil {
		tmp.Close()
		return 0, err
	}
	if err := tmp.Close(); err != nil {
		return 0, err
	}
	if err := os.Rename(tmpName, dst); err != nil {
		return 0, err
	}
	return n, nil
}
