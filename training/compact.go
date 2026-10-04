package training

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/ShinteLab/suteme"
)

// 配布用の認識器を書き出す。
//
// **学習データ全件（114MB）は配れない。** GitHub は 100MB がハード上限で、
// 受け取る側の推論も 1 局面 1.2 秒かかる。しかも中身は 24x24 の輝度そのもので、
// **並べ直せば盤の絵になる**（AGENTS.md「認識器を配る」）。
//
// クラスごとに間引くと、同じサイズ帯では NN より強い。実測（30局面・
// 手動座標・**その局面を学習済みの条件**なので絶対値は高く出る）:
//
//	model_v8.json + orient_data_v1.bin   8.7MB   90.8%
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
// 名前は受け取る側でそのまま使える正式名（`training_data_v8.bin`）にするので、
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

// ExportCompact は配布用の認識器を `dist/` に書き出す（画面の「配布用に書き出す」・`POST /api/export`）。
func ExportCompact(perClass int) ([]ExportedFile, error) {
	return ExportCompactTo(ExportOptions{PerClass: perClass})
}

// ExportOptions は配布用の書き出しの行き先と形（2026-10-04）。
type ExportOptions struct {
	// Dir は書き出し先。空なら `dist/`。**データディレクトリそのものは断る**
	// （正式名で書くので、手元の学習データ全件を上書きしてしまう）。
	Dir string
	// PerClass は 1クラスあたりに残す件数。
	PerClass int
	// Gzip は `.gz` を付けて圧縮して書く。**焼き込む側（ikkyoku）向け**で、
	// exe に入れる大きさが半分以下になる（実測 24.8MB → 10.6MB）。
	// ⚠️ **`LoadPredictor` は .gz を読まない**（そのまま置いて使う形ではなくなる）。
	Gzip bool
}

// ExportInfoFile は書き出しの記録（ExportInfo）の名前。書き出し先に一緒に置く。
const ExportInfoFile = "export.json"

// ExportInfo は書き出しの記録。**受け取る側がいつのどの書き出しかを知るため**のもので、
// 認識には使わない（ikkyoku は焼き込んだものと置いたものの新旧を Date で比べる）。
type ExportInfo struct {
	Date        time.Time `json:"date"`
	PerClass    int       `json:"per_class"`
	Samples     int       `json:"samples"`
	From        string    `json:"from"`
	FromSamples int       `json:"from_samples"`
	Gzip        bool      `json:"gzip"`
	Files       []string  `json:"files"`
}

// ExportCompactTo は配布用の認識器を o.Dir に書き出す。
//
// **盤の縁の帯（`strip_data_v1.bin`）も一緒に置く。** 無いと盤面検出が
// 157/157 → 152/157 になる（1マス滑りが直らない）ので、
// 学習データだけ渡すと**受け取った側だけ検出が別物**になる。
// 中身は変えずにそのまま複製する（間引く意味が無いほど小さい）。
//
// ⚠️ **書き出し先にある前回の同じ名前のもの（圧縮した版・しない版の両方）は消してから書く。**
// 片方だけ残ると、受け取る側が古いほうを読む。**知らない名前のファイルには触らない**
// （書き出し先は任意のディレクトリなので）。
func ExportCompactTo(o ExportOptions) ([]ExportedFile, error) {
	if o.PerClass <= 0 {
		return nil, fmt.Errorf("1クラスあたりの件数は 1 以上にしてください")
	}
	dir := o.Dir
	if dir == "" {
		dir = distDir
	}
	if same, err := sameDir(dir, "."); err == nil && same {
		return nil, fmt.Errorf("書き出し先がデータディレクトリと同じです（手元の学習データを上書きしてしまいます）: %s", dir)
	}
	data, path := loadExistingTrainingData()
	if data == nil || len(data.Samples) == 0 {
		return nil, fmt.Errorf("学習データがありません")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}

	compact := BuildCompactData(data.Samples, o.PerClass)
	if len(compact) == 0 {
		return nil, fmt.Errorf("間引いた結果が空になりました")
	}
	for _, name := range []string{suteme.DefaultDataFile, stripFile, ExportInfoFile} {
		os.Remove(filepath.Join(dir, name))
		os.Remove(filepath.Join(dir, name+".gz"))
	}

	// **受け取る側がそのまま使える名前で書く**（`LoadPredictor` は
	// ファイル名を決め打ちで探す）。`dist/` に置くのは手元の全件を守るため
	name := suteme.DefaultDataFile
	if o.Gzip {
		name += ".gz"
	}
	dst := filepath.Join(dir, name)
	if err := saveTrainingDataAs(dst, &suteme.TrainingData{Samples: compact}, o.Gzip); err != nil {
		return nil, err
	}
	out := []ExportedFile{{
		Name: name, Path: dst, Bytes: fileSize(dst), Samples: len(compact),
		Note: fmt.Sprintf("駒種の認識（k-NN）。%s の %d 件から 1クラス %d 件まで間引いたもの",
			path, len(data.Samples), o.PerClass),
	}}

	// 帯はそのまま複製。無ければ黙って飛ばす（判定器が無くても検出は動く）
	stripName := stripFile
	if o.Gzip {
		stripName += ".gz"
	}
	stripDst := filepath.Join(dir, stripName)
	if n, err := copyFileAs(stripFile, stripDst, o.Gzip); err == nil {
		out = append(out, ExportedFile{
			Name: stripName, Path: stripDst, Bytes: n,
			Note: "盤面検出の1マス滑りの補正。無いと 157/157 → 152/157 になる",
		})
	} else if !os.IsNotExist(err) {
		return nil, err
	}

	// 書き出しの記録は**最後に**書く（揃っていない途中の状態を「書き出し済み」と読ませない）。
	info := ExportInfo{
		Date: time.Now(), PerClass: o.PerClass, Samples: len(compact),
		From: path, FromSamples: len(data.Samples), Gzip: o.Gzip,
	}
	for _, f := range out {
		info.Files = append(info.Files, f.Name)
	}
	b, err := json.MarshalIndent(info, "", "  ")
	if err != nil {
		return nil, err
	}
	infoPath := filepath.Join(dir, ExportInfoFile)
	if err := os.WriteFile(infoPath, b, 0o644); err != nil {
		return nil, err
	}
	out = append(out, ExportedFile{Name: ExportInfoFile, Path: infoPath, Bytes: int64(len(b)),
		Note: "書き出しの記録（日時・件数）。受け取る側が新旧を比べるのに使う"})

	logger().Info("Exported distribution set",
		"dir", dir, "samples", len(compact), "perClass", o.PerClass, "gzip", o.Gzip)
	return out, nil
}

// sameDir は a と b が同じディレクトリを指すか（無いディレクトリは違うとみなす）。
func sameDir(a, b string) (bool, error) {
	sa, err := os.Stat(a)
	if err != nil {
		return false, err
	}
	sb, err := os.Stat(b)
	if err != nil {
		return false, err
	}
	return os.SameFile(sa, sb), nil
}

// saveTrainingDataAs は学習データを書く。gz なら圧縮して書く（一時ファイル + rename）。
func saveTrainingDataAs(dst string, data *suteme.TrainingData, gz bool) error {
	if !gz {
		return suteme.SaveTrainingData(dst, data)
	}
	raw := dst + ".raw"
	if err := suteme.SaveTrainingData(raw, data); err != nil {
		return err
	}
	defer os.Remove(raw)
	_, err := copyFileAs(raw, dst, true)
	return err
}

// copyFileAs は src を dst へ複製する（gz なら圧縮して）。書いた大きさを返す。
func copyFileAs(src, dst string, gz bool) (int64, error) {
	if !gz {
		return copyFile(src, dst)
	}
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
	zw, err := gzip.NewWriterLevel(tmp, gzip.BestCompression)
	if err != nil {
		tmp.Close()
		return 0, err
	}
	if _, err := io.Copy(zw, in); err != nil {
		tmp.Close()
		return 0, err
	}
	if err := zw.Close(); err != nil {
		tmp.Close()
		return 0, err
	}
	if err := tmp.Close(); err != nil {
		return 0, err
	}
	if err := os.Rename(tmpName, dst); err != nil {
		return 0, err
	}
	return fileSize(dst), nil
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
