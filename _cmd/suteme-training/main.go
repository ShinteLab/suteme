// suteme-training はラベリング・学習用Webサーバの起動コマンド。
// サーバ本体の実装は suteme/training パッケージにある。
//
//	suteme-training [-port 8080] [データディレクトリ]
//	suteme-training -export [-out dist] [-gzip] [-per-class 1000] [データディレクトリ]
//
// データディレクトリ（data/・training_data_v8.bin など）を省略すると
// カレントディレクトリを使う。training パッケージのパスはすべて
// カレントディレクトリからの相対なので、起動時にそこへ移る。
//
// -export はサーバを起動せず、**配布用の書き出し**（履歴タブ「配布用に書き出す」・
// `POST /api/export` と同じ `training.ExportCompact`）だけをして終わる。
// 認識器を焼き込んで配る側（ikkyoku の `task model:copy`）が、画面を開かずに
// `dist/` を作るための口（2026-10-04）。-out で書き出し先を選べる（ikkyoku なら
// `_cmd/ikkyoku/model` を指し、-gzip で exe に入れる大きさを抑える）。
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"

	"github.com/ShinteLab/suteme/training"
)

func main() {
	port := flag.String("port", "8080", "待ち受けるポート")
	export := flag.Bool("export", false, "サーバを起動せず、配布用の認識器を dist/ に書き出して終わる")
	perClass := flag.Int("per-class", training.CompactPerClass, "-export で 1クラスあたりに残す件数")
	out := flag.String("out", "", "-export の書き出し先（既定はデータディレクトリの下の dist）")
	gz := flag.Bool("gzip", false, "-export で圧縮して書く（.gz。焼き込む側向け）")
	flag.Usage = func() {
		name := filepath.Base(os.Args[0])
		fmt.Fprintf(flag.CommandLine.Output(), "使い方: %s [-port 8080] [データディレクトリ]\n", name)
		fmt.Fprintf(flag.CommandLine.Output(), "        %s -export [-out dist] [-gzip] [-per-class %d] [データディレクトリ]\n", name, training.CompactPerClass)
		flag.PrintDefaults()
	}
	flag.Parse()
	if flag.NArg() > 1 {
		flag.Usage()
		os.Exit(2)
	}
	// -out は**起動した場所からの相対**で受ける（下でデータディレクトリへ移るので、先に絶対パスにする）。
	if *out != "" {
		abs, err := filepath.Abs(*out)
		if err != nil {
			log.Fatal(err)
		}
		*out = abs
	}
	if dir := flag.Arg(0); dir != "" {
		if err := enterDataDir(dir); err != nil {
			log.Fatal(err)
		}
	}
	wd, _ := os.Getwd()
	log.Printf("データディレクトリ: %s", wd)
	if *export {
		if err := exportCompact(training.ExportOptions{Dir: *out, PerClass: *perClass, Gzip: *gz}); err != nil {
			log.Fatal(err)
		}
		return
	}
	log.Fatal(training.Serve(*port))
}

// exportCompact は配布用の書き出しをして、書いたファイルを並べる。
// **中身は画面の「配布用に書き出す」と同じ**（`training.ExportCompact`）。
// 出力先は -out（既定はデータディレクトリの下の `dist/`。手元の全件を上書きしないため。compact.go）。
func exportCompact(o training.ExportOptions) error {
	files, err := training.ExportCompactTo(o)
	if err != nil {
		return fmt.Errorf("配布用に書き出せませんでした: %w", err)
	}
	for _, f := range files {
		fmt.Printf("%s  %.1f MB", f.Path, float64(f.Bytes)/(1<<20))
		if f.Samples > 0 {
			fmt.Printf("  %d サンプル", f.Samples)
		}
		fmt.Printf("\n    %s\n", f.Note)
	}
	return nil
}

// enterDataDir はデータディレクトリへ移る。**無いディレクトリは作らない。**
// 作ると空の data/ で起動して「履歴が消えた」ように見えるため
func enterDataDir(dir string) error {
	st, err := os.Stat(dir)
	if err != nil {
		// 以前は位置引数がポートだった。古い使い方を黙って別の意味に取らない
		if _, nerr := strconv.Atoi(dir); nerr == nil {
			return fmt.Errorf("データディレクトリ %q がありません（ポートは -port %s で指定します）", dir, dir)
		}
		return fmt.Errorf("データディレクトリ %q がありません: %w", dir, err)
	}
	if !st.IsDir() {
		return fmt.Errorf("%q はディレクトリではありません", dir)
	}
	return os.Chdir(dir)
}
