// suteme-training はラベリング・学習用Webサーバの起動コマンド。
// サーバ本体の実装は suteme/training パッケージにある。
//
//	suteme-training [-port 8080] [データディレクトリ]
//
// データディレクトリ（data/・training_data_v8.bin など）を省略すると
// カレントディレクトリを使う。training パッケージのパスはすべて
// カレントディレクトリからの相対なので、起動時にそこへ移る。
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
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(), "使い方: %s [-port 8080] [データディレクトリ]\n", filepath.Base(os.Args[0]))
		flag.PrintDefaults()
	}
	flag.Parse()
	if flag.NArg() > 1 {
		flag.Usage()
		os.Exit(2)
	}
	if dir := flag.Arg(0); dir != "" {
		if err := enterDataDir(dir); err != nil {
			log.Fatal(err)
		}
	}
	wd, _ := os.Getwd()
	log.Printf("データディレクトリ: %s", wd)
	log.Fatal(training.Serve(*port))
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
