// suteme-training はラベリング・学習用Webサーバの起動コマンド。
// サーバ本体の実装は suteme/training パッケージにある。
package main

import (
	"log"
	"os"

	"github.com/ShinteLab/suteme/training"
)

func main() {
	port := "8080"
	if len(os.Args) > 1 {
		port = os.Args[1]
	}
	log.Fatal(training.Serve(port))
}
