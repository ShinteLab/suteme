package main

import (
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"

	"suteme"
)

func main() {
	files := []string{
		"./_cmd/samples/game.png",
		"./_cmd/samples/20240508102053.png",
		"./_cmd/samples/rf48936080_o.png",
	}

	f := files[0] // game.png
	img, err := load(f)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: %v\n", f, err)
		os.Exit(1)
	}

	r := suteme.Analyze(img)
	if r.Board == nil {
		fmt.Println("Board not found")
		os.Exit(1)
	}

	cats := suteme.ClassifyBoard(img, r.Board)
	for row := 0; row < 9; row++ {
		for col := 0; col < 9; col++ {
			fmt.Print(cats[row][col].String())
		}
		fmt.Println()
	}
	conf := suteme.ValidateBoard(img, r.Board)
	fmt.Printf("Confidence: %.0f%%\n", conf*100)
}

func load(f string) (image.Image, error) {
	fp, err := os.Open(f)
	if err != nil {
		return nil, err
	}
	defer fp.Close()

	img, _, err := image.Decode(fp)
	if err != nil {
		return nil, err
	}
	return img, nil
}
