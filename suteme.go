package suteme

import (
	"fmt"
	"image"
)

func LoadSFEN(img image.Image) (string, error) {
	return "", fmt.Errorf("not implemented")
}

func ViewDebug(img image.Image) error {
	r := Analyze(img)
	out := r.DrawBoard(img)

	SaveImage("debug_board.png", out)
	SaveImage("debug_edges.png", r.Edges)

	if r.Board != nil {
		fmt.Printf("Board: %v\n", r.Board.Bounds)
	} else {
		fmt.Println("Board: not found")
	}
	fmt.Println("Saved: debug_board.png, debug_edges.png")
	return nil
}
