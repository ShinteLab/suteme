package suteme

import (
	"fmt"
	"strings"
)

type Board struct {
	Rows [9]BoardRow
}

type BoardPos struct {
	Row   int
	Col   int
	Value string
}

func (p BoardPos) IsError() bool {
	if p.Row < 0 || p.Col < 0 {
		return true
	}
	return false
}

func parsePosition(b string) BoardPos {
	var p BoardPos
	p.Row = -1
	p.Col = -1
	p.Value = b
	if len(b) != 2 {
		return p
	}

	num1 := b[0:1][0]
	num2 := b[1:2][0]
	// a = 97
	// 1 == 49
	row := 9 - (num2 - (97 - 1))
	col := num1 - 49
	if row < 0 || row > 9 ||
		col < 0 || col > 9 {
		return p
	}

	p.Row = int(row)
	p.Col = int(col)

	return p
}

func (bd Board) ToSFEN() string {
	var sb strings.Builder
	for idx, row := range bd.Rows {
		sb.WriteString(row.ToSFEN())
		if idx+1 != len(bd.Rows) {
			sb.WriteString("/")
		}
	}
	return sb.String()
}

func (bd *Board) SetPiece(r, c int, p Piece) {
	bd.Rows[r].Squares[c] = BoardSquare{Piece: p}
}

func (bd Board) GetPiece(p BoardPos) Piece {
	return bd.Rows[p.Row].Squares[p.Col].Piece
}

type BoardRow struct {
	Squares [9]BoardSquare
}

func (r BoardRow) ToSFEN() string {
	var b strings.Builder
	num := 0
	for idx, s := range r.Squares {

		m := s.Piece.Mark()
		if m == "" {
			num++
		}

		if m != "" || idx+1 == len(r.Squares) {
			if num != 0 {
				b.WriteString(fmt.Sprintf("%d", num))
				num = 0
			}

			if m != "" {
				b.WriteString(m)
			}
		}
	}
	return b.String()
}

type BoardSquare struct {
	Piece Piece
}
