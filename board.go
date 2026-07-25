package suteme

import (
	"shinte/core/sfen"
	"shinte/core/usi"
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

// parsePosition は USI マス文字列を配列添字(Row=段0..8, Col=筋0..8)に変換する。
// 座標変換の仕様は core/usi に集約している。内部座標(x,y in 1..9)との対応は
// Row = 9-y, Col = 9-x。
func parsePosition(b string) BoardPos {
	x, y, ok := usi.ParseSquare(b)
	if !ok {
		return BoardPos{Row: -1, Col: -1, Value: b}
	}
	return BoardPos{Row: 9 - y, Col: 9 - x, Value: b}
}

// ToSFEN は盤面を SFEN 盤面文字列に変換する。空マスの圧縮・段区切りは
// core/sfen に委譲する。
func (bd Board) ToSFEN() string {
	return sfen.FormatBoard(func(rank, file int) string {
		return bd.Rows[rank].Squares[file].Piece.Mark()
	})
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

type BoardSquare struct {
	Piece Piece
}
