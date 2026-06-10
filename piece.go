package suteme

import "strings"

type Piece struct {
	typ  PieceType
	plus bool
	turn bool
}

func parsePiece(tag string) Piece {
	var p Piece
	p.plus = false
	p.turn = false

	val := tag
	if tag[0:1] == "+" {
		p.plus = true
		val = tag[1:]
	}
	p.typ = PieceType(val)
	return p
}

func (p Piece) IsEmpty() bool {
	if p.typ == PieceTypeNone {
		return true
	}
	return false
}

func (p Piece) Mark() string {

	if p.IsEmpty() {
		return ""
	}

	m := p.typ.Mark()

	prefix := ""
	if p.plus {
		prefix = "+"
	}

	if p.turn {
		m = strings.ToLower(m)
	}

	return prefix + m
}

type PieceType string

const (
	PieceTypePawn   PieceType = "pawn"
	PieceTypeLance  PieceType = "lance"
	PieceTypeKnight PieceType = "knight"
	PieceTypeSilver PieceType = "silver"
	PieceTypeGold   PieceType = "gold"
	PieceTypeBishop PieceType = "bishop"
	PieceTypeRook   PieceType = "rook"
	PieceTypeKing   PieceType = "king"
	PieceTypeNone   PieceType = "none"
)

func (pt PieceType) Mark() string {
	switch pt {
	case PieceTypePawn:
		return "P"
	case PieceTypeLance:
		return "L"
	case PieceTypeKnight:
		return "N"
	case PieceTypeSilver:
		return "S"
	case PieceTypeGold:
		return "G"
	case PieceTypeBishop:
		return "B"
	case PieceTypeRook:
		return "R"
	case PieceTypeKing:
		return "K"
	}
	return ""
}
