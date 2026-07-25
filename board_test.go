package suteme

import "testing"

const startBoard = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL"

func TestCountFromSFENStartpos(t *testing.T) {
	sente, gote := CountFromSFEN(startBoard)
	want := map[string]int{"P": 9, "L": 2, "N": 2, "S": 2, "G": 2, "B": 1, "R": 1, "K": 1}
	for k, v := range want {
		if sente[k] != v {
			t.Errorf("sente[%s] = %d, want %d", k, sente[k], v)
		}
		if gote[k] != v {
			t.Errorf("gote[%s] = %d, want %d", k, gote[k], v)
		}
	}
}

func TestToSFENEmpty(t *testing.T) {
	var bd Board // すべて空マス
	if got := bd.ToSFEN(); got != "9/9/9/9/9/9/9/9/9" {
		t.Errorf("empty ToSFEN = %q", got)
	}
}

func TestToSFENSinglePiece(t *testing.T) {
	var bd Board
	// 先頭段の先頭マスに先手歩、末尾マスに後手歩
	bd.Rows[0].Squares[0] = BoardSquare{Piece: Piece{typ: PieceTypePawn}}
	bd.Rows[0].Squares[8] = BoardSquare{Piece: Piece{typ: PieceTypePawn, turn: true}}
	if got := bd.ToSFEN(); got != "P7p/9/9/9/9/9/9/9/9" {
		t.Errorf("ToSFEN = %q, want %q", got, "P7p/9/9/9/9/9/9/9/9")
	}
}

func TestParsePosition(t *testing.T) {
	// "7g" は内部座標(x=3,y=7) → Row=9-7=2, Col=9-3=6
	p := parsePosition("7g")
	if p.Row != 2 || p.Col != 6 {
		t.Errorf("parsePosition(7g) = (Row=%d,Col=%d), want (2,6)", p.Row, p.Col)
	}
	if bad := parsePosition("zz"); !bad.IsError() {
		t.Errorf("parsePosition(zz) should be error")
	}
}
