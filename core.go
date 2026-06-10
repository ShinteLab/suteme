package suteme

import (
	"fmt"
	"image"
	"math"
)

const float64Equality = 1e-1

func sameF(a, b float64) bool {
	return math.Abs(a-b) <= float64Equality
}

const IntEquality = 3

func sameI(a, b int) bool {
	return math.Abs(float64(a-b)) <= IntEquality
}

type Line [2]image.Point

func NewLine(s, e image.Point) Line {
	var l Line
	l[0] = s
	l[1] = e
	return l
}

func (l Line) On(p image.Point) bool {

	p1 := l[0]
	p2 := l[1]

	if p1.X == p2.X {
		if !sameI(p1.X, p.X) {
			return false
		}
	} else if p1.Y == p2.Y {
		if !sameI(p1.Y, p.Y) {
			return false
		}
	} else {

		a1 := float64(p2.Y-p1.Y) / float64(p2.X-p1.X)
		a2 := float64(p2.Y-p.Y) / float64(p2.X-p.X)
		fmt.Println("    A1 =", a1)
		fmt.Println("    A2 =", a2)

		if !sameF(a1, a2) {
			return false
		}

		/*
			b1 := float64(p1.Y) - (a1 * float64(p1.X))
			b2 := float64(p.Y) - (a2 * float64(p.X))

			fmt.Println("    B1 =", b1)
			fmt.Println("    B2 =", b2)

			if !same(b1, b2) {
				return false
			}
		*/
	}
	return true
}

type Rect [4]image.Point

func toRect(rect image.Rectangle) Rect {
	var rtn Rect
	minP := rect.Min
	maxP := rect.Max
	rtn[0] = image.Pt(minP.X, maxP.Y)
	rtn[1] = image.Pt(maxP.X, maxP.Y)
	rtn[2] = image.Pt(maxP.X, minP.Y)
	rtn[3] = image.Pt(minP.X, minP.Y)
	return rtn
}

func (r Rect) toLine() []Line {
	rtn := make([]Line, 4)
	for i := range r {
		rtn[i] = NewLine(r[i], r[(i+1)%4])
	}
	return rtn
}
