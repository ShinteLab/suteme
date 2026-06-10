package suteme

import (
	"image"
	"math"
	"sort"
)

type HoughLine struct {
	Rho   float64
	Theta float64
	Votes int
}

func (h HoughLine) ToLine(width, height int) Line {
	a := math.Cos(h.Theta)
	b := math.Sin(h.Theta)
	x0 := a * h.Rho
	y0 := b * h.Rho

	scale := float64(width + height)
	pt1 := image.Pt(int(x0-scale*b), int(y0+scale*a))
	pt2 := image.Pt(int(x0+scale*b), int(y0-scale*a))

	return NewLine(pt1, pt2)
}

func (h HoughLine) IsHorizontal() bool {
	return math.Abs(h.Theta-math.Pi/2) < math.Pi/6
}

func (h HoughLine) IsVertical() bool {
	return h.Theta < math.Pi/6 || h.Theta > 5*math.Pi/6
}

func FindLines(img *image.Gray, threshold int) []HoughLine {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	diagonal := int(math.Ceil(math.Sqrt(float64(w*w + h*h))))
	numRho := 2*diagonal + 1
	numTheta := 180

	sinTable := make([]float64, numTheta)
	cosTable := make([]float64, numTheta)
	for t := 0; t < numTheta; t++ {
		theta := float64(t) * math.Pi / float64(numTheta)
		sinTable[t] = math.Sin(theta)
		cosTable[t] = math.Cos(theta)
	}

	acc := make([]int, numRho*numTheta)

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if img.GrayAt(x+bounds.Min.X, y+bounds.Min.Y).Y == 0 {
				continue
			}
			for t := 0; t < numTheta; t++ {
				rho := int(math.Round(float64(x)*cosTable[t] + float64(y)*sinTable[t]))
				acc[(rho+diagonal)*numTheta+t]++
			}
		}
	}

	neighborSize := 9
	lines := make([]HoughLine, 0)

	for r := 0; r < numRho; r++ {
		for t := 0; t < numTheta; t++ {
			val := acc[r*numTheta+t]
			if val < threshold {
				continue
			}
			isMax := true
			for dr := -neighborSize; dr <= neighborSize && isMax; dr++ {
				for dt := -neighborSize; dt <= neighborSize && isMax; dt++ {
					if dr == 0 && dt == 0 {
						continue
					}
					nr, nt := r+dr, t+dt
					if nr < 0 || nr >= numRho || nt < 0 || nt >= numTheta {
						continue
					}
					if acc[nr*numTheta+nt] > val {
						isMax = false
					}
				}
			}
			if isMax {
				rho := float64(r - diagonal)
				theta := float64(t) * math.Pi / float64(numTheta)
				lines = append(lines, HoughLine{Rho: rho, Theta: theta, Votes: val})
			}
		}
	}

	sort.Slice(lines, func(i, j int) bool {
		return lines[i].Votes > lines[j].Votes
	})

	return lines
}
