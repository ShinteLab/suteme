package suteme

import (
	"image"
	"math"
	"sort"
)

type Grid struct {
	Horizontal []HoughLine
	Vertical   []HoughLine
}

func FindGrid(lines []HoughLine, topN int) *Grid {
	if len(lines) < 20 {
		return nil
	}

	a1, a2 := findPerpPair(lines)
	angleTol := math.Pi / 12

	var group1, group2 []HoughLine
	for _, l := range lines {
		if thetaDist(l.Theta, a1) < angleTol {
			group1 = append(group1, l)
		} else if thetaDist(l.Theta, a2) < angleTol {
			group2 = append(group2, l)
		}
	}

	hLines := mergeNearby(group1, 25)
	vLines := mergeNearby(group2, 25)

	h := pickGridLines(hLines, 10)
	v := pickGridLines(vLines, 10)

	if h == nil || v == nil {
		return nil
	}

	return &Grid{Horizontal: h, Vertical: v}
}

func findPerpPair(lines []HoughLine) (float64, float64) {
	numBins := 180
	bins := make([]int, numBins)
	for _, l := range lines {
		bin := int(l.Theta * float64(numBins) / math.Pi)
		if bin >= numBins {
			bin = numBins - 1
		}
		bins[bin]++
	}

	smoothed := make([]int, numBins)
	radius := 5
	for i := 0; i < numBins; i++ {
		sum := 0
		for d := -radius; d <= radius; d++ {
			j := i + d
			if j >= 0 && j < numBins {
				sum += bins[j]
			}
		}
		smoothed[i] = sum
	}

	bestScore := 0
	bestBin := 0
	halfBins := numBins / 2
	flex := 20

	for i := 0; i < numBins; i++ {
		for offset := halfBins - flex; offset <= halfBins+flex; offset++ {
			j := i + offset
			if j >= numBins {
				continue
			}
			score := smoothed[i] + smoothed[j]
			if score > bestScore {
				bestScore = score
				bestBin = i
			}
		}
	}

	angle1 := (float64(bestBin) + 0.5) * math.Pi / float64(numBins)
	angle2 := angle1 + math.Pi/2
	if angle2 >= math.Pi {
		angle2 -= math.Pi
	}

	return angle1, angle2
}

func thetaDist(a, b float64) float64 {
	d := math.Abs(a - b)
	if d > math.Pi/2 {
		d = math.Pi - d
	}
	return d
}

func mergeNearby(lines []HoughLine, threshold float64) []HoughLine {
	if len(lines) == 0 {
		return nil
	}
	sort.Slice(lines, func(i, j int) bool {
		return lines[i].Rho < lines[j].Rho
	})
	merged := []HoughLine{lines[0]}
	for i := 1; i < len(lines); i++ {
		last := &merged[len(merged)-1]
		if math.Abs(lines[i].Rho-last.Rho) <= threshold {
			if lines[i].Votes > last.Votes {
				*last = lines[i]
			}
		} else {
			merged = append(merged, lines[i])
		}
	}
	return merged
}

func pickGridLines(lines []HoughLine, count int) []HoughLine {
	n := len(lines)
	if n < count/2 {
		return nil
	}

	sort.Slice(lines, func(i, j int) bool {
		return lines[i].Rho < lines[j].Rho
	})

	if n == count {
		return lines
	}

	minFound := count * 7 / 10
	bestFound := 0
	bestScore := 0
	bestSpacing := 0.0
	bestOrigin := 0.0
	var bestSlots [10]int

	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			diff := lines[j].Rho - lines[i].Rho
			if diff < 20 {
				continue
			}

			maxGap := count - 1
			if diff/float64(maxGap) < 10 {
				continue
			}

			for gap := 1; gap <= maxGap; gap++ {
				spacing := diff / float64(gap)
				if spacing < 15 {
					continue
				}

				tolerance := spacing * 0.3
				origin := lines[i].Rho
				found := 0
				score := 0
				var slots [10]int

				for k := 0; k < count; k++ {
					expected := origin + float64(k)*spacing
					bestDist := tolerance
					bestIdx := -1
					for idx, l := range lines {
						d := math.Abs(l.Rho - expected)
						if d < bestDist {
							bestDist = d
							bestIdx = idx
						}
					}
					if bestIdx >= 0 {
						slots[k] = bestIdx
						found++
						score += lines[bestIdx].Votes
					} else {
						slots[k] = -1
					}
				}

				if found < minFound {
					continue
				}

				if found > bestFound || (found == bestFound && score > bestScore) {
					bestFound = found
					bestScore = score
					bestSpacing = spacing
					bestOrigin = origin
					bestSlots = slots
				}
			}
		}
	}

	if bestFound < minFound {
		return nil
	}

	avgTheta := 0.0
	thetaCount := 0
	for _, idx := range bestSlots {
		if idx >= 0 {
			avgTheta += lines[idx].Theta
			thetaCount++
		}
	}
	if thetaCount > 0 {
		avgTheta /= float64(thetaCount)
	}

	result := make([]HoughLine, count)
	for k := 0; k < count; k++ {
		if bestSlots[k] >= 0 {
			result[k] = lines[bestSlots[k]]
		} else {
			result[k] = HoughLine{
				Rho:   bestOrigin + float64(k)*bestSpacing,
				Theta: avgTheta,
				Votes: 0,
			}
		}
	}

	return result
}

func (h HoughLine) Intersect(other HoughLine) (image.Point, bool) {
	det := math.Cos(h.Theta)*math.Sin(other.Theta) - math.Cos(other.Theta)*math.Sin(h.Theta)
	if math.Abs(det) < 1e-10 {
		return image.Point{}, false
	}
	x := (h.Rho*math.Sin(other.Theta) - other.Rho*math.Sin(h.Theta)) / det
	y := (other.Rho*math.Cos(h.Theta) - h.Rho*math.Cos(other.Theta)) / det
	return image.Pt(int(math.Round(x)), int(math.Round(y))), true
}

func (g *Grid) Intersections() [10][10]image.Point {
	var pts [10][10]image.Point
	for i, h := range g.Horizontal {
		for j, v := range g.Vertical {
			pt, ok := h.Intersect(v)
			if ok {
				pts[i][j] = pt
			}
		}
	}
	return pts
}

func (g *Grid) ExtractCell(src image.Image, pts [10][10]image.Point, row, col int) image.Image {
	tl := pts[row][col]
	br := pts[row+1][col+1]
	rect := image.Rect(tl.X, tl.Y, br.X, br.Y)
	rect = rect.Canon()
	rect = rect.Intersect(src.Bounds())
	if rect.Empty() {
		return nil
	}

	dst := image.NewRGBA(image.Rect(0, 0, rect.Dx(), rect.Dy()))
	for y := rect.Min.Y; y < rect.Max.Y; y++ {
		for x := rect.Min.X; x < rect.Max.X; x++ {
			dst.Set(x-rect.Min.X, y-rect.Min.Y, src.At(x, y))
		}
	}
	return dst
}
