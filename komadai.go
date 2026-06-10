package suteme

import (
	"fmt"
	"strings"
	"unicode"
)

// PieceLimits は先後合計の駒数上限
var PieceLimits = map[string]int{
	"P": 18, "L": 4, "N": 4, "S": 4,
	"G": 4, "B": 2, "R": 2, "K": 2,
}

// HandOrder は駒台に入りうる駒種の順序（玉は除く）
var HandOrder = []string{"P", "L", "N", "S", "G", "B", "R"}

// baseType は成駒含む駒種文字列をベース駒種に変換する
func baseType(t string) string {
	switch strings.ToUpper(t) {
	case "+P":
		return "P"
	case "+L":
		return "L"
	case "+N":
		return "N"
	case "+S":
		return "S"
	case "+B":
		return "B"
	case "+R":
		return "R"
	default:
		return strings.ToUpper(t)
	}
}

// CountFromSFEN はSFEN盤面部分（スラッシュ区切りの9行）から先後の駒数を集計する
func CountFromSFEN(sfenBoard string) (sente, gote map[string]int) {
	sente = map[string]int{}
	gote = map[string]int{}
	promoted := false
	for _, ch := range sfenBoard {
		switch {
		case ch == '/' || (ch >= '1' && ch <= '9'):
			promoted = false
		case ch == '+':
			promoted = true
		default:
			var key string
			if promoted {
				key = baseType("+" + strings.ToUpper(string(ch)))
				promoted = false
			} else {
				key = strings.ToUpper(string(ch))
			}
			if unicode.IsLower(ch) {
				gote[key]++
			} else {
				sente[key]++
			}
		}
	}
	return sente, gote
}

// PieceValidation は盤面の駒数検証結果
type PieceValidation struct {
	Sente     map[string]int `json:"sente"`
	Gote      map[string]int `json:"gote"`
	HandTotal map[string]int `json:"hand_total"` // 駒台合計（先後不明）
	Warnings  []string       `json:"warnings"`
}

// ValidatePieces はSFEN盤面部分を検証し駒台枚数を推定する
func ValidatePieces(sfenBoard string) *PieceValidation {
	sente, gote := CountFromSFEN(sfenBoard)
	v := &PieceValidation{
		Sente:     sente,
		Gote:      gote,
		HandTotal: map[string]int{},
		Warnings:  []string{},
	}
	for _, pt := range HandOrder {
		limit := PieceLimits[pt]
		total := sente[pt] + gote[pt]
		if total > limit {
			v.Warnings = append(v.Warnings,
				fmt.Sprintf("%s: %d枚（上限%d）", pt, total, limit))
		} else {
			if hand := limit - total; hand > 0 {
				v.HandTotal[pt] = hand
			}
		}
	}
	// 玉は2枚固定のはず
	if kings := sente["K"] + gote["K"]; kings != 2 {
		v.Warnings = append(v.Warnings, fmt.Sprintf("玉: %d枚（通常2枚）", kings))
	}
	return v
}
