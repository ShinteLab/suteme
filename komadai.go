package suteme

import (
	"fmt"

	"shinte/core/sfen"
)

// PieceLimits は先後合計の駒数上限
var PieceLimits = map[string]int{
	"P": 18, "L": 4, "N": 4, "S": 4,
	"G": 4, "B": 2, "R": 2, "K": 2,
}

// HandOrder は駒台に入りうる駒種の順序（玉は除く）
var HandOrder = []string{"P", "L", "N", "S", "G", "B", "R"}

// CountFromSFEN はSFEN盤面部分（スラッシュ区切りの9行）から先後の駒数を集計する。
// 盤面文字列の走査（数字・'+'・大小文字の解釈）は core/sfen に委譲する。
// 成駒はベース駒種にまとめて数える（例: "+P" は "P"）。
func CountFromSFEN(sfenBoard string) (sente, gote map[string]int) {
	sente = map[string]int{}
	gote = map[string]int{}
	// 不正な盤面でも解釈できた分だけ集計する（従来同様に寛容）ため、
	// ParseBoard のエラーは無視する。
	_ = sfen.ParseBoard(sfenBoard, func(rank, file, base int, black, promoted bool) {
		key := sfen.Letter(base)
		if black {
			sente[key]++
		} else {
			gote[key]++
		}
	})
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
