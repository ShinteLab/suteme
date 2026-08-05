package suteme

import (
	"github.com/ShinteLab/core/sfen"
)

// 駒数の上限・二歩・行き所のない駒といった**将棋の仕様は core/sfen が持つ**。
// ここにあるのは、その結果を suteme / 学習用サーバが扱ってきた形
// （SFEN 大文字キーのマップ + 日本語の警告文字列）に組み替える層。

// PieceLimits は先後合計の駒数上限（キーは SFEN の大文字）。
// 定義は core/sfen にあり、ここでは表引きの形に直しているだけ。
var PieceLimits = pieceLimits()

// HandOrder は駒台に入りうる駒種の順序（玉は除く）。core/sfen の持ち駒順に従う。
var HandOrder = handOrder()

func pieceLimits() map[string]int {
	m := map[string]int{}
	for _, base := range append([]int{sfen.King}, sfen.HandOrder...) {
		m[sfen.Letter(base)] = sfen.PieceLimit(base)
	}
	return m
}

func handOrder() []string {
	out := make([]string, 0, len(sfen.HandOrder))
	for _, base := range sfen.HandOrder {
		out = append(out, sfen.Letter(base))
	}
	return out
}

// CountFromSFEN はSFEN盤面部分（スラッシュ区切りの9行）から先後の駒数を集計する。
// 盤面文字列の走査（数字・'+'・大小文字の解釈）は core/sfen に委譲する。
// 成駒はベース駒種にまとめて数える（例: "+P" は "P"）。
// 不正な盤面でも解釈できた分だけ集計する（従来同様に寛容）。
func CountFromSFEN(sfenBoard string) (sente, gote map[string]int) {
	info := sfen.Inspect(sfenBoard, 0) // チェックはせず集計だけ
	return toLetterMap(info.Black), toLetterMap(info.White)
}

// PieceValidation は盤面の駒数検証結果
type PieceValidation struct {
	Sente     map[string]int `json:"sente"`
	Gote      map[string]int `json:"gote"`
	HandTotal map[string]int `json:"hand_total"` // 駒台合計（先後不明）
	Warnings  []string       `json:"warnings"`
}

// ValidatePieces はSFEN盤面部分を検証し駒台枚数を推定する。
// 見るのは駒数の辻褄（駒種ごとの上限・玉の数）だけ。二歩や行き所のない駒まで
// 見たい場合は sfen.Inspect / Recognize のオプションを使う。
func ValidatePieces(sfenBoard string) *PieceValidation {
	info := sfen.Inspect(sfenBoard, sfen.CheckCounts)
	v := &PieceValidation{
		Sente:     toLetterMap(info.Black),
		Gote:      toLetterMap(info.White),
		HandTotal: toLetterMap(info.Hands),
		Warnings:  []string{},
	}
	v.Warnings = append(v.Warnings, info.Messages()...)
	return v
}
