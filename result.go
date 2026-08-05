package suteme

import (
	"errors"
	"strconv"
	"strings"

	"github.com/ShinteLab/core/sfen"
)

// ErrInvalidBoard は認識した盤面が将棋の局面として成立していないことを表す。
// どの点がおかしいかは BoardError（errors.As で取り出す）か Result.Violations を見る。
var ErrInvalidBoard = errors.New("盤面が将棋の局面として成立していません")

// BoardError は検証で見つかった違反をまとめたエラー。
// エラーを返す場合でも Recognize は Result を返すので、
// 「エラーにしたうえで、どこがおかしいかも見せる」ことができる。
type BoardError struct {
	Board      string
	Violations []sfen.Violation
}

func (e *BoardError) Error() string {
	msgs := make([]string, 0, len(e.Violations))
	for _, v := range e.Violations {
		msgs = append(msgs, v.Detail)
	}
	return ErrInvalidBoard.Error() + ": " + strings.Join(msgs, " / ")
}

// Is は errors.Is(err, ErrInvalidBoard) を成立させる。
func (e *BoardError) Is(target error) bool { return target == ErrInvalidBoard }

// Unwrap は個々の違反を返す（errors.As で sfen.Violation を取り出せる）。
func (e *BoardError) Unwrap() []error {
	errs := make([]error, len(e.Violations))
	for i, v := range e.Violations {
		errs[i] = v
	}
	return errs
}

// Result は画像1枚の認識結果。
type Result struct {
	// Board は SFEN の盤面部分（'/' 区切りの9段）。
	Board string `json:"board"`
	// Confidence は盤面検出の信頼度（ValidateBoard の値）。
	Confidence float64 `json:"confidence"`
	// Black / White は盤上の駒数（SFEN 大文字がキー、成駒はベース駒に合算）。
	Black map[string]int `json:"black"`
	White map[string]int `json:"white"`
	// HandTotal は盤上に無い駒の枚数（先後の区別なし）。
	HandTotal map[string]int `json:"hand_total"`
	// BlackHand / WhiteHand は WithHandTo で割り振った持ち駒。
	// HandNone（既定）なら両方とも空。
	BlackHand map[string]int `json:"black_hand"`
	WhiteHand map[string]int `json:"white_hand"`
	// Violations は実施したチェックで見つかった違反（エラーにしたかは別）。
	Violations []sfen.Violation `json:"violations"`
	// Debug は「その答えをどう出したか」の観測情報（盤面と判定した矩形・
	// その決め方・使った推論器・マスごとの分類と確信度）。
	// 認識が外れたときの切り分け用で、判断には使わない。Recognize が常に埋める。
	Debug *Debug `json:"debug,omitempty"`

	turn string // "b" / "w"
	move int
}

// OK は違反が1つも無いことを返す。
func (r *Result) OK() bool { return len(r.Violations) == 0 }

// Filter は指定した種類の違反だけを返す。
func (r *Result) Filter(c sfen.Check) []sfen.Violation {
	var out []sfen.Violation
	for _, v := range r.Violations {
		if c.Has(v.Check) {
			out = append(out, v)
		}
	}
	return out
}

// Warnings は違反の日本語メッセージだけを返す（UI 表示用）。
func (r *Result) Warnings() []string {
	out := make([]string, 0, len(r.Violations))
	for _, v := range r.Violations {
		out = append(out, v.Detail)
	}
	return out
}

// SFEN は完全な SFEN（盤面 手番 持ち駒 手数）を返す。
// 手番と手数は WithTurn / WithMoveNumber の指定（既定は "b" と 1）。
// 持ち駒は WithHandTo の割り振り結果で、HandNone なら "-"。
func (r *Result) SFEN() string {
	return r.Board + " " + r.turn + " " + r.hands() + " " + strconv.Itoa(r.move)
}

// hands は SFEN の持ち駒欄を組み立てる。表記は core/sfen に委譲する。
func (r *Result) hands() string {
	return sfen.FormatHands(toCodeMap(r.BlackHand), toCodeMap(r.WhiteHand))
}

// newResult は sfen.Inspect の結果を Result に組み替える。
func newResult(board string, conf float64, info *sfen.BoardInfo, cfg *config) *Result {
	r := &Result{
		Board:      board,
		Confidence: conf,
		Black:      toLetterMap(info.Black),
		White:      toLetterMap(info.White),
		HandTotal:  toLetterMap(info.Hands),
		BlackHand:  map[string]int{},
		WhiteHand:  map[string]int{},
		Violations: info.Violations,
		turn:       "b",
		move:       cfg.move,
	}
	if !cfg.black {
		r.turn = "w"
	}
	switch cfg.hand {
	case HandBlack:
		r.BlackHand = toLetterMap(info.Hands)
	case HandWhite:
		r.WhiteHand = toLetterMap(info.Hands)
	}
	return r
}

// toLetterMap はベース駒コードのマップを SFEN 大文字キーのマップにする。
func toLetterMap(m map[int]int) map[string]int {
	out := make(map[string]int, len(m))
	for base, n := range m {
		if n != 0 {
			out[sfen.Letter(base)] = n
		}
	}
	return out
}

// toCodeMap は SFEN 大文字キーのマップをベース駒コードのマップに戻す。
func toCodeMap(m map[string]int) map[int]int {
	if len(m) == 0 {
		return nil
	}
	out := make(map[int]int, len(m))
	for letter, n := range m {
		if letter == "" || n == 0 {
			continue
		}
		base, _ := sfen.ParsePieceLetter(letter[0])
		if base != sfen.NotFound {
			out[base] = n
		}
	}
	return out
}
