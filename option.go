package suteme

import (
	"image"

	"github.com/ShinteLab/core/sfen"
)

// 認識(LoadSFEN / Recognize)の挙動を呼び出し側が決めるためのオプション。
//
// **画像から出てきた盤面がおかしいことは異常ではない**(中継が盤を映していない、
// 認識器が外す)。何をチェックするか・どれをエラーとして返すか・読めない駒台を
// どう埋めるかはアプリごとに違うので、既定は「全部調べるが、エラーにはしない」。
//
// 盤面のルール判定そのものは core/sfen が持つ。ここに将棋の仕様を書き足さないこと。

// HandSide は「盤上に無い駒(＝駒台にあるはずの駒)」を誰の持ち駒として扱うか。
//
// **どちらの持ち駒かは盤面からは決まらない。** 駒台を画像から読めていない現状、
// 完全な SFEN が要る用途では便宜的にどちらかへ寄せるしかないので、その選択肢。
type HandSide int

const (
	// HandNone は割り振らない(既定)。Result.HandTotal に先後不明のまま残り、
	// SFEN の持ち駒欄は "-" になる。
	HandNone HandSide = iota
	// HandBlack は不足分をすべて先手の持ち駒にする。
	HandBlack
	// HandWhite は不足分をすべて後手の持ち駒にする。
	HandWhite
)

// config は認識1回分の設定。ゼロ値ではなく defaultConfig から作る。
type config struct {
	predictor Predictor
	region    *BoardRegion // 盤面領域の明示指定(nil なら画像から検出)
	checks    sfen.Check   // 実施するチェック
	fatal     sfen.Check   // 違反をエラーとして返すチェック
	hand      HandSide
	black     bool // 手番(true=先手)
	move      int  // 手数
}

func defaultConfig() *config {
	return &config{
		checks: sfen.CheckAll,
		fatal:  0, // 既定では違反があってもエラーにしない(結果に載せるだけ)
		hand:   HandNone,
		black:  true,
		move:   1,
	}
}

func (c *config) apply(opts []Option) *config {
	for _, o := range opts {
		if o != nil {
			o(c)
		}
	}
	return c
}

// Option は認識の設定を変える関数。
type Option func(*config)

// WithPredictor は駒種推論器を明示指定する(既定はファイル探索した既定の推論器)。
func WithPredictor(p Predictor) Option {
	return func(c *config) { c.predictor = p }
}

// WithRegion は盤面領域を明示指定する。盤の座標が既に分かっている場合は
// 検出を挟まないぶん確実(AGENTS.md の「盤面座標が分かっているなら
// BoardRegionFromRect を直接使う」に相当)。
func WithRegion(br *BoardRegion) Option {
	return func(c *config) { c.region = br }
}

// WithRect は2点で盤面領域を指定する(WithRegion + BoardRegionFromRect)。
func WithRect(x1, y1, x2, y2 int) Option {
	return WithRegion(BoardRegionFromRect(x1, y1, x2, y2))
}

// WithChecks は実施するチェックを指定する(既定は sfen.CheckAll)。
// 実施しないチェックの違反は Result にも載らない。
func WithChecks(c sfen.Check) Option {
	return func(cfg *config) { cfg.checks = c }
}

// WithErrorOn は違反をエラーとして返すチェックを指定する(既定は無し＝
// 違反があっても Result に載せるだけ)。指定したチェックは自動的に実施される。
//
//	// 駒数の辻褄が合わないときだけエラーにする
//	suteme.LoadSFEN(img, suteme.WithErrorOn(sfen.CheckPieceCount|sfen.CheckKing))
func WithErrorOn(c sfen.Check) Option {
	return func(cfg *config) {
		cfg.fatal |= c
		cfg.checks |= c
	}
}

// WithStrict は全てのチェックを行い、1つでも違反があればエラーにする。
func WithStrict() Option {
	return func(cfg *config) {
		cfg.checks = sfen.CheckAll
		cfg.fatal = sfen.CheckAll
	}
}

// WithLenient は違反を一切エラーにしない(調べはする)。既定と同じ。
func WithLenient() Option {
	return func(cfg *config) { cfg.fatal = 0 }
}

// WithHandTo は盤上に無い駒をどちらの持ち駒として扱うかを決める。
func WithHandTo(side HandSide) Option {
	return func(cfg *config) { cfg.hand = side }
}

// WithTurn は Result.SFEN() が出力する手番を指定する(既定は先手)。
// **手番は盤面からは決まらない**ので、認識結果には影響しない。
func WithTurn(black bool) Option {
	return func(cfg *config) { cfg.black = black }
}

// WithMoveNumber は Result.SFEN() が出力する手数を指定する(既定は1)。
func WithMoveNumber(n int) Option {
	return func(cfg *config) { cfg.move = n }
}

// boardRegion は設定に従って盤面領域・信頼度・領域の決め方を返す。
func (c *config) boardRegion(img image.Image) (*BoardRegion, float64, RegionSource) {
	if c.region != nil {
		return c.region, ValidateBoard(img, c.region), RegionFromOption
	}
	return detectBoardRegion(img)
}
