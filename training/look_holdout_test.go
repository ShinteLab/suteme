package training

import (
	"strings"
	"testing"

	"github.com/ShinteLab/suteme"
)

// TestLookHoldout は**見た目ホールドアウト**（その盤を丸ごと学習から外す）の成績を測る。
//
// 評価タブの「見た目LOO」（`runEvaluation` の `ManualLook`）と同じ測り方
// （`holdoutPredictors` を見た目の名前で束ね、格子線へ寄せた手動座標で認識する）を、
// 画面を通さずに回して局面ごとの内訳まで出す。認識の判断を変えたときに
// 「良くなった局面と悪くなった局面」を数えるためのもの。合否判定はしない。
//
//	$env:SUTEME_SLOW=1; go test -run TestLookHoldout -timeout 60m -v ./training
func TestLookHoldout(t *testing.T) {
	requireSlow(t)
	if !chdirToData(t) {
		t.Skip("data/history.json が無いのでスキップ")
	}
	h := loadHistory()
	lookOf := func(e HistoryEntry) string { return strings.TrimSpace(e.Look) }
	kns := holdoutPredictors(h.Entries, lookOf)
	if len(kns) == 0 {
		t.Skip("見た目の判定済みの局面が無いのでスキップ")
	}

	var total EvalMetrics
	for _, e := range h.Entries {
		look := lookOf(e)
		kn, ok := kns[look]
		if look == "" || !ok || !e.IsVerified() {
			continue
		}
		tg, err := loadEvalTarget(e)
		if err != nil {
			continue
		}
		s := tg.snap
		m := EvalMetrics{}
		if g, _, err := recognizeGrid(tg.img, suteme.Predictor(kn), suteme.WithRect(s.Min.X, s.Min.Y, s.Max.X, s.Max.Y)); err == nil {
			m = gradeBoard(tg.want, g)
		}
		total.add(m)
		t.Logf("%s [%s]: %d/81 空→駒 %d 駒→空 %d 向き %d 駒種違い %d",
			e.ID, look, m.OK, m.EmptyAsPiece, m.PieceAsEmpty, m.OrientFlip, m.TypeMiss)
	}
	if total.Cells == 0 {
		t.Skip("評価できる局面が無いのでスキップ")
	}
	t.Logf("見た目LOO %d 局面 / %d 見た目: 全マス %d/%d = %.2f%%  完全一致 %d",
		total.Boards, len(kns), total.OK, total.Cells, 100*float64(total.OK)/float64(total.Cells), total.BoardsPerfect)
	t.Logf("  空→駒 %d  駒→空 %d  向き %d  駒種違い %d  駒種 %d/%d = %.2f%%",
		total.EmptyAsPiece, total.PieceAsEmpty, total.OrientFlip, total.TypeMiss,
		total.PieceOK, total.PieceCells, 100*float64(total.PieceOK)/float64(total.PieceCells))
}
