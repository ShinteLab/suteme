package suteme

import (
	"image"
	_ "image/png"
	"testing"
)

// 保存済み局面（data/）の手動指定座標を正解として ValidateBoard の
// 弁別力を測る。正しいマス割りと、ずらした／周期を変えたマス割りを
// 比べ、**正解が必ず最高スコアになる**ことを確認する。
//
// これが崩れると detectBoardRegion のフォールバックも
// minBoardConfidence による棄却も効かなくなる（旧実装の均一性のみでは
// どの候補も 1.00 を返していた）。
//
// data/ は .gitignore 対象なので、無ければスキップする。
func TestValidateBoardDiscriminates(t *testing.T) {
	dir, entries := loadSavedBoards(t)

	// 歪めた領域の名前。**並列の中で t.Errorf を呼ばない**ので、
	// 結果を持ち帰ってから順に判定する（mapSavedBoards の説明）
	badNames := []string{
		"横半マスずれ", "縦半マスずれ", "上下左右半マス縮小",
		"半周期(左上)", "半周期(中央)", "半周期(右下)",
	}
	type res struct {
		id   string
		want float64
		got  []float64
	}
	rs := mapSavedBoards(dir, entries, func(e historyEntry, img image.Image) res {
		x1, y1, x2, y2 := e.Bounds.X1, e.Bounds.Y1, e.Bounds.X2, e.Bounds.Y2
		cw, ch := (x2-x1)/9, (y2-y1)/9

		want := ValidateBoard(img, BoardRegionFromRect(x1, y1, x2, y2))

		// 位置ずれ（周期は正しい）と周期ずれ。いずれも正解を上回ってはいけない。
		// 周期が半分（マス2つぶんを1マスとみなす）は格子線に1本おきに
		// 乗るので**内側の色の均一性では区別が付かず**、盤の 1/4 が
		// 信頼度 1.00 で返っていた。ikkyoku のガイド枠の自動フィットが
		// これを掴んで枠を潰す（axisAlignment のパリティ参照）
		bad := []*BoardRegion{
			BoardRegionFromRect(x1+cw/2, y1, x2+cw/2, y2),
			BoardRegionFromRect(x1, y1+ch/2, x2, y2+ch/2),
			BoardRegionFromRect(x1+cw/2, y1+ch/2, x2-cw/2, y2-ch/2),
			BoardRegionFromRect(x1, y1, x1+cw*9/2, y1+ch*9/2),
			BoardRegionFromRect(x1+cw*9/4, y1+ch*9/4, x1+cw*27/4, y1+ch*27/4),
			BoardRegionFromRect(x2-cw*9/2, y2-ch*9/2, x2, y2),
		}
		got := make([]float64, len(bad))
		for i, br := range bad {
			got[i] = ValidateBoard(img, br)
		}
		return res{id: e.ID, want: want, got: got}
	})

	n, clear := 0, 0
	for _, r := range rs {
		n++
		if r.want >= minBoardConfidence {
			clear++
		}
		for i, name := range badNames {
			if r.got[i] > r.want {
				t.Errorf("%s: %s のスコア %.2f が正解 %.2f を上回った", r.id, name, r.got[i], r.want)
			}
		}
		// 正解が採用される画像では、歪めた領域は棄却されなければならない。
		// 正解自体が閾値に届かない画像（後述）では比較する意味が無い
		if r.want >= minBoardConfidence {
			for i, name := range badNames {
				if r.got[i] >= minBoardConfidence {
					t.Errorf("%s: %s が棄却されない conf=%.2f", r.id, name, r.got[i])
				}
			}
		}
		t.Logf("%s: 正解=%.2f 横半マス=%.2f 縦半マス=%.2f 縮小=%.2f 半周期=%.2f/%.2f/%.2f",
			r.id, r.want, r.got[0], r.got[1], r.got[2], r.got[3], r.got[4], r.got[5])
	}
	if n == 0 {
		t.Skip("画像が無いのでスキップ")
	}

	// **現在は 15/15 が閾値を超える。** 投影を方向別・中央値にする前は、
	// 実物の盤を撮った画像で片方の軸の格子線が出ず（木目・照明・駒の重なり）、
	// 正しい領域でも 0 近辺までしか上がらない画像があった（12/14）。
	// 手動指定はこの値でゲートしていないので実害は表示だけだが、
	// 自動検出は候補をこの値で選ぶので効いていないと成立しない。
	// 新しい画像 1 枚で落ちないよう、要求は 8 割に留める。
	t.Logf("正解座標が %s を超えたのは %d/%d", "minBoardConfidence", clear, n)
	if clear*10 < n*8 {
		t.Errorf("正解座標が採用されたのが %d/%d しかない", clear, n)
	}
}
