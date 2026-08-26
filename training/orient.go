package training

import (
	"log"
	"sort"

	"github.com/ShinteLab/suteme"
)

// 向きの回転照合だけを担うデータを、学習データから間引いて作る。
//
// **目的は配布**（`suteme.DefaultOrientFile` のコメント）。駒種を NN で読む
// 構成では向きの照合器が無く、分類器に落ちて大きく損をする。照合に要るのは
// 「駒サンプルへの最近傍距離」だけなので、全件は要らない。
//
// **空マス（`ClassEmpty`）は入れない。** `KNN.PieceDistance` が読み飛ばすので
// 持っていても引かれず、ファイルが太るだけ。

// OrientPerClass は向き照合データに残すクラスごとのサンプル数。
//
// **クラスあたりの上限であって、クラスの偏りを直すものではない**
// （少ないクラスを水増ししない。`BalanceData` とは別物）。
//
// 実測（30局面 / 1025 駒マス・手動座標・NN で駒種を読んだときの向きの反転）:
//
//	照合データ無し      69 件   —
//	 20/class (274件)   51 件   0.6MB
//	 60/class (768件)   18 件   1.7MB
//	120/class (1429件)  13 件   3.2MB
//	250/class (2543件)   8 件   5.7MB   ← 現在
//	500/class (4543件)   4 件  10.2MB
//
// **平らにはならないので、大きさで決めるしかない。** 相方の
// `model_v7.json` が 3MB なので、**2 つ合わせて 1 桁 MB に収まる 250** を採る。
// 増やすと推論時間も伸びる（`PieceDistance` は全件走査。185ms → 250ms/局面）。
const OrientPerClass = 250

// BuildOrientData は学習データから向き照合用のサンプルを間引いて返す。
//
// **等間隔に抜く（先頭から詰めない）。** 学習データは局面ごとに順に
// 積まれているので、先頭から取ると**古い局面の盤だけ**が残る。
// 回転照合の成績は「同じ見た目の盤を見たことがあるか」で決まる
// （CLAUDE.md「向きの回転照合」）ので、**見た目の広がりが命**。
// 等間隔なら局面をまたいで散る。
func BuildOrientData(samples []suteme.TrainingSample, perClass int) []suteme.TrainingSample {
	if perClass <= 0 {
		return nil
	}
	byClass := map[int][]int{}
	for i, s := range samples {
		if s.Label == suteme.ClassEmpty || s.Label < 0 {
			continue
		}
		byClass[s.Label] = append(byClass[s.Label], i)
	}

	classes := make([]int, 0, len(byClass))
	for c := range byClass {
		classes = append(classes, c)
	}
	sort.Ints(classes)

	var out []suteme.TrainingSample
	for _, c := range classes {
		idx := byClass[c]
		if len(idx) <= perClass {
			for _, i := range idx {
				out = append(out, samples[i])
			}
			continue
		}
		// 等間隔に perClass 件。端も必ず含める
		for k := 0; k < perClass; k++ {
			out = append(out, samples[idx[k*len(idx)/perClass]])
		}
	}
	return out
}

// rebuildOrientData は学習データから向き照合データを作り直して保存する。
// 作れたサンプル数を返す。**毎回ゼロから作る**（学習データの部分集合なので、
// 累積すると消したはずのサンプルが残る）。
func rebuildOrientData(samples []suteme.TrainingSample) (int, error) {
	out := BuildOrientData(samples, OrientPerClass)
	if len(out) == 0 {
		return 0, nil
	}
	if err := suteme.SaveTrainingData(orientFile, &suteme.TrainingData{Samples: out}); err != nil {
		return 0, err
	}
	log.Printf("Rebuilt orient data: %d samples (<= %d per class) -> %s",
		len(out), OrientPerClass, orientFile)
	return len(out), nil
}
