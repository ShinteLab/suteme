package suteme

import (
	"os"
	"path/filepath"
	"sync"
)

// 向きの回転照合だけを担う小さなデータ（配布用）。
//
// **駒種を NN（`model_v7.json`）で読む構成のための穴埋め。**
// 向きは `OrientationMatcher`（＝駒サンプルへの最近傍距離）で決めるが、
// **これを実装しているのは `*KNN` だけ**で、gobrain の `*Model` は持たない。
// 実装が無ければ分類器の幅プロファイルに落ちるので、実測（30局面・
// その局面を学習済みの条件・手動座標）で向きの反転が **0 → 69 件**、
// 全マス一致が 98.8% → **88.8%** まで落ちていた。
// **向きを外すと駒種も必ず外れる**（後手の駒は `Rotate180` してから
// 推論するため）ので、駒種違いも 27 → 189 件に増える。
//
// 一方で照合に要るのは「駒サンプルへの最近傍距離」だけなので、
// **学習データ全件（114MB）は要らない**。クラスごとに間引いたものがあれば足りる。
// 作るのは `training.BuildOrientData`、使うのはここ。
//
// **形式は学習データと同じ**（`SaveTrainingData` / `LoadTrainingData`）。
// 中身が「駒サンプルの部分集合」そのものなので、別の形式にする理由が無い。
// 取り違えないようファイル名で分ける。
const DefaultOrientFile = "orient_data_v1.bin"

var (
	orientOnce sync.Once
	orientVal  OrientationMatcher
	orientMu   sync.RWMutex
	// orientSet は SetOrientMatcher が呼ばれたか。
	// 呼ばれるまでの間だけファイルを自動で探す
	orientSet bool
)

// SetOrientMatcher は向きの回転照合に使う照合器を明示指定する。
//
// **nil は「照合器を使わない」＝向きを分類器に任せる。自動探索に戻すのは
// `ResetOrientMatcher`。** `SetStripJudge` と同じ約束にしてある
// （`SetPredictor(nil)` が自動探索に戻るのとは逆なので注意）。
// 外して測る用途が主なので、nil を自動探索にすると
// 「外したつもりがファイルを拾っていた」が起きる。
func SetOrientMatcher(om OrientationMatcher) {
	orientMu.Lock()
	orientVal, orientSet = om, true
	orientMu.Unlock()
}

// ResetOrientMatcher はファイルからの自動探索に戻す。
func ResetOrientMatcher() {
	orientMu.Lock()
	orientVal, orientSet = nil, false
	orientMu.Unlock()
}

// defaultOrientMatcher はカレント→実行ファイルのディレクトリの順に探す。
//
// **見つからなければ nil**（そのときは向きが分類器に落ちるだけで、認識は動く）。
// **推論器が `OrientationMatcher` を実装しているなら、そちらが優先で
// ここは呼ばれない**（k-NN を使う構成では読み込みも起きない）。
func defaultOrientMatcher() OrientationMatcher {
	orientMu.RLock()
	if orientSet {
		om := orientVal
		orientMu.RUnlock()
		return om
	}
	orientMu.RUnlock()

	orientOnce.Do(func() {
		dirs := []string{"."}
		if exe, err := os.Executable(); err == nil {
			dirs = append(dirs, filepath.Dir(exe))
		}
		for _, d := range dirs {
			p := filepath.Join(d, DefaultOrientFile)
			data, err := LoadTrainingData(p)
			if err != nil {
				continue
			}
			kn := NewKNN(data.Samples)
			if kn == nil {
				continue
			}
			kn.source = p
			orientMu.Lock()
			orientVal = kn
			orientMu.Unlock()
			return
		}
	})
	orientMu.RLock()
	defer orientMu.RUnlock()
	return orientVal
}
