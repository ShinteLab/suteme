package training

import (
	"os"
	"testing"
)

// 重いテストの扱い。
//
// **合否を判定しないテストを `go test ./...` で走らせない。**
// `data/` の局面が増えるほど費用が線形に伸びる一方、これらは
// **t.Error を 1 つも持たない＝絶対に落ちない**。つまり
// 既定で回しても回帰は 1 件も捕まえられないのに、
// 局面が 188 件になった時点でパッケージ全体が標準の 600 秒に
// 収まらなくなり、**回帰テストごと巻き添えで落ちるようになった**
// （タイムアウトはパッケージ単位なので、軽いテストの結果まで消える）。
//
// 判定を持つ重いテスト（`TestShiftRobustness` / `TestUnslipHoldout` /
// ルートの `TestDetectBoardMatchesManual` など）は**既定のまま**にする。
// あちらは落ちることがあるので、走らせない意味が無い。
//
// 測るときは:
//
//	go test -timeout 3h -run TestKNNHoldout -v ./training   # 1 本だけ
//	$env:SUTEME_SLOW=1; go test -timeout 6h ./training      # 全部
//
// **`SUTEME_SLOW` はファイルを書き換えるものには効かせない。**
// `SUTEME_REBUILD`（学習データの作り直し）と各 `*_DUMP` 系は
// 別の名前のまま＝「うっかり全部走らせた」で資産が変わらないようにする。
func requireSlow(t *testing.T, also ...string) {
	t.Helper()
	if os.Getenv("SUTEME_SLOW") != "" {
		return
	}
	for _, n := range also {
		if os.Getenv(n) != "" {
			return
		}
	}
	t.Skip("計測用（合否判定なし）。SUTEME_SLOW=1 で実行する")
}
