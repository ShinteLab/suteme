package training

import (
	"log/slog"
	"sync/atomic"
)

// pkgLogger は SetLogger で差し込まれた Logger。未設定（nil）なら slog.Default() を使う。
//
// **ライブラリはログの設定を持たない。** レベルや出力先は呼ぶ側（アプリ）が
// 決めた *slog.Logger を渡して決める。ここから slog.SetDefault は呼ばない。
// ハンドラがパッケージ関数で状態もパッケージ変数なので、構造体への注入ではなく
// パッケージ変数 1 つで受ける。
var pkgLogger atomic.Pointer[slog.Logger]

// SetLogger は training が使う Logger を差し替える。nil を渡すと既定（slog.Default()）に戻る。
func SetLogger(l *slog.Logger) { pkgLogger.Store(l) }

// logger は今使う Logger を返す。
//
// ⚠️ slog.Default() は**呼ぶたびに引く**こと（init で控えない）。
// アプリが後から slog.SetDefault したときに追従させるため。
func logger() *slog.Logger {
	if l := pkgLogger.Load(); l != nil {
		return l
	}
	return slog.Default()
}
