package suteme

import (
	_ "embed"
	"strings"
)

// version は suteme の版。リリース（go run _cmd/version.go）がこのファイルを書き換えて
// コミットし、そのコミットに v+版 のタグを打つので、タグの suteme ではタグと同じ値になる。
//
//go:embed _cmd/version
var version string

// Version は suteme の版（"0.2.7" など。先頭に v は付けない）を返す。
// リリースとリリースの間のコミットでは、直前にリリースした版のまま。
func Version() string { return strings.TrimSpace(version) }
