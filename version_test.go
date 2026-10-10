package suteme

import (
	"os"
	"strings"
	"testing"
)

// Version は _cmd/version の中身そのもの（リリースが書き換えるのはこのファイルだけ）
func TestVersionMatchesFile(t *testing.T) {
	data, err := os.ReadFile("_cmd/version")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := Version(), strings.TrimSpace(string(data)); got != want || got == "" {
		t.Errorf("Version() = %q, want %q", got, want)
	}
}
