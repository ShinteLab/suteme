package training

import (
	"bytes"
	"log/slog"
	"strings"
	"testing"
)

// SetLogger で渡した Logger に出ること、nil で slog.Default() に戻ること、
// そして既定は呼ぶたびに引く（後から SetDefault しても追従する）ことを確かめる。
func TestSetLogger(t *testing.T) {
	t.Cleanup(func() { SetLogger(nil) })

	var buf bytes.Buffer
	SetLogger(slog.New(slog.NewTextHandler(&buf, nil)))
	logger().Info("hello", "k", 1)
	if !strings.Contains(buf.String(), "msg=hello k=1") {
		t.Fatalf("差し込んだ Logger に出ていない: %q", buf.String())
	}

	SetLogger(nil)
	old := slog.Default()
	t.Cleanup(func() { slog.SetDefault(old) })
	var def bytes.Buffer
	slog.SetDefault(slog.New(slog.NewTextHandler(&def, nil)))
	logger().Info("fallback")
	if !strings.Contains(def.String(), "msg=fallback") {
		t.Fatalf("nil のあと slog.Default() に戻っていない: %q", def.String())
	}
}
