package training

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/ShinteLab/suteme"
)

func TestHandleVersion(t *testing.T) {
	w := httptest.NewRecorder()
	handleVersion(w, httptest.NewRequest(http.MethodGet, "/api/version", nil))
	var v buildVersion
	if err := json.Unmarshal(w.Body.Bytes(), &v); err != nil {
		t.Fatal(err)
	}
	if v.Version != suteme.Version() {
		t.Errorf("version = %q, want %q", v.Version, suteme.Version())
	}
}

// 版は外部公開をオンにしても外へは出さない（/api/status と /api/register 以外はローカル限定）
func TestVersionIsLoopbackOnly(t *testing.T) {
	resetSettings(t)
	settingsMu.Lock()
	settings = APISettings{Enabled: true, External: true}
	settingsMu.Unlock()

	r := httptest.NewRequest(http.MethodGet, "/api/version", nil)
	r.RemoteAddr = "192.168.1.50:5000"
	w := httptest.NewRecorder()
	withAccessControl(http.HandlerFunc(handleVersion)).ServeHTTP(w, r)
	if w.Code == http.StatusOK {
		t.Errorf("code = %d, want not 200", w.Code)
	}
}
