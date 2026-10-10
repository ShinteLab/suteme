package training

import (
	"encoding/json"
	"net/http"
	"runtime/debug"

	"github.com/ShinteLab/suteme"
)

// buildVersion は画面に出す suteme の版。
// 版は suteme.Version（_cmd/version を埋め込んだもの）。リリースの間のコミットでも
// 同じ値になるので、ビルド情報にコミットがあれば添える（go build / go install が付ける）。
type buildVersion struct {
	Version  string `json:"version"`
	Commit   string `json:"commit,omitempty"`
	Modified bool   `json:"modified,omitempty"`
}

func currentVersion() buildVersion {
	v := buildVersion{Version: suteme.Version()}
	if bi, ok := debug.ReadBuildInfo(); ok {
		for _, s := range bi.Settings {
			switch s.Key {
			case "vcs.revision":
				v.Commit = s.Value
				if len(v.Commit) > 7 {
					v.Commit = v.Commit[:7]
				}
			case "vcs.modified":
				v.Modified = s.Value == "true"
			}
		}
	}
	return v
}

// handleVersion は suteme の版を返す（ループバック限定。/api/status と違い外には出さない）
func handleVersion(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(currentVersion())
}
