// version は suteme のリリース（バージョンの更新・コミット・タグ打ち）を 1 コマンドにする。
// _cmd/version（このファイルの隣）が唯一の正。
//
//	go run _cmd/version.go           v+現在のバージョンのタグがあれば patch を上げてリリース。無ければ今の値でリリース
//	go run _cmd/version.go -bump     patch / minor / major を対話で選んでリリース（Enter = patch）
//	go run _cmd/version.go 1.2.3     指定したバージョンでリリース
//	go run _cmd/version.go -print    現在のバージョンを表示するだけ（何も変えない）
//
// リリースは次の順に進む。どこかで失敗したら、そこから先はやらない。
//
//  1. go.mod に replace が残っていないか確かめる（残したままタグを打つと、
//     replace と仮の版を持ったモジュールが配られる。AGENTS.md の 2026-10-05 の件）
//  2. go build ./... が通るか確かめる
//  3. _cmd/version を書き換え、それだけをコミットする（作業中の他の変更は巻き込まない）
//  4. HEAD に v+バージョンの軽量タグを打つ（既にあれば動かさない）
//
// push はしない。proxy.golang.org は「無い」を覚えるので、取りに行くのは push を
// 確かめてから（AGENTS.md）。最後に push のコマンドを表示する。
package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
)

const (
	versionFile = "_cmd/version"
	moduleName  = "suteme"
)

const inquiry = `
現在のバージョン: %s

  Enter   -> patch
  1:patch -> %s
  2:minor -> %s
  3:major -> %s
  その他  -> 中止

上げ方を選んでください (1-3)[1]: `

type ver struct {
	major, minor, patch int
}

func parseVer(s string) (ver, error) {
	vals := strings.Split(strings.TrimSpace(s), ".")
	if len(vals) != 3 {
		return ver{}, fmt.Errorf("バージョンの形式が違います（X.Y.Z）: %q", s)
	}
	var n [3]int
	for i, v := range vals {
		x, err := strconv.Atoi(v)
		if err != nil || x < 0 {
			return ver{}, fmt.Errorf("バージョンの形式が違います（X.Y.Z）: %q", s)
		}
		n[i] = x
	}
	return ver{n[0], n[1], n[2]}, nil
}

func (v ver) addMajor() ver  { return ver{v.major + 1, 0, 0} }
func (v ver) addMinor() ver  { return ver{v.major, v.minor + 1, 0} }
func (v ver) addPatch() ver  { return ver{v.major, v.minor, v.patch + 1} }
func (v ver) String() string { return fmt.Sprintf("%d.%d.%d", v.major, v.minor, v.patch) }
func (v ver) tag() string    { return "v" + v.String() }

func main() {
	bump := flag.Bool("bump", false, "patch / minor / major を対話で選んで上げる")
	printVer := flag.Bool("print", false, "現在のバージョンを表示するだけ")
	flag.Usage = func() {
		out := flag.CommandLine.Output()
		fmt.Fprintln(out, "使い方: go run _cmd/version.go [-bump | -print | X.Y.Z]")
		flag.PrintDefaults()
	}
	flag.Parse()
	if flag.NArg() > 1 || (flag.NArg() == 1 && (*bump || *printVer)) || (*bump && *printVer) {
		flag.Usage()
		os.Exit(2)
	}

	if err := run(*bump, *printVer, flag.Arg(0)); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run(bump, printVer bool, arg string) error {
	// パスはリポジトリルートからの相対で扱う。どこから実行しても同じに動くよう先に移る。
	root, err := git("rev-parse", "--show-toplevel")
	if err != nil {
		return err
	}
	if err := os.Chdir(root); err != nil {
		return err
	}

	now, err := readVersion()
	if err != nil {
		return err
	}
	if printVer {
		fmt.Println(now)
		return nil
	}

	next, err := decide(now, bump, arg)
	if err != nil {
		return err
	}
	if next == nil {
		return nil
	}
	fmt.Printf("バージョン: %s -> %s\n", now, next)

	if err := checkNoReplace(); err != nil {
		return err
	}
	fmt.Println("go build ./...")
	if err := runCmd("go", "build", "./..."); err != nil {
		return fmt.Errorf("go build ./... が通りません: %w", err)
	}
	warnDirty()

	if *next != now {
		if err := os.WriteFile(versionFile, []byte(next.String()+"\n"), 0644); err != nil {
			return err
		}
		fmt.Println("書き換え:", versionFile)
	}
	if err := commitVersion(*next); err != nil {
		return err
	}
	if err := tagHead(*next); err != nil {
		return err
	}

	branch, _ := git("rev-parse", "--abbrev-ref", "HEAD")
	fmt.Println()
	fmt.Println("push はしていません。公開するなら:")
	fmt.Printf("  git push origin %s %s\n", branch, next.tag())
	return nil
}

// decide はリリースするバージョンを決める。nil ならやることが無い。
func decide(now ver, bump bool, arg string) (*ver, error) {
	var next ver
	switch {
	case arg != "":
		v, err := parseVer(arg)
		if err != nil {
			return nil, err
		}
		next = v
	case bump:
		v, ok := inquiryVersion(now)
		if !ok {
			fmt.Println("中止しました")
			return nil, nil
		}
		next = v
	default:
		// タグがある = そのバージョンはリリース済みなのに version が上がっていない。
		at, err := tagCommit(now.tag())
		if err != nil {
			return nil, err
		}
		if at == "" {
			next = now
			break
		}
		head, err := git("rev-parse", "HEAD")
		if err != nil {
			return nil, err
		}
		// 直前のリリースのあと何もコミットしていないなら、上げるものが無い。
		if at == head {
			fmt.Printf("%s は HEAD に打ってあります。リリースするものはありません\n", now.tag())
			return nil, nil
		}
		next = now.addPatch()
		fmt.Printf("%s はリリース済み（%s）なので patch を上げます\n", now.tag(), short(at))
	}

	// 上げた先のタグがすでにあるなら、version とタグがずれている。
	// 進めると古いコミットを指すタグが残り、今のコミットにタグが付かないので止める。
	at, err := tagCommit(next.tag())
	if err != nil {
		return nil, err
	}
	if at != "" {
		return nil, fmt.Errorf("タグ %s はすでに %s にあります（%s とタグがずれていないか確かめてください）", next.tag(), short(at), versionFile)
	}
	return &next, nil
}

func inquiryVersion(now ver) (ver, bool) {
	fmt.Printf(inquiry, now, now.addPatch(), now.addMinor(), now.addMajor())
	sc := bufio.NewScanner(os.Stdin)
	sc.Scan()
	switch strings.TrimSpace(sc.Text()) {
	case "", "1":
		return now.addPatch(), true
	case "2":
		return now.addMinor(), true
	case "3":
		return now.addMajor(), true
	}
	return ver{}, false
}

func readVersion() (ver, error) {
	data, err := os.ReadFile(versionFile)
	if err != nil {
		return ver{}, err
	}
	v, err := parseVer(string(data))
	if err != nil {
		return ver{}, fmt.Errorf("%s: %w", versionFile, err)
	}
	return v, nil
}

// checkNoReplace は go.mod に replace が残っていないかを見る。
func checkNoReplace() error {
	out, err := exec.Command("go", "mod", "edit", "-json").Output()
	if err != nil {
		return fmt.Errorf("go mod edit -json: %w", err)
	}
	// Replace が無ければ JSON に "Replace" のキーが出ない。
	if bytes.Contains(out, []byte(`"Replace":`)) {
		return fmt.Errorf("go.mod に replace が残っています。外してからリリースしてください")
	}
	return nil
}

// warnDirty は version 以外の未コミットの変更があれば知らせる（タグのコミットには入らない）。
func warnDirty() {
	// 短い形式の ":!" は続く "_" をマジック文字と読んで失敗するので、長い形式で書く。
	out, err := git("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)"+versionFile)
	if err != nil {
		fmt.Println("注意: 未コミットの変更を確かめられませんでした:", err)
		return
	}
	if out == "" {
		return
	}
	fmt.Println("注意: 未コミットの変更があります。これらはタグのコミットに入りません")
	for _, l := range strings.Split(out, "\n") {
		fmt.Println("  " + l)
	}
}

// commitVersion は version に変更があれば、それだけをコミットする。
// パスを指定した git commit はそのファイルだけが対象なので、ステージ済みの他の変更も巻き込まない。
func commitVersion(v ver) error {
	if err := exec.Command("git", "diff", "--quiet", "HEAD", "--", versionFile).Run(); err == nil {
		fmt.Println("version は HEAD と同じなのでコミットしません")
		return nil
	}
	msg := fmt.Sprintf("chore: %s %s", moduleName, v)
	if err := runCmd("git", "commit", "-m", msg, "-m", "Glory to mankind.", "--", versionFile); err != nil {
		return fmt.Errorf("git commit: %w", err)
	}
	return nil
}

func tagHead(v ver) error {
	if err := runCmd("git", "tag", v.tag()); err != nil {
		return fmt.Errorf("git tag %s: %w", v.tag(), err)
	}
	head, _ := git("rev-parse", "HEAD")
	fmt.Printf("タグ %s を %s に打ちました\n", v.tag(), short(head))
	return nil
}

// tagCommit はタグが指すコミットを返す。タグが無ければ空。
func tagCommit(tag string) (string, error) {
	out, err := exec.Command("git", "rev-parse", "-q", "--verify", "refs/tags/"+tag+"^{commit}").Output()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok && ee.ExitCode() == 1 {
			return "", nil
		}
		return "", fmt.Errorf("git rev-parse %s: %w", tag, err)
	}
	return strings.TrimSpace(string(out)), nil
}

func git(args ...string) (string, error) {
	out, err := exec.Command("git", args...).Output()
	if err != nil {
		return "", fmt.Errorf("git %s: %w", strings.Join(args, " "), err)
	}
	return strings.TrimSpace(string(out)), nil
}

func runCmd(name string, args ...string) error {
	c := exec.Command(name, args...)
	c.Stdout = os.Stdout
	c.Stderr = os.Stderr
	return c.Run()
}

func short(hash string) string {
	if len(hash) > 7 {
		return hash[:7]
	}
	return hash
}
