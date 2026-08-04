# suteme

将棋盤の画像から盤面状態を解析して SFEN 形式で出力する Pure Go プロジェクト。

## モジュール / 位置づけ

独立した Go モジュール `github.com/ShinteLab/suteme`。
import パスは `github.com/ShinteLab/suteme`, `github.com/ShinteLab/suteme/training`。
`core` はタグ未発行のため `replace github.com/ShinteLab/core => ../core` の相対パス参照で引いている
（この replace を外さないこと）。プロジェクト横断の方針は親ディレクトリの `CLAUDE.md` を参照。

依存:

- `github.com/goml/gobrain` — 駒種認識の NN
- `github.com/ShinteLab/core/sfen`, `github.com/ShinteLab/core/usi` — **SFEN / USI の仕様。自前実装は持たない**
  （`board.go` / `komadai.go` / `recognize.go` から利用）
- `github.com/ShinteLab/core/web` — 共有フロント資産。`training/server.go` が `embed.FS` を配信（後述）

SFEN の駒文字マッピングや盤面文字列の組み立てを suteme 側に書き足さないこと。
必要なら `core/sfen` に足して、Go 側テストと `core/web/test.mjs` の両方を揃える。

---

## ファイル構成

| ファイル | 役割 |
|---------|------|
| `suteme.go` | 公開API: `LoadSFEN` / `LoadSFENWith` / `LoadPredictor` / `SetPredictor`・`ViewDebug` |
| `analyze.go` | `Analyze()` / `AnalyzeResult` / `DrawBoard()` |
| `detect.go` | **盤面検出**: エッジ投影 → ピーク検出 → 9x9分割 → `BoardRegion` / `BoardRegionFromRect` |
| `classify.go` | **空/先手/後手の分類**: 画像処理のみ（学習不要）・`BoardColor` |
| `validate.go` | 盤面検出の妥当性チェック（背景色の均一性）|
| `recognize.go` | **駒種認識(NN推論)**: gobrain `Model` 読み込み/推論・`CellToInput`・学習データ型 |
| `knn.go` | **駒種認識(k-NN)**: 学習不要・距離重み付き投票。現在の主認識器。`Predictor` IF |
| `training/training.go` | **学習パッケージ**: `Train` / `BalanceData` / `ClassDistribution` / `SaveModel` |
| `training/server.go` | **学習用Webサーバ**: `Serve(port)`・APIハンドラ・履歴管理・UI embed |
| `_cmd/suteme-training/` | 学習用Webサーバの起動コマンド（`training.Serve` を呼ぶだけ）|
| `komadai.go` | **駒台推定**: `ValidatePieces` / `CountFromSFEN`（盤面 + 駒台 = 全駒 検証）|
| `imaging.go` | グレースケール変換・二値化・BoxBlur・Sobel・`Rotate180` |
| `draw.go` | Bresenham 線描画・PNG 保存 |
| `core.go` | `Line` / `Rect` 型・幾何ユーティリティ |
| `board.go` | `Board` / `BoardRow` / `BoardSquare` データモデル・SFEN出力 |
| `piece.go` | `Piece` / `PieceType` データモデル |
| `hough.go` | Hough 変換（現在未使用、残置）|
| `grid.go` | グリッド検出（現在未使用、残置）|

---

## アーキテクチャ

### 盤面検出（`detect.go`）

```
入力画像
  ↓ グレースケール → BoxBlur(r=2) → Sobel エッジ検出
  ↓
Pass 1: 横方向投影（全幅）→ ピーク検出 → 等間隔8本選択（内部グリッド線）
  ↓ hSpacing 取得（マス間隔）
Pass 2: 行範囲内の縦方向投影 → 正方形制約でピーク境界ペア探索
  ↓
extendToBorders: 8本 → 上下1マス延長 → 10本（盤面外枠含む）
  ↓
9x9 = 81マスの Rectangle を生成 → BoardRegion
```

**正方形制約**: 将棋盤は正方形なので `hSpacing * 9 ≈ boardWidth`。
これにより TV 中継画像のサイドバー等のノイズを排除する。

**手動指定**: `BoardRegionFromRect(x1, y1, x2, y2)` で2点から直接 `BoardRegion` を生成可能。
UI からドラッグで盤面領域を指定する場合に使用。

**セルパディング**: `ExtractCell` は各マスを 15% 外側に拡張して抽出する。
グリッド検出の誤差でグリッド線が入り込む場合でも駒全体を捉えるため。

### 駒分類（`classify.go`）

学習データ不要の純粋な画像処理:

```
BoardColor: 盤面領域の全画素の中央値 = 盤の地色（画像ごとに1回）
  ↓
ClassifyCellWith(cell, 地色)
  パディング相当の縁を除いた内側だけで判定（ExtractCell の15%拡張に対応）
  地色から ±12%（最低±12）離れた画素を「駒」としてマスク化
  ↓ ほぼ全高を占める列・ほぼ全幅を占める行（>90%）をグリッド線／盤外として除外
  被覆率 < 15% → 空
  ↓ 行ごとの駒の幅プロファイルを取り、駒の外接範囲を求める
  幅で重み付けした重心が外接範囲の上寄り → 底辺が上 → 後手（☖）
                              下寄り → 底辺が下 → 先手（☗）
```

設計上の要点（いずれも実測で精度が変わった箇所。安易に戻さないこと）:

- **空判定を分散の絶対値でやらない。** 旧実装は `分散 < 1500` だったが、分散は
  画像の解像度・コントラスト・盤外領域の映り込みで桁が変わる。実際に
  「駒の分散が全マス 1500 未満で81マス全部が空」「空マスの分散が 1500 超で
  28マスが駒」という両方向の破綻が起きていた。地色基準の被覆率はスケール不変。
- **判定の基準色はマスの平均ではなく盤の地色。** 平均は駒・墨・盤外を含むので
  マスの中身に応じて参照点が動き、「背景と異なる」の意味が保てない。
- **向きは外接範囲内での重心で決める（マス内の絶対位置を見ない）。** 盤面領域は
  数 px ずれるのが普通で、旧実装の「上1/5 と 下1/5 の比較」は 1マスの 13% の
  ズレで反転していた。外接範囲基準なら平行移動に影響されない。
- **グリッド線は行・列単位で落とす。列を先に処理する。** 縦線を残したまま行を見ると
  全行が「ほぼ全幅が埋まっている」= グリッド線に見えてしまう。

精度は `TestClassifyAccuracy`（`classify_data_test.go`）で `data/` の保存済み局面を
正解として測れる。`go test -run TestClassifyAccuracy -v`。

### パッケージ分割の方針

- **`suteme`（コア）**: 「画像データを元に棋譜データを作成する」推論側。
  盤面検出・分類・推論（NN/k-NN）・学習データ型とJSON入出力・`LoadModel` を持つ
- **`suteme/training`**: モデルを「作る」側。NN訓練・バランシング・`SaveModel`。
  依存方向は training → suteme の一方向のみ

### 駒種認識 NN（`recognize.go` + `training/`）

gobrain の FeedForward NN を使用:

- 入力: 各マスを 24×24 グレースケールにリサイズ → 576 float64 → **平均0・分散1に標準化**（明るさ・配色差を吸収）
- クラス数: **14**（駒種のみ、向きなし: 歩/香/桂/銀/金/角/飛/玉/と/杏/圭/全/馬/龍）
- 向きは `ClassifyCellWith` が担当（NN 不使用）
- **向きの正規化**: 後手の駒は `Rotate180` で先手向きにしてから NN に入力（学習・推論とも）
- モデル保存: `model_v2.json`（JSON シリアライズ）
- 学習データ: `training_data_v2.json`（累積保存）
- 訓練パラメータ: 300 epochs / lr=0.2 / momentum=0.5、**学習前にパターンをシャッフル**（クラス順のままだと逐次学習が偏る）

**重複除去** (`MergeSamples`): `/api/train` は毎回「既存ファイルの全件 + メモリ上の
全セッションのラベル」を保存し直すため、素通しだと学習を押すたびに同じサンプルが
積み上がる（実際に 1173 件中 838 件＝71% が重複していた）。入力ベクトルの内容を
キーにマージして畳む。**後勝ち**なので、同じマスにラベルを付け直した場合は訂正が反映される。
同じ画像の同じマスからは同じ入力が得られることを利用しており、セッション ID の
管理は要らない。既存ファイルも通すので、古い重複は次の学習で自動的に解消される。

**クラスバランシング** (`BalanceData`): 中央値の3倍を上限に多数クラスをダウンサンプル、
中央値未満の少数クラスは複製でオーバーサンプル。
（最少クラス基準だと1枚しかない駒種があるだけで全データが捨てられるため中央値基準）

**注意（学習データの汚染防止）**: `LabelToClass` は不明ラベル（"none" 等）で **-1 を返す**。
学習時は class < 0 のサンプルをスキップすること。
過去にこのチェックがなく、空マスが「歩」（class 0）として大量に混入し精度が崩壊した。
旧 `training_data.json` / `model.json` はこの汚染と旧入力表現のため使用しない（v2 ファイルに移行済み）。

### 駒種認識 k-NN（`knn.go`）— 現在の主認識器

ゲーム画面の駒は毎回ほぼ同一ピクセルで描画されるため、汎化より既知パターンとの照合が
重要であり、小データでは NN より k-NN が安定する。

- `Predictor` インターフェース（`Predict(cell) (class, conf)`）で NN と差し替え可能
- 距離: 標準化済み 576 次元ベクトルのユークリッド距離（2乗）
- k=5（サンプル数未満なら全件）、距離の逆数で重み付き投票
- 信頼度 = 勝者クラスの重み比率
- **学習処理が不要**: `training_data_v2.json` から起動時に構築、/api/train 後に再構築
- サーバは k-NN 優先で推論し、gobrain モデルもあれば /api/recognize で比較用 SFEN（`sfen_nn`）も返す

### 駒台推定（`komadai.go`）

```
CountFromSFEN: SFEN盤面部分 → 先手/後手の駒数マップ
ValidatePieces: 各駒種の上限 − 盤面合計 = 駒台枚数
  超過している場合は Warnings に追加
```

駒の総数（盤面 + 駒台）は全局面で一定なので、盤面認識の誤りの検出に活用できる。

### 認識フロー（ハイブリッド方式）

```
BoardColor で盤の地色を求める（画像ごとに1回）
  ↓
各マス画像
  ↓ ClassifyCellWith(cell, 地色)
  空 → SFEN に空カウント追加
  先手/後手 → gobrain で駒種推論 → 向きに応じて大文字/小文字変換
  ↓
SFEN 文字列を生成 → ValidatePieces で駒数検証
```

### 公開 API（`suteme.go`）

`LoadSFEN(img) (string, error)` が「画像 → SFEN 盤面文字列」の入口。
**返すのは盤面部分（'/' 区切りの9段）だけ**で、手番・持ち駒・手数は付けない
（suteme の責務を「画像 → 盤面」に閉じるため。持ち駒枚数が要るなら
戻り値を `ValidatePieces` に渡す）。

```
LoadSFEN(img)
  ↓ defaultPredictor: カレントディレクトリ → 実行ファイルのディレクトリの順に探索
  │   training_data_v2.json → k-NN（優先）/ model_v2.json → gobrain
  │   ※ 見つかった結果はキャッシュ。失敗はキャッシュしない（後からファイルを置けば拾う）
  ↓ detectBoardRegion: DetectBoard → 信頼度不足なら「画像全体が盤面」で再評価
  │   ※ ikkyoku のガイド枠のように盤だけを切り出した画像はグリッド線が
  │      画像端に来て DetectBoard が外れるため
  │   ValidateBoard の値が minBoardConfidence(0.5) 未満なら ErrBoardNotFound
  ↓ RecognizeBoard
SFEN 盤面文字列
```

- `LoadSFENWith(img, Predictor)` … 推論器を明示指定（テスト・比較用）
- `LoadPredictor(dir)` … 指定ディレクトリから推論器を読む
- `SetPredictor(p)` … 既定の推論器を差し替え（nil で自動探索に戻る）
- エラーは `errors.Is` で判別する（`ErrBoardNotFound` / `ErrNoPredictor`）

**認識精度は盤面検出と学習データに依存する。** `DetectBoard` の行間隔が不均一で
グリッド線がマス内に入り込むため誤認識が出る。盤面座標が分かっているなら
`BoardRegionFromRect` + `RecognizeBoard` を直接使うほうが確実。
分類（空/先手/後手）自体はグリッド線の混入や数 px のズレに耐えるようにしてあるが、
1マス分ずれるほどの検出ミスはさすがに救えない。

---

## ラベリング・学習用 Web サーバ（`training/server.go` + `_cmd/suteme-training/`）

サーバ本体（ハンドラ・セッション管理・履歴・静的ファイル embed）は `training` パッケージの
`Serve(port)` として実装されている。`_cmd/suteme-training` はそれを起動するだけのコマンド。
UI は `training/static/index.html`（embed）。

### 共有フロント資産（`@shinte/web`）

盤の仕様（SFEN 処理・盤描画）は `core/web` に一本化している。`server.go` は
`github.com/ShinteLab/core/web`（`assets.go` の `embed.FS`）を `/shinte-web/` で配信し、`index.html` は
`import ... from "/shinte-web/sfen.js"` 等で利用する（バンドラ無し・ファイルのコピー不要）。

- `buildSFEN()` … `ShinteWeb.sfen.formatBoard` に委譲（空マス圧縮・段区切り）
- `parseSFEN()` … `ShinteWeb.sfen.toGrid` に委譲（不正 SFEN は空配列）
- 認識結果 / SFEN 一括入力時に `<shogi-board>` で盤プレビューを表示（`updateBoardPreview`）
- ラベリング/訂正の操作 UI（写真セルのクリック・分類ダイアログ等）は suteme 独自のまま

### 起動

```
go run ./_cmd/suteme-training/ 8888
```

起動時に `model_v2.json` があれば自動ロード。

### API エンドポイント

| エンドポイント | 機能 |
|--------------|------|
| `POST /api/analyze` | 画像アップロード → 盤面検出 + 分類 + 推論候補（モデルあり時）|
| `GET /api/images/:id/:type` | `original` / `edges` / `board` 画像取得 |
| `GET /api/cells/:id/:row/:col` | マス画像取得 |
| `POST /api/trainhistory` | **学習**（選択した保存済み局面: `{ ids: [履歴ID] }`）|
| `POST /api/recognize` | 推論（空/向きは ClassifyCellWith、駒種は gobrain）→ 駒数検証付き |
| `POST /api/setboard` | 手動盤面指定（`{ session, x1, y1, x2, y2 }`）→ 分類 + 推論候補 |
| `GET /api/history` | 保存済み局面一覧取得 |
| `GET /api/history/:id/image` | 保存済み局面の画像取得 |
| `POST /api/savesession` | 現在のセッション（画像 + SFEN + 盤面座標）を保存 |

### UI 構成（タブ）

**責務を分けてある。解析タブは「ラベリングして保存する」だけ、学習は履歴タブから行う。**
以前は解析タブにも「学習」ボタンがあったが、押すとメモリ上の**全セッション**のラベルを
拾う（＝そのセッション中に開いた他の画像も混ざる）ため、何が学習対象なのか画面から
分からなかった。学習の入力は「保存済みの局面」だけに一本化した。
これに伴いサーバはマスごとのラベルを保持しない（`/api/label` / `/api/labelbulk` は廃止、
`session.Labels` も削除）。ラベルは画面側の `boardLabels` だけが持ち、
保存時に SFEN になってサーバへ渡る。

- **解析タブ**: 画像アップロード → ラベリング → 認識/保存
- **履歴タブ**: 保存済み局面の一覧（サムネイル付き）。**チェックを付けて「選択した N 件で学習」**で
  そのまま学習できる（解析タブに読み込む必要はない）。行クリックは「再入力」で解析タブへ遷移。
  学習に使った／読み込んだ履歴には緑「○」マークが付く（セッション内のみ、永続化なし）

### ラベリング UI の操作フロー

1. 画像アップロード（またはCtrl+Vで貼り付け）→ 盤面検出（信頼度 % 表示）
2. 各マスを自動分類（空=紺、☗先手=緑、☖後手=赤）
3. **モデルがあれば全マスを自動推論** → `☗歩?` のように水色の候補を表示（未確定）
4. 盤面がズレている場合: 画像上でドラッグ → 手動で盤面範囲を指定
5. マス画像をクリック → ダイアログ（推論候補があれば駒種が事前選択され信頼度も表示）
   - **「○ 合ってる」**: 分類＋候補駒種で即確定（候補なし時は向きのみ確定）
   - **「空」選択**: 即確定・ダイアログ閉じる
   - **駒種選択 → 「修正」**: 変更後に確定
6. 確定済みマス: 黄色「○」マーク + labeled 表示（背景色はカテゴリ色を維持: 空=紺/先手=緑/後手=赤）
6a. **「残りを確定」**: 未入力のマスを自動分類＋推論候補でまとめて確定する（`confirmRemaining`）。
   推論候補をいったん全部受け入れて、間違っているマスだけクリックして直す流れ。
   入力済みのマスは触らないので、直した後にもう一度押しても上書きされない。
   駒種が決まらないマス（モデル未学習など）は誤ラベルになるので確定せず、件数を警告する。
   **なお空マスの確定は保存される SFEN を変えない**（`buildSFEN` は未入力でも
   自動分類が空なら空として扱い、空マスは学習サンプルにもならない）。
   手数が減るのは駒のマスのほう
7. **SFEN入力**: SFEN文字列を貼り付けて「SFENから入力」→ 81マス一括ラベル設定（画面側のみ）
8. **「保存」**: **画面のラベル入力からのみ** SFEN を生成して保存する（`buildSFEN`、
   未入力マスは自動分類が空なら空扱い、駒種未入力が残っていれば中断）→ 画像 + SFEN +
   盤面座標を `data/` に保存。保存後は保存内容を盤プレビューと文字列で表示する。
   **SFEN 欄は一括入力の手段であって保存の入力源ではない。** 以前は「SFEN欄が空でなければ
   その文字列をそのまま保存」していたため、履歴復元で入った SFEN が欄に残ったまま
   別画像を貼って保存すると、画像と対応しない SFEN が正解データとして記録された
   （実際に1件混入して削除した）。新しい画像を読むと SFEN 欄と保存ボタンはリセットされる
9. **学習は履歴タブで行う**（解析タブに学習ボタンは無い）
10. **「認識」**: モデルで推論 → 読み取り専用モードに切り替え → SFEN + 駒数表示

### 履歴・再入力・履歴からの学習

- 「保存」で局面を保存 → `data/history.json` に記録
- **履歴からの学習（`/api/trainhistory`）**: 履歴には画像・盤面座標・正解SFENが揃っているので、
  解析タブに読み込まずにサーバ側だけで学習データを組み立てられる（`samplesFromHistory`）。
  SFEN の解析は `core/sfen.ParseBoard` に委譲し、駒文字の対応表は持たない。
  後手は `Rotate180` で先手向きに正規化する（推論時と揃える）。空マスは学習対象外。
  **モデルを作り直すなら「全選択」→「学習」の 1 操作。** ゼロからなら先に
  `training_data_v2.json` を消す（消さなくても重複は `MergeSamples` が畳む）
- 履歴行クリック → 保存した画像・盤面座標・SFENを復元して再ラベリング可能（内容を直したいとき）
- 手動指定した盤面座標も復元される

### データの永続化

- `training_data_v2.json`: 学習データ累積（毎回マージ・`MergeSamples` で重複除去）
- `model_v2.json`: 学習済み FeedForward NN（起動時自動ロード）
- `data/{id}.png`: 保存した局面画像
- `data/history.json`: 保存した局面の一覧（SFEN・盤面座標含む、最大100件）

---

## 対応画像タイプ

| 画像タイプ | 盤面検出 | マス分割 | 備考 |
|-----------|---------|---------|------|
| ゲーム画面（真正面）| ◎ | ◎ | 推奨 |
| TV 中継画像 | ○ | △ | 1行ズレる場合あり |
| 写真（斜め撮影）| △ | × | 射影変換が必要（未実装）|

---

## 今後の課題

- [ ] 写真（斜め撮影）対応 → 射影変換（Homography）
- [x] 学習データの重複排除（`MergeSamples` による入力内容ベースのマージ）
- [ ] 持ち駒の認識（駒台の画像解析）
- [x] `LoadSFEN` の本実装（`suteme.go`。盤面文字列のみを返す）
- [ ] `DetectBoard` の行間隔を均一化する（検出したピーク位置をそのまま使うため
      1マスの高さが 47〜58px とばらつき、グリッド線がマス内に入り込む）
- [ ] TV 中継画像のグリッドズレ自動修正
- [x] 学習を履歴タブに一本化（解析タブの「学習」とサーバ側ラベル保持を廃止）
