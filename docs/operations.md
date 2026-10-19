# サーバーの立ち上げと更新（運用手順）

コマンドはすべてリポジトリのルート（`~/Working/poke-sim`）を基準にしています。ログの置き場所は `/data/poke-sim/` です。
認証情報はリポジトリの外に置き、表示やコミットはしないでください。

| ファイル | 中身 | 使うもの |
|---|---|---|
| `~/.config/poke-sim/showdown.env` | `PS_USERNAME=...` / `PS_PASSWORD=...` | ラダー収集（`scripts/ladder.py`） |
| `~/.config/poke-sim/obs.env` | `OBS_PASSWORD=...`（環境変数 `OBS_PASSWORD` でも可） | 対戦サポートサーバー |

## 全体像

| サーバー | 役割 | ポート / 接続先 | コード更新の反映 |
|---|---|---|---|
| PostgreSQL（docker） | 図鑑・型データ | 127.0.0.1:5433 | — |
| ローカル Showdown サーバー | ローカル対戦（`ladder.py --local`、`smoke_battle.py`） | 8000 | 起動し直し |
| ラダー収集（`scripts/collect.sh` → `scripts/ladder.py`） | 公式 Showdown で対戦してデータを集める | play.pokemonshowdown.com | 自動（対戦の合間に読み込み直す） |
| 対戦サポートサーバー（`pokesim.api.server`） | Switch の対戦を OBS で読み取って助言する | 8100（操作画面）、OBS 4455 | 起動し直し |

先読みの Node ワーカー（`services/sim-tools/*.js`）は、それを使う Python プロセスが起動したときに一緒に立ち上がります。単独で起動する必要はありません。

## PostgreSQL
```bash
docker compose up -d
docker compose ps        # 状態の確認
```

## ローカル Showdown サーバー（ポート 8000）
起動：
```bash
cd external/pokemon-showdown
setsid nohup node pokemon-showdown start --no-security 8000 \
  > /data/poke-sim/server_logs/showdown_8000.log 2>&1 < /dev/null &
ss -tlnp | grep :8000    # LISTEN が出れば起動済み
```
停止：
```bash
pkill -f "pokemon-showdown start --no-security 8000"
```
同じポートで二重に起動しないよう、起動する前に `pgrep -af "pokemon-showdown start"` で確認してください。

### Showdown 本体の更新
```bash
cd external/pokemon-showdown
git log --oneline HEAD..origin/master   # git fetch のあとで、何が入るかを確認
git pull --ff-only
node build
```
ビルドしただけでは、動いているプロセスには反映されません。ローカル Showdown サーバー、ラダー収集、対戦サポートサーバーをそれぞれ起動し直してください。対戦の記録には Showdown のバージョンが `versions.engine` として残ります。

更新の前に、対戦の処理（ダメージ、特性、メガシンカなど）の修正が入っているかを確認してください。入っていなければ、収集の途中で版を混ぜないほうがよいです。

## ラダー収集（公式 Showdown）
`scripts/collect.sh` は、`ladder.py` を20戦ずつ繰り返し実行します。止めるまで続けます。対戦が1つも保存されなかったまとまり（ログインやネットワークの不調）が続いたら、最大30分まで待ってからやり直します。
同じ`COLLECT_DIR`ではロックにより二重起動を拒否します。

起動：
```bash
mkdir -p /data/poke-sim/collect
PLANNER=on setsid nohup ~/Working/poke-sim/scripts/collect.sh > /dev/null 2>&1 < /dev/null &
```
環境変数で設定を変えられます。

| 変数 | 意味 | 既定値 |
|---|---|---|
| `PLANNER` | `off` / `shadow`（従来の方式が指し、プランナーは裏で計算するだけ）/ `on`（プランナーが指す） | `on` |
| `POLICIES` | 使う方策。複数指定すると対戦ごとに交互に使う | `checkpoints/ppo_baton_v2.zip checkpoints/ppo_baton_items.zip` |
| `EXTRA` | `ladder.py` に追加で渡すオプション | なし |
| `PYTHON` | 既存venvのPythonを直接使う場合の実行ファイル。未指定なら`uv run python` | 未指定 |
| `POKESIM_SEARCH_WORKERS` | 1判断で同じ世界を別乱数列で探索するNodeワーカー数（1本≈400MB） | `4` |
| `MIN_FREE_MB` | まとまりの開始前に待つ空きメモリの下限 | `4096` |
| `COLLECT_DIR` | ログと停止ファイルの置き場所 | `/data/poke-sim/collect` |

`EXTRA` の例は `EXTRA="--belief --opponent-policy checkpoints/bc_human_v2b.zip"` です。
選出は既定で `--lineup ladder`（この構築の実戦結果の事後分布＋相手の見せ合いで条件付け、Thompson抽選）。
2026-10-18以降は、相手の見せ合いに対する先発と控えの相性を学習し、仮説の型に対する3体全体の攻守も比較する。
メガシンカは1体に割り当てた場合ごとに評価する。相性補正だけを外した比較は`--no-preview-coverage`。
対戦サポート画面も同じ候補順位を使い、「相性評価」として表示する。表示値は勝率ではない。
検証結果と制限は[積み判断と選出の改善](audits/20261018-selection-tactics.md)を参照。
旧来の模擬対戦表は `--lineup matchup`。`--battle-timeout` は対戦の総時間ではなく、対戦メッセージが途絶えた時間で判定します。
勝敗価値を探索で使う場合は`EXTRA="--value-run runs/terminal-value-v2"`と`PLANNER=on`を指定します。
この指定の現在の既定は、[1判断最大15秒の探索と終局確認](terminal-search.md)です。
参照方策の版を含む検証条件と起動例は[価値探索の接続手順](value-search-runtime.md)を参照してください。

状態の確認：
```bash
tail /data/poke-sim/collect/status.log      # まとまりごとの結果（何戦保存できたか、終了コード）
tail -f /data/poke-sim/collect/ladder.log   # 対戦ごとの結果とエラー
pgrep -af "scripts/ladder.py --battles"     # 動いているか
```

品質停止や選択の不一致を調べるときは、対戦JSONに加えて `data/raw/ladder/diagnostics.jsonl`
を確認してください（`--out` 指定時はその出力先）。要求ID付きの送信試行・送信完了と、
品質停止の対戦タグ・要求ID・例外の詳細を記録します。対戦終了後に届いた要求もこちらに残ります。
対戦JSONのadviceには選択ラベル、変換後の注文、送信したコマンドがあり、実行技と比較できます。

停止（今のまとまりを最後まで戦ってから止まります）：
```bash
touch /data/poke-sim/collect/STOP
tail -f /data/poke-sim/collect/status.log   # "stopped ..." が出たら止まった
```
設定を変えるときは、止めてから環境変数を変えて起動し直します。`collect.sh` は起動時に `STOP` ファイルを消します。

### コードの更新
```bash
git pull --ff-only
```
`ladder.py` は、`src/`・`services/sim-tools/`・`data/processed/*_priors*.json` が変わったことを検知し、対戦と対戦の合間に終了します。`collect.sh` が新しいコードで起動し直すので、操作は要りません。

**収集が動いている間は、本体のフォルダでブランチを切り替えたり、ファイルを編集したりしないでください。** 途中のコードで対戦してしまい、比較のデータに複数の版が混ざります。開発は別の worktree で行ってください（例：`git worktree add ~/Working/wt/<name> -b <branch> origin/master`）。

注意：`/tmp` はメモリ上の領域（15GB）です。worktree をそこに作ると、`uv sync` で `.venv` が丸ごとコピーされ、すぐに満杯になります。worktree は `~/Working/wt/` などディスク上に作ってください。

## 対戦サポートサーバー（Switch + OBS）
OBS の設定：ツール → WebSocket サーバー設定で有効にし、ポート（4455）とパスワードを `obs.env` に書きます。キャプチャボードのソース名（例：`映像キャプチャデバイス (V4L2)`）を `--obs-source` に渡します。

起動（コードを更新したあとも、この手順で起動し直します）：
```bash
pkill -f "pokesim.api.server"               # 動いていれば止める
L=/data/poke-sim/server_logs/server_$(date +%Y%m%d-%H%M).log
setsid nohup uv run python -m pokesim.api.server --policy checkpoints/ppo_baton_v2.zip \
  --obs-source "映像キャプチャデバイス (V4L2)" --obs-record /data/poke-sim/obs_frames \
  > $L 2>&1 < /dev/null &
tail -f $L    # "Uvicorn running on http://127.0.0.1:8100" と出れば起動完了（Ctrl+C で抜ける）
```
操作画面：http://127.0.0.1:8100

主なオプション：

| オプション | 意味 |
|---|---|
| `--planner off\|shadow\|on` | プランナーの使い方（既定 `off`） |
| `--value-run` | 検証済み勝敗価値の実行ディレクトリ。`--planner on`または`shadow`と併用 |
| `--no-lookahead` | 先読みを使わず、方策だけで助言する |
| `--team` | 自分の構築ファイル（既定 `data/my_teams/baton.json`） |
| `--obs-record` | 画面が変わったときにフレームを保存する場所（対戦記録を作り直すのに使う） |
| `--obs-fps` | OBS から画面を取る頻度（既定 4） |

確認：
```bash
ss -tnp | grep 4455                                    # OBS と接続中なら1行出る
ls /data/poke-sim/obs_frames/$(date +%Y%m%d) | wc -l   # 保存された画面の数が増えていくか
ls -t data/raw/live/logs | head -3                     # 対戦が終わるたびに記録が増える
```
OBS を閉じたり起動し直したりしても、サーバーは自動でつなぎ直します。再接続の間隔は3秒から最大30秒です。未接続の間は、操作画面に赤字で「OBS 未接続」と出て、その間は画面も対戦も記録されません。エラーのログは1分に1回だけ出ます。

保存した画面から対戦記録を作り直す方法（読み取りを改善したときなど）：
```bash
uv run python scripts/replay_frames.py /data/poke-sim/obs_frames/<日付> --records data/processed/replayed
```

## 困ったとき

| 症状 | 確認すること |
|---|---|
| ラダーで0戦のまとまりが続く | `ladder.log` に `ReadTimeout` などのログイン・ネットワークのエラーが出ていないか。`showdown.env` は正しいか |
| 操作画面に「OBS 未接続」 | OBS は起動しているか、WebSocket サーバーが有効か、ソース名は合っているか |
| `/tmp` が満杯（`EDQUOT`） | `du -sh /tmp/*` で、`/tmp` に作った worktree の `.venv` が場所を取っていないか |
| 同じサーバーが二重に起動している | `pgrep -af "pokemon-showdown start"` / `pgrep -af pokesim.api.server` で確認し、余分なものを止める |

## 2026-10-17 のデータ消失と復旧

開発worktreeで作った `external`・`checkpoints`・`runs`・`data/raw`・`data/processed`・`data/curated` へのシンボリックリンクが
（`.gitignore` の末尾スラッシュ付きパターンはリンクに一致しないため）コミットされ、本体ツリーの `git pull` で無視対象の
実ディレクトリが自分自身を指すリンクに置き換えられて消えた。`.gitignore` は末尾スラッシュなしに修正済み（PR #28）。

- **worktreeでリンクを作らない。** 本体のデータを使うときは `POKESIM_SHOWDOWN_DIR` と絶対パスの引数を使う。
- 復元できたもの: 公開リプレイ（DB `replays` から）、PokeAPI CSV、smogon統計、エンジン（942cda71 を再クローン）、
  図鑑・各種prior・選出率・チーム群（再生成）、選出の事前証拠（`data/lineup_prior.json`）。
- 失われたもの: ラダー対戦記録、ライブ記録（OBSの保存画面 `/data/poke-sim/obs_frames` から再構成）、全checkpoint、全run。

復旧後の収集はPPO方策なしで始める:
```bash
POLICIES=none PLANNER=on EXTRA="--value-run runs/value-replays-v1 --opponent-policy net:runs/opp-replays-v1/opp-seed0.json" \
  setsid nohup ./scripts/collect.sh > /dev/null 2>&1 < /dev/null &
```
`--policy none` では探索が指し、探索に失敗した判断はランダムな合法手（記録に残る）になる。

## PR #37 の残提案の実装後

選出は静的相性の上位候補を、相手の選出・型4仮説と3種類の公開情報方策で3対3対戦させて比較する。
保護・バトンの修正後は、従来の4候補にバトン役先発の全編成を追加し、現行6体で最大24候補を比較する。
準備に使うターンとHPも含め、受け先が強化された後の終局結果で選ぶ。詳細は[履歴とバトンの監査](audits/20261018-relay-tempo.md)。
対戦中の通常設定は `rootProgress=false`、許容幅0.03固定。上位2手への集中探索と複数ターン計画は比較用に指定できる。
学習候補は較正条件を満たすまで通常ロードしない。手順と検証は[実装監査](audits/20261018-proposals37.md)を参照。

ラダーは提案・送信・実行を分け、自分側requestから実際の選出を保存する。
操作主体の既定はunknown。人が介入できない収集であると確認している場合に限り、`EXTRA`へ
`--control-mode autonomous_ai` を追加する。提案と一致したという理由だけではAI単独の実行と集計しない。

## 公開構築の型推定と追加時間での終局確認

通常の価値探索は1手最大15秒。通常探索の8.5秒・8ターンを維持し、余剰時間で公開情報方策の終局比較を行う。
各仮説と相手方策で悪化せず平均が改善する候補だけ採用し、打切り・時間切れでは通常探索の手を使う。
Showdownが通知した持ち時間が少ない場合は短縮する。
比較用に旧設定へ戻すには `--search '{"decisionMs":8500,"maxTurns":8,"terminalReview":false}'` を指定する。

`data/processed/top_teams.json` がある場合、GameWith上位構築の検証済みの型・6体を、選出と実戦中の仮説へ使う。
更新は開発worktreeで次を実行し、保存先の内容・validator結果を確認して、対戦の合間に原子的に配置する。

```sh
uv run python -m pokesim.data.team_catalog \
  --cache data/raw/team-catalog-20261018 --out data/processed/top_teams.json
```

実行中の自動ダウンロードはない。catalogのhashは記録・較正・選出cacheへ入り、変更は対戦間に検知する。
再学習したNN候補は検証で採用条件を満たさなかったため、現在のNN重みは維持する。
原本の分割、構築取得、時間延長の比較、再学習の採否は[今回の監査](audits/20261018-team-priors-refresh.md)を参照。

## 直近の反撃確認と選出の対策（2026-10-19）

通常探索と終局再検討の選択を、同じ型仮説の全合法な相手初手と3乱数列で確認する。
安全な積み・バトンは残し、瀕死交換や、倒れる際の削りで劣る候補だけを修正する。
最大15秒の予算を共有し、比較が未完了ならその比較では変更しない。
比較時に無効化する設定は`--search '{"rootCounterplay":false}'`。

選出は全20構成を少なくとも1つの先発で比較する。通常4仮説と、対策技・物理積みを含む最大2仮説、
相手の3方策を使い、各仮説の最も厳しい方策も評価する。仮説の型は実際の相手の型の断定ではない。
バトン役先発の候補も維持する。実装条件・急所を含む敗因説明の訂正・検証の制限は
[反撃と選出の監査](audits/20261019-counterplay.md)を参照。

## 受け手の生存条件に応じたバトン（2026-10-19）

expert継続はpublic-baton-v3、計画方策はpublic-plans-v3。鉄壁の必要量、晴れの残り、
回復と瞑想、呪いの解除、2段階のバトンを公開情報から再評価し、選出の模擬戦にも使う。
ルートでは従来の効用差0.03以内で計画を採用し、反撃確認・勝利証明を優先する。
`relay_plan.strategy`に根拠を記録する。NN重み・参照方策v2は維持。
判断条件、固定局面と選出の検証、未解決の局面は[今回の監査](audits/20261019-relay-goals.md)を参照。

## 集合NN候補と実戦評価の分離（2026-10-19）

探索の既定効用は`reference-material-v2`。能力ランクへの一律加点を除き、選出は実戦の事後成績も
最終スコアへ残す`preview-evidence-v5`を使う。Struggleの合法手変換も修正した。
新しい集合NNと、対応付き終局比較による行動・継続価値の学習は実行したが、採用条件は満たしていない。
収集は従来と同じ`runs/value-replays-v1` / `runs/opp-replays-v1/opp-seed0.json`を使用する。
候補の失敗を理由に品質ゲートを緩めたり、新形式の未採用重みを通常ロードしたりしない。
再現と未見期間の比較は[集合NNと学習経路の監査](audits/20261019-structured-learning.md)を参照。
