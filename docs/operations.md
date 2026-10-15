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
| `COLLECT_DIR` | ログと停止ファイルの置き場所 | `/data/poke-sim/collect` |

`EXTRA` の例は `EXTRA="--belief --opponent-policy checkpoints/bc_human_v2b.zip"` です。
勝敗価値を探索で使う場合は`EXTRA="--value-run runs/terminal-value-v2"`と`PLANNER=on`を指定します。
参照方策の版を含む検証条件と起動例は[価値探索の接続手順](value-search-runtime.md)を参照してください。

状態の確認：
```bash
tail /data/poke-sim/collect/status.log      # まとまりごとの結果（何戦保存できたか、終了コード）
tail -f /data/poke-sim/collect/ladder.log   # 対戦ごとの結果とエラー
pgrep -af "scripts/ladder.py --battles"     # 動いているか
```
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
