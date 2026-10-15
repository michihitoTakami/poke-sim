# poke-sim

Pokémon Champions シングル（3v3, `gen9championsbssregmc`）の対戦サポートツール。
計画の全体像は `docs/plan.md` を参照。
対人戦の課題に対する改善設計は [対人戦の判断品質を改善する実装設計](docs/competitive-improvement-design.md) を参照。
PR #7と10/15の実戦結果を踏まえた次の実装順・価値学習・採用基準は [勝敗を目的に改善する次期設計](docs/winning-ai-roadmap.md) を参照。
候補手を固定して終局まで比較するコマンドは [終局分岐評価の実行手順](docs/terminal-branch-evaluation.md) を参照。

## セットアップ
```bash
git clone --depth 1 https://github.com/smogon/pokemon-showdown.git external/pokemon-showdown
(cd external/pokemon-showdown && cp config/config-example.js config/config.js && npm ci --omit=optional && node build)
uv sync                                   # Python 3.12 + torch cu128
docker compose up -d                      # PostgreSQL (127.0.0.1:5433)
mkdir -p data/raw/pokeapi && for f in pokemon_species_names pokemon pokemon_forms pokemon_form_names \
  move_names ability_names abilities ability_flavor_text item_names type_names; do
  curl -sSfL -o data/raw/pokeapi/$f.csv https://raw.githubusercontent.com/PokeAPI/pokeapi/master/data/v2/csv/$f.csv; done
uv run alembic upgrade head
uv run python -m pokesim.data.ingest_dex  # Showdown の Champions 図鑑 + 日本語名を DB に投入
```

## 動作確認
```bash
(cd external/pokemon-showdown && node pokemon-showdown start --no-security 8000) &
uv run python scripts/smoke_battle.py     # poke-env でランダム対戦 10 戦
uv run pytest
uv run python -c "from pokesim.lookup.effects import lookup; print(lookup('いかく', 'ability'))"
```

## 対戦サポート（選出画面 → 選出提案 → 毎ターンの推奨 → 記録）
> 練習・自分の対戦の振り返り・配信での解説向けです。公式大会では対戦中の外部ツールによる支援は禁止されています。
> ランクマッチでの利用は各自の判断で、規約を確認してください。

```bash
uv run python -m pokesim.vision.refs              # 選出画面の照合用に HOME の画像を取得（初回のみ。data/raw/ に保存、git 管理外）
uv run python -m pokesim.live.selection pick-rates  # リプレイから相手の選出率・先発率を集計
uv run python -m pokesim.rl.myteam                # 対策技（アンコール・ほえる等）の採用率

# 自分の構築専用の学習（構築 JSON は data/my_teams/ に置く）
uv run python -m pokesim.rl.bc record --battles 4000 && uv run python -m pokesim.rl.bc fit
uv run python -m pokesim.rl.train --my-team data/my_teams/baton.json --season all --reward sparse \
    --init checkpoints/bc_baton.zip --league --opp-snapshots checkpoints/ppo_m6_league3m.zip --name ppo_baton2 --steps 1000000
uv run python -m pokesim.rl.train ... --random-lineup --name ppo_baton_any   # どの3体を選んでも動くモデル
# 相手目線の学習（真似 AI から出発し、この構築に勝つように学習）→ その相手を練習相手にして学習し直す
uv run python -m pokesim.rl.human build && uv run python -m pokesim.rl.human fit   # 人間の行動の真似 AI（bc_human）
uv run python -m pokesim.rl.train_opponent --init checkpoints/bc_human.zip --name opp_baton --steps 3000000
uv run python -m pokesim.rl.train ... --random-lineup --init checkpoints/ppo_baton_any.zip --p-heuristic 0.2 --n-envs 4 \
    --opp-snapshots checkpoints/opp_baton_seg012.zip checkpoints/opp_baton_seg001.zip checkpoints/opp_baton_seg012.zip \
    checkpoints/bc_human.zip checkpoints/ppo_m6_league3m.zip --name ppo_baton_opp --steps 3000000
cp checkpoints/ppo_baton_opp_seg010.zip checkpoints/ppo_baton_v2.zip   # 100万ステップ版を採用（評価で最良）
uv run python scripts/sim_matchups.py record --policy checkpoints/ppo_baton_any.zip --battles 12000
uv run python scripts/sim_matchups.py fit         # 選出ごとの勝率モデル

# サーバー（ブラウザで http://127.0.0.1:8100 、OBS のブラウザソースには /overlay）
uv run python -m pokesim.api.server --policy checkpoints/ppo_baton_v2.zip \
    [--obs-source "キャプチャのソース名"] [--obs-record /data/poke-sim/obs_frames]
    # OBS の選出画面を自動で認識（パスワードは環境変数 OBS_PASSWORD か ~/.config/poke-sim/obs.env の OBS_PASSWORD=...）。--obs-record は変化した画面を保存（容量が大きいので別ディスク推奨）
```

- 選出画面: OBS から自動認識（または画像をアップロード）。誤認識は候補から直せます。
- 選出: 相手の選出率・先発率と、自分の選出ごとの推定勝率を表示。
- 試合中: OBS 連携時は対戦画面を読み取り（メッセージ欄の OCR・HP バー・コマンド画面の検出）、技・交代・ひんし・ランク変化・
  状態異常・天候・設置技・メガシンカ・勝敗を自動で反映します。自分の番（コマンド画面）が来るたびに、推奨行動の確率・推定勝率・
  相手の次の行動の予測が更新されます。読み取った内容は操作画面に表示され、読み落としは手入力で補正できます（`--no-screen` で手入力のみ）。
  選出を登録しなくても、出てきたポケモンから自動で登録・修正します。
- 保存した画面で読み取りを確認: `uv run python scripts/replay_frames.py /data/poke-sim/obs_frames/20261013 [--apply]`
- 試合後: 勝敗のメッセージを読み取ると自動で（手入力時は勝ち/負けボタンで）`data/raw/live/logs/` と DB（`my_battles`, `preview_records`）に記録されます。
  記録には使ったコード・方策・選出表の版（`versions`）が入ります。選出表が別の方策用に作られたものなら起動時に警告します。

## 判断の評価基盤（[改善設計](docs/competitive-improvement-design.md) の S1〜S3）
```bash
uv run python -m pokesim.data.curate_battles     # 対戦台帳 data/curated/{battles,decisions}.jsonl + manifest.json（live/再構成の重複は統合、曖昧なものは merge_pending）
uv run python -m pokesim.data.split_manifest     # train/development/calibration と前向き test の分割（一度作ったら固定。--force で作り直し）
uv run python -m pokesim.rl.human fit --records data/processed/human_records.npz --split-manifest data/curated/split_manifest.json ...
# 共通戦略プランナー（型を知らない自分が全サンプルで同じ手を使う前提で解く）: shadow は現行の推奨と並べて記録、on はそれで行動
uv run python -m pokesim.api.server --policy checkpoints/ppo_baton_v2.zip --planner shadow
uv run python scripts/ladder.py --battles 30 --planner shadow [--sample]   # --sample: 推奨分布から抽選（seed と選択確率を記録）
uv run python -m pokesim.evaluation.fixtures -v  # E1: 既知の失敗パターンの局面（src/pokesim/evaluation/fixtures/）を旧方式とプランナーで比較
# 推定勝率の較正: calibration 分割の試合で方策の評価値→勝率を当てはめ、基準（勝率一定）より良いときだけ表示を有効化
uv run python -m pokesim.evaluation.calibration fit --source showdown --policy checkpoints/ppo_baton_items.zip
uv run python -m pokesim.evaluation.calibration report
```
- プランナーは先読みの続きを2種類の方策（greedy / tactical: メガ・回復・安全な積み・バトン/交代）で打ち、終局を ±1、
  途中局面は低信頼の代替値で評価します（`status.shadow` の `resolved` = 終局まで進んだ割合、`continuation_spread` = 2方策の評価差）。
- 推定勝率は、source・方策・続きの設定が一致する較正がある場合に、1ターン目の通常判断でだけ表示します。
  それ以降や強制交代は較正の対象外です。較正がない場合も、方策の評価値を勝率とは区別して参考表示します。
- 較正には `versions.calibration_context` 付きの記録が必要です。複数の設定がある場合は、
  `calibration fit --context <hash>` で対象を指定します。`--split` は calibration に限定しています。
- 探索期限は全方策・全サンプルを一巡する単位で確認するソフト予算です。最初の一巡は必ず実行するため、
  指定時間を超える場合がありますが、期限によって tactical だけが評価から消えることはありません。
- 攻撃できる残りが1体・0体になると、推奨の下に警告を出します（`pokesim.team.roles`）。
