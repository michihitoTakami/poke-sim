# poke-sim

Pokémon Champions シングル（3v3, `gen9championsbssregmc`）の対戦サポートツール。
計画の全体像は `docs/plan.md` を参照。

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
uv run python scripts/sim_matchups.py record --policy checkpoints/ppo_baton_any.zip --battles 12000
uv run python scripts/sim_matchups.py fit         # 選出ごとの勝率モデル

# サーバー（ブラウザで http://127.0.0.1:8100 、OBS のブラウザソースには /overlay）
uv run python -m pokesim.api.server --policy checkpoints/ppo_baton_any.zip \
    [--obs-source "キャプチャのソース名"] [--obs-record /data/poke-sim/obs_frames]
    # OBS の選出画面を自動で認識（パスワードは環境変数 OBS_PASSWORD）。--obs-record は変化した画面を保存（容量が大きいので別ディスク推奨）
```

- 選出画面: OBS から自動認識（または画像をアップロード）。誤認識は候補から直せます。
- 選出: 相手の選出率・先発率と、自分の選出ごとの推定勝率を表示。
- 試合中: 起きたこと（相手の登場・技・HP%・状態など）を入力すると、推奨行動の確率・推定勝率・相手の次の行動の予測が更新されます
  （今は手入力。画面読み取りで自動化する予定）。
- 試合後: 勝ち/負けを押すと `data/raw/live/logs/` と DB（`my_battles`, `preview_records`）に記録されます。
