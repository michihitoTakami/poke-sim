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
