# 学習した公開方策（相手モデル）と学習データの作り方

探索内の相手が手書き規則（`tactical`/`disrupt`）だと「積めば勝てる」線が過大評価される。
実戦相手の行動を学んだ方策をNodeワーカーの中で動かし、木の事前分布・ロールアウト・root予測に使う。

## データ（Stage 0）
```bash
uv run python -m pokesim.data.curate_battles --out data/curated/v2
uv run python -m pokesim.data.split_manifest --ledger data/curated/v2/battles.jsonl --out data/curated/split_manifest_v2.json \
    --shares 0.7 0.15 0.15 --cap-per-opponent 12
uv run python -m pokesim.data.prior_bundle --ledger data/curated/v2/battles.jsonl --split-manifest data/curated/split_manifest_v2.json \
    --out data/processed/scoped/priors-v2.json
# 実戦記録: 各通常ターンで自分(p1)と相手(p2)の観測行。相手の型は「公開済み ∪ 実際に選んだ手」を含む仮説
uv run python -m pokesim.learn.records --split train --priors data/processed/scoped/priors-v2.json --out data/processed/learn/ladder-train-v1.jsonl.gz
# 公開リプレイ: 両プレイヤーを順に行為者にする。自分のラダー戦は除外
uv run python -m pokesim.learn.replays --shard 0/3 --out data/processed/learn/replays-v1-0.jsonl.gz
```
観測は `services/sim-tools/observe.js` が作る。行為者のrequestと公開状態・履歴だけで、相手の非公開情報は含まない
（`tests/test_learn_records.py`）。

## 特徴量・学習・書き出し（Stage A）
- 仕様: `services/sim-tools/policy_features.schema.json`（状態342・行動89次元）。JS `policy_features.js` と
  Python `pokesim.learn.features` が同じベクトルを出す（`tests/test_policy_feature_parity.py`）。
- 学習:
```bash
uv run python -m pokesim.learn.dataset --rows data/processed/learn/replays-v1-*.jsonl.gz --sides p1 --out data/processed/learn/features-v1/replays.npz
uv run python -m pokesim.learn.dataset --rows data/processed/learn/ladder-train-v1.jsonl.gz --sides p2 --out data/processed/learn/features-v1/ladder-train-p2.npz
uv run python -m pokesim.learn.train_policy --head opp --out runs/opp-policy-v1 --pretrain .../replays.npz \
    --train .../ladder-train-p2.npz --dev .../ladder-development-p2.npz --seeds 0 1 2 --device cuda
uv run python -m pokesim.learn.evaluate_opponent --run runs/opp-policy-v1 --dev .../ladder-development-p2.npz
```
- 書き出した `opp-seed0.json` を `services/sim-tools/policy_net.js` が読む（`tests/test_policy_net_parity.py` でtorchと一致を確認）。
- 採用ゲート（オフライン）: 開発分割で top-1 ≥ 46%、top-3 ≥ 72%。実戦の記録予測（`advice[].opponent`）と同じ対戦で比較する。

## 探索への接続
```bash
uv run python scripts/ladder.py --planner on --value-run runs/terminal-value-v2 --opponent-policy net:runs/opp-policy-v1/opp-seed0.json
```
`net:` を指定すると、相手ネットが p2 の木ノードの事前分布（PUCT）、展開後最初の `rolloutNetTurns` ターンのロールアウト
（確率 `rolloutMix.opp`）、root予測の学習分（`planner.predict` の `prior`）になり、クローンPPO（bc_human）は読み込まない。
記録の `versions.opponent_policy.kind` は `net`、`summary.policy_nets` にSHAが入る。ネットが読めない判断は品質停止の対象。

## 実勝敗で学ぶ価値（Stage C）
```bash
uv run python -m pokesim.learn.value_dataset --rows data/processed/learn/replays-v1-*.jsonl.gz --sides p1 --out data/processed/learn/value-v1/replays.npz
uv run python -m pokesim.learn.value_dataset --rows data/processed/learn/ladder-train-v1.jsonl.gz --sides p1 --out data/processed/learn/value-v1/ladder-train.npz
uv run python -m pokesim.learn.train_value --out runs/value-real-v1 --pretrain .../replays.npz --train .../ladder-train.npz \
    --dev .../ladder-development.npz --reference-run runs/terminal-value-v2 --device cuda
```
特徴量は既存の `observed-value-v1`（`pokesim.value.features`）のまま、教師を参照方策の継続結果ではなく実際の勝敗にする。
`ValueEnsemble` は `provenance.target == "real_outcome"` の run を読み、`--value-run runs/value-real-v1` でそのまま探索に使える。
初回（開発108戦）: Brier 定数 0.247 / HP基準 0.211 / 現行V_ref 0.201 / 実勝敗ネット 0.176（ゲートC: 差 ≥ 0.015 かつ90%区間が0を含まない → 合格）。
これは観測の予測精度であり、探索に入れたときの手の改善は局面セットの後悔とローカル対戦で別に測る。
