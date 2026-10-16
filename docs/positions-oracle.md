# 局面セットとオラクル後悔（探索設定の採否判定）

勝率差+5ポイントの検出には1群あたり約1,550戦が要る。探索設定の比較は、実戦記録から切り出した固定局面に対する
「長時間探索（オラクル）との差」で行い、実戦は回帰の監視に使う。

```bash
# 1. 記録から局面を切り出す（敗戦の最大評価落差57、他の敗戦100、勝戦143をターン帯で層化）
uv run python -m pokesim.evaluation.positions build --out data/processed/positions/v1 --since 20261016-1613 --commit db80f64
# 2. オラクル: 8乱数列 × 60秒を合算（約1分/局面）
uv run python -m pokesim.evaluation.positions oracle --set data/processed/positions/v1 --streams 8 --budget-ms 60000 \
    --search '{"priorVisits":4,"rolloutPolicies":{"p2":["disrupt","tactical","expert"]},"maxTurns":120}'
# 3. 候補設定を実戦予算で指し、後悔（オラクル最良手との目的値差）と一致率を測る
uv run python -m pokesim.evaluation.positions eval --set data/processed/positions/v1 --name base --search '{"maxTurns":120}'
uv run python -m pokesim.evaluation.positions eval --set data/processed/positions/v1 --name workers4 --search '{"maxTurns":120,"workers":4}'
uv run python -m pokesim.evaluation.positions report --set data/processed/positions/v1
```

- `build` は `Lookahead.query(capture=True)` で、実戦と同じ観測ログ・仮説世界・設定をそのまま保存する。
  相手の型は観測から再サンプルした仮説であり、実戦時の粒子と一致する保証はない（p2側で戦った試合は履歴の向きが違う）。
- 後悔はオラクルの目的値 `(1-rho)*expected + rho*worst` で測る。オラクルは同じ継続方策の偏りを共有するので、
  これは探索の品質の指標であり勝率そのものではない。`played_regret` は実戦で指した手の後悔。
- `scripts/ladder_report.py` は探索の目的値と実勝敗の校正（ターン帯別のBrier・定数との比較）を出す。
- 切り出した記録は以後の開発データであり、未見のtestとは呼ばない。
