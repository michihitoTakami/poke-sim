# 候補手を固定して終局まで比較する

短い探索が身代わり・回復を選ぶ局面で、最初の手だけを変えて実際のシミュレーターの勝敗を比較する。
ライブの判断方法・サーバ設定・モデルは変更しない。価値関数の学習はこのPRの対象外。

## 実行

PostgreSQLのポケモンデータ、ビルド済みShowdown、既存のtrain限定prior bundleと元対戦記録が必要。
通常の開発環境で次を実行する。ローカルのShowdown対戦サーバや支援APIの起動・再起動は不要。

まず、前回の診断で「壊された身代わりを繰り返す」判断が残った2局面を比較する。

```bash
PYTHONHASHSEED=0 uv run python -m pokesim.evaluation.branch_matches generate \
  --positions docs/terminal-branches-example.json \
  --out data/processed/branch_matches/substitute-v1 \
  --worlds 6 --seeds 4 --max-turns 200
```

同じコマンドを再実行すると、保存済み局面を検証して残りから再開する。
設定・コード・入力・prior・エンジンが変わった場合は別の`--out`が必要。
`--limit 1`で最初の1局面だけ実行し、後からその引数を外して残りを実行できる。
同じ出力先での並列実行には対応しない。

集計だけならDBやシミュレーターを起動せず実行できる。

```bash
uv run python -m pokesim.evaluation.branch_matches report \
  --out data/processed/branch_matches/substitute-v1
```

既存の20局面へ広げる場合（前回の診断ファイルがある環境）：

```bash
PYTHONHASHSEED=0 uv run python -m pokesim.evaluation.branch_matches generate \
  --positions data/processed/diagnostics/20261016-postmerge/positions.json \
  --out data/processed/branch_matches/diagnostic20-v1 \
  --worlds 6 --seeds 4 --max-turns 200
```

## 比較条件と入力

入力は`[{"file": "元記録.json", "turn": 4}]`というJSON配列。`file`は`data/raw/ladder`内のファイル名。
`--records`、`--team`、`--prior-bundle`で入力場所を指定できる。
そのターンの開始までのログで局面を復元し、相手の後で公開された技・型・実際の勝敗を読み込まない。
通常ターンの開始を対象とし、強制交代の途中の局面はエラーで停止する。

各局面で相手の型と未公開の選出を固定してサンプルする。各候補手に対して同じ世界・乱数seed・
継続方策の組を使う。既定は自分`expert`、相手`tactical`および`disrupt`。
Pythonの集合の並び順でも型の抽選が変わるため、生成には起動時の`PYTHONHASHSEED`固定が必要。
この値もmanifestへ記録する。集計だけなら指定不要。
相手の最初の手も、自分の候補手を提出する前にその継続方策で選ぶ。
将来の最良計画を候補手ごとに選び直す処理は行わない。
複数の自分側継続も比較するなら、例えば次の引数を追加する。

```bash
--continuations expert/tactical expert/disrupt expert@0/tactical expert@0/disrupt
```

候補手を省略すると、復元された世界の合法手ラベルの和集合を比較する。Mega進化の有無も別の手として扱う。
入力局面に`"actions": ["move:moonblast:mega", "move:substitute:mega"]`を指定すればその候補だけ比較する。
特定世界で使えない手は失敗として残し、比較から黙って落とさない。
乱数の初期状態は共通だが、選ぶ手により乱数の消費順は変わる。
バトンタッチ後の受け先や交代イベントは、同じ対戦インスタンスの継続方策とShowdownで実行する。

## 出力と読み方

- `manifest.json`: 設定、入力とコード・エンジンのハッシュ、実験ID、用途と情報制約。
- `positions/<id>.json`: 判断時の観測ログ・自分のリクエスト、サンプルした全世界、再現用workerリクエスト、
  全試行の結果と候補別・継続方策別の集計。局面単位で一時ファイルから原子的に保存する。
- `summary.json`: 保存済み局面の集計。`report`で再作成できる。

各試行に`world`、`seed`、`continuation`、`action`、`opponentAction`、`outcome`、`target`、
進行ターン数と失敗理由を保存する。

| outcome | target | 意味 |
| --- | --- | --- |
| win | 1 | シミュレーターが自分の勝利で終局 |
| loss | 0 | 相手の勝利で終局 |
| draw | 0.5 | シミュレーターが実際に引分で終局 |
| unfinished | null | `--max-turns`で打切り |
| failed | null | 手の拒否、進行停止等の計算失敗 |

HP差・攻撃役の有無・勝利までの速さは教師値に加えない。
`terminal_mean`は**終局した試行だけ**の平均で、打切りがある場合の順位決定には使わない。
`target_bounds`は未終局・失敗をすべて負け／すべて勝ちとして補った平均の下限・上限。
これは乱数や型推定の不確実性に対する信頼区間ではない。
候補間でこの範囲が重なる場合は打切りを伸ばすか追加試行する。継続方策別の不一致も確認する。
全試行が終局していても小標本の差だけで優劣を断定しない。

## 次の価値学習へ進む条件

この版の継続方策は、サンプルした相手の未公開セットも読む。
そのため成果物に`information_scope: sampled_full_information_continuations`、
`purpose: diagnostic_development`、`training_ready: false`を付ける。
同じ公開観測でもサンプル世界ごとに将来行動が変わるため、これをそのまま情報制約を守るV_refの教師とはしない。
公開観測だけで指す継続方策への対応、学習・開発・未見評価の対戦単位の分割、観測特徴の確定が後続の作業。
今回の2局面・既に診断した20局面も未見testにはしない。既存のsplit manifestを変更しない。

まずこの診断で、短期HP評価と終局結果で手の順位がどう変わるか、未終局がどれだけ残るか、
相手の妨害方策によって結果が逆転するかを確認する。最終的な採用判断は独立した実戦比較で行う。
