# 公開観測から価値学習用データを作る

短期HP差に代えて勝敗を学ぶための、オフラインのデータ生成・出力手順。
Showdownのエンジンは固定したまま使う。対戦サーバや支援APIの再起動は不要。
モデルの学習・探索への組込みは次の工程。

## 2種類の教師を作る

| データ | 入力 | 教師の意味 | 用途 |
| --- | --- | --- | --- |
| Showdown公開リプレイ | ターン開始時点の公開状態。非公開requestはnull | 実際にその対戦者が指したときの最終勝敗 | 公開状態を読む部分の事前学習 |
| 公開観測による分岐対戦 | 公開状態・履歴と、自分自身のrequest | 固定参照方策を継続したときの終局勝敗 | V_refの学習・開発評価 |

人間同士の勝敗を、そのまま固定参照方策の勝率ラベルへ置き換えない。
まず公開状態のエンコーダをリプレイで学習し、自分の非公開情報を追加したモデルを分岐対戦で調整する想定。
現時点で共通のモデル入力として利用できるのは`observation.public`と`observation.side`。
リプレイにない自分の技・努力値・持ち物を、対戦後の公開情報から補完して入力する処理は行わない。

## 公開リプレイを追加する

既存の`data/raw/replays/*.json`を読み取り、新しい出力ディレクトリへ保存する。
このコマンドはDBを使わない。ビルド済みShowdownが必要。

```bash
uv run python -m pokesim.evaluation.replay_value \
  --out data/processed/public_value/showdown-replays-v2
```

`train.jsonl.gz`、`development.jsonl.gz`、両者の`manifest.json`を出力する。
同じ出力先を上書きしない。入力ディレクトリが別の場合は`--data-root`でプロジェクトのルートを指定する。
元JSON、生成コード、エンジン、分割ルール、除外理由、出力ファイルのSHAを保存する。

- `rl.human`と同じreplay IDのSHA1による15%の開発holdoutを使う。既存の模倣学習のholdoutを事前学習へ入れない。
- 同一対戦の全ターン・両視点は同じsplit。IDおよびログの完全一致による重複も除外する。
- 既存ledgerの全Showdown対戦と、新しい`data/raw/ladder`の識別可能な全対戦を公開コーパスから除外する。
- 勝者不明、結果重複、ターン逆転、ターンのない対戦は教師にしない。実際の引分だけ0.5にする。
- 結果行より後の退出に伴う空の`player`行で、対戦中の参加者名を消さない。
- パーサへ結果行を渡さず、各`turn`行までの公開イベントだけを状態へ反映する。
- 各対戦の全ターン・両視点の重みの合計は1。長い対戦を自動的に重くしない。

ここでの開発holdoutは既存の学習工程でも使われる開発用データであり、最終的な未見testではない。
公開状態のパーサは軽量な表現であり、全効果の持続カウンタや隠れた正確なHPまでは復元しない。

## 情報制約を守る分岐対戦を作る

PostgreSQL、既存の`data/curated/battles.jsonl`と固定split manifest、
そのtrainのみで作った`data/processed/scoped/priors.json`、自分の構築JSONが必要。
既存のsplit manifestを再生成しない。calibration/testからの出力は受け付けない。

```bash
PYTHONHASHSEED=0 uv run python -m pokesim.evaluation.branch_matches generate \
  --information-scope public_observation --split train \
  --out data/processed/public_value/train-v2 \
  --worlds 2 --seeds 2 --max-turns 200

PYTHONHASHSEED=0 uv run python -m pokesim.evaluation.branch_matches generate \
  --information-scope public_observation --split development \
  --out data/processed/public_value/development-v2 \
  --worlds 2 --seeds 2 --max-turns 200
```

既定はそのsplitのShowdown記録のターン1を起点とする。分岐後の途中局面も教師として記録する。
`--positions`で別ターンを指定する場合も、元対戦が指定splitに所属することを検証する。
同じ設定・入力・コード・エンジンなら再実行で再開できる。どれかが変わったら新しい`--out`を使う。

両者の方策へ渡すのは、JSONの公開履歴・公開状態・そのプレイヤー自身のrequestのみ。
相手の未公開セットや選出済み控えを渡さない。元記録の6体のプレビューを保持し、
サンプルした3体を相手に公開することもない。履歴中の自分の正確なHPは公開HP表示に変換する。
相手の初手は、自分の候補手を提出する前に選ぶ。

参照方策`public-reference-v1`は、公開された脅威と、自分の技・能力値を用いる軽量ヒューリスティック。
既存の全情報`expert`とは別実装で、強さの優越は確認していない。相手のダメージは公開種族情報からの概算。
学習先は、この固定参照方策・固定相手プールに対するV_refであり、最適勝率やラダー勝率ではない。

## 終局した途中局面を出力する

```bash
uv run python -m pokesim.evaluation.branch_matches export-value \
  --out data/processed/public_value/train-v2 \
  --dataset data/processed/public_value/reference-train-v1.jsonl

uv run python -m pokesim.evaluation.branch_matches export-value \
  --out data/processed/public_value/development-v2 \
  --dataset data/processed/public_value/reference-development-v1.jsonl
```

各行に公開観測、自分のrequest、合法手、参照方策の手、終局教師、元対戦ID、初手の候補、world/seed/継続方策を保存する。
対象はp1の通常判断。候補手を強制した初手、p2視点、強制交代判断は除外する。
初手の強制を通常方策のV_ref教師として扱わない。未終局・失敗試行は途中局面も含めて除外する。
残す1試行内の各行の重みの合計は1。試行間・元対戦間にも相関があるため、評価は元対戦単位で集計する。

JSONLと隣接する`.manifest.json`を保存し、上書きしない。保存済み位置のハッシュ・視点・合法手・結果の整合性を検証する。
診断用の全情報継続データは、この出力へ変換できない。

## 2026-10-16の生成確認

エンジンは`942cda71d14bf0618bd57991708a2e2734b6daba`のまま固定。
公開リプレイ4,999戦のうち、ターンのない41戦を除く4,958戦を利用した。

| データ | 学習用 | 開発評価用 |
| --- | --- | --- |
| 公開リプレイの元対戦数 | 4,218戦 | 740戦 |
| 公開リプレイの両視点の局面数 | 80,202 | 14,202 |
| 分岐対戦の元対戦数 | 34戦 | 19戦 |
| 分岐試行数 | 1,888 | 1,104 |
| 分岐対戦から出力した局面数 | 21,602 | 11,490 |

分岐試行2,992件はすべて終局し、失敗・未終局は0件。
イリュージョンの`replace`イベントを交代扱いしてHPを失う問題も修正し、既知の控えと公開された正体を分離した。
これらはサーバ上の`data/processed/public_value/`に保存した生成物であり、リポジトリへ大容量データは追加しない。
同じ元対戦から作った複数局面を、独立した対戦数として数えない。

次は公開リプレイによる事前学習とV_refの調整、開発対戦単位のBrier/log lossと判断改善の比較。
このPR自体では重み更新や実戦勝率の改善を確認していない。
