# NN入力・行動教師・実戦への接続の変更

直近の118戦は50勝68敗。解析で、控えの麻痺の持ち主、技ごとのPPゼロ、晴れの残り時間が
旧NNで同一入力になることを確認した。今回はこの欠落を解消するモデルと、候補手の終局比較から
学ぶ経路を実装・実行した。勝率が改善したという報告ではない。

## 実戦に適用する変更

- Tauntで全技を封じられた場合やPP切れの`Struggle`を、poke-envと同じ要求専用スロットへ変換する。
  提案の合法性判定、ラベルへの逆変換、送信を一致させる。通常の技として存在しないことを理由に
  棄却してランダム手へ落ち、品質ゲートで収集が止まる問題を解消する。
- 探索の既定を`reference-material-v2`へ変更。能力ランクへの一律加点を除き、終局結果、
  NN、実際の生存・HP・状態異常で評価する。鉄壁・加速・瞑想はシミュレーションとNNを通じて評価される。
  比較用のv1は明示指定時にだけ残し、記録にも異なる効用版を保存する。
- 選出は`preview-evidence-v5`。履歴モデルの事後平均と減衰済み試合数を、模擬戦による最終順位まで保持する。
  実戦成績の重みは、先発込みの試合数と別先発での同じ3体の証拠に応じて増やし、最大0.35に制限する。
  履歴がない場合は従来の模擬戦評価。選出の観測バイアスがあるため、合成スコアを勝率とは呼ばない。

## 新モデルと入力

`observed-value-sets-v2` / `nested-sets-v2`を追加した。旧`features.py`は変更せず、旧checkpointも同時に読める。
技を非線形に符号化してポケモンごとに集約し、さらにポケモンを非線形に符号化して各陣営を集約する。
技・控えの順序に依存せず、種族と状態異常、技とPPを結び付けたまま扱う。

- 公開2,956次元、本人1,297次元。hidden=128で200,510パラメータ。
- 各技に技ID・効果・PP・使用不能・欠損mask、各個体に種族・HP・状態異常・能力変化・正確な本人能力値を持つ。
- 見せ合い6体も個体ごとの符号化後に集約する。パディングはmaskし、非公開要求の欠損を架空のチームにしない。
- 公開履歴から天候の開始とupkeepを数え、通常天候は5〜8ターンの範囲、公開メガリザードンYは5ターンとして扱う。
  後から出てきた別個体の持ち物を、天候を起こした個体のものとは扱わない。
- 解析で見つけた3種類の入力衝突が解消したこと、集合の並べ替え不変性、保存・再読込の一致を検査した。

これは系列NNや不完全情報ゲーム全体の解法の完成版ではない。履歴全体を読むGRU、複数世代の自己対戦リーグ、
相手構築の事後分布の学習、選出と行動を同時に最適化する方策は、別の比較対象として残る。

## 学習・評価を実行した結果

元のリプレイ分割を保持し、学習51,826行・元対戦3,296件、開発25,475行・元対戦1,650件で3 seedを学習した。
両プレイヤーの`:p1`/`:p2`を同じ元対戦として、合計重み1、対戦単位bootstrap、分割重複検査を行うよう修正した。
モデルの大小だけでなく、旧モデルを旧入力、新モデルを新入力で評価し、対戦・ターン・教師の並びが同じことを検査する。

| 元対戦単位の開発Brier（小さい方が良い） | 値 |
| --- | ---: |
| HP等の基準 | 0.211516 |
| 旧NN | 0.197495 |
| 新NN（3 seed平均） | 0.196293 |

旧NNとの差は約−0.00120。採用条件の改善幅0.015と区間の条件を満たさなかった。
直近PR43の117戦・1,590個の正常ターン開始点でも、旧NN約0.21530、新NN約0.22278で改善を確認できない。
新−旧のBrier差の対戦相手単位95%区間は約−0.00639〜+0.02073。調査済みの開発診断であり、未見の採用試験ではない。
この1,590点は同一ターンの重複要求を除いた集合で、前回解析の1,592点とは2点異なる。

**新NN重みは採用しない。実戦は`runs/value-replays-v1`と既存の相手方策を維持する。**
新形式の候補を通常の`ValueEnsemble`でロードすると拒否する。オフライン検査だけ`allow_candidate=True`を使う。
表現の欠落を直しただけでは、公開リプレイ上の勝敗から、こちらの構築での良い手は自動的には学べないという結果でもある。
公開リプレイの本人セット・能力値にも補完による推定誤差がある。正確な実requestと、固定方策の分岐からの学習を分けて扱う。

## 行動学習と、その方策用の価値

`learn.branches`の教師を単純な平均勝敗差から変更した。

1. 同じ型仮説の4世界、各2乱数、4相手方策で候補手を終局まで比較する。相手には従来3方策に加えて
   Taunt・Haze・強制交代を使う`counterplay`を追加した。
2. 世界単位の対応付きbootstrapで差の90%区間が正、かつ相手方策のどれかに平均で逆転されない比較だけを教師にする。
   同一世界の乱数反復を独立の型仮説とは数えない。急所一発、未完了、失敗、手動操作との食い違いを正解ラベルにしない。
3. 行動順位を学習後、その行動方策を固定してもう一度分岐を生成し、その方策が実行した通常判断から継続価値を更新する。
   最初の強制介入手は価値教師から除外し、方策のhashと版を保存する。新しい集合NNもこの経路で使える。

訓練4局面・開発4局面で1,504分岐を生成し、行動順位を学習した。その方策を固定し、さらに1,504分岐を再生成した。
固定方策の継続価値の開発Brierは約0.23972→0.13901。ただし開発元対戦は4件しかなく、採用条件は満たさない。
3,008件の分岐は独立した対人対戦3,008件ではない。行動NNと継続価値の候補も実戦には導入しない。
教師が得られない少数世界では、ノイズを学習する代わりに生成不足として停止する。

## 効用変更の固定局面比較と検証

保存済みの8局面を同じ型仮説・乱数系列、各256追加試行で比較した。
7局面は同じ手。1局面では`Moonblast`から`Calm Mind`へ変わった。
その2手を4世界×2乱数×4相手方策で終局比較したところ、前者25勝7敗、後者32勝0敗、未完了・失敗0。
積みを一律に止める変更ではない。仮説上のシミュレーションの診断で、対人勝率の改善を示すものではない。

通常の判断予算は15秒のまま。今回の実戦変更の評価と、新NNの採用試験は分けて記録する。
学習run・原記録・全対戦分岐・大容量の重みは無視対象に保存し、[監査JSON](20261019-structured-learning.json)にhashを残す。
関連356テストが成功。Ruff、変更したNodeファイルの構文検査、`git diff --check`も成功した。

## 再現・次の未見データでの検査

本体ルートでの実行例。出力先は既存ファイルを指定しない。分割は作り直さない。

```sh
python -m pokesim.learn.value_dataset --rows data/processed/learn/replays-v2-0.jsonl.gz \
  data/processed/learn/replays-v2-1.jsonl.gz --sides p1 --feature-version observed-value-sets-v2 \
  --out data/processed/learn/sets-new/replays-train.npz
python -m pokesim.learn.value_dataset --rows data/processed/learn/replays-v2-2.jsonl.gz --sides p1 \
  --feature-version observed-value-sets-v2 --out data/processed/learn/sets-new/replays-dev.npz
python -m pokesim.learn.train_value --train data/processed/learn/sets-new/replays-train.npz \
  --dev data/processed/learn/sets-new/replays-dev.npz --hidden 128 --seeds 0 1 2 \
  --reference-run runs/value-replays-v1 --reference-dev data/processed/learn/value-v2/replays-dev.npz \
  --out runs/sets-new
PYTHONHASHSEED=0 python scripts/eval_value_utility.py --fixtures data/processed/relay-goals-20261019-runtime \
  --value-run runs/value-replays-v1 --out data/processed/utility-new.json
```

`learn.branches generate`は`--worlds 4 --seeds 2`以上で実行する。
`train --feature-version observed-value-sets-v2 --init-value <集合NN候補>`で新モデルを使える。
`generate --continuation-net <行動NN.json>`→`train --fixed-policy <同じ行動NN.json>`が二段階目。
少数の開発局面での改善を採用根拠にしない。次は別の対戦相手・構築、固定した未見期間、実戦A/Bで検査する。

`learn.compare_value`は、保存済みの**正確な自分のrequest**とそのターン開始までの公開ログだけから比較する。
未来の相手情報で再構成せず、元のrequestがp2でも正規化済みログのp1へ合わせる。
対戦相手単位の区間、原記録・重みのhash、除外・失敗を保存し、自動で採用可にはしない。

```sh
python -m pokesim.learn.compare_value --records data/raw/ladder --after <事前に固定した開始時刻> \
  --code <対象の配備commit> --candidate runs/value-sets-v2-20261019-final \
  --incumbent runs/value-replays-v1 --out data/processed/prospective-value-new.json
```
