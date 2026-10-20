# 候補手の比較から行動方策と継続価値を更新する

本番の原記録・重みを上書きせず、別の作業ツリーで生成・学習・採用評価を行う。
実行された手と提案の食い違いを教師にはしない。本人の手動操作も、勝った試合の全行動も、
自動的な正解ラベルにはしない。

## 固定するもの

`learn.curriculum freeze`は、締切前の保存済み試合から通常判断を選ぶ。
同じ対戦相手または同じ見せ合い6体で連結する試合は、同じグループに置く。
各グループから最大1試合、各試合から指定数の局面を選び、全てのターン・型仮説・乱数・
候補手を同じ分割へ固定する。既存分割は`--inherit`で継承し、連結グループ内で分割が
矛盾する試合は隔離する。既存の#37開発用試合も学習へ移さない。
選出や勝敗の将来情報で候補手の教師を選ばない。

原記録SHA、グループ、分割、対象局面は`cohort.json`に残る。
過去に分析した構築の歴史的なホールドアウトであり、将来の対人勝率の証明ではない。

## 行動の比較と学習

`learn.curriculum generate`は公開された型だけの事前分布を使う。
本人の非公開情報は当該ターンの保存済みリクエストから復元する。
相手側には、その側のリクエストと公開ログだけを渡す。未来の相手の型・技を渡さない。
通常は4型仮説・2乱数・4相手方策で全合法候補手を比較する。
終局まで進まない試行、失敗した試行は敗戦ラベルにしない。

行動順位の教師は、対応付き比較の90%区間が正で、どの相手方策にも平均で逆転されない組だけ。
最低4つの**異なるシミュレーション状態**を要求する。同じ世界の乱数反復や、同じ状態の
重複サンプルを独立した型仮説として数えない。

方策の初期学習には、参照方策がシミュレーション中に実行した継続行動を使う。
介入したルート手は除き、グループごとの上限と決定的なサンプリングで長い試合の偏りを抑える。
これは参照方策の蒸留であり、実行者の手を勝ち手として教える処理ではない。
その後、候補手の比較で順位を学習し、開発用の後悔量でエポックを選ぶ。

`--rank-only`で方策を保存・固定した後、同じ型仮説・乱数で分岐を生成し直す。
この固定方策の、介入後の通常判断から継続価値を学習する。
値を学習した方策と実行する方策のhashを一致させる。

今回の方策は通常ターンだけを学習するので、JSONに`decision_scope: move`を持つ。
探索のpriorと継続、分岐の実行役はこの範囲を確認する。
バトン後や瀕死後の交代先選択には既存の参照方策を使い、未学習の出力を適用しない。
範囲のない既存の相手モデルは、従来の全フェーズ対応として読む。

保存する軌跡は、完走した後に本人のルートと指定数の継続判断へ間引ける。
実行行動・乱数・終局を変えずに容量を抑え、介入マークと終局ラベルを保持する。

## 採用条件

通常の判断予算・現行構築は固定する。採用のために条件を緩めない。

1. 固定方策の価値が、現行NNの**全seedの平均**を同じ継続状態で上回ること。
   初期乱数モデルとの比較だけでは採用できない。
   Brier差が−0.015以下で、グループ単位90%区間の上限が0未満であること。
2. 行動順位の後悔量も参照方策を上回り、差のグループ単位90%区間の上限が0未満であること。
   固定した学習済み方策を比較基準に流用しない。
3. 20以上の独立グループで、参照方策と学習済み方策そのものの続きを比較する。
   同じ型仮説・乱数・相手方策の完走比較で、平均改善0.03以上、90%区間の下限が0より大きいこと。
   比較欠損がある場合は採用しない。

3は「弱い方策の敗戦を予測できるようになっただけ」で1を満たすことを防ぐ。
統計条件だけを通った候補は保留し、`learn.promotion`の比較を一度実行する。
ランタイムも、グループ化した新候補にこの比較の成功記録を要求する。
条件を満たしても、実戦レートの上昇を保証するものではない。

## 実行例

環境のPythonと固定したShowdownエンジンを指定し、`PYTHONHASHSEED=0`で実行する。
`--out`には新しいディレクトリを指定する。生成済みの成果物は同じ入力・コードの時だけ再利用する。
事前分布は、空のtrain台帳から`prior_bundle --allow-empty`で作った、公開構築のみのsnapshotを使う。

```sh
python -m pokesim.learn.curriculum freeze --records /path/to/data/raw/ladder \
  --before YYYYMMDD-HHMMSS --inherit data/processed/new-learning-split.json \
  --groups 50 20 20 --positions-per-battle 2 --out /new/experiment/cohort
python -m pokesim.learn.curriculum generate --records /path/to/data/raw/ladder \
  --cohort /new/experiment/cohort --priors /new/published-priors.json --out /new/experiment/planned
python -m pokesim.learn.branches train --data /new/experiment/planned \
  --out /new/policy --rank-only --epochs 40 --distill-per-group 16
python -m pokesim.learn.curriculum generate --records /path/to/data/raw/ladder \
  --cohort /new/experiment/cohort --priors /new/published-priors.json \
  --policy /new/policy/ours.json --out /new/experiment/learned
python -m pokesim.learn.branches train --data /new/experiment/learned \
  --fixed-policy /new/policy/ours.json --init-value runs/value-replays-v1 --epochs 40 --out /new/value
python -m pokesim.learn.promotion --before /new/experiment/planned \
  --after /new/experiment/learned --run /new/value
```

実験後の候補・データは保持する。本番への反映は、採用結果、状態復元・合法手・固定予算での
動作検証を確認して行う。対戦中にコード・重みを入れ替えない。
