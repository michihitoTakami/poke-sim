# 勝敗価値を探索とShowdownへ接続する

`--planner on --value-run runs/terminal-value-v2`で、未終局の探索葉をV_refで採点する。
終局は勝+1・負−1・引分0、途中は`2 * V_ref - 1`。HP差・勝つ速さ・攻撃役の有無を加点しない。
本人のrequestと公開状態だけをモデルへ渡し、全葉をまとめて3モデルで推論する。
モデルが使えないときは理由を記録して既存の方策へ戻る。HP採点への自動切替は行わない。

## 情報制約と継続方策の版

継続は自分`expert`、相手`tactical/disrupt`の公開観測版に固定する。
全情報版の`expert@...`や計画選び直しをこの価値と混用しない。
通常ターンの探索後に強制交代が残った場合、その選択を参照方策で解決してから通常判断の価値を呼ぶ。
こちらの強制交代を比較するときも、実際の交代・設置技・特性・バトンのイベントを公開観測へ反映する。
Showdown上で自分がp2の場合、元ログを自分=p1へ正規化する。

接続検証で次の問題が見つかったため、参照方策は`public-reference-v2`へ更新した。

- 復元時に実戦のニックネームが変わると、その後の公開HP・技等を追跡できない。
  場の1体の名前だけを対応づけ、相手の本当の種族・型・控えを公開観測へ追加しない。
- 未公開の拘束特性で交代が拒否された後に、継続が失敗する。
  実際に拒否されてサーバーから更新された本人のrequestで指し直す。相手の未公開特性を先に読まない。
  比較対象として強制したrootの候補手は別の手へすり替えず、使えない候補は失敗として残す。
  拒否された途中の行動試行は、実行済み判断として学習データへ出力しない。

旧`terminal-value-v1`はv1参照用の過去実験であり、新しい継続との組合わせは起動時に拒否する。
v2で学習・開発の終局データを再生成し、同じ特徴量・既定ハイパーパラメータ・3seedで再学習したものを使う。
公開リプレイの原ログを読む処理にはニックネームの復元問題はなく、元の公開コーパスを再利用する。

## 検証と記録

```bash
PYTHONHASHSEED=0 uv run python -m pokesim.evaluation.value_search \
  --value-run runs/terminal-value-v2 --split calibration \
  --out runs/value-search-reference-v2-validation
```

モデル・探索条件を固定してから、学習・epoch選択に使っていない元対戦の別ターンを評価する。
6つの仮説世界から選んだ手を、別の6世界・seed 32〜39・相手2方策で終局まで比較する。
HP基準も同じ公開継続・同じ短期探索を使う。既存の全情報プランナーとのパッケージ全体比較ではない。
探索のseedを変えたときの手の変化と、応答時間も保存する。

このcalibration集合は初版の接続確認でも閲覧しているため、以後はこの接続の開発ゲート用として扱う。
未見の最終testとは呼ばない。元のsplit manifestを書き換えず、使用IDと除外した短い記録を実験manifestへ残す。
失敗・未終局は勝敗に置き換えない。全元対戦での差の上下限と、完了対戦だけの平均・対戦単位の区間を区別する。
実際のラダー勝率は、稼働後の記録から別に評価する。

ラダー記録の`advice[].status.search.value`には、モデル・データID・重みSHA、学習価値を使った葉の数、
終局数・失敗数・推論時間が残る。値がない判断は「モデル使用済み」と数えない。

## Showdownで起動する

通常の運用は[運用手順](operations.md)を参照。ローカルShowdown（8000）と公式ラダーへの接続は別プロセス。
先読みの変更はPythonとNodeワーカーの再起動で反映する。エンジン本体は`942cda71`のまま使う。

まずローカルの接続確認:

```bash
uv run python scripts/ladder.py --local --local-opponent heuristic --battles 2 \
  --planner on --value-run runs/terminal-value-v2 --policy checkpoints/ppo_baton_v2.zip \
  --out data/processed/diagnostics/value-search-local-v2
```

公式ラダー収集:

```bash
PLANNER=on EXTRA="--value-run runs/terminal-value-v2" \
  setsid nohup ./scripts/collect.sh > /dev/null 2>&1 < /dev/null &
```

別worktreeで既存venvを使う場合は`PYTHON=/元リポジトリ/.venv/bin/python`と、
そのworktreeの`src`・ルートを含む`PYTHONPATH`を設定できる。`uv sync`で別venvを作る必要はない。
対戦ログ・停止ファイルは既定で`/data/poke-sim/collect/`。
モデルを外すときは収集を停止し、`EXTRA`から`--value-run`を外して起動し直す。
本体のフォルダは収集実行中に編集しない。

重い探索はラダークライアントのイベントループ外で実行し、WebSocketのkeepaliveを止めない。
Nodeワーカーの返答が30秒を超えた場合は終了させ、その判断を既存方策へ戻す。次の判断でワーカーを起動し直す。
公開対戦・ネットワークが600秒停止した場合も収集のまとまりを終了させ、収集スクリプトが起動し直す。
`--battle-timeout`で後者の秒数を変更できる。

OBSサポートサーバーにも`--planner on --value-run runs/terminal-value-v2`を渡せる。
今回はユーザーの指定によりOBSの起動・再起動は行わず、Showdown側だけを稼働させる。
