# Pokémon Champions シングル 対戦サポートツール 実装計画

## Context
Pokémon Champions のランクマッチ（シングル: 6体見せ 3体選出）で、
OBS に取り込んだ対戦映像から盤面を読み取り、
(a) 相手が選出しそうな 3 体を確率で示し、
(b) 毎ターン「自分の構築で取るべき行動」を確率（推定勝率と方策の分布）で提示するツールを作る。
学習は手元の GPU で行い、最新の上位構築・使用率データを元にシミュレータで自己対戦させる。
作業ディレクトリ `/home/michihito/Working/poke-sim` は空なので、ゼロから構築する。

確認済みの前提
- 対象: Pokémon Champions シングル（3v3）。結果は OBS オーバーレイと Web UI の両方に出す。
- OBS 連携: 映像の画面認識（OCR とテンプレートマッチング）。
- 環境: RTX 5060 Ti 16GB（Blackwell / sm_120 → PyTorch は CUDA 12.8 以降のビルドが必須）、Node v22、システムの Python は 3.14。
  → torch 等の対応状況を考え、**uv で Python 3.12 の venv** を作る。
- Showdown 本家は 2026年4月に Champions 形式（BSS シングル Reg M-A など）を追加済み。
  ただし今のレギュレーション名と format ID は実装時に `data/formats` で確認する。
- 使用率: pokemon-zone.com に Champions ランクマのシーズン別データがある（API は無く HTML。利用規約を確認してからスクレイピング）。Showdown 側の Champions 形式は smogon.com/stats の chaos JSON とリプレイ API から取得できる。

## 全体アーキテクチャ
```
[Switch/スマホ] → キャプチャボード → OBS
                                   │ obs-websocket v5 (GetSourceScreenshot 3〜5fps)
                                   ▼
 vision/  画面種別の判定 → 選出画面の6体認識 / ログOCR / HPバー読み取り
                                   │ イベント (Showdown プロトコル風に正規化)
                                   ▼
 state/   BattleState トラッカー ── 相手の型推定 (ベイズ: 使用率の事前分布 × 観測)
                                   │
          ┌────────────────────────┼─────────────────────────┐
          ▼                        ▼                         ▼
 models/selection          models/policy (方策/価値NN)     calc/ (@smogon/calc)
 相手の選出3体の確率        + 推論時探索 (型を何通りか      ダメージ計算・確定数
 + 自分の推奨選出           サンプルして Showdown シミュで検証)
          └────────────────────────┬─────────────────────────┘
                                   ▼
 api/ FastAPI + WebSocket ──► web/ オーバーレイ (OBS ブラウザソース) & 操作用の Web UI
                                   ▲
 db/ PostgreSQL: 図鑑・技・特性・道具・使用率スナップショット・リプレイ・自分の対戦ログ
                                   ▲
 sim/ ローカルの Pokémon Showdown サーバー + poke-env (自己対戦・RL 学習・評価)
```

## 使う外部ライブラリ・ツール
| 用途 | ライブラリ / ツール |
|---|---|
| バトルシミュレータ | `smogon/pokemon-showdown`（ローカルで `node pokemon-showdown start --no-security`） |
| Python から対戦させる | `poke-env`（Player / Gymnasium 環境、並列対戦） |
| ダメージ計算 | `@smogon/calc`（Node の小さな HTTP サービスにする。Champions 未対応の部分はパッチ） |
| オフライン RL の参考実装・データ | `UT-Austin-RPL/metamon`（Showdown リプレイを一人称の軌跡に変換、BC/オフライン RL） |
| 学習 | PyTorch (cu128)、必要なら `torch.compile`、ログは TensorBoard |
| マスタデータ | Showdown の `data/*.ts`（pokedex, moves, abilities, items, learnsets）と PokeAPI（日本語名・特性の日本語説明） |
| 使用率 | pokemon-zone.com（Champions ランクマ）、smogon.com/stats（Showdown の Champions BSS） |
| リプレイ | `replay.pokemonshowdown.com/search.json?format=...` と `/{id}.json` |
| OBS | `obsws-python`（obs-websocket v5） |
| 画像認識 | OpenCV（テンプレートマッチング・HP バーの色）、PaddleOCR（日本語）、必要に応じて小さな CNN 分類器 |
| DB | PostgreSQL 16（docker compose）、SQLAlchemy 2 + Alembic |
| API / UI | FastAPI + uvicorn + WebSocket、フロントは Vite + React（OBS ブラウザソース用に透過 CSS） |

## ディレクトリ構成（新規）
```
poke-sim/
  pyproject.toml / uv.lock / docker-compose.yml (postgres, showdown, calc)
  external/pokemon-showdown/        # git clone（Champions mod を確認・必要ならパッチ）
  services/calc/                    # @smogon/calc の HTTP ラッパー (Node)
  src/pokesim/
    data/      ingest_showdown.py, ingest_pokeapi.py, scrape_usage.py, fetch_replays.py
    db/        models.py, alembic/
    lookup/    abilities.py, moves.py   # 特性・技の検索と効果のタグ付け（要件8）
    sim/       teams.py (使用率からチームを生成), players.py (heuristic/NN Player), eval.py
    state/     battle_state.py, belief.py (相手の型推定)
    models/    encoders.py, selection.py, policy.py, search.py
    train/     bc.py, selfplay_ppo.py, selection_train.py
    vision/    obs_client.py, scene.py, preview.py, ocr_log.py, hpbar.py, event_parser.py
    api/       server.py
  web/                              # overlay と control UI
  data/raw, data/processed, checkpoints/
```

## フェーズ別の計画

### P0 環境構築
- `uv python install 3.12`、torch cu128 版を入れ、`torch.cuda.get_device_capability()==(12,0)` を確認。
- docker compose で postgres。Showdown を clone して起動し、Champions のシングル形式の ID を `config/formats.ts` から特定する。

### P1 データ基盤と DB（要件7・8）
- テーブル: `species`, `forms`（メガ含む）, `abilities`（効果タグ: 天候, 威嚇, 浮遊, 半減実, 被弾時発動…）, `moves`, `items`, `learnsets`, `type_chart`, `names_i18n`（日英、OCR 照合用）,
  `usage_snapshots`(season, regulation, species, 道具/特性/技/努力値配分/同時選出の分布 JSONB),
  `replays`, `preview_records`（6体見せ→実際に出た3体）, `my_teams`, `my_battles`（OBS で記録した自分の対戦）。
- Showdown の `data/` を Node スクリプトで JSON に出し、Python で取り込む。日本語名と特性の説明は PokeAPI から。
- `lookup/` で「特性・技・道具を名前（日英・表記ゆれ）で検索 → 効果・ダメージ計算への影響・行動への影響」を返す API を作る（UI のホバー表示や型推定から使う）。
- 使用率スクレイパーは日次で動かし、シーズンごとのスナップショットとして保存する。

### P2 シミュレータとベースライン
- `sim/teams.py`: 使用率の分布から「ありそうな 6 体＋型」をサンプリングするチーム生成器と、上位構築を手で登録する仕組み。
- poke-env で Random / MaxDamage / ヒューリスティック（calc で確定数を見る）Player を作り、評価用のリーグ（勝率表）を作る。
- 3v3 の選出はチームプレビュー時の `teamPreview()` を上書きして扱う。

### P3 選出予測モデル（要件3）
- 入力: 相手の 6 体と自分の 6 体（種族・タイプ・種族値・使用率から見た型の埋め込み）。
- 出力: 相手の 3 体の組み合わせ C(6,3)=20 通りの確率と先発の確率。自分の選出 20 通り × 先発 3 通りの推定勝率も出す。
- 学習データ:
  1. Showdown の Champions BSS リプレイ（プレビュー→場に出た個体。出なかった個体は部分観測として扱う）
  2. pokemon-zone の同時選出率
  3. 自己対戦の結果（選出ごとの勝率）
- モデル: Set Transformer（6体の順序に依存しない）、数十 MB 規模なので 16GB で十分。
- 評価: 相手の 3 体の top-1 / top-3 の的中率、対数尤度。

### P4 行動の最適化（要件4・6）
1. **相手の型推定 `state/belief.py`**: 使用率を事前分布にして、観測（受けたダメージを calc で逆算、素早さの順、判明した技・特性・道具、メガの有無）でベイズ更新する。粒子（型の候補）を N 個持つ。
2. **方策/価値ネットワーク**: 盤面（両者の場と控え・HP・状態・天候/場・判明した情報＋推定の分布）→ 行動（技 4 / 交代 2 / メガ等の有無）の確率と勝率。
3. **学習の段階**:
   - (a) 行動クローニング: metamon の手法でリプレイを一人称の軌跡に変換して教師あり学習。
   - (b) PPO 自己対戦: poke-env で並列に対戦（シミュレータは CPU 側で数十プロセス、推論は GPU でまとめて処理）。相手は過去のチェックポイント＋ヒューリスティック bot のリーグ。相手のチームは P2 の生成器から最新の使用率で生成する。
4. **推論時の探索 `models/search.py`**: 粒子から相手の型を K 通りサンプル → 各世界で Showdown のバトルを複製して、方策を事前分布とした 1〜2 ターン先の探索（同時手番なので行列ゲームとして解く）→ 行動ごとの期待勝率と推奨の確率分布を出す。時間制限（例: 2〜5 秒）付き。
- 出力例: 「ハバタクカミ: ムーンフォース 62% / 交代→サーフゴー 28% …（推定勝率 58%）」と、その理由（確定数、相手の想定する型の上位）。

### P5 OBS の画面認識（要件1）
- `obs_client.py`: obs-websocket v5 の `GetSourceScreenshot` でキャプチャソースを 3〜5fps 取得する。
- `scene.py`: 画面種別の分類（プレビュー / 行動選択 / 演出中 / 交代選択 / 結果）。まずは固定 ROI と特徴的な UI 要素のテンプレートで、足りなければ小さな CNN。
- `preview.py`: 相手 6 体のアイコンをテンプレートマッチング（実際のキャプチャからテンプレート集を作るツールも用意）。
- `ocr_log.py`: バトルメッセージを PaddleOCR (ja) で読み、`names_i18n` で正規化して「相手の○○の××！」「効果は抜群だ」「○○の特性△△」などを正規表現でイベント化する。
- `hpbar.py`: 相手の HP バーは緑/黄/赤の画素比率で %、自分の HP は数値を OCR。
- `event_parser.py`: Showdown プロトコル風のイベントに変換して `BattleState` に流す。シミュレータと同じ状態表現を使うことで、モデルをそのまま使い回せる。
- 開発時は録画ファイルを再生してオフラインでテストする（記録した対戦は `my_battles` に保存し、学習データにも回す）。

### P6 オーバーレイと Web UI
- FastAPI が状態と推論結果を WebSocket で配信する。
- `web/overlay`: 透過背景、選出予測と行動の推奨を短く表示（OBS ブラウザソースに設定）。
- `web/control`: 認識結果を手で修正する UI（誤認識の補正が実用上必須）、自分の構築の登録、特性・技の検索。

### P7 レギュレーション更新への追従
- 使用率の取得 → チーム生成器の更新 → 選出モデルの再学習 → PPO のファインチューニング、を 1 コマンドで流すパイプライン（夜間に回す）。

## 進め方の優先順位
P0→P1→P2→P3 を先に作れば、手入力でも「選出予測＋ダメージ計算」の形で使える。
P4（RL）と P5（画面認識）は並行して進められる。P4 は時間がかかるので、最初はヒューリスティック＋探索で動くものを出し、学習済みの方策に順に置き換える。

## リスク・注意点
- Showdown の Champions 実装と実機の仕様の差（メガ等の仕様、新しい技・特性）。実機のログと突き合わせて差分を mod でパッチする。
- @smogon/calc が Champions 固有の要素に未対応の可能性がある → ラッパー側で補う。
- pokemon-zone の利用規約・アクセス頻度に注意する（日次 1 回、キャッシュ）。
- 公式大会では対戦中の外部ツールによる支援は禁止されている。リアルタイムの支援は練習・振り返り・配信での解説用途に限るのが無難。

## 検証方法
- P0: `python -c "import torch; print(torch.cuda.is_available())"`、Showdown のローカルサーバーで Champions 形式の対戦が作れること。
- P1: `pytest` で取り込み件数・日英名の対応・特性の検索（例:「いかく」「Intimidate」が同じ結果を返す）。
- P2/P4: `sim/eval.py` で各 bot 同士を 1000 戦させた勝率表。NN 方策が MaxDamage とヒューリスティック bot に有意に勝ち越すこと。
- P3: 取っておいたリプレイでの選出の top-3 的中率（ランダムの 3/20 と比べる）。
- P5: 録画した対戦を再生してイベント列を抽出し、手でラベルを付けたものとの一致率（プレビューの 6 体認識 95% 以上が目安）。
- E2E: OBS で録画を再生 → オーバーレイに選出予測と各ターンの推奨が数秒以内に出ること。
