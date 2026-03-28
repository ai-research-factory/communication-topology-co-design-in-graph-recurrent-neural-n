# Communication Topology Co-Design in Graph Recurrent Neural Network Based Distributed Control

## Project ID
proj_c589ff16

## Taxonomy
GNN

## Current Cycle
2

## Objective
Implement, validate, and iteratively improve the paper's approach with production-quality standards.


## Design Brief
### Problem
This paper addresses the challenge of designing distributed controllers for multi-agent systems, where both the control logic for each agent and the communication network connecting them are critical for performance. Traditional methods often pre-define a fixed communication topology, which may be suboptimal or overly dense, leading to high communication costs. The paper proposes a novel approach to solve this by co-designing the controller and the communication topology simultaneously.

The core idea is to parameterize a distributed controller using a Graph Recurrent Neural Network (GRNN), where the graph's adjacency matrix itself is a learnable parameter. By applying L1 regularization to this adjacency matrix during training, the system is encouraged to find a sparse communication graph, effectively pruning unnecessary communication links. This allows for an end-to-end optimization that jointly discovers an efficient control policy and a cost-effective communication network, explicitly managing the trade-off between control performance and communication sparsity.

### Datasets
Synthetic multi-agent consensus dataset. To be generated in `src/data.py`. The generator will create episodes with N agents, each with a state vector. The initial states are sampled randomly, and the goal is for all agents to reach a consensus state (e.g., the average of their initial states).

### Targets
Minimize a composite loss function: `Loss = PerformanceLoss + lambda * L1_norm(AdjacencyMatrix)`.
- The `PerformanceLoss` measures how well the agents achieve their control objective (e.g., MSE of agent states from the consensus value).
- The `L1_norm` term penalizes dense communication graphs.

### Model
The model is a Graph Recurrent Neural Network (GRNN). At each time step, agent states are updated using a GNN layer. The crucial aspect is that the GNN's adjacency matrix is not fixed but is a set of learnable parameters, representing the weights of communication links. An L1 penalty is applied to these weights during training, which forces many of them to zero, resulting in a sparse, learned communication graph. The GRNN's hidden state captures the temporal dynamics of the control system.

### Training
Training is performed over multiple simulated episodes. In each episode, a set of agents is initialized with random states. The GRNN controller is unrolled for a fixed number of time steps, generating control actions. The performance loss is calculated over the trajectory (e.g., sum of squared distances from the consensus state). This is combined with the L1 regularization term on the adjacency matrix to form the total loss, which is then backpropagated through time to update the GRNN weights and the adjacency matrix.

### Evaluation
The primary evaluation is to generate a trade-off curve plotting control performance against communication sparsity. This is achieved by training and evaluating the model across a range of L1 regularization strengths (lambda). For each lambda, the final trained model is evaluated on a set of unseen initial conditions. Performance is measured by metrics like final consensus error or settling time. Sparsity is measured by the percentage of zero-weight edges in the learned adjacency matrix. The resulting curve demonstrates the effectiveness of the co-design approach.


## データ取得方法（共通データ基盤）

**合成データの自作は禁止。以下のARF Data APIからデータを取得すること。**

### ARF Data API
```bash
# OHLCV取得 (CSV形式)
curl -o data/aapl_1d.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=AAPL&interval=1d&period=5y"
curl -o data/btc_1h.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=BTC/USDT&interval=1h&period=1y"
curl -o data/nikkei_1d.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=^N225&interval=1d&period=10y"

# JSON形式
curl "https://ai.1s.xyz/api/data/ohlcv?ticker=AAPL&interval=1d&period=5y&format=json"

# 利用可能なティッカー一覧
curl "https://ai.1s.xyz/api/data/tickers"
```

### Pythonからの利用
```python
import pandas as pd
API = "https://ai.1s.xyz/api/data/ohlcv"
df = pd.read_csv(f"{API}?ticker=AAPL&interval=1d&period=5y")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")
```

### ルール
- **リポジトリにデータファイルをcommitしない** (.gitignoreに追加)
- 初回取得はAPI経由、以後はローカルキャッシュを使う
- data/ディレクトリは.gitignoreに含めること



## ★ 今回のタスク (Cycle 2)


### Phase 2: 複合損失関数と学習ループの実装 [Track ]

**Track**:  (A=論文再現 / B=近傍改善 / C=独自探索)
**ゴール**: 性能損失とL1正則化項を組み合わせた複合損失関数を用いて、完全な学習ループを実装する。

**具体的な作業指示**:
1. `src/training.py`に`train_epoch`関数を実装する。この関数は、シミュレーションエピソードを複数回実行する。
2. 各エピソード内で、`GRNNController`をTタイムステップにわたって展開し、状態の軌跡を記録する。
3. 性能損失を計算する。例：全エージェントの状態がその平均値（コンセンサス目標）からどれだけ離れているかの二乗誤差の総和。
4. 複合損失を `loss = performance_loss + lambda * torch.norm(model.A, p=1)` として計算する。
5. `loss.backward()`を呼び出し、オプティマイザでモデルの重みと隣接行列`A`を更新する。
6. `scripts/train.py`を作成し、この学習プロセスを実行するCLIを実装する。学習中の損失と隣接行列のL1ノルムをログに出力する。

**期待される出力ファイル**:
- src/training.py
- scripts/train.py
- reports/cycle_2/training_log.txt

**受入基準 (これを全て満たすまで完了としない)**:
- 学習ループがエラーなく完了する
- 訓練損失がエポックを通じて減少傾向を示す
- 隣接行列`A`のL1ノルムが学習の進行とともに変化する




## データ問題でスタックした場合の脱出ルール

レビューで3サイクル連続「データ関連の問題」が指摘されている場合:
1. **データの完全性を追求しすぎない** — 利用可能なデータでモデル実装に進む
2. **合成データでのプロトタイプを許可** — 実データが不足する部分は合成データで代替し、モデルの基本動作を確認
3. **データの制約を open_questions.md に記録して先に進む**
4. 目標は「論文の手法が動くこと」であり、「論文と同じデータを揃えること」ではない







## 全体Phase計画 (参考)

✓ Phase 1: 環境とGRNNコアモデルの実装 — エージェントのダイナミクスをシミュレートする環境と、学習可能な隣接行列を持つ基本的なGRNNモデルを実装する。
→ Phase 2: 複合損失関数と学習ループの実装 — 性能損失とL1正則化項を組み合わせた複合損失関数を用いて、完全な学習ループを実装する。
  Phase 3: 評価フレームワークとベースライン指標の実装 — 学習済みモデルの性能とグラフのスパース性を測定する評価フレームワークを実装する。
  Phase 4: 性能-スパース性トレードオフ曲線の再現 — L1正則化係数λを変化させてモデルの学習と評価を繰り返し、論文の中心的な結果である性能対スパース性のトレードオフ曲線をプロットする。
  Phase 5: GRNNハイパーパラメータ最適化 — Optunaを用いて、固定されたスパース性レベルでの性能を最大化するGRNNのハイパーパラメータ（隠れ層の次元、学習率）を探索する。
  Phase 6: システム規模に対するロバスト性検証 — 学習時と異なるエージェント数でモデルを評価し、学習した制御ポリシーの般化性能を検証する。
  Phase 7: 代替システムダイナミクスの導入 — より複雑な二重積分器ダイナミクスを環境に実装し、提案手法が同様に機能するかを検証する。
  Phase 8: 代替GNN層（GAT）の実験 — Graph Attention Network (GAT) をGRNNのコア層として使用し、性能と学習されるトポロジーがどう変化するかを調査する。
  Phase 9: 学習済みトポロジーの分析 — 異なるスパース性レベルで学習された通信グラフのトポロジー的特性を分析する。
  Phase 10: 包括的なテクニカルレポートの生成 — 全フェーズの結果を統合し、論文再現の成果をまとめた包括的なテクニカルレポートを生成する。
  Phase 11: 最終化とエグゼクティブサマリー — コードの最終的なクリーンアップ、テストカバレッジの向上、および非技術者向けの日本語エグゼクティブサマリーを作成する。


## 評価原則
- **主指標**: Sharpe ratio (net of costs) on out-of-sample data
- **Walk-forward必須**: 単一のtrain/test splitでの最終評価は不可
- **コスト必須**: 全メトリクスは取引コスト込みであること
- **安定性**: Walk-forward窓の正の割合を報告
- **ベースライン必須**: 必ずナイーブ戦略と比較

## 再現モードのルール（論文忠実度の維持）

このプロジェクトは**論文再現**が目的。パフォーマンス改善より論文忠実度を優先すること。

### パラメータ探索の制約
- **論文で既定されたパラメータをまず実装し、そのまま評価すること**
- パラメータ最適化を行う場合、**論文既定パラメータの近傍のみ**を探索（例: 論文が12ヶ月なら [6, 9, 12, 15, 18] ヶ月）
- 論文と大きく異なるパラメータ（例: 月次論文に対して日次10営業日）で良い結果が出ても、それは「論文再現」ではなく「独自探索」
- 独自探索で得た結果は `customMetrics` に `label: "implementation-improvement"` として記録し、論文再現結果と明確に分離

### データ条件の忠実度
- 論文のデータ頻度（日次/月次/tick）にできるだけ合わせる
- ユニバース規模が論文より大幅に小さい場合、その制約を `docs/open_questions.md` に明記
- リバランス頻度・加重方法も論文に合わせる



## 禁止事項
- 未来情報を特徴量やシグナルに使わない
- 全サンプル統計でスケーリングしない (train-onlyで)
- テストセットでハイパーパラメータを調整しない
- コストなしのgross PnLだけで判断しない
- 時系列データにランダムなtrain/test splitを使わない
- APIキーやクレデンシャルをコミットしない
- **新しい `scripts/run_cycle_N.py` や `scripts/experiment_cycleN.py` を作成しない。既存の `src/` 内ファイルを修正・拡張すること**
- **合成データを自作しない。必ずARF Data APIからデータを取得すること**
- **「★ 今回のタスク」以外のPhaseの作業をしない。1サイクル=1Phase**
- **論文が既定するパラメータから大幅に逸脱した探索を「再現」として報告しない**

## Git / ファイル管理ルール
- **データファイル(.csv, .parquet, .h5, .pkl, .npy)は絶対にgit addしない**
- `__pycache__/`, `.pytest_cache/`, `*.pyc` がリポジトリに入っていたら `git rm --cached` で削除
- `git add -A` や `git add .` は使わない。追加するファイルを明示的に指定する
- `.gitignore` を変更しない（スキャフォールドで設定済み）
- データは `data/` ディレクトリに置く（.gitignore済み）
- 学習済みモデルは `models/` ディレクトリに置く（.gitignore済み）

## 出力ファイル
以下のファイルを保存してから完了すること:
- `reports/cycle_2/metrics.json` — 下記スキーマに従う（必須）
- `reports/cycle_2/technical_findings.md` — 実装内容、結果、観察事項

### metrics.json 必須スキーマ
```json
{
  "sharpeRatio": 0.0,
  "annualReturn": 0.0,
  "maxDrawdown": 0.0,
  "hitRate": 0.0,
  "totalTrades": 0,
  "transactionCosts": { "feeBps": 10, "slippageBps": 5, "netSharpe": 0.0 },
  "walkForward": { "windows": 0, "positiveWindows": 0, "avgOosSharpe": 0.0 },
  "customMetrics": {}
}
```
- 全フィールドを埋めること。Phase 1-2で未実装のメトリクスは0.0/0で可。
- `customMetrics`に論文固有の追加メトリクスを自由に追加してよい。
- `docs/open_questions.md` — 未解決の疑問と仮定
- `README.md` — 今回のサイクルで変わった内容を反映して更新（セットアップ手順、主要な結果、使い方など）
- `docs/open_questions.md` に以下も記録:
  - ARF Data APIで問題が発生した場合（エラー、データ不足、期間の短さ等）
  - CLAUDE.mdの指示で不明確な点や矛盾がある場合
  - 環境やツールの制約で作業が完了できなかった場合

## 標準バックテストフレームワーク

`src/backtest.py` に以下が提供済み。ゼロから書かず、これを活用すること:
- `WalkForwardValidator` — Walk-forward OOS検証のtrain/test split生成
- `calculate_costs()` — ポジション変更に基づく取引コスト計算
- `compute_metrics()` — Sharpe, 年率リターン, MaxDD, Hit rate算出
- `generate_metrics_json()` — ARF標準のmetrics.json生成

```python
from src.backtest import WalkForwardValidator, BacktestConfig, calculate_costs, compute_metrics, generate_metrics_json
```

## Key Commands
```bash
pip install -e ".[dev]"
pytest tests/
python -m src.cli run-experiment --config configs/default.yaml
```

Commit all changes with descriptive messages.
