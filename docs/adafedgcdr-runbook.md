# AdaFedGCDR 运行手册（自适应冷启动验证版）

本手册描述 `model/adafedgcdr/` 离线实验管线的输入、命令、输出与结论边界。该管线验证的问题是：目标域零交互的用户能否利用可靠的源域知识获得更好的推荐，以及随着目标域交互增长、目标域偏好权重上升后推荐是否继续改善，同时低可靠源域是否被降权。

**范围声明：** 本管线只产出离线可复现实验结果，不将任何结果称为线上效果，也不声称新增任何隐私保证。原始 `model/fedgcdr/` 保持为论文基线实现，本管线只复用其 `Client`、`Server`、`MLP`、`GAT`，不修改其算法定义，也不导入会在加载时执行训练的 `main.py`。

## 环境准备

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv venv --python 3.11 .venv
UV_CACHE_DIR="$PWD/.uv-cache" uv lock
UV_CACHE_DIR="$PWD/.uv-cache" uv sync --extra dev
.venv/bin/python -c 'import sys; print(sys.version_info[:2])'   # 预期 (3, 11)
```

所有训练与测试都必须在 `.venv` 中进行，不向全局 Python 安装依赖。运行测试：

```bash
.venv/bin/python -m pytest tests -q
```

## 获取 Amazon 原始数据并转换

原始数据来自 UCSD Amazon Review Data (2018)（Ni et al., EMNLP 2019）：https://nijianmo.github.io/amazon/index.html 。使用按领域的 "ratings only" CSV（无表头，四列 `item,user,rating,timestamp`，时间戳为 Unix **秒**），例如：

- Books：https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Books.csv
- CDs and Vinyl：https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/CDs_and_Vinyl.csv
- Movies and TV：https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Movies_and_TV.csv

其余领域的文件名与 `legacy/Data_Proc.py` 中的列表一致，URL 前缀均为 `https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/`。下载后放入本地目录（如 `data/raw/`，不入 Git），再用转换脚本生成契约 CSV：

```bash
.venv/bin/python scripts/prepare_amazon_events.py \
  --domain Books=data/raw/Books.csv --domain CDs=data/raw/CDs_and_Vinyl.csv \
  --target Books --user-core 10 --item-core 10 \
  --out data/private/events.csv
```

- 每行评分视为一次正向交互，不按评分过滤；时间戳自动乘 1000 转为毫秒；`--domain` 可重复多次加入更多源域，支持 `.gz`。
- `--user-core` / `--item-core` 控制每领域用户/物品最少交互数，可用 `--user-core Books=96` 形式对单域覆盖（论文参考阈值见 `legacy/Data_Proc.py`）。Books 全量约 5100 万行，不过滤会让 `train-base` 不可行。
- 脚本不去重：重复 `(user_id, domain, item_id)` 由管线 `load_events` 保留最早时间戳；core 过滤前的计数也包含全部行。
- 脚本结束时输出总用户数、桥接用户数与预估合格用户数（未去重口径，权威口径以 `prepare` 的审计文件为准）。合格用户过少时先调整领域组合或阈值，再进入四命令流程。

## 输入数据契约

输入为带时间戳的本地 CSV，默认放在 `data/private/events.csv`（该目录已被 `.gitignore` 排除，数据不得提交到 Git）：

```csv
user_id,domain,item_id,timestamp_ms
u001,CDs,B000001,1262304000000
u001,Books,B000002,1293840000000
u001,Books,B000003,1293840060000
```

- 列名固定为 `user_id,domain,item_id,timestamp_ms`；`user_id`、`item_id`、`domain` 按字符串处理，`timestamp_ms` 为整数 Unix 毫秒（无法转整数时按行号报错）。
- 一行代表一次正向交互；重复的 `(user_id, domain, item_id)` 只保留时间最早的一条。
- `domain` 使用数据源中的领域名（如 Amazon 的 `Books`、`CDs`）。
- 若手头只有原仓库的 `implicit.json`：它没有时间戳契约。只有在确认其列表顺序确实代表发生时间后才能转成上述 CSV；不能核实时，实验名称必须写成“稀疏历史模拟”，不得称为“随时间积累交互”的验证。

## 时间截断与用户隔离

- **源域时间截断：** 每个用户的源域历史只使用其**第一次目标域交互之前**（`timestamp_ms` 严格小于首次目标域时间戳）的事件，保证冷启动评估时这些行为已经存在。完全没有目标域事件的用户不受截断，全部源域历史可用。A/B/C/D 四组用户应用同一规则，避免训练与推理输入定义不一致。
- **用户分组：** 桥接用户按 `sha256(f"{seed}:{user_id}")` 前 8 字节映射到 `[0,0.6)/[0.6,0.8)/[0.8,0.9)/[0.9,1)`，落入 `base/fusion/validation/test`（A 60% / B 20% / C 10% / D 10%），名单写入 `split.json`，四组互不交叉。
- **数据用途隔离：**
  - 基础模型（源域 GAT + 目标域模型）只使用 A 组的目标域物品与交互标签；B/C/D 的目标域物品与标签不得进入基础模型、物品向量或早停。B/C/D 的首个目标域时间只能用于截断其源域历史。
  - 目标域物品目录由 A 组的目标域交互建立；目录外的评估标签计入“不可推荐”覆盖率，不混入排序指标。
  - B 组只训练融合器参数；C 组只用于早停与超参数选择，不更新参数；D 组只用于最终评估，不能进入训练函数（代码层面会拒绝 `test` 组用例）。
  - 测试用户的目标域 ID embedding 不从原始模型读取；新用户表示只由源域向量、目标域历史物品向量与融合门控构成。
- **合格用户：** 只有目标域交互 ≥ 11 条且可用源域交互 ≥ 1 条的用户进入主评估（k 最大为 10，需要预测第 11 条）；其余用户的数量与原因在 `prepare` 的审计文件中单独报告。

## 阶段定义（k 阈值）

同一用户取 `k ∈ {0,1,3,5,10}` 条目标域历史构造状态，预测其第 `k+1` 条目标域交互。目标域历史按 `timestamp_ms` 升序，同一时间戳用 `item_id` 升序打破平局；下一条事件只作为标签。`k=0` 即零交互冷启动，此时目标域融合权重精确为 0；完全没有源域但 `k>0` 的用户目标域权重强制为 1。

## 四个命令

统一入口：`python -m model.adafedgcdr prepare|train-base|train-fusion|evaluate`（以下用 `.venv/bin/python -m model.adafedgcdr`）。所有命令都会校验数据与拆分的 SHA-256：`train-base` 对照 `prepare` 写出的审计文件（防止 prepare 之后数据被改动），`train-fusion` 与 `evaluate` 对照 checkpoint 中保存的哈希，不匹配时直接报错退出。

### 1. prepare —— 拆分与数据审计

```bash
.venv/bin/python -m model.adafedgcdr prepare \
  --events data/private/events.csv --target Books --seed 2026 --split artifacts/split.json
```

| 参数 | 说明 |
|---|---|
| `--events` | 输入 CSV 路径 |
| `--target` | 目标域名称（源域 = 其余全部领域） |
| `--seed` | 拆分种子（默认 2026） |
| `--split` | 输出拆分清单路径 |

输出：`artifacts/split.json`（`target_domain`、`seed`、四组用户名单，按字典序）与 `artifacts/split.json.audit.json`（用户总数、合格数、排除原因计数、分组大小、数据/拆分 SHA-256）。

### 2. train-base —— 训练并导出基础表示

```bash
.venv/bin/python -m model.adafedgcdr train-base \
  --events data/private/events.csv --split artifacts/split.json \
  --source-rounds 150 --target-rounds 150 --device cpu --out artifacts/base.pt
```

| 参数 | 说明 |
|---|---|
| `--events` / `--split` | 输入数据与拆分清单 |
| `--source-rounds` | 各源域 `kt_stage(False)` 的固定轮数（不做测试集选轮） |
| `--target-rounds` | 目标域 `kt_stage(True)` 的固定轮数 |
| `--embedding-size` | 嵌入维度（默认 16） |
| `--local-epoch` | 本地训练轮数（默认 3） |
| `--user-batch` | 用户批大小（默认 16） |
| `--device` | `cpu` 或 `cuda:N` |
| `--dp` / `--no-dp` | 导出时是否加差分隐私噪声（默认开） |
| `--out` | 输出 checkpoint 路径 |

`base.pt` 字段：`domain_names`（源域名单，与向量第 1 维对齐）、`source_vectors[user_id]`（`[源域数, D]`）、`source_counts`、`source_mask`（缺失源域为 `False` 且向量为零，不把噪声当历史）、`target_items`（字典序物品→索引）、`target_item_vectors`、`target_train_users`（仅 A 组）、`split_sha256`，以及完整配置、种子与数据 SHA-256。**不保存任何 B/C/D 的目标域用户 embedding。**

### 3. train-fusion —— 按冷启动阶段训练融合器

```bash
.venv/bin/python -m model.adafedgcdr train-fusion \
  --events data/private/events.csv --split artifacts/split.json --base artifacts/base.pt \
  --candidate-count 100 --out artifacts/fusion.pt
```

| 参数 | 说明 |
|---|---|
| `--base` | 基础表示 checkpoint（会被哈希校验） |
| `--candidate-count` | 每个样本的候选数（负物品数 + 1 个正物品，默认 100） |
| `--lr` / `--max-epochs` / `--patience` | 学习率（默认 1e-3）、最大 epoch（默认 50）、早停耐心（默认 5） |
| `--batch-size` / `--device` | 批大小（默认 64）、设备 |
| `--out` | 输出 checkpoint 路径 |

训练只优化 `AdaptiveFusion` 参数，基础向量全部冻结；损失按 k 分阶段取均值后五阶段等权平均，样本数不均不会改变阶段权重；早停与最优 checkpoint 依据 C 组验证 NDCG@10。训练日志逐 epoch 输出五阶段验证指标，并在结尾输出早停轮次与随机种子。

### 4. evaluate —— 测试组对照评估

```bash
.venv/bin/python -m model.adafedgcdr evaluate \
  --events data/private/events.csv --split artifacts/split.json \
  --base artifacts/base.pt --fusion artifacts/fusion.pt \
  --candidate-count 100 --out artifacts/metrics.json
```

| 参数 | 说明 |
|---|---|
| `--fusion` | 融合器 checkpoint（会被哈希校验） |
| `--report` | 报告输出路径（默认与 `--out` 同目录的 `report.md`） |
| 其余 | 同上 |

输出 `metrics.json` 与 `report.md`。所有方法使用完全相同的用户、标签、候选集与目标域目录。

## 负采样与候选集

- 每个正样本配 `candidate_count - 1` 个负物品：来自目标域目录、且该用户**从未交互过**（含未来事件），候选列表最后一位固定为正物品，损失与排序的正样本索引即 `candidate_count - 1`。
- 相同 `(seed, user_id, k)` 必然产生相同候选；不同 k 的候选相互独立。
- 可用负物品不足时报错退出，通过调低 `--candidate-count` 解决（合成小目录即属此情形）。

## 指标定义

- **HR@5/10、NDCG@5/10**：正物品排名（同分按候选顺序打破平局）；NDCG 折扣为 `1/log2(rank+2)`。
- **覆盖率@10**：该方法在该 k 的全部 top-10 列表中出现过的物品数占目标域目录的比例。
- **置信区间**：用户级 bootstrap，重采样 1000 次、经验百分位 95% 区间；固定种子后完全可复现。方法间比较用同一批用户的配对差值 bootstrap。
- **五 k 用户等权平均 NDCG@10**：先对每个用户在参与的 k 上取均值，再对用户等权平均。`target-only` 在 `k=0` 不适用（不填零混入平均），故不参与五 k 平均比较。
- **负迁移用户比例**：同一 k 下 `full` 的 NDCG@10 低于 `target-only` 的用户占比；整体口径为 k∈{1,3,5,10} 的用户等权均值对比。
- **不可推荐比例**：标签不在 A 组建立的目标域目录中的样本占比（单独计数，不混入排序指标）。
- **缺失源域比例**：每条样本缺失源域数占源域总数的均值。

## 基线与消融（同一候选集）

| 方法 | 源域权重 | 融合权重 |
|---|---|---|
| `source-only` | 等权 | 0（纯源域） |
| `target-only` | — | 1（仅 k>0；k=0 不适用） |
| `fixed-0.5` | 等权 | 固定 0.5 |
| `dynamic-only` | 等权 | 交互数门控 `k/(k+softplus(τ))` |
| `reliability-only` | 可靠性门控 | 固定 0.5 |
| `full` | 可靠性门控 | 交互数门控 |

所有融合基线在 `k=0` 将目标域权重设为 0；完全缺失源域且 `k>0` 时目标域权重强制为 1。无个性化信号（既无源域也无历史）的样本由评估层按目标域流行度（A 组目标域交互频次）排序，不用随机向量打分。

## 有效性判据（预先固定）

“方法有效”当且仅当同时满足：

1. 五个 k 的用户等权平均 NDCG@10 高于最强**可比**基线（可比 = 五 k 均有定义：`source-only`、`fixed-0.5`、`dynamic-only`、`reliability-only`）；
2. 配对用户 bootstrap 的差值 95% 区间下界大于 0；
3. `k=0` 的 NDCG@10 不低于 `source-only`。

未达到时如实报告，不通过调测试集阈值改变结论。

## 本地隐私边界

- 源域向量导出时先按 `Client.l2_clip` 裁剪，`--dp` 开启则按当前 `eps`（默认 8）、`delta`（默认 1e-5）公式**加噪一次**并缓存进 `base.pt`；评估阶段不会重复生成新噪声。
- 这是集中式离线导出，仅供本地研究；不能据此宣称上线后的联邦隐私保证。把 `base.pt` 中的用户表示导出到生产服务器需要单独的隐私设计与预算核算。
- `base.pt`、`fusion.pt`、`metrics.json` 等产物都写入被忽略的 `artifacts/`，不入 Git。

## 已知限制与结论边界

- **样本选择偏差：** Amazon 数据的 user-core/item-core 过滤会留下更活跃的用户与更热门的物品，实验样本不代表真实用户分布。
- **真实新用户更稀疏：** 真实刚进入目标域的用户历史可能短于过滤阈值，主评估覆盖的“≥11 条目标域交互”用户不一定代表最稀疏的真实冷启动用户。
- **离线结论：** 一切指标都是离线评估，不是线上效果；本管线未新增隐私保证。
- **与旧 FedGCDR 不同表：** 旧评估依赖已训练的目标域用户 ID embedding，不与零交互结果同表比较；若未来加入归纳式适配器，必须沿用同一用户隔离协议。
- **图扩展门控是独立扩展：** 把可靠性权重接入 `GATLayer` 注意力的方案只有在本版显示稳定收益后才另行开计划实施；`GAT.forward` 现在并未使用传入的 `domain_attention` 参数，不能当作已完成的可靠性门控。

## 合成数据烟雾测试

仓库不含真实数据时，用合成数据验证管线完整性（**合成数字只用于烟雾测试，不是任何效果指标**）：

```bash
.venv/bin/python tests/make_synthetic_events.py
.venv/bin/python -m model.adafedgcdr prepare --events artifacts/synthetic/events.csv --target Books --seed 2026 --split artifacts/synthetic/split.json
.venv/bin/python -m model.adafedgcdr train-base --events artifacts/synthetic/events.csv --split artifacts/synthetic/split.json --source-rounds 1 --target-rounds 1 --device cpu --out artifacts/synthetic/base.pt
.venv/bin/python -m model.adafedgcdr train-fusion --events artifacts/synthetic/events.csv --split artifacts/synthetic/split.json --base artifacts/synthetic/base.pt --candidate-count 10 --out artifacts/synthetic/fusion.pt
.venv/bin/python -m model.adafedgcdr evaluate --events artifacts/synthetic/events.csv --split artifacts/synthetic/split.json --base artifacts/synthetic/base.pt --fusion artifacts/synthetic/fusion.pt --candidate-count 10 --out artifacts/synthetic/metrics.json
```

预期：四命令退出码 0；`metrics.json` 含五个 k、无 NaN、六个方法齐全。合成数据的行为是确定性构造的，不包含真实的跨域迁移结构，因此 `report.md` 的判据结论只反映管线正确性。
