# AdaFedGCDR 自适应冷启动验证版 Implementation Plan

**Goal:** 构建可复现的离线实验，验证目标域零交互时利用可靠源域知识、交互增加后提高目标域偏好权重，并抑制低可靠迁移是否改善推荐。

**Architecture:** 保留 `model/fedgcdr/` 作为原始基线，新增独立的 `model/adafedgcdr/` 实验管线。先用互不重叠的桥接用户训练基础表示、融合器及评估模型，再按用户的目标域交互前缀模拟冷启动过程。第一版在表示融合处控制源域参与程度；只有确认收益后，才把同一可靠性权重接入图注意力。

**Tech Stack:** Python 3.11、PyTorch、NumPy、pytest、uv；输入为带时间戳的本地 CSV，输出为本地 PyTorch checkpoint、JSON 指标和 Markdown 报告。

## Global Constraints

- 本计划只覆盖离线可验证版本，不将实验结果称为线上效果，也不声称新增隐私保证。
- 任何依赖安装、训练或测试前，先进入项目 `.venv`；不向全局 Python 安装依赖。
- 保留原始 `model/fedgcdr/` 行为；实验脚本不导入会在 import 时直接执行训练的 `main.py`。
- 基础模型只能用基础训练组的目标域物品和交互标签优化参数；B/C/D 的首个目标域时间仅可用于截断其源域历史，B/C/D 的目标域物品和标签不得用于基础模型、物品向量或早停。
- 源域信息只使用评估时点之前可得的交互；测试用户的目标域 ID embedding 不得从原始模型中读取。
- 每个训练任务结束时创建中文 Git 提交，正文用分条列表；本计划文档本身不自动提交。
- 仓库目前没有 `data/`、依赖文件或预处理脚本。真实数据验证以带时间戳的输入到位为前提；缺数据时仅运行合成数据测试，不填造指标。

---

## 范围与验收

验证对象是**源域已有行为、目标域刚进入或行为很少的用户**。同一用户使用 `k ∈ {0,1,3,5,10}` 条目标域历史构造状态，预测其第 `k+1` 条目标域交互。目标域历史按 `timestamp_ms` 升序；对同一时间戳用 `item_id` 升序打破平局。只有至少 11 条目标域交互、至少 1 条可用源域交互的用户进入主评估；其余用户数量与原因单独报告。

主要验收：

1. 无用户交叉、无目标域未来交互泄漏；`k=0` 时目标域权重精确为 0。
2. 给定相同输入、锁文件和种子，拆分、负采样、训练结果可重复；缺失源域、未知物品、空历史均有确定行为。
3. 对每个 `k` 输出 HR@5/10、NDCG@5/10、覆盖率和用户级 bootstrap 95% 置信区间；报告所有基线与消融，不只报告最优结果。
4. “方法有效”的判据预先固定为：五个 `k` 的用户等权平均 NDCG@10 高于最强可比基线，配对用户 bootstrap 的差值 95% 区间下界大于 0，且 `k=0` 的 NDCG@10 不低于源域单独推荐。未达到时如实报告，不通过调测试集阈值改变结论。

## 数据契约与防泄漏策略

输入路径：`data/private/events.csv`，UTF-8，列名固定为 `user_id,domain,item_id,timestamp_ms`。`user_id` 和 `item_id` 当作字符串，`domain` 使用 README/数据源中的领域名，`timestamp_ms` 是整数 Unix 毫秒。一个事件对应一次正向交互；重复的 `(user_id, domain, item_id)` 仅保留时间最早的一条。数据不得提交到 Git。

桥接用户按稳定哈希划入四组：基础训练 A 60%、融合训练 B 20%、验证 C 10%、测试 D 10%。拆分以用户为单位，种子 `2026`，写入 `artifacts/split.json`。源域模型可以使用 B/C/D 用户在其**第一次目标域交互之前**的源域行为，因为这些行为在冷启动时已经存在；A 用户也按相同规则截断，避免训练和推理输入定义不一致。基础目标域模型只使用 A 的目标域行为。B 的目标域行为只训练融合器，C 只用于早停与超参数选择，D 只用于最终评估。目标域物品目录由 A 的目标域交互建立；后续标签不在该目录中的样本计入“不可推荐”覆盖率，不混入排序指标。

原仓库的 `implicit.json` 未随仓库提交，也没有时间戳契约。若手头仅有此文件，先核实其列表顺序确实代表发生时间；不能核实时不得把随机遮蔽称作“随时间积累交互”的验证。

## 文件结构与接口

| 文件 | 职责与接口 |
|---|---|
| `pyproject.toml`、`uv.lock`、`.gitignore` | 固定 Python 环境、依赖与本地数据排除规则 |
| `model/adafedgcdr/data.py` | `load_events(path) -> list[Event]`；`build_split(events, target, seed) -> SplitManifest`；`make_cases(events, manifest, k_values) -> list[Case]` |
| `model/adafedgcdr/base.py` | `train_base(events, manifest, config) -> BaseArtifacts`；仅调用 `model/fedgcdr/` 的 `Client`、`Server`、`MLP`、`GAT`，输出对新用户可用的源域向量与目标物品向量 |
| `model/adafedgcdr/fusion.py` | `AdaptiveFusion.forward(source_vec, source_mask, source_count, history_vec, history_mask) -> (user_vec, target_weight, source_weight, personalized_mask)` |
| `model/adafedgcdr/experiment.py` | `train_fusion(artifacts, cases, config) -> checkpoint`；`evaluate(checkpoint, cases, catalog) -> metrics`；统一实现负采样、基线、bootstrap |
| `model/adafedgcdr/__main__.py` | `python -m model.adafedgcdr prepare|train-base|train-fusion|evaluate` 命令入口 |
| `tests/make_synthetic_events.py`、`tests/test_ada_data.py`、`tests/test_ada_base.py`、`tests/test_ada_fusion.py`、`tests/test_ada_experiment.py` | 可重复的合成数据与定向回归测试 |
| `docs/adafedgcdr-runbook.md` | 数据格式、命令、输出解释与已知限制 |

`BaseArtifacts` 保存 `domain_names: list[str]`、`source_vectors: dict[str, Tensor[S,D]]`、`source_counts: dict[str, Tensor[S]]`、`source_mask: dict[str, BoolTensor[S]]`、`target_items: dict[str,int]`、`target_item_vectors: Tensor[I,D]`、`target_train_users: list[str]`、`split_sha256: str`。**不保存或读取 B/C/D 的目标域用户 embedding**。`Case` 保存 `user_id`、`k`、`history_item_ids`、`label_item_id`；下一条事件只作为标签，不进入表示。

数据对象的字段类型固定为：`Event(user_id: str, domain: str, item_id: str, timestamp_ms: int)`；`SplitManifest(target_domain: str, seed: int, users: dict[str, tuple[str, ...]])`，其中键为 `base/fusion/validation/test`；`Case(user_id: str, k: int, history_item_ids: tuple[str, ...], label_item_id: str)`。四组用户名单和物品映射都按字典序保存，避免字典遍历顺序影响结果。

## Task 1: 固定环境和数据契约

**Files:** Create `.gitignore`, `pyproject.toml`, `model/adafedgcdr/__init__.py`, `model/adafedgcdr/data.py`, `tests/make_synthetic_events.py`, `tests/test_ada_data.py`; write `uv.lock`。

**Interfaces:** Consumes `data/private/events.csv`；produces `Event`、`SplitManifest`、`Case` 和稳定拆分 JSON，供后续任务使用。

- [ ] **Step 1: 建立项目虚拟环境和依赖声明。** `pyproject.toml` 的核心内容为：

  ```toml
  [build-system]
  requires = ["setuptools>=69"]
  build-backend = "setuptools.build_meta"

  [project]
  name = "adafedgcdr"
  version = "0.1.0"
  requires-python = ">=3.11,<3.12"
  dependencies = ["torch>=2.2,<3", "numpy>=1.26,<3", "tqdm>=4.66,<5"]

  [project.optional-dependencies]
  dev = ["pytest>=8,<10"]

  [tool.setuptools.packages.find]
  where = ["."]
  include = ["model*"]
  ```

  `.gitignore` 包含 `.venv/`、`.uv-cache/`、`data/private/`、`artifacts/`、`__pycache__/`、`.pytest_cache/`。运行：

  ```bash
  UV_CACHE_DIR="$PWD/.uv-cache" uv venv --python 3.11 .venv
  UV_CACHE_DIR="$PWD/.uv-cache" uv lock
  UV_CACHE_DIR="$PWD/.uv-cache" uv sync --extra dev
  .venv/bin/python -c 'import sys; print(sys.version_info[:2])'
  ```

  预期最后输出 `(3, 11)`；若依赖下载受限，记录错误并继续完成不依赖训练的文档，不改用全局解释器。

- [ ] **Step 2: 先写拆分与前缀测试。** 合成至少 100 个桥接用户，每人在一个源域有 1 条、目标域有 12 条按时间排序的交互。关键断言为：

  ```python
  groups = [set(split.users[name]) for name in ("base", "fusion", "validation", "test")]
  assert not any(groups[i] & groups[j] for i in range(4) for j in range(i + 1, 4))
  case = next(c for c in cases if c.k == 0)
  assert case.history_item_ids == ()
  assert case.label_item_id == target_events[case.user_id][0].item_id
  assert next(c for c in cases if c.user_id == case.user_id and c.k == 3).label_item_id == target_events[case.user_id][3].item_id
  ```

  运行 `.venv/bin/python -m pytest tests/test_ada_data.py -q`；预期因接口尚未实现而失败。

- [ ] **Step 3: 实现数据读取、去重、稳定拆分及前缀生成。** `Event` 用 frozen dataclass；`timestamp_ms` 无法转为整数时抛出含文件行号的 `ValueError`。用户拆分依据 `sha256(f"{seed}:{user_id}".encode()).digest()[:8]` 转出的 64 位数落入 `[0,.6)、[.6,.8)、[.8,.9)、[.9,1)`。`make_cases` 只为合格用户生成 `k ∈ (0,1,3,5,10)`，且 `label_item_id = target_events[k].item_id`。不可推荐标签单独计数；不得通过替换标签掩盖覆盖率。

  `tests/make_synthetic_events.py` 的生成逻辑固定为：

  ```python
  import csv
  from pathlib import Path

  output = Path("artifacts/synthetic/events.csv")
  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open("w", newline="", encoding="utf-8") as stream:
      writer = csv.writer(stream)
      writer.writerow(("user_id", "domain", "item_id", "timestamp_ms"))
      for user in range(100):
          writer.writerow((f"u{user}", "CDs", f"s{user % 30}", 0))
          for index in range(12):
              writer.writerow((f"u{user}", "Books", f"t{(user * 13 + index) % 200}", 1000000 + index))
  ```

- [ ] **Step 4: 再运行测试与提交。** `.venv/bin/python -m pytest tests/test_ada_data.py -q` 预期全过；执行 `git add .gitignore pyproject.toml uv.lock model/adafedgcdr tests/make_synthetic_events.py tests/test_ada_data.py`，提交标题 `建立冷启动数据契约`，正文两条分别写环境锁定与防泄漏拆分。

## Task 2: 导出没有目标域用户泄漏的基础表示

**Files:** Create `model/adafedgcdr/base.py`, `tests/test_ada_base.py`; modify `model/fedgcdr/party.py` **仅在合成烟雾测试暴露出确定的设备或空批次错误时**修复相应行，不改变原算法定义。

**Interfaces:** Consumes `Event`、`SplitManifest`；produces `BaseArtifacts`，其 `source_vectors[user_id]` 形状是 `[源域数, embedding_size]`，`target_item_vectors` 形状是 `[物品数, embedding_size]`。

- [ ] **Step 1: 先写隔离测试。** 用 spy 包装目标域 `Client.train_gat` 和 `Client.train`，断言训练调用的用户集合始终是 `split.users["base"]`；只改写 B/C/D 用户**第一次目标域事件之后**的目标域事件、保留第一次事件时间戳与所有源域事件，以相同种子重跑 `train_base`，断言目标域物品向量哈希不变。断言 `BaseArtifacts` 不含 B/C/D 的目标域 `U` 行。运行 `.venv/bin/python -m pytest tests/test_ada_base.py -q`，预期失败。

- [ ] **Step 2: 实现事件到旧模型的适配。** 使用 `BaseConfig(target_domain: str, embedding_size: int = 16, source_rounds: int = 150, target_rounds: int = 150, local_epoch: int = 3, user_batch: int = 16, lr_gat: float = 0.001, eps: float = 8.0, delta: float = 1e-5, dp: bool = True, device: str = "cpu", seed: int = 2026)`；适配旧模型时补齐其 `args.num_domain`、`args.num_users`。源域用户集合取截断后的源域事件；目标域 `Server.clients` 仅取 A 用户，并为 A 重建从 0 开始的 `user_dic[target_name]`。新入口按固定轮数调用各源域 `kt_stage(False)`，随后调用目标域 `kt_stage(True)`；不使用旧 `main.py` 中的测试集选最优轮次。默认 CPU 烟雾测试各 1 轮，正式轮数由 `BaseConfig` 传入。源域缺失时 `source_mask=False`、对应向量为零，不把高斯噪声当作可靠源域历史。

- [ ] **Step 3: 明确导出与隐私边界。** 有历史的源域向量先按当前 `Client.l2_clip` 裁剪，并在 `dp=True` 时按当前 `eps`、`delta` 公式加噪一次，再经对应 `MLP` 映射；离线导出结果缓存到 checkpoint，不能在每次评估时重复生成新噪声。保存模型配置、种子、数据与拆分的 SHA-256；`artifacts/` 不纳入 Git。这个集中式导出仅供离线研究，不能据此宣称上线后的联邦隐私保证。

- [ ] **Step 4: 运行隔离测试与合成烟雾测试。** `.venv/bin/python -m pytest tests/test_ada_base.py -q` 全过；合成样本导出必须具有全部预期用户、源域掩码和非空目标物品目录。提交标题 `导出隔离的跨域基础表示`，正文分条写用户隔离与表示导出。

## Task 3: 实现可靠性门控与随交互增长的融合

**Files:** Create `model/adafedgcdr/fusion.py`, `tests/test_ada_fusion.py`。

**Interfaces:** Consumes `BaseArtifacts` 的对齐源域向量和目标物品向量；produces `user_vec`、`target_weight`、`source_weight`、`personalized_mask`。所有张量首维均为 batch；`source_mask=False` 的权重精确为 0。

- [ ] **Step 1: 先写定向测试。** 覆盖 `k=0`、`k=1`、`k=10`、源域缺失、全部源域缺失、单个源域极低可靠性以及反向传播。关键断言：

  ```python
  assert torch.allclose(target_weight[k == 0], torch.zeros_like(target_weight[k == 0]))
  assert torch.all(target_weight[k == 10] > target_weight[k == 1])
  assert torch.count_nonzero(source_weight[~source_mask]) == 0
  assert torch.isfinite(user_vec).all()
  ```

  运行 `.venv/bin/python -m pytest tests/test_ada_fusion.py -q`，预期失败。

- [ ] **Step 2: 实现最小可训练门控。** 对有历史源域计算 `logit_s = b_s + a·log(1+source_count_s) + c·cos(source_vec_s, target_history_mean)`；`k=0` 时余弦项设为 0。用掩码 softmax 得到 `r_s`，缺失源域置零并重新归一化。`h_source = Σ r_s source_vec_s`，`h_target = mean(已见目标物品向量)`；`w(k)=k/(k+softplus(tau_raw))`，`k=0` 时强制为 0，完全没有源域但 `k>0` 时强制为 1。最终 `h=(1-w)h_source+w h_target`。`personalized_mask = source_mask.any(1) | history_mask.any(1)`；为 False 时由评估层使用目标域流行度排序，不用随机向量打分。`forward` 至少满足以下实现骨架：

  ```python
  count = history_mask.sum(dim=1)
  local = (history_vec * history_mask[..., None]).sum(dim=1) / count.clamp_min(1)[:, None]
  agreement = torch.nn.functional.cosine_similarity(source_vec, local[:, None, :], dim=-1)
  agreement = torch.where(count[:, None] > 0, agreement, torch.zeros_like(agreement))
  logits = domain_bias + count_scale * torch.log1p(source_count) + agreement_scale * agreement
  source_weight = torch.softmax(logits.masked_fill(~source_mask, -1e9), dim=1) * source_mask
  source_weight = source_weight / source_weight.sum(dim=1, keepdim=True).clamp_min(1e-8)
  source = (source_weight[..., None] * source_vec).sum(dim=1)
  target_weight = count / (count + torch.nn.functional.softplus(tau_raw))
  target_weight = torch.where(source_mask.any(dim=1), target_weight, (count > 0).float())
  user_vec = (1 - target_weight[:, None]) * source + target_weight[:, None] * local
  personalized_mask = source_mask.any(dim=1) | (count > 0)
  return user_vec, target_weight, source_weight, personalized_mask
  ```

- [ ] **Step 3: 运行测试与提交。** `.venv/bin/python -m pytest tests/test_ada_fusion.py -q` 全过；固定相同输入连续调用两次应得到相同输出。提交标题 `加入可靠性门控与动态融合`，正文分条写源域降权和目标域权重规则。

## Task 4: 按冷启动阶段训练融合器

**Files:** Create `model/adafedgcdr/experiment.py`, `model/adafedgcdr/__main__.py`, `tests/test_ada_experiment.py`。

**Interfaces:** Consumes B 组 `Case`、`BaseArtifacts`；produces `artifacts/fusion.pt`。C 组仅供早停，不更新任何参数；D 组不能进入训练函数。

- [ ] **Step 1: 先写阶段均衡与负采样测试。** 每个正样本配 99 个目标域目录内、该用户从未交互过的负物品；候选列表第 100 位固定为正物品。相同 `(seed,user_id,k)` 必须产生相同候选。`k=0,1,3,5,10` 的阶段损失先各自取均值，再对五阶段等权平均；测试样本数不同也不能改变阶段权重。运行 `.venv/bin/python -m pytest tests/test_ada_experiment.py -q`，预期失败。

- [ ] **Step 2: 实现训练循环。** 冻结 `BaseArtifacts` 中的源域与目标物品向量，只优化 `AdaptiveFusion` 的参数；批次候选分数为 `scores = torch.einsum("bd,bcd->bc", user_vec, candidate_item_vectors)`，正样本索引为 `candidate_count - 1`，损失使用 `torch.nn.functional.cross_entropy(scores, labels)`。固定种子 `2026`、学习率 `1e-3`、最多 50 epoch、验证 NDCG@10 早停耐心 5 epoch；保存验证最优 checkpoint 与完整配置。对目标目录不足 100 个可用物品的合成测试，允许 `candidate_count` 配置为较小值，正样本始终放最后。

- [ ] **Step 3: 实现四个命令。** `prepare` 读取 CSV 并写拆分及数据审计；`train-base` 写 `artifacts/base.pt`；`train-fusion` 写 `artifacts/fusion.pt`；`evaluate` 写 `artifacts/metrics.json`。命令必须检查数据 SHA-256 和拆分 SHA-256 与 checkpoint 相符，不匹配时明确报错。训练日志逐轮输出五阶段验证指标、早停轮次和随机种子。

- [ ] **Step 4: 运行测试与合成端到端。** `.venv/bin/python -m pytest tests/test_ada_experiment.py -q` 全过；以下命令用合成 CSV 完成四阶段：

  ```bash
  .venv/bin/python tests/make_synthetic_events.py
  .venv/bin/python -m model.adafedgcdr prepare --events artifacts/synthetic/events.csv --target Books --seed 2026 --split artifacts/synthetic/split.json
  .venv/bin/python -m model.adafedgcdr train-base --events artifacts/synthetic/events.csv --split artifacts/synthetic/split.json --source-rounds 1 --target-rounds 1 --device cpu --out artifacts/synthetic/base.pt
  .venv/bin/python -m model.adafedgcdr train-fusion --events artifacts/synthetic/events.csv --split artifacts/synthetic/split.json --base artifacts/synthetic/base.pt --candidate-count 10 --out artifacts/synthetic/fusion.pt
  .venv/bin/python -m model.adafedgcdr evaluate --events artifacts/synthetic/events.csv --split artifacts/synthetic/split.json --base artifacts/synthetic/base.pt --fusion artifacts/synthetic/fusion.pt --candidate-count 10 --out artifacts/synthetic/metrics.json
  ```

  预期所有命令退出码 0，`metrics.json` 含五个 `k` 且没有 NaN。提交标题 `建立分阶段训练与实验命令`，正文分条写训练和可复现性校验。

## Task 5: 基线、统计评估与报告

**Files:** Modify `model/adafedgcdr/experiment.py`, `model/adafedgcdr/__main__.py`; create `docs/adafedgcdr-runbook.md`; modify `README.md`; extend `tests/test_ada_experiment.py`。

**Interfaces:** Consumes D 组 `Case` 和锁定的最优 checkpoint；produces `artifacts/metrics.json`、`artifacts/report.md`。所有方法使用完全相同的用户、标签、候选和目标物品目录。

- [ ] **Step 1: 先写指标测试。** 用手工分数检查正物品在第一位时 `HR@5=HR@10=NDCG@5=NDCG@10=1`，在第六位时 `HR@5=NDCG@5=0`、`HR@10=1`；对每个用户 bootstrap 1000 次，固定种子后置信区间完全复现。运行 `.venv/bin/python -m pytest tests/test_ada_experiment.py -q`，预期新测试失败。

- [ ] **Step 2: 实现统一对照。** 同一候选集上评估 `source-only`（源域等权）、`target-only`（`k>0`）、`fixed-0.5`、`dynamic-only`（交互数门控、源域等权）、`reliability-only`（源域门控、固定融合）、`full`（两种门控）。所有融合基线在 `k=0` 时将目标域权重设为 0；`k=0` 的 target-only 标为不适用，不能填零混入平均。报告每阶段的样本数、不可推荐物品比例、缺失源域比例、负迁移用户比例及配对用户 bootstrap 95% 区间。旧 FedGCDR 的现有评估依赖已训练目标域用户 ID，不直接与零交互结果同表比较；若未来加入归纳式适配器，必须遵循同一用户隔离协议。

- [ ] **Step 3: 写运行手册与结论边界。** README 链接手册；手册给出输入 CSV 示例、四命令的完整参数、输出字段、阶段阈值、源域时间截断、用户级数据隔离和本地隐私边界。报告清楚说明 Amazon 的 user-core/item-core 过滤造成的样本选择偏差，以及真实新用户可能短于过滤阈值。

- [ ] **Step 4: 全量测试与数据关口。** `.venv/bin/python -m pytest tests -q` 全过。若 `data/private/events.csv` 不存在，到此停止真实训练，仅提交代码和合成验证结果；若存在，按固定种子运行完整实验，检查每个 `k` 的可评估用户数与覆盖率，再生成报告。提交标题 `完成冷启动对照评估`，正文分条写基线、置信区间和复现说明。

## 后续独立扩展：图扩展门控

第一版的可靠性权重已经降低低可靠源域在**表示融合**中的参与。仅当 Task 5 显示稳定收益，再开独立计划修改 `model/fedgcdr/model.py:GATLayer.forward` 和 `GAT.forward`：把 `log(r_s+ε)` 加到用户到源域虚拟节点的注意力 logit，`r_s=0` 时删除边；在相同 A/B/C/D 用户协议下重新训练并与表示融合版比较。此扩展必须额外测试：全源域屏蔽时无 NaN、注意力掩码确实为零、原始 `GAT.forward` 不传权重时输出不变。当前 `domain_attention` 参数虽被传入，`GAT.forward` 并未使用它，因此不能把它当作已经完成的可靠性门控。

## 执行前检查与风险

- **真实数据缺席：** 当前仓库没有 `data/`；先确认合法来源、时间戳和领域映射。没有时间戳时可以做随机遮蔽消融，但报告名称必须改为“稀疏历史模拟”，不能写成时间演进或线上冷启动。
- **旧入口脆弱：** `main.py` 在模块加载时执行训练，并读取未定义的 `args.random_seed`；新命令不得复用该入口。`Server.kt_stage` 的空批次/设备处理须由合成烟雾测试覆盖，修复时保留原始模型逻辑。
- **零交互可靠性不可直接观测：** `k=0` 只能利用跨用户学到的先验、源域历史量与缺失掩码；目标域一致性信号只能在 `k>0` 时加入。
- **缓存向量与隐私：** 离线 `base.pt` 含用户表示，必须留在本地忽略目录；从客户端导出到生产服务器需要单独的隐私设计和预算核算。
- **性能与稳定性：** 先用 CPU 合成测试验证形状和边界，再用实际数据与明确设备参数训练；真实实验记录耗时、显存、可评估用户数及运行失败原因。

## 自检清单

- [x] 逐条对应原目标：冷启动源域优先、目标行为逐步接管、低可靠来源降权、零交互推理、用户级无泄漏评估。
- [x] 检查所有后续任务引用的接口与本文件“文件结构与接口”一致。
- [x] 检查任务步骤没有未定义输入、隐含用户确认或依赖不存在的仓库数据。
- [x] 在真实数据到位前，不宣称效果已被验证。
