# 四域实验交接记录

本目录保存2026年10月9日完成的四域受控预算实验，供另一台电脑查看结果和继续分析。三个训练种子、共15组训练全部完成，2026年10月9日23:14（北京时间）取回结果。

## 结果入口

- `report.md`：对照、消融、固定标签、源域分组及敏感性检查。
- `summary.json`：跨种子的汇总指标和标准差。
- `primary-comparisons.json`：配对用户bootstrap区间与候选/用户一致性审计。
- `seed-*/{sparse-k1,sparse-k3,sparse-k5,sparse-k10,cold}/metrics.json`：每组指标、门控验证曲线、骨干微调选轮和覆盖审计。
- `seed-*/training.log.gz`：完整原始训练日志；`training-stages.log`：去掉进度条后的阶段索引。
- `tests/`：70项回归测试和最后11项协议检查的日志，两者有重叠，不能相加计算测试总数。
- `embedding-diagnostics.json`：已完成模型的只读诊断，包括门控参数、向量尺度、方向和评分相关性。
- `*.png`、`*.pdf`：主对照、固定标签检查和冷启动消融图。
- `analysis-notes.md`：当前根因分析证据和待验证问题。

这是公开可共享的结果包，只包含汇总指标和训练日志。逐用户指标、用户划分名单、原始交互数据、用户/物品向量checkpoint及服务器密码未包含在Git仓库中。公开包中的`run.json`仅用于查看配置和耗时，不能代替私有完整实验目录执行checkpoint诊断或重算bootstrap。

## 本轮设置与结论

目标域Books；源域CDs、Movies、Clothing；4348名合格用户，104180条去重交互。先前1000人试运行的103名测试用户划入验证组，避免将已查看结果的开发用户继续作为主测试用户。最终A/B/C/D人数为2572/858/548/370。

每个源域5轮；目标域普通GAT5轮、迁移GAT5轮；MF最多20轮；三个门控消融独立训练、最多50轮，验证早停patience=5。原版Client/Server/GAT/MLP未修改；当前对比入口使两个推荐头共享MF微调后的骨干和物品向量。因此，这是特定共享骨干下的表示头对照，不等同于复现原论文，也不等同于直接运行旧AdaFedGCDR入口。

| 协议 | 完整融合头 | 对照 | 用户等权平均NDCG@10 |
|---|---:|---|---:|
| 少交互 | 0.089119 | 原版FedGCDR | 0.222159 |
| 严格冷启动 | 0.063521 | 目标域流行度 | 0.249635 |

本轮不支持完整融合头有效。下一步应先检查表示与评分器的匹配、源域表示退化和门控训练，再考虑新的效果实验。不能把这组负结果改写成性能提升。

## 另一台电脑继续工作

先拉取`main`分支，查看本目录与`docs/paper-experiment-protocol.md`。标准开发环境按项目运行手册建立；本次服务器实际环境为Python 3.10.16、PyTorch 2.7.1+cu126、Tesla P4，属于另行记录的实验环境。绘图还需要`matplotlib>=3.7,<3.11`。

现有训练服务器上的完整项目位于：

```text
/data/coding/AdaFedGCDR-paper-20261009-r1
```

在服务器项目目录内，关键私有路径为：

```text
artifacts/input/events-4348.csv
artifacts/input/development-users.json
artifacts/paper-suite-4348/
artifacts/paper-suite-4348/workers/seed-2026/seed-2026/cold/backbone.pt
```

服务器连接信息需要通过自己的私有方式带到另一台电脑。不要把连接密码加入Git。三个种子均有完整source、backbone及门控checkpoint，可以在原服务器继续诊断，无需重训。

只读诊断命令（在具有完整私有结果目录及PyTorch的环境执行）：

```bash
python -m scripts.diagnose_paper_embeddings \
  --results artifacts/paper-suite-4348 \
  --events artifacts/input/events-4348.csv \
  --out artifacts/embedding-diagnostics.json
```

原始四域ratings CSV在第一台电脑的`ratings-full/`，没有上传服务器。需要从原始CSV重新生成同一子集时：

```bash
python -m scripts.prepare_paper_subset \
  --ratings-dir ratings-full --out artifacts/paper-four-domain --seed 2026
```

`--limit 1000`可重建最初试运行子集。全量本轮输入CSV的SHA-256为：

```text
df9945b70887fac226d5915e6966b5bae9cf26a1d64035df55ffe572968bc482
```

需要新一轮训练时，使用新的输出目录，并记录协议、数据和代码版本；目前尚未启动新的改进训练。已完成测试集结果不得用于反复选择模型或超参数。
