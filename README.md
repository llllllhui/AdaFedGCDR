<div align="center">

<h2 class="papername"> Federated Graph Learning for Cross-Domain Recommendation </h2>
<div>
<div>
    Ziqi Yang<sup>1,2</sup>, Zhaopeng Peng<sup>1,2</sup>, Zihui Wang<sup>1,2</sup>, Jianzhong Qi<sup>3</sup>, Chaochao Chen<sup>4</sup>,Weike Pan<sup>5</sup>, Chenglu Wen<sup>1,2</sup>, Cheng Wang<sup>1,2</sup>, Xiaoliang Fan<sup>1,2∗</sup>

</div>

<div><sup>1</sup>Fujian Key Laboratory of Sensing and Computing for Smart Cities, Xiamen University, China</div>
<div><sup>2</sup>Key Laboratory of Multimedia Trusted Perception and Efficient Computing,Ministry of Education of China, Xiamen University, China</div>
<div><sup>3</sup>School of Computing and Information Systems, The University of Melbourne, Australia</div>
<div><sup>4</sup>College of Computer Science and Technology, Zhejiang University Hangzhou, China</div>
<div><sup>5</sup>College of Computer Science and Software Engineering, Shenzhen University Shenzhen, China</div>
<div>*Corresponding author</div>
<br>
<div>

[The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS 2024)](https://neurips.cc/Conferences/2024)

[[Paper]](https://arxiv.org/abs/2410.08249)

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FLafinHana%2FFedGCDR&count_bg=%2379C83D&title_bg=%23555555&icon=github.svg&icon_color=%23E7E7E7&title=visitors&edge_flat=false)](https://hits.seeyoufarm.com)

</div>
</div>
<br>

</div>



## Introduction

This is the github repository of *Federated Graph Learning for Cross-Domain Recommendation*. In this apper, we propose a federated graph learning framework to preserve intra and inter domain privacy while mitigating negative transfer in cross-domain recommendation.

Our method is consist of two key modules, the positive knowledge transfer module and the positive knowledge activation module. The first module can extract domain konwledge form graph attention networks in a privacy perserving mode and has a theoretical guarantee. The second module performs graph expansion with virtual social links and incorporates a fine-tuning process to filter potientially harmful information and make maximum use of external knowledge.

The architecture of the proposed model:

<div align="center">
<img src='./assets/FedGCDR.png' width='100%'>
</div>

## Benchmark

We conduct experiments on 16 popular domains of the Amazon datasets:

<div align="center">
<img src='./assets/dataset.png' width='100%'>
</div>

## Evaluation results

Here we list the recommendation performance comparison of our method and baselines. Please refer to our paper for more details.

<div align="center">
<img src='./assets/results.png' width='100%'>
</div>


## AdaFedGCDR 冷启动验证版

面向“目标域零交互到行为渐多”场景的离线验证管线位于 `model/adafedgcdr/`，在保留原始 `model/fedgcdr/` 基线不变的前提下，加入可靠性门控与随交互增长的表示融合。数据契约、命令、指标定义与结论边界见运行手册：[docs/adafedgcdr-runbook.md](./docs/adafedgcdr-runbook.md)。

四域三种子对比已完成，代码、训练日志、汇总结果、图表与跨电脑交接说明见 [实验交接记录](./experiments/2026-10-09-books-four-domain/README.md)，评估设置见 [论文实验协议](./docs/paper-experiment-protocol.md)。本轮固定预算结果不支持当前融合头优于原版或流行度基线；下一步诊断见 [分析笔记](./experiments/2026-10-09-books-four-domain/analysis-notes.md)。原始交互数据、逐用户记录、checkpoint和连接凭据不包含在公开仓库中。

## Citation

If you find this work useful for your research, please kindly cite our paper:
```
@article{yang2024federated,
  title={Federated Graph Learning for Cross-Domain Recommendation},
  author={Yang, Ziqi and Peng, Zhaopeng and Wang, Zihui and Qi, Jianzhong and Chen, Chaochao and Pan, Weike and Wen, Chenglu and Wang, Cheng and Fan, Xiaoliang},
  journal={arXiv preprint arXiv:2410.08249},
  year={2024}
}
```

