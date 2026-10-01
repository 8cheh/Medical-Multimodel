# d. 运行结果

59 份产物：实验结果 JSON 29 份（其中 4 份为文档级核验产物：报告 PDF、中期报告 PDF、交付包完整性、幻灯片）、运行日志 14 份、图 16 张。报告中的每个数值都能由这些文件回算——`check_numbers.json` 记录 **239 个数值、0 失配**。

## 4.1 五个终点的核心结果（测试集，病例级互斥划分）

| 终点 | 患病率 | 规则基线 AUROC | 模型 AUROC | AUPRC | 检出率 | 假报警/小时 |
|---|---|---|---|---|---|---|
| 低血压 hypotension | 12.48% | 0.9022 | 0.9536 | 0.8423 | 99.85% | 1.72 |
| 低氧 hypoxemia | 1.13% | 0.2445 | 0.9723 | 0.5179 | 95.90% | 0.52 |
| 心动过缓 bradycardia | 3.74% | 0.5960 | 0.9889 | 0.8851 | 99.34% | 0.34 |
| 通气不足 hypoventilation | 0.53% | 0.6527 | 0.8787 → **0.9202** | 0.0820 → **0.2011** | 87.32% → **93.0%** | 1.97 |
| 高碳酸血症 hypercapnia | 0.49% | 0.9469 | 0.9858 → 0.9838 | 0.7938 → **0.8219** | 100.00% | 1.31 |

说明（避免误读）：

- 低氧与心动过缓的"规则基线"是同一个现成 `MAP<65` 报警器——它并不针对这两个风险，AUROC 低于 0.5 正说明现成报警器**不覆盖**这些风险；通气不足与高碳酸血症的规则基线是各自针对性的报警规则（呼吸频率 <6、ETCO2 >50）。
- 通气不足一行的箭头表示"基础 67 维 → 追加匹配的功能通道特征（`main_func`）"：加通道后 AUROC 0.8787→0.9202、AUPRC 0.0820→0.2011、检出 87.32%→93.0%。
- 高碳酸血症相反：其触发量本身已在基础特征里（`Solar8000/ETCO2`），追加 `Primus/ETCO2` 后 AUROC 0.9858→0.9838（配对 bootstrap 区间跨 0，无增益）——这正是本项目"模态价值 = 信息是否已在特征里"的核心论断。

## 4.2 其它已完成实验

| 主题 | 关键结果 | 产物 |
|---|---|---|
| 报警策略帕累托 | persist 1/2/3 → 检出 98.03%/96.87%/92.65%，假报警 0.58/0.22/0.10 每小时 | `alarm_pareto.json` |
| 校准与决策曲线 | Brier 0.0432（低血压）/ 0.0079（低氧），ECE 0.0025 / 0.0024，pt=0.10 净收益 +0.0983 / +0.0069 | `calibration_*.json` |
| 信号质量门控 | 可用窗口 94.49%，假报警 0.22→0.21，检出事件 1331→1267（收益小、代价明确，不作默认） | `quality_gate.json` |
| 工作点 | 验证集选出的 6 档工作点（含 sens≥0.90 档：灵敏度 0.8876、特异度 0.8706） | `operating_points.json` |
| 预测视界 | 5/10/15 min：模型 AUROC 0.9536→0.9100→0.8848，规则 0.9022→0.8349→0.7954 | `p0_report_h600.json`、`p0_report_h900.json` |
| 优化消融 | 长窗（900 s）AUROC +0.0051；追加功能通道特征 +0.0009 | `p0_report_h900opt.json` |
| 波形增量 | PPG 55 维、动脉压 55 维对三个终点均无正增量（配对检验区间跨 0） | `p0_report_multimodal_h300.json`、`p0_significance.json` |
| 提前量配对检验 | 推翻原"+1.65 min"说法：配对中位提前量差 0 s | `logs/p0_lead_h300.log` |
| 多种子 | seed 42/7/2024/12345 → AUROC 0.9537/0.9519/0.9602/0.9552 | `logs/verify_seeds.log` |
| 流式等价性 | 3 例、67/67 特征一致、分数最大偏差 2.973e-08、中位延迟 1.25–2.16 ms/决策 | `logs/stream_check.log` |

## 4.3 文件索引

| 组 | 文件 |
|---|---|
| 主结果 | `p0_report.json`（低血压 5 min）、`p0_report_h600.json`、`p0_report_h900.json`、`p0_report_h900opt.json`（优化消融） |
| 其它终点 | `p0_report_hypoxemia.json`、`p0_report_bradycardia.json`、`p0_report_hypoventilation.json`、`p0_report_hypercapnia.json` |
| 多模态 | `p0_report_multimodal_h300.json`、`p0_report_multimodal_h900.json`、`p0_report_hypoxemia_mm.json`、`p0_report_ppg_h300.json` / `p0_report_ppg_h600.json` / `p0_report_ppg_h900.json` |
| 专项分析 | `alarm_pareto.json`、`calibration_hypotension.json`、`calibration_hypoxemia.json`、`operating_points.json`、`quality_gate.json`、`p0_significance.json` |
| 训练记录 | `report_main.json`、`report_fused.json`、`report_fused_final.json` |
| 数值核查 | `check_numbers.json`（239 值 / 0 失配） |
| 日志 | `logs/`：数据集构建、各终点评测、优化实验、提前量、多种子、流式核验 |
| 图 | `figures/fig01–fig16`：数据量、PPG 覆盖与波形、对齐、种子稳健性、ROC/PR、工作点、特征增益、PPG 消融、基线对比、视界、优化、显示原型、系统架构 |

## 4.4 复核方式

```console
python tools/check_all.py                   # 一键运行全部核查（含下列两项）
python tools/check_report_numbers.py        # 报告数值回算：239 值 0 失配
python tools/check_algorithm_delivery.py    # 本目录与仓库逐字节一致
```

图中 fig12–fig16 与本目录 JSON 一一对应；fig01–fig11 来自更早阶段（PPG 波形与 VLA 实验），保留以便追溯结论来源。
