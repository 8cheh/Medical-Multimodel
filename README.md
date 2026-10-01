# 算法实现

「基于多模态的生理监测」项目的算法交付包。报告与汇报幻灯片里出现的**每一个数字**，都能在本目录内找到产生它的代码与结果文件；目录内所有副本与仓库原件**逐字节一致**，可用一条命令复核。

## 一、目录结构

| 子目录 | 内容 | 副本数 |
|---|---|---|
| `a_数据来源/` | 数据出处、通道清单、下载量、存储策略的证据文件 | 6 |
| `b_算法说明/` | 端到端架构图（正文见 `b_算法说明/README.md`） | 1 |
| `c_源代码/` | 全部实现脚本：`remote/`（流水线 22 个）、`tools/`（20 个 Python + `make_slides.js`）、`requirements.txt` | 44 |
| `d_运行结果/` | 实验结果 JSON 29 份（含 4 份文档核验产物）、运行日志 14 份、图 16 张 | 59 |
| `inventory.json`、`MANIFEST.json` | 普查数字与逐文件 sha256 | 2 |

合计 110 个副本 + 5 份说明（本文件与四个子目录的 `README.md`）。仓库侧代码规模、结果产物数与报告插图数见 `inventory.json`（每次重新生成时刷新）。

## 二、如何复核

```
python tools/build_algorithm_delivery.py    # 重新生成：复制、重算 sha256、刷新普查数字
python tools/check_algorithm_delivery.py    # 校验本目录
```

校验三件事：

1. 每个副本的 sha256 == 清单记录值 == 仓库原件**当前**值——任一方向漂移都会报错，因此本目录不可能与仓库代码悄悄脱钩。
   例外：`check_all.json`、`verify_pdf*.json`、`verify_slides.json` 四份**核验产物**每次运行都会被重写（含耗时与字节数），它们是**构建时快照**，只按清单哈希校验完整性，不要求与实时文件同步；要刷新它们，重新运行 `tools/build_algorithm_delivery.py`。
2. 目录内不存在清单之外的残留文件（上一次运行留下的旧副本会与报告自相矛盾，因此视为错误）；
3. `inventory.json` 的普查数字与仓库、本地数据现状一致（防止说明里引用过期数字）。

## 三、算法链路与入口脚本

数据获取 → 特征提取 → 数据集构建 → 模型训练 → 终点评测 → 报警策略 → 校准与决策曲线 → 流式推理 → 显示原型

| 环节 | 入口脚本 |
|---|---|
| 数值轨道下载 | `remote/dl_tracks.py`、`remote/dl_vital.py` |
| 波形下载（PPG / 有创动脉压） | `remote/dl_ppg.py` |
| 波形特征 | `remote/build_ppg_features.py`、`remote/build_abp_features.py` |
| 数据集（1 Hz 网格 / 窗口 / 标签 / 67 维特征） | `remote/build_dataset.py` |
| 训练与阈值选择 | `remote/train_model.py`、`remote/train_fused.py` |
| 终点评测（含配对显著性检验） | `remote/p0_eval.py` |
| 报警策略 | `remote/alarm_policy.py`、`remote/alarm_pareto.py` |
| 校准与决策曲线 | `remote/calibration_dca.py` |
| 信号质量门控 | `remote/quality_gate.py` |
| 流式推理 | `remote/stream_monitor.py` |
| 显示原型 | `remote/make_monitor_display.py` |

## 四、运行环境

`c_源代码/requirements.txt`（Python 3 + LightGBM、pandas、numpy、PyWavelets 等）。训练与评测脚本经 `tools/remote.py`（SSH/SFTP）在计算节点执行；数据路径以 `D:\vitaldb_local` 本地镜像为准。

## 五、数据不随本目录提交

原始数值轨道与波形共约 105 GB（95,402 个 CSV + 2,500 个 npy），存放在仓库之外的本地镜像；本目录只保留派生特征、结果与清单（约 60 MB），因此可随报告一并提交、便于携带。出处与体积见 `a_数据来源/README.md`。

## 六、已知边界（不全藏起来）

- **单中心回顾性数据**：全部结果来自 VitalDB 单一中心，没有外部验证集——这是最大的科学缺口；
- **灵敏度**：默认阈值（0.5）下低血压终点漏报约 35%，落地需按目标灵敏度重选工作点（候选见 `d_运行结果/operating_points.json`）；
- **提前量**：模型相对现成 `MAP<65` 报警**没有**事件级优势——配对比较的中位提前量差为 0 s（详见 `d_运行结果/logs/p0_lead_h300.log`）；
- **波形模态**：PPG（+55 维）与有创动脉压（+55 维）对低血压/低氧/通气不足三个终点均无正增量；
- **肌松（TOF/NMT）**：VitalDB 通道索引中不存在该通道，本数据集无法评测；
- **未评测终点**：`tachycardia`（HR>120）已在代码中定义，本轮未评测；
- **通道覆盖面**：196 路索引通道中本机下载 51 路，其余 145 路未取。
