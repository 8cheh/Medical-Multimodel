# a. 数据来源

本目录回答一个具体问题：**报告里的数字是从哪份数据里算出来的**。所有结论的证据文件都在这里。

## 1.1 数据出处

- **数据集**：VitalDB——术中高保真多参数生命体征数据库（首尔大学医院，去标识化后公开发布）。引用：LEE H, PARK Y, YOON SB, et al. *VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients*. Scientific Data, 2022. DOI: 10.1038/s41597-022-01411-5。
- **获取方式**：官方 Python 客户端 / HTTPS 接口（`api.vitaldb.net`）逐轨道下载，本机镜像目录 `D:\vitaldb_local`。
- **伦理与合规**：使用已公开的去标识化数据，未新增采集、未接触可识别信息；本项目为回顾性分析，不含前瞻性干预。

## 1.2 通道清单（`channel_registry.json`）

| 项目 | 数值 |
|---|---|
| 索引通道总数 | 196 路 |
| 设备 | 11 台：BIS、CardioQ、EV1000、FMS、Invos、Orchestra、Primus、SNUADC、Solar8000、Vigilance、Vigileo |
| 功能分组 | 8 组：循环、呼吸、肺力学、麻醉深度、肌松、输注、脑氧、体温 |
| 本机可用 | 51 路 |

清单由 `tools/build_channel_registry.py` 从本地轨道索引直接统计生成，不是人工填写。

## 1.3 实际下载量

| 内容 | 文件数 | 体积 |
|---|---|---|
| 数值轨道 CSV | 95,402 | 37.13 GB |
| 有创动脉压波形（`.npy`） | 2,500 | 67.59 GB |
| 动脉压特征（parquet） | 1 | 0.32 GB |
| PPG 特征（parquet） | 1 | 0.33 GB |
| 窗口数据集（parquet） | 1 | 0.17 GB |

数值轨道与波形合计约 104.7 GB。窗口数据集规模：**1,553,854 个决策窗口 / 3,495 例手术**（`../d_运行结果/logs/p0_dataset.log` 首行）。

## 1.4 功能通道下载记录（`func_download.json`）

| 项目 | 数值 |
|---|---|
| 请求通道 | 41 路（呼吸、肺力学、麻醉深度、输注、脑氧等） |
| 落盘文件 | 66,375 |
| 体积 | 32.38 GiB |
| 下载器统计 | ok=66,375，fail=1 |

唯一的失败条目是长期重试仍未成功的单文件，占请求量的 0.0015%——因此报告写"41 路、66,375 个文件、32.38 GiB"，而不是"41 路全部成功"。

## 1.5 存储策略：波形是临时文件

本机磁盘余量有限（见 `disk_volumes.txt`），因此原始波形**只作为中间物**：下载 → 提取特征 → 立即删除，仅保留派生特征与清单。策略与实测记录见 `ppg_local_store.md`、`abp_local_store.md`。

这带来一个必须说明的后果：**波形无法二次回放**，但所有下游数字只依赖特征矩阵与标签，因此报告结论仍可复现；若要复核波形本身，需要重新下载（脚本 `remote/dl_ppg.py` 在交付包内）。

## 1.6 本目录文件

| 文件 | 作用 |
|---|---|
| `channel_registry.json` | 196 路索引、11 台设备、8 个功能组、51 路本机可用 |
| `func_download.json` | 功能通道下载统计，含逐通道文件数/体积 |
| `ppg_local_store.md`、`abp_local_store.md` | PPG / 动脉压波形的下载—提取—删除记录 |
| `ppg_manifest.json` | PPG 波形清单 |
| `disk_volumes.txt` | 采集时的磁盘余量快照 |

## 1.7 数据层面的局限

1. **单中心**：手术类型、麻醉习惯、监护配置集中，外推性未知；
2. **通道覆盖**：196 路中仅下载 51 路，其余 145 路（EV1000/Vigileo/Vigilance/CardioQ、Invos 脑氧、FMS、部分 Orchestra 药物通道、BIS 脑电波形）未取；
3. **肌松监测无数据源**：肌松（neuromuscular_block）分组在通道索引中没有可下载通道，因此"肌松深度监测"只能作为后续工作，而不是本轮未完成项；
4. **时间基准不统一**：不同设备采样率与起始时刻不同，跨通道对齐需要零阶保持（处理方式见 `../b_算法说明/README.md` 2.1），最大容忍陈旧度 15 s。
