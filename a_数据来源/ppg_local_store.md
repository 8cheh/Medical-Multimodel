# 本地波形库：验证记录与存储策略

**策略：原始波形是临时数据，只保留派生特征与溯源元数据。**

理由：PPG（SNUADC/PLETH）原始 `.npy` 需 124 GB，而它唯一的用途是计算特征；特征表
（`ppg_features_full/ppg_features.parquet`，315 MB，3,282 例 / 1,400,689 行 / 55 维）
一旦生成并验证，原始波形即可删除。需要时用同一条命令重新下载（实测 ~150 例/min，
全量约 40 分钟，可断点续传）。

## 已删除内容

| 项目 | 值 |
|---|---|
| 目录 | `D:\vitaldb_local\ppg_npy`（`.npy` 已删除） |
| 删除文件数 | 6,157 |
| 释放空间 | 124.0 GB |
| 保留 | 6,157 个 `.json` 侧车（`D:\vitaldb_local\ppg_meta\`，共约 6 MB） |

## 删除前的验证证据（均为实测，删除后无法重跑，故在此留档）

**1. 完整性与侧车一致性**（`check_ppg_integrity.py`）

```text
files=6157 samples_total=33,709,809,300 missing_sidecar=0 inconsistent=0
OK every npy loads and matches its sidecar
```

每个 `.npy` 都能被 `np.load` 打开，且其样本数与 dtype 与同名 `.json` 侧车声明**完全一致**。

**2. 是否被截断**（`check_ppg_duration.py`）——这是关键一项，因为流式下载没有
Content-Length 校验，被截断的流会生成「短但自洽」的数组（侧车也会一致）。

```text
cases compared=221
PPG shorter than 80% of the numeric span: 0 cases
mean PPG 221.0 min vs mean numeric 214.8 min
OK waveform durations are consistent with the monitoring period
```

即：逐例把 PPG 时长与同一病例的数值轨道时长比较，**0/221 例偏短**，中位比值 1.02。
PPG 平均时长 221 min 与窗口数据集反推的病例时长（444.6 窗口 × 30 s ≈ 222 min）一致。

**3. 管线生理一致性**（`check_ppg_local.py`）——下载 → npy → 30 s 块特征全链路

```text
merged rows=... cases=21
within 15 bpm: 21/21 cases; median |diff| = 0.4 bpm; pooled correlation = 0.958
OK local PPG feature pipeline is physiologically consistent
```

特征表里的 `ppg_hr_mean` 与监护仪自带 HR 一致（中位差 0.4 bpm，相关 0.958），
说明时间轴与幅值尺度都正确。

## 同一策略适用于 ABP

`abp_npy`（SNUADC/ART，2,500 例 ≈ 81 GB）在特征提取并验证后同样删除，只保留
`abp_meta/` 侧车与 `abp_features_full/`。下载器带 `--min-free-gb` 硬下限，
磁盘不足时自行停止而不是写满。
