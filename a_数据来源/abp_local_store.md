# ABP（有创动脉压波形）本地库：验证记录与结果

**结论：管线可用。** 关键意义在于对比：PPG 与 ABP 都是「新增波形模态」，但 ABP 携带
的是**压力**形态信息（每搏脉压、上升支斜率、PPV），而 PPG 携带的是**容积/搏动**信息
——标签本身是压力事件。这正是「哪一模态值得加」这个问题的核心。

## 数据与派生量

| 项目 | 值 |
|---|---|
| 轨道 | `SNUADC/ART`（约 500 Hz） |
| 下载范围 | 2,500 例（与「数值 + PPG」队列取交集，且该例有 ART 轨道） |
| 下载量 | 62.95 GB，0 失败，压缩比 2.23× |
| 派生特征 | `abp_features_full/abp_features.parquet`：1,058,390 行 × 65 维，0 错误 |
| 特征内容 | 每搏脉压（pp/pp_sd/PPV）、上升支最大斜率 dP/dt、上升时间占比、收缩/舒张值、波形偏度/峰度、每 30 s 搏动数；再按 10 个 30 s 块聚合 |

原始 `.npy`（62.95 GB）按既定策略在特征提取后删除，仅保留侧车元数据。

## 验证（对照监护仪自身数值）

```text
cases=2500 merged rows=1058390
HR within 15 bpm: 2470/2500 cases; median |diff| = 1.6 bpm; correlation = 0.953
beat count per 30 s block: median 34.6
pulse pressure vs numeric (SBP-DBP): Spearman = 0.941
waveform coverage of windows: median 1.000
OK ABP feature pipeline is physiologically consistent
```

- **心率**：由每搏间期反推，与监护仪 HR 中位差 1.6 bpm、相关 0.953。
- **每搏脉压**：与监护仪自身的 SBP−DBP 秩相关 **0.941**。波形幅度是 native 单位
  （非 mmHg），所以绝对量级不可比，但秩一致性说明形态与量级都对。
- **PPV**：均值 12.0%，处于生理合理区间（机械通气下常见 5%–20%）。

## 一个被检查拦下的真实 bug（记录在案）

首版提取器把「上升支」写成 `v[trough:p0+1]`，而 `trough > p0`，因此**切片恒为空**，
每搏都被丢弃 → 所有形态特征为 NaN。它在真实数据上看起来像「该模态没有信息」，
极可能被误读为「ABP 无增益」的阴性结论。

发现方式不是代码审查，而是**先设的门槛检查**（`check_abp_local.py`：心率对不上就停），
随后用合成信号定位（`diag_abp_detect.py`：原始数据能检出 40 个搏动，函数却返回 NaN）。
修正后新增合成自检 `remote/check_abp_features.py`：对已知心率/脉压的合成波形断言
恢复值（50/60/90/120/150 bpm 全部命中，pp 39.9 对真值 40.0），并断言平坦信号与
过慢信号必须返回 NaN 而不是编造数值。
