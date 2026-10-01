# c. 源代码

本目录与仓库原件**逐字节一致**（sha256 见 `../MANIFEST.json`）。`remote/` 是算法与流水线，`tools/` 是数据清单、制图与核查工具。文件顶部 docstring 说明了各自职责，下表是索引。

## 3.1 `remote/`　流水线（22 个，按链路顺序）

| 文件 | 作用 |
|---|---|
| `dl_vital.py` | 下载完整 `.vital` 打包文件（权威原始数据） |
| `dl_tracks.py` | 按轨道下载数值 CSV |
| `dl_ppg.py` | 下载波形轨道，落盘即压缩为二进制 |
| `build_ppg_features.py` | PPG 波形特征，与数值决策时刻对齐 |
| `build_abp_features.py` | 有创动脉压波形特征，与 PPG 共用 30 s 块网格 |
| `build_dataset.py` | **核心**：构建数据集（1 Hz 网格、窗口、标签、67 维特征）；含 `ENDPOINTS` 定义与 `window_row`，被流式推理复用 |
| `train_model.py` | 训练与评估低血压预测器 |
| `train_fused.py` | 波形到底有没有增益：同一固定划分上训练两变体对比 |
| `p0_eval.py` | **核心**：基线、事件级指标与配对显著性检验 |
| `redefine_eval.py` | 按评测协议重算（含 onset 口径） |
| `operating_points.py` | 给出临床可用工作点（不止 0.5 阈值） |
| `dump_scores.py` | 导出测试集分数与特征重要度，供报告绘制真实 ROC/PR |
| `calibration_dca.py` | 校准（Brier/ECE）与决策曲线分析 |
| `alarm_policy.py` | 报警层：把逐窗分数变成监护仪报警（含回差与归并） |
| `alarm_pareto.py` | 报警策略帕累托前沿：各（持续, 回差）设置的代价与收益 |
| `quality_gate.py` | 信号质量门控能否减少假报警、代价是多少 |
| `stream_monitor.py` | **流式推理**：喂样本即出风险与报警 |
| `make_monitor_display.py` | 显示原型：单例的通道、风险曲线与报警 |
| `check_p0_eval.py` | `p0_eval` 非平凡逻辑自检（不需要数据） |
| `check_abp_features.py` | 动脉压提取器合成自检 |
| `check_build_refactor.py` | 抽取后的事件定义仍能复现标签 |
| `check_extra_merge.py` | 两种 `--extra-features` 合并模式的小样本测试 |

## 3.2 `tools/`　清单、制图与核查（20 个 Python + 1 个 Node）

| 文件 | 作用 |
|---|---|
| `build_channel_registry.py` | 设备/通道清单：项目涉及的全部设备与本机实有 |
| `pick_functional_channels.py` | 从索引中挑出值得下载的功能通道 |
| `record_downloads.py` | 记录功能通道下载的真实产出（41 路 / 66,375 文件 / 32.38 GiB） |
| `remote.py` | 计算节点的 SSH/SFTP 助手 |
| `capture.py` | 在远端执行脚本并把 stdout 存为 UTF-8 |
| `make_figures.py` | 渲染报告全部 16 张图 |
| `make_pdf.py` | 由 Markdown 生成自包含 PDF |
| `normalize_report_headings.py` | 规范报告标题层级并修正锚点 |
| `check_report_numbers.py` | **数值回算**：报告中的每个数值能否从产物复现（239 值 0 失配） |
| `verify_pdf.py` | 核验 PDF 真实的页数/图数/数值/链接/乱码 |
| `final_audit.py` | 面向交付物的仓库自审（27 项） |
| `build_algorithm_delivery.py` | 生成 `算法实现/` 交付包并重算 sha256 与普查数字 |
| `check_algorithm_delivery.py` | 校验交付包与仓库双向一致、无残留文件 |
| `ppg_signal_check.py` | 校验下载的 PPG 样本并提取报告所需数字 |
| `verify_ppg.py` | PPG 数组真实性核验（是否为脉搏波形） |
| `verify_ppg_align.py` | PPG 时间轴与数值轨道对齐核验 |
| `make_slides.js` | 生成汇报幻灯片；每页数字同样从 `results/` 产物读取（Node + pptxgenjs） |
| `make_slides_pdf.py` | 幻灯片转 PDF 并核验：确认 PDF 与当前 pptx 一致、无占位文本，再渲染每页供视觉检查 |
| `check_all.py` | **一键运行全部核查**：逐项执行并汇总到 `results/check_all.json` |
| `build_latex_midterm.py` | 编译 `report/latex/MIDTERM.tex`（Tectonic，无需系统 TeX）并核验其 PDF：页数、中文字形、引用数值、模板占位字段 |
| `check_freshness.py` | 时间戳闸门：任何渲染产物（PDF / pptx）落后于它的来源即报错 |

## 3.3 依赖

`requirements.txt` 为该流水线的运行环境。三个自检脚本（`check_p0_eval.py`、`check_abp_features.py`、`check_build_refactor.py`）不需要数据集即可运行，便于先验证环境再跑全流程。
