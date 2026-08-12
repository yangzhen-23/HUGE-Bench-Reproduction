# HUGE-Bench Reproduction

本仓库用于对 **HUGE-Bench: A Benchmark for High-Level UAV Vision-Language-Action Tasks** 进行分阶段复现与实验验证。

目前项目仍处于复现初期。现阶段主要目标不是完整复现论文中的全部 VLA 模型实验，而是首先完成对 HUGE-Bench 数据组织、任务结构、评估流程和实验结果形式的理解，并构建一个可以在普通本地环境运行的轻量级复现流程。

后续将在 **AutoDL GPU 环境** 中继续完成官方代码运行、真实 Benchmark 流程验证以及 VLA 模型推理实验。

------

## 1. 项目目标

HUGE-Bench 面向高层级无人机 Vision-Language-Action（VLA）任务，用于评估智能体根据自然语言指令，在复杂三维环境中完成多阶段无人机任务的能力。

本复现项目采用逐步推进的方式：

> **先理解 Benchmark → 再跑通评估流程 → 最后进行模型推理与实验复现**

目前重点关注以下内容：

- HUGE-Bench 数据与任务结构
- Scene / Task / Episode / Stage 等层级关系
- 数据统计与可视化
- Benchmark 数据合法性检查
- 基础评估指标流程验证
- 实验结果自动生成与汇总
- 为后续 AutoDL GPU 复现建立基础

------

## 2. 当前阶段

### Stage 0：Lightweight Reproduction

**状态：已完成基础版本**

目前首先实现了一个轻量级、本地可运行的 HUGE-Bench 分析与复现框架。

这一阶段主要用于：

- 熟悉 HUGE-Bench 数据格式
- 分析任务和场景分布
- 分析 Episode 与 Stage 结构
- 验证数据读取流程
- 验证部分评估逻辑
- 自动生成实验统计表
- 自动生成可视化结果
- 自动生成轻量复现报告

本阶段尽量避免：

- 下载完整大规模场景数据
- 加载大型 VLA 模型
- 进行高显存 GPU 推理
- 完整复现论文所有模型结果

因此，该阶段主要面向 **Benchmark 理解、数据分析与基础实验验证**，并不代表 HUGE-Bench 的完整模型复现。

------

## 3. 当前项目结构

```text
HUGE-Bench-Reproduction/
│
└── lightweight_reproduction/
    │
    ├── README.md
    ├── FINAL_QA.md
    ├── pyproject.toml
    │
    ├── src/
    │   └── huge_lightweight/
    │       ├── analysis.py
    │       ├── cli.py
    │       ├── loader.py
    │       ├── metric_smoke.py
    │       ├── models.py
    │       ├── plots.py
    │       ├── report.py
    │       └── validation.py
    │
    ├── tests/
    │   ├── test_analysis.py
    │   ├── test_cli.py
    │   ├── test_loader.py
    │   ├── test_metric_smoke.py
    │   ├── test_plots.py
    │   ├── test_report.py
    │   └── test_validation.py
    │
    └── outputs/
        ├── figures/
        ├── tables/
        ├── summary.json
        ├── run_manifest.json
        └── HUGE_Bench_轻量复现报告.*
```

------

## 4. 当前已完成内容

### 4.1 数据读取

实现基础的数据加载与结构解析流程，用于读取和整理 HUGE-Bench 相关数据。

主要包括：

- Dataset 信息读取
- Task 信息解析
- Scene 信息解析
- Episode 信息解析
- Stage 信息解析

------

### 4.2 数据统计分析

当前已经实现部分基础统计功能，包括：

- 数据集整体规模统计
- 不同任务的数据分布
- Task × Scene 分布统计
- Episode 长度统计
- Episode 中 Stage 数量统计
- 不同任务 Stage 持续时间统计
- Annotation 来源统计

------

### 4.3 可视化结果

当前轻量复现可以生成多种统计图：

```text
01_split_overview.png
02_task_distribution.png
03_task_scene_heatmap.png
04_episode_length_distribution.png
05_stages_per_episode.png
06_stage_duration_by_task.png
07_annotation_provenance.png
08_example_stage_timeline.png
```

这些结果主要用于理解 HUGE-Bench 数据集的组成、任务结构以及轨迹阶段信息。

------

### 4.4 统计表格

当前可以生成：

```text
dataset_overview.csv
stage_statistics.csv
task_scene_matrix.csv
task_statistics.csv
validation_report.csv
```

用于进一步分析不同任务、场景和阶段的数据分布情况。

------

### 4.5 Validation

实现了基础的数据验证流程，用于检查：

- 数据字段完整性
- Episode / Stage 结构
- 数据之间的关联关系
- 部分异常数据情况

并生成：

```text
validation_report.csv
```

------

### 4.6 Metric Smoke Test

目前实现的是评估指标的 **Smoke Test**。

主要目的不是完整复现论文最终指标，而是确认：

> 数据 → Prediction / Trajectory → Metric

这一基本评估链路可以正常运行。

完整的官方评估指标将在后续 GPU / AutoDL 阶段继续验证。

------

### 4.7 自动化实验报告

当前流程能够自动整理实验结果并生成：

```text
HUGE_Bench_轻量复现报告.md
HUGE_Bench_轻量复现报告.html
```

用于快速查看：

- 数据统计
- 可视化结果
- 数据验证结果
- 当前复现状态

------

## 5. 当前阶段的定位

当前工作的准确定位是：

```text
HUGE-Bench
    │
    ├── 数据理解                 ✓
    ├── Task / Scene 分析        ✓
    ├── Episode / Stage 分析     ✓
    ├── 数据统计                 ✓
    ├── 数据可视化               ✓
    ├── 数据 Validation          ✓
    ├── Metric Smoke Test        ✓
    │
    ├── 官方 Benchmark Pipeline  →
    ├── 真实场景数据加载          →
    ├── VLA 模型推理              →
    ├── 官方指标计算              →
    └── 完整论文结果复现          →
```

因此，目前属于：

> **Stage 0 — Lightweight / Local Reproduction**

主要解决“理解 Benchmark 并建立最小可运行实验框架”的问题。

------

## 6. 后续复现计划

后续实验将主要迁移至 **AutoDL GPU 环境**。

计划按照以下路线逐步推进。

### Stage 1：官方环境与代码验证

目标：

- 配置 CUDA / PyTorch / Python 环境
- Clone HUGE-Bench 官方代码
- 安装官方依赖
- 验证核心模块能够正常 import
- 跑通官方基础脚本

预期结果：

```text
Environment
    ↓
Official HUGE-Bench Repository
    ↓
Dependencies
    ↓
Basic Script
    ↓
Successful Execution
```

------

### Stage 2：最小 Benchmark Pipeline

这一阶段不会直接运行完整数据集，而是优先建立一个最小闭环：

```text
HUGE-Bench Data
      ↓
One Scene
      ↓
One Task
      ↓
Few Episodes
      ↓
Prediction / Trajectory
      ↓
Official Evaluation
      ↓
Metrics
```

核心目标是证明：

> HUGE-Bench 官方数据、推理结果和评估代码能够真正形成端到端闭环。

------

### Stage 3：VLA Baseline 推理

在 Benchmark Pipeline 跑通之后，再逐步尝试论文中的代表性模型。

计划关注的模型包括：

- OpenVLA
- FastVLM
- MemoryVLA
- π0
- π0.5
- Depth-aware π0.5

考虑到不同模型的：

- 参数规模
- GPU 显存需求
- 环境依赖
- 模型权重大小
- 数据要求

将优先选择 **计算资源消耗较小、依赖较简单的模型** 进行最小规模实验。

------

### Stage 4：官方评估指标复现

进一步验证 HUGE-Bench 中面向：

- Task completion
- Process / Stage
- Trajectory
- Collision
- Flight safety

等维度的评估方法。

目标是使用真实模型输出计算官方 Benchmark 指标，而不仅仅进行 Smoke Test。

------

### Stage 5：扩大实验规模

在单任务 / 单场景实验成功之后，再逐渐扩展到：

```text
Few Episodes
     ↓
One Task
     ↓
Multiple Tasks
     ↓
Multiple Scenes
     ↓
Multiple Models
```

最终尝试复现论文中具有代表性的部分实验结果。

------

## 7. 整体复现路线

```text
Level 0
Local Lightweight Reproduction
数据分析 / 可视化 / Validation
            │
            ▼
Level 1
Official Benchmark Pipeline
少量 Scene / Task / Episode
            │
            ▼
Level 2
VLA Model Inference
运行至少一个 Baseline
            │
            ▼
Level 3
Official Evaluation
真实模型输出 + 官方指标
            │
            ▼
Level 4
Extended Reproduction
多模型 / 多任务 / 多场景
```

本项目优先保证：

> **每一个阶段都形成一个可以验证的实验闭环。**

而不是一开始就追求完整复现全部论文结果。

------

## 8. 复现原则

本仓库遵循以下原则：

### Minimal First

优先运行：

- 最少数据
- 最少模型
- 最少显存
- 最短实验时间

在流程完全正确后，再逐步扩大实验规模。

### Reproducibility

尽可能记录：

- Python 版本
- CUDA 版本
- PyTorch 版本
- GPU 型号
- Package 版本
- 实验配置
- 输入数据
- 输出结果

### Progressive Verification

每增加一个实验模块，都优先验证：

```text
Input
  ↓
Pipeline
  ↓
Output
  ↓
Metric
```

避免在完整大规模实验中再定位基础问题。

------

## 9. 当前进度

| 模块                 | 状态 |
| -------------------- | ---- |
| HUGE-Bench 论文理解  | ✅    |
| 数据结构分析         | ✅    |
| Task / Scene 统计    | ✅    |
| Episode / Stage 分析 | ✅    |
| 数据可视化           | ✅    |
| 数据 Validation      | ✅    |
| Metric Smoke Test    | ✅    |
| 自动实验报告         | ✅    |
| 官方环境配置         | ⏳    |
| 官方数据 Pipeline    | ⏳    |
| AutoDL GPU 实验      | ⏳    |
| VLA Baseline 推理    | ⏳    |
| 官方评估指标         | ⏳    |
| 多模型完整实验       | ⏳    |

------

## 10. Disclaimer

当前仓库中的 `lightweight_reproduction` 主要用于 HUGE-Bench 的学习、分析以及最小复现验证。

当前结果：

- **不等价于论文完整实验结果**
- **不代表官方 benchmark performance**
- **尚未完成全部 VLA 模型推理**

后续实验将逐步迁移至真实 HUGE-Bench 官方数据、官方评估流程及 GPU VLA 模型推理。

------

## 11. Reference

**HUGE-Bench: A Benchmark for High-Level UAV Vision-Language-Action Tasks**

本仓库仅用于论文学习、科研复现与实验研究。

------

## 12. Status

```text
Current Stage:
Stage 0 — Lightweight Reproduction ✅

Next Stage:
Stage 1 — AutoDL + Official HUGE-Bench Pipeline ⏳
```

**Work in progress.**