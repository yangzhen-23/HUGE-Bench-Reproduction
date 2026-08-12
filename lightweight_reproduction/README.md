# HUGE-Bench 轻量本地复现

回答：我主要复现和验证了 HUGE-Bench 的数据与多阶段标注结构。官方每条高级无人机轨迹会被划分为多个子任务阶段，例如朝向目标、飞至目标上方和下降到指定高度。轻量复现将这些阶段标注标准化为 episode 和 stage 记录，然后检查轨迹数量、阶段数量、任务类型、阶段边界、阶段连续性和完整覆盖关系，同时通过 SHA-256 等方式验证公开标注文件完整性。这里并没有运行 VLA 模型或复现论文的模型性能。

本项目在离线、CPU-only 环境中复现 HUGE-Bench 官方公开数据的任务结构、多阶段标注统计与标注完整性检查。

允许得出的结论只有：**复现了 HUGE-Bench 官方公开数据的任务结构、多阶段标注统计与标注完整性检查。** 不应扩展为以下结论：

1. 未复现 PI0/PI0.5 模型性能。
2. 未复现论文 TCR、nDTW、NSP、CR 或 CSPL 主实验数值。
3. 未完成 3DGS 渲染、闭环飞行或碰撞评估。
4. `recovered_from_released_actions` 不等同于原始人工阶段标注。

## 前置条件与输入

需要 Python >= 3.11、NumPy >= 1.26、matplotlib >= 3.8、Pillow >= 10；开发和验证另需 pytest。流程只使用 CPU，不需要 GPU、JAX、OpenPI、Isaac Sim，也不需要批量下载 Hugging Face 数据。

默认目录结构要求标注 sidecar 位于 `HUGE-Bench/trajectory_generation/stage_annotations/`，`--repo-root` 指向官方嵌套仓库 `HUGE-Bench/`。运行清单会记录该嵌套仓库的当前 commit；CLI 只读取它，不执行 Git 写操作。

`run_manifest.json` 的 `command` 保存入口边界可知的命令 provenance：模块入口会如实记录 `Python 可执行文件 -m huge_lightweight.cli ...`，console-script 入口记录其真实进程命令；程序化调用 `main(argv)` 因不存在外部 launcher，使用明确的 canonical 表示 `huge-lightweight ...`。直接调用 `run_pipeline(..., command=...)` 时则逐 token 原样保存调用方给出的序列。

## Windows PowerShell：安装并运行

```powershell
cd lightweight_reproduction
& 'D:\anaconda3\python.exe' -m pip install -e .
huge-lightweight `
  --annotations-root ..\HUGE-Bench\trajectory_generation\stage_annotations `
  --repo-root ..\HUGE-Bench `
  --output .\outputs
```

完成 editable install 后，也可用模块入口：

```powershell
& 'D:\anaconda3\python.exe' -m huge_lightweight.cli `
  --annotations-root ..\HUGE-Bench\trajectory_generation\stage_annotations `
  --repo-root ..\HUGE-Bench `
  --output .\outputs
```

若暂不安装，可显式设置 src-layout 的 `PYTHONPATH`：

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
& 'D:\anaconda3\python.exe' -m huge_lightweight.cli `
  --annotations-root ..\HUGE-Bench\trajectory_generation\stage_annotations `
  --repo-root ..\HUGE-Bench `
  --output .\outputs
```

使用完毕后请恢复此前的 `PYTHONPATH`，或执行 `Remove-Item Env:PYTHONPATH` 删除本次设置。

## 成功门与文件清单

官方 sidecar 的终端成功门应为：

```text
LIGHTWEIGHT_REPRODUCTION_OK episodes=6168 segments=27539
```

成功目录共 17 个生成文件：`tables/` 下 5 个 CSV，`figures/` 下 8 个 PNG，Markdown 报告、HTML 报告、`summary.json` 和最后写入的 `run_manifest.json`。清单的 `outputs` 对前 16 个文件记录流式 SHA-256 与字节数，不对自身做哈希。官方集成还要求 `resources.peak_python_memory_bytes < 314572800`（小于 300 MiB）。已有输出目录中的无关文件会保留，仅覆盖这些已知文件名。

标注验证不通过时进程返回退出码 2，只生成 `tables/validation_report.csv` 和诊断 `summary.json`，并在 stderr 输出简短失败诊断。若工作树因 Windows CRLF 与发布的 LF 字节不同，验证报告会同时列出 raw 与 canonical 指纹；只有 canonical 大小和 SHA-256 同时匹配时才按 LF 规范化通过，可据此排查换行转换问题。

## 测试与离线性

```powershell
& 'D:\anaconda3\python.exe' -m pytest tests -v
& 'D:\anaconda3\python.exe' -m compileall -q src
```

加载、验证、统计、指标 smoke、绘图和报告生成均为本地离线操作，运行时不会访问网络。指标 smoke 仅对 `HUGE-Bench/metric.py` 做合成轨迹兼容性探测；SKIP 不阻断标注复现成功。

## 下一步边界

本地数据与标注层验收后，下一阶段是 AutoDL Stage A；模型权重、GPU 推理、模拟器和闭环评估不属于本地轻量流程，也不应由本命令隐式启动。
