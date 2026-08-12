# HUGE-Bench 轻量本地复现最终 QA

## 最终状态

- 正式成功门：`LIGHTWEIGHT_REPRODUCTION_OK episodes=6168 segments=27539`
- 正式运行时间（UTC）：`2026-08-12T19:39:55.903962+00:00` 至 `2026-08-12T19:40:02.971576+00:00`
- fresh 测试：76 passed（35.02 s）；`compileall`：exit 0。
- 官方计数：6,168 episodes、27,539 stage segments、2,059,490 frames；split episodes 为 5,175 / 576 / 417；8 tasks、7 scenes。
- 官方仓库 commit：`d000b99794e9da85dff117db61ef874659b91ae8`
- 输入 manifest SHA-256：`eef03c4acb5ade46c64e12c799d480b8bdf918f30ec0d93058692f27ea4c95e7`
- 标注验证：147,263 PASS / 0 FAIL。
- 指标兼容 smoke：PASS；仅为 synthetic compatibility probe，不是论文结果。
- 资源：measured=true；Python traced peak 99,371,512 bytes（94.77 MiB），低于 300 MiB；pipeline elapsed 7.067 s。
- 输出审计：17 个已知文件；manifest 声明 16 个排序指纹、无 self-hash；以 1 MiB 分块复算 SHA-256/bytes，0 mismatch；无无关额外文件。

## Artifact Tool CSV QA

| CSV | Sheet / used range | 数据行 | 状态 |
|---|---|---:|---|
| `dataset_overview.csv` | `dataset_overview!A1:K5` | 4 | PASS |
| `task_statistics.csv` | `task_statistics!A1:K33` | 32 | PASS |
| `task_scene_matrix.csv` | `task_scene_matrix!A1:H9` | 8 | PASS |
| `stage_statistics.csv` | `stage_statistics!A1:L39` | 38 | PASS |
| `validation_report.csv` | `validation_report!A1:F147264` | 147,263 | PASS |

五表均由 `Workbook.fromCSV` 实际导入并经 `workbook.inspect` 核验：单一精确 sheet、精确 header、无额外 header/ragged row、末行可达。`dataset_overview overall`、task overall 合计、task-scene 合计和 validation PASS 数分别为 6,168 / 6,168 / 6,168 / 147,263，与 `summary.json` 一致。

## 图像与 nature-figure QA

| 图 | Mode | 尺寸 px | DPI | 二进制 / 视觉 |
|---|---|---:|---:|---|
| `01_split_overview.png` | RGBA | 1316×775 | 180.01 | PASS / PASS |
| `02_task_distribution.png` | RGBA | 1316×752 | 180.01 | PASS / PASS |
| `03_task_scene_heatmap.png` | RGBA | 1314×856 | 180.01 | PASS / PASS（单图复核） |
| `04_episode_length_distribution.png` | RGBA | 1316×752 | 180.01 | PASS / PASS |
| `05_stages_per_episode.png` | RGBA | 1316×752 | 180.01 | PASS / PASS |
| `06_stage_duration_by_task.png` | RGBA | 1316×821 | 180.01 | PASS / PASS |
| `07_annotation_provenance.png` | RGBA | 1316×775 | 180.01 | PASS / PASS |
| `08_example_stage_timeline.png` | RGBA | 1316×890 | 180.01 | PASS / PASS（单图复核） |

八图均按批准文件名/顺序存在，RGB/RGBA、尺寸至少 800×450、双轴 DPI 在 180±0.1、每个 RGB 通道标准差非零。原分辨率 contact sheet 显示无空白、裁切、重叠或不可读主标签；密集热图和时间线另行原分辨率复核通过。

`nature-figure` 通用 journal preflight 原始结果为 11 PASS、1 WARN、2 FAIL，且无其他发现。三项任务特定透明 waiver：缺少 SVG/PDF（批准交付为 PNG-only offline report）；缺少 TIFF（同一交付约束）；180 DPI 低于通用 300 DPI（批准报告契约精确要求 180 DPI，且已由 PNG binary probe 验证）。

## 报告安全与范围

Markdown 与 HTML 均精确引用 8 图和 6 个可下载表格/summary 链接，包含指标标签与 provenance caveat，不含 Task-6 placeholder。HTML 无 script、事件处理器、外部 URL、data URI 或不安全相对路径。

允许结论只有：**复现了 HUGE-Bench 官方公开数据的任务结构、多阶段标注统计与标注完整性检查。**

明确排除：

1. 未复现 PI0/PI0.5 模型性能。
2. 未复现论文 TCR、nDTW、NSP、CR 或 CSPL 主实验数值。
3. 未完成 3DGS 渲染、闭环飞行或碰撞评估。
4. `recovered_from_released_actions` 不等同于原始人工阶段标注。

## ZIP 交付

- 文件名：`HUGE_Bench_轻量本地复现包.zip`
- 字节数：`20,670,128`
- 条目数：`37`
- SHA-256：external definitive hash，见 delivery implementation report 与包外 `zip_sha256.txt`。
- `ZipFile.testzip()`：PASS；条目按 POSIX 路径排序且无绝对路径、反斜杠、`..`、重复项、symlink、cache 或 forbidden root。
- ZIP 不含 `node_modules`、QA 脚本/contact sheet、cache、官方数据/仓库、论文/PPT/DOCX、旧 `stage_a/` 或 `.superpowers/`。
