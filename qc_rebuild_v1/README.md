# QC_Rebuild_v1：胚胎F0逐帧质控共享包

本目录整理自2026-11-13导出的QC项目快照，用于组内交接、结果追溯和复用。
原项目快照完整保存在`archive/source_snapshot.tar.gz`，91个原始文件的SHA256已核对。
原GitHub仓库的`README.md`、`scripts/frame_qc/`保留为历史版本，本目录独立新增。

## 先看已确认的版本

| 项目 | 本包记录 |
|---|---|
| 模型 | 二分类ResNet18，seed 42，最佳checkpoint epoch 6 |
| 权重SHA256 | `54b65cbb9d626051b2b9c6f1827eef215ba4f5b2da48183128ec7f3aafad1e30` |
| 类别与分数 | 0=KEEP，1=DELETE；`softmax(logits, dim=1)[:,1]` |
| 判定方向 | `score_delete >= threshold`为DELETE候选 |
| 训练/验证 | 24/6个胚胎；11,607/2,454帧；从ImageNet初始化 |
| 独立测试 | 15个胚胎、6,940帧；冻结测试阈值0.001 |
| 独立测试结果 | 6,101张KEEP全部保留，839张DELETE全部识别；该测试集误删/漏检均为0 |
| 原始全量F0推理 | 704个胚胎、342,363帧；阈值0.001；KEEP 296,005，DELETE 46,358，REVIEW 0 |
| 后续开发审计 | 包含0.01、0.013、0.02候选方案的历史记录 |
| 最终部署阈值 | 上传快照中没有找到批准最终部署策略的记录；测试协议`deployment_approved=false` |

独立测试集上的0误判只描述这6,940张已标注帧，不能作为整个F0数据集零误删的保证。
0.013文件夹是候选审计，不代表0.013已经冻结，也不能替换原来0.001的独立测试结论。
`audit_0p013_unreviewed_200.csv`的200张新样本尚无人工KEEP/DELETE判定，不能据此计算新一轮实测误判率。

## 预处理与输入范围

`code/qc_image_io_v1.py`和`code/qc_preprocess_v1.py`从已验证运行目录逐字节复制，没有修改。

1. 严格解码；仅当JPEG被判定为截断时，尝试允许截断解码，并记录`TRUNCATED_ALLOWED`。
2. 转灰度，逐帧P1–P99拉伸；百分位上下限相等时保留原强度。
3. 完整视野等比例缩放并黑色填充为224×224；不是胚胎bbox裁剪。
4. 重复为RGB三通道，应用ImageNet mean/std。

这是**F0技术可用性评分**，不是生物学活性/胚胎存活预测，也不是已验证的七焦面联合QC。
F0候选DELETE不意味着其他6个焦平面也应删除。
保留原始RUN、时间戳和时间缺口；下游使用mask时不要压缩时间轴、重新编号或自动删除原图。

## 目录

| 位置 | 用途 |
|---|---|
| `code/infer_qc.py` | 新增可移植推理入口；显式指定路径和阈值；保留CSV行顺序和原字段 |
| `code/qc_image_io_v1.py`、`qc_preprocess_v1.py` | 原始固定解码与预处理 |
| `tools/recheck_thresholds.py` | 从已有人工标签重新计算各阈值在该复核样本上的误判 |
| `tools/collect_missing_from_server.py` | 仅打包缺失的模型/全量逐帧评分/独立测试CSV，不运行训练或推理 |
| `manifests/` | 已修正200帧审计标签、923帧标签登记、待标注200帧、45组QC名单、704组上游划分 |
| `reports/` | 原独立测试记录、原全量统计、训练配置、候选阈值历史与重算结果 |
| `archive/` | 完整原始快照与原始文件清单；历史路径和记录保持原样 |
| `docs/` | CSV字段、版本差异、验证结果、GitHub上传步骤 |

## A. 先复现200帧阈值比较：只需要Python标准库

在本目录内运行：

```bash
python3 tools/recheck_thresholds.py \
  --labels manifests/reviewed_200_corrected.csv \
  --output /tmp/qc_reviewed_200_thresholds.csv
```

输出文件必须是新路径。11个阈值默认0.010至0.020，步长0.001。
人工标签更正后的这一批200帧实际KEEP=130、DELETE=70。
例如0.01下误删30、错误保留0；0.013下误删2、错误保留13。
这些是混合抽样复核集的观察计数，并非整个F0的误判率；边界富集样本不能直接代表全量。
历史外推表及其假设单独存放于`reports/historical_threshold_extrapolation.csv`和对应JSON。
后者不是实测全量错误数。

## B. 运行模型评分：需要另行提供权重和原图

本小包没有模型权重和原始图像。补齐权重后，放在`weights/model.pt`或其他路径。
建议先使用原运行兼容环境：Python 3.8.10、torch 2.4.1、torchvision 0.19.1、NumPy 1.24.4、Pillow 10.4.0。
CPU运行可安装`requirements-inference.txt`；CUDA安装方式按接收方机器选择，原记录使用cu121构建。

```bash
python3 -m pip install -r requirements-inference.txt
python3 code/infer_qc.py --help
```

输入CSV至少包含`embryo_id,run,image_path`；可附带`upstream_split`或其他原始列。
使用相对图片路径时：

```bash
python3 code/infer_qc.py \
  --index /path/to/f0_manifest.csv \
  --checkpoint /path/to/model.pt \
  --image-root /path/to/embryo_dataset \
  --threshold 0.001 \
  --output-dir /path/to/new_qc_scoring_run \
  --device cpu
```

这里的0.001用于复用**原测试/原全量评分规则**，不是建议冻结为新的部署阈值。
若CSV保留了旧服务器绝对路径，例如`/home/gpuserver2/datasets/nantes_v3/extracted/f0/embryo_dataset/胚胎/文件.jpeg`，则额外加入：

```text
--old-image-prefix /home/gpuserver2/datasets/nantes_v3/extracted/f0/embryo_dataset
```

同时将`--image-root`设置为接收方实际的`embryo_dataset`目录。
仅指定`--image-root`不会覆盖绝对路径；必须明确指定旧前缀，避免猜测或误映射。
与例子不同的原始路径，应使用CSV实际对应的旧前缀。

该入口验证固定权重SHA256、epoch和两份预处理脚本哈希；不自动下载权重。
阈值必须显式提供，无默认部署阈值。所有输出都是候选评分，`deployment_approved=false`。
输出`predictions.csv`及`run_record.json`；不写回输入、不删除图像，也不修改训练/验证/测试划分。
解码或预处理失败的帧标记REVIEW，保留该行，不强行归为DELETE。
同一`embryo_id/run`重复时停止，以避免把多个焦平面混在同一张F0索引中。

新入口是本次交接新增的包装器，不是当时生成历史报告的原始脚本。
由于当前小包未附权重，尚未端到端验证模型输出与原始逐帧分数一致；补齐后应先用少量相同图像在CPU核对分数与解码模式。
原始`run_qc.py`位于归档中，仍包含原服务器路径和固定的0.001阈值，不能直接视为通用命令行工具。

## C. 在原服务器补齐缺失文件

将`tools/collect_missing_from_server.py`上传到服务器任意位置，执行：

```bash
python3 /path/to/collect_missing_from_server.py \
  --project-root /home/zhl/projects/QC_Rebuild_v1
```

它会在`/home/zhl/projects/`生成新的`QC_rebuild_supplement_时间.zip`，包含：

- 已确认SHA256的F0批次`model.pt`和`f0_manifest.csv`；
- `per_embryo/`的原始逐帧评分，预期704个CSV、342,363行；
- 独立测试运行目录中的逐帧清单、预测、人工标签及每胚胎统计（存在时）；
- 原始训练/标注数据目录中的小型索引与协议文件（存在时）。

原始图像不打包，模型不重训，QC不重跑。若缺少文件，会在导出报告中明确标记PARTIAL_EXPORT。
发现已知文件哈希不符时停止，不把不同版本混在一起。
补充包用于组内传输；模型权重和逐帧大表不要求加入Git历史。
完整重训仍需原始图像、ImageNet初始化权重及训练脚本引用的完整审计/冒烟协议；本包没有声称完整重训已经可复现。

## D. GitHub上传与共享

将整个`qc_rebuild_v1`目录放在现有仓库根目录，原版文件保持不变。
本目录的`README.md`在GitHub进入目录后会自动显示。
具体步骤见`docs/GITHUB_UPLOAD.md`。

## 当前验证范围

已核对91个源文件哈希、两份预处理哈希、CSV字段与行数；已重新计算并对齐11个阈值下的200帧审计误判计数。
新增脚本已通过语法、帮助命令和适用的标准库功能检查。
未获得权重/原图，未实际执行模型推理、训练或独立测试重跑。
