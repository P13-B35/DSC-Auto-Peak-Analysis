# DSC熔融峰全自动解析 使用说明

本文件夹包含DSC算法代码、依赖、使用说明和输入模板。论文原始实测数据、训练数据、验证数据及预训练权重未包含在内。

## 文件用途

| 文件/目录 | 用途 |
|---|---|
| `src/peakbot/dsc_peakbot/` | DSC预处理、网络、检峰、边界、基线和热参数函数 |
| `train.py` | 同一数据划分下的120/240点双尺度训练入口 |
| `shared_dataset.py` | 共享数据校验、清单与旧目录合并 |
| `train_family_models.py` | 训练实现，供入口调用 |
| `fs_workflow.py` | 调用原FS函数，处理路径、信号方向和文件读写 |
| `fit_all_fs_peaks.py` | 保留的FS拟合函数 |
| `generate_all_fs_synthetic.py` | 保留的FS合成曲线函数 |
| `dsc_peakbot_cli.py` | 双模型联合检测、保留单模型检测及热参数计算 |
| `templates/` | 只有表头的输入模板，不是实验数据 |
| `examples/make_demo.py` | 生成明确标记为人工曲线的运行示例 |
| `requirements.txt` | 根据本机已安装版本整理的依赖 |
| `CITATION.cff` | 软件引用信息 |
| `SHA256SUMS.txt` | 文件校验值 |

目录名保留`peakbot`是为了维持原有Python导入关系，其内部只收集DSC所需模块。

## 安装及输入

在此目录打开终端，使用独立Python环境：

```text
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

本地检查环境为Python 3.8.2、TensorFlow 2.13.0、CPU。依赖文件固定了本机版本；其他系统和GPU环境需另行配置。旧`.xls`格式另需`requirements-xls.txt`，建议使用`.xlsx`。

每条曲线一个Excel文件，首行为表头，第一列为`temperature`（℃），第二列为`heatflow`（mW），温度严格递增，熔融吸热峰向下。不能把电压或W/g直接当作mW输入。

训练标注使用`templates/references.tsv`的列顺序，类型为`melt_peak`、`baseline`或`artifact`。训练和验证按完整曲线划分：同一实测曲线及其派生合成曲线不能同时进入训练与验证。

## 训练自己的模型

在`my_dataset/splits/shared/real_train`与`real_val`中分别放训练、验证曲线及`references.tsv`。只按完整曲线划分一次，两种模型均使用该划分下的全部曲线。采用FS增强时，把全部FS合成曲线与标注放在`my_dataset/synthetic`，两种模型也共用这一批合成数据。

```text
python train.py --data-root my_dataset --family both --variant real_only
python train.py --data-root my_dataset --family both --variant real_fs
```

两种模型的名称表示观察尺度：2℃窗口重采样至120点，4℃窗口重采样至240点；不再要求把曲线人工分成窄峰和宽峰两组。标签仍是熔融峰、基线和伪峰。原始截窗点数随实际温度间隔变化，网络输入长度固定。网络结构、基线和热参数计算函数沿用现有实现。

输出在`runs`下，终端打印实际模型路径，并保存数据名单与内容摘要。增加`--prepare-only`可只检查数据并导出共享数据清单，不启动训练。更改`--seed`可训练不同随机种子模型，数据划分不会随训练种子改变。旧版分目录数据的合并方法见`python shared_dataset.py --help`；合并写入新目录，原数据保留。

## FS拟合与合成

用`templates/fs_manifest.csv`填写自己曲线的路径、材料、拟合区间和中心初值。`derived_curve`支持相对于清单文件的路径。`peak_family`使用原拟合规则支持的`narrow_metal`、`crystalline_organic`或`broad_polymer`。

```text
python fs_workflow.py fit --manifest my_fs_manifest.csv --output workspace/fs_fit --endotherm down
python fs_workflow.py generate --params workspace/fs_fit/fs_fit_params.csv --output my_dataset --n-per-material 24
```

包装入口将曲线复制为内部向上的拟合模板，调用已有拟合函数；生成后输出向下的mW合成曲线，不改动原始文件。合成曲线只用于训练或软件检查，不是新的实测数据。

原FS脚本的部分参数列名仍带`uV`，属于历史字段名，数值单位随输入信号。本入口明确使用mW，不进行电压到热流的标定。拟合边界、接受判据和扰动规则均保留，包括80℃附近的原有温区限制及幅值下限；将其用于不同温区或信号尺度前，应查看源函数，而非假定无需调整。

## 检测和热参数计算

训练完成后，用实际模型路径替换下面的`model.h5`：

```text
python dsc_peakbot_cli.py detect my_curves --family narrow --model model.h5 --output workspace/detections.tsv
python dsc_peakbot_cli.py quantify my_curves --detections workspace/detections.tsv --metadata my_metadata.csv --output workspace/thermal.csv
```

`my_metadata.csv`按`templates/metadata.csv`填写每个文件的实际质量（mg）及升温速率（℃/min）。如果所有文件的质量和速率确实相同，可用`--mass-mg 10 --heating-rate 10`代替`--metadata`；不能给不同质量文件统一填一个质量。

热参数入口只计算`melt_peak`，采用检测文件中保存的起止边界，调用已有基线、积分、峰温和外推温度函数。外推温度输出含状态列，表示切线交点或边界回退。新联合检测入口如下，多随机种子权重分别列在对应尺度后面：

```text
python dsc_peakbot_cli.py detect-joint my_curves --narrow-models small_seed1.h5 small_seed2.h5 --broad-models large_seed1.h5 large_seed2.h5 --heating-rate 10 --width-threshold-min 0.4602999429067669 --output workspace/joint_detections.tsv
python dsc_peakbot_cli.py quantify my_curves --detections workspace/joint_detections.tsv --metadata my_metadata.csv --output workspace/thermal.csv
```

两种尺度判别同一组极值候选，各自对多个种子的三类概率取均值，再筛选和去重。同一候选仅一路保留时采用对应边界规则；两路保留时，使用共同二次背景估计半高温度宽度，再除以升温速率得到分钟单位的时间宽度，按给定分界选择规则。测宽或边界失败保留状态，不作为正常热参数静默计算。

例子中的0.4602999429067669分钟是可追溯的历史开发值，不是这次新模型的重新标定结果，因此入口要求显式提供分界。不同曲线的升温速率不一致时，使用`--metadata`填写实际速率。概率、SNR和去重间距可在`detect-joint --help`中查看。原稿的已有性能数值不会自动变成修改后程序的评价结果。

## 人工示例

```text
python examples/make_demo.py --output workspace/demo
python train.py --data-root workspace/demo --family both --variant real_only --epochs 1
python dsc_peakbot_cli.py quantify workspace/demo/splits/narrow/real_val --detections workspace/demo/ARTIFICIAL_narrow_detections.tsv --metadata workspace/demo/narrow_metadata.csv --output workspace/demo/thermal.csv
```

人工示例用于检查程序能否运行，不复现论文性能。人工曲线、模型权重和测试输出在运行时生成，未作为实测数据包含在发布包中。

本次实际执行的检查见`VALIDATION.md`，文件校验值见`SHA256SUMS.txt`。检查使用本机已有环境，尚未执行全新环境安装；上传后还需从仓库重新下载检查一次。

## 当前实现说明

本次更新落实共享曲线训练和双尺度联合检测。默认每种尺度训练一个种子；论文中的每尺度五模型集成可通过`--seeds 20260713 20260714 20260715 20260716 20260717`指定。FS来源记录缺失或不完整时会标记为未核验，但不会阻止训练。

两处已确认的旧行为保持现状：定量基线拟合默认排除80℃以下的点；边界搜索到达搜索限位时，可能返回尚未满足阈值条件的位置。本次未继续修改这两处算法。更新范围见`CHANGELOG.md`。

## 发布前

仓库URL已写入软件引用信息；实际发布版本、论文引用及软件存档DOI在获得后填写。
