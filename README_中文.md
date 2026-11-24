# DSC熔融峰全自动解析 使用说明

本文件夹包含DSC算法代码、依赖、使用说明和输入模板。论文原始实测数据、训练数据、验证数据及预训练权重未包含在内。

## 文件用途

| 文件/目录 | 用途 |
|---|---|
| `src/peakbot/dsc_peakbot/` | DSC预处理、网络、检峰、边界、基线和热参数函数 |
| `train.py` | 120/240点模型的便携训练入口 |
| `train_family_models.py` | 保留的训练实现，供入口调用 |
| `fs_workflow.py` | 调用原FS函数，处理路径、信号方向和文件读写 |
| `fit_all_fs_peaks.py` | 保留的FS拟合函数 |
| `generate_all_fs_synthetic.py` | 保留的FS合成曲线函数 |
| `dsc_peakbot_cli.py` | 单模型检测及读取检测结果计算热参数 |
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

数据目录见英文README。`my_dataset/splits/narrow/real_train`与`real_val`各放Excel和`references.tsv`，宽峰对应`broad`目录。联合训练还需`synthetic_narrow`或`synthetic_broad`中的合成曲线及标注。

```text
python train.py --data-root my_dataset --family narrow --variant real_only
python train.py --data-root my_dataset --family broad --variant real_fs
```

入口选择已有120/240点配置，对应2/4℃目标温区。输出在`runs`下，终端会打印实际模型路径。原计算函数和网络结构没有为本次整理而修改。保留的旧底层入口包含历史默认配置，建议使用本说明给出的`train.py`。

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

检测入口按原函数导出各类候选，热参数入口只计算`melt_peak`，采用检测文件中保存的起止边界，调用已有基线、积分、峰温和外推温度函数。外推温度输出含状态列，表示切线交点或边界回退。两模型可分别运行，本次新增入口没有另行设计算法来综合两个模型的结果。

## 人工示例

```text
python examples/make_demo.py --output workspace/demo
python train.py --data-root workspace/demo --family narrow --variant real_only --epochs 1
python dsc_peakbot_cli.py quantify workspace/demo/splits/narrow/real_val --detections workspace/demo/ARTIFICIAL_narrow_detections.tsv --metadata workspace/demo/narrow_metadata.csv --output workspace/demo/thermal.csv
```

人工示例用于检查程序能否运行，不复现论文性能。人工曲线、模型权重和测试输出在运行时生成，未作为实测数据包含在发布包中。

本次实际执行的检查见`VALIDATION.md`，文件校验值见`SHA256SUMS.txt`。检查使用本机已有环境，尚未执行全新环境安装；上传后还需从仓库重新下载检查一次。

## 发布前

仓库URL、实际发布版本及软件存档DOI在创建后填写，并相应更新软件引用信息。
