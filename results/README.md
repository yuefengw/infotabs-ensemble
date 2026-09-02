# Results

本目录保存 InfoTabS 五个 split 的最终预测结果、模型运行信息和结果检查记录。
结果已经检查过：样本数量完整，样本编号能够对应，预测标签格式正确，没有缺失或重复的最终结果。

## 标签和字段

预测标签只有三种：

- `E`：Entailment，表格内容支持该假设。
- `C`：Contradiction，表格内容与该假设相矛盾。
- `N`：Neutral，表格内容无法确定该假设是否成立。

宽表中的主要字段如下：

- `Split`：数据集分区，取值为 `train`、`dev`、`alpha1`、`alpha2` 或 `alpha3`。
- `Sample`：样本编号，例如 `dev_000001`。这是不同文件之间对应同一个样本的主要依据。
- `Gold`：InfoTabS 数据集提供的参考标签。
- `Qwen_Thinking`：Qwen3-8B 开启 thinking 条件的预测。
- `Qwen_NoThinking`：Qwen3-8B 关闭 thinking 条件的预测。
- `Llama`：Llama 3.1 8B 的预测，使用 Ollama `llama3.1:8b` Q4_K_M 量化模型。
- `Gemma`：Gemma 3 12B IT 的预测。
- `Mistral`：Mistral Nemo Instruct 2407 的预测。
- `DeepSeek`：DeepSeek-R1-Distill-Qwen-7B 的预测。
- `Phi`：Phi-4-mini-instruct 的预测。

## 文件说明

### `all_predictions.csv`

所有 split 合并后的宽表，共 23,738 条数据行。每一行是一个样本，每个模型占一列，
适合直接查看各模型预测、计算准确率、统计模型之间的一致性，以及作为后续 ensemble
或误差分析的输入。

### `train_predictions.csv`

只包含 `train` split 的宽表，共 16,538 条数据行。主要用于训练集上的模型比较、
错误样本检查和后续实验准备，不表示模型在训练过程中更新了参数。

### `dev_predictions.csv`

只包含 `dev` split 的宽表，共 1,800 条数据行。通常用于开发阶段的模型比较和参数选择。

### `alpha1_predictions.csv`

只包含 `alpha1` split 的宽表，共 1,800 条数据行。用于 InfoTabS 的第一组评估数据。

### `alpha2_predictions.csv`

只包含 `alpha2` split 的宽表，共 1,800 条数据行。用于 InfoTabS 的第二组评估数据。

### `alpha3_predictions.csv`

只包含 `alpha3` split 的宽表，共 1,800 条数据行。用于 InfoTabS 的第三组评估数据。

### `predictions_long.csv`

统一的长表，共 166,166 条数据行，即每个样本对应 7 个模型或模型条件。
它适合使用 pandas、Excel、R 或数据库进行分组统计和可视化。

字段说明：

- `Split`、`Sample`、`Gold`：样本所属 split、样本编号和参考标签。
- `Model`：模型的完整名称，例如 `Qwen/Qwen3-8B`。
- `Condition`：模型运行条件，例如 `thinking`、`no_thinking` 或 `default`。
- `Prediction`：该模型对该样本的预测标签。
- `RawOutputPath`：对应的原始预测 JSONL 文件路径，便于回看模型输出。

### `coverage_report.json`

结果覆盖情况的 JSON 汇总。它记录每个模型、每个条件和每个 split 的：

- 预期样本数；
- 已完成样本数；
- 无效最终标签数量；
- 失败尝试数量；
- 缺失、重复、额外样本和参考标签不一致的样本列表。

顶层 `valid` 表示整个结果集合是否通过检查；当前值为 `true`。

### `run_metadata.json`

本次结果的完整运行记录，包括：

- 数据集来源、固定版本和各 split 的样本数；
- 使用的模型、模型条件和推理后端；
- Ollama Llama 的模型标签和量化方式；
- Python、CUDA 和主要依赖版本；
- 推理参数、运行设备和输出文件清单；
- 结果生成时间以及对应的覆盖报告路径。

它用于复现实验环境、确认结果来源和追踪不同运行之间的差异。

### `reasoning_capabilities.json`

记录各模型是否支持可控的 thinking 开关，以及检测到的原因。
该文件说明为什么 Qwen 有 `thinking` 和 `no_thinking` 两种结果，而其他模型只保留
`default` 条件；它不存放模型预测本身。

### `README.md`

就是当前说明文件，介绍本目录中的结果文件、字段、标签和使用方式。

## 常用使用方式

查看所有模型在某个 split 上的预测：打开对应的 `*_predictions.csv`。

计算某个模型的准确率：将模型列与 `Gold` 列逐行比较。例如统计 Llama 时，比较
`Llama == Gold` 的行数，再除以该 split 的总行数。

按模型条件统计：使用 `predictions_long.csv`，按 `Model`、`Condition` 和 `Split`
分组即可得到准确率、标签分布和模型间一致性。

查看某条样本的完整信息：先用 `Sample` 在宽表或长表中定位，再根据
`RawOutputPath` 到 `predictions_raw/` 中查看该模型的原始响应。
