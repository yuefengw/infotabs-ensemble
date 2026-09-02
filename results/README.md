# Results

此目录包含经过 `scripts/validate_predictions.py` 校验后，由
`scripts/build_prediction_table.py` 生成的正式结果。

## 宽表

`all_predictions.csv` 和各 split 的 `*_predictions.csv` 每行对应一个 InfoTabS 样本：

```text
Split, Sample, Gold, Qwen_Thinking, Qwen_NoThinking,
Llama, Gemma, Mistral, DeepSeek, Phi
```

预测标签为 `E`、`C`、`N`。Llama 已通过 Ollama `llama3.1:8b` 完成五个 split，
该列为实际解析结果，不是 `NA` 占位值。

## 长表

`predictions_long.csv` 每行对应一个样本和一个模型条件，列为：

```text
Split, Sample, Gold, Model, Condition, Prediction, RawOutputPath
```

## 审计文件

- `coverage_report.json`：每个模型/条件/split 的完成数和一致性检查。
- `reasoning_capabilities.json`：模型 chat template 的 thinking 开关检测结果。
- `run_metadata.json`：运行环境、包版本、模型位置和输出清单。
