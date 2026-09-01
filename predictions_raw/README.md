# Raw predictions

每个目录对应一个模型条件，每个 split 包含三类文件：

- `<split>.jsonl`：最终有效预测，每个 `sample_id` 恰好一行。
- `<split>.invalid.jsonl`：历史无效输出和重试审计记录。
- `<split>.metadata.json`：该次运行的模型、条件、GPU、生成参数和时间信息。

目录名称：

```text
qwen_thinking/
qwen_no_thinking/
gemma_default/
mistral_default/
deepseek_default/
phi_default/
```

`validate_predictions.py` 只接受 `E`、`C`、`N` 三种最终标签，并检查缺失、重复、
额外样本及 gold 一致性。
