# Model directory

模型权重不提交到 Git。请将官方 Hugging Face 快照放入以下目录：

```text
models/qwen3-8b
models/gemma-3-12b-it
models/mistral-nemo-instruct-2407
models/deepseek-r1-distill-qwen-7b
models/phi-4-mini-instruct
```

每个目录应包含 `config.json`、官方 tokenizer 文件、safetensors 权重及模型仓库要求的
自定义代码。使用 `scripts/download_model.py` 下载，使用 `scripts/verify_model.py` 验证。
