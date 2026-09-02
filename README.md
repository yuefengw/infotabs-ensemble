# InfoTabS Multi-LLM Predictions

这是一个面向 [InfoTabS](https://huggingface.co/datasets/table-benchmark/infotabs)
表格自然语言推理任务的多模型、离线优先预测流水线。仓库包含推理脚本、环境配置、
原始模型输出和汇总结果；模型权重、数据集、离线 wheel 包和运行日志不会提交到 Git。

当前结果覆盖 InfoTabS 的 `train`、`dev`、`alpha1`、`alpha2`、`alpha3` 五个 split。
可用模型条件全部通过覆盖校验：每个条件、每个 split 都没有缺失、重复、额外样本、
最终无效标签或 gold 不一致。

## 已运行模型

| 表中列名 | 模型 | 条件 | 后端 |
|---|---|---|---|
| `Qwen_Thinking` | `Qwen/Qwen3-8B` | thinking | vLLM |
| `Qwen_NoThinking` | `Qwen/Qwen3-8B` | no thinking | vLLM |
| `Llama` | `meta-llama/Llama-3.1-8B-Instruct` | default | Ollama `llama3.1:8b`（Q4_K_M），已完成五个 split |
| `Gemma` | `google/gemma-3-12b-it` | default | vLLM |
| `Mistral` | `mistralai/Mistral-Nemo-Instruct-2407` | default | vLLM |
| `DeepSeek` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | default | vLLM |
| `Phi` | `microsoft/Phi-4-mini-instruct` | default | vLLM |

只有 Qwen3 的官方 chat template 在本实验环境中确认支持可控的
`enable_thinking=True/False`，因此只有 Qwen 同时运行 Thinking 和 NoThinking。
Llama 的 Hugging Face 权重需要额外授权，因此改用官方 Ollama `llama3.1:8b`
标签。该模型已完成五个 split 的正式批量预测，结果已写入宽表、长表和
`predictions_raw/llama_ollama_default/`。

## 最终结果

- 样本总数：23,738（`train=16,538`，其余四个 split 各 1,800）。
- 宽表：23,738 条数据行。
- 长表：166,166 条数据行，即每个样本 7 个模型/条件位置，包含 Llama 预测。
- 校验状态：`results/coverage_report.json` 中 `valid=true`。

Llama 准确率（由 `results/all_predictions.csv` 计算）：

| Split | Correct / Total | Accuracy |
|---|---:|---:|
| `train` | 10,249 / 16,538 | 61.97% |
| `dev` | 1,095 / 1,800 | 60.83% |
| `alpha1` | 1,085 / 1,800 | 60.28% |
| `alpha2` | 1,057 / 1,800 | 58.72% |
| `alpha3` | 1,135 / 1,800 | 63.06% |
| 全部 | 14,621 / 23,738 | 61.59% |

关键文件：

| 文件 | 用途 |
|---|---|
| `results/all_predictions.csv` | 五个 split 的统一宽表 |
| `results/train_predictions.csv` | 训练 split 的宽表 |
| `results/{dev,alpha1,alpha2,alpha3}_predictions.csv` | 各评估 split 的宽表 |
| `results/predictions_long.csv` | 统一长表，适合统计和下游分析 |
| `results/coverage_report.json` | 每个模型/条件/split 的覆盖校验报告 |
| `results/run_metadata.json` | 环境、包版本、模型和运行元数据 |
| `predictions_raw/` | 模型逐样本原始输出、解析标签和无效尝试审计记录 |

## 目录结构

```text
.
|-- data/                    # 本地数据目录；数据内容被 Git 忽略
|-- env/                     # 在线/离线环境安装和 Ollama 辅助脚本
|-- logs/                    # 运行日志；被 Git 忽略
|-- models/                  # 官方模型快照；被 Git 忽略
|-- predictions_raw/         # 逐模型 JSONL 预测和 metadata
|-- results/                 # 最终宽表、长表、校验报告和元数据
|-- scripts/                 # 下载、预处理、推理、校验和制表脚本
|-- MODEL_MANIFEST.tsv       # 模型 ID、预期本地路径和状态
|-- STATUS.md                # 本次运行和部署状态摘要
|-- requirements-top-level.txt
|-- requirements-lock.txt
`-- README.md
```

### 主要脚本

| 脚本 | 用途 |
|---|---|
| `scripts/download_model.py` | 从 Hugging Face 下载单个官方模型快照 |
| `scripts/verify_model.py` | 检查本地模型配置、tokenizer 和 safetensors 完整性 |
| `scripts/check_env.py` | 输出 Python、CUDA、PyTorch、vLLM 和 GPU 状态 |
| `scripts/preprocess_infotabs.py` | 将官方 parquet 转换成稳定的 JSONL 输入 |
| `scripts/inspect_reasoning.py` | 检测模型 chat template 是否支持 thinking 开关 |
| `scripts/test_model.py` | 对本地 Hugging Face 模型做单 GPU smoke test |
| `scripts/run_model.py` | 使用 vLLM 推理，支持断点续跑和 guided 标签重试 |
| `scripts/run_ollama.py` | 使用本机 Ollama API 运行 Llama |
| `scripts/validate_predictions.py` | 校验缺失、重复、非法标签和 gold 一致性 |
| `scripts/build_prediction_table.py` | 校验通过后生成宽表、长表和元数据 |
| `scripts/prepare_pending_shards.py` | 将未成功样本拆成互不重叠的并行分片 |
| `scripts/merge_prediction_shards.py` | 按 `sample_id` 合并分片结果 |

## 环境要求

已验证环境：

```text
Ubuntu 20.04
Python 3.10.18
CUDA 12.4 runtime
PyTorch 2.6.0+cu124
vLLM 0.8.5
Transformers 4.51.3
NVIDIA RTX A6000
```

vLLM 模型使用单模型单 GPU、BF16、`tensor_parallel_size=1`、`temperature=0`；
Llama 使用 Ollama Q4_K_M 量化权重、单 GPU 和 `temperature=0`。

### 在线安装

建议新建独立环境，然后安装固定的顶层依赖：

```bash
conda create -n infotabs python=3.10 -y
conda activate infotabs
python -m pip install -r requirements-top-level.txt
python -m pip check
python scripts/check_env.py
```

`requirements-lock.txt` 是原服务器完整环境快照，适合审计；在新机器上优先使用
`requirements-top-level.txt`，避免把与机器相关的间接依赖强行锁死。

### 离线安装

在可联网的 Windows 机器上为 Linux CPython 3.10 准备 wheel：

```powershell
.\env\prepare_linux_wheels.ps1
```

将 `env/wheels/` 上传到 Linux 服务器后运行：

```bash
bash env/install_offline.sh
```

wheel 文件体积很大，已被 `.gitignore` 排除，不会随仓库分发。

## 准备数据

数据集不会提交到本仓库。下载固定 revision：

```bash
huggingface-cli download table-benchmark/infotabs \
  --repo-type dataset \
  --revision 5e26f025fbbf2055f6905cf553ad9c81048891ea \
  --local-dir data/infotabs
```

转换五个 split：

```bash
python scripts/preprocess_infotabs.py
```

输出位置为 `data/processed/{train,dev,alpha1,alpha2,alpha3}.jsonl`，同时生成
`data/processed/dataset_manifest.json`。每条输入必须包含：

```json
{
  "sample_id": "dev_000000",
  "split": "dev",
  "table": "...",
  "hypothesis": "...",
  "gold": "E"
}
```

标签为 `E`（Entailment）、`C`（Contradiction）、`N`（Neutral）。

## 准备模型

下载前可先查看预计大小：

```bash
python scripts/download_model.py qwen3-8b --dry-run
```

下载和验证：

```bash
python scripts/download_model.py qwen3-8b
python scripts/verify_model.py --model models/qwen3-8b --hash
```

支持的本地目录列在 `MODEL_MANIFEST.tsv`。模型目录必须包含完整的
`config.json`、tokenizer 文件和 safetensors 权重。整个 `models/` 内容被 Git 忽略。

## 运行推理

先做单模型测试：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/test_model.py \
  --model models/qwen3-8b
```

运行一个模型条件和一个 split：

```bash
# Qwen Thinking
CUDA_VISIBLE_DEVICES=0 python scripts/run_model.py \
  --model qwen \
  --condition thinking \
  --split dev \
  --batch-size 16

# Qwen NoThinking
CUDA_VISIBLE_DEVICES=1 python scripts/run_model.py \
  --model qwen \
  --condition no_thinking \
  --split dev \
  --batch-size 16

# 其他模型使用 default
CUDA_VISIBLE_DEVICES=2 python scripts/run_model.py \
  --model gemma \
  --condition default \
  --split dev \
  --batch-size 16
```

`--model` 可使用 `qwen`、`gemma`、`mistral`、`deepseek`、`phi`。
脚本会读取已有的有效 `sample_id` 并只处理缺失样本，因此同一命令可安全断点续跑。
解析失败的原始响应会保存在同目录的 `*.invalid.jsonl` 中，不应删除。

如果常规输出未形成可解析标签，可对缺失项执行约束重试：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_model.py \
  --model qwen \
  --condition thinking \
  --split dev \
  --batch-size 16 \
  --max-tokens 32 \
  --guided-labels
```

## 校验并生成结果表

仅当所有计划中的模型输出完成后运行：

```bash
python scripts/validate_predictions.py
python scripts/build_prediction_table.py
```

第一条命令写入 `results/coverage_report.json`，任何缺失、重复、额外样本、无效最终
标签或 gold 不一致都会返回非零退出码。第二条命令要求覆盖报告为有效状态，然后写入
`results/*.csv` 和 `results/run_metadata.json`。

## 原始预测格式

`predictions_raw/<model>_<condition>/<split>.jsonl` 的每行包含：

```json
{
  "sample_id": "dev_000000",
  "split": "dev",
  "gold": "E",
  "model": "Qwen/Qwen3-8B",
  "model_key": "qwen",
  "condition": "thinking",
  "raw_output": "...",
  "parsed_prediction": "E"
}
```

## Ollama Llama

Llama 3.1 的 Hugging Face 仓库受 gated license 保护。本项目提供 Ollama 路径，
服务器已完成 Q4_K_M 量化模型的安装和离线导入。服务固定使用 GPU 5，并关闭
Vulkan 后端：

```bash
CUDA_VISIBLE_DEVICES=5 bash env/start_ollama_user.sh
python scripts/test_ollama.py
python scripts/run_ollama.py --model llama
```

不传 `--split` 时会运行全部五个 split。若模型的长解释在默认 `256` token 上限前
没有输出最终标签，可使用 `--concise --max-tokens 32`，要求 Ollama 只返回一个标签。
本次批量结果对这类未完成样本使用了该续跑方式；已有有效预测保持不变，所有最终标签
仍经过同一解析器和覆盖校验。

## 复现注意事项

- 模型权重和 InfoTabS 数据集受各自许可证约束，请从官方来源获取。
- 不要混用多张可见 GPU；每个 `run_model.py` 进程应只设置一个 GPU。
- 不要在推理进程仍写文件时运行最终校验或制表。
- 不要删除 `*.invalid.jsonl`；它们用于审计首次解析失败和后续重试。
- `Llama` 已完成批量预测，其结果列为解析后的 `E/C/N` 标签。
- 本仓库尚未声明开源许可证；代码使用权限以仓库所有者后续添加的许可证为准。
