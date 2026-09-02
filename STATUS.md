# Experiment status

Last updated: 2026-09-01 (Asia/Shanghai)

## Runtime

- Server: Ubuntu 20.04.6 LTS, 8 x RTX A6000.
- Environment: Python 3.10.18, PyTorch 2.6.0+cu124, vLLM 0.8.5,
  Transformers 4.51.3.
- vLLM inference: one model per GPU, BF16, `tensor_parallel_size=1`,
  `temperature=0`.
- Ollama Llama inference: Q4_K_M quantization, GPU 5, `temperature=0`.
- The final run completed and all inference sessions exited normally.

## Dataset

- Source: `table-benchmark/infotabs`.
- Revision: `5e26f025fbbf2055f6905cf553ad9c81048891ea`.
- Splits: `train=16538`; `dev=alpha1=alpha2=alpha3=1800`.
- Total samples: 23,738.

## Prediction coverage

The following seven runnable conditions completed every sample in all five splits:

- Qwen3-8B Thinking
- Qwen3-8B NoThinking
- Llama-3.1-8B-Instruct default (Ollama Q4_K_M)
- Gemma-3-12B-IT default
- Mistral-Nemo-Instruct-2407 default
- DeepSeek-R1-Distill-Qwen-7B default
- Phi-4-mini-instruct default

Final validation result:

```text
valid=true
missing=[]
duplicate=[]
extra=[]
invalid_final=0
gold_mismatch=[]
```

Historical invalid attempts remain in `predictions_raw/*/*.invalid.jsonl` for
auditability. Guided decoding retries filled the corresponding missing final
predictions without deleting those records.

## Llama

`meta-llama/Llama-3.1-8B-Instruct` was run through the official Ollama tag
`llama3.1:8b` because the Hugging Face account does not have access to the gated
checkpoint. Ollama 0.33.2 used the Q4_K_M quantized model on GPU 5 with
`OLLAMA_VULKAN=0` and `OLLAMA_CONTEXT_LENGTH=4096`. All five splits completed:
train=16,538 and each of dev/alpha1/alpha2/alpha3=1,800. Llama accuracy is
61.97% on train, 60.83% on dev, 60.28% on alpha1, 58.72% on alpha2, 63.06% on
alpha3, and 61.59% overall. The final tables contain parsed `E/C/N` predictions
rather than `NA`. Samples truncated at the default 256-token output limit were
completed with the documented concise retry mode; all final rows passed the same
coverage and label validation.

## Outputs

- `results/all_predictions.csv`: 23,738 data rows.
- `results/predictions_long.csv`: 166,166 data rows.
- `results/train_predictions.csv`: 16,538 data rows.
- `results/coverage_report.json`: final structured validation report.
- `predictions_raw/llama_ollama_default/`: Llama predictions and retained invalid attempts.
