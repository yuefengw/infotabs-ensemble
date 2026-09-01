# Experiment status

Last updated: 2026-09-01 (Asia/Shanghai)

## Runtime

- Server: Ubuntu 20.04.6 LTS, 8 x RTX A6000.
- Environment: Python 3.10.18, PyTorch 2.6.0+cu124, vLLM 0.8.5,
  Transformers 4.51.3.
- Inference: one model per GPU, BF16, `tensor_parallel_size=1`,
  `temperature=0`.
- The final run completed and all inference sessions exited normally.

## Dataset

- Source: `table-benchmark/infotabs`.
- Revision: `5e26f025fbbf2055f6905cf553ad9c81048891ea`.
- Splits: `train=16538`; `dev=alpha1=alpha2=alpha3=1800`.
- Total samples: 23,738.

## Prediction coverage

The following six runnable conditions completed every sample in all five splits:

- Qwen3-8B Thinking
- Qwen3-8B NoThinking
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

`meta-llama/Llama-3.1-8B-Instruct` was not run because the Hugging Face account
did not have access to the gated checkpoint. The final tables preserve a
`Llama=NA` column so the schema remains stable.

## Outputs

- `results/all_predictions.csv`: 23,738 data rows.
- `results/predictions_long.csv`: 166,166 data rows.
- `results/train_predictions.csv`: 16,538 data rows.
- `results/coverage_report.json`: final structured validation report.
