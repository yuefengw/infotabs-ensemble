# Logs directory

环境安装、模型验证、smoke test、正式推理和 guided 重试日志写入此目录。
日志可能很大且包含机器路径，因此默认被 Git 忽略。正式的结构化运行信息位于
`results/run_metadata.json` 和 `predictions_raw/*/*.metadata.json`。
