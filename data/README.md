# Data directory

InfoTabS 数据集及预处理后的 JSONL 不提交到 Git。

预期结构：

```text
data/
|-- infotabs/data/*.parquet
`-- processed/{train,dev,alpha1,alpha2,alpha3}.jsonl
```

下载和转换命令见仓库根目录 `README.md` 的“准备数据”章节。
