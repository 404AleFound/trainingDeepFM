# Process the raw data
如下图所示，criteo 的原始数据输出如下：
![alt text](./assets/the_output_of_raw_data.png)

数据包含一个目标标签（lable）：
- 13 个稠密特征，也即数值特征，分别命名为 `I1` 到 `I13`；
- 26 个稀疏特征，也即类别特征，分别命名为 `C1` 到 `C26`。

## 问题2
数据处理


## 问题1

### 问题描述
数据集太大，无法通过readCSV的方式全部读取

### 解决方法
自动检查：检测是否已经存在一个同名的 .parquet 文件（比如你的输入是 train.txt，它会自动去找 train.parquet）。
极速读取：如果有这个 .parquet，它就不会再去跑耗时的 Pandas 逐行字符串解析和 LabelEncoder 了，而是直接 pd.read_parquet(parquet_path) 几秒钟读入内存。
静默生成：如果你是第一次跑（或者把原始 TXT 文件换了路径没有成套的 parquet），它会先正常读 TXT、做清洗和归一化、做 Label Encode，并且在处理完后自动调用 data.to_parquet(...) 保存在旁边。
### 
原因 

### 解决方式分块读取