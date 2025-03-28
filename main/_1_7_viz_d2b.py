import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import json
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 读取 JSON 文件（请修改文件路径）
file_path = '../raw_data/cleaned_without_micro_data.json'  # 修改为实际文件路径

# 读取 JSON 数据（假设 JSON 结构是列表，每个元素是 {"body": "文本内容"}）
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 转换为 DataFrame
df = pd.DataFrame(data)

# 确保 "body" 列存在
if "body" not in df.columns:
    raise ValueError("列 'body' 不存在，请检查 JSON 文件")

# 选择 BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

# 计算每行 "body" 的 token 数
df['token_count'] = df['body'].astype(str).apply(lambda x: len(tokenizer.tokenize(x)))

# 统计 token 分布
token_stats = df['token_count'].describe()
print(token_stats)

# 绘制 token 长度分布直方图
plt.figure(figsize=(10, 6))
plt.hist(df['token_count'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel("Token Count")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('d2b.svg')
plt.show()
