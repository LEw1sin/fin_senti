import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 读取 Excel 文件（请替换 'your_file.xlsx' 为你的文件路径）
file_path = '../processed_data/train_senti_BERT.xlsx'  
# file_path = '../processed_data/6927条.xlsx' 
df = pd.read_excel(file_path)

# 确保 "内容_Content" 列存在
if "内容_Content" not in df.columns:
    raise ValueError("列 '内容_Content' 不存在，请检查 Excel 文件")

# 选择 GPT-4 兼容的 tokenizer（或其他模型，如 'bert-base-chinese'）
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

# 计算每行的 token 数
df['token_count'] = df['内容_Content'].astype(str).apply(lambda x: len(tokenizer.tokenize(x)))

# 统计 token 分布
token_stats = df['token_count'].describe()
print(token_stats)

# 绘制 token 长度分布直方图
plt.figure(figsize=(10, 6))
plt.hist(df['token_count'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel("Token Count")
plt.ylabel("Frequency")
# plt.title("Distribution of Token Count in '内容_Content'")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('d1.svg')
plt.show()