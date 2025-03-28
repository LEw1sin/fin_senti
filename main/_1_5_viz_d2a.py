import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import json
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

file_path = '../raw_data/all_micro_data.json'  # 修改为实际文件路径

# 读取 JSON 数据（假设 JSON 结构是列表，每个元素是 {"body": "文本内容"}）
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

from tqdm import tqdm
all_data_list = []

for company_short, text_list_date in tqdm(data.items()):
    for text_date, text_list in text_list_date.items():
        all_data_list = [*all_data_list, *text_list]

all_data_dict = [{'index': i, 'body': body} for i, body in enumerate(all_data_list)]

df = pd.DataFrame(all_data_dict)

print(df.head())

df.to_csv('../raw_data/all_micro_data.csv', index=False)

# df = pd.read_csv('../raw_data/all_micro_data.csv')
# 选择 BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

# 计算每行 "body" 的 token 数
df['token_count'] = df['body'].astype(str).apply(lambda x: len(tokenizer.tokenize(x)))

# 统计 token 分布
token_stats = df['token_count'].describe()
print(token_stats)

df_filtered = df[df['token_count'] <= 512]

# 绘制 token 长度分布直方图
plt.figure(figsize=(10, 6))
plt.hist(df_filtered['token_count'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel("Token Count")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('all_micro_data_token_count.svg')
plt.show()