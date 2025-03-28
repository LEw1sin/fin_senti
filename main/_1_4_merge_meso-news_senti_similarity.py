import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

meso_senti_path = os.path.abspath('../raw_data/merged_data_except_2020_senti.json')
meso_similarity_path = os.path.abspath('../raw_data/merged_data_except_2020_similarity.json')

with open(meso_similarity_path, 'r', encoding='utf-8') as file:
    meso_similarity_data = json.load(file)
with open(meso_senti_path, 'r', encoding='utf-8') as file:
    meso_senti_data  = json.load(file)

distance_values = []

for news_iter in tqdm(meso_similarity_data):
    policies_list = news_iter['policies']
    for policy in policies_list:
        distance_values.append(policy['distance'])

# 确保所有值在 0 到 1 范围内
assert all(0 <= d <= 1 for d in distance_values), "Distance values are not in the range 0 to 1."

# 绘制分布直方图
bins = np.linspace(0, 1, 11)  # 将 0-1 分为 10 个区间（0.1 间隔）
plt.hist(distance_values, bins=bins, edgecolor='k', alpha=0.7)
plt.title("Policy Distance Distribution (0-1)")
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.grid(True)
plt.xticks(bins)
plt.show()

# 输出基本统计信息
print(f"Total items: {len(distance_values)}")
print(f"Mean distance: {np.mean(distance_values):.2f}")
print(f"Median distance: {np.median(distance_values):.2f}")
print(f"Min distance: {np.min(distance_values):.2f}")
print(f"Max distance: {np.max(distance_values):.2f}")

def filter_policy(news_iter):
    output = []
    for policy_help in news_iter:
        if policy_help['distance'] > 0.6:
            output.append(policy_help)
    return output[:3]

for news_in_senti in tqdm(meso_senti_data):
    for news_in_similarity in meso_similarity_data:
        if news_in_similarity['body'] == news_in_senti['body']:
            news_in_senti['policies'] = filter_policy(news_in_similarity['policies'])
            break

print(len(meso_senti_data))
with open(os.path.abspath('../raw_data/merged_data_except_2020_senti_similarity.json'), 'w', encoding='utf-8') as file:
    json.dump(meso_senti_data, file, ensure_ascii=False, indent=4)

# with open('../raw_data/merged_data_except_2020_senti_similarity.json', 'r', encoding='utf-8') as file:
#     meso_senti_data = json.load(file)
# print(len(meso_senti_data))

data_2020_senti_similarity_path = '../raw_data/data_2020_senti_similarity.json'
with open(data_2020_senti_similarity_path, 'r', encoding='utf-8') as file:
    data_2020_senti_similarity = json.load(file)
print(len(data_2020_senti_similarity))

all_meso_senti_similarity =[]
all_meso_senti_similarity.extend(meso_senti_data)
all_meso_senti_similarity.extend(data_2020_senti_similarity)
print(len(all_meso_senti_similarity))
with open('../processed_data/all_meso_senti_similarity.json', 'w', encoding='utf-8') as file:
    json.dump(all_meso_senti_similarity, file, ensure_ascii=False, indent=4)



