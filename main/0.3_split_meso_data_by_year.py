from collections import defaultdict
import json
import os
from tqdm import tqdm
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 读取 JSON 文件
with open('../raw_data/cleaned_without_micro_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 确保数据按日期排序
data_sorted = sorted(data, key=lambda x: x['date'] if 'date' in x else '')

# 将每一年的数据分组
year_groups = defaultdict(list)
for item in data_sorted:
    if 'date' in item:
        year = item['date'].split('-')[0]
        year_groups[year].append(item)

# 将每一年的数据单独保存为新的 JSON 文件
for year, items in year_groups.items():
    file_path = f'../raw_data/data_{year}.json'
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(items)} items for year {year} to {file_path}")
