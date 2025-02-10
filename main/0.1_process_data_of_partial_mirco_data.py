from datetime import datetime
import json
from tqdm import tqdm
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open('../raw_data/partial_micro_data.json', 'r', encoding='utf-8') as original_file:
    original_file_data = json.load(original_file)

    for company_short, temp in tqdm(original_file_data.items()):
        for date_str in list(temp.keys()):
            if '-' in date_str:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            else:
                date_obj = datetime.strptime(date_str, "%Y.%m.%d")
            formatted_date = date_obj.strftime("%Y-%m-%d")

            # 修改键名
            temp[formatted_date] = temp[date_str]
            del temp[date_str]

with open('../raw_data/processed_partial_micro_data.json', 'w', encoding='utf-8') as original_file:
    json.dump(original_file_data, original_file, ensure_ascii=False, indent=4)

    # 统计微观数据的数量
    total_count = 0
    for company_short, temp in tqdm(original_file_data.items(), desc='统计第一版微观数据数量'):
        for date_str, text_list in temp.items():
            for text in text_list:
                total_count += 1
    print(total_count)
