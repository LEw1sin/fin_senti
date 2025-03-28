import os
import json
from tqdm import tqdm
import re
import pandas as pd

def my_gen():
    value = None
    while True:
        new_value = yield value
        if new_value is not None:
            value = new_value

def clean_text(text):
    if "文章编号" in text:
        text = re.split(r"文章编号", text)[0]
    if "原文连接" in text:
        text = re.split(r"原文连接", text)[0]
    if "免责声明" in text:
        text = re.split(r"免责声明", text)[0]
    if "本内容经慧科的电子服务提供" in text:
        text = re.split(r"本内容经慧科的电子服务提供", text)[0]

    return text.strip()

def process_blank_in_text(text):
    text = re.sub(r'。+', "。", text)      # 替换连续的 \n 为 "。"
    return text

def load_txt_file(file_path):
    data = []
    
    # 打开txt文件并逐行读取
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 去除每行的前后空白字符（例如换行符、空格）
            data.append(line.strip())
    
    return data

def judge(element,lenth=5120):
    element_list = []
    while(len(element.get('body', ''))>lenth):
        new_element = element.copy()
        new_element['body'] = element['body'][:lenth]
        element['body'] = element['body'][lenth:]
        element_list.append(new_element)
    element_list.append(element)
    return element_list



def judge_micro(element,company,lenth=5120):
    element_list = []
    while(len(element.get('body', ''))>lenth):
        new_element = element.copy()
        new_element['body'] = element['body'][:lenth]
        element['body'] = element['body'][lenth:]
        if company in new_element['body']:
            element_list.append(new_element)
    if company in element['body']:
        element_list.append(element)
    return element_list

class JSONGenerator:
    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            try:
                # 加载 JSON 文件的内容
                data = json.load(file)
                for element in tqdm(data,desc=self.file_path):
                    yield element
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from file {self.file_path}: {e}")

def extract_contexts(element, a: str, b: str, context_length: int = 512) -> list:
    """
    从长字符串 b 中截取每次短字符串 a 出现位置的上下文，长度为 context_length。
    
    :param a: 短字符串
    :param b: 长字符串
    :param context_length: 总上下文长度（默认 512）
    :return: 包含上下文的字符串列表
    """
    half_length = context_length // 2
    results = []
    start_pos = 0

    while True:
        # 找到下一个 a 的位置
        new_element = element.copy()
        start_idx = b.find(a, start_pos)
        if start_idx == -1:
            break

        # 确定前文和后文的范围
        end_idx = start_idx + len(a)
        pre_start = max(0, start_idx - half_length)
        post_end = min(len(b), end_idx + half_length)

        # 动态调整分配
        pre_context = b[pre_start:start_idx]
        post_context = b[end_idx:post_end]
        pre_context_len = len(pre_context)
        post_context_len = len(post_context)

        # 如果前文不足，后文补足长度
        if pre_context_len < half_length:
            extra_len = half_length - pre_context_len
            post_end = min(len(b), post_end + extra_len)
            post_context = b[end_idx:post_end]

        # 如果后文不足，前文补足长度
        if post_context_len < half_length:
            extra_len = half_length - post_context_len
            pre_start = max(0, pre_start - extra_len)
            pre_context = b[pre_start:start_idx]

        # 拼接上下文
        context = pre_context + a + post_context
        new_element['body'] = context
        results.append(new_element)

        # 更新起始位置，继续查找下一个 a
        start_pos = end_idx

    return results


def load_json_files(file_path):
    without_micro_data = []
    micro_data = []
    all_bond_company_df = pd.read_excel('../processed_data/all_bond_company_data.xlsx', sheet_name='all')
    company_short_list = []
    for index, row in all_bond_company_df.iterrows():
        company_short_list.append(row['发行人中文简称'])
    company_short_set = set(company_short_list)
    all_data = my_gen()
    next(all_data)

    json_generator = JSONGenerator(file_path)
    try:
        for element in json_generator:

            if len(element.get('body', '')) > 0:
                element['body'] = process_blank_in_text(element.get('body', ''))
                element['body'] = clean_text(element.get('body', ''))
                all_data.send(element)
                flag = 0
                for company in company_short_set:
                    if company in element['body']:
                        flag = 1
                        # if len(element.get('body', '')) > 5120:
                        #     micro_data.extend(judge_micro(element, company, lenth=512))
                        # else:
                        #     micro_data.append(element)
                        micro_data.extend(extract_contexts(element, company, element['body'], context_length=512))
                        break
                if flag == 0:
                    if len(element.get('body', '')) > 5120:
                        without_micro_data.extend(judge(element))
                    else:
                        # 只保留原始元素
                        without_micro_data.append(element)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file: {e}")
    
    cleaned_without_micro_data = [item for item in tqdm(without_micro_data,desc='无微观数据最终筛选') if len(item.get('body', "")) > 100]
    cleaned_micro_data = [item for item in tqdm(micro_data,desc='微观数据最终筛选') if len(item.get('body', "")) > 100]
    return cleaned_micro_data,cleaned_without_micro_data, all_data

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    directory = '../raw_data/all_data.json'
    adding_micro_data, after_deleting_micro_data, all_data= load_json_files(directory)
    with open('../raw_data/adding_micro_data.json', 'w', encoding='utf-8') as all_micro_data_json_file:
        json.dump(adding_micro_data, all_micro_data_json_file, ensure_ascii=False, indent=4)
        print(len(adding_micro_data))
    with open('../raw_data/cleaned_without_micro_data.json', 'w', encoding='utf-8') as after_deleting_micro_data_json_file:
        json.dump(after_deleting_micro_data, after_deleting_micro_data_json_file, ensure_ascii=False, indent=4)
        print(len(after_deleting_micro_data))

    all_bond_company_df = pd.read_excel('../processed_data/all_bond_company_data.xlsx', sheet_name='all')
    company_short_list = []
    for index, row in all_bond_company_df.iterrows():
        company_short_list.append(row['发行人中文简称'])
    company_short_set = set(company_short_list)
    all_data = my_gen()
    next(all_data)
    with open('../raw_data/processed_partial_micro_data.json', 'r', encoding='utf-8') as original_file:
        original_file_data = json.load(original_file)
    for element in tqdm(adding_micro_data):

        for company in company_short_set:
            if company in element['body']:
                processed_date = element['date']
                if company not in original_file_data.keys():
                    original_file_data[company] = {}
                if processed_date not in original_file_data[company].keys():
                    original_file_data[company][processed_date] = []
                original_file_data[company][processed_date].append(element['body'])
    
    with open('../raw_data/all_micro_data.json', 'w', encoding='utf-8') as all_micro_data_json_file:
        json.dump(original_file_data, all_micro_data_json_file, ensure_ascii=False, indent=4)
        sum=0
        for company in tqdm(original_file_data,desc='统计最终微观数据数量'):
            for date in original_file_data[company]:
                original_text_list = original_file_data[company][date]
                for text in original_text_list:
                    sum+=1
        print(sum)