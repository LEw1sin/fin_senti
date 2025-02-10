import torch
import torch.nn as nn
import torch.functional as F
from pytorch_transformers import BertPreTrainedModel, BertModel
from transformers import BertTokenizer
import argparse
import torch
import torch.nn.functional as F
from pytorch_transformers import (WEIGHTS_NAME, BertConfig)
import os
import json
import pandas as pd
from datetime import datetime,timedelta
from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy
import re
import numpy as np
from scipy.interpolate import UnivariateSpline
    
class MLP(torch.nn.Module):
    def __init__(self, args, layer_list=[768*3, 768,64], dropout=0.5):
        """
        :param output_n: int 输出神经元个数
        :param layer_list: list(int) 每层隐藏层神经元个数
        :param dropout: float 训练完丢掉多少
        """
        super(MLP, self).__init__()
        self.output_n = args.sentiment_nums
        self.num_layer = len(layer_list)
        self.layer_list = layer_list
        self.hidden_layer = nn.ModuleList(
            [nn.Sequential(nn.Linear(layer_list[i], layer_list[i+1], bias=True),
                          torch.nn.BatchNorm1d(layer_list[i+1]),
                          nn.ReLU(),
                          nn.Dropout(dropout)
            ) for i in range(self.num_layer-1)]
        )

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(layer_list[-1], self.output_n, bias=True),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        for layer in self.hidden_layer:
            x = layer(x)
        x = self.output_layer(x)
        return x

def find_all_full_positions2(lst, sub_lst):
    positions = []
    for i in range(0, len(lst) - len(sub_lst) + 1):
        if lst[i] == sub_lst[0]:
            for j in range(1, len(sub_lst)):
                if lst[i + j] != sub_lst[j]: break
                elif j == len(sub_lst) - 1:
                    positions.append((i, i+j))
    return positions

def process_blank_in_text(text):
    text = re.sub(r'\u0020',"", text)
    text = re.sub(r'\u3000',"", text)
    text = re.sub(r'\xa0',"", text)
    text = re.sub(r'\n',"", text)
    text = re.sub(r'\t',"", text)
    text = re.sub(r'\u2002',"", text)
    text = re.sub(r'\u2003',"", text)
    return text

def validate_date(date_str):
    try:
        datetime.strptime(date_str, "%Y.%m.%d")
        return True
    except ValueError:
        return False

def add_dict_values(dict1, dict2):
    """
    将两个字典的相同键的值相加，返回新的字典。
    假设两个字典的键完全相同。
    
    :param dict1: 第一个字典
    :param dict2: 第二个字典
    :return: 相同键的值相加的新字典
    """
    if dict1.keys() != dict2.keys():
        raise ValueError("两个字典的键必须完全相同！")

    # 遍历键，将值相加
    result = {key: dict1[key] + dict2[key] for key in dict1.keys()}
    return result

def spline_smoothing(values, smoothing_factor=2):
    date_numbers = np.arange(len(values))
    
    # 样条平滑
    spline = UnivariateSpline(date_numbers, values, s=smoothing_factor)
    smoothed_values = spline(date_numbers)
    
    return smoothed_values

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", default="", type=str,
                            help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--model_name_or_path", default='../bert-base-chinese', type=str)
    parser.add_argument("--sentiment_nums", default=3, type=int)
    parser.add_argument("--do_lower_case",default=True)
    parser.add_argument("--device",default='cuda:0')
    args = parser.parse_args()

    #--------宏观中观部分--------
    graph = pd.read_excel('../processed_data/graph.xlsx', index_col=0)

    matrix_dict_raw = graph.to_dict(orient='list')

    matrix_dict = {}
    for k,v in matrix_dict_raw.items():
        matrix_dict[v[0]] = v[1:]

    industry_index = {
        0: "农林牧渔", 1: "基础化工", 2: "钢铁", 3: "有色金属", 4: "电子", 5: "汽车", 
        6: "家用电器", 7: "食品饮料", 8: "纺织服饰", 9: "轻工制造", 10: "医药生物", 
        11: "公用事业", 12: "交通运输", 13: "房地产", 14: "商贸零售", 15: "旅游及景区", 
        16: "教育（含体育）", 17: "本地生活服务", 18: "专业服务", 19: "酒店餐饮", 
        20: "银行", 21: "非银金融", 22: "建筑材料", 23: "建筑装饰", 24: "电力设备", 
        25: "机械设备", 26: "国防军工", 27: "计算机", 28: "电视广播", 29: "游戏", 
        30: "广告营销", 31: "影视院线", 32: "数字媒体", 33: "社交", 34: "出版", 
        35: "通信", 36: "煤炭", 37: "石油石化", 38: "环保", 39: "美容护理"
    }


    fields = [
        "农林牧渔", "基础化工", "钢铁", "有色金属", "电子", "汽车", "家用电器", "食品饮料",
        "纺织服饰", "轻工制造", "医药生物", "公用事业", "交通运输", "房地产", "商贸零售",
        "旅游及景区", "教育（含体育）", "本地生活服务", "专业服务", "酒店餐饮", "银行",
        "非银金融", "建筑材料", "建筑装饰", "电力设备", "机械设备", "国防军工", "计算机",
        "电视广播", "游戏", "广告营销", "影视院线", "数字媒体", "社交", "出版", "通信",
        "煤炭", "石油石化", "环保", "美容护理"
    ]

    policy_dict = defaultdict(lambda: defaultdict(list))
    meso_senti_similarity_path = '../processed_data/all_meso_senti_similarity.json'
    # meso_senti_similarity_path = '../raw_data/data_2020_senti_similarity.json'
    with open(meso_senti_similarity_path, 'r', encoding='utf-8') as file:
        meso_senti_similarity_data = json.load(file)

    for news in tqdm(meso_senti_similarity_data):
        for policy in news['policies']:
            policy_name = policy['policy_name']
            date = news['date']
            if news['sentiment'] != '0' and news['sentiment'] != '-1' and news['sentiment'] != '1':
                continue
            sentiment = policy['distance'] * float(news['sentiment'])
            # sentiment = float(news['sentiment'])
            policy_dict[policy_name][date].append(sentiment)

    start_date = datetime.strptime("2013-01-01", "%Y-%m-%d")
    end_date = datetime.strptime("2023-12-31", "%Y-%m-%d")
    # start_date = datetime.strptime("2020-01-01", "%Y-%m-%d")
    # end_date = datetime.strptime("2020-12-31", "%Y-%m-%d")

    for policy_name in tqdm(policy_dict):
        current_date = start_date
        while current_date <= end_date:
            current_date_str = current_date.strftime("%Y-%m-%d")
            sentiment_list = policy_dict[policy_name][current_date_str]
            if sentiment_list:
                policy_dict[policy_name][current_date_str] = np.sum(sentiment_list)
            else:
                policy_dict[policy_name][current_date_str] = 0
            current_date += timedelta(days=1) 

    industry_dict = defaultdict(lambda: defaultdict(list))

    for policy_name in tqdm(policy_dict):
        for date in policy_dict[policy_name]:
            for i, policy2industry in enumerate(matrix_dict[policy_name]):
                if policy2industry == 1:
                    industry = industry_index[i]
                    industry_dict[industry][date].append(policy_dict[policy_name][date])

    for industry in tqdm(industry_dict):
        current_date = start_date
        while current_date <= end_date:
            current_date_str = current_date.strftime("%Y-%m-%d")
            sentiment_list = industry_dict[industry][current_date_str]
            if sentiment_list:
                industry_dict[industry][current_date_str] = np.sum(sentiment_list)
            else:
                industry_dict[industry][current_date_str] = 0
            current_date += timedelta(days=1)  
    
    #归一化
    # for industry in industry_dict:
    #     industry_senti_list = industry_dict[industry].values()
    #     industry_max_senti = max(industry_senti_list)
    #     industry_min_senti = min(industry_senti_list)
    #     current_date = start_date
    #     normalized_values = {}
        
    #     # Normalize to [0, 1]
    #     while current_date <= end_date:
    #         current_date_str = current_date.strftime("%Y-%m-%d")
    #         normalized_value = (industry_dict[industry][current_date_str] - industry_min_senti) / \
    #                         (industry_max_senti - industry_min_senti)
    #         normalized_values[current_date_str] = normalized_value
    #         current_date += timedelta(days=1)
        
    #     # Adjust to [-1, 1] and ensure min = -1, max = 1
    #     actual_min = min(normalized_values.values())
    #     actual_max = max(normalized_values.values())
        
    #     scale_factor = 2 / (actual_max - actual_min)
    #     shift_factor = -1 - scale_factor * actual_min
        
    #     for date_str in normalized_values:
    #         industry_dict[industry][date_str] = scale_factor * normalized_values[date_str] + shift_factor

    date_list = list(industry_dict[industry].keys())

    for industry in industry_dict:
        industry_senti_list_order = sorted(industry_dict[industry].items(), key=lambda x: datetime.strptime(x[0], '%Y-%m-%d'))
        industry_senti_list_order_dict = {date: sentiment for date, sentiment in deepcopy(industry_senti_list_order)}
        industry_senti_list = np.array(list(industry_senti_list_order_dict.values()))
        mean = np.mean(industry_senti_list)
        std = np.std(industry_senti_list)
        industry_dict[industry] = (industry_senti_list - mean) / std

    company2industry_path = '../processed_data/company2industry.json'
    company2industry_dict = defaultdict(lambda: list())

    with open(company2industry_path, 'r', encoding='utf-8') as file:
        company2industry_data = json.load(file)

    for company_data in company2industry_data['results']:
        company_short = company_data['company']
        similarity_list = company_data['similarity']
        max_distance = float('-inf')  
        max_index = -1  
        index_list = []
        for i, item in enumerate(similarity_list):
            if item["distance"] > max_distance:
                max_distance = item["distance"]
                max_index = i
            if item["distance"] >= 0.7:
                index_list.append(i)
        if max_distance < 0.7:
            company2industry_dict[company_short].append(similarity_list[max_index]['industry'])
        else:
            for index in index_list:
                company2industry_dict[company_short].append(similarity_list[index]['industry'])

    company_meso_senti_dict = defaultdict(lambda: 0)
    for company_short, industry_list in tqdm(company2industry_dict.items()):
        company_meso_senti_list = []
        for industry in industry_list:
            company_meso_senti_list.append(industry_dict[industry])
        company_meso_senti_dict[company_short] = np.mean(np.array(company_meso_senti_list), axis=0)

    #--------微观部分--------

    company_micro_senti_dict_path = '../processed_data/company_micro_sentiment.json'
    with open(company_micro_senti_dict_path, 'r', encoding='utf-8') as file:
        company_micro_senti_dict = json.load(file)
    
    company_final_senti_dict = dict()  

    for company_short, company_date_micro_senti in tqdm(company_micro_senti_dict.items()):
        company_micro_senti = np.array([np.float64(sentiment) for _, sentiment in company_date_micro_senti.items()])
        company_micro_senti = company_micro_senti + company_meso_senti_dict[company_short]
        company_final_senti = spline_smoothing(company_micro_senti, smoothing_factor=10)
        company_final_senti_date = {date:senti for date, senti in zip(date_list, company_final_senti)}
        company_final_senti_dict[company_short] = company_final_senti_date

    with open('../processed_data/company_sentiment.json', 'w', encoding='utf-8') as file:
        json.dump(company_final_senti_dict, file, ensure_ascii=False)


    final_dataframe = pd.read_csv("../processed_data/bond_data_prenormalized.csv")

    # 复制原始 DataFrame
    standardized_final_dataframe = final_dataframe.copy()

    # 选择需要标准化的列
    columns_to_standardize = standardized_final_dataframe.columns[1:51]

    # 计算标准化 (z 分数)
    standardized_final_dataframe[columns_to_standardize] = (
        standardized_final_dataframe[columns_to_standardize] - standardized_final_dataframe[columns_to_standardize].mean()
    ) / standardized_final_dataframe[columns_to_standardize].std()

    # 查看标准化后的 DataFrame
    standardized_final_dataframe.head()

    standardized_final_dataframe["日期"] = pd.to_datetime(standardized_final_dataframe["日期"])
    filtered_df = standardized_final_dataframe[(standardized_final_dataframe["日期"] >= start_date) & (standardized_final_dataframe["日期"] <= end_date)]
    print(len(filtered_df))
    filtered_df.head()

    all_bond_company_df = pd.read_excel('../processed_data/all_bond_company_data.xlsx', sheet_name='all')
    company_short_list = []
    for index, row in all_bond_company_df.iterrows():
        company_short_list.append(row['发行人中文简称'])
    company_short_set = set(company_short_list)

    filtered_df = filtered_df[filtered_df['发行人中文简称'].isin(company_short_set)]
    filtered_df.count()

    def compute_sentiment(row, company_final_senti_dict=company_final_senti_dict):
        date = row["日期"]
        company_short = row["发行人中文简称"]
        idx_date = date - pd.Timedelta(days=1)
        idx_date = idx_date.strftime("%Y-%m-%d")
        return company_final_senti_dict[company_short][idx_date]

    filtered_df['sentiment'] = filtered_df.apply(compute_sentiment, axis=1)
    filtered_df.to_csv("../processed_data/bond_data_normalized_w_senti.csv", index=False)

    # config = BertConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, 
    #     num_labels=args.sentiment_nums)

    # BERT = BertModel.from_pretrained(args.model_name_or_path, config=config)
    # MLP = MLP(args=args)

    # bert_checkpoint = torch.load('./0.0_train_BERT/BERT_best_MSE.pth', map_location=torch.device('cpu'))
    # BERT.load_state_dict(bert_checkpoint)

    # mlp_checkpoint = torch.load('./0.0_train_BERT/MLP_best_MSE.pth', map_location=torch.device('cpu'))
    # MLP.load_state_dict(mlp_checkpoint)

    # BERT.to(device=args.device)
    # MLP.to(device=args.device)

    # BERT.eval()
    # MLP.eval()

    # tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, 
    #                                             do_lower_case=args.do_lower_case)

    # all_bond_company_df = pd.read_excel('../processed_data/all_bond_company_data.xlsx', sheet_name='all')
    # company_short_list = []
    # for index, row in all_bond_company_df.iterrows():
    #     company_short_list.append(row['发行人中文简称'])
    # company_short_set = set(company_short_list)

    # all_micro_data_path = '../raw_data/all_micro_data.json'

    # with open(all_micro_data_path, 'r', encoding='utf-8') as file:
    #     all_micro_data = json.load(file)
    
    # company_iter_dict = defaultdict(lambda: 0)
    # company_final_senti_dict = dict()
    # bond_dict = dict()

    # for company_short in tqdm(company_short_set):
    #     date_text_list = all_micro_data.get(company_short, None)
    #     if date_text_list:
    #         company_iter_dict.clear()
    #         for date, text_list in date_text_list.items():
    #             date = datetime.strptime(date, "%Y-%m-%d")
    #             if date >= start_date and date <= end_date:
    #                 day_tensor = torch.tensor([[0.0, 0.0, 0.0]])
    #                 for text in text_list:
    #                     text = process_blank_in_text(text)
    #                     text = text[:512]
    #                     text_num = 0
                        
    #                     company_tokenized = tokenizer.encode_plus(
    #                         company_short,                      # Sentence to encode.
    #                         add_special_tokens=True,     # Add '[CLS]' and '[SEP]'
    #                         max_length=20,               # Pad & truncate all sentences.
    #                         padding=True,
    #                         return_attention_mask=True,   # Construct attn. masks.
    #                         return_tensors='pt',         # Return pytorch tensors.
    #                         truncation=True
    #                     )
    #                     company_tokenized_ids = company_tokenized['input_ids']
    #                     company_tokenized_masks = company_tokenized['attention_mask']
    #                     company_tokenized_ids = company_tokenized_ids.to(device=args.device)
    #                     company_tokenized_masks = company_tokenized_masks.to(device=args.device)
    #                     text_tokenized = tokenizer.encode_plus(
    #                         text,                      # Sentence to encode.
    #                         add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
    #                         max_length=512,           # Pad & truncate all sentences.
    #                         padding=True,
    #                         return_attention_mask=True,   # Construct attn. masks.
    #                         return_tensors='pt',         # Return pytorch tensors.
    #                         truncation=True
    #                     )
    #                     text_tokenized_ids = text_tokenized['input_ids']
    #                     text_tokenized_masks = text_tokenized['attention_mask']
    #                     text_tokenized_ids = text_tokenized_ids.to(device=args.device)
    #                     text_tokenized_masks = text_tokenized_masks.to(device=args.device)

    #                     # 统计mask中1的个数
    #                     cnt = company_tokenized_masks.sum().item()
    #                     positions = find_all_full_positions2(
    #                         text_tokenized_ids[0].tolist(), company_tokenized_ids[0].tolist()[1:cnt-1])
    #                     if positions:
    #                         with torch.no_grad():
    #                             output = BERT(text_tokenized_ids, text_tokenized_masks)
    #                             tensor = torch.cat(
    #                                 [output[0][0][i: j+1] for (i, j) in positions], dim=0)  # [n * l, 768] n: 实体匹配cnt; l: 实体名称长度
    #                             mean_val = torch.mean(
    #                                 tensor, dim=0)  # [768]
    #                             max_val = torch.max(
    #                                 tensor, dim=0).values  # [768]
    #                             input_tensor = torch.cat(
    #                                 [mean_val, max_val, output[1][0]], dim=0)  # [768 * 3]
    #                             input_tensor = input_tensor.unsqueeze(
    #                                 0).to(device=args.device)  # [1, 2304]
    #                             sentiment_predict = MLP(input_tensor)
    #                             sentiment_predict = sentiment_predict.to(device='cpu')
    #                             text_num += 1
    #                             day_tensor += sentiment_predict

    #                 day_tensor /= text_num
    #                 if not torch.isnan(day_tensor).any():
    #                     day_tensor = day_tensor.tolist()
    #                     max_index = day_tensor[0].index(max(day_tensor[0]))
    #                     if max_index == 0:
    #                         sentiment = -1
    #                     elif max_index == 1:
    #                         sentiment = 0
    #                     elif max_index == 2:
    #                         sentiment = 1
    #                     date = date.strftime("%Y-%m-%d")
    #                     company_iter_dict[date] = sentiment
                
    #     current_date = start_date
    #     while current_date <= end_date:
    #         current_date_str = current_date.strftime("%Y-%m-%d")
    #         _ = company_iter_dict[current_date_str]
    #         current_date += timedelta(days=1)

    #     # 使用深拷贝创建新的字典
    #     company_list = sorted(company_iter_dict.items(), key=lambda x: datetime.strptime(x[0], '%Y-%m-%d'))
    #     company_micro_senti = np.array([np.float64(sentiment) for _, sentiment in deepcopy(company_list)])
    #     company_micro_senti = company_micro_senti + company_meso_senti_dict[company_short]
    #     company_final_senti = spline_smoothing(company_micro_senti)
    #     company_final_senti_date = {date:senti for date, senti in zip(date_list, company_final_senti)}
    #     company_final_senti_dict[company_short] = company_final_senti_date

    # with open('../processed_data/company_sentiment.json', 'w', encoding='utf-8') as file:
    #     json.dump(company_final_senti_dict, file, ensure_ascii=False)