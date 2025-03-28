import torch
import torch.nn as nn
import torch.functional as F
import argparse
import torch
import torch.nn.functional as F
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
import pywt

wavelet = 'db4'  # 选择小波基函数
level = 6  # 分解的级数


def smoothing(values, smoothing_factor=2):
    # date_numbers = np.arange(len(values))
    
    # # 样条平滑
    # spline = UnivariateSpline(date_numbers, values, s=smoothing_factor)
    # smoothed_values = spline(date_numbers)

    coeffs = pywt.wavedec(values, wavelet, level=level)
    coeffs_smoothed = [coeffs[0]] + [np.zeros_like(coeffs[i]) for i in range(1, len(coeffs))]
    wavelet_smoothed = pywt.waverec(coeffs_smoothed, wavelet)
    smoothed_values = wavelet_smoothed[:len(values)]
    return smoothed_values

def custom_agg(group):
    result = {}
    for col in group.columns:
        if pd.api.types.is_numeric_dtype(group[col]):
            result[col] = group[col].mean()  # 数值列取均值
        else:
            result[col] = group[col].iloc[0]  # 非数值列取第一行的值
    return pd.Series(result)

if __name__ == "__main__":
    smoothing_factor= 16

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
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

    for news in tqdm(meso_senti_similarity_data, desc='Calculating policy sentiment'):
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

    for policy_name in tqdm(policy_dict, desc='Summing policy sentiment'):
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

    for policy_name in tqdm(policy_dict, desc='Mapping policy sentiment to industry'):
        for date in policy_dict[policy_name]:
            for i, policy2industry in enumerate(matrix_dict[policy_name]):
                if policy2industry == 1:
                    industry = industry_index[i]
                    industry_dict[industry][date].append(policy_dict[policy_name][date])

    for industry in tqdm(industry_dict, desc='Summing industry sentiment'):
        current_date = start_date
        while current_date <= end_date:
            current_date_str = current_date.strftime("%Y-%m-%d")
            sentiment_list = industry_dict[industry][current_date_str]
            if sentiment_list:
                industry_dict[industry][current_date_str] = np.sum(sentiment_list)
            else:
                industry_dict[industry][current_date_str] = 0
            current_date += timedelta(days=1)  
    

    date_list = list(industry_dict[industry].keys())
    all_meso_values = []

    for industry in industry_dict:
        industry_senti_list_order = sorted(industry_dict[industry].items(), key=lambda x: datetime.strptime(x[0], '%Y-%m-%d'))
        industry_senti_list_order_dict = {date: sentiment for date, sentiment in deepcopy(industry_senti_list_order)}
        industry_senti_list = np.array(list(industry_senti_list_order_dict.values()))
        all_meso_values.append(industry_senti_list)
        # mean = np.mean(industry_senti_list)
        # std = np.std(industry_senti_list)
        industry_dict[industry] = industry_senti_list


    all_meso_values = np.array(all_meso_values)
    all_meso_values_mean = all_meso_values.mean()
    all_meso_values_std = all_meso_values.std()

    for industry in industry_dict:
        industry_dict[industry] = (industry_dict[industry] - all_meso_values_mean) / all_meso_values_std

    my_dict = dict(industry_dict)
    np.save('../processed_data/industry_dict_sum_pre-spline3.npy', my_dict)

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
    for company_short, industry_list in tqdm(company2industry_dict.items(), desc='Mapping industry sentiment to company'):
        company_meso_senti_list = []
        for industry in industry_list:
            company_meso_senti_list.append(industry_dict[industry])
        company_meso_senti_dict[company_short] = np.mean(np.array(company_meso_senti_list), axis=0) 



    #--------微观部分--------

    company_micro_senti_dict_path = '../processed_data/company_micro_sentiment_mean.json'
    with open(company_micro_senti_dict_path, 'r', encoding='utf-8') as file:
        company_micro_senti_dict = json.load(file)
    
    company_final_senti_dict = dict()  
    company_senti_only_micro_dict = dict()
    company_senti_only_meso_dict = dict()
    company_senti_wo_smooth_dict = dict()

    company_final_senti_all_values = []
    for company_short, company_date_micro_senti in tqdm(company_micro_senti_dict.items(), desc='Calculating final sentiment'):
        company_micro_senti = np.array([np.float64(sentiment) for _, sentiment in company_date_micro_senti.items()])
        company_final_senti = company_micro_senti + company_meso_senti_dict[company_short]
        company_senti_only_meso = company_meso_senti_dict[company_short]
        company_senti_only_micro = company_micro_senti
        company_final_senti_aft_smooth = smoothing(company_final_senti, smoothing_factor=smoothing_factor)
        # company_senti_only_meso_aft_smooth = smoothing(company_senti_only_meso, smoothing_factor=smoothing_factor)
        # company_senti_only_micro_aft_smooth = smoothing(company_senti_only_micro, smoothing_factor=smoothing_factor)
        # company_final_senti_all_values.append(company_final_senti)
        # company_final_senti_dict[company_short] = company_final_senti
        company_senti_wo_smooth_date = {date:senti for date, senti in zip(date_list, company_final_senti)}
        company_final_senti_date = {date:senti for date, senti in zip(date_list, company_final_senti_aft_smooth)}
        company_senti_only_meso_date = {date:senti for date, senti in zip(date_list, company_senti_only_meso)}
        company_senti_only_micro_date = {date:senti for date, senti in zip(date_list, company_senti_only_micro)}
        company_final_senti_dict[company_short] = company_final_senti_date
        company_senti_only_meso_dict[company_short] = company_senti_only_meso_date
        company_senti_only_micro_dict[company_short] = company_senti_only_micro_date
        company_senti_wo_smooth_dict[company_short] = company_senti_wo_smooth_date

    # all_final_values = np.array(company_final_senti_all_values)
    # all_final_values_mean = all_final_values.mean()
    # all_final_values_std = all_final_values.std()

    # for company_short, company_final_senti in tqdm(company_final_senti_dict.items()):
    #     company_final_senti = (company_final_senti - all_final_values_mean) / all_final_values_std
    #     company_final_senti_date = {date:senti for date, senti in zip(date_list, company_final_senti)}
    #     company_final_senti_dict[company_short] = company_final_senti_date

    company_final_senti_dict_path = f'../processed_data/company_sentiment_{wavelet}_{level}_sd.json'
    with open(company_final_senti_dict_path, 'w', encoding='utf-8') as file:
        json.dump(company_final_senti_dict, file, ensure_ascii=False)
    
    company_senti_only_meso_dict_path = f'../processed_data/company_sentiment_only_meso_{wavelet}_{level}_sd.json'
    with open(company_senti_only_meso_dict_path, 'w', encoding='utf-8') as file:
        json.dump(company_senti_only_meso_dict, file, ensure_ascii=False)
    
    company_senti_only_micro_dict_path = f'../processed_data/company_sentiment_only_micro_{wavelet}_{level}_sd.json'
    with open(company_senti_only_micro_dict_path, 'w', encoding='utf-8') as file:
        json.dump(company_senti_only_micro_dict, file, ensure_ascii=False)

    company_senti_wo_smooth_dict_path = f'../processed_data/company_sentiment_wo_smooth_{wavelet}_{level}_sd.json'
    with open(company_senti_wo_smooth_dict_path, 'w', encoding='utf-8') as file:
        json.dump(company_senti_wo_smooth_dict, file, ensure_ascii=False)

    print("Incorporating sentiment into bond data...")

    def split_list_np(lst, ratios=(0.7, 0.1, 0.2)):
        lst = np.array(lst)
        total = len(lst)
        indices = np.random.permutation(total)

        split1 = int(total * ratios[0])
        split2 = split1 + int(total * ratios[1])

        return lst[indices[:split1]].tolist(), lst[indices[split1:split2]].tolist(), lst[indices[split2:]].tolist()

    train_txt_path = "../processed_data/nn_data/split/train_files.txt"
    val_txt_path = "../processed_data/nn_data/split/val_files.txt"
    test_txt_path = "../processed_data/nn_data/split/test_files.txt"

    if os.path.exists(train_txt_path) and os.path.exists(val_txt_path) and os.path.exists(test_txt_path):
        train_files = open(train_txt_path, "r").read().split("\n")
        val_files = open(val_txt_path, "r").read().split("\n")
        test_files = open(test_txt_path, "r").read().split("\n")

    else:
        train_files, val_files, test_files = split_list_np(file_names)
        print("Total files:", len(file_names))
        print("Train files:", len(train_files))
        print("Validation files:", len(val_files))
        print("Test files:", len(test_files))


        os.makedirs(os.path.dirname(train_txt_path), exist_ok=True)
        os.makedirs(os.path.dirname(val_txt_path), exist_ok=True)
        os.makedirs(os.path.dirname(test_txt_path), exist_ok=True)

        with open(train_txt_path, "w") as f:
            f.write("\n".join(train_files))

        with open(val_txt_path, "w") as f:
            f.write("\n".join(val_files))

        with open(test_txt_path, "w") as f:
            f.write("\n".join(test_files))

    import os
    import pandas as pd
    from tqdm import tqdm

    final_dataframe = pd.read_csv("../processed_data/bond_data_prenormalized.csv")

    # start_data = '2020-01-01'
    # end_date = '2020-12-31'
    start_date = '2013-01-01'
    end_date = '2023-12-31'

    final_dataframe["日期"] = pd.to_datetime(final_dataframe["日期"])
    filtered_df = final_dataframe[(final_dataframe["日期"] >= start_date) & (final_dataframe["日期"] <= end_date)]
    filtered_df.drop(columns=['债券分类违约概率', '区域违约概率', '到期收益率', '剩余期限'], inplace=True)
    filtered_df.head()


    all_bond_company_df = pd.read_excel('../processed_data/all_bond_company_data.xlsx', sheet_name='all')
    company_short_list = []
    for index, row in all_bond_company_df.iterrows():
        company_short_list.append(row['发行人中文简称'])
    company_short_set = set(company_short_list)

    filtered_df = filtered_df[filtered_df['发行人中文简称'].isin(company_short_set)]
    filtered_df.count()

    import json

    company_final_senti_dict['紫光集团']['2020-01-01']

    def compute_sentiment(row, company_senti_dict=company_final_senti_dict):
        date = row["日期"]
        company_short = row["发行人中文简称"]
        idx_date = date.strftime("%Y-%m-%d")
        return company_senti_dict[company_short][idx_date]

    filtered_df['sentiment'] = filtered_df.apply(
    lambda row: compute_sentiment(row, company_senti_dict=company_final_senti_dict), axis=1
    )

    filtered_df['micro_sentiment'] = filtered_df.apply(
    lambda row: compute_sentiment(row, company_senti_dict=company_senti_only_micro_dict), axis=1
    )

    filtered_df['meso_sentiment'] = filtered_df.apply(
    lambda row: compute_sentiment(row, company_senti_dict=company_senti_only_meso_dict), axis=1
    )

    filtered_df['sentiment_wo_smooth'] = filtered_df.apply(
    lambda row: compute_sentiment(row, company_senti_dict=company_senti_wo_smooth_dict), axis=1
    )

    print(filtered_df.head())


    # 复制原始 DataFrame
    standardized_final_dataframe = filtered_df.copy()

    # 选择需要标准化的列
    columns_to_standardize = list(standardized_final_dataframe.columns[1:47])

    # 计算标准化 (z 分数)
    standardized_final_dataframe[columns_to_standardize] = (
        standardized_final_dataframe[columns_to_standardize] - standardized_final_dataframe[columns_to_standardize].mean()
    ) / standardized_final_dataframe[columns_to_standardize].std()

    # grouped = standardized_final_dataframe.groupby('file_name')

    # train_path = f"../processed_data/nn_data/train_{wavelet}_{level}_sd2"
    # val_path = f"../processed_data/nn_data/val_{wavelet}_{level}_sd2"
    # test_path = f"../processed_data/nn_data/test_{wavelet}_{level}_sd2"

    # os.makedirs(train_path, exist_ok=True)
    # os.makedirs(val_path, exist_ok=True)
    # os.makedirs(test_path, exist_ok=True)

    # max_rows = 0
    # min_rows = 1000000

    # for file_name, group in tqdm(grouped):
    #     file_path = f"{file_name}.npy"
    #     max_rows = max(max_rows, len(group))  # 更新最大行数

    #     if len(group) <= 30:
    #         continue

    #     min_rows = min(min_rows, len(group))  # 更新最小行数

    #     # 选择第 1-50 列（索引 1-50）和第 57 列（索引 57）
    #     selected_data = group.iloc[:, list(range(1, 47)) + [53] + [54] + [55] + [56]].to_numpy()

    #     if file_name in train_files:
    #         save_path = os.path.join(train_path, file_path)
    #     elif file_name in val_files:
    #         save_path = os.path.join(val_path, file_path)
    #     elif file_name in test_files:
    #         save_path = os.path.join(test_path, file_path)

    #     np.save(save_path, selected_data)  # 只保存数据，不包含列名

    # print(f"Grouped 中的最大行数: {max_rows}")
    # print(f"Grouped 中的最小行数: {min_rows}")


    standardized_final_dataframe_path = "../processed_data/bond_data_normalized_w_senti.csv"
    standardized_final_dataframe["日期"] = pd.to_datetime(standardized_final_dataframe["日期"], format="%Y-%m-%d")

    # 按 '日期' 和 '发行人中文简称' 排序，确保数据有序
    standardized_final_dataframe = standardized_final_dataframe.sort_values(by=["发行人中文简称", "日期"])

    # 设定多级索引，以 '日期' 和 '发行人中文简称' 共同唯一标识每一行
    standardized_final_dataframe = standardized_final_dataframe.set_index(["日期", "发行人中文简称"])

    result_df = standardized_final_dataframe.groupby(["日期", "发行人中文简称"]).apply(custom_agg)

    # 获取 t+1、t+2、t+3 的因变量（后一天、后两天、后三天的 '风险价差'）
    standardized_final_dataframe["y_t1"] = standardized_final_dataframe.groupby("发行人中文简称")["风险价差"].shift(-1)
    standardized_final_dataframe["y_t2"] = standardized_final_dataframe.groupby("发行人中文简称")["风险价差"].shift(-2)
    standardized_final_dataframe["y_t3"] = standardized_final_dataframe.groupby("发行人中文简称")["风险价差"].shift(-3)
    standardized_final_dataframe["y_t4"] = standardized_final_dataframe.groupby("发行人中文简称")["风险价差"].shift(-4)

    # 先确定完整的索引
    valid_indices = standardized_final_dataframe[["y_t1", "y_t2", "y_t3", "y_t4"]].notna().all(axis=1)  # 确保所有因变量均非空

    # 过滤出完整数据
    standardized_final_dataframe = standardized_final_dataframe.loc[valid_indices]

    # 保存数据
    standardized_final_dataframe.to_csv(standardized_final_dataframe_path.replace('.csv', '_shift.csv'), index=True)