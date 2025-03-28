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
from _0_0_train_BERT.models import *

def process_blank_in_text(text):
    text = re.sub(r'\u0020',"", text)
    text = re.sub(r'\u3000',"", text)
    text = re.sub(r'\xa0',"", text)
    text = re.sub(r'\n',"", text)
    text = re.sub(r'\t',"", text)
    text = re.sub(r'\u2002',"", text)
    text = re.sub(r'\u2003',"", text)
    return text


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", default="", type=str,
                            help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--model_name_or_path", default='../bert-base-chinese', type=str)
    parser.add_argument("--sentiment_nums", default=3, type=int)
    parser.add_argument("--do_lower_case",default=True)
    parser.add_argument("--device",default='cuda:7')
    args = parser.parse_args()

    start_date = datetime.strptime("2013-01-01", "%Y-%m-%d")
    end_date = datetime.strptime("2023-12-31", "%Y-%m-%d")
    # start_date = datetime.strptime("2020-01-01", "%Y-%m-%d")
    # end_date = datetime.strptime("2020-12-31", "%Y-%m-%d")

    #--------微观部分--------
    config = BertConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, 
        num_labels=args.sentiment_nums)

    BERT = BertModel.from_pretrained(args.model_name_or_path, config=config)
    MLP = MLP(config=config,args=args)

    bert_checkpoint = torch.load('./_0_0_train_BERT/BERT_best_MSE.pth', map_location=torch.device('cpu'))
    BERT.load_state_dict(bert_checkpoint)

    mlp_checkpoint = torch.load('./_0_0_train_BERT/MLP_best_MSE.pth', map_location=torch.device('cpu'))
    MLP.load_state_dict(mlp_checkpoint)

    BERT.to(device=args.device)
    MLP.to(device=args.device)

    BERT.eval()
    MLP.eval()

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, 
                                                do_lower_case=args.do_lower_case)

    all_bond_company_df = pd.read_excel('../processed_data/all_bond_company_data.xlsx', sheet_name='all')
    company_short_list = []
    for index, row in all_bond_company_df.iterrows():
        company_short_list.append(row['发行人中文简称'])
    company_short_set = set(company_short_list)

    all_micro_data_path = '../raw_data/all_micro_data.json'

    with open(all_micro_data_path, 'r', encoding='utf-8') as file:
        all_micro_data = json.load(file)
    
    company_iter_dict = defaultdict(lambda: 0)
    company_final_senti_dict = dict()
    bond_dict = dict()

    mapping = torch.tensor([-1, 0, 1])

    with torch.no_grad():
        for company_short in tqdm(company_short_set):
            date_text_list = all_micro_data.get(company_short, None)
            if date_text_list:
                company_iter_dict.clear()
                for date, text_list in date_text_list.items():
                    date = datetime.strptime(date, "%Y-%m-%d")
                    day_tensor = 0
                    text_num = 0
                    text_list_tensor = []
                    sentiment_list_oneday = []
                    if date >= start_date and date <= end_date:
                        for text in text_list:
                            text = process_blank_in_text(text)
                            text = text[:512]

                            company_tokenized = tokenizer.encode_plus(
                                company_short,                      # Sentence to encode.
                                add_special_tokens=True,     # Add '[CLS]' and '[SEP]'
                                max_length=20,               # Pad & truncate all sentences.
                                padding=True,
                                return_attention_mask=True,   # Construct attn. masks.
                                return_tensors='pt',         # Return pytorch tensors.
                                truncation=True
                            )
                            company_tokenized_ids = company_tokenized['input_ids']
                            company_tokenized_masks = company_tokenized['attention_mask']
                            company_tokenized_ids = company_tokenized_ids.to(device=args.device)
                            company_tokenized_masks = company_tokenized_masks.to(device=args.device)
                            text_tokenized = tokenizer.encode_plus(
                                text,                      # Sentence to encode.
                                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                max_length=512,           # Pad & truncate all sentences.
                                padding=True,
                                return_attention_mask=True,   # Construct attn. masks.
                                return_tensors='pt',         # Return pytorch tensors.
                                truncation=True
                            )
                            text_tokenized_ids = text_tokenized['input_ids']
                            text_tokenized_masks = text_tokenized['attention_mask']
                            text_tokenized_ids = text_tokenized_ids.to(device=args.device)
                            text_tokenized_masks = text_tokenized_masks.to(device=args.device)

                            # 统计mask中1的个数
                            cnt = company_tokenized_masks.sum().item()
                            positions = find_all_full_positions2(
                                text_tokenized_ids[0].tolist(), company_tokenized_ids[0].tolist()[1:cnt-1])
                            if positions:
                                output = BERT(text_tokenized_ids, text_tokenized_masks)
                                tensor = torch.cat(
                                    [output[0][0][i: j+1] for (i, j) in positions], dim=0)  # [n * l, 768] n: 实体匹配cnt; l: 实体名称长度
                                mean_val = torch.mean(
                                    tensor, dim=0)  # [768]
                                max_val = torch.max(
                                    tensor, dim=0).values  # [768]
                                input_tensor = torch.cat(
                                    [mean_val, max_val, output[1][0]], dim=0)  # [768 * 3]
                                input_tensor = input_tensor.unsqueeze(
                                    0).to(device=args.device)  # [1, 2304]
                                text_list_tensor.append(input_tensor)
                                text_num += 1

                        if text_num  > 0:
                            text_list_tensor = torch.cat(text_list_tensor, dim=0)
                            sentiment_predict = MLP(text_list_tensor).cpu()
                            max_indices = torch.argmax(sentiment_predict, dim=1)
                            text_list_sentiment = mapping[max_indices].to(torch.float32)
                            day_tensor = float(torch.mean(text_list_sentiment))
                            date = date.strftime("%Y-%m-%d")
                            company_iter_dict[date] = day_tensor
                    
            current_date = start_date
            while current_date <= end_date:
                current_date_str = current_date.strftime("%Y-%m-%d")
                _ = company_iter_dict[current_date_str]
                current_date += timedelta(days=1)

            # 使用深拷贝创建新的字典
            company_list = sorted(company_iter_dict.items(), key=lambda x: datetime.strptime(x[0], '%Y-%m-%d'))
            company_micro_senti = {date:sentiment for date, sentiment in deepcopy(company_list)}
            company_final_senti_dict[company_short] = company_micro_senti

    with open('../processed_data/company_micro_sentiment_mean.json', 'w', encoding='utf-8') as file:
        json.dump(company_final_senti_dict, file, ensure_ascii=False)