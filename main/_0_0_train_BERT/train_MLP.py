import torch
import torch.nn as nn
import torch.functional as F
from pytorch_transformers import BertPreTrainedModel, BertModel
from transformers import BertTokenizer
import argparse
import csv
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from pytorch_transformers import (WEIGHTS_NAME, BertConfig)
from models import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


e = 2.718281828459045
loss_fn = F.mse_loss

def train(args, MLP, BERT, tokenizer, device):
    # 读取 Excel 文件
    data = pd.read_excel('../../processed_data/train_senti_BERT.xlsx')
    data = data[['公司', '摘要', '悲观', '中性', '乐观']]
    # 定义数值列
    numerical_cols = ['悲观', '中性', '乐观']
    # cross_attention = CrossAttention(embed_dim=args.embedding_dim, num_heads=8)
    # cross_attention.to(device=device)
    # numerical_cols = ['val']
    # 划分训练集和验证集
    train_data, valid_data = train_test_split(data, train_size=0.7, random_state=42)

    # 创建训练集和验证集的 Dataset 对象
    train_dataset = CustomDataset(train_data, numerical_cols)
    valid_dataset = CustomDataset(valid_data, numerical_cols)

    # 创建 DataLoader 对象
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=custom_collate)
    valid_loader = DataLoader(valid_dataset, batch_size=args.eval_batch_size, shuffle=True, collate_fn=custom_collate)
    optimizer_one = torch.optim.Adam(MLP.parameters(), lr=args.learning_rate,
                              weight_decay=1e-5)
    optimizer_two = torch.optim.Adam(BERT.parameters(), lr=args.learning_rate*0.1,
                              weight_decay=1e-7)
    # optimizer_three = torch.optim.Adam(cross_attention.parameters(), lr=args.learning_rate,
    #                           weight_decay=1e-5)
    best_loss = float('inf')
    # best_loss = 0.04277444130121546
    best_acc = 0
    train_loss = []
    train_acc_list = []
    valid_loss = []
    valid_acc_list = []

    for epoch in range(args.num_train_epochs):
        MLP.train()
        BERT.train()
        # cross_attention.train()
        train_losses = train_acc = 0
        for numerical_batch, text_batch in tqdm(train_loader):
            optimizer_one.zero_grad()
            optimizer_two.zero_grad()
            # optimizer_three.zero_grad()
            numerical_batch = numerical_batch.to(device=device)

            company_input_ids_list = []
            company_attention_masks_list = []
            abstract_input_ids_list = []
            abstract_attention_masks_list = []
            position_list = [[] for _ in range(args.train_batch_size)]
            embed_list = [[] for _ in range(args.train_batch_size)]
            pool_list = [[] for _ in range(args.train_batch_size)]
            for i,text_data in enumerate(text_batch):
                company_text = text_data['公司'].lower()
                abstract_text = text_data['摘要'].lower()
                abstract_text = abstract_text.replace("\u3000", '')

                # 对 '公司' 列进行 tokenize
                company_inputs = tokenizer.encode_plus(
                            company_text,                      # Sentence to encode.
                            add_special_tokens=True, # Add '[CLS]' and '[SEP]'
                            max_length=20,           # Pad & truncate all sentences.
                            pad_to_max_length=True,
                            return_attention_mask=True,   # Construct attn. masks.
                            return_tensors='pt',  # Return pytorch tensors.
                            truncation=True     
                    )
                company_input_ids_list.append(company_inputs['input_ids'])
                company_attention_masks_list.append(company_inputs['attention_mask'])


                # 对 '摘要' 列进行 tokenize
                abstract_inputs = tokenizer.encode_plus(
                            abstract_text,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 512,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',  # Return pytorch tensors.
                            truncation=True     
                    )
                abstract_input_ids_list.append(abstract_inputs['input_ids'])
                abstract_attention_masks_list.append(abstract_inputs['attention_mask'])

                # 统计mask中1的个数
                cnt = company_inputs['attention_mask'].sum().item()
                positions = find_all_full_positions2(abstract_inputs['input_ids'][0].tolist(), company_inputs['input_ids'][0].tolist()[1:cnt-1])
                position_list[i].append(positions)


            # 将 input_ids 和 attention_masks 转换为张量
            company_input_ids = torch.cat(company_input_ids_list, dim=0).to(device=device)
            company_attention_masks = torch.cat(company_attention_masks_list, dim=0).to(device=device)
            abstract_input_ids = torch.cat(abstract_input_ids_list, dim=0).to(device=device)
            abstract_attention_masks = torch.cat(abstract_attention_masks_list, dim=0).to(device=device)

            # 将数据输入到 BERT 模型中
            company_outputs = BERT(company_input_ids, company_attention_masks)
            abstract_outputs = BERT(abstract_input_ids, abstract_attention_masks)

            for item in range(args.train_batch_size): 
                for position in position_list[item]:
                    for (i,j) in position:
                        for t in range(i,j+1):
                            embed_list[item].append(abstract_outputs[0][item][t].cpu().detach().numpy())
                tensor = torch.tensor(embed_list[item])
                mean_val = torch.mean(tensor, dim=0)
                max_val = torch.max(tensor, dim=0)
                # pool_list[item] = torch.cat((mean_val,max_val[0]),dim=0)
                pool_list[item] = mean_val

            input_tensor = torch.stack(pool_list, dim=0).to(device=device)
            pred = MLP(torch.cat((input_tensor,abstract_outputs[1]),dim=1))

            label = numerical_batch.clone().detach()
            pred_classes = torch.argmax(pred, dim=1)
            label_classes = torch.argmax(label, dim=1) 

            train_acc += torch.sum(pred_classes == label_classes).item() / args.train_batch_size

            loss = loss_fn(pred, label)

            loss.backward()
            optimizer_one.step()
            optimizer_two.step()
            # optimizer_three.step()
            train_losses += loss.item()

        train_acc /= len(train_loader)
        train_losses /= len(train_loader)
        print('train_loss:' + str(train_losses) + f' in epoch{epoch}')
        print('train_acc:' + str(train_acc) + f' in epoch{epoch}')
        train_loss.append(train_losses)
        train_acc_list.append(train_acc)

        MLP.eval()
        BERT.eval()
        valid_losses = 0
        valid_acc = 0
        d = {}
        with torch.no_grad():
            for numerical_batch, text_batch in tqdm(valid_loader):
                numerical_batch = numerical_batch.to(device=device)

                company_input_ids_list = []
                company_attention_masks_list = []
                abstract_input_ids_list = []
                abstract_attention_masks_list = []
                position_list = [[] for _ in range(args.eval_batch_size)]
                embed_list = [[] for _ in range(args.eval_batch_size)]
                pool_list = [[] for _ in range(args.eval_batch_size)]
                for i,text_data in enumerate(text_batch):
                    company_text = text_data['公司'].lower()
                    abstract_text = text_data['摘要'].lower()
                    abstract_text = abstract_text.replace("\u3000", '')

                    # 对 '公司' 列进行 tokenize
                    company_inputs = tokenizer.encode_plus(
                                company_text,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length=20,           # Pad & truncate all sentences.
                                pad_to_max_length = True,
                                return_attention_mask = True,   # Construct attn. masks.
                                return_tensors = 'pt',  # Return pytorch tensors.
                                truncation=True     
                        )
                    company_input_ids_list.append(company_inputs['input_ids'])
                    company_attention_masks_list.append(company_inputs['attention_mask'])

                    # 对 '摘要' 列进行 tokenize
                    abstract_inputs = tokenizer.encode_plus(
                                abstract_text,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = 512,           # Pad & truncate all sentences.
                                pad_to_max_length = True,
                                return_attention_mask = True,   # Construct attn. masks.
                                return_tensors = 'pt',  # Return pytorch tensors.
                                truncation=True     
                        )
                    abstract_input_ids_list.append(abstract_inputs['input_ids'])
                    abstract_attention_masks_list.append(abstract_inputs['attention_mask'])

                    # 统计mask中1的个数
                    cnt = company_inputs['attention_mask'].sum().item()
                    positions = find_all_full_positions2(abstract_inputs['input_ids'][0].tolist(), company_inputs['input_ids'][0].tolist()[1:cnt-1])
                    position_list[i].append(positions)

                # 将 input_ids 和 attention_masks 转换为张量
                company_input_ids = torch.cat(company_input_ids_list, dim=0).to(device=device)
                company_attention_masks = torch.cat(company_attention_masks_list, dim=0).to(device=device)
                abstract_input_ids = torch.cat(abstract_input_ids_list, dim=0).to(device=device)
                abstract_attention_masks = torch.cat(abstract_attention_masks_list, dim=0).to(device=device)

                # 将数据输入到 BERT 模型中
                company_outputs = BERT(company_input_ids, company_attention_masks)
                abstract_outputs = BERT(abstract_input_ids, abstract_attention_masks)

                for item in range(args.eval_batch_size): 
                    for position in position_list[item]:
                        for (i,j) in position:
                            for t in range(i,j+1):
                                embed_list[item].append(abstract_outputs[0][item][t].cpu().detach().numpy())
                    tensor = torch.tensor(embed_list[item])
                    if tensor.nelement() != 0:
                        mean_val = torch.mean(tensor, dim=0)
                        max_val = torch.max(tensor, dim=0)
                        # pool_list[item] = torch.cat((mean_val,max_val[0]),dim=0)
                        pool_list[item] = mean_val

                    else:
                        pool_list.remove(pool_list[-1])

                input_tensor = torch.stack(pool_list, dim=0).to(device=device)
                pred = MLP(torch.cat((input_tensor,abstract_outputs[1]),dim=1))

                label = numerical_batch.clone().detach()
                pred_classes = torch.argmax(pred, dim=1)  # [8,1]
                label_classes = torch.argmax(label, dim=1)

                for res in pred_classes:
                    d[res.item()] = d.get(res.item(), 0) + 1

                valid_acc += torch.sum(pred_classes == label_classes).item() / args.eval_batch_size

                loss = loss_fn(pred, label)
                valid_losses += loss.item()

        print(d)
        valid_acc /= len(valid_loader)
        valid_losses /= len(valid_loader)  # 计算平均损失
        print('valid_loss:' + str(valid_losses) + f' in epoch{epoch}')
        print('valid_acc:' + str(valid_acc) + f' in epoch{epoch}')
        valid_loss.append(valid_losses)
        valid_acc_list.append(valid_acc)

        if best_loss > valid_losses:
            patience_counter = 0
            best_loss = valid_losses
            best_acc = valid_acc
            print(f'save the model in epoch{epoch}')
            logging.info(f'save the model in epoch{epoch}')
            torch.save(MLP.state_dict(), 'MLP_best_MSE_mean.pth')
            torch.save(BERT.state_dict(), 'BERT_best_MSE_mean.pth')
        # elif epoch == args.num_train_epochs-1:
        #     torch.save(MLP.state_dict(), 'MLP_last_MSE.pth')
        #     torch.save(BERT.state_dict(), 'BERT_last_MSE.pth')

    epochs = list(range(epoch+1))  
    # 绘制曲线图
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, valid_loss, label='Valid Loss')
    plt.plot(epochs, train_acc_list, label='Train Acc')
    plt.plot(epochs, valid_acc_list, label='Valid Acc')
    # 添加图例
    plt.legend()

    # 添加标题和标签
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylabel('Acc')
    # 显示网格线
    plt.grid(True)

    # 显示图形
    plt.savefig('loss&Acc.png')


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()

    ## 必须的参数
    parser.add_argument("--model_name_or_path", default='../../bert-base-chinese', type=str)
    ## 可选参数
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--num_train_epochs", default=100, type=float)
    parser.add_argument("--do_lower_case", default=True)
    parser.add_argument("--embedding_dim", default=768, type=int)
    parser.add_argument("--sentiment_nums", default=3, type=int)
    parser.add_argument("--device", default='cuda:2', type=str)
    args = parser.parse_args()

    config = BertConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, 
        num_labels=args.sentiment_nums)

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, 
                do_lower_case=args.do_lower_case)
    BERT =BertModel.from_pretrained(args.model_name_or_path, config=config)
    MLP = MLP(config=config,args=args)
    # checkpoint = torch.load('/project/Finbert/Fink_NER_ED_Web/main/FinBERT_best_DICE_MSE_fcn_3bloc.pth', map_location=torch.device('cpu'))
    # BERT_subset.load_state_dict(checkpoint)

    # checkpoint = torch.load('/project/Finbert/Fink_NER_ED_Web/main/MLP_best_MSE_mlp2.pth', map_location=torch.device('cpu'))
    # MLP.load_state_dict(checkpoint)

    num_cuda_devices = torch.cuda.device_count()

    print("可用的CUDA设备数量：", num_cuda_devices)

    torch.manual_seed(15832775322209013207)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    MLP.to(device)
    BERT.to(device)
    train(args, MLP, BERT, tokenizer, device)
