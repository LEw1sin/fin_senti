import torch
import torch.nn as nn
import torch.functional as F
from pytorch_transformers import BertPreTrainedModel, BertModel
from transformers import BertTokenizer
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

from pytorch_transformers import (WEIGHTS_NAME, BertConfig)
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, dataframe, numerical_cols):
        self.numerical_data = dataframe[numerical_cols].values.astype(float)  # 数值列数据
        self.text_data = dataframe.drop(columns=numerical_cols)  # 非数值列数据
    
    def __len__(self):
        return len(self.numerical_data)
    
    def __getitem__(self, idx):
        numerical_tensor = torch.tensor(self.numerical_data[idx], dtype=torch.float)
        text_data = self.text_data.iloc[idx]  # 获取对应索引的非数值列数据
        return numerical_tensor, text_data

def custom_collate(batch):
    numerical_batch, text_batch = zip(*batch)
    numerical_batch = torch.stack(numerical_batch, dim=0)
    return numerical_batch, text_batch


class MLP(torch.nn.Module):
    def __init__(self, config=None, args=None, output_n=5, layer_list=[768*3, 768,64], dropout=0.5, rnn_dim=128):
        """
        :param output_n: int 输出神经元个数
        :param layer_list: list(int) 每层隐藏层神经元个数
        :param dropout: float 训练完丢掉多少
        """
        super(MLP, self).__init__()
        self.output_n = args.sentiment_nums
        self.num_layer = len(layer_list)
        self.layer_list = layer_list
        # self.bilstm = nn.LSTM(config.hidden_size*2, rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
        # self.dropout = nn.Dropout(dropout)
        # 隐藏层
        self.hidden_layer = nn.ModuleList(
            [nn.Sequential(nn.Linear(layer_list[i], layer_list[i+1], bias=True),
                          torch.nn.BatchNorm1d(layer_list[i+1]),
                          nn.ReLU(),
            ) for i in range(self.num_layer-1)]
        )

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(layer_list[-1], self.output_n, bias=True),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        # x, _ = self.bilstm(x)
        # x = self.dropout(x)
        for layer in self.hidden_layer:
            x = layer(x)
        x = self.output_layer(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # 增加维度使得查询向量适配多头注意力的形状
        Q = Q.unsqueeze(1)  # Query维度变为 [batch_size, 1, embed_dim]
        
        # 重塑为多头注意力需要的形状
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算得分
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 注意力权重和输出
        attention = F.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)
        
        # 降低一个维度
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        output = output.squeeze(1)  # Remove the '1' dimension
        return output


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

# Function to find all occurrences of the substring in the text.
def find_all_full_positions(text, substring):
    text = text.tolist()
    start = 0
    positions = []
    while True:
        start = text.find(substring, start)
        if start == -1:  # no more occurrences found
            break
        end = start + len(substring)
        if end >= 511:
            break
        positions.append((start, end))
        start += len(substring)  # move past the last found substring
    return positions

def find_all_full_positions2(lst, sub_lst):
    positions = []
    for i in range(0, len(lst) - len(sub_lst) + 1):
        if lst[i] == sub_lst[0]:
            for j in range(1, len(sub_lst)):
                if lst[i + j] != sub_lst[j]: break
                elif j == len(sub_lst) - 1:
                    positions.append((i, i+j))
    return positions

def my_delete(my_list):
    for item in my_list:
        if not item:
            my_list.remove(item)
    return my_list

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# convert company name to token number
with open(os.path.abspath('../../bert-base-chinese/vocab.txt'), "r") as f:
    lines = f.readlines()
    d = {}
    for idx, line in enumerate(lines):
        d[line.strip()] = idx

def company2number(text):
    return [d[word.lower()] for word in text]


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()

    ## 必须的参数
    parser.add_argument("--model_name_or_path", default='../../bert-base-chinese', type=str)
    parser.add_argument("--output_dir", default='../output_test', type=str)

    ## 可选参数
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3") 
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--do_train", default=True, type=boolean_string)
    parser.add_argument("--do_eval", default=True, type=boolean_string)
    parser.add_argument("--do_test", default=True, type=boolean_string)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=3e-6, type=float)
    parser.add_argument("--num_train_epochs", default=10, type=float)
    parser.add_argument("--warmup_proprotion", default=0.1, type=float)
    parser.add_argument("--use_weight", default=1, type=int)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--fp16", default=False)
    parser.add_argument("--loss_scale", type=float, default=0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--do_lower_case",default=True)
    parser.add_argument("--logging_steps", default=200, type=int)
    parser.add_argument("--clean", default=False, type=boolean_string, help="clean the output dir")
    parser.add_argument("--need_birnn", default=False, type=boolean_string)
    parser.add_argument("--rnn_dim", default=256, type=int)
    parser.add_argument("--embedding_dim", default=768, type=int)
    parser.add_argument("--sentiment_nums", default=3, type=int)
    args = parser.parse_args()

    config = BertConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, 
        num_labels=args.sentiment_nums)


    model_subset = BertModel.from_pretrained(args.model_name_or_path, config=config)
    model_linear = MLP(config=config,args=args)

    # 加载模型参数
    checkpoint = torch.load('./BERT_best_MSE.pth', map_location=torch.device('cpu'))
    # 从检查点中加载模型状态字典
    model_subset.load_state_dict(checkpoint)
    
    # 加载模型参数
    checkpoint2 = torch.load('./MLP_best_MSE.pth', map_location=torch.device('cpu'))
    # 从检查点中加载模型状态字典
    model_linear.load_state_dict(checkpoint2)

    text = "亚太转债"
    text2 = "常熟银行"
    text3 = "转债配置窗口打开    随着A股涨势放缓,本周以来可转债市场也终结连续上涨局面,个券分化加大。21日,中证转债指数小幅上涨,个券涨跌参半。市场人士表示,转债市场筑底特征明显,试错成本降低,可以在坚守底仓的情况下,把握一些局部机会。回归震荡 借着股市反弹的暖风,11月中旬转债市场走出一波连续上涨的行情。从指数走势上看,11月12日至19日,中证转债指数一连6日上涨。这一局面在20日被打破,当天A股上证综指下跌超过2%,中证转债指数随之出现接近1%的回调,单日跌幅创1个月最大。 21日,转债市场稍稍企稳,中证转债指数先抑后扬,收盘微涨0.08%。个券表现分化,两市正常交易的94只可转债中,有54只上涨,40只下跌。上涨转债对应的正股也以上涨居多,反之,正股多出现了下跌。但值得注意的是,转债个券表现似乎不及正股。例如,在上涨的个券中,涨幅最大的是康泰转债(4.47%),其正股康泰生物上涨5.45%;常熟转债、鼎信转债分别以1.11%和1.03%的涨幅紧随其后,但也都低于正股的涨幅,常熟银行上涨2.07%,鼎信通讯上涨4.77%。还有一部分正股在上涨,转债则出现下跌,例如,广电转债、亚太转债分别下跌4%和2%,而正股双双上涨超过1%。"
    # text3 = "29日地产债走势跌涨不一    中证网讯 数据显示，截至29日收盘，地产债走势分化，涨跌不一，其中“20时代05”涨超24%，“20阳城04”和“19世茂G3”涨超9%，而“20阳城03”跌超9%。"
    # text3 = '上交所:泰晶转债盘中临时停牌 停牌前涨20.03%　　上证报中国证券网讯 上交所7月1日公告称,泰晶转债(113503)今日上午交易出现异常波动。根据有关规定,自9时30分开始暂停泰晶转债(113503)交易,自10时00分起恢复交易。停牌前,15宜华02报价179.87元,上涨20.03%。如该债券交易再次出现异常波动,可实施第二次盘中临时停牌,停牌时间持续至今日14时57分。'
    text3 = text3.replace("\u3000", '')
    position_list = []
    embed_list = []
    b = []

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, 
                    do_lower_case=args.do_lower_case)
    input = tokenizer.encode_plus(
                        text,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 20,           # Pad & truncate all sentences.
                        padding = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',  # Return pytorch tensors.
                        truncation=True     
                   )
    input_ids = input['input_ids']
    attention_masks = input['attention_mask']

    input2 = tokenizer.encode_plus(
                        text2,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 512,           # Pad & truncate all sentences.
                        padding = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',  # Return pytorch tensors.
                        truncation=True     
                   )
    input_ids2 = input2['input_ids']
    attention_masks2 = input2['attention_mask']

    input3 = tokenizer.encode_plus(
                        text3,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 512,           # Pad & truncate all sentences.
                        padding = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',  # Return pytorch tensors.
                        truncation=True     
                   )
    input_ids3 = input3['input_ids']
    attention_masks3 = input3['attention_mask']

    # 统计mask中1的个数
    cnt = input['attention_mask'].sum().item()
    positions = find_all_full_positions2(input3['input_ids'][0].tolist(), input['input_ids'][0].tolist()[1:cnt-1])

    model_subset.eval()
    model_linear.eval()
    with torch.no_grad():
        output = model_subset(input_ids,attention_masks)
        output2 = model_subset(input_ids2,attention_masks2)
        output3 = model_subset(input_ids3,attention_masks3)

        temp = output2[1] - output[1]
        for (i,j) in positions:
            for t in range(i,j+1):
                b.append(output3[0][0][t])

        tensor = torch.stack(b, dim=0)
        mean_val = torch.mean(tensor, dim=0)
        max_val = torch.max(tensor, dim=0)
        input_tensor = torch.cat((mean_val,max_val[0]),dim=0)
        input_tensor = input_tensor.unsqueeze(0)

        # attention = cross_attention(output[1], output3[0],output3[0])
        # sentiment_predict = model_linear(attention)
        # sentiment_predict = model_linear(input_tensor)
        sentiment_predict = model_linear(torch.cat((input_tensor, output3[1]), dim=1))
        
        print(f"{text}:{sentiment_predict}")






