from torch.utils.data import Dataset
import torch
import numpy as np
import os
from functools import wraps
import logging
from torch.nn.utils import rnn as rnn_utils
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from scipy.stats import t
from sklearn.utils import resample

class TimeSeriesDataset(Dataset):
    def __init__(self, split_path, max_seq_length=2200, cache=True):
        self.file_list = os.listdir(split_path)  # 获取文件列表
        self.max_seq_length = max_seq_length  # 设定最大长度（用于截断）
        self.split_path = split_path

        if cache:
            self.get_all_data()
            self.getitem = self.getitem_w_cache
        else:
            self.getitem = self.getitem_wo_cache

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data, lengths = self.getitem(idx)
        return data, lengths
    
    def get_all_data(self):
        self.data_list = []
        self.lengths_list = []
        for idx in tqdm(range(len(self.file_list)),desc='Loading data',unit='file'):
            data, lengths = self.getitem_wo_cache(idx)
            self.data_list.append(data)
            self.lengths_list.append(lengths)

    def getitem_wo_cache(self, idx):
        file_path = self.file_list[idx]
        data = np.load(os.path.join(self.split_path, file_path))
        data = torch.from_numpy(data).float()
        seq_length = data.shape[0]

        if self.max_seq_length:
            if seq_length > self.max_seq_length:
                data = data[:self.max_seq_length]  # 截断
            else:
                pad_size = self.max_seq_length - seq_length
                pad_tensor = torch.zeros((pad_size, data.shape[1]))  # 填充 0
                data = torch.cat([data, pad_tensor], dim=0)

        return data, seq_length
    
    def getitem_w_cache(self, idx):
        return self.data_list[idx], self.lengths_list[idx]

def setup_logging(log_file_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def get_log_dir(args, train_eval=''):
        from datetime import datetime
        import os
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        log_dir_suffix = datetime.now().strftime("%m-%d-%H-%M-%S")
        train_eval = f'{train_eval}_' if train_eval else ''
        senti = 'senti_' if args.senti else ''
        loss = '_'.join(args.loss_list)
        log_dir = f'../logistics/{train_eval}{args.model}_{args.t_type}_{args.window}_{loss}_{senti}{log_dir_suffix}/'
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

class RNNRegressor(nn.Module):
    def __init__(self, args, input_size, hidden_size=128):
        super(RNNRegressor, self).__init__()
        self.args = args

        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(0.5)

        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(0.25)

        self.lstm3 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout3 = nn.Dropout(0.125)

        self.lstm4 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm5 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)  # 输出回归值

    def forward(self, data_truncated):

        out, _ = self.lstm1(data_truncated)
        # out = self.dropout1(out)

        out, _ = self.lstm2(out)
        # out = self.dropout2(out)

        out, _ = self.lstm3(out)
        # out = self.dropout3(out)

        out, _ = self.lstm4(out)
        out, _ = self.lstm5(out)

        out = self.fc(out) #(batch_size, seq_len, 1)
        out = out[:,-1,:]  # (batch_size, 1)
        return out  
    
class TransformerEncoderRegressor(nn.Module):
    def __init__(self, args, input_size):
        super(TransformerEncoderRegressor, self).__init__()
        self.args = args
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=5)
        self.fc = nn.Linear(input_size, 1)  # 输出回归值
    def forward(self, data_truncated):
        out = self.transformer_encoder(data_truncated)
        out = self.fc(out)
        out = out[:,-1,:]  # 形状变为 (batch_size, seq_len)
        return out
    
class RNN_Transformer_Hybrid_Regressor(nn.Module):
    def __init__(self, args, input_size, hidden_size=128):
        super(RNN_Transformer_Hybrid_Regressor, self).__init__()
        self.args = args

        self.lstm1 = nn.LSTM(input_size, hidden_size//2, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size//2, hidden_size, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.fc = nn.Linear(hidden_size, 1)  # 输出回归值

    def forward(self, data_truncated):
        out, _ = self.lstm1(data_truncated)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)

        out = self.transformer_encoder(out)
        out = self.fc(out)
        out = out[:,-1,:]  # 形状变为 (batch_size, seq_len)
        return out  # 形状应为 (batch_size, seq_len)


class Evaluator(object):
    def __init__(self, loss_list=None, weight_list=None):
        self.loss_list = loss_list
        self.weight_list = weight_list
    def loss(self, pred, target):
        total_loss = 0
        target = target.to(pred.device)
        for (loss, weight) in zip(self.loss_list, self.weight_list):
            if loss == 'mse':
                loss_fn = F.mse_loss(pred, target)
                total_loss = total_loss + loss_fn*weight
            elif loss == 'huber':
                loss_fn = F.smooth_l1_loss(pred, target)
                total_loss = total_loss + loss_fn*weight
            elif loss == 'rmse':
                loss_fn = torch.sqrt(F.mse_loss(pred, target))
                total_loss = total_loss + loss_fn*weight
        return total_loss
    def mse(self, pred, target):
        return F.mse_loss(pred, target)
    def mae(self, pred, target):
        return F.l1_loss(pred, target)
    def rmse(self, pred, target):
        return torch.sqrt(F.mse_loss(pred, target))
    def mape(self, pred, target):
        return torch.mean(torch.abs((target - pred) / (target + 1e-8)))

def collate_decorator(t_type, senti):
    def decorator(func):
        @wraps(func)
        def wrapper(batch):
            data, lengths = zip(*batch)  # 拆分数据和序列长度
            
            # 处理自变量（data_filtered）
            if senti:
                data_filtered = [d[:, list(range(44)) + [46]] for d in data]  # 取 0-47 列和第 50 列
            else:
                data_filtered = [d[:, list(range(44))] for d in data]  # 取 0-47 列
            
            # 处理因变量（target）
            if t_type == 't0':
                target = [d[:, 45] for d in data]  # 当前行的第 49 列
            elif t_type == 't1':
                target = [d[1:, 45] for d in data]  # 下一行的第 49 列
                lengths = [l-1 for l in lengths]  # 序列长度减 1
            elif t_type == 't2':
                target = [d[2:, 45] for d in data]  # 下两行的第 49 列
                lengths = [l-2 for l in lengths]  # 序列长度减 2
            elif t_type == 't3':
                target = [d[3:, 45] for d in data]  # 下三行的第 49 列
                lengths = [l-3 for l in lengths]  # 序列长度减 3
            elif t_type == 't4':
                target = [d[4:, 45] for d in data]
                lengths = [l-4 for l in lengths]
            else:
                raise ValueError("Invalid t_type value. Choose from ['t0', 't1', 't2', 't3', 't4']")
            
            # 填充序列到相同长度
            data_padded = rnn_utils.pad_sequence(data_filtered, batch_first=True, padding_value=0)
            target_padded = rnn_utils.pad_sequence(target, batch_first=True, padding_value=0)
            
            return data_padded, target_padded, torch.tensor(lengths, dtype=torch.long)
        return wrapper
    return decorator

def load_model(model, path):
    for file in os.listdir(path):
        if file.endswith('.pth'):
            model.load_state_dict(torch.load(os.path.join(path, file), weights_only=True))
            return model
        
def build_model(args):
    if args.model == 'lstm':
        net = RNNRegressor(args, input_size=45, hidden_size=args.max_channel) if args.senti else RNNRegressor(args, input_size=44, hidden_size=args.max_channel)
    elif args.model == 'transformer':
        net = TransformerEncoderRegressor(args, input_size=45) if args.senti else TransformerEncoderRegressor(args, input_size=44)
    elif args.model == 'hybrid':
        net = RNN_Transformer_Hybrid_Regressor(args, input_size=45, hidden_size=args.max_channel) if args.senti else RNN_Transformer_Hybrid_Regressor(args, input_size=44, hidden_size=args.max_channel)

    return net

def neural_net_feature_importance(model, X):
    """
    计算基于PyTorch神经网络模型的特征重要性。
    
    参数:
        model: 一个PyTorch神经网络模型。
        X: 输入张量，形状为 (seq_len, batch_size, feature_num)。
    
    返回:
        importance: 特征重要性值，形状为 (feature_num,)。
    """
    
    # 确保输入张量需要梯度
    X.requires_grad_(True)
    
    # 清空模型的梯度缓存
    model.zero_grad()
    
    # 获取模型的输出
    output = model(X)
    
    # 计算输出的均值
    output_mean = output.mean()
    
    # 反向传播计算梯度
    output_mean.backward()
    
    # 获取输入特征的梯度
    gradients = X.grad
    
    # 清空模型的梯度缓存
    model.zero_grad()
    
    # 计算特征重要性（梯度的绝对值）
    # 对序列维度 (seq_len) 和批次维度 (1) 取平均
    with torch.no_grad():
        importances = torch.abs(gradients).mean(dim=(0, 1)).cpu().numpy()
    
    # 分离输出以避免梯度传播
    output = output.detach()
    del gradients, output_mean, X
    torch.cuda.empty_cache()

    
    return importances, output

def permutation_importance(model, src, target, metric_fn):
    """
    计算基于特征置换的特征重要性。

    参数:
        model: PyTorch 神经网络模型。
        src: 输入张量，形状为 (batch_size, seq_len, feature_num)。
        target: 目标张量，与 `model(src)` 形状匹配。
        metric_fn: 评价指标函数，例如 `torch.nn.functional.mse_loss`。

    返回:
        importance_scores: 每个特征的影响程度，形状为 (feature_num,)。
    """
    # 计算基线评分
    with torch.no_grad():
        baseline_score = metric_fn(model(src), target).detach().cpu().item()

    importance_scores = []

    for feature_idx in range(src.shape[2]):  # 遍历每个特征
        src_permuted = src.clone()
        src_permuted[:, :, feature_idx] = src_permuted[torch.randperm(src.shape[0]), :, feature_idx]
        
        with torch.no_grad():
            permuted_score = metric_fn(model(src_permuted), target).detach().cpu().item()

        # 计算特征的重要性
        importance_scores.append(abs(permuted_score - baseline_score) / (baseline_score + 1e-8))

    return np.array(importance_scores)

def independent_t_test(x, y):
    """
    计算独立样本 t 值和 p 值
    参数:
        x: Tensor, 形状 (batch_size, 1)
        y: Tensor, 形状 (batch_size, 1)
    返回:
        t 值 (标量), p 值 (标量)
    """
    n_x, n_y = x.shape[0], y.shape[0]
    mean_x, mean_y = x.mean(), y.mean()
    var_x, var_y = x.var(unbiased=True), y.var(unbiased=True)

    # 计算 t 值
    t_value = (mean_x - mean_y) / torch.sqrt(var_x / n_x + var_y / n_y)
    
    # 计算自由度 df
    numerator = (var_x / n_x + var_y / n_y) ** 2
    denominator = ((var_x / n_x) ** 2) / (n_x - 1) + ((var_y / n_y) ** 2) / (n_y - 1)
    df = numerator / denominator

    # 计算 p 值 (双尾检验)
    p_value = 2 * (1 - t.cdf(abs(t_value.item()), df.item()))

    return t_value.item(), p_value

def permutation_test(series_a, series_b, n_permutations=50, metric='mean'):
    observed_diff = torch.mean(series_a) - torch.mean(series_b)  # 可替换为其他指标
    combined = torch.concatenate([series_a, series_b])
    perm_diffs = []
    for _ in range(n_permutations):
        permuted = resample(combined, replace=False)
        perm_a = permuted[:len(series_a)]
        perm_b = permuted[len(series_a):]
        perm_diff = torch.mean(perm_a) - torch.mean(perm_b)
        perm_diffs.append(torch.abs(perm_diff))

    perm_diffs = torch.tensor(perm_diffs).to(series_a.device)
    p_value = (perm_diffs >= torch.abs(observed_diff)).float().mean()
    return p_value

def get_data_window(args, src, target):
    window = args.window
    step = int(args.t_type[-1])
    src_list = [src[:,i:i+window,:] for i in range(0, src.size(1)-window-step+1)]
    target_list = [target[:,i] for i in range(window+step-1, src.size(1))]
    src_stack = torch.cat(src_list, dim=0)
    target_stack = torch.cat(target_list, dim=0).unsqueeze(1)
    return src_stack, target_stack

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # dataset = TimeSeriesDataset('../processed_data/nn_data/train')
    # print(len(dataset))
    # print(dataset[0][0].shape)
    # print(dataset[0][1])
    # net = TransformerEncoderRegressor(None, input_size=49).to(device='cuda:5')
    # src = torch.rand(655, 1, 49).to(device='cuda:5')  # (S, N, E) = (序列长度, 批次大小, 特征维度)
    # target = torch.rand(655, 1).to(device='cuda:5')  # (S, N) = (序列长度, 批次大小)
    # 前向传播
    # output = net(src)
    # metric_fn = lambda y_pred, y_true: ((y_pred - y_true) ** 2).mean()  # 均方误差
    # importance_scores = permutation_importance(net, src, target, metric_fn)
    # print("Permutation importance scores:", importance_scores)
    src = torch.rand(708,1).to(device='cuda:5')
    tgt = torch.rand(708,1).to(device='cuda:5')
    permutation_test(src, tgt, n_permutations=50, metric='mean')

