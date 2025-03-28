import torch
import os
from _3_0_dataset_utils import *
import math
from torch.optim import lr_scheduler
from tqdm import tqdm
import wandb
import argparse

def main(args, net):
    logging.info(args)
    # if there is a pretrained model, load it
    if os.path.exists(args.pretrained_weights) and args.pretrained:
        load_model(net, args.pretrained_weights)
        logging.info(f"load weights from {args.pretrained_weights}")

    net = net.to(args.device)
    train_dataset_path = args.train_dataset_path
    val_dataset_path = args.val_dataset_path
    
    train_dataset = TimeSeriesDataset(train_dataset_path, cache=args.cache)
    val_dataset = TimeSeriesDataset(val_dataset_path, cache=args.cache)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    pg = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.RMSprop(pg, lr=args.lr, weight_decay=args.l2_norm, momentum=0.9)

    def lf(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            return ((1 + math.cos((epoch - args.warmup_epochs) * math.pi / (args.epochs - args.warmup_epochs))) / 2) * (1 - args.lr) + args.lr
        
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    model_train_acc = []
    model_train_loss = []
    model_valid_acc = []
    model_valid_loss = []
    best_loss = float('inf')

    patience = 0 # early stop patience
    for epoch in range(args.epochs):
        train_losses = train_epoch(args, net, optimizer, train_loader, epoch, scheduler)
        model_train_loss.append(train_losses)

        valid_losses = valid_epoch(args, net, valid_loader, epoch)
        model_valid_loss.append(valid_losses)

        # select the best model
        if valid_losses < best_loss:
            best_loss = valid_losses
            logging.info(f'save the model in epoch{epoch+1}')
            torch.save(net.state_dict(), os.path.join(args.log_dir, f'best_model.pth'))
            patience = 0

def train_epoch(args, net, optimizer, train_loader, epoch, scheduler):
    train_loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]", unit="batch")
    train_losses = forward(args, 'train', net, train_loader, optimizer, scheduler)
    logging.info('train_loss:' + str(train_losses) + f' in epoch{epoch+1}')
    logging.info(f"Epoch {epoch+1}: Learning Rate: {scheduler.get_last_lr()}")
    if args.wandb:
        wandb.log({'train_loss': train_losses}, step=epoch)
        wandb.log({'learning_rate': optimizer.param_groups[0]['lr']}, step=epoch)

    return train_losses

@torch.no_grad()
def valid_epoch(args, net, valid_loader, epoch):
    valid_loader = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validation]", unit="batch")
    valid_losses = forward(args, 'eval', net, valid_loader)
    logging.info('valid_loss:' + str(valid_losses) + f' in epoch{epoch+1}')
    if args.wandb:
        wandb.log({'valid_loss': valid_losses}, step=epoch)

    return valid_losses

def forward(args, mode, net, loader, optimizer=None, scheduler=None):
    net.train() if mode == 'train' else net.eval()
    total_losses = 0
    evaluator = Evaluator(args.loss_list, args.loss_weight_list)
    for data_padded, target_padded, lengths in loader:
        data_truncated = data_padded[:, :lengths.item(), :].to(device=args.device)
        target = target_padded[:, :lengths.item()].to(device=args.device)

        src_stack, target_stack = get_data_window(args, data_truncated, target)

        y_pred = net(src_stack)
        y_pred_diff = y_pred[1:, :] - y_pred[:-1, :]
        target_stack_diff = target_stack[1:, :] - target_stack[:-1, :]
        loss = evaluator.loss(y_pred_diff, target_stack_diff)

        # loss = evaluator.loss(y_pred, target_stack)  # 计算损失
        total_losses = total_losses + loss.item()
        if mode == 'train':
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
  
    if mode == 'train':
        scheduler.step()

    return total_losses / len(loader)

if __name__ == '__main__':
    file_path = os.path.abspath(__file__)
    dir_path = os.path.dirname(file_path)
    os.chdir(dir_path)
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--device', default='cuda:7', type=str, help='device to use for training / testing')
    parser.add_argument('--max_channel', default=64, type=int, help='max channel')
    parser.add_argument('--loss_list', default=['rmse'], type=list, help='the loss function list')
    parser.add_argument('--loss_weight_list', default=[1], type=list, help='the loss weight list')
    parser.add_argument('--t_type', default='t2', type=str, help='the t_type')
    parser.add_argument('--senti', default=True, type=bool, help='whether to use sentiment')
    parser.add_argument('--window', default=7, type=int, help='the length of the rolling window')
    parser.add_argument('--model', default='transformer', type=str, help='architecture of the model')
    parser.add_argument('--patience', default=15, type=float, help='the patience of early stop')
    parser.add_argument('--wandb', default=True, type=bool, help='whether to use wandb')
    parser.add_argument('--cache', default=True, type=bool, help='whether to load data to cache')
    parser.add_argument('--batch_size', default=1, type=int, help='input batch size for training')
    parser.add_argument('--pretrained', default=False, type=bool, help='load pretrained model weights')
    parser.add_argument('--pretrained_weights', default='../logistics/train_hybrid_t1_02-19-20-07-44', type=str, help='pretrained weights path')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--l2_norm', default=1e-7, type=float, help='the coefficient of l2 norm')
    parser.add_argument('--epochs', default=50, type=int, help='training and testing epochs')
    parser.add_argument('--warmup_epochs', default=5, type=int, help='training warmup epochs')
    parser.add_argument('--train_dataset_path', default='../processed_data/nn_data/train_db4_6_sd2', type=str, help='training dataset path')
    parser.add_argument('--val_dataset_path', default='../processed_data/nn_data/val_db4_6_sd2', type=str, help='validation dataset path')
    args = parser.parse_args()
    args.log_dir = get_log_dir(args, train_eval='train_duration')
    args.log_file_path = os.path.join(args.log_dir, "training.log")
    setup_logging(args.log_file_path)
    if args.wandb:
        wandb.init(project='fin_senti', config=args, name=args.log_dir.split('/')[-2])
    net = build_model(args)
    @collate_decorator(t_type=args.t_type, senti=args.senti)
    def collate_fn(batch):
        pass
    main(args, net)