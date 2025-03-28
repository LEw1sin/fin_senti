import torch
import os
from _3_0_dataset_utils import *
from tqdm import tqdm
import argparse
import shap

# feature_dict = {0: '中间价:美元兑人民币', 1: 'Shibor:3月', 2: '制造业PMI', 3: '宏观经济景气指数:先行指数', 4: 'PPI:当月同比', 5: 'CPI:当月同比', 6: 'GDP:不变价:当季同比', 7: '社会融资规模存量:期末同比', 8: '所属申万一级行业指数', 9: '债券分类违约概率', 10: '区域违约概率', 11: '营业收入', 12: '营业成本', 13: '利润总额', 14: '流动资产', 15: '非流动资产', 16: '资产总计', 17: '流动负债', 18: '非流动负债', 19: '负债合计', 20: '股东权益合计', 21: '经营活动现金流', 22: '投资活动现金流', 23: '筹资活动现金流', 24: '总现金流', 25: '流动比率', 26: '速动比率', 27: '超速动比率', 28: '资产负债率(%)', 29: '产权比率(%)', 30: '有形净值债务率(%)', 31: '销售毛利率(%)', 32: '销售净利率(%)', 33: '资产净利率(%)', 34: '营业利润率(%)', 35: '平均净资产收益率(%)', 36: '营运周期(天)', 37: '存货周转率', 38: '应收账款周转率', 39: '流动资产周转率', 40: '股东权益周转率', 41: '总资产周转率', 42: '授信剩余率', 43: '授信环比变动', 44: '担保授信比', 45: '同期国债利率', 46: '成交量', 47: '剩余期限', 48: '到期收益率', 49: '风险价差', 50: 'sentiment'}
feature_dict = {
    0: '中间价:美元兑人民币',
    1: 'Shibor:3月',
    2: '制造业PMI',
    3: '宏观经济景气指数:先行指数',
    4: 'PPI:当月同比',
    5: 'CPI:当月同比',
    6: 'GDP:不变价:当季同比',
    7: '社会融资规模存量:期末同比',
    8: '所属申万一级行业指数',
    9: '营业收入',
    10: '营业成本',
    11: '利润总额',
    12: '流动资产',
    13: '非流动资产',
    14: '资产总计',
    15: '流动负债',
    16: '非流动负债',
    17: '负债合计',
    18: '股东权益合计',
    19: '经营活动现金流',
    20: '投资活动现金流',
    21: '筹资活动现金流',
    22: '总现金流',
    23: '流动比率',
    24: '速动比率',
    25: '超速动比率',
    26: '资产负债率(%)',
    27: '产权比率(%)',
    28: '有形净值债务率(%)',
    29: '销售毛利率(%)',
    30: '销售净利率(%)',
    31: '资产净利率(%)',
    32: '营业利润率(%)',
    33: '平均净资产收益率(%)',
    34: '营运周期(天)',
    35: '存货周转率',
    36: '应收账款周转率',
    37: '流动资产周转率',
    38: '股东权益周转率',
    39: '总资产周转率',
    40: '授信剩余率',
    41: '授信环比变动',
    42: '担保授信比',
    43: '同期国债利率',
    44: '成交量',
    45: '风险价差',
    46: 'sentiment'
}
def print_importance(args,feature_importance):
    feature_importance_with_names = [(feature_dict[i], importance.item()) for i, importance in enumerate(feature_importance)]
    if args.senti:
        last_key = max(feature_dict.keys())  # 获取字典的最后一个键
        feature_importance_with_names[-1] = (feature_dict[last_key], feature_importance_with_names[-1][1])
    feature_importance_with_names_sorted = sorted(feature_importance_with_names, key=lambda x: x[1], reverse=True)
    for i, (name, importance) in enumerate(feature_importance_with_names_sorted):
        logging.info(f"{i}  {name}: {importance}")

def main(args, net, weights_path):
    logging.info(args)
    load_model(net, weights_path)
    logging.info(f"load weights from {weights_path}")

    net = net.to(args.device)
    net.eval()
    # net.train()
    test_dataset_path = args.test_dataset_path
    test_dataset = TimeSeriesDataset(test_dataset_path, cache=args.cache)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    metric_fn = lambda y_pred, y_true: torch.sqrt(F.mse_loss(y_pred, y_true))  # rmse
    evaluator = Evaluator()
    mse, mae, rmse, mape, importance_scores_shap, importance_scores_permutation = 0, 0, 0, 0, 0, 0
    pred_list = []

    with torch.no_grad():
        for data_padded, target_padded, lengths in tqdm(test_loader):
            data_truncated = data_padded[:, :lengths.item(), :].to(device=args.device)
            target = target_padded[:, :lengths.item()].to(device=args.device)
            src_stack, target_stack = get_data_window(args, data_truncated, target)
            batch_size = src_stack.size(0)

            # explainer = shap.DeepExplainer(net, src_stack)
            # shap_values = explainer.shap_values(src_stack)
            y_pred = net(src_stack)
            importance_scores_i_permutation = permutation_importance(net, src_stack, target_stack, metric_fn)
            mse_i = evaluator.mse(y_pred, target_stack)
            mae_i = evaluator.mae(y_pred, target_stack)
            rmse_i = evaluator.rmse(y_pred, target_stack)
            mape_i = evaluator.mape(y_pred, target_stack)

            mse += mse_i/batch_size
            mae += mae_i/batch_size
            rmse += rmse_i/batch_size
            mape += mape_i/batch_size
            # importance_scores_shap += shap_values
            importance_scores_permutation += importance_scores_i_permutation

            pred_list.append(y_pred)

    mse /= len(test_loader)
    mae /= len(test_loader)
    rmse /= len(test_loader)
    mape /= len(test_loader)
    # importance_scores_shap /= len(test_loader)
    importance_scores_permutation /= len(test_loader)

    # logging.info(f"mse: {mse}, mae: {mae}, rmse: {rmse}, mape: {mape}")
    # logging.info("SHAP Feature Importance:")
    # print_importance(args, importance_scores_shap)

    logging.info("Permutation-based Feature Importance:")
    print_importance(args, importance_scores_permutation)

    return pred_list, mse, mae, rmse, mape

if __name__ == '__main__':
    file_path = os.path.abspath(__file__)
    dir_path = os.path.dirname(file_path)
    os.chdir(dir_path)
    parser = argparse.ArgumentParser(description="Validation")
    parser.add_argument('--device', default='cuda:3', type=str, help='device to use for training / testing')
    parser.add_argument('--net_weights_wo_senti', default='../logistics/train_transformer_t2_7_rmse_03-13-12-41-06', type=str, help='pretrained weights path wo senti')
    parser.add_argument('--net_weights_w_senti', default='../logistics/train_micro_meso_transformer_t2_7_rmse_senti_03-21-13-04-41', type=str, help='pretrained weights path w senti')
    parser.add_argument('--t_type', default='t2', type=str, help='the t_type')
    parser.add_argument('--senti', default=True, type=bool, help='whether to use sentiment')
    parser.add_argument('--model', default='transformer', type=str, help='architecture of the model')
    parser.add_argument('--max_channel', default=64, type=int, help='max channel')
    parser.add_argument('--window', default=7, type=int, help='the length of the rolling window')
    parser.add_argument('--cache', default=True, type=bool, help='whether to load data to cache')
    parser.add_argument('--loss_list', default=['rmse'], type=list, help='the loss function list')
    parser.add_argument('--test_dataset_path', default='../processed_data/nn_data/test_db4_6_sd2', type=str, help='test dataset path')
    args = parser.parse_args()
    args.log_dir = get_log_dir(args, train_eval='eval')
    args.log_file_path = os.path.join(args.log_dir, "eval.log")
    setup_logging(args.log_file_path)

    args.senti = False
    net_wo_senti = build_model(args)
    @collate_decorator(t_type=args.t_type, senti=False)
    def collate_fn(batch):
        pass
    y_wo_senti_list, mse_wo_senti, mae_wo_senti, rmse_wo_senti, mape_wo_senti = main(args, net_wo_senti, args.net_weights_wo_senti)

    args.senti = True
    net_w_senti = build_model(args)
    @collate_decorator(t_type=args.t_type, senti=True)
    def collate_fn(batch):
        pass
    y_w_senti_list, mse_w_senti, mae_w_senti, rmse_w_senti, mape_w_senti = main(args, net_w_senti, args.net_weights_w_senti)

    logging.info(f"wo_mse: {mse_wo_senti}, wo_mae: {mae_wo_senti}, wo_rmse: {rmse_wo_senti}, wo_mape: {mape_wo_senti}")
    logging.info(f"w_mse: {mse_w_senti}, w_mae: {mae_w_senti}, w_rmse: {rmse_w_senti}, w_mape: {mape_w_senti}")
    logging.info(f"delta mse: {mse_wo_senti - mse_w_senti}, delta mae: {mae_wo_senti - mae_w_senti}, delta rmse: {rmse_wo_senti - rmse_w_senti}, delta mape: {mape_wo_senti - mape_w_senti}")
    logging.info(f"delta persent mse : {(mse_wo_senti - mse_w_senti) * 100 / mse_wo_senti}, delta persent mae : {(mae_wo_senti - mae_w_senti) * 100/ mae_wo_senti}, delta persent rmse : {(rmse_wo_senti - rmse_w_senti) * 100/ rmse_wo_senti}, delta persent mape : {(mape_wo_senti - mape_w_senti) * 100/ mape_wo_senti}")

    # t_stat, p_value = 0, 0
    # for y_wo_senti, y_w_senti in zip(y_wo_senti_list, y_w_senti_list):
    #     t_stat_i, p_value_i = independent_t_test(y_wo_senti, y_w_senti)
    #     t_stat += t_stat_i
    #     p_value += p_value_i
    # t_stat /= len(y_wo_senti_list)
    # p_value /= len(y_wo_senti_list)
    # logging.info(f"t-statistic: {t_stat}, p-value: {p_value}")
    p_value = 0
    for y_wo_senti, y_w_senti in tqdm(zip(y_wo_senti_list, y_w_senti_list), total=len(y_wo_senti_list), desc="Calculating p-value"):
        p_value_i = permutation_test(y_wo_senti, y_w_senti, n_permutations=50)
        p_value += p_value_i

    p_value /= len(y_wo_senti_list)
    logging.info(f"p-value: {p_value}")

    if p_value < 0.05:
        logging.info("The difference is significant")
    else:
        logging.info("The difference is not significant")