import pandas as pd
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

data = pd.read_csv("../processed_data/bond_data_normalized_w_senti_shift.csv")

df = pd.DataFrame(data)

feature_mapping_corrected = {
    '中间价:美元兑人民币': 'USDCNYC',
    'Shibor:3月': 'Shibor',
    '制造业PMI': "Manufacturing PMI",
    '宏观经济景气指数:先行指数': 'Macroeconomic Climate Index',
    'PPI:当月同比': 'PPI',
    'GDP:不变价:当季同比': 'GDP',
    'CPI:当月同比': 'CPI',
    '社会融资规模存量:期末同比': 'AFRE',
    '同期国债利率': 'Yield on Government Bonds',
    '所属申万一级行业指数': 'SWS Primary Industry Index',
    '成交量': 'Trading Volume',
    '营业收入': 'Operating Revenue',
    '营业成本': 'Operating Costs',
    '利润总额': 'Total Profit',
    '流动资产': 'Current Assets',
    '非流动资产': 'Non-Current Assets',
    '资产总计': 'Total Assets',
    '流动负债': 'Current Liabilities',
    '非流动负债': 'Non-Current Liabilities',
    '负债合计': 'Total Liabilities',
    '股东权益合计': 'Total Shareholders’ Equity',
    '经营活动现金流': 'Cash Flow from Operations',
    '投资活动现金流': 'Cash Flow from Investment',
    '筹资活动现金流': 'Cash Flow from Finance',
    '总现金流': 'Total Cash Flow',
    '流动比率': 'Current Ratio',
    '速动比率': 'Quick Ratio',
    '超速动比率': 'Super Quick Ratio',
    '资产负债率(%)': 'Debt-to-Asset Ratio',
    '产权比率(%)': 'Equity Ratio',
    '有形净值债务率(%)': 'Tangible Net Worth Debt Ratio',
    '销售毛利率(%)': 'Gross Profit Margin',
    '销售净利率(%)': 'Net Profit Margin',
    '资产净利率(%)': 'Return on Assets',
    '营业利润率(%)': 'Operating Profit Margin',
    '平均净资产收益率(%)': 'Average Return on Equity',
    '营运周期(天)': 'Operating Cycle',
    '存货周转率': 'Inventory Turnover Ratio',
    '应收账款周转率': 'Accounts Receivable Turnover Ratio',
    '流动资产周转率': 'Current Asset Turnover Ratio',
    '股东权益周转率': 'Shareholders’ Equity Turnover Ratio',
    '总资产周转率': 'Total Asset Turnover Ratio',
    '授信剩余率': 'Remaining Credit Utilization Ratio',
    '授信环比变动': 'Month-over-Month Change in Credit',
    '担保授信比': 'Secured Credit Ratio',
    'sentiment': 'Sentiment'
}

import seaborn as sns
import matplotlib.pyplot as plt


selected_columns = list(df.columns[2:47]) + ['sentiment']

# 过滤 DataFrame 只保留所选列
selected_df = df[selected_columns]

selected_df.rename(columns=feature_mapping_corrected, inplace=True)

# 计算相关性矩阵
correlation_matrix = selected_df.corr()

# 可视化相关性矩阵
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", linewidths=0.5)

# 旋转 X 轴标签 180° 并居中
plt.xticks(rotation=270)

# 旋转 Y 轴标签 0°
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig("correlation_matrix.pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)

plt.show()
