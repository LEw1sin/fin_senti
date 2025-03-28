import json

with open("../processed_data/company_sentiment_db4_6_sd.json", 'r', encoding='utf-8') as file:
    std_data = json.load(file)

with open("../processed_data/company_sentiment_only_meso_db4_6_sd.json", 'r', encoding='utf-8') as file:
    wo_meso_data = json.load(file)

with open("../processed_data/company_sentiment_only_micro_db4_6_sd.json", 'r', encoding='utf-8') as file:
    wo_micro_data = json.load(file)

with open("../processed_data/company_sentiment_wo_smooth_db4_6_sd.json", 'r', encoding='utf-8') as file:
    wo_spline_data = json.load(file)

company_short = '宁德时代'

company_std_data = std_data[company_short]
company_wo_meso_data = wo_meso_data[company_short]
company_wo_micro_data = wo_micro_data[company_short]
company_wo_spline_data = wo_spline_data[company_short]

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 将字典转换为 Pandas DataFrame
df = pd.DataFrame({
    "Standard": company_std_data,
    "w/o Meso-Level": company_wo_meso_data,
    "w/o Micro-Level": company_wo_micro_data,
    "w/o Spline Smoothing": company_wo_spline_data,

}).T  # 转置使日期成为列

df = df.T  # 转置回来，使日期成为索引

import numpy as np

# 绘制折线图
plt.figure(figsize=(12, 6))

plt.plot(df.index, df['Standard'], label='All-inclusive Factors', linewidth=1.5, alpha=0.8)
plt.plot(df.index, df['w/o Meso-Level'], label='Only Meso-level Sentiment', linewidth=1.5, alpha=0.5)
plt.plot(df.index, df['w/o Micro-Level'], label='Only Micro-level Sentiment', linewidth=1.5, alpha=0.5)
plt.plot(df.index, df['w/o Spline Smoothing'], label='w/o Wavelet Smoothing', linewidth=1.5, alpha=0.1)


years = np.arange(2013, 2024)
days_per_year = [365 + (1 if year in [2016, 2020] else 0) for year in years]
xticks_start_pos = np.cumsum([0] + days_per_year[:-1])

plt.xticks(xticks_start_pos, years)

# 设置图表信息
plt.xlabel("year", fontsize=16)
plt.ylabel("Sentiment", fontsize=16)
plt.legend()

plt.grid(True)
plt.savefig('company_line.svg')
# 显示图像
plt.show()