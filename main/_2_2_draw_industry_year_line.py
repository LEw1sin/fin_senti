import pandas as pd

# Define the industries and their English translations
industries = {
    "农林牧渔": "Agriculture, Forestry, Livestock, and Fishery",
    "基础化工": "Basic Chemicals",
    "钢铁": "Steel",
    "有色金属": "Non-ferrous Metals",
    "电子": "Electronics",
    "汽车": "Automobile",
    "家用电器": "Household Appliances",
    "食品饮料": "Food and Beverage",
    "纺织服饰": "Textiles and Apparel",
    "轻工制造": "Light Industry Manufacturing",
    "医药生物": "Pharmaceuticals and Biotechnology",
    "公用事业": "Utilities",
    "交通运输": "Transportation",
    "房地产": "Real Estate",
    "商贸零售": "Trade and Retail",
    "旅游及景区": "Tourism and Scenic Areas",
    "教育（含体育）": "Education (Including Sports)",
    "本地生活服务": "Local Life Services",
    "专业服务": "Professional Services",
    "酒店餐饮": "Hospitality and Catering",
    "银行": "Banking",
    "非银金融": "Non-bank Financial Services",
    "建筑材料": "Building Materials",
    "建筑装饰": "Building Decoration",
    "电力设备": "Electrical Equipment",
    "机械设备": "Machinery and Equipment",
    "国防军工": "Defense and Military Industry",
    "计算机": "Computer",
    "电视广播": "Television and Broadcasting",
    "游戏": "Gaming",
    "广告营销": "Advertising and Marketing",
    "影视院线": "Film and Cinema",
    "数字媒体": "Digital Media",
    "社交": "Social Media",
    "出版": "Publishing",
    "通信": "Telecommunications",
    "煤炭": "Coal",
    "石油石化": "Petroleum and Petrochemicals",
    "环保": "Environmental Protection",
    "美容护理": "Beauty and Personal Care"
}

# Convert dictionary to DataFrame
df_industries = pd.DataFrame(list(industries.items()), columns=["行业(中文)", "Industry(English)"])

# Save to Excel
excel_path = '../processed_data/industries_translation.xlsx'
df_industries.to_excel(excel_path, index=False)

import pandas as pd

df = pd.read_excel('../processed_data/industries_translation.xlsx')

# Convert DataFrame to dictionary
industries_dict = pd.Series(df_industries["Industry(English)"].values, index=df_industries["行业(中文)"]).to_dict()

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# 创建示例数据
data = np.load('../processed_data/industry_dict_sum_pre-spline2.npy', allow_pickle=True).item()

years = np.arange(2013, 2024)
days_per_year = [365 + (1 if year in [2016, 2020] else 0) for year in years]
xticks_start_pos = np.cumsum([0] + days_per_year[:-1])

fig, ax = plt.subplots(figsize=(120, 60))

space = 2
total_rows = (1 + space) * len(data)
heatmap_data = np.full((total_rows, 4017), np.nan)

yticks_positions = []
yticks_labels = []

all_values = []

# 提取所有值用于统一归一化
for key, values in data.items():
    all_values.append(values)

all_values = np.array(all_values)
flattened_values = all_values.flatten()
mean_value = flattened_values.mean()
std_value = flattened_values.std()

# 控制颜色过渡程度的超参数 sigma，越大越平滑
sigma = 3  # 可调整，推荐范围：2-20

for idx, (key, values) in enumerate(data.items()):
    # 归一化为z-score
    z_values = (values - mean_value) / std_value
    # z_values = values

    # 在每根热力条内部应用高斯平滑
    smooth_values = gaussian_filter1d(z_values, sigma=sigma)

    row_position = idx * (space + 1)
    heatmap_data[row_position, :] = smooth_values

    yticks_positions.append(row_position)
    key_en = industries_dict[key]
    yticks_labels.append(key_en)

cax = ax.imshow(heatmap_data, aspect='auto', cmap='bwr', interpolation='None', vmin=-1, vmax=1)

ax.set_xticks(xticks_start_pos)
ax.set_xticklabels(years, fontsize=72)
ax.set_xlim(0, heatmap_data.shape[1])

ax.set_yticks(yticks_positions)
ax.set_yticklabels(yticks_labels, fontsize=48)

# 去除上方和右方边界
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

cbar = fig.colorbar(cax, orientation='vertical', ax=ax, fraction=0.02, pad=0.02)
cbar.ax.tick_params(labelsize=72)
cbar.outline.set_visible(False)

plt.tight_layout()
plt.savefig('year_line3.svg')
plt.show()
