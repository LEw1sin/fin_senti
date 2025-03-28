import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL

time_series=np.load('../processed_data/industry_dict_sum_pre-spline3.npy', allow_pickle=True).item()


time_series_industry = time_series['汽车']

# 转换为 Pandas Series（时间索引可选）
df = pd.Series(time_series_industry, index=pd.date_range(start="2013-01-01", periods=len(time_series_industry), freq="D"))

stl = STL(df)
res = stl.fit()
fig = res.plot()

ax_resid = fig.axes[-1]  # Resid 子图通常是最后一个
for line in ax_resid.get_lines():
    line.set_markersize(1)  


ax_seasonal = fig.axes[-2]  # Resid 是最后一个，Seasonal 是倒数第二个
for line in ax_seasonal.get_lines():
    line.set_linewidth(0.5)  

colors = ['#D9958F', '#3B7D23', '#589BFF', '#7030A0']  # 蓝色、橙色、绿色、红色
axes = fig.axes  
for ax, color in zip(axes, colors):
    for line in ax.get_lines():  # 修改该子图中的所有线条颜色
        line.set_color(color)

plt.savefig('decomposition.svg')
plt.show()