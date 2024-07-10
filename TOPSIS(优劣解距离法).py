import numpy as np
import pandas as pd

data = pd.DataFrame(
    {'人均专著': [0.1, 0.2, 0.4, 0.9, 1.2],
     '生师比': [5, 6, 7, 10, 2],
     '科研经费': [5000, 6000, 7000, 10000, 400],
     '逾期毕业率': [4.7, 5.6, 6.7, 2.3, 1.8]},
    index=['院校' + i for i in list('ABCDE')])

'''
1.指标正向化
2.结合公式理解代码
3.需要根据题目选择具体的处理方法
'''

# 极小型转为极大型指标
def dataDirection_1(datas, offset=0):
    def normalization(data):
        return 1 / (data + offset)

    return list(map(normalization, datas))

# 中间型指标转为极大型指标
def dataDirection_2(datas, x_min, x_max):
    def normalization(data):
        if data <= x_min or data >= x_max:
            return 0
        elif data > x_min and data < (x_min + x_max) / 2:
            return 2 * (data - x_min) / (x_max - x_min)
        elif data < x_max and data >= (x_min + x_max) / 2:
            return 2 * (x_max - data) / (x_max - x_min)

    return list(map(normalization, datas))

# 区间型指标转为极大型指标
# [x_min, x_max]最佳稳定区间, [x_minimum, x_maximum]容忍区间
def dataDirection_3(datas, x_min, x_max, x_minimum, x_maximum):
    def normalization(data):
        if data >= x_min and data <= x_max:
            return 1
        elif data <= x_minimum or data >= x_maximum:
            return 0
        elif data > x_max and data < x_maximum:
            return 1 - (data - x_max) / (x_maximum - x_max)
        elif data < x_min and data > x_minimum:
            return 1 - (x_min - data) / (x_min - x_minimum)

    return list(map(normalization, datas))

# 极小型指标转为极大型
minimum_list = dataDirection_1(data.loc[:, "逾期毕业率"])
minimum_array = np.array(minimum_list)
minimum_4f = np.round(minimum_array, 6)
print(minimum_4f)

# 区间型指标转为极大型
maximum_list = dataDirection_3(data.loc[:, "生师比"], 5, 6, 2, 12)
maximum_array = np.array(maximum_list)
maximum_4f = np.round(maximum_array, 6)
print(maximum_4f)

# 结果：
# [0.212766 0.178571 0.149254 0.434783 0.555556]
# [1.       1.       0.833333 0.333333 0.]

# 指标正向化结果
index_Isotropy = pd.DataFrame()
index_Isotropy["人均专著"] = data["人均专著"]
index_Isotropy["生师比"] = maximum_4f
index_Isotropy["科研经费"] = data["科研经费"]
index_Isotropy["逾期毕业率"] = minimum_4f
print(index_Isotropy)

# 这里采用归一化，数据处于0-1之间，便于可视化
data_normalization = index_Isotropy / np.sqrt((index_Isotropy ** 2).sum())
print(data_normalization)

def entropyWeight(data):
    data = np.array(data)
    # 计算第j个指标下第i个样本所占的比重，相对熵计算中用到的概率
    P = data / data.sum(axis=0)  # 压缩行
    # 计算熵值
    E = np.nansum(-P * np.log(P) / np.log(len(data)), axis=0)
    # 信息效用值
    d = (1 - E)
    # 计算权系数
    W = d / d.sum()
    return W
entropyWeight(data)


def topsis(data, weight=None):
    # 最优最劣方案（最大值Z^+ 和 最小值）
    Z = pd.DataFrame([data.max(), data.min()], index=['正理想解', '负理想解'])

    # 距离
    weight = entropyWeight(data) if weight is None else np.array(weight)
    Result = data.copy()
    Result['正理想解'] = np.sqrt(((data - Z.loc['正理想解']) ** 2 * weight).sum(axis=1))  # 评价对象与最大值的距离
    Result['负理想解'] = np.sqrt(((data - Z.loc['负理想解']) ** 2 * weight).sum(axis=1))

    # 综合得分指数
    Result['综合得分指数'] = Result['负理想解'] / (Result['负理想解'] + Result['正理想解'])
    Result['排序'] = Result.rank(ascending=False)['综合得分指数']

    return Result, Z, weight

# # 人工赋权重的结果
weight = [0.2, 0.3, 0.4, 0.1]
Result, Z, weight = topsis(data_normalization, weight)

# 把归一化的结果列的顺序改一下，把‘生师比’放第一个，便于下图展示，不改也可以
data_normalization=data_normalization[['生师比','人均专著',  '逾期毕业率', '科研经费']]
#data_normalization

from math import pi
import matplotlib.pyplot as plt
import matplotlib as mpl
# 全局设置字体
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['SimSun']  # 设置为黑体，适用于中文
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 目标数量
categories = list(data_normalization)[0:]
N = len(categories)

# 角度
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# 绘图初始化
plt.figure(dpi=150)
plt.style.use('ggplot')
ax = plt.subplot(111, polar=True)


# 设置第一处
ax.set_theta_offset(pi / 2)  # 设置最上面为0°
ax.set_theta_direction(-1)  # 设置正方形为顺时针

# 添加背景信息
plt.title("研究生院试评估")
plt.xticks(angles[:-1], categories)  # 改变轴标签rotation = 30
plt.xticks(rotation=pi / 4, fontsize=6)
ax.set_rlabel_position(30)  # 极径标签显示位置
# ax.set_rgrids(np.arange(0.1,0.9,0.1))
plt.yticks(np.arange(0.1, 0.9, 0.1), ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8"], color="grey",
           size=5)  # 设置极径标签区间
plt.ylim(0, 0.8)  # 设置极径范围

# 添加数据图

# 第一个
values = data_normalization.loc["院校A"].values.flatten().tolist()
values += values[:1]  # 首尾相连
ax.plot(angles, values, linewidth=1, linestyle='solid', color='#d62728', marker='.', markersize=4, label="院校A")
ax.fill(angles, values, '#d62728', alpha=0.2)  # 填充线条

# 第二个
values = data_normalization.loc["院校B"].values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', color='#1f77b4', marker='.', markersize=4, label="院校B")
ax.fill(angles, values, '#1f77b4', alpha=0.2)

# 第三个
values = data_normalization.loc["院校C"].values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', color='#9467bd', marker='.', markersize=4, label="院校C")
ax.fill(angles, values, '#9467bd', alpha=0.2)

# 第四个
values = data_normalization.loc["院校D"].values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', color='#7f7f7f', marker='.', markersize=4, label="院校D")
ax.fill(angles, values, '#7f7f7f', alpha=0.2)

# 第五个
values = data_normalization.loc["院校E"].values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', color='#ff7f0e', marker='.', markersize=4, label="院校E")
ax.fill(angles, values, '#ff7f0e', alpha=0.2)

# 最优解
values = Z.loc["正理想解"].values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', color='#2ca02c', marker='.', markersize=4, label="最优解")
ax.fill(angles, values, '#2ca02c', alpha=0.2)

# 最劣解
values = Z.loc["负理想解"].values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', color='#e377c2', marker='.', markersize=4, label="最劣解")
ax.fill(angles, values, '#e377c2', alpha=0.2)

# 添加图例
plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1))

# 显示
plt.show()