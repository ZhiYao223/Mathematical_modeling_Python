'''
print
'''
# a = input("输入一个数：")
# if a == 2:
#     print(a)
#     print("Ture")
#     print("a = 1")
# print("Go!")


# import numpy as np
# np.random.seed(123444) #固定随机种子
# print(np.random.rand())#0-1之间随机浮点
# arr = np.random.randint(0,100,16).reshape(4,4)
# print(np.sum(arr[arr<=10]))
#
#
# import LearnPandass as pd
#
# # 创建数据框
# data = {
#     '学生': ['小明', '小红', '小刚'],
#     '数学': [85, 79, 93],
#     '英语': [78, 88, 81],
#     '科学': [92, 84, 76]
# }
#
# df = pd.DataFrame(data)
#
# # 计算每个学生的平均成绩
# df['平均成绩'] = df[['数学', '英语', '科学']].mean(axis=1)
#
# # 打印数据框
# print(df)

import pandas as pd

data = {
    'name': ['John', 'Anna', 'Peter', 'Linda'],
    'age':  [28, 24, 35, 32],
    'gender': ['M', 'F', 'M', 'F']
}

df = pd.DataFrame(data)
print(df)

#从Numpy数组创建
data = [
    ['John', 28, 'M'],
    ['Anna', 24, 'F'],
    ['Peter', 35, 'M'],
    ['Linda', 32, 'F']
]
df = pd.DataFrame(data,columns=['name','age','gender'])
print(df)

#从CSV文件读取
df = pd.read_csv('data.csv')
print(df)

#访问某一列
print(df['name'])
#访问某几列
print(df[['name','age']])
#访问某一行
print(df.loc[0]) #使用行标签
print(df.iloc[0])#使用行位置

#按条件过滤
filtered_df = df[df['age'] > 30]
print(filtered_df)

#计算平均值
mean_age = df['age'].mean()
print(mean_age)
#分组聚合
grouped = df.groupby('gender').mean()
print(grouped)

#转置
print(df.T)
#改变形状
reshaped_df = df.melt(id_vars=['name'],value_vars=['age','gender'])
print(reshaped_df)
