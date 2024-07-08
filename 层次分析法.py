'''
层次分析法传送门：https://blog.csdn.net/qq_25990967/article/details/122820595

'''
import  numpy as np
import pandas as pd
p = np.matrix('8 7 6 8;7 8 8 7') #每一行代表一个对象的指标评分
print(p)

#A为自己构造的输入判别矩阵
A = np.array([[1,3,1,1/3],[1/3,1,1/2,1/5],[1,2,1,1/3],[3,5,3,1]])
print(A)

#查看行数和列数
[rows,cols] = A.shape
print(rows,cols)
n = cols

#求特征向量
V,D = np.linalg.eig(A)
print("特征值：\n",V)
print("特征向量：\n",D)

#找到最大特征值和最大特征向量
max_eigen_value = np.max(V)
print("最大特征值：\n",max_eigen_value)

#最大特征向量
k = [i for i in range(len(V)) if V[i] == np.max(V)]
max_eigenvector = -D[:,k]
print("最大特征向量：\n",max_eigenvector)

#计算权重
weight = np.zeros((n,1))
for i in range(0,n):
        weight[i] = max_eigenvector[i]/np.sum(max_eigenvector)
Q = weight
#print(weight)

#一致性检验
CI = (max_eigen_value - n)/(n - 1)
RI = [0.000001,0.0000001,0.58,0.9,1.12,1.24,1.32,1.41,1.45,1.49,1.52,1.54,1.56,1.58,1.59]

#判断是否通过一致性检验
CR = CI/RI[n-1]
print("CR = ",CR)
if CR >= 0.1:
        print('没有通过一致性检验\n')
else:
        print('通过一致性检验\n')

#计算评分 显示出所有评分对象的评分值
score = p*weight
for i in range(len(score)):
        print('object_score {}:'.format(i),float(score[i]))


