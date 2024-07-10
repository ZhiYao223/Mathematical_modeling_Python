#TOPSIS算法：逼近理想解排序法/优劣解距离法
'''
TOPSIS法是一种理想目标相似性的题序选优技术,
在多目标决策分析中是一种非常有效的方法。它通过归一化后(去量纲化)
的数据规范化矩阵，找出多个目标中最优目标和最劣目标(分别用理归想一解化和反理想解表示),
分别计算各评价目标与理想解和反理想解的距离,获得各目标与理想解的贴近度,
按理想解贴近度的大小排序,以此作为评价目标优劣的依据。贴近度取值在0~1之间,
该值愈接近1,表示相应的评价目标越接近最优水平:反之,该值愈接近0,表示评价目标越接近最劣水平,

基本步骤：
1.将原始矩阵正向化
将原始矩阵正向化，就是要将所有的指标类型统一转化为极大型指标。
2.正向矩阵标准化
其主要目的就是去除量纲的影响，保证不同评价指标在同一个数量级，
且数据大小排序不变。
3.计算得分并归一化
'''
'''
列表推导式（List Comprehension）  ans = [[(maxx-e)] for e in x]
列表推导式是一种简洁的语法，用于创建新的列表。它的基本形式如下：
[expression for item in iterable]
expression 是你希望应用于每个 item 的操作或计算。
item 是从 iterable（如列表、元组、集合等）中取出的每一个元素。
for e in x: 这部分表示从可迭代对象 x 中逐个取出元素，并将每个元素暂时命名为 e。
[(maxx - e)]: 这是对每个 e 进行的操作。在这里，对于每个 e，计算 maxx - e 的值，
并将结果放入一个新的单元素列表中（即 [(maxx - e)]）。
整个列表推导式生成一个包含这些新列表的列表。
'''
#Reshape函数是用于改变数组形状的函数，允许将数组重新组织成不同的形状，而不改变数组中的数据
#举例代码：
# import numpy as np
# #创建一个数组
# arr = np.array([1,2,3,4,5,6])
# # 将一维数组reshape成二维数组
# reshaped_arr = arr.reshape(2,3)
# print("原始数组：\n",arr)
# print("Reshaped 数组：\n",reshaped_arr)
#当reshpe(-1,3)前者为-1时，表示自动识别所需行数，列数为第二个值
#reshape(-1,1)代表把谁转换成列向量.

#np.hstack 是numpy库中的函数，用于在水平方向上沿着列，将多个数组堆叠在一起。
# import  numpy as np
# arr1 = np.array([[1],[2],[3]])
# arr2 = np.array([[4,],[5],[6]]) #创建两个一维数组
# stacked_arr = np.hstack((arr1,arr2))#将两数组水平堆叠
# print("数组1：\n",arr1)
# print("数组2：\n",arr2)
# print("水平堆叠后的数组：\n",stacked_arr)

import  numpy as np #导入numpy库
import pandas as pd
#   从用户输入中接收参评数目和指标数目，并将输入的字符串转换为数值
print("请输入参评数目：")
n = int(input())    #接收参评数目，使用数据类型转换
print("请输入指标数目：")
m = int(input())     #接收指标数目

#接收用户输入的类型矩阵，该矩阵指示了每个指标的类型(极大型、极小型)
print("请输入类型矩阵，1：极大型，2：极小型，3：中间型，4：区间型")
kind = input().split(" ") #将输入的字符串按空格分割，形成列表

#接收用户输入的矩阵并转化为numpy数组
print("请输入矩阵：")
A = np.zeros(shape = (n,m)) #初始化一个n行m列的全零矩阵A
for i in range(n):
    A[i] = input().split(" ")#接收每行输入的数据
    A[i] = list(map(float,A[i])) #将接收到的字符串列表转换为浮点数列表
print("输入矩阵为：\n{}".format(A)) #打印输入的矩阵A

#极小型指标转化为极大型指标的函数
def minTomax(maxx,x):
    x = list(x)#将输入的指标数据转换为列表
    #列表推导式，整体上，其作用是遍历列表x中的每个元素，对每个元素计算max-e的值，
    #并将结果构建成一个新的列表ans
    ans = [[(maxx-e)] for e in x] #计算最大值与每个指标值的差，并将其放入新列表中
    return np.array(ans)#将列表转换为numpy数组并返回

#中间型指标转化为极大型指标的函数
def midTomax(bestx,x): #bestx为中间型指标的最佳值
    x = list(x) #将输入的指标数据转换为列表
    h = [abs(e-bestx) for e in x] #计算每个指标值与最优值之间的绝对差
    M = max(h) #找到最大的差值
    if M == 0:
        M = 1 #防止最大差值为0的情况，M为分母，防止代码报错
    ans = [[(1-e/M)] for e in h]#计算每个差值占最大差值的比例，并从1中减去，得到新指标值
    return np.array(ans) #返回处理后的numpy数组

#区间型指标转化为极大型指标的函数
def regTomax(lowx,highx,x):
    x = list(x) #将输入的指标数据转换为列表
    M = max(lowx-min(x),max(x)-highx) #计算指标值超出区间的最大值
    if M == 0:
        M = 1 #防止最大距离为0的情况，因为要作分母
    ans = []
    for i in range(len(x)):
        if x[i] < lowx:
            ans.append([1-(lowx-x[i]/M)])#如果指标值小于下限，则计算其与下限的距离比例
        elif x[i]>highx:
            ans.append([(1-(x[i]-highx)/M)])#如果指标值大于上限，则计算其与上限的距离比例
        else:
            ans.append([1])#如果指标值在区间内，则直接取为1
    return np.array(ans) #返回处理后的numpy数组

#统一指标类型，将所有指标转化为极大型指标
X = np.zeros(shape=(n,1))
for i in range(m):
    if kind[i] == "1":#如果当前指标为极大值，则直接使用原值
        v = np.array(A[:,i])
    elif kind[i] == "2": #如果当前指标为极小值，调用minTomax函数转换
        maxA = max(A[:,i])
        v = minTomax(maxA,A[:,i])
    elif kind[i] == "3":#如果当前指标为中间型，调用midTomax函数转换
        print("类型三，请输入最优值：")
        bestA = eval(input()) #eval包含int与float两种格式
        v = midTomax(bestA,A[:,i])
    elif kind[i] == "4":#如果当前指标为区间型，调用regTomax函数转换
        print("类型四，请输入区间[a,b]值a：")
        lowA = eval(input())
        print("类型四，请输入区间[a,b]值b：")
        highA = eval(input())
        v = regTomax(lowA,highA,A[:,i])
    if i == 0:
        X = v.reshape(-1,1)#如果是第一个指标，直接替换X数组
    else:
        X = np.hstack([X, v.reshape(-1,1)])#如果不是第一个指标，则将新指标列拼接到X数组上
print("统一指标后矩阵为，\n{}".format(X)) #打印处理后的矩阵X

#对统一指标后的矩阵X进行标准化处理
X = X.astype('float') #确保X矩阵的数据类型为浮点型
for j in range(m):
    X[:,j] = X[:,j]/np.sqrt(sum(X[:,j]**2))#对每一列数据进行性归一化处理，即除以改列的欧几里得范数
print("标准化矩阵为：\n{}".format(X))#打印标准化后的矩阵X

#最大值与最小值距离的计算
x_max = np.max(X,axis=0)#计算标准化矩阵每列的最大值
x_min = np.min(X,axis=0)#计算标准化矩阵每列的最小值
d_z = np.sqrt(np.sum(np.square((X-np.tile(x_max,(n,1)))),axis = 1)) #计算每个参评对象与最优情况的距离d+
d_f = np.sqrt(np.sum(np.square((X-np.tile(x_min,(n,1)))),axis = 1)) #计算每个参评对象与最劣情况的距离d-
print("每个指标的最大值：", x_max)
print("每个指标的最小值：", x_min)
print("d+向量：",d_z)
print("d-向量：",d_f)

#计算每个参评对象的评分排名
s = d_f/(d_z+d_f)#根据d+和d-计算评分s，其中s接近于1则表示较优，接近于0则表示较劣
Score = 100*s/sum(s) #将得分s转化为百分制，便于比较
for i in range(len(Score)):
    print(f"第{i+1}个标准化后百分制得分为：{Score[i]}") #打印每个参评对象的得分



