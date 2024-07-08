import numpy as np
#数组的创建
arr = np.array([1,2,3,4,5])
print(arr)
print(type(arr))

arr = np.array([[1,2,3],[2,4,5],[2,3,4],[5,6,0]])
print(arr)
print(type(arr))
print("数组形状",arr.shape) #四行三列(4,3)

#索引和切片
print(arr[0]) #第一行
print(arr[0:3])#前三行
print(arr[1][2]) #第二行第三个元素5

#运算 对应位置上的元素加减乘除
print([1,2,3]+[4,5,6])
print(np.array([1,2,3])+np.array([4,5,6]))
print(np.array([1,2,3])*np.array([4,5,6]))

#数组形状操作
arr = np.array([[1,2,3],[2,4,5],[2,3,4],[5,6,0]])
print(arr)
print("数组形状是：",arr.shape)
new_arr = arr.reshape(2,6)
print(new_arr)
print("新的数组形状是:",new_arr.shape)
print(new_arr)
new_arr_T = new_arr.transpose()
print("新的数组转置的形状是：\n",new_arr_T)

#线性代数 统计
arr1 = np.array([1,2,3])
arr2 = np.array([4,5,6])
arr1_dot_arr2 = np.dot(arr1,arr2)
print(arr1_dot_arr2)
print("一维数组的平均值：",arr1.mean())

arr = np.array([[3,1,2],[2,4,3],[5,4,3],[6,4,5]])
print("二维数组的平均值：",arr.mean())
print("二维数组的平均值：",np.mean(arr))
print("数组最大值：",arr.max())
print("数组最小值：",arr.min())
print("数组最大值：",np.max(arr))
print("数组的标准差：",arr.std())
print("数组的和：",arr.sum())
print("数组的排序\n",np.sort(arr))
print("数组的排序：",np.sort(arr.reshape(-1)))# 一行排序
print(arr > 10)
print(arr[arr < 3])
print(arr[(arr > 3) & (arr < 5)])

#保存和导入
np.save("arr",arr)

import numpy as np
print("显示导入的数组：")
arr = np.load("arr.npy")
print(arr)



