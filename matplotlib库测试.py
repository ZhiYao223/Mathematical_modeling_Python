import numpy as np
import matplotlib.pyplot as plt

#中文转化
from matplotlib.pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus']=False


x = np.linspace(0,10,10)
print(x)
y = np.sin(x)

x2 = np.linspace(0,10,100)
y2 = np.sin(x2)

# plt.plot(x,y)
# plt.title("y = sin(x)")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()

plt.scatter(x,y,marker = '*',c='r',label="数据点")
plt.plot(x2,y2,linestyle = '--',label='折线')
plt.legend()
plt.show()
