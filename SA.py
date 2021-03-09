"""
模拟退火算法(Simulated annealing algorithm ，SA)

"""

import numpy as np
import matplotlib.pyplot as plt
import pdb
import pandas as pd
import math
import time

dataframe = pd.read_csv("./data/TSP10cities.tsp",sep=" ",header=None)#数据读取
v = dataframe.iloc[:,1:3]
coordinates= np.array(v)#经纬度即坐标矩阵
train_v= np.array(v)
train_d=train_v
numcity = coordinates.shape[0] #城市个数


#得到距离矩阵的函数
def getdistmat():#定义距离函数，返回每个城市之间的距离矩阵，采用欧式距离计算
    distmat = np.zeros((numcity,numcity))
    for i in range(numcity):
        for j in range(numcity):
            distmat[i,j] = math.sqrt(np.sum((train_v[i,:]-train_d[j,:])**2))
    return distmat


def initpara():
    alpha = 0.99
    t = (1,100)
    markovlen = 1000

    return alpha,t,markovlen

start = time.perf_counter()

distmat = getdistmat() #得到距离矩阵


pathnew = np.arange(numcity)
#valuenew = np.max(num)

pathcurrent = pathnew.copy()
valuecurrent =99000  #np.max这样的源代码可能同样是因为版本问题被当做函数不能正确使用，应取一个较大值作为初始值
#print(valuecurrent)

pathbest =pathnew.copy()
lengthbest = 99000 #np.max

alpha,t2,markovlen = initpara()
t = t2[1]

result = [] #记录迭代过程中的最优解
while t > t2[0]:
    for i in np.arange(markovlen):

        #下面的两交换和三角换是两种扰动方式，用于产生新解
        if np.random.rand() > 0.5:# 交换路径中的这2个节点的顺序
            # np.random.rand()产生[0, 1)区间的均匀随机数
            while True:#产生两个不同的随机数
                loc1 = np.int(np.ceil(np.random.rand()*(numcity-1)))
                loc2 = np.int(np.ceil(np.random.rand()*(numcity-1)))
                ## print(loc1,loc2)
                if loc1 != loc2:
                    break
            pathnew[loc1],pathnew[loc2] =pathnew[loc2],pathnew[loc1]
        else: #三交换
            while True:
                loc1 = np.int(np.ceil(np.random.rand()*(numcity-1)))
                loc2 = np.int(np.ceil(np.random.rand()*(numcity-1)))
                loc3 = np.int(np.ceil(np.random.rand()*(numcity-1)))

                if((loc1 != loc2)&(loc2 != loc3)&(loc1 != loc3)):
                    break

            # 下面的三个判断语句使得loc1<loc2<loc3
            if loc1 > loc2:
                loc1,loc2 = loc2,loc1
            if loc2 > loc3:
                loc2,loc3 = loc3,loc2
            if loc1 > loc2:
                loc1,loc2 = loc2,loc1

            #下面的三行代码将[loc1,loc2)区间的数据插入到loc3之后
            tmplist =pathnew[loc1:loc2].copy()
            pathnew[loc1:loc3-loc2+1+loc1] = pathnew[loc2:loc3+1].copy()
            pathnew[loc3-loc2+1+loc1:loc3+1] = tmplist.copy()

        valuenew = 0
        for i in range(numcity-1):
            valuenew += distmat[pathnew[i]][pathnew[i+1]]
        valuenew += distmat[pathnew[0]][pathnew[numcity-1]]
       # print (valuenew)
        if valuenew<valuecurrent: #接受该解

            #更新solutioncurrent 和solutionbest
            valuecurrent = valuenew
            pathcurrent = pathnew.copy()

            if valuenew < lengthbest:
                lengthbest = valuenew
                pathbest = pathnew.copy()
        else:#按一定的概率接受该解
            if np.random.rand() < np.exp(-(valuenew-valuecurrent)/t):
                valuecurrent = valuenew
                pathcurrent = pathnew.copy()
            else:
                pathnew = pathcurrent.copy()
    t = alpha*t
    result.append(lengthbest)
    print (t) #程序运行时间较长，打印t来监视程序进展速度
#用来显示结果
end = time.perf_counter()

print("程序的运行时间是：%s" % (end - start))

bestpath = pathbest
print("最短路径和为：%s" % (lengthbest))
for i in range(numcity):
    print(pathbest[i],end=" ")
    print("-->",end=" ")


plt.plot(coordinates[:,0],coordinates[:,1],'r.',marker='*',markersize=8)
plt.xlim([-100,4000])
plt.ylim([-100,4000])

for i in range(numcity-1):#
    m,n = int(bestpath[i]),int(bestpath[i+1])
    # print (m,n)
    plt.plot([coordinates[m][0],coordinates[n][0]],[coordinates[m][1],coordinates[n][1]],'k')
# plt.plot([coordinates[bestpath[0]][0],coordinates[n][0]],[coordinates[bestpath[0]][1],coordinates[n][1]],'b')
plt.plot([coordinates[int(bestpath[0])][0],coordinates[n][0]],[coordinates[int(bestpath[0])][1],
                                                                   coordinates[n][1]],'b')
ax=plt.gca()
ax.set_title("best path(SA)")
ax.set_xlabel('X_axis')
ax.set_ylabel('Y_axis')

# plt.savefig('Best Path_Ant.png',dpi=1000,bbox_inches='tight')
plt.savefig('best path(SA).png',dpi=1000)
plt.close()

plt.plot(np.array(result))
plt.ylabel("bestvalue")
plt.xlabel("t")
plt.show()