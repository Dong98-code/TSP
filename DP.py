"""
动态规划(Dynamic Programming，DP)
"""
import pandas as pd
import numpy as np
import math
import time
import matplotlib.pyplot as plt

dataframe = pd.read_csv("./data/TSP25cities.tsp",sep=" ",header=None)#读取TSP数据文件
# print(dataframe)
v = dataframe.iloc[:,1:3]#提取位置坐标
# print(v)

coordinates= np.array(v)
train_v= np.array(v)
train_d=train_v
numcity = coordinates.shape[0]
bestpath = np.zeros((numcity,))
dist = np.zeros((train_v.shape[0],train_d.shape[0]))#设置空矩阵，以计算两个城市之间的距离信息

#计算距离矩阵
for i in range(train_v.shape[0]):
    for j in range(train_d.shape[0]):
        dist[i,j] = math.sqrt(np.sum((train_v[i,:]-train_d[j,:])**2))#采用欧式距离计算

"""
N:城市数
s:二进制表示，遍历过得城市对应位为1，未遍历为0
dp:动态规划的距离数组
dist：城市间距离矩阵
sumpath:目前的最小路径总长度
Dtemp：当前最小距离
path:记录下一个应该到达的城市
"""

N=train_v.shape[0]
path = np.ones((2**(N+1),N))
dp = np.ones((2**(train_v.shape[0]+1),train_d.shape[0]))*-1

def TSP(s,init,num):
    if dp[s][init] !=-1 :
        return dp[s][init]
    if s==(1<<(N)):
        return dist[0][init]
    sumpath=1000000000
    for i in range(N):
        if s&(1<<i):
            m=TSP(s&(~(1<<i)),i,num+1)+dist[i][init]
            if m<sumpath:
                sumpath=m
                path[s][init]=i
    dp[s][init]=sumpath
    return dp[s][init]

if __name__ == "__main__":
    init_point=0
    s=0
    for i in range(1,N+1):
        s=s|(1<<i)
    start = time.perf_counter()
    distance=TSP(s,init_point,0)
    end = time.perf_counter()
    s=0b11111111110
    init=0
    num=0
    print(distance)
    while True:
        print(path[s][init])
        init=int(path[s][init])

        bestpath[num,]=init
        s=s&(~(1<<init))
        num+=1
        if num>9:
            break
    print("程序的运行时间是：%s"%(end-start))

    bestpath[numcity-1,]=0
    plt.plot(coordinates[:,0],coordinates[:,1],'r.',marker = "*",markersize=8)
    plt.xlim([-100,4000])
    plt.ylim([-100,4000])

for i in range(numcity-1):
    m,n = int(bestpath[i]),int(bestpath[i+1])
    # print (m,n)
    plt.plot([coordinates[m][0],coordinates[n][0]],[coordinates[m][1],coordinates[n][1]],'k')
plt.plot([coordinates[int(bestpath[0])][0],coordinates[n][0]],[coordinates[int(bestpath[0])][1],
                                                                   coordinates[n][1]],'b')

ax=plt.gca()
ax.set_title("best path(DP)")
ax.set_xlabel('X_axis')
ax.set_ylabel('Y_axis')

# plt.savefig('Best Path_Ant.png',dpi=1000,bbox_inches='tight')
plt.savefig('best path(DP).png',dpi=1000)
plt.close()