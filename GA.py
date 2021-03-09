"""
遗传算法(Genetic Algorithm ，GA )
"""

import math, random, time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

 #读取坐标文件
dataframe = pd.read_csv("./data/TSP100cities.tsp",sep=" ",header=None)#数据读取
v = dataframe.iloc[:,1:3]
coordinates= np.array(v)#经纬度即坐标矩阵
train_v= np.array(v)
train_d=train_v
numcity =coordinates.shape[0]  # 总的城市数
remain_cities = [i for i in range(numcity)]

numcity =coordinates.shape[0]  # 总的城市数
origin = 0  # 设置起点和终点
population_size = 100  # 种群数
mutation_rate = 0.3  # 变异概率

remain_cities.remove(origin)  # 迭代过程中变动的城市
remain_count = numcity - 1  # 迭代过程中变动的城市数
indexs = list(i for i in range(remain_count))

# 计算邻接矩阵
def getdistmat():#定义距离函数，返回每个城市之间的距离矩阵，采用欧式距离计算
    distmat = np.zeros((numcity, numcity))
    for i in range(numcity):
        for j in range(numcity):
            distmat[i,j] = math.sqrt(np.sum((train_v[i,:]-train_d[j,:])**2))
    return distmat
distmat=getdistmat()





def route_mile_cost(route):
    distance=0
    distance+=distmat[origin][route[0]]
    for i in range(len(route)):
        if i==len(route)-1:
            distance += distmat[origin][route[i]]
        else:
            distance += distmat[route[i]][route[i + 1]]
    return distance


# 获取当前邻居城市中距离最短的1个
def nearest_city(current_city,remain_cities):
    temp_min = float('inf')
    next_city = None
    for i in range(len(remain_cities)):
        distance = distmat[current_city][remain_cities[i]]
        if distance < temp_min:
            temp_min = distance
            next_city = remain_cities[i]
    return next_city


def greedy_initial_route(remain_cities):
    '''
    采用贪婪算法生成初始解：从第一个城市出发找寻与其距离最短的城市并标记，
    然后继续找寻与第二个城市距离最短的城市并标记，直到所有城市被标记完。
    最后回到第一个城市(起点城市)
    '''
    cand_cities = remain_cities[:]
    current_city = origin
    initial_route = []
    while len(cand_cities) > 0:
        next_city = nearest_city(current_city, cand_cities)  # 找寻最近的城市及其距离
        initial_route.append(next_city)  # 将下一个城市添加到路径列表中
        current_city = next_city  # 更新当前城市
        cand_cities.remove(next_city)  # 更新未定序的城市
    return initial_route


# 物竞天择，适者生存
def selection(population):
    '''
    选出父代个体
    '''
    M = population_size
    parents = []
    for i in range(M):
        if random.random() < (1 - i / M):
            parents.append(population[i])
    return parents


def CPX(parent1, parent2):
    '''
    交叉繁殖：CX与PX的混合双亲产生两个子代
    '''
    cycle = []
    start = parent1[0]
    cycle.append(start)
    end = parent2[0]
    while end != start:
        cycle.append(end)
        end = parent2[parent1.index(end)]
    child = parent1[:]
    cross_points = cycle[:]
    if len(cross_points) < 2:
        cross_points = random.sample(parent1, 2)
    k = 0
    for i in range(len(parent1)):
        if child[i] in cross_points:
            continue
        else:
            for j in range(k, len(parent2)):
                if parent2[j] in cross_points:
                    continue
                else:
                    child[i] = parent2[j]
                    k = j + 1
                    break
    return child


# 变异
def mutation(children, mutation_rate):
    '''
    子代变异
    '''
    for i in range(len(children)):
        if random.random() < mutation_rate:
            child = children[i]
            new_child = child[:]
            index = sorted(random.sample(indexs, 2))
            L = index[1] - index[0] + 1
            for j in range(L):
                new_child[index[0] + j] = child[index[1] - j]
            path = [origin] + child + [origin]
            a, b = index[0] , index[1]
            d1 = distmat[path[a - 1] - 1][path[a] - 1] + distmat[path[b] - 1][path[b + 1] - 1]
            d2 = distmat[path[a - 1] - 1][path[b] - 1] + distmat[path[a] - 1][path[b + 1] - 1]
            if d2 < d1:
                children[i] = new_child

    return children


def get_best_current(population):
    '''
    将种群的个体按照里程排序，并返回当前种群中的最优个体及其里程
    '''
    graded = [[route_mile_cost(x), x] for x in population]#x为一个路径顺序
    graded = sorted(graded)
    population = [x[1] for x in graded]
    return graded[0][0], graded[0][1], population





def main(iter_count):
    # 初始化种群
    population0 = greedy_initial_route(remain_cities)#greedy_..函数得到的是初始路线
    # print("初始路线：")
    initial_route=[origin]+population0+[origin]
    population=[population0]


    # population = []
    for i in range(population_size - 1):
        # 随机生成个体
        individual = remain_cities[:]
        random.shuffle(individual)
        population.append(individual)
    mile_cost, result, population = get_best_current(population)
    record = [mile_cost]  # 记录每一次繁殖的最优值
    i = 0
    while i < iter_count:
        # 选择繁殖个体群
        parents = selection(population)
        # 交叉繁殖
        target_count = population_size - len(parents)  # 需要繁殖的数量(保持种群的规模)
        children = []
        while len(children) < target_count:
            parent1, parent2 = random.sample(parents, 2)
            child1 = CPX(parent1, parent2)
            child2 = CPX(parent2, parent1)
            children.append(child1)
            children.append(child2)
        # 父代变异
        parents = mutation(parents, 1)
        # 子代变异
        children = mutation(children, mutation_rate)
        # 更新种群
        population = parents + children
        # 更新繁殖结果
        mile_cost, result, population = get_best_current(population)
        record.append(mile_cost)  # 记录每次繁殖后的最优解
        i += 1
        if i%10==0:
            print("进化次数：%s"%i)

    route = [origin] + result+[origin]
    return initial_route,route, mile_cost, record

#主程序部分
start = time.perf_counter()
N = 50000  # 进化次数
initial_path,bestpath, lengthbest, record = main(N)
time_end = time.time()
end=time.perf_counter()
print("程序的运行时间是：%s" % (end - start)+" s")
print("最优路径总和:%s" % (lengthbest))
for i in range(numcity):
    print(bestpath[i], end=" ")
    print("-->",end=" ")

#绘制路线图

#初始路线
plt.plot(coordinates[:, 0], coordinates[:, 1], 'r.', marker='*', markersize=8)
plt.xlim([-100,3500])
plt.ylim([-100,3500])

for i in range(numcity - 1):#
    m,n = int(initial_path[i]), int(initial_path[i + 1])
    plt.plot([coordinates[m][0], coordinates[n][0]], [coordinates[m][1], coordinates[n][1]], 'k')#中间的路线
plt.plot([coordinates[int(initial_path[0])][0], coordinates[n][0]], [coordinates[int(initial_path[0])][1],
                                                                 coordinates[n][1]], 'b')#蓝色表示头尾相连
ax=plt.gca()
ax.set_title("initial path(GA)-TSP%dcities"%numcity)
ax.set_xlabel('X_axis')
ax.set_ylabel('Y_axis')

plt.savefig('D:/XD/Documents/研究生/课件及作业/智能优化/结果/GA/initial path(GA)-TSP%dcities.png'
            %numcity,dpi=1000,bbox_inches='tight')
plt.show()

plt.close()
#最终路线
plt.plot(coordinates[:, 0], coordinates[:, 1], 'r.', marker='*', markersize=8)
plt.xlim([-100,4000])
plt.ylim([-100,4000])

for i in range(numcity - 1):#
    m,n = int(bestpath[i]), int(bestpath[i + 1])
    plt.plot([coordinates[m][0], coordinates[n][0]], [coordinates[m][1], coordinates[n][1]], 'k')#中间的路线
plt.plot([coordinates[int(bestpath[0])][0], coordinates[n][0]], [coordinates[int(bestpath[0])][1],
                                                                 coordinates[n][1]], 'b')#蓝色表示头尾相连
ax=plt.gca()
ax.set_title("best path-TSP%dcities"%numcity)
ax.set_xlabel('X_axis')
ax.set_ylabel('Y_axis')

plt.savefig('D:/XD/Documents/研究生/课件及作业/智能优化/结果/GA/best path(GA)-TSP%dcities.png'
            %numcity,dpi=1000,bbox_inches='tight')
plt.show()
plt.close()


# 绘制迭代过程图
X_axis= [i for i in range(N + 1)]  # 横坐标
Y_axis = record[:]  # 纵坐标
plt.xlim(0, N)
plt.xlabel('进化次数', fontproperties="SimSun")
plt.ylabel('最佳路径和', fontproperties="SimSun")
plt.title("进化过程图",fontproperties="SimSun")
plt.plot(X_axis, Y_axis, '-')
plt.grid()
plt.savefig('D:/XD/Documents/研究生/课件及作业/智能优化/结果/GA/进化过程图-TSP%dcities.png'
            %numcity,dpi=1000,bbox_inches='tight')
plt.show()
