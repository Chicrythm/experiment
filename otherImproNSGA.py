#-------------- NSGA-II 算法实现-----------
## 问题定义
from deap import creator, base, algorithms, tools
import numpy as np
from scenario_overall import Assignment
import matplotlib.pyplot as plt
import time
import xlsxwriter
startTime = time.time()
workbook = xlsxwriter.Workbook("otherImprovement.xlsx")
worksheet = workbook.add_worksheet()

creator.create('MultiObjMin', base.Fitness, weights=(-1.0, -1.0))
creator.create('Individual', list, fitness=creator.MultiObjMin)

## 个体编码
def uniform(low, up):
    # 用均匀分布生成个体
    a = [np.random.uniform(a,b) for a,b in zip(low, up)]
    b = []

    for item in a:
        b.append(round(item, 2))

    return b


toolbox = base.Toolbox()
NDim = 5 # 变量数为3
low = [0, 0, 0, 0, 0] # 变量下界
up = [1, 5, 10, 5, 10] # 变量上界

toolbox.register('attr_float', uniform, low, up)
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

## 评价函数
def ZDT3(ind):
    f1, f2 = Assignment.RUN_GRA(ind[0], ind[1], ind[2], ind[3], ind[4])
    if (f1 < 10):
        return 100, -100
    return -f1, f2

toolbox.register('evaluate', ZDT3)

## 注册工具
toolbox.register('selectGen1', tools.selTournament, tournsize=2)
toolbox.register('select', tools.emo.selTournamentDCD) # 该函数是binary tournament，不需要tournsize
toolbox.register('mate', tools.cxSimulatedBinaryBounded, eta=20.0, low=low, up=up)
toolbox.register('mutate', tools.mutPolynomialBounded, eta=20.0, low=low, up=up, indpb=1.0/NDim)

## 遗传算法主程序
# 参数设置
toolbox.popSize = 60
toolbox.maxGen = 10
toolbox.cxProb = 0.7
toolbox.mutateProb = 0.2

# 迭代部分
# 第一代
pop = toolbox.population(toolbox.popSize) # 父代
fitnesses = toolbox.map(toolbox.evaluate, pop)
for ind, fit in zip(pop,fitnesses):
    ind.fitness.values = fit
fronts = tools.emo.sortNondominated(pop, k=toolbox.popSize)
# 将每个个体的适应度设置为pareto前沿的次序
for idx, front in enumerate(fronts):
    for ind in front:
        ind.fitness.values = (idx+1),
# 创建子代
offspring = toolbox.selectGen1(pop, toolbox.popSize) # binary Tournament选择
offspring = algorithms.varAnd(offspring, toolbox, toolbox.cxProb, toolbox.mutateProb)

iterData = []

# 第二代之后的迭代
for gen in range(1, toolbox.maxGen):
    combinedPop = pop + offspring # 合并父代与子代
    # 评价族群
    fitnesses = toolbox.map(toolbox.evaluate, combinedPop)
    for ind, fit in zip(combinedPop,fitnesses):
        ind.fitness.values = fit
    plt.plot(-ind.fitness.values[0], ind.fitness.values[1], 'r.', ms=2)
    # 快速非支配排序
    fronts = tools.emo.sortNondominated(combinedPop, k=toolbox.popSize, first_front_only=False)
    # 拥挤距离计算
    for front in fronts:
        tools.emo.assignCrowdingDist(front)
    # 环境选择 -- 精英保留
    pop = []
    for front in fronts:
        pop += front
    pop = toolbox.clone(pop)
    pop = tools.selNSGA2(pop, k=toolbox.popSize, nd='standard')

    # 为作图保留下空间
    for i in range(len(pop)):
        lis = []
        lis.append(pop[i][0])
        lis.append(pop[i][1])
        lis.append(pop[i][2])
        lis.append(pop[i][3])
        lis.append(pop[i][4])
        lis.append(-pop[i].fitness.values[0])
        lis.append(pop[i].fitness.values[1])
        iterData.append(lis)
    # 创建子代
    offspring = toolbox.select(pop, toolbox.popSize)
    offspring = toolbox.clone(offspring)
    offspring = algorithms.varAnd(offspring, toolbox, toolbox.cxProb, toolbox.mutateProb)

for i in range(len(iterData)):
    worksheet.write_row('A' + str(i + 1), iterData[i])

endTime = time.time()
ace = endTime - startTime
print('------------------------time', ace)
frontData = []
for ind in front:
    lis = []
    lis.append(ind[0])
    lis.append(ind[1])
    lis.append(ind[2])
    lis.append(ind[3])
    lis.append(ind[4])
    lis.append(-ind.fitness.values[0])
    lis.append(ind.fitness.values[1])
    frontData.append(lis)

worksheet2 = workbook.add_worksheet()
for i in range(len(frontData)):
    worksheet2.write_row('A' + str(i + 1), frontData[i])

workbook.close()
