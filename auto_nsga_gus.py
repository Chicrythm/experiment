## 问题定义
from deap import creator, base, algorithms, tools
import numpy as np
import pulp as pl
import time
import xlsxwriter
import math


def KM(Q, La, L, Lsum, dayFlights, index_sum):
  row = len(Q)
  col = len(Q[0])

  # build a optimal problem
  pro = pl.LpProblem('Min(connection and coverage)', pl.LpMinimize)
  # build variables for the optimal problem
  lpvars = [[pl.LpVariable("x" + str(i) + "y" + str(j), lowBound=0, cat='Integer') for j in range(col)]
            for i in range(row)]

  # build optimal function
  all = pl.LpAffineExpression()
  for i in range(0, row):
    for j in range(0, col):
      all += Q[i][j] * lpvars[i][j]

  pro += all
  # build constraint for each day of role
  for j in range(0, col):
    pro += pl.LpConstraint(pl.LpAffineExpression([(lpvars[i][j], 1) for i in range(0, row)]), -1, "Lsum" + str(j),
                           Lsum[j])

  # build constraint for each agent
  for i in range(0, row):
    pro += pl.LpConstraint(pl.LpAffineExpression([(lpvars[i][j], 1) for j in range(0, col)]), 0, "La" + str(i), La[i])

  # build constraint for each role
  count = 0
  for k in range(0, len(dayFlights)):
    for j in range(0, col):
      pro += pl.LpConstraint(pl.LpAffineExpression([(lpvars[i][j], 1) for i in range(index_sum[k], index_sum[k + 1])]),
                             -1, "L" + str(count), L[k][j])
      count += 1

  # solve optimal problem
  status = pro.solve()
  print("Assignment Status: ", pl.LpStatus[status])
  print("Final Assignment Result", pl.value(pro.objective))

  # get the result of T matrix
  T = [[lpvars[i][j].varValue for j in range(col)] for i in range(row)]
  return [T, pl.value(pro.status), pl.value(pro.objective)]

def KMV(Q, La, L, Lsum, dayFlights, index_sum, sum_ave):
    row = len(Q)
    col = len(Q[0])
    # len_relationMat = len(dimension_relationMat)

    # build a optimal problem
    pro = pl.LpProblem('Min(connection and coverage)', pl.LpMinimize)
    # build variables for the optimal problem
    lpvars = [[pl.LpVariable("x" + str(i) + "y" + str(j), lowBound=0, cat='Integer') for j in range(col)]
              for i in range(row)]

    # build optimal function
    all = pl.LpAffineExpression()
    for i in range(0, row):
      for j in range(0, col):
        all += ((Q[i][j] - sum_ave)**2 / row) * lpvars[i][j]

    pro += all
    # build constraint for each day of role
    for j in range(0, col):
      pro += pl.LpConstraint(pl.LpAffineExpression([(lpvars[i][j], 1) for i in range(0, row)]), -1, "Lsum" + str(j),
                             Lsum[j])

    # build constraint for each agent
    for i in range(0, row):
      pro += pl.LpConstraint(pl.LpAffineExpression([(lpvars[i][j], 1) for j in range(0, col)]), 0, "La" + str(i), La[i])

    # build constraint for each role
    count = 0
    for k in range(0, len(dayFlights)):
      for j in range(0, col):
        pro += pl.LpConstraint(
          pl.LpAffineExpression([(lpvars[i][j], 1) for i in range(index_sum[k], index_sum[k + 1])]), -1,
          "L" + str(count), L[k][j])
        count += 1

    # solve optimal problem
    status = pro.solve()
    print("Assignment Status: ", pl.LpStatus[status])
    print("Final Assignment Result", pl.value(pro.objective))

    # get the result of T matrix
    T = [[lpvars[i][j].varValue for j in range(col)] for i in range(row)]
    return [T, pl.value(pro.status), pl.value(pro.objective)]

def assignment(Q_Matrix, La, L, Lsum, dayFlights, index_sum, city, flights, sum_aver, devCoe, ratioCoe):

  TMatrix_GRA, result_GRA, performance_GRA = KM(Q_Matrix, La, L, Lsum, dayFlights, index_sum)

  lisGRA = []
  for i in range(len(TMatrix_GRA)):
    for j in range(len(TMatrix_GRA[0])):
      for k in range(int(TMatrix_GRA[i][j])):
        lisGRA.append(Q_Matrix[i][j])

  initPerformance = sum(lisGRA)

  aver = initPerformance / flights
  init_sum = 0
  for i in range(len(lisGRA)):
    init_sum += (lisGRA[i] - aver) ** 2

  init_var = init_sum / flights
  Q_martix_co = np.zeros([flights, city])
  for i in range(len(Q_Matrix)):
    Q_martix_co[i] = Q_Matrix[i].copy()

  for i in range(flights):
    for j in range(city):
      # 缩小较小的部分
      if(Q_martix_co[i][j] < sum_aver and abs(Q_martix_co[i][j] - sum_aver) > devCoe * init_var and flights > 50):
        Q_martix_co[i][j] = abs(Q_martix_co[i][j] - sum_aver) / ratioCoe
      # 把超出范围的
      elif(Q_martix_co[i][j] > sum_aver and abs(sum_aver - Q_martix_co[i][j]) > devCoe * init_var and flights > 50):
        Q_martix_co[i][j] = abs(Q_martix_co[i][j] - sum_aver) * ratioCoe
      else:
        Q_martix_co[i][j] = abs(Q_martix_co[i][j] - sum_aver)

  min_Qco = 10
  max_Qco = 0
  for i in range(len(Q_martix_co)):
    maxLen = max(Q_martix_co[i])
    minLen = min(Q_martix_co[i])
    if max_Qco < maxLen:
      max_Qco = maxLen
    if min_Qco > minLen:
      min_Qco = minLen

  for i in range(len(Q_martix_co)):
    for j in range(len(Q_martix_co[0])):
      Q_martix_co[i][j] = (Q_martix_co[i][j] - min_Qco) / (max_Qco - min_Qco)

  TMatrix_EXGRA, result_EXGRA, performance_EXGRA = KM(Q_martix_co, La, L, Lsum, dayFlights, index_sum)

  lisEXGRA = []
  # 求第二次指派后的T矩阵得分
  for i in range(len(TMatrix_EXGRA)):
    for j in range(len(TMatrix_EXGRA[0])):
      if TMatrix_EXGRA[i][j] != 0:
        lisEXGRA.append(Q_Matrix[i][j])

  improve_performance = sum(lisEXGRA)

  sum_aver_after = improve_performance / flights

  init_sum = 0
  for i in range(len(lisEXGRA)):
    init_sum += (lisEXGRA[i] - sum_aver_after) ** 2

  improve_var = init_sum / flights

  var3 = (init_var - improve_var) * 100 / improve_var
  var3 = round(var3, 2)
  var4 = (improve_performance - initPerformance) * 100 / initPerformance
  var4 = round(var4, 2)

  return var3,var4

def GMRACF(Q_Matrix, La, L, Lsum, dayFlights, index_sum, flights):

  TMatrix_GRA, result_GRA, performance_GRA = KM(Q_Matrix, La, L, Lsum, dayFlights, index_sum)
  lisGRA = []
  for i in range(len(TMatrix_GRA)):
    for j in range(len(TMatrix_GRA[0])):
      if TMatrix_GRA[i][j] != 0:
        lisGRA.append(Q_Matrix[i][j])

  initPerformance = sum(lisGRA)

  bestTeam_aver = initPerformance / flights
  init_sum = 0
  for i in range(len(lisGRA)):
    init_sum += (lisGRA[i] - bestTeam_aver) ** 2

  init_var = init_sum / flights

  # GMRACF
  # TMatrix_GMRACF, result_GMRACF, performance_GMRACF = KMV(Q_Matrix, La, L, Lsum, dayFlights, index_sum, teamAver)
  #
  # GMRACFlis = []
  # for i in range(len(TMatrix_GMRACF)):
  #   for j in range(len(TMatrix_GMRACF[0])):
  #     if TMatrix_GMRACF[i][j] != 0:
  #       GMRACFlis.append(Q_Matrix[i][j])
  #
  # GMRACFper = sum(GMRACFlis)
  #
  # GMRACFaver = GMRACFper / flights
  #
  # GMRACFsum = 0
  # for i in range(len(GMRACFlis)):
  #   GMRACFsum += ((GMRACFlis[i] - GMRACFaver) ** 2)
  #
  # GMRACFvar = GMRACFsum / flights

  # IGMRACF
  TMatrix_IGMRACF, result_IGMRACF, performance_IGMRACF = KMV(Q_Matrix, La, L, Lsum, dayFlights, index_sum, bestTeam_aver)
  lisIGMRACF = []
  for i in range(len(TMatrix_IGMRACF)):
    for j in range(len(TMatrix_IGMRACF[0])):
      if TMatrix_IGMRACF[i][j] != 0:
        lisIGMRACF.append(Q_Matrix[i][j])

  IGMRACFper = sum(lisIGMRACF)

  IGMRACFaver = IGMRACFper / flights

  IGMRACFsum = 0
  for i in range(len(lisIGMRACF)):
    IGMRACFsum += (lisIGMRACF[i] - IGMRACFaver) ** 2

  IGMRACFvar = IGMRACFsum / flights

  # GMRACFImp = (init_var - GMRACFvar) * 100 / GMRACFvar
  # GMRACFLoss = (GMRACFper - initPerformance) * 100 / initPerformance
  IGMRACFImp = (init_var - IGMRACFvar) * 100 / IGMRACFvar
  IGMRACFLoss = (IGMRACFper - initPerformance) * 100 / initPerformance


  return [IGMRACFImp, IGMRACFLoss]

def nsga(Q_Matrix, La, L, Lsum, dayFlights, index_sum, city, flights):
  startTime = time.time()
  creator.create('MultiObjMin', base.Fitness, weights=(-1.0, -1.0))
  creator.create('Individual', list, fitness=creator.MultiObjMin)

  ## 个体编码
  def uniform(low, up):
    # 用均匀分布生成个体
    return [np.random.uniform(a, b) for a, b in zip(low, up)]

  toolbox = base.Toolbox()
  NDim = 3  # 变量数为3
  low = [0, 0, 0]  # 变量下界
  up = [0.35, 10, 20]  # 变量上界

  toolbox.register('attr_float', uniform, low, up)
  toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.attr_float)
  toolbox.register('population', tools.initRepeat, list, toolbox.individual)

  ## 评价函数
  def ZDT3(ind):

    f1, f2 = assignment(Q_Matrix, La, L, Lsum, dayFlights, index_sum, city, flights, ind[0], ind[1], ind[2])
    if (f1 < 0 or f2 > 25):
      return 100, -100
    return -f1, f2

  toolbox.register('evaluate', ZDT3)

  ## 注册工具
  toolbox.register('selectGen1', tools.selTournament, tournsize=2)
  toolbox.register('select', tools.emo.selTournamentDCD)  # 该函数是binary tournament，不需要tournsize
  toolbox.register('mate', tools.cxSimulatedBinaryBounded, eta=20.0, low=low, up=up)
  toolbox.register('mutate', tools.mutPolynomialBounded, eta=20.0, low=low, up=up, indpb=1.0 / NDim)

  ## 遗传算法主程序
  # 参数设置
  toolbox.popSize = 80
  toolbox.maxGen = 10
  toolbox.cxProb = 0.7
  toolbox.mutateProb = 0.2

  # 迭代部分
  # 第一代
  pop = toolbox.population(toolbox.popSize)  # 父代
  fitnesses = toolbox.map(toolbox.evaluate, pop)
  for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit
  fronts = tools.emo.sortNondominated(pop, k=toolbox.popSize)
  # 将每个个体的适应度设置为pareto前沿的次序
  for idx, front in enumerate(fronts):
    for ind in front:
      ind.fitness.values = (idx + 1),
  # 创建子代
  offspring = toolbox.selectGen1(pop, toolbox.popSize)  # binary Tournament选择
  offspring = algorithms.varAnd(offspring, toolbox, toolbox.cxProb, toolbox.mutateProb)

  # 第二代之后的迭代
  for gen in range(1, toolbox.maxGen):
    combinedPop = pop + offspring  # 合并父代与子代
    # 评价族群
    fitnesses = toolbox.map(toolbox.evaluate, combinedPop)
    for ind, fit in zip(combinedPop, fitnesses):
      ind.fitness.values = fit
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

    # 创建子代
    offspring = toolbox.select(pop, toolbox.popSize)
    offspring = toolbox.clone(offspring)
    offspring = algorithms.varAnd(offspring, toolbox, toolbox.cxProb, toolbox.mutateProb)

  endTime = time.time()
  ace = endTime - startTime
  # print('------------------------time', ace)
  frontData = []

  for ind in front:
    lis = []
    lis.append(ind[0])
    lis.append(ind[1])
    lis.append(ind[2])
    lis.append(-ind.fitness.values[0])
    lis.append(ind.fitness.values[1])
    frontData.append([ind[0], ind[1], ind[2], -ind.fitness.values[0], ind.fitness.values[1]])
  worksheet = workbook.add_worksheet()
  for i in range(len(frontData)):
    worksheet.write_row('A' + str(i + 1), frontData[i])
  tempLis = []
  for i in range(len(frontData)):
    if frontData[i][4] < 15:
      tempLis.append([(15 - frontData[i][4]), frontData[i][3], frontData[i][4]])
  tempLis.sort()
  return [tempLis[0][1],tempLis[0][2]]

if __name__ == '__main__':

  # Statistical data
  sumFDImp = 0
  sumFDLoss = 0
  sumGMRACFImp = 0
  sumGMRACFLoss = 0
  sumIGMRACFImp = 0
  sumIGMRACFLoss = 0

  workbook = xlsxwriter.Workbook("gus50-60.xlsx")

  dayFlightMin = 50
  dayFlightMax = 60
  count = 1
  while count <= 5:
    count += 1
    # 每日航班
    dayFlights = np.random.randint(dayFlightMin, dayFlightMax, 7)
    # 航班累加总计
    index_sum = [0]
    for i in range(len(dayFlights)):
      index_sum.append(index_sum[i] + dayFlights[i])
    flights = sum(dayFlights)
    city = 20
    Lsum = np.random.randint(40, 60, city)
    La = np.ones(flights)
    A = np.random.randint(80, 121, city)
    omiga = np.random.randint(10, 31, city)
    omiga = omiga / 100
    fai = np.random.randint(8, 13, city)
    x = np.linspace(3, 24, 2100)
    c = []
    for i in range(city):
      a = A[i] * np.sinc(omiga[i] * x + fai[i]) + 7
      c.append(a)
    L = np.zeros((21, 20))
    for i in range(len(c)):
      for j in range(21):
        L[j][i] = math.floor(c[i][j * 100])


    Q_Matrix = []
    martix = np.random.normal(0.5, 0.1, city * flights)
    max_gus = max(martix)
    min_gus = min(martix)
    for i in range(len(martix)):
      martix[i] = (martix[i] - min_gus) / (max_gus - min_gus)
    index = 0
    while index < len(martix):
      Q_Matrix.append(martix[index:index + city])
      index += city

    [FDImp,FDLoss] = nsga(Q_Matrix, La, L, Lsum, dayFlights, index_sum, city, flights)

    print('FDImp',FDImp, 'FDLoss', FDLoss)

    [IGMRACFImp, IGMRACFLoss] = GMRACF(Q_Matrix, La, L, Lsum, dayFlights, index_sum, flights)
    print('IGMRACFImp', IGMRACFImp, 'IGMRACFLoss', IGMRACFLoss)

    # Statistical data
    sumFDImp += FDImp
    sumFDLoss += FDLoss
    sumIGMRACFImp += IGMRACFImp
    sumIGMRACFLoss += IGMRACFLoss


  # Statistical data
  averFDImp = sumFDImp / (count - 1)
  averFDLoss = sumFDLoss / (count - 1)
  sumIGMRACFImp = sumIGMRACFImp / (count - 1)
  sumIGMRACFLoss = sumIGMRACFLoss / (count - 1)
  a = []
  b = []
  b.append(averFDImp)
  b.append(averFDLoss)
  b.append(sumIGMRACFImp)
  b.append(sumIGMRACFLoss)
  a.append(b)

  print('averFDImp', averFDImp)
  print('averFDLoss', averFDLoss)
  print('sumIGMRACFImp', sumIGMRACFImp)
  print('sumIGMRACFLoss', sumIGMRACFLoss)

  worksheet = workbook.add_worksheet()
  worksheet.write_row('A2', a[0])

  workbook.close()