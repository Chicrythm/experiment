#!/usr/bin/python3
import numpy as np
import pulp as pl
import random
import xlsxwriter
import math


# GRACCF的约束条件
class Assignment:
  @classmethod
  # maxDrone: the amount of the drones
  def KM(cls, Q, La, L, Lsum):
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
        all += Q[i][j] * lpvars[i][j]

    pro += all
    # build constraint for each day of role
    for j in range(0, col):
      pro += pl.LpConstraint(pl.LpAffineExpression([(lpvars[i][j], 1) for i in range(0, row)]), -1, "Lsum" + str(j), Lsum[j])

    # build constraint for each agent
    for i in range(0, row):
      pro += pl.LpConstraint(pl.LpAffineExpression([(lpvars[i][j], 1) for j in range(0, col)]), 0, "La" + str(i), La[i])

    # build constraint for each role
    count = 0
    for k in range(0, len(dayFlights)):
      for j in range(0, col):
        pro += pl.LpConstraint(pl.LpAffineExpression([(lpvars[i][j], 1) for i in range(index_sum[k], index_sum[k+1])]), -1, "L" + str(count), L[k][j])
        count += 1
        # pro += lpvars[i][j] <= L[k, j]

    # for k in range(0, len_relationMat):
    #   pro += lpvars_GRACCF[k] * 2 <= lpvars[dimension_relationMat[k][0]][dimension_relationMat[k][1]] + \
    #          lpvars[dimension_relationMat[k][2]][dimension_relationMat[k][3]]

    # solve optimal problem
    status = pro.solve()
    print("Assignment Status: ", pl.LpStatus[status])
    print("Final Assignment Result", pl.value(pro.objective))

    # get the result of T matrix
    T = [[lpvars[i][j].varValue for j in range(col)] for i in range(row)]
    return [T, pl.value(pro.status), pl.value(pro.objective)]


#
if __name__ == '__main__':
  devCoe = 1
  ratioCoe = 10
  var1 = []
  var2 = []
  workbook = xlsxwriter.Workbook("uni1.xlsx")
  worksheet = workbook.add_worksheet()
  dayFlights = np.random.randint(20, 80, 7)
  worksheet.write_row('A2',dayFlights)


  index_sum = [0]
  for i in range(len(dayFlights)):
    index_sum.append(index_sum[i] + dayFlights[i])
  worksheet.write_row('A3',index_sum)
  flights = sum(dayFlights)
  print('sum_flights = ', flights)
  city = 20
  Lsum = np.random.randint(40, 60, city)
  worksheet.write_row('A4', Lsum)
  La = np.ones(flights)


  # 80 - 120 A mean the dev is 4 - 6
  A = np.random.randint(80, 121, city)
  # omiga coef use 0.1 - 0.3
  omiga = np.random.randint(10, 31, city)
  omiga = omiga / 100
  # fai use 8 - 12
  fai = np.random.randint(8, 13, city)
  x = np.linspace(3, 24, 2100)
  # base value 7 mean the vibration center is 7
  c = []
  for i in range(city):
    a = A[i] * np.sinc(omiga[i] * x + fai[i]) + 7
    c.append(a)
  L = np.zeros((21, 20))
  for i in range(len(c)):
    for j in range(21):
      L[j][i] = math.floor(c[i][j * 100])

  for i in range(len(L)):
    worksheet.write_row('A' + str(i + 6), L[i])


  sum_la = flights
  Q_Matrix = []

  martix = np.random.random_sample(city * flights)
  index = 0
  while index < len(martix):
    Q_Matrix.append(martix[index:index + city])
    index += city

  worksheet = workbook.add_worksheet()
  for i in range(len(Q_Matrix)):
    worksheet.write_row('A' + str(i + 1), Q_Matrix[i])

  TMatrix_GRA, result_GRA, performance_GRA = Assignment.KM(Q_Matrix, La, L, Lsum)
  lisGRA = []

  for i in range(len(TMatrix_GRA)):
    for j in range(len(TMatrix_GRA[0])):
      for k in range(int(TMatrix_GRA[i][j])):
        lisGRA.append(Q_Matrix[i][j])

  initPerformance = sum(lisGRA)

  sum_aver = initPerformance / flights
  init_sum = 0
  for i in range(len(lisGRA)):
    init_sum += (lisGRA[i] - sum_aver) ** 2

  init_var = init_sum / flights

  lisGRA = []
  Q_martix_co = np.zeros([flights, city])
  for i in range(len(Q_Matrix)):
    Q_martix_co[i] = Q_Matrix[i].copy()

  for i in range(flights):
    for j in range(city):
      if (Q_martix_co[i][j] < sum_aver and abs(Q_martix_co[i][j] - sum_aver) > devCoe * init_var and flights > 50):
        Q_martix_co[i][j] = abs(Q_martix_co[i][j] - sum_aver) / ratioCoe
      # 把超出范围的
      elif (Q_martix_co[i][j] > sum_aver and abs(sum_aver - Q_martix_co[i][j]) > devCoe * init_var and flights > 50):
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

  TMatrix_EXGRA, result_EXGRA, performance_EXGRA = Assignment.KM(Q_martix_co, La, L, Lsum)

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
  var4 = (improve_performance - initPerformance) * 100 / initPerformance
  print("init var: ", init_var)
  print("improve var: ", improve_var)
  print("init performance: ", initPerformance)
  print("imporve performance: ", improve_performance)
  print("VAR :", var3)
  print("PER :", var4)

  var1.append(var3)
  var2.append(var4)
  workbook.close()
