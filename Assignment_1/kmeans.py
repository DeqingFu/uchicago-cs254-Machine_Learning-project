import random
import sys
import matplotlib.pyplot as plt

C_final = []
for i in range(3):
  C_final.append([])
clusters = []
for i in range(3):
  clusters.append([])

def parse():
  data = []
  f = open("toydata.txt","r")
  while True:
    line = f.readline()
    if line == "":
      break
    x = float(line.split()[0])
    y = float(line.split()[1])
    data.append((x,y))
  return data

def random_init(m, data):
  gamma_1 = random.randint(0,499)
  m[0] = data[gamma_1]
  while True:
    gamma_2 = random.randint(0,499)
    if gamma_2 != gamma_1:
      break
  m[1] = data[gamma_2]
  while True:
    gamma_3 = random.randint(0,499)
    if gamma_3 != gamma_1 and gamma_3 != gamma_2:
      break
  m[2] = data[gamma_3]
  #plt.plot([m[0][0], m[1][0], m[2][0]], [m[0][1],m[1][1],m[2][1]], "ro", c = 'black')

def distance_sqr(x, y):
  return (x[0]-y[0])**2 + (x[1]-y[1])**2

def argmin(x,m):
  ret = -1
  minimum = sys.maxsize
  for i in range(3):
    d = distance_sqr(x, m[i])
    if d < minimum:
      minimum = d
      ret = i
  return ret

def J_avg_sqr(m, C, data):
  sum = 0
  for j in range(3):
    for index in C[j]:
      sum += distance_sqr(data[index], m[j])
  return sum

def vec_add(x, y):
  return (x[0]+y[0],x[1]+y[1])

def scalar_mul(scalar, vec):
  return (scalar * vec[0], scalar * vec[1])

def optimal_init(m, data):
  gamma_1 = random.randint(0,499)
  m[0] = data[gamma_1]
  from numpy.random import choice
  prob = [None] * 500
  d_sqr = [None] * 500
  for i in range(500):
    d_sqr[i] = distance_sqr(data[i], m[0])
  d_sqr_sum = reduce(lambda a, b: a + b, d_sqr)
  for i in range(500):
    prob[i] = d_sqr[i]/float(d_sqr_sum)
  gamma_2 = choice(range(0,500), 1, p=prob)
  m[1] = data[gamma_2[0]]
  prob = [None] * 500
  d_sqr = [None] * 500
  for i in range(500):
    d_sqr[i] = min(distance_sqr(data[i], m[0]), distance_sqr(data[i], m[1]))
  d_sqr_sum = reduce(lambda a, b: a + b, d_sqr)
  for i in range(500):
    prob[i] = d_sqr[i]/float(d_sqr_sum)
  gamma_3 = choice(range(0,500), 1, p=prob)
  m[2] = data[gamma_3[0]]
  #plt.plot([m[0][0], m[1][0], m[2][0]], [m[0][1],m[1][1],m[2][1]], "ro", c = 'black')

def kmeans(plus = False):
  #clutering is 3
  global C_final
  global clusters
  C_final = []
  for i in range(3):
    C_final.append([])
  clusters = []
  for i in range(3):
    clusters.append([])
  data = parse()
  n = len(data)
  m = [None, None, None]
  if plus:
    optimal_init(m, data)
  else:
    random_init(m, data)
  J = 0
  J_arr = []
  while True:
    C = []
    for _ in range(3):
      C.append([])
    for i in range(n):
      C[argmin(data[i],m)].append(i)
    for j in range(3):
      coef = 1.0/len(C[j])
      sum = (0,0)
      for index in C[j]:
        sum = vec_add(sum, data[index])
      m[j] = scalar_mul(coef, sum)
    J_new = J_avg_sqr(m, C, data)
    J_arr.append(J_new)
    if J_new == J:
      for i in range(3):
        C_final[i] = C[i]
        for index in range(500):
          if index in C_final[0]:
            clusters[0].append(data[index])
          if index in C_final[1]:
            clusters[1].append(data[index])
          if index in C_final[2]:
            clusters[2].append(data[index])
      return J_arr
    else:
      J = J_new
      

def kmeans_plot():
  J_arr = kmeans()
  for i in range(3):
    x = [d[0] for d in clusters[i]]
    y = [d[1] for d in clusters[i]]
    if i == 0:
      clr = 'r'
    elif i == 1:
      clr = 'b'
    else:
      clr = 'g'
    plt.plot(x,y, ".", color = clr)
  plt.show()

def kmeans_function_plot():
  for i in range(20):
    J_arr = kmeans()
    x = range(1, len(J_arr)+1)
    plt.plot(x, J_arr)
  plt.show()

def kmeans_pp_plot():
  J_arr = kmeans(True)
  for i in range(3):
    x = [d[0] for d in clusters[i]]
    y = [d[1] for d in clusters[i]]
    if i == 0:
      clr = 'r'
    elif i == 1:
      clr = 'b'
    else:
      clr = 'g'
    plt.plot(x,y, ".", color = clr)
  plt.show()

def kmeans_pp_function_plot():
  for i in range(20):
    J_arr = kmeans(True)
    x = range(1, len(J_arr)+1)
    plt.plot(x, J_arr)
  plt.show()


kmeans_plot()
kmeans_function_plot()
kmeans_pp_plot()
kmeans_pp_function_plot()


