import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import copy
import sys
import math

# parsing data
color_map={1:"green",2:"yellow",3:"blue",4:"red"}
def parse_3ddata():
  data = []
  color_data = []
  f = open("3Ddata-1.txt", "r")
  while True:
    line = f.readline()
    if line == "":
      break
    l = line.split()
    xi = np.ndarray((3,1))
    xi[0][0] = float(l[0])
    xi[1][0] = float(l[1])
    xi[2][0] = float(l[2])
    data.append(xi)
    color_data.append(color_map[int(l[3])])
  return data, color_data

data,color_data = parse_3ddata()
n = len(data)

### ---- Principal Component Analysis ---- ###
#pre-processing of data
def PCA():
  mean = np.zeros((3,1))
  for i in range(n):
    mean+=data[i]
  mean *= 1.0/n
  centered_data = copy.deepcopy(data)
  for i in range(n):
    centered_data[i] = data[i] - mean

  #compute covariance_matrix
  covariance_matrix = np.zeros((3,3))
  for i in range(n):
    xi = centered_data[i]
    covariance_matrix += np.dot(xi, xi.T)
  covariance_matrix *= 1.0/n

  #finding principal components
  eigenvalues, eigenvectors_matrix = np.linalg.eig(covariance_matrix)
  eigenvectors = eigenvectors_matrix.T
  #first principal component
  index_1 = np.argmax(eigenvalues)
  p1 = eigenvectors[index_1]
  #second principal component
  second_eigenvalues = np.delete(eigenvalues, index_1, 0)
  second_eigenvectors = np.delete(eigenvectors, index_1, 0)
  index_2 = np.argmax(second_eigenvalues)
  p2 = second_eigenvectors[index_2]

  #projecting to the new space 
  pca_matrix = np.ndarray((2,3))
  pca_matrix[0] = p1
  pca_matrix[1] = p2
  pca_data = [None] * n
  for i in range(n):
    pca_data[i] = ((np.dot(pca_matrix,centered_data[i])).T)[0]

  #plotting
  pca_xs = []
  pca_ys = []
  for i in range(n):
    pca_xs.append(pca_data[i][0])
    pca_ys.append(pca_data[i][1])
  plt.scatter(pca_xs,pca_ys,color=color_data)
  plt.show()

### ---- ISOMAP ---- ###
def distance(xi, xj):
  return math.sqrt((xi[0][0]-xj[0][0])**2 + (xi[1][0]-xj[1][0])**2 + (xi[2][0]-xj[2][0])**2)

def isomap():
  graph = np.zeros((n,n))
  k = 10
  #initializing the matrix with distance of k nearest neighbors
  for i in range(n):
    distances = []
    for j in range(n):
      if i != j:
        graph[i][j] = sys.maxsize
      distances.append((j, distance(data[i],data[j])))
      # Because for j in range(n),
      # there always exists an j = i such that d(x[i],x[j]) = 0
      # we should pop that from the k nearest neighbors
      k_neighbors_distance = sorted(distances, key = lambda x: x[1])[1:k+1]
      k_nearest_neighbors = [x[0] for x in k_neighbors_distance]
    for x in k_nearest_neighbors:
      dist = distance(data[i], data[x])
      graph[i][x] = dist
      graph[x][i] = dist

  #Floyd-Warshall Algorithm
  for k in range(n):
    for i in range(n):
      for j in range(n):
        if graph[i][j] > graph[i][k] + graph[k][j]:
          graph[i][j] = graph[i][k] + graph[k][j]

  # Make sure it's symmetric
  for i in range(n):
    for j in range(n):
      if i > j:
        graph[i][j] = graph[j][i]

  P = np.identity(n) - 1.0/n * np.ones((n,n))
  Gram = -0.5 * np.dot(np.dot(P,graph**2),P)

  lambdas, Q = np.linalg.eig(Gram)

  Lambda = np.zeros((n,n))
  lambdas = sorted(lambdas, reverse = True)
  Lambda[0][0] = lambdas[0]
  Lambda[1][1] = lambdas[1]

  ISOMAP_matrix = np.dot(Q, Lambda)

  iso_xs = []
  iso_ys = []
  for i in range(n):
    iso_xs.append(ISOMAP_matrix[i][0])
    iso_ys.append(ISOMAP_matrix[i][1])
  plt.scatter(iso_xs,iso_ys,color=color_data)
  plt.show()

if __name__ == "__main__":
  x = input("Enter 1 for PCA and 2 for ISOMAP:")
  if x == 1:
    PCA()
  elif x == 2:
    isomap()
  else: 
    print "Invalid"
