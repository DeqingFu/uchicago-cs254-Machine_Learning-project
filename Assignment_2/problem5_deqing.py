import matplotlib.pyplot as plt
import numpy as np
import copy
import sys
import math

###parsing data###
#parsing digits
digits = []
f = open("train35.digits", "r")
while True:
  line = f.readline()
  if line == "":
    break
  vec = np.asarray([int(x) for x in line.split()] + [1])
  digits.append(vec)

labels = []
g = open("train35.labels", "r")
while True:
  line = g.readline()
  if line == "":
    break
  label = int(line.split()[0])
  labels.append(label)

n = len(labels)

def test_mistake(digits, labels, w):
  mis = 0
  t = 0
  for x in digits:
    if np.dot(w, np.asarray(x).T) >= 0:
      predict = 1
    else:
      predict = -1
    if predict != labels[t]:
      mis += 1
    t += 1
  return float(mis)/float(t)

mistakes = []
def perceptron_train(digits, labels, w = np.zeros(785)):
  n = len(digits)
  mistake = 0
  global mistakes
  for t in range(n):
    if np.dot(w, digits[t].T) >= 0:
      predict = 1
    else:
      predict = -1
    if (predict == -1 and labels[t] == 1):
      w = w + digits[t]
      mistake += 1
    if (predict == 1 and labels[t] == -1):
      w = w - digits[t]
      mistake += 1
    mistakes.append(mistake)
  return w

perceptron_train(digits, labels)
#plt.plot(range(n),mistakes,'.')
#plt.show()
'''
#trivial perceptron's prediction 
def perceptron_test(digits, labels):
  w = np.asarray([0] * 785)
  w = perceptron_train(digits, labels, w)
  h = open("test35.digits", "r")
  out = open("test35.predictions.trivial", "w")
  while True:
    line = h.readline()
    if line == "":
      break
    vec = np.asarray([int(x) for x in line.split()] + [1])
    if np.dot(w, vec.T) >= 0:
      out.write("1\n")
    else:
      out.write("-1\n")

perceptron_test(digits, labels)
'''
#preprocess of data for cross validation
#partition as 10
p = 10
k_digits = []
k_labels = []
part = n/p
for i in range(p):
  k_digits.append(digits[i*part:(i+1)*part])
  k_labels.append(labels[i*part:(i+1)*part])


epsilon = []
for M in range (1,11):
  error = []
  for i in range(p):
    test_digits = k_digits[i]
    test_labels = k_labels[i]
    for j in range(p): 
      w = np.asarray([0] * 785)
      for _ in range(M):
        if i != j:
          w = perceptron_train(k_digits[j], k_labels[j], w)
    error.append(test_mistake(test_digits,test_labels,w))
  print M, np.mean(error)
  epsilon.append(np.mean(error))

print "The M that minimizes the error is,", 1 + np.argmin(epsilon)

#cross validation M times
new_w = np.zeros(785)
for _ in range(6):
  new_w = perceptron_train(digits, labels, new_w)

h = open("test35.digits", "r")
out = open("test35.predictions", "w")
while True:
  line = h.readline()
  if line == "":
    break
  vec = np.asarray([int(x) for x in line.split()] + [1])
  if np.dot(new_w, vec.T) >= 0:
    out.write("1\n")
  else:
    out.write("-1\n")






