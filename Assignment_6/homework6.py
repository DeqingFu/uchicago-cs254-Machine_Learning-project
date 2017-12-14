from __future__ import print_function
import numpy as np

def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

def sigmoid_derivative(output):
  return output*(1-output)

#loading data
print("loading data...")
images = np.loadtxt("TrainDigitX.csv", delimiter=',') 
labels = np.loadtxt("TrainDigitY.csv", delimiter=',')
test_images = np.loadtxt("TestDigitX.csv", delimiter=',') 
test_labels = np.loadtxt("TestDigitY.csv", delimiter=',')
N_data = len(images)
N_holdout = int(N_data / 5)
holdout_x = images[N_data - N_holdout:N_data]
holdout_y = labels[N_data - N_holdout:N_data]
N_train = N_data - N_holdout
N_input  = 785 #28*28+1
N_output = 10
print("data loaded")

def train(N_hidden = 256, epoch = 55, learning_rate = 0.1, experiment = False):
  N_hidden += 1 #adding the bias neuron to the hidden layer

  # Initial weights are normalized (column normalized)
  # For the weight from neuron s to neuron t: w_{s->t} = weight[s][t]
  # Given a layer, the weight respective to that layer 
  # to neuron t of the next layer is the t-th column of the weight matrix
  # weight from input layer to hidden layer
  weight_1 = np.random.random((N_input, N_hidden))
  weight_1 /= weight_1.sum(axis=0)[:None]
  # weight from hidden layer to output layer
  weight_2 = np.random.random((N_hidden, N_output))
  weight_2 /= weight_2.sum(axis=0)[:None]

  for ep in range(epoch):
    for i in range(N_train):
      # Neural Network Layers
      input_layer = np.append(images[i], [1])
      hidden_layer = sigmoid(np.dot(input_layer, weight_1))
      output_layer = sigmoid(np.dot(hidden_layer, weight_2))

      # generating traing label
      data_label = np.zeros((1,N_output))
      data_label[0][int(labels[i])] = 1

      error = (output_layer - data_label)
      #deltas of output layer
      delta_2 = 2 * sigmoid_derivative(output_layer) * error 
      #deltas of hidden layer
      delta_1 = sigmoid_derivative(hidden_layer) * np.dot(weight_2, delta_2.T).T 
      #updating weights
      weight_2 = weight_2 - learning_rate * np.dot(np.atleast_2d(hidden_layer).T, np.atleast_2d(delta_2))
      weight_1 = weight_1 - learning_rate * np.dot(np.atleast_2d(input_layer).T, np.atleast_2d(delta_1))
    if experiment == True:
      if (ep+1)%5 == 0: 
        print("epoch:", ep+1)
        test(weight_1, weight_2)
    else:
      # Holdout Error
      err = 0
      for i in range(N_holdout):
        input_layer = np.append(holdout_x[i], [1]).reshape(1,N_input)
        hidden_layer = sigmoid(np.dot(input_layer, weight_1))
        output_layer = sigmoid(np.dot(hidden_layer, weight_2))
        pred = np.argmax(output_layer)
        if pred != holdout_y[i]:
          err += 1
      if ep%10 != 0:
        continue
      print("holdout error: ", err/float(N_holdout))
  #save weights
  np.savetxt("weight_1.txt", weight_1, delimiter = ",")
  np.savetxt("weight_2.txt", weight_2, delimiter = ",")

def test(weight_1, weight_2):
  # Testing Error
  N_test = len(test_images)
  corr = 0
  for i in range(N_test):
    input_layer = np.append(test_images[i], [1]).reshape(1, N_input)
    hidden_layer = sigmoid(np.dot(input_layer, weight_1))
    output_layer = sigmoid(np.dot(hidden_layer, weight_2))
    pred = np.argmax(output_layer)
    if pred == test_labels[i]:
      corr += 1
  print("testing error: ", round(100 * (1 - corr/float(N_test)),3), "%")

def predictDigitX():
  #loading weigts
  weight_1 = np.loadtxt("weight_1.txt", delimiter = ",")
  weight_2 = np.loadtxt("weight_2.txt", delimiter = ",")

  data_x = np.loadtxt("TestDigitX.csv", delimiter = ",")
  f = open("PredDigitY.csv", "w+")
  N = len(data_x)
  for i in range(N):
    image = data_x[i]
    input_layer = np.append(image, [1])
    hidden_layer = sigmoid(np.dot(input_layer, weight_1))
    output_layer = sigmoid(np.dot(hidden_layer, weight_2))
    pred = np.argmax(output_layer)
    f.write(str(pred)+"\n")
  f.close()

def predictDigitX2():
  #loading weigts
  weight_1 = np.loadtxt("weight_1.txt", delimiter = ",")
  weight_2 = np.loadtxt("weight_2.txt", delimiter = ",")

  data_x = np.loadtxt("TestDigitX2.csv", delimiter = ",")
  f = open("PredDigitY2.csv", "w+")
  N = len(data_x)
  for i in range(N):
    image = data_x[i]
    input_layer = np.append(image, [1])
    hidden_layer = sigmoid(np.dot(input_layer, weight_1))
    output_layer = sigmoid(np.dot(hidden_layer, weight_2))
    pred = np.argmax(output_layer)
    f.write(str(pred)+"\n")
  f.close()

def experiment():
  # experiment of learning rate
  # set N_hidden = 256 and epochs = 40
  etas = [1,0.5,0.1,0.05,0.01,0.001]
  for eta in etas:
    print("------ Learning Rate:", eta, "------")
    train(256, 40, eta)
    # load weights
    weight_1 = np.loadtxt("weight_1.txt", delimiter = ",")
    weight_2 = np.loadtxt("weight_2.txt", delimiter = ",")
    test(weight_1, weight_2)
    print("--------------------------")
  
  # experiment of N_hidden
  # set epochs = 40, and learning rate being 0.01
  N_hiddens = [32,64,128,256]
  for N_hidden in N_hiddens:
    print("------ Hidden Units:", N_hidden, "------")
    train(N_hidden, 40, 0.1)
    # load weights
    weight_1 = np.loadtxt("weight_1.txt", delimiter = ",")
    weight_2 = np.loadtxt("weight_2.txt", delimiter = ",")
    test(weight_1, weight_2)
    print("--------------------------")
  # experiment of epochs
  # set N_hidden = 256, and learning rate being 0.01
  epoch_max = 200
  train(256, epoch_max, 0.1, True)

if __name__ == "__main__":
  print("begin trainig")
  train()
  print("training finished")
  # Loading data
  weight_1 = np.loadtxt("weight_1.txt", delimiter = ',')
  weight_2 = np.loadtxt("weight_2.txt", delimiter = ',')
  print("begin testing")
  test(weight_1, weight_2)

