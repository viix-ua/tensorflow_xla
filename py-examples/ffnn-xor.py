
# medium title source: Intuition + Mathematics + Python behind Basic 3 Layers Neural Network (Mar 4, 2020)

import numpy as np

# Cost function
def sse(y_true, y_pred):
  return 0.5 * np.sum(np.power(y_pred - y_true, 2))

# Activation function
def sigmoid(X):
  return 1.0 / (1.0 + np.exp(-X))

# Activation prime function
def sigmoid_prime(x):
  sig = sigmoid(x)
  return sig * (1.0 - sig)

# Feed Forward Neural Network
class FFNN:
  def __init__(self, input_layer, hidden_layer, output_layer):
    # Number of neurons for each layer
    self.hidden_layer = hidden_layer
    self.output_layer = output_layer
    self.input_layer = input_layer
    # Initialisation of parameters W1, W2, B1, B2
    self.W1 = np.random.rand(input_layer, hidden_layer)
    self.W2 = np.random.rand(hidden_layer, output_layer)
    self.B1 = np.zeros(hidden_layer)
    self.B2 = np.zeros(output_layer)


  # Method to train our model
  def fit(self, X_train, y_train, learning_rate=0.1, epochs=10000):
    # Epoch loop
    for epoch in range(epochs):
      error = 0.0
      # Training-set Loop
      for x, y_true in zip(X_train, y_train):
        
        # Forward Propagation
        h1_ = np.dot(x, self.W1) + self.B1
        h1 = sigmoid(h1_)
        y_ = np.dot(h1, self.W2) + self.B2
        y = sigmoid(y_)
        
        # Error computation
        error += sse(y_true, y)
        
        # Backward Propagation
        dW1 = np.zeros((self.input_layer, self.hidden_layer))
        dW2 = np.zeros((self.hidden_layer, self.output_layer))
        dB1 = np.zeros(self.hidden_layer)
        dB2 = np.zeros(self.output_layer)

        for k in range(self.output_layer):
          dB2[k] = (y[k] - y_true[k]) * sigmoid_prime(y_[k])

        for j in range(self.hidden_layer):
          for k in range(self.output_layer):
            dW2[j][k] =  dB2[k] * h1[j]

        for j in range(self.hidden_layer):
          dB1[j] = sum([dB2[k] * self.W2[j][k] * sigmoid_prime(h1_[j]) for k in range(self.output_layer)])

        for i in range(self.input_layer):
          for j in range(self.hidden_layer):
            dW1[i][j] = dB1[j] * x[i]

        # Gradient Descente
        self.B1 -= learning_rate * dB1
        self.W1 -= learning_rate * dW1
        self.B2 -= learning_rate * dB2
        self.W2 -= learning_rate * dW2

      # Show error at each epoch
      error /= X_train.shape[0]
      print(f"[INFO]: epoch = {epoch + 1} | error = {error}")

  # Method to do a single prediction
  def predict(self, X_test):
    # Forward Propagation
    h1_ = np.dot(X_test, self.W1) + self.B1
    h1 = sigmoid(h1_)
    y_ = np.dot(h1, self.W2) + self.B2
    y = sigmoid(y_)
    return y

# Dataset
x_train = np.array([[0,0], [0,1], [1,0], [1,1]])
y_train = np.array([[0], [1], [1], [0]])

# Network object creation and training
HIDDEN_NODE_RATIO = 4
input_nodes = 2
output_nodes = 1
hidden = int( HIDDEN_NODE_RATIO * np.sqrt(input_nodes * output_nodes) ) # hidden=5

fnn = FFNN(input_nodes, hidden, output_nodes)
fnn.fit(x_train, y_train)

#Prediction
print()
print('Input:', x_train)
#print('Prediction:', np.around(fnn.predict(x_train)))
print('Prediction:', fnn.predict(x_train))
print('Label:', y_train)
