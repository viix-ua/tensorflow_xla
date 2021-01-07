
# https://www.sqlshack.com/implement-artificial-neural-networks-anns-in-sql-server/

# An Artificial Neural Network (ANN) can be considered as a classification and as a forecasting technique.
# In Microsoft Neural Network uses tanh as the hidden layer activation function and sigmoid function for the output layer.

# HIDDEN_NODE_RATIO

# This parameter specifies a number used in determining the number of nodes in the hidden layer. 
# The algorithm calculates the number of nodes in the hidden layer as HIDDEN_NODE_RATIO * sqrt({the number of input nodes} * {the number of output nodes}).
# By default HIDDEN_NODE_RATIO = 4
