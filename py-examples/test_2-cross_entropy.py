# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f, encoding='latin1')
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory

  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10 # = num_classes

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print('Training set', train_dataset.shape, train_labels.shape)   # (20000, 28, 28) (20000,)
print('Validation set', valid_dataset.shape, valid_labels.shape) # (1000, 28, 28) (1000,)
print('Test set', test_dataset.shape, test_labels.shape)         # (1000, 28, 28) (1000,)


# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
train_subset = 225#10000

graph = tf.Graph()
with graph.as_default():
  # Input data.
  # Load the training, validation and test data into constants that are
  # attached to the graph.
  tf_train_dataset = tf.constant(train_dataset[:train_subset, :])# = train_subset x 784
  tf_train_labels = tf.constant(train_labels[:train_subset])     # = train_subset x num_labels

  # Variables.
  # These are the parameters that we are going to be training. The weight
  # matrix will be initialized using random values following a (truncated)
  # normal distribution. The biases get initialized to zero.
  weights = tf.Variable(
    tf.truncated_normal([image_size * image_size, num_labels]))
  biases = tf.Variable(tf.zeros([num_labels]))

  # Training computation.
  # We multiply the inputs with the weight matrix, and add biases. We compute
  # the softmax and cross-entropy (it's one operation in TensorFlow, because
  # it's very common, and it can be optimized). We take the average of this
  # cross-entropy across all training examples: that's our loss.
  logits = tf.matmul(tf_train_dataset, weights) + biases #train_subset x num_labels
  ce = tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)
  loss = tf.reduce_mean(ce)

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print(tf_train_dataset.eval(), [tf_train_dataset])
  print(tf_train_labels.eval(), [tf_train_labels])
  print("******************************")