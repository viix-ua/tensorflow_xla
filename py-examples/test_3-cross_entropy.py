# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf

cex = tf.Variable([[]])

x_logits = tf.Variable(np.array([
    [1., 1., 2., 4., 6., 1.],
    [ 1., 1., 8., 10., 12., 1.]]),
    dtype=tf.float32)

x_labels = tf.Variable(np.array([
    [1., 0., 0., 0., 0., 1.],
    [1., 1., 0., 0., 0., 0.]])
    , dtype=tf.float32)

# (x_logits == x_labels) should be equal size
cex = tf.nn.softmax_cross_entropy_with_logits(logits=x_logits, labels=x_labels)


init = tf.global_variables_initializer()
init_local = tf.local_variables_initializer()

with tf.Session() as sess:
  sess.run([init, init_local])

  print(cex.eval(), [cex])
  print("******************************")