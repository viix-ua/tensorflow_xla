from __future__ import print_function
import numpy as np
import tensorflow as tf

patch_x = 2
patch_y = 3
channels = 1 #greyscale
depth = 2

# size = 2x1x5x7 (NCHW)
x = tf.constant([
[
    [
    [1., 1., 1., 1.,  1.,  1., 1.],
    [1., 2., 1., 1.,  1.,  1., 1.],
    [1., 1., 2., 4.,  6.,  1., 1.],
    [1., 1., 8., 10., 12., 1., 1.],
    [1., 1., 1., 1.,  1.,  1., 1.]
    ]
]
,
[
    [
    [0., 0., 0., 0.,  0.,  0., 0.],
    [0., 2., 1., 1.,  1.,  1., 0.],
    [0., 1., 2., 4.,  6.,  1., 0.],
    [0., 1., 8., 10., 12., 1., 0.],
    [0., 0., 0., 0.,  0.,  0., 0.]
    ]
]
])

# convert from NCHW to NHWC with a well chosen permutation
# perm[0] = 0  # output dimension 0 will be 'N', which was dimension 0 in the input
# perm[1] = 2  # output dimension 1 will be 'H', which was dimension 2 in the input
# perm[2] = 3  # output dimension 2 will be 'W', which was dimension 3 in the input
# perm[3] = 1  # output dimension 3 will be 'C', which was dimension 1 in the input

x = tf.transpose(x, [0, 2, 3, 1])

# Variables.
layer1_weights = tf.Variable(tf.truncated_normal(
      [patch_y, patch_x, channels, depth], stddev=0.1))
layer1_biases = tf.Variable(tf.fill([depth], 1.5))

###################################################################################
kernel = tf.Variable(tf.ones([patch_y, patch_x, channels, depth], name = 'kernel'))

convSame = tf.nn.conv2d(x, kernel, [1, 2, 2, 1], padding = 'SAME');

hidden = tf.nn.relu(convSame + layer1_biases)
###################################################################################

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    print(layer1_biases.eval(), [layer1_biases])

    print(convSame.eval(), [convSame])
    print(hidden.eval(), [hidden])
    print("***********************************")
