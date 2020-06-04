#pragma once

// https://stackoverflow.com/questions/37890394/tensorflow-gradientdescentoptimizer-how-does-it-connect-to-tf-variables

// y = W * x + b
// cost_func = tf.nn.l2_loss(y_ - y)  # squared error
// trainer = tf.train.GradientDescentOptimizer(0.01).minimize(cost_func)

// https://learningtensorflow.com/linear_equations/

// https://stackoverflow.com/questions/42468292/gradientdescentoptimizer-got-wrong-result

// print(s.solve([[1, 2], [1, 3]], [3, 4]))

#include "array2d.h"
#include "reference_util.h"

