
/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <initializer_list>

#include "array2d.h"
#include "array3d.h"

#include "reference_util.h"

#include "test_helpers.h"

//#include "padding_test.cc"


//#include "reference_util_test.cc"
//#include "literal_util_test.cc"

#include "test_utils.h"
#include "ptr_util.h"

#include "xla_data.pb.h"

#include "test_utils.h"
/////////////////////////////////////////////
#include "computation.h"
#include "computation_builder.h"
/////////////////////////////////////////////
//#include "pad_test.cc"
#include "client_library_test_base.h"
#include "arithmetic.h"

//#include "convolution_test.cc"

////////////////////////////////////////////
//#include "convolution_variants_test.cc"
#include <assert.h>

//#include "reduce_window_test.cc"

//#include "reshape_test.cc"

//#include "pooling_test.cpp"

#include "image.h"

#include "nnet.h"

using namespace tensorflow;

// no need Bazel, Protobuff, GTest, Eigen, CudaNN and other external lib-dependency.

// https://www.tensorflow.org/get_started/mnist/beginners
// https://www.tensorflow.org/get_started/mnist/pros


// TODO: GradientDescentOptimizer
// https://stackoverflow.com/questions/41918795/minimize-a-function-of-one-variable-in-tensorflow
// https://stackoverflow.com/questions/33919948/how-to-set-adaptive-learning-rate-for-gradientdescentoptimizer
// https://stackoverflow.com/questions/37921781/what-does-opt-apply-gradients-do-in-tensorflow

// https://stackoverflow.com/questions/42468292/gradientdescentoptimizer-got-wrong-result -->> sgd impl
// https://stackoverflow.com/questions/38067443/how-to-use-tensorflow-gradient-descent-optimizer-to-solve-optimization-problems

// https://stackoverflow.com/questions/34911276/cannot-gather-gradients-for-gradientdescentoptimizer-in-tensorflow
// https://stackoverflow.com/questions/45901523/tensorflow-optimizer-minimize-function

// https://stackoverflow.com/questions/35226428/how-do-i-get-the-gradient-of-the-loss-at-a-tensorflow-variable
// https://stackoverflow.com/questions/46255221/difference-between-gradientdescentoptimizer-and-adamoptimizer-in-tensorflow *

// conv2d:
// https://stackoverflow.com/questions/35565312/is-there-a-convolution-function-in-tensorflow-to-apply-a-sobel-filter


//#include "GradientDescentOptimizer.h"
#include "trainer_base_lr_sgd.h"


namespace xla
{
   template <typename TType> inline
   void solve(const xla::Array2D<TType>& coeff, const std::vector<TType>& y)
   {
      xla::Array2D<TType> _a = coeff;
      xla::Array2D<TType> _b(coeff.Height(), 1, y);
      xla::Array2D<TType> _x(coeff.Height(), 1, TType(0));

      TType learn_rate = 0.1;
      size_t max_epoch = 1;

      for (size_t i = 0; i < max_epoch; i++)
      {
         auto grad_loss = MakeMatrixMul(*xla::Transpose(_a), *MakeMatrixMul(_a, _x) - _b);

         grad_loss->mul(learn_rate);
         _x = _x - (*grad_loss);

         TType loss = 0.f; // ReferenceUtil::ReduceMean(xla::Square(*MakeMatrixMul(_a, _x) - _b));

         if (loss < 0.0001f)
         {
            break;
         }
      }
   }
}


int main()
{
   xla::Array2D<double> x = { { 5.0, 6.0, 4.0 } };          // matrix (n_classes, n_features)
   xla::Array2D<double> w = { { 1.0 }, { 2.0 }, { 3.0 } };  // matrix (n_features, n_classes)

   xla::Array2D<double> b = { { 0.0 } };                    // matrix (n_classes, n_classes)

   auto mmul = xla::MakeMatrixMul(x, w);

   auto y = *mmul + b;

   LOG_MSG("", y);

   ////////////////////////////////////////


   // SGD: https://stackoverflow.com/questions/41822308/how-tf-gradients-work-in-tensorflow
   // + src: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/linear_regression.ipynb

   //////////////////////////////////////////
   // TODO: softmax, cross_entropy
   // http://www.geeksforgeeks.org/softmax-regression-using-tensorflow/
   // https://stackoverflow.com/questions/34240703/difference-between-tensorflow-tf-nn-softmax-and-tf-nn-softmax-cross-entropy-with
   // https://stackoverflow.com/questions/34240703/whats-the-difference-between-softmax-and-softmax-cross-entropy-with-logits
   // TODO: https://www.oreilly.com/ideas/visualizing-convolutional-neural-networks


   // TODO: implement https://www.tensorflow.org/versions/r0.12/resources/xla_prerelease

   /*
   xla::ReferenceUtilTest referenceUtilTest;
   referenceUtilTest.run();

   xla::PoolingTest poolingTest;
   poolingTest.run();

   xla::ConvolutionTest convolutionTest;
   convolutionTest.run();

   xla::ConvolutionVariantsTest convolutionVariantsTest;
   convolutionVariantsTest.run();

   xla::ReduceWindowTest reduceWindowTest;
   reduceWindowTest.run();

   xla::PadTest padTest;
   padTest.run();
   */

   // TODO: error now
   //xla::PaddingTest paddingTest;
   //paddingTest.run();

   // TODO:
   // mnist_deep.py

   // TODO:
   // https://www.tensorflow.org/versions/master/api_docs/python/tf/reshape

   // MNIST for training:
   // https://www.tensorflow.org/get_started/mnist/beginners 
   // -->> https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/mnist/mnist_softmax.py

   // https://www.tensorflow.org/get_started/mnist/pros

   // TODO: reduce_sum, reduce_mean
   //TODO: dropout

   return 0;
}
