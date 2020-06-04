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

#include "reference_util.h"

#include <cmath>
#include <memory>

#include "array2d.h"
#include "array4d.h"
#include "padding.h"
#include "literal_util.h"
#include "ptr_util.h"
#include "literal_test_util.h"
#include "xla_data.pb.h"
//#include "test.h"

// https://www.tensorflow.org/api_guides/python/nn#Convolution
// https://www.tensorflow.org/api_guides/python/nn#Notes_on_SAME_Convolution_Padding
// http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html

namespace xla {
namespace {

   class PoolingTest
   {
   public:
      PoolingTest()
      {
      }
      
      void MaxPool_1x2x4x4_1x1();
      void MaxPool_1x2x4x4_2x2();

      void MaxPool_1x2x3x3_1x1();
      void MaxPool_1x2x3x3_2x2();

      void run();
   };

   void PoolingTest::run()
   {
      MaxPool_1x2x4x4_1x1();
      MaxPool_1x2x4x4_2x2();

      MaxPool_1x2x3x3_1x1();
      MaxPool_1x2x3x3_2x2();
   }

   void PoolingTest::MaxPool_1x2x4x4_1x1()
   {
      const xla::Array4D<float> input4d({
      {  // batch_1
         {  // channel_1
            { 3., 7., -2., 1. },
            { 1., 9.,  6., 2. },
            { 2., 3.,  8., 4. },
            { 5., -9., 6., 3. }
         }
         ,
         {  // channel_2
            { 3., -11., 12.,  5. },
            { 2.,  -3.,  4.,  0. },
            {-2.,  10., -8., -2. },
            { 7.,   1.,  0.,  6. }
         }
      }
      });

      const xla::Array4D<float> expected_same({
      {  // batch_1
         {  // channel_1
            { 9.,  9.,  6.,  2. },
            { 9.,  9.,  8.,  4. },
            { 5.,  8.,  8.,  4. },
            { 5.,  6.,  6.,  3. }
         }
         ,
         {  // channel_2
            { 3.,  12.,  12., 5. },
            { 10., 10.,  4.,  0. },
            { 10., 10.,  6.,  6. },
            { 7.,   1.,  6.,  6. }
         }
      }
      });

      const xla::Array4D<float> expected_valid({
      {  // batch_1
         {  // channel_1
            { 9.,  9.,  6. },
            { 9.,  9.,  8. },
            { 5.,  8.,  8. }
         }
         ,
         {  // channel_2
            { 3.,  12.,  12. },
            { 10., 10.,   4. },
            { 10., 10.,   6. }
         }
      }
      });

      auto result_same = xla::ReferenceUtil::Max_Pool(input4d, { 2, 2 }, { 1, 1 }, xla::Padding::kSame);

      ASSERT_EQ(*result_same, expected_same);

      auto result_valid = xla::ReferenceUtil::Max_Pool(input4d, { 2, 2 }, { 1, 1 }, xla::Padding::kValid);

      ASSERT_EQ(*result_valid, expected_valid);
   }

   void PoolingTest::MaxPool_1x2x4x4_2x2()
   {
      const xla::Array4D<float> input4d({
      {  // batch_1
         {  // channel_1
            { 3., 7., -2., 1. },
            { 1., 9.,  6., 2. },
            { 2., 3.,  8., 4. },
            { 5., -9., 6., 3. }
         }
         ,
         {  // channel_2
            { 3., -11., 12.,  5. },
            { 2.,  -3.,  4.,  0. },
            {-2.,  10., -8., -2. },
            { 7.,   1.,  0.,  6. }
         }
      }
      });

      const xla::Array4D<float> expected_same({
      {  // batch_1
         {  // channel_1
            { 9.,  6. },
            { 5.,  8. }
         }
         ,
         {  // channel_2
            { 3.,  12.},
            { 10., 6. }
         }
      }
      });

      const xla::Array4D<float> expected_valid({
      {  // batch_1
         {  // channel_1
            { 9.,  6. },
            { 5.,  8. }
         }
         ,
         {  // channel_2
            { 3.,  12. },
            { 10., 6. }
         }
      }
      });

      auto result_same = xla::ReferenceUtil::Max_Pool(input4d, { 2, 2 }, { 2, 2 }, xla::Padding::kSame);

      ASSERT_EQ(*result_same, expected_same);

      auto result_valid = xla::ReferenceUtil::Max_Pool(input4d, { 2, 2 }, { 2, 2 }, xla::Padding::kValid);

      ASSERT_EQ(*result_valid, expected_valid);
   }

   void PoolingTest::MaxPool_1x2x3x3_1x1()
   {
      const xla::Array4D<float> input_3x3({
      {  // batch_1
         {  // channel_1
            { 3., 7., -2. },
            { 1., 9.,  6. },
            { 2., 3., -4. }
         }
         ,
         {  // channel_2
            { 3., -11., 12. },
            { 2.,  -3.,  4. },
            {-2.,  10., -8. }
         }
      }
      });

      const xla::Array4D<float> expected_same({
      {  // batch_1
         {  // channel_1
            { 9.,  9.,  6. },
            { 9.,  9.,  6. },
            { 3.,  3.,  -4. }
         }
         ,
         {  // channel_2
            { 3.,  12.,  12.},
            { 10., 10.,  4. },
            { 10., 10., -8. }
         }
      }
      });

      const xla::Array4D<float> expected_valid({
      {  // batch_1
         {  // channel_1
            { 9.,  9. },
            { 9.,  9. }
         }
         ,
         {  // channel_2
            { 3.,  12. },
            { 10., 10. }
         }
      }
      });

      auto result_same = xla::ReferenceUtil::Max_Pool(input_3x3, { 2, 2 }, { 1, 1 }, xla::Padding::kSame);

      ASSERT_EQ(*result_same, expected_same);

      auto result_valid = xla::ReferenceUtil::Max_Pool(input_3x3, { 2, 2 }, { 1, 1 }, xla::Padding::kValid);

      ASSERT_EQ(*result_valid, expected_valid);
   }

   void PoolingTest::MaxPool_1x2x3x3_2x2()
   {
      const xla::Array4D<float> input_3x3({
      {  // batch_1
         {  // channel_1
            { 3., 7., -2. },
            { 1., 9.,  6. },
            { 2., 3., -4. }
         }
         ,
         {  // channel_2
            { 3., -11., 12. },
            { 2.,  -3.,  4. },
            {-2.,  10., -8. }
         }
      }
      });

      const xla::Array4D<float> expected_same({
      {  // batch_1
         {  // channel_1
            { 9.,  6. },
            { 3., -4. }
         }
         ,
         {  // channel_2
            { 3.,  12.},
            { 10., -8. }
         }
      }
      });

      const xla::Array4D<float> expected_valid({
      {  // batch_1
         {  // channel_1
            { 9. }
         }
         ,
         {  // channel_2
            { 3. }
         }
      }
      });

      auto result_same = xla::ReferenceUtil::Max_Pool(input_3x3, { 2, 2 }, { 2, 2 }, xla::Padding::kSame);

      ASSERT_EQ(*result_same, expected_same);

      auto result_valid = xla::ReferenceUtil::Max_Pool(input_3x3, { 2, 2 }, { 2, 2 }, xla::Padding::kValid);

      ASSERT_EQ(*result_valid, expected_valid);
   }


}  // ns
}  // xla
