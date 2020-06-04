/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

//#include "tensorflow/core/kernels/eigen_pooling.h"

//#include "tensorflow/core/framework/types.h"

#include "test_helpers.h"
//#include "tensorflow/core/platform/test.h"

#include "array4d.h"
#include "padding.h"
#include "reference_util.h"


namespace Eigen {

namespace {
void EigenApprox(float a, float b) {
  ASSERT_TRUE(std::abs(a - b) <= std::min(std::abs(a), std::abs(b)) * 1e-3);
}

std::unique_ptr<xla::Array4D<float>> 
SpatialMaxPooling(const xla::Array4D<float>& input, int patch_rows, int patch_cols, int stride, int stride, xla::Padding padding)
{
   auto result = xla::MakeUnique<xla::Array4D<float>>(window_counts[0], window_counts[1], window_counts[2], window_counts[3]);

   return result;
}

}

class EigenPoolingTest
{
public:

   void Simple();
   //void SimpleRowMajor();

   void Strided();
   //void StridedRowMajor();

   void run()
   {
      Simple();
      Strided();
   }
};

void EigenPoolingTest::Simple() 
{
  const int depth = 10;
  const int input_rows = 5;
  const int input_cols = 5;
  const int num_batches = 13;
  const int patch_rows = 4;
  const int patch_cols = 4;
  const int output_rows = 2;
  const int output_cols = 2;

  xla::Array4D<float> input(num_batches, depth, input_rows, input_cols);

////////////////////////////////////////////////////////////////////////

  std::vector<tensorflow::int64> dim_lengths{ input.n1(), input.n2(), input.n3(), input.n4() };

  std::vector<tensorflow::int64> window_counts(window.size(), 0);
  std::vector<tensorflow::int64> pad_low(window.size(), 0);
  for (tensorflow::int64 i = 0; i < window.size(); ++i)
  {
     window_counts[i] =
        xla::ReferenceUtil::WindowCount(dim_lengths[i], window[i], stride[i], padding);
     pad_low[i] = padding_both[i].first;
  }

  EXPECT_EQ(result.n1(), num_batches);
  EXPECT_EQ(result.n2(), depth);
  EXPECT_EQ(result.n3(), output_rows);
  EXPECT_EQ(result.n4(), output_cols);

////////////////////////////////////////////////////////////////////////
  xla::Array4D<float> result(num_batches, depth, output_rows, output_cols);
  
  input.FillRandom(11.f);
  //result.setRandom();
  //result = result.constant(-1000.f);
  result.FillWithMultiples(-1000.f);

  // Max pooling using a 4x4 window and a stride of 1.
  const int stride = 1;
  result = Eigen::SpatialMaxPooling(input, patch_rows, patch_cols, stride, stride, xla::Padding::kValid);

  EXPECT_EQ(result.n1(), num_batches);
  EXPECT_EQ(result.n2(), depth);
  EXPECT_EQ(result.n3(), output_rows);
  EXPECT_EQ(result.n4(), output_cols);


  for (int b = 0; b < num_batches; ++b) 
  {
    for (int d = 0; d < depth; ++d) 
    {
      for (int i = 0; i < output_rows; ++i) 
      {
        for (int j = 0; j < output_cols; ++j) 
        {
          float expected = -10000.f;
          for (int r = 0; r < patch_rows; ++r) 
          {
            for (int c = 0; c < patch_cols; ++c) 
            {
              expected = (std::max)(expected, input(b, d, r + i, c + j));
            }
          }
          if (result(b, d, i, j) != expected) 
          {
            std::cout << "at d=" << d << " b=" << b << " i=" << i << " j=" << j
                      << " " << result(b, d, i, j) << " vs " << expected
                      << std::endl;
          }
          EigenApprox(result(d, i, j, b), expected);
        }
      }
    }
  }
}

/*
void EigenPoolingTest::SimpleRowMajor() 
{
  const int depth = 10;
  const int input_rows = 5;
  const int input_cols = 5;
  const int num_batches = 13;
  const int patch_rows = 4;
  const int patch_cols = 4;
  const int output_rows = 2;
  const int output_cols = 2;

  xla::Array4D<float, RowMajor> input(num_batches, input_cols, input_rows, depth);
  xla::Array4D<float, RowMajor> result(num_batches, output_cols, output_rows,
                                    depth);
  input = input.constant(11.0f) + input.random();
  result.setRandom();
  result = result.constant(-1000.f);

  // Max pooling using a 4x4 window and a stride of 1.
  const int stride = 1;
  result = SpatialMaxPooling(input, patch_rows, patch_cols, stride, stride,
                             PADDING_VALID);

  EXPECT_EQ(result.dimension(3), depth);
  EXPECT_EQ(result.dimension(2), output_rows);
  EXPECT_EQ(result.dimension(1), output_cols);
  EXPECT_EQ(result.dimension(0), num_batches);

  for (int b = 0; b < num_batches; ++b) {
    for (int d = 0; d < depth; ++d) {
      for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < output_cols; ++j) {
          float expected = -10000.f;
          for (int r = 0; r < patch_rows; ++r) {
            for (int c = 0; c < patch_cols; ++c) {
              expected = (std::max)(expected, input(b, c + j, r + i, d));
            }
          }
          if (result(b, j, i, d) != expected) {
            std::cout << "at d=" << d << " b=" << b << " i=" << i << " j=" << j
                      << " " << result(b, j, i, d) << " vs " << expected
                      << std::endl;
          }
          EigenApprox(result(b, j, i, d), expected);
        }
      }
    }
  }
}
*/

void EigenPoolingTest::Strided() 
{
  const int depth = 10;
  const int input_rows = 5;
  const int input_cols = 5;
  const int num_batches = 13;
  const int patch_rows = 3;
  const int patch_cols = 3;
  const int output_rows = 2;
  const int output_cols = 2;

  xla::Array4D<float> input(depth, input_rows, input_cols, num_batches);
  xla::Array4D<float> result(depth, output_rows, output_cols, num_batches);
  
  input = input.constant(11.0f) + input.random();
  result.setRandom();

  // Max pooling using a 3x3 window and a stride of 2.
  int stride = 2;
  result = Eigen::SpatialMaxPooling(input, patch_rows, patch_cols, stride, stride, xla::Padding::kValid);

  EXPECT_EQ(result.n1(), num_batches);
  EXPECT_EQ(result.n2(), depth);
  EXPECT_EQ(result.n3(), output_rows);
  EXPECT_EQ(result.n4(), output_cols);


  for (int b = 0; b < num_batches; ++b) {
    for (int d = 0; d < depth; ++d) {
      for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < output_cols; ++j) {
          float expected = -10000.f;
          for (int r = 0; r < patch_rows; ++r) {
            for (int c = 0; c < patch_cols; ++c) {
              expected = (std::max)(
                  expected, input(d, r + stride * i, c + stride * j, b));
            }
          }
          if (result(d, i, j, b) != expected) {
            std::cout << "at d=" << d << " b=" << b << " i=" << i << " j=" << j
                      << " " << result(d, i, j, b) << " vs " << expected
                      << std::endl;
          }
          EigenApprox(result(d, i, j, b), expected);
        }
      }
    }
  }
}

/*
void EigenPoolingTest::StridedRowMajor() 
{
  const int depth = 10;
  const int input_rows = 5;
  const int input_cols = 5;
  const int num_batches = 13;
  const int patch_rows = 3;
  const int patch_cols = 3;
  const int output_rows = 2;
  const int output_cols = 2;

  Tensor<float, 4, RowMajor> input(num_batches, input_cols, input_rows, depth);
  Tensor<float, 4, RowMajor> result(num_batches, output_cols, output_rows,
                                    depth);
  input = input.constant(11.0f) + input.random();
  result.setRandom();

  // Max pooling using a 3x3 window and a stride of 2.
  int stride = 2;
  result = SpatialMaxPooling(input, patch_rows, patch_cols, stride, stride,
                             PADDING_VALID);

  EXPECT_EQ(result.dimension(3), depth);
  EXPECT_EQ(result.dimension(2), output_rows);
  EXPECT_EQ(result.dimension(1), output_cols);
  EXPECT_EQ(result.dimension(0), num_batches);

  for (int b = 0; b < num_batches; ++b) {
    for (int d = 0; d < depth; ++d) {
      for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < output_cols; ++j) {
          float expected = -10000.f;
          for (int r = 0; r < patch_rows; ++r) {
            for (int c = 0; c < patch_cols; ++c) {
              expected = (std::max)(
                  expected, input(b, c + stride * j, r + stride * i, d));
            }
          }
          if (result(b, j, i, d) != expected) {
            std::cout << "at d=" << d << " b=" << b << " i=" << i << " j=" << j
                      << " " << result(b, j, i, d) << " vs " << expected
                      << std::endl;
          }
          EigenApprox(result(b, j, i, d), expected);
        }
      }
    }
  }
}
*/

}  // namespace Eigen
