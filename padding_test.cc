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

#include "padding.h"

//#include "test.h"

#include "test_helpers.h"

//#include "tensorflow/compiler/xla/client/padding.h"

//#include "tensorflow/core/platform/test.h"

#include "window_util.h"
#include "reference_util.h"

#include "client_library_test_base.h"

// https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t

// https://www.tensorflow.org/api_docs/python/tf/pad

/*
* VALID padding.The easiest case, means no padding at all.Just leave your data the same it was.
* SAME padding sometimes called HALF padding.It is called SAME because for a convolution 
   with a stride = 1, (or for pooling) it should produce output of the same size as the input.
   It is called HALF because for a kernel of size k, p = k / 2
* FULL padding is the maximum padding which does not result in a convolution 
   over just padded elements. For a kernel of size k, this padding is equal to k - 1.
*/
namespace xla
{
namespace {

inline 
std::pair<int64, int64> ComputeSamePadding(int padding_size)
{
   std::pair<int64, int64> padding_low_high;

   //# if even.. easy..
   if (padding_size % 2 == 0)
   {
      int64 pad_low = padding_size / 2;
      int64 pad_high = pad_low;

      padding_low_high.first = pad_low;
      padding_low_high.second = pad_high;

      LOG_MSG("", pad_low, pad_high);
   }
   //# if odd
   else
   {
      int64 pad_low = (int64)floor(double(padding_size) / 2.0);
      int64 pad_high = (int64)floor(double(padding_size) / 2.0) + 1;

      padding_low_high.first = pad_low;
      padding_low_high.second = pad_high;

      LOG_MSG("", pad_low, pad_high);
   }
   return padding_low_high;
}

//# strides[image index, y, x, depth]
//# padding 'SAME' or 'VALID'
//# bottom and right sides always get the one additional padded pixel(if padding is odd)
inline 
std::pair<int64, int64> getDimPadding(int input, int filter, int stride, xla::Padding padding)
{
   if (padding == xla::Padding::kSame)
   {
      int output = (int)ceil(float(input) / float(stride));


      int padding_size = ((output - 1) * stride + filter - input);


      //// now get padding
      return ComputeSamePadding(padding_size);

   }
   else if (padding == xla::Padding::kValid)
   {
      int output = (int)ceil(float(input - filter + 1) / float(stride));
      LOG_MSG("", output);

      return std::pair<int64, int64>(0, 0);
   }
   else
   {
      assert(false);
      return std::pair<int64, int64>(0, 0);
   }
}

inline void test_array_convolution()
{
   int64 window_cntr = xla::ReferenceUtil::WindowCount(4, 2, 1, xla::Padding::kValid);

/*
"VALID" = without padding:
   inputs:         1  2  3  4  5  6  7  8  9  10 11 (12 13)
                  |________________|                dropped
                                 |_________________|

"SAME" = with zero padding:
               pad|                                      |pad
   inputs:      0 |1  2  3  4  5  6  7  8  9  10 11 12 13|0  0
               |________________|
                              |_________________|
                                             |________________|

*/
   auto array_ops_valid = getDimPadding(13, 6, 5, xla::Padding::kValid);

   auto array_ops_same = getDimPadding(13, 6, 5, xla::Padding::kSame);

   auto cnt1 = xla::ReferenceUtil::WindowCount(13, 6, 5, xla::Padding::kValid);
   auto cnt2 = xla::ReferenceUtil::WindowCount(13, 6, 5, xla::Padding::kSame);

   auto padding1 = xla::MakePadding({ 13 }, { 6 }, { 5 }, xla::Padding::kValid);
   auto padding2 = xla::MakePadding({ 13 }, { 6 }, { 5 }, xla::Padding::kSame);

///////////////////////////////////////////////////

   //auto array_ops_2 = getOutputDim(4, 2, 2, xla::Padding::kSame);

   auto array_ops_2 = getDimPadding(2, 2, 2, xla::Padding::kSame);

   auto cnt22 = xla::ReferenceUtil::WindowCount(4, 2, 2, xla::Padding::kSame);

   LOG_MSG("", window_cntr, cnt1, cnt2, cnt22);
   LOG_MSG("", array_ops_valid, array_ops_same, array_ops_2);
}


// Tests MakePadding utility function for various cases.

  // A convenience function to test padding for a single dimension.
  inline std::pair<int64, int64> ComputePadding(int64 input_dimension,
                                         int64 window_dimension,
                                         int64 window_stride, Padding padding)
  {
    return MakePadding({input_dimension}, {window_dimension}, {window_stride},
                       padding)[0];
  }


class PaddingTest : public ClientLibraryTestBase
{
public:
   PaddingTest() { run(); }

   void ValidPaddingWithStrideOne();
   void ValidPaddingWithStrideThree();
   void SamePaddingWithOddWindow();
   void SamePaddingWithEvenWindow();
   void SamePaddingWithOddWindowWithStride();
   void SamePaddingWithEvenWindowWithStride();
   void SamePaddingForWindowSizeOne();
   void SamePaddingForWindowLargerThanInput();
   void NonNegativePadding();

   void run();
};


void PaddingTest::ValidPaddingWithStrideOne()
{
  const auto padding = ComputePadding(10, 5, 1, xla::Padding::kValid);
  EXPECT_EQ(padding.first, 0);
  EXPECT_EQ(padding.second, 0);

  const auto padding_test = getDimPadding(10, 5, 1, xla::Padding::kValid);

  EXPECT_EQ(padding.first, padding_test.first);
  EXPECT_EQ(padding.second, padding_test.second);
}


void PaddingTest::ValidPaddingWithStrideThree()
{
  const auto padding = ComputePadding(10, 5, 3, xla::Padding::kValid);
  EXPECT_EQ(padding.first, 0);
  EXPECT_EQ(padding.second, 0);

  const auto padding_test = getDimPadding(10, 5, 3, xla::Padding::kValid);

  EXPECT_EQ(padding.first, padding_test.first);
  EXPECT_EQ(padding.second, padding_test.second);
}


void PaddingTest::SamePaddingWithOddWindow()
{
  const auto padding = ComputePadding(10, 7, 1, xla::Padding::kSame);
  EXPECT_EQ(padding.first, 3);
  EXPECT_EQ(padding.second, 3);

  const auto padding_test = getDimPadding(10, 7, 1, xla::Padding::kSame);

  EXPECT_EQ(padding.first, padding_test.first);
  EXPECT_EQ(padding.second, padding_test.second);
}


void PaddingTest::SamePaddingWithEvenWindow()
{
  const auto padding = ComputePadding(10, 6, 1, xla::Padding::kSame);
  EXPECT_EQ(padding.first, 2);
  EXPECT_EQ(padding.second, 3);

  const auto padding_test = getDimPadding(10, 6, 1, xla::Padding::kSame);

  EXPECT_EQ(padding.first, padding_test.first);
  EXPECT_EQ(padding.second, padding_test.second);
}


void PaddingTest::SamePaddingWithOddWindowWithStride()
{
  const auto padding = ComputePadding(10, 7, 3, xla::Padding::kSame);
  EXPECT_EQ(padding.first, 3);
  EXPECT_EQ(padding.second, 3);

  const auto padding_test = getDimPadding(10, 7, 3, xla::Padding::kSame);

  EXPECT_EQ(padding.first, padding_test.first);
  EXPECT_EQ(padding.second, padding_test.second);
}


void PaddingTest::SamePaddingWithEvenWindowWithStride()
{
  const auto padding = ComputePadding(10, 6, 4, xla::Padding::kSame);
  EXPECT_EQ(padding.first, 2);
  EXPECT_EQ(padding.second, 2);

  const auto padding_test = getDimPadding(10, 6, 4, xla::Padding::kSame);

  EXPECT_EQ(padding.first, padding_test.first);
  EXPECT_EQ(padding.second, padding_test.second);
}


void PaddingTest::SamePaddingForWindowSizeOne()
{
  const auto padding = ComputePadding(10, 1, 1, xla::Padding::kSame);
  EXPECT_EQ(padding.first, 0);
  EXPECT_EQ(padding.second, 0);

  const auto padding_test = getDimPadding(10, 1, 1, xla::Padding::kSame);

  EXPECT_EQ(padding.first, padding_test.first);
  EXPECT_EQ(padding.second, padding_test.second);
}


void PaddingTest::SamePaddingForWindowLargerThanInput()
{
  const auto padding = ComputePadding(10, 20, 1, xla::Padding::kSame);
  EXPECT_EQ(padding.first, 9);
  EXPECT_EQ(padding.second, 10);

  const auto padding_test = getDimPadding(10, 20, 1, xla::Padding::kSame);

  EXPECT_EQ(padding.first, padding_test.first);
  EXPECT_EQ(padding.second, padding_test.second);
}

// This used to trigger a case with negative padding.
void PaddingTest::NonNegativePadding()
{
  const auto padding = ComputePadding(4, 1, 2, xla::Padding::kSame);
  EXPECT_EQ(padding.first, 0);
  EXPECT_EQ(padding.second, 0);

  const auto padding_test = getDimPadding(4, 1, 2, xla::Padding::kSame);

  EXPECT_EQ(padding.first, padding_test.first);
  EXPECT_EQ(padding.second, padding_test.second);
}

void PaddingTest::run()
{
   ValidPaddingWithStrideOne();
   ValidPaddingWithStrideThree();
   SamePaddingWithOddWindow();
   SamePaddingWithEvenWindow();
   SamePaddingWithOddWindowWithStride();
   SamePaddingWithEvenWindowWithStride();
   SamePaddingForWindowSizeOne();
   SamePaddingForWindowLargerThanInput();
   NonNegativePadding();
}

}  // }
}  // namespace xla
