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

// Tests of convolution variants -- kernel sizes, padding, and strides --
// in small sized data.

#include <algorithm>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "array4d.h"
#include "computation_builder.h"
//#include "tensorflow/compiler/xla/client/local_client.h"
#include "padding.h"
//#include "tensorflow/compiler/xla/legacy_flags/cpu_compiler_flags.h"
#include "reference_util.h"
#include "client_library_test_base.h"
#include "literal_test_util.h"
//#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "xla_data.pb.h"
//#include "test.h"
#include "base.h"

namespace xla {
namespace {

class ConvolutionVariantsTest : public ClientLibraryTestBase 
{
   public:
#if XLA_TEST_BACKEND_GPU
  // XLA:GPU sometimes uses FFT convolution which isn't as precise as spatial
  // convolution. So relax the absolute error threshold.
  ErrorSpec error_spec_ = ErrorSpec(1e-1.ff, 1e-5f);
#else
  ErrorSpec error_spec_ = ErrorSpec(1e-4f, 1e-2f);
#endif

  void Minimal();
  void MinimalWithBatch();

  void Flat1x1();
  void Deep1x1();

  void Filter1x2in1x2();
  void Filter1x2in1x3();
  void Filter1x2in2x2();
  void Filter2x1in2x2();
  void Filter2x2in2x2();

  void Filter1x2in2x3WithDepthAndBatch();

  void Filter1x1stride1x2in1x4();

  void Filter1x1stride1x2in1x5();

  void Filter1x3stride1x2in1x4();

  void Filter1x3stride1x2in1x5();

  void Filter1x1stride2x2in3x3();

  void Filter3x1in1x1Padded();
  void Filter5x1in3x1Padded();
  void Filter3x3in2x2Padded();

  void Filter1x1in2x1WithPaddingAndDepth();

  void Filter2x2Stride1x1Input3x3();

  void Filter1x2Stride1x1Input1x3();

  void Filter2x1x8x8Input1x1x8x8();

  void Filter1x1x1x1Input16x1x1x1();
  void Filter1x1x2x2Input16x1x2x2();

  void Filter1x1x2x2Input3x1x2x2();

  void Filter1x1x8x8Input16x1x8x8();

  void Filter2x2x8x8Input1x2x8x8();
  void Filter2x2x8x8Input2x2x8x8();
  void Filter2x2x8x8Input32x2x8x8();
  void Filter16x16x1x1Input16x16x1x1();

  void FlatRhsDilation();
  void FlatLhsDilation1D();
  void FlatLhsDilation();

  void NegativePaddingOnBothEnds();
  void NegativePaddingLowAndPositivePaddingHigh();
  void PositivePaddingLowAndNegativePaddingHigh();
  void PositivePaddingAndDilation();
  void NegativePaddingAndDilation();

  void RandomData_Input1x1x2x3_Filter2x1x1x2();
  void RandomData_Input1x16x1x1_Filter1x16x1x1();

  void RandomData_Input16x16x1x1_Filter1x16x1x1();

  void RandomData_Input16x16x1x1_Filter16x16x1x1();

  void RandomData_Input16x16x16x16_Filter16x16x16x16();

  void Filter1x2x1x1Input1x2x3x1GeneralPadding();

  void Filter1x1x1x1Input1x2x3x1GeneralPadding();

  void Filter1x1x1x1Input1x2x3x1NoPadding();
  void Filter1x1x2x3Input1x2x3x2NoPadding();

  void BackwardInputLowPaddingLessThanHighPadding();

  void BackwardInputLowPaddingGreaterThanHighPadding();

  void BackwardInputEvenPadding();

  void BackwardInputWithNegativePaddingHigh();
  void BackwardFilterLowPaddingLessThanHighPadding();

  void BackwardFilterLowPaddingGreaterThanHighPadding();
  void BackwardFilterEvenPadding();

  //////////////////////////////////////

  void run();

};

void ConvolutionVariantsTest::Minimal() 
{
  ComputationBuilder builder(TestName());

  const Array4D<float> input_array(1, 1, 1, 1, {2});
  auto input = builder.ConstantR4FromArray4D<float>(input_array);

  const Array4D<float> filter_array(1, 1, 1, 1, {3});
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);

  builder.Conv(input, filter, {1, 1}, Padding::kValid);

  const Array4D<float> expected(1, 1, 1, 1, {6});


  auto conv4d = ReferenceUtil::ConvArray4D(input_array, filter_array, { 1, 1 }, Padding::kValid);
  ASSERT_TRUE(*conv4d == expected);

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::MinimalWithBatch() 
{
  ComputationBuilder builder(TestName());

  const Array4D<float> input_array(5, 1, 1, 1, {1, 2, 3, 4, 5});
  auto input = builder.ConstantR4FromArray4D<float>(input_array);

  const Array4D<float> filter_array(1, 1, 1, 1, {2});
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);

  builder.Conv(input, filter, {1, 1}, Padding::kValid);

  const Array4D<float> expected(5, 1, 1, 1, {2, 4, 6, 8, 10});


  auto conv4d = ReferenceUtil::ConvArray4D(input_array, filter_array, { 1, 1 }, Padding::kValid);
  ASSERT_TRUE(*conv4d == expected);

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::Flat1x1() 
{
  ComputationBuilder builder(TestName());

  Array4D<float> input_array(2, 1, 3, 4);
  input_array.FillWithMultiples(1);
  auto input = builder.ConstantR4FromArray4D<float>(input_array);

  const Array4D<float> filter_array(1, 1, 1, 1, {2.3f});
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);

  builder.Conv(input, filter, {1, 1}, Padding::kValid);

  Array4D<float> expected(2, 1, 3, 4);

  expected.FillWithMultiples(2.3f);

  auto conv4d = ReferenceUtil::ConvArray4D(input_array, filter_array, { 1, 1 }, Padding::kValid);
  ASSERT_TRUE(*conv4d == expected);

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::Deep1x1() 
{
  ComputationBuilder builder(TestName());

  Array4D<float> input_array(1, 2, 1, 1, {10, 1});
  auto input = builder.ConstantR4FromArray4D<float>(input_array);

  const Array4D<float> filter_array(3, 2, 1, 1, {1, 2, 3, 4, 5, 6});
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);

  builder.Conv(input, filter, {1, 1}, Padding::kValid);

  Array4D<float> expected(1, 3, 1, 1, {12, 34, 56});


  auto conv4d = ReferenceUtil::ConvArray4D(input_array, filter_array, { 1, 1 }, Padding::kValid);
  ASSERT_TRUE(*conv4d == expected);

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::Filter1x2in1x2() 
{
  ComputationBuilder builder(TestName());

  Array4D<float> input_array(1, 1, 1, 2, {1, 2});
  auto input = builder.ConstantR4FromArray4D<float>(input_array);

  const Array4D<float> filter_array(1, 1, 1, 2, {10, 1});
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);

  builder.Conv(input, filter, {1, 1}, Padding::kValid);

  Array4D<float> expected(1, 1, 1, 1, {12});


  auto conv4d = ReferenceUtil::ConvArray4D(input_array, filter_array, { 1, 1 }, Padding::kValid);
  ASSERT_TRUE(*conv4d == expected);

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::Filter1x2in1x3() 
{
  ComputationBuilder builder(TestName());

  Array4D<float> input_array(1, 1, 1, 3, {1, 2, 3});
  auto input = builder.ConstantR4FromArray4D<float>(input_array);

  const Array4D<float> filter_array(1, 1, 1, 2, {10, 1});
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);

  builder.Conv(input, filter, {1, 1}, Padding::kValid);

  Array4D<float> expected(1, 1, 1, 2, {12, 23});


  auto conv4d = ReferenceUtil::ConvArray4D(input_array, filter_array, { 1, 1 }, Padding::kValid);
  ASSERT_TRUE(*conv4d == expected);

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::Filter1x2in2x2() 
{
  ComputationBuilder builder(TestName());

  Array4D<float> input_array(1, 1, 2, 2, {1, 2, 3, 4});
  auto input = builder.ConstantR4FromArray4D<float>(input_array);

  const Array4D<float> filter_array(1, 1, 1, 2, {10, 1});
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);

  builder.Conv(input, filter, {1, 1}, Padding::kValid);

  Array4D<float> expected(1, 1, 2, 1, {12, 34});


  auto conv4d = ReferenceUtil::ConvArray4D(input_array, filter_array, { 1, 1 }, Padding::kValid);
  ASSERT_TRUE(*conv4d == expected);

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::Filter2x1in2x2() 
{
  ComputationBuilder builder(TestName());

  Array4D<float> input_array(1, 1, 2, 2, {1, 2, 3, 4});
  auto input = builder.ConstantR4FromArray4D<float>(input_array);

  const Array4D<float> filter_array(1, 1, 2, 1, {10, 1});
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);

  builder.Conv(input, filter, {1, 1}, Padding::kValid);

  Array4D<float> expected(1, 1, 1, 2, {13, 24});


  auto conv4d = ReferenceUtil::ConvArray4D(input_array, filter_array, { 1, 1 }, Padding::kValid);
  ASSERT_TRUE(*conv4d == expected);

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::Filter2x2in2x2() 
{
  ComputationBuilder builder(TestName());

  Array4D<float> input_array(1, 1, 2, 2, {1, 2, 3, 4});
  auto input = builder.ConstantR4FromArray4D<float>(input_array);

  const Array4D<float> filter_array(1, 1, 2, 2, {1000, 100, 10, 1});
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);

  builder.Conv(input, filter, {1, 1}, Padding::kValid);

  Array4D<float> expected(1, 1, 1, 1, {1234});


  auto conv4d = ReferenceUtil::ConvArray4D(input_array, filter_array, { 1, 1 }, Padding::kValid);
  ASSERT_TRUE(*conv4d == expected);

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::Filter1x2in2x3WithDepthAndBatch() 
{
  ComputationBuilder builder(TestName());

  Array4D<float> input_array(
      2, 2, 2, 3, {0, 1, 2, 3, 4, 5,  6,  7,  8,  9,  0, 0,    // plane 0
                   0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 0, 0});  // plane 1
  auto input = builder.ConstantR4FromArray4D<float>(input_array);

  const Array4D<float> filter_array(
      2, 2, 1, 2, {1000, 100, 10, 1, 0.1f, 0.01f, 0.001f, 0.0001f});
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);

  builder.Conv(input, filter, {1, 1}, Padding::kValid);

  Array4D<float> expected(
      2, 2, 2, 2,
      {167, 1278, 3490, 4500, 0.0167f, 0.1278f, 0.3490f, 0.4500f,    // plane 0
       334, 2556, 6980, 9000, 0.0334f, 0.2556f, 0.6980f, 0.9000f});  // plane 1


  auto conv4d = ReferenceUtil::ConvArray4D(input_array, filter_array, { 1, 1 }, Padding::kValid);
  ASSERT_TRUE(*conv4d == expected);

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::Filter1x1stride1x2in1x4() 
{
  ComputationBuilder builder(TestName());

  Array4D<float> input_array(1, 1, 1, 4, {1, 2, 3, 4});
  auto input = builder.ConstantR4FromArray4D<float>(input_array);

  const Array4D<float> filter_array(1, 1, 1, 1, {10});
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);

  builder.Conv(input, filter, {1, 2}, Padding::kValid);

  Array4D<float> expected(1, 1, 1, 2, {10, 30});

  // TODO:
  auto conv4d = ReferenceUtil::ConvArray4D(input_array, filter_array, { 1, 2 }, Padding::kValid);
  ASSERT_TRUE(*conv4d == expected);

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::Filter1x1stride1x2in1x5() 
{
  ComputationBuilder builder(TestName());

  Array4D<float> input_array(1, 1, 1, 5, {1, 2, 3, 4, 5});
  auto input = builder.ConstantR4FromArray4D<float>(input_array);

  const Array4D<float> filter_array(1, 1, 1, 1, {10});
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);

  builder.Conv(input, filter, {1, 2}, Padding::kValid);

  Array4D<float> expected(1, 1, 1, 3, {10, 30, 50});

  // TODO:
  auto conv4d = ReferenceUtil::ConvArray4D(input_array, filter_array, { 1, 2 }, Padding::kValid);
  ASSERT_TRUE(*conv4d == expected);

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::Filter1x3stride1x2in1x4() 
{
  ComputationBuilder builder(TestName());

  Array4D<float> input_array(1, 1, 1, 4, {1, 2, 3, 4});
  auto input = builder.ConstantR4FromArray4D<float>(input_array);

  const Array4D<float> filter_array(1, 1, 1, 3, {100, 10, 1});
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);

  builder.Conv(input, filter, {1, 2}, Padding::kValid);

  Array4D<float> expected(1, 1, 1, 1, {123});

  // TODO:
  auto conv4d = ReferenceUtil::ConvArray4D(input_array, filter_array, { 1, 2 }, Padding::kValid);
  ASSERT_TRUE(*conv4d == expected);

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::Filter1x3stride1x2in1x5() 
{
  ComputationBuilder builder(TestName());

  Array4D<float> input_array(1, 1, 1, 5, {1, 2, 3, 4, 5});
  auto input = builder.ConstantR4FromArray4D<float>(input_array);

  const Array4D<float> filter_array(1, 1, 1, 3, {100, 10, 1});
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);

  builder.Conv(input, filter, {1, 2}, Padding::kValid);

  Array4D<float> expected(1, 1, 1, 2, {123, 345});

  // TODO:
  auto conv4d = ReferenceUtil::ConvArray4D(input_array, filter_array, { 1, 2 }, Padding::kValid);
  ASSERT_TRUE(*conv4d == expected);

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::Filter1x1stride2x2in3x3() 
{
  ComputationBuilder builder(TestName());

  Array4D<float> input_array(1, 1, 3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto input = builder.ConstantR4FromArray4D<float>(input_array);

  const Array4D<float> filter_array(1, 1, 1, 1, {10});
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);

  builder.Conv(input, filter, {2, 2}, Padding::kValid);

  Array4D<float> expected(1, 1, 2, 2, {10, 30, 70, 90});

  // TODO:
  auto conv4d = ReferenceUtil::ConvArray4D(input_array, filter_array, { 2, 2 }, Padding::kValid);
  ASSERT_TRUE(*conv4d == expected);

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::Filter3x1in1x1Padded() 
{
  ComputationBuilder builder(TestName());

  Array4D<float> input_array(1, 1, 1, 1, {1});
  auto input = builder.ConstantR4FromArray4D<float>(input_array);

  const Array4D<float> filter_array(1, 1, 1, 3, {10, 20, 30});
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);

  builder.Conv(input, filter, {1, 1}, Padding::kSame);

  Array4D<float> expected(1, 1, 1, 1, {20});

  // TODO:
  auto conv4d = ReferenceUtil::ConvArray4D(input_array, filter_array, { 1, 1 }, Padding::kSame);
  ASSERT_TRUE(*conv4d == expected);

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::Filter5x1in3x1Padded() 
{
  ComputationBuilder builder(TestName());

  Array4D<float> input_array(1, 1, 1, 3, {1, 2, 3});
  auto input = builder.ConstantR4FromArray4D<float>(input_array);

  const Array4D<float> filter_array(1, 1, 1, 5, {10000, 1000, 100, 10, 1});
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);

  builder.Conv(input, filter, {1, 1}, Padding::kSame);

  Array4D<float> expected(1, 1, 1, 3, {123, 1230, 12300});

  // TODO:
  auto conv4d = ReferenceUtil::ConvArray4D(input_array, filter_array, { 1, 1 }, Padding::kSame);
  ASSERT_TRUE(*conv4d == expected);

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::Filter3x3in2x2Padded() 
{
  ComputationBuilder builder(TestName());

  Array4D<float> input_array(1, 1, 2, 2, {1, 2, 3, 4});
  auto input = builder.ConstantR4FromArray4D<float>(input_array);

  const Array4D<float> filter_array(1, 1, 3, 3, {10000, 0, 1000,  // row 0
                                                 0, 100, 0,       // row 1
                                                 10, 0, 1});      // row 2
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);

  builder.Conv(input, filter, {1, 1}, Padding::kSame);

  Array4D<float> expected(1, 1, 2, 2, {104, 230, 2300, 10400});

  // TODO:
  auto conv4d = ReferenceUtil::ConvArray4D(input_array, filter_array, { 1, 1 }, Padding::kSame);
  ASSERT_TRUE(*conv4d == expected);

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::Filter1x1in2x1WithPaddingAndDepth() 
{
  ComputationBuilder builder(TestName());

  Array4D<float> input_array(1, 2, 1, 2, {1, 2, 3, 4});
  auto input = builder.ConstantR4FromArray4D<float>(input_array);

  const Array4D<float> filter_array(1, 2, 1, 1, {10, 1});
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);

  builder.Conv(input, filter, {1, 1}, Padding::kSame);

  Array4D<float> expected(1, 1, 1, 2, {13, 24});

  // TODO:
  auto conv4d = ReferenceUtil::ConvArray4D(input_array, filter_array, { 1, 1 }, Padding::kSame);
  ASSERT_TRUE(*conv4d == expected);

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::Filter2x2Stride1x1Input3x3() 
{
  ComputationBuilder builder(TestName());

  Array4D<float> input_array(1, 1, 3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto input = builder.ConstantR4FromArray4D<float>(input_array);

  const Array4D<float> filter_array(1, 1, 2, 2, {7, 13, 17, 23});
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);

  builder.Conv(input, filter, {1, 1}, Padding::kValid);

  Array4D<float> expected(1, 1, 2, 2, {216, 276, 396, 456});

  // TODO:
  auto conv4d = ReferenceUtil::ConvArray4D(input_array, filter_array, { 1, 1 }, Padding::kValid);
  ASSERT_TRUE(*conv4d == expected);

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::Filter1x2Stride1x1Input1x3() 
{
  ComputationBuilder builder(TestName());

  Array4D<float> input_array(1, 1, 1, 3, {1, 2, 3});
  auto input = builder.ConstantR4FromArray4D<float>(input_array);

  const Array4D<float> filter_array(1, 1, 1, 2, {7, 13});
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);

  builder.Conv(input, filter, {1, 1}, Padding::kValid);

  Array4D<float> expected(1, 1, 1, 2, {33, 53});

  // TODO:
  auto conv4d = ReferenceUtil::ConvArray4D(input_array, filter_array, { 1, 1 }, Padding::kValid);
  ASSERT_TRUE(*conv4d == expected);

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::Filter2x1x8x8Input1x1x8x8() 
{
  ComputationBuilder builder(TestName());

  std::vector<float> input_data(64);
  std::iota(input_data.begin(), input_data.end(), 0.0f);
  Array4D<float> input_array(1, 1, 8, 8, input_data);
  auto input = builder.ConstantR4FromArray4D<float>(input_array);

  std::vector<float> filter_data(128);
  std::fill(filter_data.begin(), filter_data.begin() + 64, 1.0f);
  std::fill(filter_data.begin() + 64, filter_data.begin() + 128, 2.0f);
  const Array4D<float> filter_array(2, 1, 8, 8, filter_data);
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);

  builder.Conv(input, filter, {1, 1}, Padding::kValid);

  Array4D<float> expected(1, 2, 1, 1, {2016, 4032});

  // TODO:
  auto conv4d = ReferenceUtil::ConvArray4D(input_array, filter_array, { 1, 1 }, Padding::kValid);
  ASSERT_TRUE(*conv4d == expected);

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::Filter1x1x1x1Input16x1x1x1() 
{
  ComputationBuilder builder(TestName());

  std::vector<float> input_data(16 * 1 * 1 * 1);
  std::iota(input_data.begin(), input_data.end(), 1.f);
  Array4D<float> input_array(16, 1, 1, 1, input_data);
  auto input = builder.ConstantR4FromArray4D<float>(input_array);

  std::vector<float> filter_data(1 * 1 * 1 * 1);
  std::iota(filter_data.begin(), filter_data.end(), 1.f);
  const Array4D<float> filter_array(1, 1, 1, 1, filter_data);
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);

  builder.Conv(input, filter, {1, 1}, Padding::kValid);

  std::vector<float> expected_data = {1, 2,  3,  4,  5,  6,  7,  8,
                                      9, 10, 11, 12, 13, 14, 15, 16};
  Array4D<float> expected(16, 1, 1, 1, expected_data);

  // TODO:
  auto conv4d = ReferenceUtil::ConvArray4D(input_array, filter_array, { 1, 1 }, Padding::kValid);
  ASSERT_TRUE(*conv4d == expected);

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::Filter1x1x2x2Input16x1x2x2() 
{
  ComputationBuilder builder(TestName());

  constexpr int bs = 16;
  constexpr int kx = 2;
  constexpr int ky = 2;
  Array4D<float> input_array(bs, 1, ky, kx);
  for (int i0 = 0; i0 < bs; ++i0) {
    for (int i2 = 0; i2 < ky; ++i2) {
      for (int i3 = 0; i3 < kx; ++i3) {
        input_array(i0, 0, i2, i3) = i0 + 1.f;
      }
    }
  }
  auto input = builder.ConstantR4FromArray4D<float>(input_array);

  std::vector<float> filter_data(1 * 1 * ky * kx);
  std::iota(filter_data.begin(), filter_data.end(), 1.f);
  const Array4D<float> filter_array(1, 1, ky, kx, filter_data);
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);

  builder.Conv(input, filter, {1, 1}, Padding::kValid);


  std::vector<float> expected_data(bs);
  for (int i = 0; i < bs; ++i) 
  {
    expected_data[i] = static_cast<float>(10 * (i + 1));
  }
  Array4D<float> expected(bs, 1, 1, 1, expected_data);

  //TODO:
  auto conv4d = ReferenceUtil::ConvArray4D(input_array, filter_array, { 1, 1 }, Padding::kValid);
  ASSERT_TRUE(*conv4d == expected);

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::Filter1x1x2x2Input3x1x2x2() 
{
  ComputationBuilder builder(TestName());

  constexpr int kx = 2;
  constexpr int ky = 2;
  constexpr int bs = 3;
  Array4D<float> input_array(bs, 1, ky, kx);
  for (int i0 = 0; i0 < bs; ++i0) {
    for (int i2 = 0; i2 < ky; ++i2) {
      for (int i3 = 0; i3 < kx; ++i3) {
        input_array(i0, 0, i2, i3) = static_cast<float>(i0 + i2 + i3 + 1);
      }
    }
  }
  auto input = builder.ConstantR4FromArray4D<float>(input_array);

  std::vector<float> filter_data(1 * 1 * ky * kx);
  std::iota(filter_data.begin(), filter_data.end(), 1.f);
  const Array4D<float> filter_array(1, 1, ky, kx, filter_data);
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);

  builder.Conv(input, filter, {1, 1}, Padding::kValid);

  std::vector<float> expected_data = {
      23, 33, 43,
  };
  Array4D<float> expected(bs, 1, 1, 1, expected_data);

  auto conv4d = ReferenceUtil::ConvArray4D(input_array, filter_array, { 1, 1 }, Padding::kValid);
  ASSERT_TRUE(*conv4d == expected);

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::Filter1x1x8x8Input16x1x8x8() 
{
  ComputationBuilder builder(TestName());

  Array4D<float> input_array(16, 1, 8, 8);
  for (int i0 = 0; i0 < 16; ++i0) {
    for (int i2 = 0; i2 < 8; ++i2) {
      for (int i3 = 0; i3 < 8; ++i3) {
        input_array(i0, 0, i2, i3) = static_cast<float>(i0 + i2 + i3 + 1);
      }
    }
  }
  auto input = builder.ConstantR4FromArray4D<float>(input_array);

  std::vector<float> filter_data(1 * 1 * 8 * 8);
  std::iota(filter_data.begin(), filter_data.end(), 1.f);
  const Array4D<float> filter_array(1, 1, 8, 8, filter_data);
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);

  builder.Conv(input, filter, {1, 1}, Padding::kValid);

  std::vector<float> expected_data = {
      19664, 21744, 23824, 25904, 27984, 30064, 32144, 34224,
      36304, 38384, 40464, 42544, 44624, 46704, 48784, 50864,
  };
  Array4D<float> expected(16, 1, 1, 1, expected_data);

  auto conv4d = ReferenceUtil::ConvArray4D(input_array, filter_array, { 1, 1 }, Padding::kValid);
  ASSERT_TRUE(*conv4d == expected);

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::Filter2x2x8x8Input1x2x8x8() 
{
  ComputationBuilder builder(TestName());

  std::vector<float> input_data(2 * 8 * 8);
  std::iota(input_data.begin(), input_data.end(), 0.f);
  Array4D<float> input_array(1, 2, 8, 8, input_data);
  auto input = builder.ConstantR4FromArray4D<float>(input_array);

  std::vector<float> filter_data(2 * 2 * 8 * 8);
  std::fill(filter_data.begin(), filter_data.begin() + filter_data.size() / 4,
            1.f);
  std::fill(filter_data.begin() + filter_data.size() / 4,
            filter_data.begin() + filter_data.size() / 2, 2.f);
  std::fill(filter_data.begin() + filter_data.size() / 2,
            filter_data.begin() + 3 * filter_data.size() / 4, 3.f);
  std::fill(filter_data.begin() + 3 * filter_data.size() / 4, filter_data.end(),
            4.f);
  const Array4D<float> filter_array(2, 2, 8, 8, filter_data);
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);

  builder.Conv(input, filter, {1, 1}, Padding::kValid);

  Array4D<float> expected(1, 2, 1, 1, {14240, 30496});

  auto conv4d = ReferenceUtil::ConvArray4D(input_array, filter_array, { 1, 1 }, Padding::kValid);
  ASSERT_TRUE(*conv4d == expected);

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::Filter2x2x8x8Input2x2x8x8() 
{
  ComputationBuilder builder(TestName());

  std::vector<float> input_data(2 * 2 * 8 * 8);
  std::iota(input_data.begin(), input_data.end(), 0.f);
  Array4D<float> input_array(2, 2, 8, 8, input_data);
  auto input = builder.ConstantR4FromArray4D<float>(input_array);

  std::vector<float> filter_data(2 * 2 * 8 * 8);
  std::fill(filter_data.begin(), filter_data.begin() + filter_data.size() / 4,
            1.f);
  std::fill(filter_data.begin() + filter_data.size() / 4,
            filter_data.begin() + filter_data.size() / 2, 2.f);
  std::fill(filter_data.begin() + filter_data.size() / 2,
            filter_data.begin() + 3 * filter_data.size() / 4, 3.f);
  std::fill(filter_data.begin() + 3 * filter_data.size() / 4, filter_data.end(),
            4.f);
  const Array4D<float> filter_array(2, 2, 8, 8, filter_data);
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);

  builder.Conv(input, filter, {1, 1}, Padding::kValid);

  Array4D<float> expected(2, 2, 1, 1, {14240, 30496, 38816, 87840});


  auto conv4d = ReferenceUtil::ConvArray4D(input_array, filter_array, { 1, 1 }, Padding::kValid);
  ASSERT_TRUE(*conv4d == expected);

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::Filter2x2x8x8Input32x2x8x8() 
{
  ComputationBuilder builder(TestName());

  std::vector<float> input_data(32 * 2 * 8 * 8);
  std::iota(input_data.begin(), input_data.end(), 0.f);
  Array4D<float> input_array(32, 2, 8, 8, input_data);
  auto input = builder.ConstantR4FromArray4D<float>(input_array);

  std::vector<float> filter_data(2 * 2 * 8 * 8);
  std::fill(filter_data.begin(), filter_data.begin() + filter_data.size() / 4,
            1.f);
  std::fill(filter_data.begin() + filter_data.size() / 4,
            filter_data.begin() + filter_data.size() / 2, 2.f);
  std::fill(filter_data.begin() + filter_data.size() / 2,
            filter_data.begin() + 3 * filter_data.size() / 4, 3.f);
  std::fill(filter_data.begin() + 3 * filter_data.size() / 4, filter_data.end(),
            4.f);
  const Array4D<float> filter_array(2, 2, 8, 8, filter_data);
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);

  builder.Conv(input, filter, {1, 1}, Padding::kValid);

  std::vector<float> expected_data = {
      14240,       30496,       38816,   87840,   63392,       145184,  87968,
      202528,      112544,      259872,  137120,  317216,      161696,  374560,
      186272,      431904,      210848,  489248,  235424,      546592,  260000,
      603936,      284576,      661280,  309152,  718624,      333728,  775968,
      358304,      833312,      382880,  890656,  407456,      948000,  432032,
      1005344,     456608,      1062688, 481184,  1120032,     505760,  1177376,
      530336,      1.23472e+06, 554912,  1292064, 579488,      1349408, 604064,
      1406752,     628640,      1464096, 653216,  1.52144e+06, 677792,  1578784,
      702368,      1636128,     726944,  1693472, 751520,      1750816, 776096,
      1.80816e+06,
  };
  Array4D<float> expected(32, 2, 1, 1, expected_data);
  // The output elements can be larger than 1e+5, making the absolute error
  // large sometimes. So, we focus on relative errors for this test case.


  auto conv4d = ReferenceUtil::ConvArray4D(input_array, filter_array, { 1, 1 }, Padding::kValid);
  ASSERT_TRUE(*conv4d == expected);

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::Filter16x16x1x1Input16x16x1x1() 
{
  ComputationBuilder builder(TestName());

  Array4D<float> input_array(16, 16, 1, 1);
  Array4D<float> filter_array(16, 16, 1, 1);
  for (int i0 = 0; i0 < 16; ++i0) {
    for (int i1 = 0; i1 < 16; ++i1) {
      input_array(i0, i1, 0, 0) = static_cast<float>(1000 * i0 + i1);
      filter_array(i0, i1, 0, 0) = 1.f;
    }
  }

  auto input = builder.ConstantR4FromArray4D<float>(input_array);
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);
  builder.Conv(input, filter, {1, 1}, Padding::kValid);

  Array4D<float> expected(16, 16, 1, 1);
  for (int i0 = 0; i0 < 16; ++i0) {
    for (int i1 = 0; i1 < 16; ++i1) {
      expected(i0, i1, 0, 0) = static_cast<float>(16000 * i0 + 120);
    }
  }


  auto conv4d = ReferenceUtil::ConvArray4D(input_array, filter_array, { 1, 1 }, Padding::kValid);
  ASSERT_TRUE(*conv4d == expected);

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::FlatRhsDilation() 
{
  ComputationBuilder builder(TestName());

  std::vector<float> input_data(1 * 1 * 4 * 6);
  std::iota(input_data.begin(), input_data.end(), 0.f);
  Array4D<float> input_array(1, 1, 4, 6, input_data);

  Array4D<float> filter_array(1, 1, 2, 3, {1, 10, 100, 2, 20, 200});
  auto input = builder.ConstantR4FromArray4D<float>(input_array);
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);
  builder.ConvGeneralDilated(
      /*lhs=*/input, /*rhs=*/filter, /*window_strides=*/{}, /*padding=*/{},
      /*lhs_dilation=*/{}, /*rhs_dilation=*/{2, 2},
      ComputationBuilder::CreateDefaultConvDimensionNumbers());

  Array4D<float> expected(1, 1, 2, 2, {3924, 4257, 5922, 6255});

  auto conv4d = ReferenceUtil::ConvArray4DGeneralDimensionsDilated(
     input_array, 
     filter_array, 
     { 1, 1 }, 
     Padding::kValid,
     {1, 1},
     {2, 2},
     ComputationBuilder::CreateDefaultConvDimensionNumbers()
  );
  ASSERT_TRUE(*conv4d == expected);

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::FlatLhsDilation1D() 
{
  ComputationBuilder builder(TestName());

  std::vector<float> input_data(1 * 1 * 1 * 5);
  std::iota(input_data.begin(), input_data.end(), 1.f);
  Array4D<float> input_array(1, 1, 1, 5, input_data);

  Array4D<float> filter_array(1, 1, 1, 2, {10, 1});
  auto input = builder.ConstantR4FromArray4D<float>(input_array);
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);
  builder.ConvGeneralDilated(
      /*lhs=*/input, /*rhs=*/filter, /*window_strides=*/{}, /*padding=*/{},
      /*lhs_dilation=*/{1, 2}, /*rhs_dilation=*/{},
      ComputationBuilder::CreateDefaultConvDimensionNumbers());

  Array4D<float> expected(1, 1, 1, 8, {10, 2, 20, 3, 30, 4, 40, 5});

  auto conv4d = ReferenceUtil::ConvArray4DGeneralDimensionsDilated(
     input_array,
     filter_array,
     { 1, 1 },
     Padding::kValid,
     { 1, 2 },
     { 1, 1 },
     ComputationBuilder::CreateDefaultConvDimensionNumbers()
  );
  ASSERT_TRUE(*conv4d == expected);

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::FlatLhsDilation() 
{
  ComputationBuilder builder(TestName());

  std::vector<float> input_data(1 * 1 * 3 * 4);
  std::iota(input_data.begin(), input_data.end(), 1.f);
  Array4D<float> input_array(1, 1, 3, 4, input_data);

  Array4D<float> filter_array(1, 1, 4, 3, {100, 10, 1,  //
                                           200, 20, 2,  //
                                           300, 30, 3,  //
                                           400, 40, 4});
  auto input = builder.ConstantR4FromArray4D<float>(input_array);
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);
  builder.ConvGeneralDilated(
      /*lhs=*/input, /*rhs=*/filter, /*window_strides=*/{2, 1},
      /*padding=*/{{1, 0}, {0, 0}}, /*lhs_dilation=*/{3, 2},
      /*rhs_dilation=*/{},
      ComputationBuilder::CreateDefaultConvDimensionNumbers());

  Array4D<float> expected(1, 1, 3, 5, {204, 40, 406, 60, 608,       //
                                       1518, 180, 1821, 210, 2124,  //
                                       4146, 460, 4651, 510, 5156});

  //auto conv4d = ReferenceUtil::ConvArray4DGeneralDimensionsDilated(
  //   input_array,
  //   filter_array,
  //   { 2, 1 },
  //   Padding::kValid,
  //   { 3, 2 },
  //   { 1, 1 },
  //   ComputationBuilder::CreateDefaultConvDimensionNumbers()
  //);
  //ASSERT_TRUE(*conv4d == expected);


  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::NegativePaddingOnBothEnds() 
{
  ComputationBuilder builder(TestName());

  std::vector<float> input_data(1 * 1 * 1 * 5);
  std::iota(input_data.begin(), input_data.end(), 1.f);
  Array4D<float> input_array(1, 1, 1, 5, input_data);

  Array4D<float> filter_array(1, 1, 1, 2, {10, 1});
  auto input = builder.ConstantR4FromArray4D<float>(input_array);
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);
  builder.ConvGeneral(
      /*lhs=*/input, /*rhs=*/filter, /*window_strides=*/{},
      /*padding=*/{{0, 0}, {-1, -1}},
      ComputationBuilder::CreateDefaultConvDimensionNumbers());

  Array4D<float> expected(1, 1, 1, 2, {23, 34});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::NegativePaddingLowAndPositivePaddingHigh() 
{
  ComputationBuilder builder(TestName());

  std::vector<float> input_data(1 * 1 * 1 * 5);
  std::iota(input_data.begin(), input_data.end(), 1.f);
  Array4D<float> input_array(1, 1, 1, 5, input_data);

  Array4D<float> filter_array(1, 1, 1, 2, {10, 1});
  auto input = builder.ConstantR4FromArray4D<float>(input_array);
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);
  builder.ConvGeneral(
      /*lhs=*/input, /*rhs=*/filter, /*window_strides=*/{},
      /*padding=*/{{0, 0}, {-1, 2}},
      ComputationBuilder::CreateDefaultConvDimensionNumbers());

  Array4D<float> expected(1, 1, 1, 5, {23, 34, 45, 50, 0});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::PositivePaddingLowAndNegativePaddingHigh() 
{
  ComputationBuilder builder(TestName());

  std::vector<float> input_data(1 * 1 * 1 * 5);
  std::iota(input_data.begin(), input_data.end(), 1.f);
  Array4D<float> input_array(1, 1, 1, 5, input_data);

  Array4D<float> filter_array(1, 1, 1, 2, {10, 1});
  auto input = builder.ConstantR4FromArray4D<float>(input_array);
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);
  builder.ConvGeneral(
      /*lhs=*/input, /*rhs=*/filter, /*window_strides=*/{},
      /*padding=*/{{0, 0}, {2, -1}},
      ComputationBuilder::CreateDefaultConvDimensionNumbers());

  Array4D<float> expected(1, 1, 1, 5, {0, 1, 12, 23, 34});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::PositivePaddingAndDilation() 
{
  ComputationBuilder builder(TestName());

  std::vector<float> input_data(1 * 1 * 1 * 5);
  std::iota(input_data.begin(), input_data.end(), 1.f);
  Array4D<float> input_array(1, 1, 1, 5, input_data);

  Array4D<float> filter_array(1, 1, 1, 2, {10, 1});
  auto input = builder.ConstantR4FromArray4D<float>(input_array);
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);
  builder.ConvGeneralDilated(
      /*lhs=*/input, /*rhs=*/filter, /*window_strides=*/{},
      /*padding=*/{{0, 0}, {3, 2}},
      /*lhs_dilation=*/{1, 2}, /*rhs_dilation=*/{1, 2},
      ComputationBuilder::CreateDefaultConvDimensionNumbers());

  // input:
  //   [1, 2, 3, 4, 5] --dilate-> [1, 0, 2, 0, 3, 0, 4, 0, 5]
  //                   ---pad---> [0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 0]
  // filter:
  //   [10, 1] --dilate-> [10, 0, 1]
  Array4D<float> expected(1, 1, 1, 12,
                          {0, 1, 0, 12, 0, 23, 0, 34, 0, 45, 0, 50});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::NegativePaddingAndDilation() 
{
  ComputationBuilder builder(TestName());

  std::vector<float> input_data(1 * 1 * 1 * 5);
  std::iota(input_data.begin(), input_data.end(), 1.f);
  Array4D<float> input_array(1, 1, 1, 5, input_data);

  Array4D<float> filter_array(1, 1, 1, 2, {10, 1});
  auto input = builder.ConstantR4FromArray4D<float>(input_array);
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);
  builder.ConvGeneralDilated(
      /*lhs=*/input, /*rhs=*/filter, /*window_strides=*/{},
      /*padding=*/{{0, 0}, {-3, -2}},
      /*lhs_dilation=*/{1, 2}, /*rhs_dilation=*/{1, 2},
      ComputationBuilder::CreateDefaultConvDimensionNumbers());

  // input:
  //   [1, 2, 3, 4, 5] --dilate-> [1, 0, 2, 0, 3, 0, 4, 0, 5]
  //                   ---pad---> [0, 3, 0, 4]
  // filter:
  //   [10, 1] --dilate-> [10, 0, 1]
  Array4D<float> expected(1, 1, 1, 2, {0, 34});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::RandomData_Input1x1x2x3_Filter2x1x1x2() 
{
  constexpr int bs = 1;
  constexpr int iz = 1;
  constexpr int oz = 2;
  constexpr int iy = 2;
  constexpr int ix = 3;
  constexpr int ky = 1;
  constexpr int kx = 2;
  std::mt19937 rng;
  std::uniform_real_distribution<float> distribution;
  std::vector<float> input_data(bs * iz * iy * ix);
  for (float& f : input_data) {
    f = distribution(rng);
  }
  std::vector<float> kernel_data(oz * iz * ky * kx);
  for (float& f : kernel_data) {
    f = distribution(rng);
  }

  Array4D<float> input_array(bs, iz, iy, ix, input_data);
  Array4D<float> filter_array(oz, iz, ky, kx, kernel_data);

  ComputationBuilder builder(TestName());
  auto input = builder.ConstantR4FromArray4D<float>(input_array);
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);
  builder.Conv(input, filter, {1, 1}, Padding::kValid);

  std::unique_ptr<Array4D<float>> expected = ReferenceUtil::ConvArray4D(
      input_array, filter_array, {1, 1}, Padding::kValid);

  ComputeAndCompareR4<float>(&builder, *expected, {}, error_spec_);
}

void ConvolutionVariantsTest::RandomData_Input1x16x1x1_Filter1x16x1x1() 
{
  constexpr int bs = 1;
  constexpr int iz = 16;
  constexpr int oz = 1;
  constexpr int iy = 1;
  constexpr int ix = 1;
  constexpr int ky = 1;
  constexpr int kx = 1;
  std::mt19937 rng;
  std::uniform_real_distribution<float> distribution;
  std::vector<float> input_data(bs * iz * iy * ix);
  for (float& f : input_data) {
    f = distribution(rng);
  }
  std::vector<float> kernel_data(oz * iz * ky * kx);
  for (float& f : kernel_data) {
    f = distribution(rng);
  }

  Array4D<float> input_array(bs, iz, iy, ix, input_data);
  Array4D<float> filter_array(oz, iz, ky, kx, kernel_data);

  ComputationBuilder builder(TestName());
  auto input = builder.ConstantR4FromArray4D<float>(input_array);
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);
  builder.Conv(input, filter, {1, 1}, Padding::kValid);

  std::unique_ptr<Array4D<float>> expected = ReferenceUtil::ConvArray4D(
      input_array, filter_array, {1, 1}, Padding::kValid);

  ComputeAndCompareR4<float>(&builder, *expected, {}, error_spec_);
}

void ConvolutionVariantsTest::RandomData_Input16x16x1x1_Filter1x16x1x1() 
{
  constexpr int bs = 16;
  constexpr int iz = 16;
  constexpr int oz = 1;
  constexpr int iy = 1;
  constexpr int ix = 1;
  constexpr int ky = 1;
  constexpr int kx = 1;
  std::mt19937 rng;
  std::uniform_real_distribution<float> distribution;
  std::vector<float> input_data(bs * iz * iy * ix);
  for (float& f : input_data) {
    f = distribution(rng);
  }
  std::vector<float> kernel_data(oz * iz * ky * kx);
  for (float& f : kernel_data) {
    f = distribution(rng);
  }

  Array4D<float> input_array(bs, iz, iy, ix, input_data);
  Array4D<float> filter_array(oz, iz, ky, kx, kernel_data);

  ComputationBuilder builder(TestName());
  auto input = builder.ConstantR4FromArray4D<float>(input_array);
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);
  builder.Conv(input, filter, {1, 1}, Padding::kValid);

  std::unique_ptr<Array4D<float>> expected = ReferenceUtil::ConvArray4D(
      input_array, filter_array, {1, 1}, Padding::kValid);

  ComputeAndCompareR4<float>(&builder, *expected, {}, error_spec_);
}

void ConvolutionVariantsTest::RandomData_Input16x16x1x1_Filter16x16x1x1() 
{
  constexpr int bs = 16;
  constexpr int iz = 16;
  constexpr int oz = 16;
  constexpr int iy = 1;
  constexpr int ix = 1;
  constexpr int ky = 1;
  constexpr int kx = 1;
  std::mt19937 rng;
  std::uniform_real_distribution<float> distribution;
  std::vector<float> input_data(bs * iz * iy * ix);
  for (float& f : input_data) {
    f = distribution(rng);
  }
  std::vector<float> kernel_data(oz * iz * ky * kx);
  for (float& f : kernel_data) {
    f = distribution(rng);
  }

  Array4D<float> input_array(bs, iz, iy, ix, input_data);
  Array4D<float> filter_array(oz, iz, ky, kx, kernel_data);

  ComputationBuilder builder(TestName());
  auto input = builder.ConstantR4FromArray4D<float>(input_array);
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);
  builder.Conv(input, filter, {1, 1}, Padding::kValid);

  std::unique_ptr<Array4D<float>> expected = ReferenceUtil::ConvArray4D(
      input_array, filter_array, {1, 1}, Padding::kValid);

  ComputeAndCompareR4<float>(&builder, *expected, {}, error_spec_);
}

void ConvolutionVariantsTest::RandomData_Input16x16x16x16_Filter16x16x16x16() 
{
  constexpr int bs = 16;
  constexpr int iz = 16;
  constexpr int oz = 16;
  constexpr int iy = 16;
  constexpr int ix = 16;
  constexpr int ky = 16;
  constexpr int kx = 16;
  std::mt19937 rng;
  std::uniform_real_distribution<float> distribution;
  std::vector<float> input_data(bs * iz * iy * ix);
  for (float& f : input_data) {
    f = distribution(rng);
  }
  std::vector<float> kernel_data(oz * iz * ky * kx);
  for (float& f : kernel_data) {
    f = distribution(rng);
  }

  Array4D<float> input_array(bs, iz, iy, ix, input_data);
  Array4D<float> filter_array(oz, iz, ky, kx, kernel_data);

  ComputationBuilder builder(TestName());
  auto input = builder.ConstantR4FromArray4D<float>(input_array);
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);
  builder.Conv(input, filter, {1, 1}, Padding::kValid);

  std::unique_ptr<Array4D<float>> expected = ReferenceUtil::ConvArray4D(
      input_array, filter_array, {1, 1}, Padding::kValid);


  auto conv4d = ReferenceUtil::ConvArray4D(input_array, filter_array, { 1, 1 }, Padding::kValid);
  ASSERT_TRUE(*conv4d == *expected);


  ComputeAndCompareR4<float>(&builder, *expected, {}, error_spec_);
}

void ConvolutionVariantsTest::Filter1x2x1x1Input1x2x3x1GeneralPadding() 
{
  ComputationBuilder builder(TestName());

  std::vector<float> input_data(1 * 2 * 3 * 1);
  std::iota(input_data.begin(), input_data.end(), 1.f);
  Array4D<float> input_array(1, 2, 3, 1, input_data);
  auto input = builder.ConstantR4FromArray4D<float>(input_array);

  std::vector<float> filter_data(1 * 2 * 1 * 1);
  std::iota(filter_data.begin(), filter_data.end(), 1.f);
  Array4D<float> filter_array(1, 2, 1, 1, filter_data);
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);

  ConvolutionDimensionNumbers dnums;
  // NHWC input format.
  dnums.set_batch_dimension(0);
  dnums.add_spatial_dimensions(1);
  dnums.add_spatial_dimensions(2);
  dnums.set_feature_dimension(3);

  // Tensorflow filter shape: [ H, W, inC, outC ]
  dnums.add_kernel_spatial_dimensions(0);
  dnums.add_kernel_spatial_dimensions(1);
  dnums.set_kernel_input_feature_dimension(2);
  dnums.set_kernel_output_feature_dimension(3);

  // Tests padding sizes that don't correspond either to SAME or VALID padding.
  builder.ConvGeneral(input, filter, {1, 1}, {{2, 1}, {2, 3}}, dnums);

  std::vector<float> expected_data = {
      0, 0, 0,  0,  0, 0, 0,  //
      0, 0, 0,  0,  0, 0, 0,  //
      0, 2, 5,  8,  3, 0, 0,  //
      0, 8, 14, 17, 6, 0, 0,  //
      0, 0, 0,  0,  0, 0, 0   //
  };
  Array4D<float> expected(1, 5, 7, 1, expected_data);
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::Filter1x1x1x1Input1x2x3x1GeneralPadding() 
{
  ComputationBuilder builder(TestName());

  std::vector<float> input_data(1 * 2 * 3 * 1);
  std::iota(input_data.begin(), input_data.end(), 1.f);
  Array4D<float> input_array(1, 2, 3, 1, input_data);
  auto input = builder.ConstantR4FromArray4D<float>(input_array);

  std::vector<float> filter_data(1 * 1 * 1 * 1);
  std::iota(filter_data.begin(), filter_data.end(), 2.f);
  Array4D<float> filter_array(1, 1, 1, 1, filter_data);
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);

  ConvolutionDimensionNumbers dnums;
  // NHWC input format.
  dnums.set_batch_dimension(0);
  dnums.add_spatial_dimensions(1);
  dnums.add_spatial_dimensions(2);
  dnums.set_feature_dimension(3);

  // Tensorflow filter shape: [ H, W, inC, outC ]
  dnums.add_kernel_spatial_dimensions(0);
  dnums.add_kernel_spatial_dimensions(1);
  dnums.set_kernel_input_feature_dimension(2);
  dnums.set_kernel_output_feature_dimension(3);

  // Tests padding sizes that don't correspond either to SAME or VALID padding.
  builder.ConvGeneral(input, filter, {1, 1}, {{2, 1}, {2, 3}}, dnums);

  std::vector<float> expected_data = {
      0, 0, 0, 0,  0,  0, 0, 0,  //
      0, 0, 0, 0,  0,  0, 0, 0,  //
      0, 0, 2, 4,  6,  0, 0, 0,  //
      0, 0, 8, 10, 12, 0, 0, 0,  //
      0, 0, 0, 0,  0,  0, 0, 0   //
  };
  Array4D<float> expected(1, 5, 8, 1, expected_data);
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::Filter1x1x1x1Input1x2x3x1NoPadding() 
{
  ComputationBuilder builder(TestName());

  std::vector<float> input_data(1 * 2 * 3 * 1);
  std::iota(input_data.begin(), input_data.end(), 1.f);
  Array4D<float> input_array(1, 2, 3, 1, input_data);
  auto input = builder.ConstantR4FromArray4D<float>(input_array);

  std::vector<float> filter_data(1 * 1 * 1 * 1);
  std::iota(filter_data.begin(), filter_data.end(), 2.f);
  Array4D<float> filter_array(1, 1, 1, 1, filter_data);
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);

  ConvolutionDimensionNumbers dnums;
  // NHWC input format.
  dnums.set_batch_dimension(0);
  dnums.add_spatial_dimensions(1);
  dnums.add_spatial_dimensions(2);
  dnums.set_feature_dimension(3);

  // Tensorflow filter shape: [ H, W, inC, outC ]
  dnums.add_kernel_spatial_dimensions(0);
  dnums.add_kernel_spatial_dimensions(1);
  dnums.set_kernel_input_feature_dimension(2);
  dnums.set_kernel_output_feature_dimension(3);

  // Tests zero padding sizes. This can use matmul for computation.
  builder.ConvGeneral(input, filter, {1, 1}, {{0, 0}, {0, 0}}, dnums);

  std::vector<float> expected_data = {
      2, 4,  6,  //
      8, 10, 12,
  };
  Array4D<float> expected(1, 2, 3, 1, expected_data);
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

void ConvolutionVariantsTest::Filter1x1x2x3Input1x2x3x2NoPadding() 
{
  ComputationBuilder builder(TestName());

  std::vector<float> input_data(1 * 2 * 3 * 2);
  std::iota(input_data.begin(), input_data.end(), 1.f);
  Array4D<float> input_array(1, 2, 3, 2, input_data);
  auto input = builder.ConstantR4FromArray4D<float>(input_array);

  std::vector<float> filter_data(1 * 1 * 2 * 3);
  std::iota(filter_data.begin(), filter_data.end(), 2.f);
  Array4D<float> filter_array(1, 1, 2, 3, filter_data);
  auto filter = builder.ConstantR4FromArray4D<float>(filter_array);

  ConvolutionDimensionNumbers dnums;
  // NHWC input format.
  dnums.set_batch_dimension(0);
  dnums.add_spatial_dimensions(1);
  dnums.add_spatial_dimensions(2);
  dnums.set_feature_dimension(3);

  // Tensorflow filter shape: [ H, W, inC, outC ]
  dnums.add_kernel_spatial_dimensions(0);
  dnums.add_kernel_spatial_dimensions(1);
  dnums.set_kernel_input_feature_dimension(2);
  dnums.set_kernel_output_feature_dimension(3);

  // Tests zero padding sizes. This can use matmul for computation.
  builder.ConvGeneral(input, filter, {1, 1}, {{0, 0}, {0, 0}}, dnums);

  std::vector<float> expected_data = {
      12, 15,  18,   //
      26, 33,  40,   //
      40, 51,  62,   //
      54, 69,  84,   //
      68, 87,  106,  //
      82, 105, 128,  //
  };
  Array4D<float> expected(1, 2, 3, 3, expected_data);
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

// Regression test for b/32034796.
//
// XLA:GPU fuses
//   Conv([1,2,3], Reverse([5,6]), padding_low=1)
// into
//   BackwardInputConv([1,2,3], [5,6], padding_low=0, padding_high=1)
void ConvolutionVariantsTest::BackwardInputLowPaddingLessThanHighPadding() 
{
  ComputationBuilder builder(TestName());

  auto gradients = builder.ConstantR4FromArray4D<float>(
      Array4D<float>(1, 1, 1, 3, /*values=*/{1, 2, 3}));
  auto weights = builder.ConstantR4FromArray4D<float>(
      Array4D<float>(1, 1, 1, 2, /*values=*/{5, 6}));
  auto mirrored_weights = builder.Rev(weights, {2, 3});
  builder.ConvWithGeneralPadding(gradients, mirrored_weights,
                                 /*window_strides=*/{1, 1},
                                 /*padding=*/{{0, 0}, {1, 0}});
  ComputeAndCompareR4<float>(&builder, {{{{5.f, 16.f, 27.f}}}}, {}, error_spec_);
}

// XLA:GPU fuses
//   Conv([1], Reverse([1,10,100]), padding_high=3, base_dilation=3)
// into
//   BackwardInputConv([1], [1,10,100], stride=3, padding=(2,1))
void ConvolutionVariantsTest::BackwardInputLowPaddingGreaterThanHighPadding() 
{
  ComputationBuilder builder(TestName());

  auto gradients = builder.ConstantR4FromArray4D<float>(
      Array4D<float>(1, 1, 1, 1, /*values=*/{1}));
  auto weights = builder.ConstantR4FromArray4D<float>(
      Array4D<float>(1, 1, 1, 3, /*values=*/{1, 10, 100}));
  auto mirrored_weights = builder.Rev(weights, {2, 3});
  builder.ConvGeneralDilated(
      gradients, mirrored_weights,
      /*window_strides=*/{1, 1},
      /*padding=*/{{0, 0}, {0, 3}},
      /*lhs_dilation=*/{1, 3}, /*rhs_dilation=*/{},
      ComputationBuilder::CreateDefaultConvDimensionNumbers());
  ComputeAndCompareR4<float>(&builder, {{{{100, 0}}}}, {}, error_spec_);
}

// XLA:GPU fuses
//   Conv([1], Reverse([1,10,100]), padding=(1,1))
// into
//   BackwardInputConv([1], [1,10,100], padding=(1,1))
void ConvolutionVariantsTest::BackwardInputEvenPadding() 
{
  ComputationBuilder builder(TestName());

  auto gradients = builder.ConstantR4FromArray4D<float>(
      Array4D<float>(1, 1, 1, 1, /*values=*/{1}));
  auto weights = builder.ConstantR4FromArray4D<float>(
      Array4D<float>(1, 1, 1, 3, /*values=*/{1, 10, 100}));
  auto mirrored_weights = builder.Rev(weights, {2, 3});
  builder.ConvWithGeneralPadding(gradients, mirrored_weights,
                                 /*window_strides=*/{1, 1},
                                 /*padding=*/{{0, 0}, {1, 1}});
  ComputeAndCompareR4<float>(&builder, {{{{10}}}}, {}, error_spec_);
}

// HLO pattern
//   Conv([1,2,3], Reverse([1,10], padding_high=2)
// could be fused to
//   BackwardInputConv([1,2,3], [1,10], padding_low=1, padding_high=-1)
//
// However, XLA:GPU doesn't actually fuse it because PadInsertion doesn't
// support negative padding on backward convolution yet (b/32744257).
void ConvolutionVariantsTest::BackwardInputWithNegativePaddingHigh() 
{
  ComputationBuilder builder(TestName());

  auto gradients = builder.ConstantR4FromArray4D<float>(
      Array4D<float>(1, 1, 1, 3, /*values=*/{1, 2, 3}));
  auto weights = builder.ConstantR4FromArray4D<float>(
      Array4D<float>(1, 1, 1, 2, /*values=*/{1, 10}));
  auto mirrored_weights = builder.Rev(weights, {2, 3});
  builder.ConvWithGeneralPadding(gradients, mirrored_weights,
                                 /*window_strides=*/{1, 1},
                                 /*padding=*/{{0, 0}, {0, 2}});

  ComputeAndCompareR4<float>(&builder, {{{{12, 23, 30, 0}}}}, {}, error_spec_);
}

void ConvolutionVariantsTest::BackwardFilterLowPaddingLessThanHighPadding() 
{
  ComputationBuilder builder(TestName());

  // activations:      1,2,3,4  ---pad--> 0,1,2,3,4,0,0
  // gradients:        100,10,1 -dilate-> 100,0,10,0,1
  // weight gradients: 24,130,240
  //
  // This pattern will be fused to backward convolution with padding=(1,2).
  auto activations = builder.ConstantR4FromArray4D<float>(
      Array4D<float>(1, 1, 1, 4, /*values=*/{1, 2, 3, 4}));
  auto gradients = builder.ConstantR4FromArray4D<float>(
      Array4D<float>(1, 1, 1, 3, /*values=*/{100, 10, 1}));
  auto forward_conv = builder.ConvGeneralDilated(
      activations, gradients,
      /*window_strides=*/{1, 1},
      /*padding=*/{{0, 0}, {1, 2}},
      /*lhs_dilation=*/{}, /*rhs_dilation=*/{1, 2},
      ComputationBuilder::CreateDefaultConvDimensionNumbers());
  builder.Transpose(forward_conv, {0, 1, 2, 3});

  ComputeAndCompareR4<float>(&builder, {{{{24, 130, 240}}}}, {}, error_spec_);
}

void ConvolutionVariantsTest::BackwardFilterLowPaddingGreaterThanHighPadding() 
{
  ComputationBuilder builder(TestName());

  // activations:      1,2,3,4  ---pad--> 0,0,1,2,3,4
  // gradients:        100,10,1 -dilate-> 100,0,10,0,1
  // weight gradients: 13,24
  //
  // This pattern will be fused to backward convolution with padding=(2,1).
  // Note: both (2,1) and (2,0) are valid padding for the backward convolution
  // because the stride is 2.
  auto activations = builder.ConstantR4FromArray4D<float>(
      Array4D<float>(1, 1, 1, 4, /*values=*/{1, 2, 3, 4}));
  auto gradients = builder.ConstantR4FromArray4D<float>(
      Array4D<float>(1, 1, 1, 3, /*values=*/{100, 10, 1}));
  auto forward_conv = builder.ConvGeneralDilated(
      activations, gradients,
      /*window_strides=*/{1, 1},
      /*padding=*/{{0, 0}, {2, 0}},
      /*lhs_dilation=*/{}, /*rhs_dilation=*/{1, 2},
      ComputationBuilder::CreateDefaultConvDimensionNumbers());
  builder.Transpose(forward_conv, {0, 1, 2, 3});

  ComputeAndCompareR4<float>(&builder, {{{{13, 24}}}}, {}, error_spec_);
}

void ConvolutionVariantsTest::BackwardFilterEvenPadding() 
{
  ComputationBuilder builder(TestName());

  // activations:      1,2,3,4  ---pad--> 0,0,1,2,3,4,0
  // gradients:        100,10,1 -dilate-> 100,0,10,0,1
  // weight gradients: 13,24,130
  //
  // This pattern will be fused to backward convolution with padding=(2,2).
  // Note: both (2,1) and (2,2) are valid padding for the backward convolution
  // because the stride is 2. ConvolutionFolding prefers (2,2) because cuDNN
  // supports even padding only -- using (2,1) would need extra effort of
  // canonicalization.
  auto activations = builder.ConstantR4FromArray4D<float>(
      Array4D<float>(1, 1, 1, 4, /*values=*/{1, 2, 3, 4}));
  auto gradients = builder.ConstantR4FromArray4D<float>(
      Array4D<float>(1, 1, 1, 3, /*values=*/{100, 10, 1}));
  auto forward_conv = builder.ConvGeneralDilated(
      activations, gradients,
      /*window_strides=*/{1, 1},
      /*padding=*/{{0, 0}, {2, 1}},
      /*lhs_dilation=*/{}, /*rhs_dilation=*/{1, 2},
      ComputationBuilder::CreateDefaultConvDimensionNumbers());
  builder.Transpose(forward_conv, {0, 1, 2, 3});

  ComputeAndCompareR4<float>(&builder, {{{{13, 24, 130}}}}, {}, error_spec_);
}

void ConvolutionVariantsTest::run()
{
   Minimal();
   MinimalWithBatch();

   Flat1x1();
   Deep1x1();

   Filter1x2in1x2();
   Filter1x2in1x3();
   Filter1x2in2x2();
   Filter2x1in2x2();
   Filter2x2in2x2();

   Filter1x2in2x3WithDepthAndBatch();

   Filter1x1stride1x2in1x4();

   Filter1x1stride1x2in1x5();

   Filter1x3stride1x2in1x4();

   Filter1x3stride1x2in1x5();

   Filter1x1stride2x2in3x3();

   Filter3x1in1x1Padded();
   Filter5x1in3x1Padded();
   Filter3x3in2x2Padded();

   Filter1x1in2x1WithPaddingAndDepth();

   Filter2x2Stride1x1Input3x3();

   Filter1x2Stride1x1Input1x3();

   Filter2x1x8x8Input1x1x8x8();

   Filter1x1x1x1Input16x1x1x1();
   Filter1x1x2x2Input16x1x2x2();

   Filter1x1x2x2Input3x1x2x2();

   Filter1x1x8x8Input16x1x8x8();

   Filter2x2x8x8Input1x2x8x8();
   Filter2x2x8x8Input2x2x8x8();
   Filter2x2x8x8Input32x2x8x8();
   Filter16x16x1x1Input16x16x1x1();

   FlatRhsDilation();
   FlatLhsDilation1D();
   FlatLhsDilation();

   NegativePaddingOnBothEnds();
   NegativePaddingLowAndPositivePaddingHigh();
   PositivePaddingLowAndNegativePaddingHigh();
   PositivePaddingAndDilation();
   NegativePaddingAndDilation();

   RandomData_Input1x1x2x3_Filter2x1x1x2();
   RandomData_Input1x16x1x1_Filter1x16x1x1();

   RandomData_Input16x16x1x1_Filter1x16x1x1();

   RandomData_Input16x16x1x1_Filter16x16x1x1();

   RandomData_Input16x16x16x16_Filter16x16x16x16();

   Filter1x2x1x1Input1x2x3x1GeneralPadding();

   Filter1x1x1x1Input1x2x3x1GeneralPadding();

   Filter1x1x1x1Input1x2x3x1NoPadding();
   Filter1x1x2x3Input1x2x3x2NoPadding();

   BackwardInputLowPaddingLessThanHighPadding();

   BackwardInputLowPaddingGreaterThanHighPadding();

   BackwardInputEvenPadding();

   BackwardInputWithNegativePaddingHigh();
   BackwardFilterLowPaddingLessThanHighPadding();

   BackwardFilterLowPaddingGreaterThanHighPadding();
   BackwardFilterEvenPadding();
}

}  // namespace
}  // namespace xla
//
//int main(int argc, char** argv) {
//  std::vector<tensorflow::Flag> flag_list;
//  xla::legacy_flags::AppendCpuCompilerFlags(&flag_list);
//  xla::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
//  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
//  if (!parse_result) {
//    LOG(ERROR) << "\n" << usage;
//    return 2;
//  }
//  testing::InitGoogleTest(&argc, argv);
//  if (argc > 1) {
//    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
//    return 2;
//  }
//  return RUN_ALL_TESTS();
//}
