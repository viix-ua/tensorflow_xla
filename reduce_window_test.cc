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

// Tests the reduce-window XLA operation.

#include <limits>
#include <memory>

#include "array2d.h"
#include "array3d.h"
#include "array4d.h"
#include "computation_builder.h"
#include "arithmetic.h"
//#include "tensorflow/compiler/xla/client/local_client.h"
#include "padding.h"
//#include "tensorflow/compiler/xla/legacy_flags/cpu_compiler_flags.h"
#include "reference_util.h"
#include "shape_util.h"
#include "client_library_test_base.h"
#include "literal_test_util.h"
//#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "xla_data.pb.h"
#include "array_slice.h"
//#include "test.h"
#include "base.h"
#include "test_helpers.h"

namespace xla {

class ReduceWindowTest : public ClientLibraryTestBase 
{
 public:
  ReduceWindowTest() : builder_(TestName()) 
  {}

  void ReduceWindowAdd(xla::ComputationDataHandle input,
                       tensorflow::gtl::ArraySlice<int64> window_dimensions,
                       tensorflow::gtl::ArraySlice<int64> window_strides,
                       xla::Padding padding) {
    builder_.ReduceWindow(input, builder_.ConstantR0<float>(0.0f),
                          CreateScalarAddComputation(F32, &builder_),
                          window_dimensions, window_strides, padding);
  }

  void ReduceWindowMax(xla::ComputationDataHandle input,
                       tensorflow::gtl::ArraySlice<int64> window_dimensions,
                       tensorflow::gtl::ArraySlice<int64> window_strides,
                       xla::Padding padding) {
    builder_.ReduceWindow(
        input, builder_.ConstantLiteral(xla::LiteralUtil::MinValue(F32)),
        CreateScalarMax(), window_dimensions, window_strides, padding);
  }

  void ReduceWindowMin(xla::ComputationDataHandle input,
                       tensorflow::gtl::ArraySlice<int64> window_dimensions,
                       tensorflow::gtl::ArraySlice<int64> window_strides,
                       xla::Padding padding) {
    builder_.ReduceWindow(input,
                          builder_.ConstantLiteral(xla::LiteralUtil::MaxValue(F32)),
                          CreateScalarMinComputation(F32, &builder_),
                          window_dimensions, window_strides, padding);
  }

  xla::ComputationBuilder builder_;

///////////////////////////////////////////////////////////
  inline void ZeroElementSmall();
  inline void NonSquareSmall();
  inline void MiddleDimsSmall();
  inline void Along2ndMinorDim();
  inline void AmongMajor2DimsMediumSize();
  inline void AmongMajor2DimsMediumSizeLargePadding();
  inline void ReduceR4AmongXYMinorSmall();
  inline void ReduceR4AmongXYMinorSmallOverlapped();
  inline void MaxTrivial();
  inline void Add3In3();
  inline void Add4In16Stride4();
  inline void Min3In5Stride2();
  inline void Max3In3();
  inline void Add2In3();
  inline void Add3In5Stride2();
  inline void Max4In16Stride4();
  inline void Max4In16Stride3();
  inline void Max4In16Stride8();
  inline void Max3In5Stride2();
  inline void Max3In5Stride1();
  inline void Add3In4Stride2();
  inline void Add2In3SamePad();
  inline void Add3In3SamePad();
  inline void Add3In3Stride3SamePad();
  inline void Add2x2In2x2Overlapped();
  inline void Add2x2In2x2Disjoint();
  inline void Add1x1x2In2x1x2();
  inline void Add1x1x2In2x1x3Stride1x1x2();
  inline void Add1x1x2In2x1x3SamePad();
  inline void NonstandardReduceFunction();

  ///////////////////////////////////////////
  inline void run();
};

void ReduceWindowTest::ZeroElementSmall() 
{
  xla::Array4D<float> input_array(1, 0, 2, 1);

  const auto input = builder_.ConstantR4FromArray4D<float>(input_array);
  xla::Padding padding = xla::Padding::kSame;
  ReduceWindowAdd(input, {1, 1, 2, 1}, {1, 1, 1, 1}, padding);

  auto res = ReferenceUtil::ReduceWindow4D(input_array, {1, 1, 2, 1},
                                              {1, 1, 1, 1}, padding);

  ComputeAndCompareR4<float>(&builder_, *res, {}, xla::ErrorSpec(1e-3f, 1e-3f));
}

void ReduceWindowTest::NonSquareSmall() 
{
  xla::Array4D<float> input_array(1, 2, 2, 1);
  input_array.FillRandom(2.f);

  const auto input = builder_.ConstantR4FromArray4D<float>(input_array);
  xla::Padding padding = xla::Padding::kSame;
  ReduceWindowAdd(input, {1, 1, 2, 1}, {1, 1, 1, 1}, padding);

  auto res = ReferenceUtil::ReduceWindow4D(input_array, {1, 1, 2, 1},
                                              {1, 1, 1, 1}, padding);

  ComputeAndCompareR4<float>(&builder_, *res, {}, xla::ErrorSpec(1e-3f, 1e-3f));
}

void ReduceWindowTest::MiddleDimsSmall() 
{
  xla::Array4D<float> input_array(1, 3, 3, 1);
  input_array.FillRandom(2.f);

  const auto input = builder_.ConstantR4FromArray4D<float>(input_array);
  xla::Padding padding = xla::Padding::kSame;
  ReduceWindowAdd(input, {1, 1, 1, 1}, {1, 2, 2, 1}, padding);

  auto res = xla::ReferenceUtil::ReduceWindow4D(input_array, {1, 1, 1, 1},
                                              {1, 2, 2, 1}, padding);

  ComputeAndCompareR4<float>(&builder_, *res, {}, xla::ErrorSpec(1e-3f, 1e-3f));
}

void ReduceWindowTest::Along2ndMinorDim() 
{
  xla::Array4D<float> input_array(3, 6, 7, 32);
  input_array.FillRandom(2.f);

  // The parameters of this reduction mimic feature norm (e.g. LRN).
  int lrn_diameter = 7;  // diameter = 2*radius + 1 --> must be odd
  const auto input = builder_.ConstantR4FromArray4D<float>(input_array);
  xla::Padding padding = xla::Padding::kSame;
  ReduceWindowAdd(input, {1, 1, lrn_diameter, 1}, {1, 1, 1, 1}, padding);

  auto res = xla::ReferenceUtil::ReduceWindow4D(input_array, {1, 1, lrn_diameter, 1}, {1, 1, 1, 1}, padding);

  ComputeAndCompareR4<float>(&builder_, *res, {}, xla::ErrorSpec(1e-3f, 1e-3f));
}

void ReduceWindowTest::AmongMajor2DimsMediumSize() 
{
  xla::Array4D<float> input_array(9, 12, 4, 89);
  input_array.FillRandom(2.0f);

  int win_len = 3;
  int win_stride = 2;

  const auto input_data_handle =
      builder_.ConstantR4FromArray4D<float>(input_array);

  xla::Padding padding = xla::Padding::kSame;
  // Reduce only along the x and y dimensions, according to the win_len.
  ReduceWindowAdd(input_data_handle, {win_len, win_len, 1, 1},
                  {win_stride, win_stride, 1, 1}, padding);

  auto result = xla::ReferenceUtil::ReduceWindow4D(
      input_array, {win_len, win_len, 1, 1}, {win_stride, win_stride, 1, 1}, padding);

  ComputeAndCompareR4<float>(&builder_, *result, {}, xla::ErrorSpec(1e-3f, 1e-3f));
}

// TODO(b/32173947): Test support for arbitrary-sized padding.
void ReduceWindowTest::AmongMajor2DimsMediumSizeLargePadding() 
{
  xla::Array4D<float> input_array(9, 12, 4, 89);  // simulate Dim0IsMinor layout
  input_array.FillRandom(2.0f);

  int64 rank = 4;
  int win_len = 3;
  int win_stride = 2;

  const auto input_data_handle =
      builder_.ConstantR4FromArray4D<float>(input_array);

  xla::Padding padding = xla::Padding::kSame;
  // Reduce only along the x and y dimensions, according to the win_len.
  // Create padding vector with large padding values in the reduction dims.
  std::vector<std::pair<int64, int64>> low_high_padding;
  low_high_padding.resize(rank, {4, 4});

  builder_.ReduceWindowWithGeneralPadding(
      input_data_handle, builder_.ConstantR0<float>(0.0f),
      CreateScalarAddComputation(F32, &builder_), {win_len, win_len, 1, 1},
      {win_stride, win_stride, 1, 1}, low_high_padding);

  auto result = xla::ReferenceUtil::ReduceWindow4D(
      input_array, {win_len, win_len, 1, 1},
      {win_stride, win_stride, 1, 1}, padding);

  ComputeAndCompareR4<float>(&builder_, *result, {}, xla::ErrorSpec(1e-3f, 1e-3f));
}

// TODO(b/31809540): Implement minor dim reduction to reduce num of reshapes.
void ReduceWindowTest::ReduceR4AmongXYMinorSmall() 
{
  xla::Array4D<float> input_array(2, 2, 4, 16);

  xla::Array2D<float> yx({{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f,
                      11.f, 12.f, 13.f, 14.f, 15.f},
                     {16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f,
                      25.f, 26.f, 27.f, 28.f, 29.f, 30.f, 31.f},
                     {32.f, 33.f, 34.f, 35.f, 36.f, 37.f, 38.f, 39.f, 40.f,
                      41.f, 42.f, 43.f, 44.f, 45.f, 46.f, 47.f},
                     {48.f, 49.f, 50.f, 51.f, 52.f, 53.f, 54.f, 55.f, 56.f,
                      57.f, 58.f, 59.f, 60.f, 61.f, 62.f, 63.f}});
  input_array.FillWithYX(yx);

  int win_len = 2;
  int win_stride = 2;
  const auto input = builder_.ConstantR4FromArray4D<float>(input_array);
  xla::Padding padding = xla::Padding::kValid;
  ReduceWindowAdd(input, {1, 1, win_len, win_len},
                  {1, 1, win_stride, win_stride}, padding);

  auto res = xla::ReferenceUtil::ReduceWindow4D(
      input_array, {1, 1, win_len, win_len},
      {1, 1, win_stride, win_stride}, padding);
  ComputeAndCompareR4<float>(&builder_, *res, {}, xla::ErrorSpec(1e-3f, 1e-3f));
}

// TODO(b/31809540): Implement minor dim reduction to reduce num of reshapes.
void ReduceWindowTest::ReduceR4AmongXYMinorSmallOverlapped() 
{
  constexpr int64 p = 2;
  constexpr int64 z = 2;
  constexpr int64 y = 4;
  constexpr int64 x = 16;
  xla::Array4D<float> input_array(p, z, y, x);

  xla::Array2D<float> yx({{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f,
                      11.f, 12.f, 13.f, 14.f, 15.f},
                     {16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f,
                      25.f, 26.f, 27.f, 28.f, 29.f, 30.f, 31.f},
                     {32.f, 33.f, 34.f, 35.f, 36.f, 37.f, 38.f, 39.f, 40.f,
                      41.f, 42.f, 43.f, 44.f, 45.f, 46.f, 47.f},
                     {48.f, 49.f, 50.f, 51.f, 52.f, 53.f, 54.f, 55.f, 56.f,
                      57.f, 58.f, 59.f, 60.f, 61.f, 62.f, 63.f}});
  input_array.FillWithYX(yx);

  int win_len = 4;
  int win_stride = 2;
  const auto input = builder_.ConstantR4FromArray4D<float>(input_array);
  ReduceWindowAdd(input, {1, 1, win_len, win_len},
                  {1, 1, win_stride, win_stride}, xla::Padding::kValid);

  // Expected result
  xla::Array2D<float> yx_result({{408.f, 440.f, 472.f, 504.f, 536.f, 568.f, 600.f}});
  xla::Array4D<float> expected(p, z, 1, 7);
  expected.FillWithYX(yx_result);
  ComputeAndCompareR4<float>(&builder_, expected, {}, xla::ErrorSpec(1e-3f, 1e-3f));

  auto reduce4d = xla::ReferenceUtil::ReduceWindow4D(input_array, { 1, 1, win_len, win_len }, { 1, 1, win_stride, win_stride }, xla::Padding::kValid);
  ASSERT_TRUE(*reduce4d == expected);
}

void ReduceWindowTest::MaxTrivial() 
{
  const auto input = builder_.ConstantR1<float>({42});
  ReduceWindowMax(input, {1}, {1}, xla::Padding::kValid);
  ComputeAndCompareR1<float>(&builder_, {42}, {}, xla::ErrorSpec(0.0001f));
}

void ReduceWindowTest::Add3In3() 
{
  const auto input = builder_.ConstantR1<float>({20, 100, 3});
  ReduceWindowAdd(input, {3}, {1}, xla::Padding::kValid);
  ComputeAndCompareR1<float>(&builder_, {123}, {}, xla::ErrorSpec(0.0001f));
}

void ReduceWindowTest::Add4In16Stride4() 
{
  const auto input = builder_.ConstantR1<float>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ReduceWindowAdd(input, {4}, {4}, xla::Padding::kValid);
  ComputeAndCompareR1<float>(&builder_, {10, 26, 42, 58}, {},
     xla::ErrorSpec(0.0001f));
}

void ReduceWindowTest::Min3In5Stride2()
{
  const auto input = builder_.ConstantR1<float>({10000, 1000, 100, 10, 1});
  ReduceWindowMin(input, {3}, {2}, xla::Padding::kValid);
  ComputeAndCompareR1<float>(&builder_, {100, 1}, {}, xla::ErrorSpec(0.0001f));
}

void ReduceWindowTest::Max3In3() 
{
  const auto input = builder_.ConstantR1<float>({20, 100, 3});
  ReduceWindowMax(input, {3}, {1}, xla::Padding::kValid);
  ComputeAndCompareR1<float>(&builder_, {100}, {}, xla::ErrorSpec(0.0001f));
}

void ReduceWindowTest::Add2In3() 
{
  const auto input = builder_.ConstantR1<float>({100, 10, 1});
  ReduceWindowAdd(input, {2}, {1}, xla::Padding::kValid);
  ComputeAndCompareR1<float>(&builder_, {110, 11}, {}, xla::ErrorSpec(0.0001f));
}

void ReduceWindowTest::Add3In5Stride2() 
{
  const auto input = builder_.ConstantR1<float>({10000, 1000, 100, 10, 1});
  ReduceWindowAdd(input, {3}, {2}, xla::Padding::kValid);
  ComputeAndCompareR1<float>(&builder_, {11100, 111}, {}, xla::ErrorSpec(0.0001f));
}

void ReduceWindowTest::Max4In16Stride4() 
{
  const auto input = builder_.ConstantR1<float>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ReduceWindowMax(input, {4}, {4}, xla::Padding::kValid);
  ComputeAndCompareR1<float>(&builder_, {4, 8, 12, 16}, {}, xla::ErrorSpec(0.0001f));
}

void ReduceWindowTest::Max4In16Stride3() 
{
  const auto input = builder_.ConstantR1<float>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ReduceWindowMax(input, {4}, {3}, xla::Padding::kValid);
  ComputeAndCompareR1<float>(&builder_, {4, 7, 10, 13, 16}, {},
     xla::ErrorSpec(0.0001f));
}

void ReduceWindowTest::Max4In16Stride8() 
{
  const auto input = builder_.ConstantR1<float>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ReduceWindowMax(input, {4}, {8}, xla::Padding::kValid);
  ComputeAndCompareR1<float>(&builder_, {4, 12}, {}, xla::ErrorSpec(0.0001f));
}

void ReduceWindowTest::Max3In5Stride2() 
{
  const auto input = builder_.ConstantR1<float>({10000, 1000, 100, 10, 1});
  ReduceWindowMax(input, {3}, {2}, xla::Padding::kValid);
  ComputeAndCompareR1<float>(&builder_, {10000, 100}, {}, xla::ErrorSpec(0.0001f));
}

void ReduceWindowTest::Max3In5Stride1() 
{
  const auto input = builder_.ConstantR1<float>({10000, 1000, 100, 10, 101});
  ReduceWindowMax(input, {3}, {1}, xla::Padding::kValid);
  ComputeAndCompareR1<float>(&builder_, {10000, 1000, 101}, {},
     xla::ErrorSpec(0.0001f));
}

void ReduceWindowTest::Add3In4Stride2() 
{
  const auto input = builder_.ConstantR1<float>({1000, 100, 10, 1});
  ReduceWindowAdd(input, {3}, {2}, Padding::kValid);
  ComputeAndCompareR1<float>(&builder_, {1110}, {}, ErrorSpec(0.0001f));
}

void ReduceWindowTest::Add2In3SamePad() 
{
  const auto input = builder_.ConstantR1<float>({100, 10, 1});
  ReduceWindowAdd(input, {2}, {1}, xla::Padding::kSame);
  ComputeAndCompareR1<float>(&builder_, {110, 11, 1}, {}, xla::ErrorSpec(0.0001f));
}

void ReduceWindowTest::Add3In3SamePad() 
{
  const auto input = builder_.ConstantR1<float>({100, 10, 1});
  ReduceWindowAdd(input, {3}, {1}, xla::Padding::kSame);
  ComputeAndCompareR1<float>(&builder_, {110, 111, 11}, {}, xla::ErrorSpec(0.0001f));
}

void ReduceWindowTest::Add3In3Stride3SamePad() 
{
  const auto input = builder_.ConstantR1<float>({100, 10, 1});
  ReduceWindowAdd(input, {3}, {2}, xla::Padding::kSame);
  ComputeAndCompareR1<float>(&builder_, {110, 11}, {}, xla::ErrorSpec(0.0001f));
}

void ReduceWindowTest::Add2x2In2x2Overlapped() 
{
   xla::Array2D<float> input_array({{1.2f, -2.5f, 0.9f, 1.0f},
                              {3.7f, 0.2f, -1.0f, -0.2f},
                              {-0.4f, 2.7f, 1.1f, 2.2f},
                              {0.6f, 1.7f, 1.4f, -0.2f}});
  auto input = builder_.ConstantR2FromArray2D<float>(input_array);
  ReduceWindowAdd(input, {2, 2}, {1, 1}, xla::Padding::kValid);
  xla::Array2D<float> expected(
      {{2.6f, -2.4f, 0.7f}, {6.2f, 3.0f, 2.1f}, {4.6f, 6.9f, 4.5f}});
  ComputeAndCompareR2<float>(&builder_, expected, {}, ErrorSpec(0.0001f));

  auto reduce2d = xla::ReferenceUtil::ReduceWindow2D(input_array, { 2, 2 }, { 1, 1 }, xla::Padding::kValid);
  ASSERT_TRUE(*reduce2d == expected);
}

void ReduceWindowTest::Add2x2In2x2Disjoint() 
{
   xla::Array2D<float> input_array({{1.2f, -2.5f, 0.9f, 1.0f},
                              {3.7f, 0.2f, -1.0f, -0.2f},
                              {-0.4f, 2.7f, 1.1f, 2.2f},
                              {0.6f, 1.7f, 1.4f, -0.2f}});
  auto input = builder_.ConstantR2FromArray2D<float>(input_array);
  ReduceWindowAdd(input, {2, 2}, {2, 2}, xla::Padding::kValid);
  xla::Array2D<float> expected({
      {2.6f, 0.7f}, {4.6f, 4.5f},
  });
  ComputeAndCompareR2<float>(&builder_, expected, {}, xla::ErrorSpec(0.0001f));

  auto reduce2d = xla::ReferenceUtil::ReduceWindow2D(input_array, { 2, 2 }, { 2, 2 }, xla::Padding::kValid);
  ASSERT_TRUE(*reduce2d == expected);
}

void ReduceWindowTest::Add1x1x2In2x1x2() 
{
  xla::Array3D<float> input_array(2, 1, 2);
  input_array(0, 0, 0) = 1000;
  input_array(0, 0, 1) = 100;
  input_array(1, 0, 0) = 10;
  input_array(1, 0, 1) = 1;
  auto input = builder_.ConstantR3FromArray3D<float>(input_array);

  ReduceWindowAdd(input, {1, 1, 2}, {1, 1, 1}, xla::Padding::kValid);

  xla::Array3D<float> expected(2, 1, 1);
  expected(0, 0, 0) = 1100;
  expected(1, 0, 0) = 11;
  ComputeAndCompareR3<float>(&builder_, expected, {}, xla::ErrorSpec(0.0001f));

  auto reduce3d = xla::ReferenceUtil::ReduceWindow3D(input_array, { 1, 1, 2 }, { 1, 1, 1 }, xla::Padding::kValid);
  ASSERT_TRUE(*reduce3d == expected);
}

void ReduceWindowTest::Add1x1x2In2x1x3Stride1x1x2() 
{
  xla::Array3D<float> input_array(2, 1, 3);
  input_array(0, 0, 0) = 100;
  input_array(0, 0, 1) = 10;
  input_array(0, 0, 2) = 1;
  input_array(1, 0, 0) = 500;
  input_array(1, 0, 1) = 50;
  input_array(1, 0, 2) = 5;
  auto input = builder_.ConstantR3FromArray3D<float>(input_array);

  ReduceWindowAdd(input, {1, 1, 2}, {1, 1, 2}, xla::Padding::kValid);

  xla::Array3D<float> expected(2, 1, 1);
  expected(0, 0, 0) = 110;
  expected(1, 0, 0) = 550;
  ComputeAndCompareR3<float>(&builder_, expected, {}, xla::ErrorSpec(0.0001f));

  auto reduce3d = xla::ReferenceUtil::ReduceWindow3D(input_array, { 1, 1, 2 }, { 1, 1, 2 }, xla::Padding::kValid);
  ASSERT_TRUE(*reduce3d == expected);
}

void ReduceWindowTest::Add1x1x2In2x1x3SamePad() 
{
  xla::Array3D<float> input_array(2, 1, 3);
  input_array(0, 0, 0) = 100;
  input_array(0, 0, 1) = 10;
  input_array(0, 0, 2) = 1;
  input_array(1, 0, 0) = 500;
  input_array(1, 0, 1) = 50;
  input_array(1, 0, 2) = 5;
  auto input = builder_.ConstantR3FromArray3D<float>(input_array);

  ReduceWindowAdd(input, {1, 1, 2}, {1, 1, 1}, xla::Padding::kSame);

  xla::Array3D<float> expected(2, 1, 3);
  expected(0, 0, 0) = 110;
  expected(0, 0, 1) = 11;
  expected(0, 0, 2) = 1;
  expected(1, 0, 0) = 550;
  expected(1, 0, 1) = 55;
  expected(1, 0, 2) = 5;
  ComputeAndCompareR3<float>(&builder_, expected, {}, xla::ErrorSpec(0.0001f));

  auto reduce3d = xla::ReferenceUtil::ReduceWindow3D(input_array, { 1, 1, 2 }, { 1, 1, 1 }, xla::Padding::kSame);
  ASSERT_TRUE(*reduce3d == expected);
}

// Tests a reduction function that is not a simple add/min/max/etc.
void ReduceWindowTest::NonstandardReduceFunction() 
{
  xla::Array4D<float> input_array(1, 2, 2, 1);
  input_array(0, 0, 0, 0) = 1;
  input_array(0, 0, 1, 0) = 2;
  input_array(0, 1, 0, 0) = 3;
  input_array(0, 1, 1, 0) = 4;

  const auto input = builder_.ConstantR4FromArray4D<float>(input_array);
  xla::Padding padding = xla::Padding::kValid;

  const Shape scalar = xla::ShapeUtil::MakeShape(F32, {});
  auto b = builder_.CreateSubBuilder("unusual");
  auto lhs = b->Parameter(0, scalar, "lhs");
  auto rhs = b->Parameter(1, scalar, "rhs");
  b->Min(b->Add(lhs, rhs), b->ConstantR0<float>(8.0f));
  Computation reduce_fn = b->BuildAndNoteError();

  builder_.ReduceWindow(input, builder_.ConstantR0<float>(3.0f), reduce_fn,
                        /*window_dimensions=*/{1, 1, 2, 1},
                        /*window_strides=*/{1, 1, 1, 1}, padding);

  xla::Array4D<float> expected(1, 2, 1, 1);
  expected(0, 0, 0, 0) = 6;
  expected(0, 1, 0, 0) = 8;

  ComputeAndCompareR4<float>(&builder_, expected, {}, xla::ErrorSpec(1e-3f, 1e-3f));
}

void ReduceWindowTest::run()
{
   ZeroElementSmall();
   NonSquareSmall();
   MiddleDimsSmall();
   Along2ndMinorDim();
   AmongMajor2DimsMediumSize();
   AmongMajor2DimsMediumSizeLargePadding();
   ReduceR4AmongXYMinorSmall();
   ReduceR4AmongXYMinorSmallOverlapped();
   MaxTrivial();
   Add3In3();
   Add4In16Stride4();
   Min3In5Stride2();
   Max3In3();
   Add2In3();
   Add3In5Stride2();
   Max4In16Stride4();
   Max4In16Stride3();
   Max4In16Stride8();
   Max3In5Stride2();
   Max3In5Stride1();
   Add3In4Stride2();
   Add2In3SamePad();
   Add3In3SamePad();
   Add3In3Stride3SamePad();
   Add2x2In2x2Overlapped();
   Add2x2In2x2Disjoint();
   Add1x1x2In2x1x2();
   Add1x1x2In2x1x3Stride1x1x2();
   Add1x1x2In2x1x3SamePad();
   NonstandardReduceFunction();
}


}  // namespace xla

/*
int main(int argc, char** argv) {
  std::vector<tensorflow::Flag> flag_list;
  xla::legacy_flags::AppendCpuCompilerFlags(&flag_list);
  xla::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << "\n" << usage;
    return 2;
  }
  testing::InitGoogleTest(&argc, argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return 2;
  }
  return RUN_ALL_TESTS();
}
*/