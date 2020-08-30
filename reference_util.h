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

#ifndef TENSORFLOW_COMPILER_XLA_REFERENCE_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_REFERENCE_UTIL_H_

#include <array>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "array2d.h"
#include "array3d.h"
#include "array4d.h"
#include "padding.h"
#include "ptr_util.h"
#include "xla_data.pb.h"
#include "array_slice.h"
#include "macros.h"
#include "types.h"

namespace xla {

// Utility class for reference implementations of linear algebra routines.
class ReferenceUtil {
 public:
  // Returns the result of a transpose operation on the input matrix.
  static std::unique_ptr<Array2D<float>> TransposeArray2D(
      const Array2D<float>& operand);

  // Returns the result of a matrix multiply `lhs x rhs`.
  static std::unique_ptr<Array2D<float>> MatmulArray2D(
      const Array2D<float>& lhs, const Array2D<float>& rhs);
  static std::unique_ptr<Array2D<double>> MatmulArray2D(
      const Array2D<double>& lhs, const Array2D<double>& rhs);

  // Converts the input operand to use f64 values instead of f32 values.
  static std::unique_ptr<Array2D<double>> Array2DF32ToF64(
      const Array2D<float>& input);

  // Returns the result of a convolution `lhs <conv> rhs`, with the default
  // convolution dimension numbers returned from
  // ComputationBuilder::CreateDefaultConvDimensionNumbers().
  static std::unique_ptr<Array4D<float>> ConvArray4D(
      const Array4D<float>& lhs, const Array4D<float>& rhs,
      std::pair<int64, int64> kernel_stride, Padding padding);

  // Returns the result of a convolution `lhs <conv> rhs`, with the given
  // convolution dimension numbers.
  static std::unique_ptr<Array4D<float>> ConvArray4DGeneralDimensions(
      const Array4D<float>& lhs, const Array4D<float>& rhs,
      std::pair<int64, int64> kernel_stride, Padding padding,
      ConvolutionDimensionNumbers dimension_numbers);

  // Returns the result of a convolution `lhs <conv> rhs`, with the given
  // dilation factors.
  static std::unique_ptr<Array4D<float>> ConvArray4DGeneralDimensionsDilated(
      const Array4D<float>& lhs, const Array4D<float>& rhs,
      std::pair<int64, int64> stride, Padding padding,
      std::pair<int64, int64> lhs_dilation,
      std::pair<int64, int64> rhs_dilation, ConvolutionDimensionNumbers dnums);

  // Returns the result of a convolution `lhs <conv> rhs`, with the default
  // convolution dimension numbers returned from
  // ComputationBuilder::CreateDefaultConvDimensionNumbers().
  static std::unique_ptr<Array3D<float>> ConvArray3D(const Array3D<float>& lhs,
                                                     const Array3D<float>& rhs,
                                                     int64 kernel_stride,
                                                     Padding padding);

  // Returns the result of a convolution `lhs <conv> rhs`.
  static std::unique_ptr<Array3D<float>> ConvArray3DGeneralDimensionsDilated(
      const Array3D<float>& lhs, const Array3D<float>& rhs, int64 kernel_stride,
      Padding padding, int64 lhs_dilation, int64 rhs_dilation,
      const ConvolutionDimensionNumbers& dnums);

  // Returns the result of a separable  convolution with the given parameters.
  // kernel_stride and padding applies to the depthwise convolution during
  // the separable convolution. pointwise_weights.depth() must be equal to
  // input.depth() * depthwise_weights.planes().
  static std::unique_ptr<Array4D<float>> SeparableConvArray4D(
      const Array4D<float>& input, const Array4D<float>& depthwise_weights,
      const Array4D<float>& pointwise_weights,
      std::pair<int64, int64> kernel_stride, Padding padding);

  // Returns the result of reducing a matrix to a column vector. init is the
  // initial value for the reduce operation, and reduce_function is the function
  // to apply for each reduction step.
  static std::unique_ptr<std::vector<float>> ReduceToColArray2D(
      const Array2D<float>& matrix, float init,
      const std::function<float(float, float)>& reduce_function);

  // Returns the result of reducing a matrix to a row vector. init is the
  // initial value for the reduce operation, and reduce_function is the function
  // to apply for each reduction step.
  static std::unique_ptr<std::vector<float>> ReduceToRowArray2D(
      const Array2D<float>& matrix, float init,
      const std::function<float(float, float)>& reduce_function);

  // Performs a R2=>R1 reduction by reducing away the dimension specified in
  // 'dimension_to_reduce'.
  template <typename T>
  static std::vector<T> ReduceR2ToR1(const Array2D<T>& input,
                                     int dimension_to_reduce, T init,
                                     const std::function<T(T, T)>& freduce) {
    std::vector<T> result(dimension_to_reduce == 0 ? input.n2() : input.n1(),
                          init);
    for (int i0 = 0; i0 < input.n1(); ++i0) {
      for (int i1 = 0; i1 < input.n2(); ++i1) {
        int output = dimension_to_reduce == 0 ? i1 : i0;
        result[output] = freduce(result[output], input(i0, i1));
      }
    }
    return result;
  }

  // Returns the result of reducing the 4D array to a vector, reducing away
  // the dimensions specified in dims.
  static std::vector<float> Reduce4DTo1D(
      const Array4D<float>& array, float init,
      tensorflow::gtl::ArraySlice<int64> dims,
      const std::function<float(float, float)>& reduce_function);

  // Broadcast 1D dimension to 4D, from the dimension `broadcast_from_dim`.
  static std::unique_ptr<Array4D<float>> Broadcast1DTo4D(
      const std::vector<float>& array, const std::vector<int64>& bounds,
      int64 broadcast_from_dim);

  // Returns the result of reducing the 3D array to a 2D array, reducing away
  // the dimensions specified in dims.
  static std::unique_ptr<Array2D<float>> Reduce3DTo2D(
      const Array3D<float>& array, float init,
      tensorflow::gtl::ArraySlice<int64> dims,
      const std::function<float(float, float)>& reduce_function);

  // Applies map_function to each element in the input (2D array) and returns
  // the result.
  static std::unique_ptr<Array2D<float>> MapArray2D(
      const Array2D<float>& matrix,
      const std::function<float(float)>& map_function);

  // Applies map_function to each pair of corresponding elements in the two
  // inputs arrays and returns the result.
  static std::unique_ptr<Array2D<float>> MapArray2D(
      const Array2D<float>& lhs, const Array2D<float>& rhs,
      const std::function<float(float, float)>& map_function);

  // Number of windows in a given dimension. Calculation taken from
  // xla::MakePadding().
  static int64 WindowCount(int64 unpadded_width, int64 window_len, int64 stride,
                           Padding padding);

  static std::unique_ptr<Array2D<float>> ReduceWindow2D(
     const Array2D<float>& operand, 
     const tensorflow::gtl::ArraySlice<int64>& window,
     const tensorflow::gtl::ArraySlice<int64>& stride, Padding padding);

  // Performs a 3D window reduction with Add as the function to apply.
  static std::unique_ptr<Array3D<float>> ReduceWindow3D(
     const Array3D<float>& operand, 
     const tensorflow::gtl::ArraySlice<int64>& window,
     const tensorflow::gtl::ArraySlice<int64>& stride, Padding padding);

  // Performs a 4D window reduction with Add as the function to apply.
  static std::unique_ptr<Array4D<float>> ReduceWindow4D(
     const Array4D<float>& operand,
     const tensorflow::gtl::ArraySlice<int64>& window,
     const tensorflow::gtl::ArraySlice<int64>& stride, Padding padding)
  {
     return ReduceWindow4DAdd(operand, 0.0f, window, stride, padding);
  }

  // Performs a 4D window reduction with Add as the function to apply.
  static std::unique_ptr<Array4D<float>> ReduceWindow4DAdd(
      const Array4D<float>& operand, float init,
      const tensorflow::gtl::ArraySlice<int64>& window,
      const tensorflow::gtl::ArraySlice<int64>& stride, Padding padding);

  // Batch normalize data.
  static std::unique_ptr<Array4D<float>> BatchNorm4D(
      const Array4D<float>& input, const Array4D<float>& mean,
      const Array4D<float>& var, const Array4D<float>& scale,
      const Array4D<float>& offset, float epsilon);

  // Performs select and scatter with Greater Than or equal as the select, plus
  // as the scatter, and Same Padding.
  static std::unique_ptr<Array4D<float>> SelectAndScatter4DGePlus(
      const Array4D<float>& operand, const Array4D<float>& source, float init,
      const tensorflow::gtl::ArraySlice<int64>& window,
      const tensorflow::gtl::ArraySlice<int64>& stride, bool same_padding);

  // Concatenates the lhs and rhs arrays along the concatenate_dimension.
  // E.g. if concatenate_dimension is 0, the "n1"/height dimension is
  // concatenated, so the arrays are stacked on top of each other.
  template <typename T>
  static std::unique_ptr<Array2D<T>> Concat2D(const Array2D<T>& lhs,
                                              const Array2D<T>& rhs,
                                              int concatenate_dimension) {
    CHECK(0 <= concatenate_dimension && concatenate_dimension < 2);
    auto result = MakeUnique<Array2D<T>>(
        concatenate_dimension == 0 ? lhs.n1() + rhs.n1() : lhs.n1(),
        concatenate_dimension == 1 ? lhs.n2() + rhs.n2() : lhs.n2());
    for (int64 i0 = 0; i0 < result->n1(); ++i0) {
      for (int64 i1 = 0; i1 < result->n2(); ++i1) {
        // If we exceed the bounds of the LHS, draw from the RHS, where the
        // result index is adjusted by the number of values present in the LHS.
        (*result)(i0, i1) = i0 < lhs.n1() && i1 < lhs.n2()
                                ? lhs(i0, i1)
                                : rhs(i0 >= lhs.n1() ? i0 - lhs.n1() : i0,
                                      i1 >= lhs.n2() ? i1 - lhs.n2() : i1);
      }
    }
    return result;
  }

  // Concatenates the lhs and rhs 3D arrays along the concatenate_dimension. lhs
  // and rhs must have the same dimensions except for the concatenate dimension.
  template <typename T>
  static std::unique_ptr<Array3D<T>> Concat3D(const Array3D<T>& lhs,
                                              const Array3D<T>& rhs,
                                              int concatenate_dimension) {
    CHECK(0 <= concatenate_dimension && concatenate_dimension < 3);
    std::vector<int64> lhs_dims = {lhs.n1(), lhs.n2(), lhs.n3()};
    std::vector<int64> rhs_dims = {rhs.n1(), rhs.n2(), rhs.n3()};
    std::vector<int64> out_dims = {rhs.n1(), rhs.n2(), rhs.n3()};
    for (int i = 0; i < 3; ++i) {
      if (i != concatenate_dimension) {
        out_dims[i] = lhs_dims[i];
        CHECK_EQ(lhs_dims[i], rhs_dims[i]);
      } else {
        out_dims[i] = lhs_dims[i] + rhs_dims[i];
      }
    }
    auto result = MakeUnique<Array3D<T>>(out_dims[0], out_dims[1], out_dims[2]);
    for (int64 i0 = 0; i0 < result->n1(); ++i0) {
      for (int64 i1 = 0; i1 < result->n2(); ++i1) {
        for (int64 i2 = 0; i2 < result->n3(); ++i2) {
          (*result)(i0, i1, i2) =
              i0 < lhs.n1() && i1 < lhs.n2() && i2 < lhs.n3()
                  ? lhs(i0, i1, i2)
                  : rhs(i0 >= lhs.n1() ? i0 - lhs.n1() : i0,
                        i1 >= lhs.n2() ? i1 - lhs.n2() : i1,
                        i2 >= lhs.n3() ? i2 - lhs.n3() : i2);
        }
      }
    }
    return result;
  }

  // Concatenates the lhs and rhs 4D arrays along the concatenate_dimension. lhs
  // and rhs must have the same dimensions except for the concatenate dimension.
  template <typename T>
  static std::unique_ptr<Array4D<T>> Concat4D(const Array4D<T>& lhs,
                                              const Array4D<T>& rhs,
                                              int concatenate_dimension) {
    CHECK(0 <= concatenate_dimension && concatenate_dimension < 4);
    std::vector<int64> lhs_dims = {lhs.n1(), lhs.n2(), lhs.n3(), lhs.n4()};
    std::vector<int64> rhs_dims = {rhs.n1(), rhs.n2(), rhs.n3(), rhs.n4()};
    std::vector<int64> out_dims = {rhs.n1(), rhs.n2(), rhs.n3(), rhs.n4()};
    for (int i = 0; i < 4; ++i) {
      if (i != concatenate_dimension) {
        out_dims[i] = lhs_dims[i];
        CHECK_EQ(lhs_dims[i], rhs_dims[i]);
      } else {
        out_dims[i] = lhs_dims[i] + rhs_dims[i];
      }
    }
    auto result = MakeUnique<Array4D<T>>(out_dims[0], out_dims[1], out_dims[2],
                                         out_dims[3]);
    for (int64 i0 = 0; i0 < result->n1(); ++i0) {
      for (int64 i1 = 0; i1 < result->n2(); ++i1) {
        for (int64 i2 = 0; i2 < result->n3(); ++i2) {
          for (int64 i3 = 0; i3 < result->n4(); ++i3) {
            (*result)(i0, i1, i2, i3) =
                i0 < lhs.n1() && i1 < lhs.n2() && i2 < lhs.n3() && i3 < lhs.n4()
                    ? lhs(i0, i1, i2, i3)
                    : rhs(i0 >= lhs.n1() ? i0 - lhs.n1() : i0,
                          i1 >= lhs.n2() ? i1 - lhs.n2() : i1,
                          i2 >= lhs.n3() ? i2 - lhs.n3() : i2,
                          i3 >= lhs.n4() ? i3 - lhs.n4() : i3);
          }
        }
      }
    }
    return result;
  }

  // Slices with modulo-wrapping.
  template <typename T>
  static std::vector<T> ModSlice1D(const tensorflow::gtl::ArraySlice<T>& input,
                                   int64 start, int64 size) {
    std::vector<T> result;
    for (int64 i = 0; i < size; ++i) {
      result.push_back(input[(start + i) % input.size()]);
    }
    return result;
  }

  // Slices the input array given starting indices in each dimension and limit
  // indices in each dimension.
  template <typename T>
  static std::unique_ptr<Array2D<T>> Slice2D(const Array2D<T>& input,
                                             std::array<int64, 2> starts,
                                             std::array<int64, 2> limits) {
    CHECK_LE(starts[0], input.n1());
    CHECK_LE(starts[1], input.n2());
    CHECK_LE(limits[0], input.n1());
    CHECK_LE(limits[1], input.n2());
    auto result =
        MakeUnique<Array2D<T>>(limits[0] - starts[0], limits[1] - starts[1]);
    for (int64 i0 = 0; i0 < result->n1(); ++i0) {
      for (int64 i1 = 0; i1 < result->n2(); ++i1) {
        (*result)(i0, i1) = input(starts[0] + i0, starts[1] + i1);
      }
    }
    return result;
  }

  template <typename T>
  static std::unique_ptr<Array3D<T>> Slice3D(const Array3D<T>& input,
     std::array<int64, 3> starts,
     std::array<int64, 3> limits) {
     CHECK_LE(starts[0], input.n1());
     CHECK_LE(starts[1], input.n2());
     CHECK_LE(starts[2], input.n3());
     CHECK_LE(limits[0], input.n1());
     CHECK_LE(limits[1], input.n2());
     CHECK_LE(limits[2], input.n3());
     auto result = MakeUnique<Array3D<T>>(
        limits[0] - starts[0], limits[1] - starts[1], limits[2] - starts[2]);
     for (int64 i0 = 0; i0 < result->n1(); ++i0) {
        for (int64 i1 = 0; i1 < result->n2(); ++i1) {
           for (int64 i2 = 0; i2 < result->n3(); ++i2) {
              (*result)(i0, i1, i2) =
                 input(starts[0] + i0, starts[1] + i1, starts[2] + i2);
           }
        }
     }
     return result;
  }

  template <typename T>
  static std::unique_ptr<Array4D<T>> Slice4D(const Array4D<T>& input,
                                             std::array<int64, 4> starts,
                                             std::array<int64, 4> limits) {
    CHECK_LE(starts[0], input.n1());
    CHECK_LE(starts[1], input.n2());
    CHECK_LE(starts[2], input.n3());
    CHECK_LE(starts[3], input.n4());
    CHECK_LE(limits[0], input.n1());
    CHECK_LE(limits[1], input.n2());
    CHECK_LE(limits[2], input.n3());
    CHECK_LE(limits[3], input.n4());
    auto result =
        MakeUnique<Array4D<T>>(limits[0] - starts[0], limits[1] - starts[1],
                               limits[2] - starts[2], limits[3] - starts[3]);
    for (int64 i0 = 0; i0 < result->n1(); ++i0) {
      for (int64 i1 = 0; i1 < result->n2(); ++i1) {
        for (int64 i2 = 0; i2 < result->n3(); ++i2) {
          for (int64 i3 = 0; i3 < result->n4(); ++i3) {
            (*result)(i0, i1, i2, i3) = input(starts[0] + i0, starts[1] + i1,
                                              starts[2] + i2, starts[3] + i3);
          }
        }
      }
    }
    return result;
  }

  // Applies map_function to each element in the input (2D array) and returns
  // the result.
  // (row, column) index of each element is also provided as arguments to
  // map_function.
  static std::unique_ptr<Array2D<float>> MapWithIndexArray2D(
      const Array2D<float>& matrix,
      const std::function<float(float, int64, int64)>& map_function);

  // Applies map_function to each element in the input (4D array) and returns
  // the result.
  template <typename F>
  static std::unique_ptr<Array4D<float>> MapArray4D(const Array4D<float>& input,
                                                    F&& map_function) {
    return MapWithIndexArray4D(input,
                               [&](float value, int64, int64, int64, int64) {
                                 return map_function(value);
                               });
  }

  // Applies map_function to each element in the input (4D array) and returns
  // the result.
  // (plane, depth, height, width) index of each element is also provided as
  // arguments to map_function.
  template <typename F>
  static std::unique_ptr<Array4D<float>> MapWithIndexArray4D(
      const Array4D<float>& input, F&& map_function) {
    auto result = MakeUnique<Array4D<float>>(input.planes(), input.depth(),
                                             input.height(), input.width());
    for (int64 plane = 0; plane < input.planes(); ++plane) {
      for (int64 depth = 0; depth < input.depth(); ++depth) {
        for (int64 height = 0; height < input.height(); ++height) {
          for (int64 width = 0; width < input.width(); ++width) {
            (*result)(plane, depth, height, width) =
                map_function(input(plane, depth, height, width), plane, depth,
                             height, width);
          }
        }
      }
    }
    return result;
  }

  // Applies map_function to each pair of elements in the input lhs and rhs
  // (4D array) and returns the result.
  template <typename F>
  static std::unique_ptr<Array4D<float>> MapArray4D(const Array4D<float>& lhs,
                                                    const Array4D<float>& rhs,
                                                    F&& map_function) {
    return MapWithIndexArray4D(
        lhs, rhs, [&](float lhs, float rhs, int64, int64, int64, int64) {
          return map_function(lhs, rhs);
        });
  }

  // Applies map_function to each pair of element in lhs and rhs (4D array) and
  // returns the result.
  // (plane, depth, height, width) index of each element is also provided as
  // arguments to map_function.
  template <typename F>
  static std::unique_ptr<Array4D<float>> MapWithIndexArray4D(
      const Array4D<float>& lhs, const Array4D<float>& rhs, F&& map_function) {
    auto result = MakeUnique<Array4D<float>>(lhs.planes(), lhs.depth(),
                                             lhs.height(), lhs.width());
    for (int64 plane = 0; plane < lhs.planes(); ++plane) {
      for (int64 depth = 0; depth < lhs.depth(); ++depth) {
        for (int64 height = 0; height < lhs.height(); ++height) {
          for (int64 width = 0; width < lhs.width(); ++width) {
            (*result)(plane, depth, height, width) = map_function(
                lhs(plane, depth, height, width),
                rhs(plane, depth, height, width), plane, depth, height, width);
          }
        }
      }
    }
    return result;
  }

  // Returns the result of a 2D pad on an input matrix.
  static std::unique_ptr<Array2D<float>> PadArray2D(
      const Array2D<float>& operand, const PaddingConfig& padding,
      const float pad);

  // Returns the result of a 3D pad on an input matrix.
  static Array3D<float> PadArray3D(const Array3D<float>& operand,
                                   const PaddingConfig& padding,
                                   const float pad);

  // Returns the result of a 4D pad on an input array.
  static Array4D<float> PadArray4D(const Array4D<float>& operand,
                                   const PaddingConfig& padding,
                                   const float pad);


  template <typename NativeT>
  static std::unique_ptr<Array4D<NativeT>> ReLu(const xla::Array4D<NativeT>& input)
  {
     auto great_zero = [](NativeT value) { return (value > 0.f) ? value : 0.f; };
     return ReferenceUtil::MapArray4D(input, great_zero);
  }


  // Log_SoftMax: logits - log(reduce_sum(exp(logits), dim))
  // Log_Sigmoid: y = log(1 / (1 + exp(-x))). For numerical stability, we use y = -tf.nn.softplus(-x).
  // l2_loss: output = sum(t ** 2) / 2
  // l2_normalize: output = x / sqrt(max(sum(x**2), epsilon))


  static std::unique_ptr<xla::Array4D<float>> Max_Pool(
     const xla::Array4D<float>& operand,
     const tensorflow::gtl::ArraySlice<tensorflow::int64>& window,
     const tensorflow::gtl::ArraySlice<tensorflow::int64>& stride, 
     xla::Padding padding_in);


  template <typename NativeT>
  static NativeT ReduceMean(const xla::Array4D<NativeT>& input)
  {
     NativeT result = xla::Sum<NativeT>(input.flatten());

     if (input.num_elements() > 0)
     {
        result /= (NativeT)input.num_elements();
     }
     return result;
  }

  template <typename NativeT>
  static NativeT ReduceMean(const xla::Array2D<NativeT>& input)
  {
     NativeT result = xla::Sum<NativeT>(input.flatten());

     if (input.num_elements() > 0)
     {
        result /= (NativeT)input.num_elements();
     }
     return result;
  }

  template <typename NativeT>
  static NativeT ReduceSum(const xla::Array4D<NativeT>& input)
  {
     return xla::Sum<NativeT>(input.flatten());
  }

  // more faster version of ConvArray4D
  template <typename TType>
  static std::unique_ptr<Array4D<TType>> Conv2D(
     const xla::Array4D<TType>& input,
     const xla::Array2D<TType>& kernel,
     const tensorflow::gtl::ArraySlice<int64>& stride,
     xla::Padding padding)
  {
     CHECK_GE(input.size(2), kernel.size(0));
     CHECK_GE(input.size(3), kernel.size(1));

     std::vector<int64> dim_lengths{ input.n3(), input.n4() };

     auto padding_both = xla::MakePadding(dim_lengths, { kernel.height(), kernel.width() }, stride, padding);

     std::vector<int64> window_counts(kernel.rank(), 0);
     std::vector<int64> pad_low(kernel.rank(), 0);

     for (int64 i = 0; i < kernel.rank(); ++i)
     {
        window_counts[i] = WindowCount(dim_lengths[i], kernel.size(i), stride[i], padding);
        pad_low[i] = padding_both[i].first;
     }
     auto result = MakeUnique<Array4D<TType>>(input.size(0), 1, window_counts[0], window_counts[1]);

     for (int64 i0 = 0; i0 < input.size(0); i0++)
     {
      for (int64 i2 = 0; i2 < result->size(2); ++i2)
      {
         for (int64 i3 = 0; i3 < result->size(3); ++i3)
         {
            TType mul_accum = 0.f;

            for (int64 i1 = 0; i1 < input.size(1); i1++)
            {
                 int64 i2_base = i2 * stride[0] - pad_low[0];
                 int64 i3_base = i3 * stride[1] - pad_low[1];

                 for (int64 y_win = 0; y_win < kernel.height(); ++y_win)
                 {
                    for (int64 x_win = 0; x_win < kernel.width(); ++x_win)
                    {
                       if (i2_base + y_win >= 0 && i3_base + x_win >= 0 &&
                          i2_base + y_win < input.n3() &&
                          i3_base + x_win < input.n4())
                       {                         
                          mul_accum += input(i0, i1, i2_base + y_win, i3_base + x_win) * kernel(y_win, x_win);
                       }
                    }
                 }
              }
              (*result)(i0, 0, i2, i3) = mul_accum;
           }
        }
     }

     return result;
  }

  template <typename TType>
  static void Bias_Add(xla::Array4D<TType>& input, const std::vector<TType>& bias)
  {
     CHECK_EQ(input.size(3), int64(bias.size()));

     for (int64 i0 = 0; i0 < input.size(0); i0++)
     {
        for (int64 i1 = 0; i1 < input.size(1); i1++)
        {
           for (int64 i2 = 0; i2 < input.size(2); i2++)
           {
              for (int64 i3 = 0; i3 < input.size(3); i3++)
              {
                 input(i0, i1, i2, i3) += bias[i3];
              }
           }
        }
     }
  }

  template <typename TType>
  static void SoftMax(xla::Array4D<TType>& input)
  {
     for (int64 i0 = 0; i0 < input.size(0); i0++)
     {
        for (int64 i1 = 0; i1 < input.size(1); i1++)
        {
           for (int64 i2 = 0; i2 < input.size(2); i2++)
           {
              TType ExpSumm = 0.f;
              for (int64 i3 = 0; i3 < input.size(3); i3++)
              {
                 TType e = std::exp(input(i0, i1, i2, i3));
                 input(i0, i1, i2, i3) = e;
                 ExpSumm += e;
              }

              for (int64 i3 = 0; i3 < input.size(3); i3++)
              {
                 input(i0, i1, i2, i3) = input(i0, i1, i2, i3) / ExpSumm;
              }
           }
        }
     }
  }


  template <typename TType>
  static std::unique_ptr<Array4D<TType>> SoftMax_Cross_Entropy_With_Logits(const xla::Array4D<TType>& input, const xla::Array4D<TType>& logics)
  {
     // https://stackoverflow.com/questions/34240703/difference-between-tensorflow-tf-nn-softmax-and-tf-nn-softmax-cross-entropy-with-logits

     xla::Array4D<TType> calc = input;

     xla::ReferenceUtil::SoftMax(calc);

     xla::Log(calc.flatten());

     // -tf.reduce_sum(y_true * tf.log(y_hat_softmax), 1)   // axis = y-axis
     calc * logics;

     auto result = xla::MakeUnique <xla::Array4D<TType> >(calc.size(0), calc.size(1), calc.size(2), 1);

     for (int64 i0 = 0; i0 < calc.size(0); i0++)
     {
        for (int64 i1 = 0; i1 < calc.size(1); i1++)
        {
           for (int64 i2 = 0; i2 < calc.size(2); i2++)
           {
              for (int64 i3 = 0; i3 < calc.size(3); i3++)
              {
                 (*result)(i0, i1, i2, 0) -= calc(i0, i1, i2, i3);
              }
           }
        }
     }
     return result;
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ReferenceUtil);
};


}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_REFERENCE_UTIL_H_
