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

#ifndef TENSORFLOW_COMPILER_XLA_ARRAY3D_H_
#define TENSORFLOW_COMPILER_XLA_ARRAY3D_H_

#include "types.h"
#include "str_util.h"
#include "strcat.h"
#include "logging.h"
#include "macros.h"

#include "tensor_array.h"
#include "stringprintf.h"
#include "ptr_util.h"

namespace xla {

// Simple 3D array structure.
//
// The data layout in major-to-minor order is: n1, n2, n3.
template <typename T>
class Array3D : public TensorArray<T>
{
 public:

   using TensorArray<T>::values_;
   using TensorArray<T>::num_elements;


  // Creates an array of dimensions n1 x n2 x n3, uninitialized values.
  Array3D(const int64 n1, const int64 n2, const int64 n3)
     : TensorArray<T>({ n1, n2, n3 }, (n1 * n2 * n3))
     , n1_(n1), n2_(n2), n3_(n3)
  {}

  // Creates an array of dimensions n1 x n2 x n3, initialized to value.
  Array3D(const int64 n1, const int64 n2, const int64 n3, const T value)
     : TensorArray<T>({ n1, n2, n3 }, (n1 * n2 * n3), value)
     , n1_(n1), n2_(n2), n3_(n3)
  {}

  Array3D(const int64 n1, const int64 n2, const int64 n3, const std::vector<T>& input_array)
     : TensorArray<T>({ n1, n2, n3 }, input_array)
     , n1_(n1), n2_(n2), n3_(n3)
  {
     CHECK_EQ(n1 * n2 * n3, input_array.size());
  }

  // Creates an array from the given nested initializer list. The outer
  // initializer list is the first dimension, and so on.
  //
  // For example {{{1, 2}, {3, 4}, {5, 6}, {7, 8}},
  //              {{9, 10}, {11, 12}, {13, 14}, {15, 16}},
  //              {{17, 18}, {19, 20}, {21, 22}, {23, 24}}}
  // results in an array with n1=3, n2=4, n3=2.
  Array3D(std::initializer_list<std::initializer_list<std::initializer_list<T>>>
              values)
      : Array3D(values.size(), values.begin()->size(),
                values.begin()->begin()->size()) {
    int64 n1 = 0;
    for (auto n1_it = values.begin(); n1_it != values.end(); ++n1_it, ++n1) {
      int64 n2 = 0;
      for (auto n2_it = n1_it->begin(); n2_it != n1_it->end(); ++n2_it, ++n2) {
        int64 n3 = 0;
        for (auto n3_it = n2_it->begin(); n3_it != n2_it->end();
             ++n3_it, ++n3) {
          (*this)(n1, n2, n3) = *n3_it;
        }
      }
    }
  }

  T& operator()(const int64 n1, const int64 n2, const int64 n3) {
    CHECK_LT(n1, n1_);
    CHECK_LT(n2, n2_);
    CHECK_LT(n3, n3_);
    return values_[n1 * n2_ * n3_ + n2 * n3_ + n3];
  }

  const T& operator()(const int64 n1, const int64 n2, const int64 n3) const {
    CHECK_LT(n1, n1_);
    CHECK_LT(n2, n2_);
    CHECK_LT(n3, n3_);
    return values_[n1 * n2_ * n3_ + n2 * n3_ + n3];
  }

  bool operator == (const Array3D<T>& rhs) const
  {
     bool result = (n1() == rhs.n1()) && (n2() == rhs.n2()) && (n3() == rhs.n3());

     if (result)
     {
        CHECK_EQ(num_elements(), rhs.num_elements());
        for (size_t i = 0; i < values_.size() && result; i++)
        {
           result = (std::abs(values_[i] - rhs.values_[i]) < 0.000001f);
        }
     }
     return result;
  }

  // Access to the array's dimensions.
  int64 n1() const { return n1_; }
  int64 n2() const { return n2_; }
  int64 n3() const { return n3_; }

  const T* data() const { return const_cast<Array3D*>(this)->values_.data(); }

  string ToString() const 
  {
     std::vector<string> pieces = {
        tensorflow::strings::Printf("z=%lld,y=%lld,x=%lld {\n", 
        n1(), n2(), n3()) };


        for (int64 depth = 0; depth < n1(); ++depth) {
           pieces.push_back("    {\n");
           for (int64 height = 0; height < n2(); ++height) {
              pieces.push_back("      {");
              for (int64 width = 0; width < n3(); ++width) {
                 pieces.push_back(tensorflow::strings::StrCat(
                    (*this)(depth, height, width), ", "));
              }
              pieces.push_back("},\n");
           }
           pieces.push_back("    },\n");
        }
        pieces.push_back("  },\n");

     return tensorflow::str_util::Join(pieces, "");
  }

 private:
  int64 n1_;   // depth
  int64 n2_;   // height
  int64 n3_;   // width
};


template <typename T>
void MatrixMul(const xla::Array3D<T>& lhs, const xla::Array3D<T>& rhs, xla::Array3D<T>& result)
{
   // multiply lsh(d, p, r) * rhs(d, r, q) = result(d, p, q)

   CHECK_EQ(lhs.Width(), rhs.Height());
   CHECK_EQ(lhs.Height(), result.Height());
   CHECK_EQ(rhs.Width(), result.Width());

   CHECK_EQ(lhs.Depth(), rhs.Depth());
   CHECK_EQ(rhs.Depth(), result.Depth());

   assert(lhs.Width() == rhs.Height());
   assert(lhs.Height() == result.Height());
   assert(rhs.Width() == result.Width());

   int64 i = 0;
   int64 j = 0;
   int64 r = 0;

   for (int64 d = 0; d < result.Depth(); d++)
   {
      for (i = 0; i < lhs.Height(); i++)
      {
         for (j = 0; j < rhs.Width(); j++)
         {
            result(d, i, j) = 0.f;
            for (r = 0; r < lhs.Width(); r++)
            {
               result(d, i, j) += lhs(d, i, r) * rhs(d, r, j);
            }
         }
      }
   }

}

template <typename T>
std::unique_ptr<xla::Array3D<T>> MakeMatrixMul(const xla::Array3D<T>& lhs, const xla::Array3D<T>& rhs)
{
   assert(lhs.n1() == rhs.n1());

   std::unique_ptr<xla::Array3D<T>> result = xla::MakeUnique<xla::Array3D<T>>(lhs.n1(), lhs.n2(), rhs.n3());
   xla::MatrixMul(lhs, rhs, *result);

   return result;
}


}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_ARRAY3D_H_
