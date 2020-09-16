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

#ifndef _TENSOR_ARRAY_H_
#define _TENSOR_ARRAY_H_

#include <initializer_list>
#include <vector>

#include "integral_types.h"
#include "types.h"

namespace xla {

// General N dimensional array class with arbitrary value type.
template <typename T>
class TensorArray
{
   TensorArray();

   explicit TensorArray(const int64 n1);

   std::vector<int64> dimensions_;

public:

   // Type inference can have a hard time parsing very deep initializer list
   // nests, especially if one or more dimensions is one as the compiler just
   // sees a single-element integer initializer. These typedefs allow casting
   // explicitly with less typing.
   using InitializerList1D = std::initializer_list<T>;
   using InitializerList2D = std::initializer_list<InitializerList1D>;
   using InitializerList3D = std::initializer_list<InitializerList2D>;
   using InitializerList4D = std::initializer_list<InitializerList3D>;

   using value_type = T;

   explicit TensorArray(const std::vector<int64>& sizes)
      : dimensions_(sizes.begin(), sizes.end())
   {
   }

   explicit TensorArray(const std::vector<int64>& sizes, T value)
      : dimensions_(sizes.begin(), sizes.end())
   {
   }

   explicit TensorArray(const std::vector<int64>& sizes, const std::vector<T>& input_array)
      : dimensions_(sizes.begin(), sizes.end())
   {
   }

   // return num_dimension
   int64 rank() const
   {
      return dimensions_.size();
   }

   // return size of specified dimension
   int64 size(int64 dim) const
   {
      if (dim < int64(dimensions_.size()) && (dim >= 0))
      {
         return dimensions_[dim];
      }
      else
      {
         return 0;
      }
   }

   const std::vector<int64>& dimensions() const
   {
      return dimensions_;
   }

   int64 Height() const
   {
      return size(rank() - 2);
   }

   int64 Width() const
   {
      return size(rank() - 1);
   }

   int64 Depth() const
   {
      return size(rank() - 3);
   }

   int64 Batch() const
   {
      return size(rank() - 4);
   }
};

}  // namespace xla

#endif
