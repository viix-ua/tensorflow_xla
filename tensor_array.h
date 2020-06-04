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

class TensorArray
{
   TensorArray();

   explicit TensorArray(const int64 n1);

   std::vector<int64> dimensions_;

public:

   TensorArray(const int64 n1, const int64 n2)
   {
      dimensions_.push_back(n1);
      dimensions_.push_back(n2);
   }

   TensorArray(const int64 n1, const int64 n2, const int64 n3)
   {
      dimensions_.push_back(n1);
      dimensions_.push_back(n2);
      dimensions_.push_back(n3);
   }

   TensorArray(const int64 n1, const int64 n2, const int64 n3, const int64 n4)
   {
      dimensions_.push_back(n1);
      dimensions_.push_back(n2);
      dimensions_.push_back(n3);
      dimensions_.push_back(n4);
   }

   inline
   int64 rank() const
   {
      return dimensions_.size();
   }

   // return size of specified dimension
   inline
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

   inline
   const std::vector<int64>& dimensions() const
   {
      return dimensions_;
   }

   inline
   int64 Height() const
   {
      return size(rank() - 2);
   }

   inline
   int64 Width() const
   {
      return size(rank() - 1);
   }

   inline
   int64 Depth() const
   {
      return size(rank() - 3);
   }

   inline
   int64 Batch() const
   {
      return size(rank() - 4);
   }
};

}  // namespace xla

#endif
