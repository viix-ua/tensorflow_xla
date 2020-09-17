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

#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_LIB_ARITHMETIC_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_LIB_ARITHMETIC_H_

#include <memory>

#include "xla_data.pb.h"

namespace xla {

   template <typename T>
   T Elu(T x)
   {
      const T alpha = 1.f;
      return (x < T(0.f)) ? (alpha * (std::exp(x) - T(1.f))) : x;
   }

   template <typename T>
   T Exponential(T x)
   {
      return std::exp(x);
   }

   template <typename T>
   T HardSigmoid(T x)
   {
      const T y = (x * T(0.2f)) + T(0.5f);

      if (y <= 0.f) {
         return 0.f;
      }
      else if (y >= T(1.f)) {
         return 1.f;
      }
      else {
         return x;
      }
   }

   template <typename T>
   T Linear(T x)
   {
      return x;
   }

   template <typename T>
   T Relu(T x)
   {
      return std::max(x, T(0.f));
   }

   template <typename T>
   T Selu(T x)
   {
      const T alpha = 1.67326324f;
      const T scale = 1.05070098f;

      return (x > 0.f) ? (scale * x) : (scale * alpha * (std::exp(x) - T(1.f)));
   }

   template <typename T>
   T Sigmoid(T x)
   {
      return T(1.f) / (T(1.f) + std::exp(-x));
   }

   template <typename T>
   T SigmoidSign(T x)
   {
      if (x >= 0) {
         return Sigmoid(x);
      }
      else {
         const T z = std::exp(x);
         return z / (T(1.f) + z);
      }
   }

   template <typename T>
   T SoftPlus(T x)
   {
      return std::log(T(1.f) + std::exp(x));
   }

   template <typename T>
   T SoftSign(T x)
   {
      return x / (std::abs(x) + T(1.f));
   }

   template <typename T>
   T Swish(T x)
   {
      return x * Sigmoid(x);
   }

   template <typename T>
   T Tanh(T x)
   {
      //return (std::exp(x) - std::exp(-x)) / (std::exp(x) + std::exp(-x));
      return std::tanh(x);
   }

   template <typename T>
   T Square(T x)
   {
      return x * x;
   }

}  // xla

namespace xla {

class Computation;
class ComputationBuilder;

// Creates a scalar add computation and returns it.
Computation CreateScalarAddComputation(PrimitiveType type,
                                       ComputationBuilder* builder);

// Creates a scalar ge computation and returns it.
Computation CreateScalarGeComputation(PrimitiveType type,
                                      ComputationBuilder* builder);

// Creates a scalar max computation and returns it.
Computation CreateScalarMaxComputation(PrimitiveType type,
                                       ComputationBuilder* builder);

// Creates a scalar min computation and returns it.
Computation CreateScalarMinComputation(PrimitiveType type,
                                       ComputationBuilder* builder);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_LIB_ARITHMETIC_H_
