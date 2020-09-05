

#ifndef TENSORFLOW_COMPILER_XLA_ARRAY1D_H_
#define TENSORFLOW_COMPILER_XLA_ARRAY1D_H_

#include <vector>
#include <math.h>

#include "types.h"


namespace xla
{
   template <typename TType>
   static void Log(std::vector<TType>& flatten)
   {
      for (size_t i = 0; i < flatten.size(); i++)
      {
         flatten[i] = std::log(flatten[i]);
      }
   }

   template <typename TType>
   static void Square(std::vector<TType>& flatten)
   {
      for (size_t i = 0; i < flatten.size(); i++)
      {
         flatten[i] = flatten[i] * flatten[i];
      }
   }

   template <typename TType>
   static TType Sum(const std::vector<TType>& flatten)
   {
      TType accumulator = TType(0);

      for (size_t i = 0; i < flatten.size(); i++)
      {
         accumulator += flatten[i];
      }
      return accumulator;
   }
}  // ns

#endif
