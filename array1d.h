

#ifndef TENSORFLOW_COMPILER_XLA_ARRAY1D_H_
#define TENSORFLOW_COMPILER_XLA_ARRAY1D_H_

#include <vector>
#include <math.h>

#include "types.h"


namespace xla 
{
   template <typename TType>
   static void SoftPlus(std::vector<TType>& flatten)
   {
      for (size_t i = 0; i < flatten.size(); i++)
      {
         flatten[i] = std::log(TType(1.0f) + std::exp(flatten[i]));
      }
   }

   template <typename TType>
   static void SoftSign(std::vector<TType>& flatten)
   {
      for (size_t i = 0; i < flatten.size(); i++)
      {
         flatten[i] = flatten[i] / (fabs(flatten[i]) + 1.f);
      }
   }

   template <typename TType>
   static void Log(std::vector<TType>& flatten)
   {
      for (size_t i = 0; i < flatten.size(); i++)
      {
         flatten[i] = std::log(flatten[i]);
      }
   }

   template <typename TType>
   static void Sigmoid(std::vector<TType>& flatten)
   {
      for (size_t i = 0; i < flatten.size(); i++)
      {
         flatten[i] = TType(1.f) / (TType(1.f) + std::exp(-flatten[i]));
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

   // Exponential linear unit a(exp(features) - 1) if < 0, else feature
   template <typename TType>
   static void Elu(std::vector<TType>& flatten)
   {
      for (size_t i = 0; i < flatten.size(); i++)
      {
         if (flatten[i] <= 0.f)
         {
            flatten[i] = std::exp(flatten[i]) - TType(1.f);  // a(exp(x) - 1)
         }
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
