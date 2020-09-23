#pragma once

#include <memory>
#include <list>
#include <string>

#include "ptr_util.h"
#include "array4d.h"
#include "arithmetic.h"

namespace xla
{
/* functions:
https://www.tensorflow.org/api_docs/python/tf/nn/
*/

enum ActivationType {
   eElu,
   eExponential,
   eHardSigmoid,
   eLinear,
   eRelu,
   eSelu,
   eSigmoid,
   eSigmoidSign,
   eSoftPlux,
   eSoftSign,
   eSwish,
   eTanh
};


template <typename T>
class Activation
{
public:

   explicit Activation(ActivationType activation)
      : mType(activation)
      , mValue(T(0))
   {}

   T operator ()(T x)
   {
      static std::function<T(T)> fn[] = {
         xla::Elu<T>,
         xla::Exponential<T>,
         xla::HardSigmoid<T>,
         xla::Linear<T>,
         xla::Relu<T>,
         xla::Selu<T>,
         xla::Sigmoid<T>,
         xla::SigmoidSign<T>,
         xla::SoftPlus<T>,
         xla::SoftSign<T>,
         xla::Swish<T>,
         xla::Tanh<T>
      };

      mValue += x;
      return fn[mType](x);
   }

   operator T() const
   {
      return mValue;
   }

private:

   ActivationType mType;
   T mValue = 0.f;
};

}  // xla

namespace xla
{
   class LayerBase
   {
   public:

   };

   class InputLayer : public LayerBase
   {
   public:
      explicit InputLayer(const xla::Array4D<float>& input){}
   };

   class ConvLayer : public LayerBase
   {
   public:
      explicit ConvLayer(std::string str) {};
   };

   class MaxPoolLayer : public LayerBase
   {
   public:
      explicit MaxPoolLayer(std::string str) {};
   };

   class FullyConnLayer : public LayerBase
   {
   public:
      explicit FullyConnLayer(std::string str) {};
   };

   class SoftMaxLayer : public LayerBase
   {
   public:
      explicit SoftMaxLayer(std::string str) {};
   };

   class ReLuLayer : public LayerBase
   {
   public:
      explicit ReLuLayer(std::string str){}
   };

   class NNet
   {
   public:
      LayerBase* AddLayer(LayerBase* linked_layer, LayerBase* layer);
      LayerBase* AddLayer(const xla::Array4D<float>* input);

   public:
      std::list<std::unique_ptr<LayerBase>> mLayers;

   };

   inline
   LayerBase* NNet::AddLayer(LayerBase* linked_layer, LayerBase* layer)
   {
      mLayers.push_back(std::unique_ptr<LayerBase>(layer));
      return layer;
   }

   inline
   LayerBase* NNet::AddLayer(const xla::Array4D<float>* input)
   {
      mLayers.push_back(std::unique_ptr<LayerBase>(new InputLayer(*input)));
      return mLayers.back().get();
   }
}
