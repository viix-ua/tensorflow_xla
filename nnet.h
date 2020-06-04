#pragma once

#include <memory>
#include <list>
#include <string>

#include "ptr_util.h"
#include "array4d.h"

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
