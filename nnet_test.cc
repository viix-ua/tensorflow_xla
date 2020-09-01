
#include "nnet.h"

#include "default_logging.h"

void run()
{
   using namespace xla;

   NNet nnet;

   auto img_224x224 = MakeUnique<xla::Array4D<float>>(1, 3, 224, 224);

   LayerBase* input = nnet.AddLayer(img_224x224.get());

   LayerBase* conv1_1 = nnet.AddLayer(input,    new ConvLayer("conv1_1"));
   LayerBase* conv1_2 = nnet.AddLayer(conv1_1,  new ConvLayer("conv1_2"));
   LayerBase* pool1   = nnet.AddLayer(conv1_2,  new MaxPoolLayer("pool1"));
   
   LayerBase* conv2_1 = nnet.AddLayer(pool1,    new ConvLayer("conv2_1"));
   LayerBase* conv2_2 = nnet.AddLayer(conv2_1,  new ConvLayer("conv2_2"));
   LayerBase* pool2   = nnet.AddLayer(conv2_2,  new MaxPoolLayer("pool2"));

   LayerBase* conv3_1 = nnet.AddLayer(pool2,    new ConvLayer("conv3_1"));
   LayerBase* conv3_2 = nnet.AddLayer(conv3_1,  new ConvLayer("conv3_2"));
   LayerBase* conv3_3 = nnet.AddLayer(conv3_2,  new ConvLayer("conv3_3"));
   LayerBase* pool3   = nnet.AddLayer(conv3_3,  new MaxPoolLayer("pool3"));

   LayerBase* conv4_1 = nnet.AddLayer(pool3,    new ConvLayer("conv4_1"));
   LayerBase* conv4_2 = nnet.AddLayer(conv4_1,  new ConvLayer("conv4_2"));
   LayerBase* conv4_3 = nnet.AddLayer(conv4_2,  new ConvLayer("conv4_3"));
   LayerBase* pool4   = nnet.AddLayer(conv4_3,  new MaxPoolLayer("pool4"));

   LayerBase* conv5_1 = nnet.AddLayer(pool4,    new ConvLayer("conv5_1"));
   LayerBase* conv5_2 = nnet.AddLayer(conv5_1,  new ConvLayer("conv5_2"));
   LayerBase* conv5_3 = nnet.AddLayer(conv5_2,  new ConvLayer("conv5_3"));
   LayerBase* pool5   = nnet.AddLayer(conv5_3,  new MaxPoolLayer("pool5"));

   LayerBase* fc6     = nnet.AddLayer(pool5,    new FullyConnLayer("fc6"));
   
   LayerBase* relu6   = nnet.AddLayer(fc6,      new ReLuLayer("relu6"));

   LayerBase* fc7     = nnet.AddLayer(relu6,    new FullyConnLayer("fc7"));
   LayerBase* relu7   = nnet.AddLayer(fc7,      new ReLuLayer("relu7"));

   LayerBase* fc8     = nnet.AddLayer(relu7,    new FullyConnLayer("fc8"));

   LayerBase* prob    = nnet.AddLayer(fc8,      new SoftMaxLayer("prob"));

   LOG_MSG("", prob);
}
