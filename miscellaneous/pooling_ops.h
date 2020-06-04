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

// XLA specific pooling ops.

#ifndef TENSORFLOW_COMPILER_TF2XLA_KERNELS_POOLING_OPS_H_
#define TENSORFLOW_COMPILER_TF2XLA_KERNELS_POOLING_OPS_H_

/*
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/conv_grad_ops.h"
#include "tensorflow/core/kernels/pooling_ops_common.h"
*/

#include "array4d.h"
#include "padding.h"
#include "test_helpers.h"

namespace tensorflow {
namespace {

// Superclass of pooling ops.
class PoolingOp // : public XlaOpKernel 
{
 public:

  PoolingOp(const std::vector<int32>& ksize,
     const std::vector<int32>& stride, xla::Padding padding,
     const xla::Array4D<double>& tensor_in_shape)
  {
    ASSERT_TRUE(ksize.size() == 4);
    ASSERT_TRUE(stride.size() == 4);

    for (int i = 0; i < 4; ++i) 
    {
      ksize_.push_back(ksize[i]);
      stride_.push_back(stride[i]);
    }
    padding_ = padding;

    //OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding));
    //padding_ = (padding == VALID) ? xla::Padding::kValid : xla::Padding::kSame;
  }

  void Compile() 
  {
    //xla::ComputationDataHandle input = ctx->Input(0);
    //const TensorShape input_shape = ctx->InputShape(0);

    //const DataType type = input_type(0);
    //xla::ComputationDataHandle pooled = ctx->builder()->ReduceWindow(
    //    input, InitValue(ctx->builder(), type), *Reduction(ctx, type), ksize_,
    //    stride_, padding_);
    //ctx->SetOutput(0, PostProcessOutput(ctx, pooled, type, input_shape));
  }

 protected:
  std::vector<int64> ksize_;
  std::vector<int64> stride_;
  xla::Padding padding_;
};


}  // anonymous namespace
}  // namespace tensorflow

#endif
