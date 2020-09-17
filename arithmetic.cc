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

#include "arithmetic.h"

#include <string>

#include "computation.h"
#include "computation_builder.h"
#include "shape_util.h"
#include "types.h"
#include "xla_data.pb.h"

namespace xla {

Computation CreateScalarAddComputation(PrimitiveType type,
                                       ComputationBuilder* builder) {
  const Shape scalar = ShapeUtil::MakeShape(type, {});
  auto b = builder->CreateSubBuilder("add_" + ShapeUtil::HumanString(scalar));
  auto lhs = b->Parameter(0, scalar, "lhs");
  auto rhs = b->Parameter(1, scalar, "rhs");
  b->Add(lhs, rhs);
  return b->BuildAndNoteError();
}

Computation CreateScalarGeComputation(PrimitiveType type,
                                      ComputationBuilder* builder) {
  const Shape scalar = ShapeUtil::MakeShape(type, {});
  auto b = builder->CreateSubBuilder("ge_" + ShapeUtil::HumanString(scalar));
  auto lhs = b->Parameter(0, scalar, "lhs");
  auto rhs = b->Parameter(1, scalar, "rhs");
  b->Ge(lhs, rhs);
  return b->BuildAndNoteError();
}

Computation CreateScalarMaxComputation(PrimitiveType type,
                                       ComputationBuilder* builder) {
  const Shape scalar = ShapeUtil::MakeShape(type, {});
  auto b = builder->CreateSubBuilder("max_" + ShapeUtil::HumanString(scalar));
  auto lhs = b->Parameter(0, scalar, "lhs");
  auto rhs = b->Parameter(1, scalar, "rhs");
  b->Max(lhs, rhs);
  return b->BuildAndNoteError();
}

Computation CreateScalarMinComputation(PrimitiveType type,
                                       ComputationBuilder* builder) {
  const Shape scalar = ShapeUtil::MakeShape(type, {});
  auto b = builder->CreateSubBuilder("min_" + ShapeUtil::HumanString(scalar));
  auto lhs = b->Parameter(0, scalar, "lhs");
  auto rhs = b->Parameter(1, scalar, "rhs");
  b->Min(lhs, rhs);
  return b->BuildAndNoteError();
}

}  // namespace xla
