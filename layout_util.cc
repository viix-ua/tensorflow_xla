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

#include "layout_util.h"

#include <stddef.h>
#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "layout_util_flags.h"
//#include "tensorflow/compiler/xla/protobuf_util.h"
#include "shape_util.h"
//#include "tensorflow/compiler/xla/status_macros.h"
#include "types.h"
#include "util.h"
#include "errors.h"


//#include "tensorflow/compiler/xla/legacy_flags/layout_util_flags.h"
//#include "tensorflow/compiler/xla/protobuf_util.h"
//#include "tensorflow/compiler/xla/shape_util.h"
//#include "tensorflow/compiler/xla/status_macros.h"
//#include "tensorflow/compiler/xla/types.h"
//#include "tensorflow/compiler/xla/util.h"
//#include "tensorflow/core/lib/core/errors.h"
#include "numbers.h"
#include "str_util.h"
#include "strcat.h"
#include "logging.h"
//#include "tensorflow/core/platform/protobuf.h"

using namespace tensorflow::errors;

namespace xla {
namespace {


using DimensionOrder = legacy_flags::DefaultLayout::DimensionOrder;

// Internal helper for GetDefaultLayoutForShape and SetToDefaultLayout. Sets
// minor_to_major to the value that represents the default layout.
void SetDefaultLayoutToContainer(
    tensorflow::protobuf::RepeatedField<tensorflow::protobuf_int64>*
        minor_to_major) {
  const int size = minor_to_major->size();
  legacy_flags::LayoutUtilFlags* flags = legacy_flags::GetLayoutUtilFlags();
  auto default_layout = flags->xla_default_layout;
  switch (default_layout.dimension_order) {
    case DimensionOrder::kMajorToMinor:
      for (int i = 0; i < size; ++i) {
        minor_to_major->Set(i, size - 1 - i);
      }
      break;
    case DimensionOrder::kMinorToMajor:
      for (int i = 0; i < size; ++i) {
        minor_to_major->Set(i, i);
      }
      break;
    case DimensionOrder::kRandom:
      for (int i = 0; i < size; ++i) {
        minor_to_major->Set(i, i);
      }
      std::shuffle(
          minor_to_major->begin(), minor_to_major->end(),
          std::mt19937(default_layout.seed != 0 ? static_cast<unsigned int>(default_layout.seed)
                                                : std::random_device()()));
  }
}

}  // namespace


Layout LayoutUtil::MakeLayout(
    tensorflow::gtl::ArraySlice<int64> minor_to_major) {
  Layout layout;
  for (int64 dimension_number : minor_to_major) {
    layout.add_minor_to_major(dimension_number);
  }
  return layout;
}

namespace {

// Internal helper that creates a default layout for an array of the given rank.
Layout CreateDefaultLayoutForRank(int rank) {
  Layout layout;
  tensorflow::protobuf::RepeatedField<tensorflow::protobuf_int64>*
      minor_to_major = layout.mutable_minor_to_major();
  minor_to_major->Resize(rank, 0);
  SetDefaultLayoutToContainer(minor_to_major);
  return layout;
}

}  // namespace

Layout LayoutUtil::GetDefaultLayoutForShape(const Shape& shape) {
  // A Layout proto corresponds to a single array, not a tuple.
  DCHECK(!ShapeUtil::IsTuple(shape));
  return CreateDefaultLayoutForRank(shape.dimensions_size());
}

Layout LayoutUtil::GetDefaultLayoutForR2() {
  return CreateDefaultLayoutForRank(2);
}

Layout LayoutUtil::GetDefaultLayoutForR3() {
  return CreateDefaultLayoutForRank(3);
}

Layout LayoutUtil::GetDefaultLayoutForR4() {
  return CreateDefaultLayoutForRank(4);
}

void LayoutUtil::SetToDefaultLayout(Shape* shape) {
  if (ShapeUtil::IsTuple(*shape)) {
    // Tuple shape.
    for (auto& element_shape : *shape->mutable_tuple_shapes()) {
      SetToDefaultLayout(&element_shape);
    }
  } else {
    tensorflow::protobuf::RepeatedField<tensorflow::protobuf_int64>*
        minor_to_major = shape->mutable_layout()->mutable_minor_to_major();
    minor_to_major->Resize(shape->dimensions_size(), 0);
    SetDefaultLayoutToContainer(minor_to_major);
  }
}

void LayoutUtil::SetToDefaultLayout(ProgramShape* program_shape) {
  for (auto& parameter_shape : *program_shape->mutable_parameters()) {
    LayoutUtil::SetToDefaultLayout(&parameter_shape);
  }
  LayoutUtil::SetToDefaultLayout(program_shape->mutable_result());
}


tensorflow::Status LayoutUtil::ValidateLayoutInShape(
    const Shape& shape) {
  if (ShapeUtil::IsTuple(shape)) {
    // Tuple shape.
    if (shape.has_layout()) {
      return InvalidArgument("tuple should not have a layout field");
    }
    for (auto& element_shape : shape.tuple_shapes()) {
      TF_RETURN_IF_ERROR(ValidateLayoutInShape(element_shape));
    }
    return tensorflow::Status::OK();
  } else if (ShapeUtil::Rank(shape) == 0 && !shape.has_layout()) {
    // A scalar without a layout is ok.
    return tensorflow::Status::OK();
  } else {
    // Array shape.
    if (!shape.has_layout()) {
      return InvalidArgument("shape does not have a layout");
    }
    return ValidateLayoutForShape(shape.layout(), shape);
  }
}

tensorflow::Status LayoutUtil::ValidateLayoutForShape(
    const Layout& layout, const Shape& shape) {
  if (ShapeUtil::IsTuple(shape)) {
    return InvalidArgument("a single Layout is not valid for tuple shapes");
  }

  if (layout.minor_to_major_size() != ShapeUtil::Rank(shape)) {
     return InvalidArgument(
        "layout minor_to_major field contains %d elements, "
        "but shape is rank %lld");// : {%s}; shape: %s",
        //layout.minor_to_major_size(), ShapeUtil::Rank(shape),
        //tensorflow::str_util::Join(layout.minor_to_major(), ", ").c_str(),
        //shape.ShortDebugString().c_str());
  }

  std::vector<bool> dimensions_in_layout(ShapeUtil::Rank(shape), false);
  for (int i = 0; i < ShapeUtil::Rank(shape); ++i) {
    int64 dim = layout.minor_to_major(i);
    if (dim < 0 || dim >= ShapeUtil::Rank(shape)) {
      return InvalidArgument(
          "layout minor_to_major field has out-of-bounds value");
    }
    if (dimensions_in_layout[dim]) {
      return InvalidArgument(
          "layout minor_to_major field has duplicate values");
    }
    dimensions_in_layout[dim] = true;
  }

  if (layout.padded_dimensions_size() > 0)
  {
    if (layout.padded_dimensions_size() != ShapeUtil::Rank(shape)) {
      return InvalidArgument(
          "layout has %d padded dimensions, but shape is rank %lld",
          layout.padded_dimensions_size(), ShapeUtil::Rank(shape));
    }
    for (int i = 0; i < layout.padded_dimensions_size(); ++i) {
      if (layout.padded_dimensions(i) < shape.dimensions(i)) {
        return InvalidArgument(
            "for dimension %d, dimension padding (%lld) is smaller than "
            "the dimension size (%lld) of the shape",
            i, layout.padded_dimensions(i), shape.dimensions(i));
      }
    }
  }
  return tensorflow::Status::OK();
}


void LayoutUtil::ClearLayout(Shape* shape) {
  shape->clear_layout();
  for (auto& element_shape : *shape->mutable_tuple_shapes()) {
    ClearLayout(&element_shape);
  }
}

void LayoutUtil::ClearLayout(ProgramShape* program_shape) {
  for (auto& parameter_shape : *program_shape->mutable_parameters()) {
    LayoutUtil::ClearLayout(&parameter_shape);
  }
  LayoutUtil::ClearLayout(program_shape->mutable_result());
}

bool LayoutUtil::IsMonotonicWithDim0Minor(const Layout& layout) {
  return std::is_sorted(layout.minor_to_major().begin(),
                        layout.minor_to_major().end());
}

bool LayoutUtil::IsMonotonicWithDim0Major(const Layout& layout) {
  return std::is_sorted(layout.minor_to_major().begin(),
                        layout.minor_to_major().end(), std::greater<int64>());
}

bool LayoutUtil::IsPadded(const Shape& shape) {
  if (ShapeUtil::IsTuple(shape) || !HasLayout(shape) ||
      shape.layout().padded_dimensions_size() == 0) {
    return false;
  }
  CHECK_EQ(shape.dimensions_size(), shape.layout().padded_dimensions_size());
  for (int i = 0; i < shape.dimensions_size(); ++i) {
    if (shape.layout().padded_dimensions(i) > shape.dimensions(i)) {
      return true;
    }
  }
  return false;
}

bool LayoutUtil::HasLayout(const Shape& shape) {
  if (ShapeUtil::IsTuple(shape)) {
    // Tuple shape: all subshapes must have a layout.
    return std::all_of(shape.tuple_shapes().begin(), shape.tuple_shapes().end(),
                       [](const Shape& s) { return HasLayout(s); });
  }
  // A scalar trivially always has a layout.
  return (ShapeUtil::Rank(shape) == 0 ||
          (shape.has_layout() && (shape.layout().minor_to_major_size() > 0)));
}

bool LayoutUtil::HasLayout(const ProgramShape& program_shape) {
  for (auto& parameter_shape : program_shape.parameters()) {
    if (!LayoutUtil::HasLayout(parameter_shape)) {
      return false;
    }
  }
  return LayoutUtil::HasLayout(program_shape.result());
}

bool LayoutUtil::Equal(const Layout& lhs, const Layout& rhs) 
{
   // TODO:
   // all Layouts inherited from tensorflow::protobuf::Message
   //return protobuf_util::ProtobufEquals(lhs, rhs); // include protobuf_util.h/.cc
   return false;  // temporary
}

int64 LayoutUtil::Major(const Layout& layout, int physical_dimension_number)
{
  CHECK_LE(0, physical_dimension_number);
  CHECK_LT(physical_dimension_number, layout.minor_to_major_size());
  return Minor(layout,
               layout.minor_to_major_size() - 1 - physical_dimension_number);
}

int64 LayoutUtil::Minor(const Layout& layout, int physical_dimension_number)
{
  CHECK_LE(0, physical_dimension_number);
  CHECK_LT(physical_dimension_number, layout.minor_to_major_size());
  return layout.minor_to_major(physical_dimension_number);
}
/*
std::vector<int64> LayoutUtil::MakeLogicalToPhysical(
    const Layout& layout) {
  std::vector<int64> logical_to_physical(layout.minor_to_major_size());
  for (int64 physical = 0; physical < logical_to_physical.size(); ++physical) {
    const int64 logical = Major(layout, physical);
    logical_to_physical[logical] = physical;
  }
  return logical_to_physical;
}
*/
string LayoutUtil::HumanString(const Layout& layout) {
  return tensorflow::strings::StrCat(
      "{", tensorflow::str_util::Join(layout.minor_to_major(), ","), "}");
}
// TODO:

namespace {

// Internal helper for recursively copying layouts.
tensorflow::Status CopyLayoutInternal(const Shape& src, Shape* dst)
{
  if (ShapeUtil::IsTuple(src) != ShapeUtil::IsTuple(*dst)) {
    return InvalidArgument(
        "cannot copy layout from shape: shape structure differs");
  }
  if (ShapeUtil::IsTuple(src))
  {
    if (ShapeUtil::TupleElementCount(src) != ShapeUtil::TupleElementCount(*dst))
    {
      return InvalidArgument(
          "cannot copy layout from shape: tuple element count differs");
    }
    for (int i = 0; i < ShapeUtil::TupleElementCount(src); ++i)
    {
      TF_RETURN_IF_ERROR(CopyLayoutInternal(src.tuple_shapes(i),
                                            dst->mutable_tuple_shapes(i)));
    }
  } else {
    if (src.has_layout()) {
      if (ShapeUtil::Rank(src) != ShapeUtil::Rank(*dst)) {
        return InvalidArgument("cannot copy layout from shape: ranks differs");
      }
      TF_RETURN_IF_ERROR(
          LayoutUtil::ValidateLayoutForShape(src.layout(), *dst));
      *dst->mutable_layout() = src.layout();
    } else {
      dst->clear_layout();
    }
  }
  return tensorflow::Status::OK();
}

}  // namespace


tensorflow::Status LayoutUtil::CopyLayoutBetweenShapes(const Shape& src,
                                                       Shape* dst) {
  return CopyLayoutInternal(src, dst);
}

bool LayoutUtil::LayoutsInShapesEqual(const Shape& lhs,
                                                   const Shape& rhs) {
  if (ShapeUtil::IsTuple(lhs) != ShapeUtil::IsTuple(rhs)) {
    return false;
  }
  if (ShapeUtil::IsTuple(lhs)) {
    if (ShapeUtil::TupleElementCount(lhs) !=
        ShapeUtil::TupleElementCount(rhs)) {
      return false;
    }
    for (int i = 0; i < ShapeUtil::TupleElementCount(lhs); ++i) {
      if (!LayoutsInShapesEqual(lhs.tuple_shapes(i), rhs.tuple_shapes(i))) {
        return false;
      }
    }
    return true;
  } else {
    return ShapeUtil::Rank(lhs) == ShapeUtil::Rank(rhs) &&
           LayoutUtil::Equal(lhs.layout(), rhs.layout());
  }
}

bool LayoutUtil::AreDimensionsConsecutive(
    const Layout& layout, tensorflow::gtl::ArraySlice<int64> dims) {
  std::vector<int64> positions_in_layout;
  for (int64 dim : dims) {
    positions_in_layout.push_back(
        PositionInContainer(layout.minor_to_major(), dim));
  }
  std::sort(positions_in_layout.begin(), positions_in_layout.end());
  for (size_t i = 1; i < positions_in_layout.size(); ++i) {
    if (1 != positions_in_layout[i] - positions_in_layout[i - 1]) {
      return false;
    }
  }
  return true;
}

}  // namespace xla
