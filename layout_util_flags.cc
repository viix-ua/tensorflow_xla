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

// Legacy flags for XLA's layout_util module.

#include <mutex>  // NOLINT(build/c++11): only using std::call_once, not mutex.
#include <vector>

#include "layout_util_flags.h"

namespace xla {
namespace legacy_flags {

// Pointers to the string value of the xla_default_layout flag and the flag
// descriptor, initialized via raw_flags_init.
//static std::string* raw_flag;
//static std::vector<tensorflow::Flag>* flag_list;
static std::once_flag raw_flags_init;


// Pointer to the parsed value of the flags, initialized via flags_init.
static LayoutUtilFlags* flags;
static std::once_flag flags_init;

// Allocate *flags.  Called via call_once(&flags_init,...).
static void AllocateFlags() {
  //std::call_once(raw_flags_init, &AllocateRawFlag);
  flags = new LayoutUtilFlags;
  flags->xla_default_layout.dimension_order =
      DefaultLayout::DimensionOrder::kMajorToMinor;
  flags->xla_default_layout.seed = 0;
  //if (!ParseDefaultLayout(*raw_flag, &flags->xla_default_layout)) 
  {
  //  flags = nullptr;
  }
}

// Return a pointer to the LayoutUtilFlags struct;
// repeated calls return the same pointer.
// This should be called only after Flags::Parse() has returned.
LayoutUtilFlags* GetLayoutUtilFlags() {
  std::call_once(flags_init, &AllocateFlags);
  return flags;
}

}  // namespace legacy_flags
}  // namespace xla
