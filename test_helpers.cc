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

#include "test_helpers.h"
#include "types.h"

#include "default_logging.h"

//#include "tensorflow/core/platform/regexp.h"  // need for ContainsRegex, HasSubstr

tensorflow::internal::LogMessage& LogMessage_check(const char* fname, int line, int severity, bool value)
{
	static tensorflow::internal::LogMessage msg(fname, line, tensorflow::INFO);

	if (severity == tensorflow::FATAL && !value)
	{
      msg(fname, line, tensorflow::ERROR, true);
		tensorflow::internal::LogMessageFatal(fname, line);
	}
	else if(!value)
	{
		msg(fname, line, tensorflow::ERROR, true);
	}
	else
	{
		msg(fname, line, severity, false);
	}

	return msg;
}

namespace xla {
namespace testing {

AssertionResult::AssertionResult(const AssertionResult& other)
    : success_(other.success_),
      message_(other.message_ != nullptr ? new std::string(*other.message_)
                                         : static_cast<std::string*>(nullptr)) 
{
}

// Returns the assertion's negation. Used with EXPECT/ASSERT_FALSE.
AssertionResult AssertionResult::operator!() const {
  AssertionResult negation(!success_);
  if (message_ != nullptr) negation << *message_;
  return negation;
}

AssertionResult& AssertionResult::operator=(const AssertionResult& ar) {
  success_ = ar.success_;
  message_.reset(ar.message_ != nullptr ? new std::string(*ar.message_)
                                        : nullptr);
  return *this;
}

AssertionResult AssertionFailure() { return AssertionResult(false); }

AssertionResult AssertionSuccess() { return AssertionResult(true); }

// TODO:
//std::function<bool(tensorflow::StringPiece)> ContainsRegex(
//    const tensorflow::StringPiece regex) 
//{
//  return [regex](const tensorflow::StringPiece to_test) 
//  {
//    if (RE2::PartialMatch(
//            tensorflow::RegexpStringPiece(to_test.data(), to_test.size()),
//            tensorflow::RegexpStringPiece(regex.data(), regex.size()))) 
//    {
//      return true;
//    } 
//    else 
//    {
//      LOG(ERROR) << "Expected to find " << regex << " in " << to_test;
//      return false;
//    }
//  };
//}
//
//std::function<bool(tensorflow::StringPiece)> HasSubstr(
//    const tensorflow::StringPiece part) 
//{
//  return [part](const tensorflow::StringPiece whole) {
//    return whole.contains(part);
//  };
//}

}  // namespace testing
}  // namespace xla
