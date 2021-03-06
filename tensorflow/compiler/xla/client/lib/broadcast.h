/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_LIB_BROADCAST_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_LIB_BROADCAST_H_

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

// Broadcasts 'input' up to shape 'output_dims', using TensorFlow broadcasting
// rules. Supports broadcasting a dimension of size x to size x*y, i.e., tiling.
StatusOr<XlaOp> BroadcastTo(XlaOp input, absl::Span<int64_t const> output_dims);

// Both ops are broadcasted to the same dimensions, so that each dimension is
// the max of the two.
// An InvalidArgument will be returned if the operations are of different rank
// or they share a dimension where they are unequal and neither is 1.
Status BroadcastOpsToSame(XlaOp* lhs, xla::XlaOp* rhs);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_LIB_BROADCAST_H_
