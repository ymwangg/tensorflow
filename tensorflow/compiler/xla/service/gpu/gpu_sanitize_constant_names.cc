/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/gpu_sanitize_constant_names.h"

#include <memory>
#include <set>
#include <vector>

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/llvm_ir/buffer_assignment_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

namespace gpu {

bool GetEnvBool(const char* name, bool defval) {
  const char* env = getenv(name);
  if (env == nullptr) {
    return defval;
  }
  if (std::strcmp(env, "true") == 0) {
    return true;
  }
  if (std::strcmp(env, "false") == 0) {
    return false;
  }
  int int_env;
  bool has_int = absl::SimpleAtoi(env, &int_env);
  return has_int && int_env != 0;
}

StatusOr<bool> GpuSanitizeConstantNames::Run(HloModule* module) {
  bool changed = false;

  NameUniquer instr_name_uniquer(/*separator=*/"_");
  // Collect the names used for the non-constant HLO instructions.+
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instr : computation->instructions()) {
      if (instr->opcode() == HloOpcode::kConstant) {
        continue;
      }

      const std::string& old_name = instr->name();
      instr->UniquifyName(&instr_name_uniquer);
      CHECK_EQ(old_name, instr->name());
    }
  }

  // Sanitize the names for the constant HLO instructions and make them unique.
  // This is not merged into the above loop because we don't want this pass to
  // change the names of non-constant instructions, that is, if a constant HLO
  // conflicts with a non-constant HLO, we change the name of the constant HLO
  // even though the non-constant HLO comes after in the HLO module.
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instr : computation->instructions()) {
      if (instr->opcode() != HloOpcode::kConstant) {
        continue;
      }
      std::string sanitized_name = llvm_ir::SanitizeConstantName(*instr);
      instr->SetAndSanitizeName(sanitized_name);
      instr->UniquifyName(&instr_name_uniquer);
      changed = true;
    }
  }

  static const bool fold_constant = GetEnvBool("XLA_FOLD_CONSTANT", false);
  if (fold_constant) {
    std::vector<HloInstruction*> reduce_list;
    for (HloInstruction* instr : module->entry_computation()->instructions()) {
      if (instr->opcode() == HloOpcode::kReduce) {
        reduce_list.push_back(instr);
      }
    }
    for (HloInstruction* instr : reduce_list) {
      module->entry_computation()->CreateFusionInstruction(
          {instr}, HloInstruction::FusionKind::kLoop);
    }

    for (HloComputation* computation : module->computations()) {
      for (HloInstruction* instr : computation->instructions()) {
        if (instr->opcode() == HloOpcode::kFusion) {
          HloFusionInstruction* fused_instr =
              DynCast<HloFusionInstruction>(instr);
          // std::cout << fused_instr->ToProto().SerializeAsString() <<
          // std::endl;
          std::vector<HloInstruction*> fuse_list;
          for (HloInstruction* operand : fused_instr->operands()) {
            if (operand->opcode() == HloOpcode::kConstant) {
              fuse_list.push_back(operand);
            }
          }
          for (HloInstruction* operand : fuse_list) {
            std::cout << xla::ShapeUtil::HumanString(operand->shape())
                      << xla::ShapeUtil::ByteSizeOfElements(operand->shape())
                      << std::endl;
            fused_instr->FuseInstruction(operand);
            changed = true;
          }
          // std::cout << fused_instr->ToProto().SerializeAsString() <<
          // std::endl;
        }
      }
    }
  }
  return changed;
}  // namespace gpu

}  // namespace gpu
}  // namespace xla
