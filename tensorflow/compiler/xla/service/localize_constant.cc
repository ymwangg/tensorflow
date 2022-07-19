#include "tensorflow/compiler/xla/service/localize_constant.h"

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

StatusOr<bool> LocalizeConstant::Run(HloModule* module) {
  bool changed = false;

  std::vector<HloInstruction*> reduce_instructions;
  for (HloInstruction* instr: module->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kReduce) {
      reduce_instructions.push_back(instr);
    }
  }
  for (HloInstruction* instr: reduce_instructions) {
    module->entry_computation()->CreateFusionInstruction({instr}, HloInstruction::FusionKind::kLoop);
  }

  for (HloComputation* comp : module->computations()) {
    for (HloInstruction* instruction : comp->instructions()) {
      if (instruction->opcode() == HloOpcode::kFusion) {
        HloFusionInstruction* fusion_instr = DynCast<HloFusionInstruction>(instruction);
        std::vector<HloInstruction*> const_operands;
        for (HloInstruction* operand: fusion_instr->operands()) {
          if (operand->opcode() == HloOpcode::kConstant) {
            const_operands.push_back(operand);
          }
        }
        for (HloInstruction* operand: const_operands) {
          // std::cout << "localize constant operands" << std::endl;
          fusion_instr->FuseInstruction(operand);
          changed = true;
        }
      }
    }
  }
  return changed;
}

}  // namespace xla
