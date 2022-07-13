#include "tensorflow/compiler/xla/service/localize_constant.h"

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

StatusOr<bool> LocalizeConstant::Run(HloModule* module) {
  bool changed = false;
  for (HloComputation* comp : module->MakeNonfusionComputations()) {
    for (HloInstruction* instruction : comp->MakeInstructionPostOrder()) {
      if (instruction->HasSideEffect() || !instruction->shape().IsArray()) {
        continue;
      }
      if (instruction->IsConstant()) {
        Shape shape = instruction->shape();
        if (!LayoutUtil::HasLayout(shape)) {
          LayoutUtil::SetToDefaultLayout(&shape);
        }
        std::cout << "Replacing constant op" << std::endl;
        TF_RETURN_IF_ERROR(comp->ReplaceWithNewInstruction(
            instruction,
            HloInstruction::CreateConstant(Literal::CreateFromShape(shape))));
        changed = true;
      }
    }
  }
  return changed;
}

}  // namespace xla
