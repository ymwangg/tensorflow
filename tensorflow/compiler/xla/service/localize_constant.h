#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_LOCALIZE_CONSTANT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_LOCALIZE_CONSTANT_H_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

// HLO pass that replaces zero sized Hlos with a zero sized constant literal.
namespace xla {
class LocalizeConstant : public HloModulePass {
 public:
  StatusOr<bool> Run(HloModule* module) override;
  absl::string_view name() const override {
    return "localize_constant";
  }
};
}  // namespace xla
#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_LOCALIZE_CONSTANT_H_
