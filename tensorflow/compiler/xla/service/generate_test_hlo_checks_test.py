# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for generate_test_hlo_checks."""

from absl.testing import absltest

from tensorflow.compiler.xla.service import generate_test_hlo_checks


class GenerateTestHloChecksTest(absltest.TestCase):

  def test_replacement(self):
    input_hlo = """
%param.0 # Do not replace if it's not CHECK'd.
// CHECK: %computation { # Do not replace computations
// CHECK: %param.0 = parameter(0) # Replace
// CHECK: %param_1 = parameter(1)
// CHECK-NEXT: %add.1 = add(%param.0, %param_1) # Replace for any CHECK-directive
// CHECK-NEXT: ROOT %reduce = reduce(%add.1)
"""
    self.assertEqual(
        generate_test_hlo_checks.replace_instruction_names(input_hlo), """
%param.0 # Do not replace if it's not CHECK'd.
// CHECK: %computation { # Do not replace computations
// CHECK: [[INSTR_0:%[^ ]+]] = parameter(0) # Replace
// CHECK: [[INSTR_1:%[^ ]+]] = parameter(1)
// CHECK-NEXT: [[INSTR_2:%[^ ]+]] = add([[INSTR_0]], [[INSTR_1]]) # Replace for any CHECK-directive
// CHECK-NEXT: ROOT [[INSTR_3:%[^ ]+]] = reduce([[INSTR_2]])
""")


if __name__ == '__main__':
  absltest.main()
