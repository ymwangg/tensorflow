#!/bin/bash
export XLA_FLAGS="--xla_dump_to=./hlo_graph --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=.*"
nvprof ../bazel-bin/tensorflow/compiler/xla/tools/replay_computation_gpu --use_fake_data=true --num_runs=1 --print_result=false $1
