#!/bin/bash
nvprof ../bazel-bin/tensorflow/compiler/xla/tools/replay_computation_gpu --use_fake_data=true --num_runs=100 --print_result=false $1
