#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

log_dir=${RUNNER_TEMP:-"$repo_root/build-coverage"}/logs
mkdir -p "$log_dir"

# Run the complete portable CPU and GPU regression suites.
test/run_tests -d test/python -j 32 -v 1 -w 3 \
  --exe build-coverage/python/opensn \
  --run-log "$log_dir/nightly_coverage_log.txt"

test/run_tests -d test/python --gpu -j 32 -v 1 -w 3 \
  --exe build-coverage/python/opensn \
  --run-log "$log_dir/nightly_coverage_gpu_log.txt"

# Reuse two exact-solution regressions with the native build. Together they
# exercise the generic mixed-cell and fixed-node SIMD sweep implementations.
test/run_tests -j 2 -v 1 \
  -t test/python/modules/linear_boltzmann_solvers/transport_steady/transport_mixed_cell_infinite_medium.py \
     test/python/framework/mesh/pvtu_roundtrip_transport.py \
  --exe build-coverage-native/python/opensn \
  --run-log "$log_dir/nightly_coverage_native_simd_log.txt"

./build-coverage/test/opensn-unit
mpiexec -n 2 ./build-coverage/test/opensn-unit \
  --gtest_filter='MathTest.ParallelVector:DataTypesTest.00:SweepCommunicatorTest.HasPrivateMessageContext:CMFDCoarseMesh.*:DiffusionMIP.MMSL2ConvergenceDistributed:OrthogonalMeshGeneratorTest.DistributedGenerationMatchesStandardMesh:VectorGhostCommunicatorTest.*'
mpiexec -n 4 ./build-coverage/test/opensn-unit \
  --gtest_filter='SDMTestPWLC.*:SDMTestPWLD.*:TutorialsTest.Pwlc1:TutorialsTest.Pwlc2'

export PYTHONPATH="$repo_root/build-coverage${PYTHONPATH:+:$PYTHONPATH}"
test/run_tests -d doc/source/tutorials -j 32 -v 1 -w 3 \
  --exe build-coverage/python/opensn --engine=jupyter \
  --run-log "$log_dir/nightly_coverage_tutorials_log.txt"
