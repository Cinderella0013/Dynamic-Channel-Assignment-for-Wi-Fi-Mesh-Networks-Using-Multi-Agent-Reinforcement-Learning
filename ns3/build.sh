#!/usr/bin/env bash
# Build the ns-3 mesh bridge inside WSL: bash /mnt/e/MESH/ns3/build.sh [NS3_ROOT]
# Copies ns3/mesh_bridge.cc into scratch/ and builds build/scratch/ns3.47-mesh_bridge.
set -euo pipefail
ROOT="${1:-$HOME/ns3_validation/ns-3.47}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cp "$HERE/mesh_bridge.cc" "$ROOT/scratch/mesh_bridge.cc"
cd "$ROOT"
cmake -S . -B cmake-cache > /dev/null   # re-glob scratch/ so the new target exists
cmake --build cmake-cache --target scratch_mesh_bridge -j"$(nproc)"
ls -l build/scratch/ns3.47-mesh_bridge
