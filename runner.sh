#!/bin/bash

set -Eeuo pipefail

SCRIPT_DIR=$(dirname $(readlink -f $0))
export PYTHONPATH=$(readlink -f "${SCRIPT_DIR}")
export OPENBLAS_NUM_THREADS=1

$SCRIPT_DIR/env/bin/python $@
