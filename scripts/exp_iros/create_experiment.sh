#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/config_experiment.yaml"

source "${SCRIPT_DIR}/../ubash.sh" || exit 1

cd ${PROJECT_DIR}

${VPY_BIN} inverse_dynamics/experiment_config.py \
           --fp_config ${CONFIG_PATH}
