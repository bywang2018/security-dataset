#!/usr/bin/env bash

PYTHON=${PYTHON:-"python3"}

CONFIG=$1
GPUS=$2
PORT=${PORT:-29501}

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
