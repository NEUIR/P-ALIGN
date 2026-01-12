#!/bin/bash

NPROC_PER_NODE=1
NNODES=1
RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=29330
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1 
export DISABLE_VERSION_CHECK=1
CUDA_VISIBLE_DEVICES="GPU device id" torchrun \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    src/train.py example.yaml > output.log 2>&1 &
