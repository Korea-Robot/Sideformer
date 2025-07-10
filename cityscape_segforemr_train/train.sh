#!/bin/bash

# run_multi_gpu.sh
# A100 GPU 6개를 사용한 Multi-GPU 학습 실행 스크립트

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^docker0,lo

# 분산 학습 설정
NUM_GPUS=6
MASTER_PORT=29501

echo "Multi-GPU 학습 시작"
echo "사용 GPU 수: $NUM_GPUS"
echo "사용 GPU: $CUDA_VISIBLE_DEVICES"
echo "Master Port: $MASTER_PORT"

# torchrun을 사용한 분산 학습 실행
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    train.py

echo "Multi-GPU 학습 완료"