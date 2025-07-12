#!/bin/bash
CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8722 --max-model-len 15472
# CUDA_VISIBLE_DEVICES=0,1 vllm serve meta-llama/Llama-3.3-70B-Instruct --port 8722 --max-model-len 8192 --tensor-parallel-size 2 --gpu-memory-utilization 0.95
# CUDA_VISIBLE_DEVICES=0 vllm serve /work/yihong/ckpt/checkpoint-1254 \
#   --port 8722 \
#   --max-model-len 15472 \