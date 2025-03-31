#!/bin/bash
CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen2.5-7B-Instruct --port 8722 --max-model-len 15472