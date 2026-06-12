#!/bin/bash
# LLM: Qwen2.5-7B-Instruct on GPU 4,5 → port 8005
# VLM: Qwen2.5-VL-7B-Instruct on GPU 6,7 → port 8010
echo "Starting Qwen2.5-7B-Instruct (LLM) on GPU 4,5 at port 8005..."
CUDA_VISIBLE_DEVICES=4,5 VLLM_USE_V1=0 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 2 \
    --port 8005 \
    --host 0.0.0.0 \
    --served-model-name Qwen/Qwen2.5-7B-Instruct \
    --max-model-len 32768 \
    &

LLM_PID=$!
echo "LLM server PID: $LLM_PID"

echo "Starting Qwen2.5-VL-7B-Instruct (VLM) on GPU 6,7 at port 8010..."
CUDA_VISIBLE_DEVICES=6,7 VLLM_USE_V1=0 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --tensor-parallel-size 2 \
    --port 8010 \
    --host 0.0.0.0 \
    --served-model-name Qwen/Qwen2.5-VL-7B-Instruct \
    --max-model-len 32768 \
    --limit-mm-per-prompt '{"image": 10}' \
    &

VLM_PID=$!
echo "VLM server PID: $VLM_PID"

echo "Both servers starting. PIDs: LLM=$LLM_PID VLM=$VLM_PID"
echo "Check status: curl http://localhost:8005/health && curl http://localhost:8010/health"
