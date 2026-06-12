#!/bin/bash
# LLM: Qwen3-30B-A3B-Instruct-2507 on GPU 0,1,2,3 → port 8005  (tensor-parallel=4)
# VLM: Qwen3-VL-8B-Instruct      on GPU 4,5,6,7 → port 8010  (tensor-parallel=4)
# Requires: poster-forest conda env (vLLM 0.12.0+)

PYTHON=python

echo "Starting Qwen3-30B-A3B-Instruct-2507 (LLM) on GPU 0,1,2,3 at port 8005 (TP=4)..."
CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --tensor-parallel-size 4 \
    --port 8005 \
    --host 0.0.0.0 \
    --served-model-name Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --enforce-eager \
    --gpu-memory-utilization 0.90 \
    &

LLM_PID=$!
echo "LLM server PID: $LLM_PID"

echo "Starting Qwen3-VL-8B-Instruct (VLM) on GPU 4,5,6,7 at port 8010 (TP=4)..."
# Fix: add gpu-memory-utilization, expandable_segments to prevent OOM on large images
# Fix: add max_pixels limit (4M pixels) to cap oversized images before processing
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PYTHON -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --tensor-parallel-size 4 \
    --port 8010 \
    --host 0.0.0.0 \
    --served-model-name Qwen/Qwen3-VL-8B-Instruct \
    --gpu-memory-utilization 0.85 \
    --limit-mm-per-prompt '{"image": 10}' \
    --mm-processor-kwargs '{"max_pixels": 1003520}' \
    &

VLM_PID=$!
echo "VLM server PID: $VLM_PID"

echo "Both servers starting. PIDs: LLM=$LLM_PID VLM=$VLM_PID"
echo "Check status: curl -s http://localhost:8005/health && curl -s http://localhost:8010/health"
