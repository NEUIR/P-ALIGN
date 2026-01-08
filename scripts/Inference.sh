#!/bin/bash
# 用法：bash run_batch_infer.sh

MODEL_PATH="your model path"
 
INPUT_FILES=(
    "input test data path1"
    "input test data path2"
)

OUTPUT_FILES=(
    "result path1"
    "result path2"
)

# echo ">>> Starting batch inference..."

export CUDA_VISIBLE_DEVICES=6

nohup python src/test_pro.py \
    --model "$MODEL_PATH" \
    --input_files "${INPUT_FILES[@]}" \
    --output_files "${OUTPUT_FILES[@]}" \
    --batch_size 1000 \
    --n 3 \
    --temperature 0.6 \
    --top_p 0.9 \
    --max_tokens 4096 \
    > output/log/result.log 2>&1 &

echo "✅ All jobs finished."
