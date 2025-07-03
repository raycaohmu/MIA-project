#!/bin/bash

# 设置变量
MODEL_PATH="./models/pretrained/NuLite/NuLite-H-Weights.pth"
WSI_PATH="./data/wsi"
OUTPUT_PATH="./data/wsi_output"

# 确保路径存在
mkdir -p "$OUTPUT_PATH"

# 检查必要文件和目录
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    exit 1
fi

if [ ! -d "$WSI_PATH" ]; then
    echo "Error: WSI directory not found at $WSI_PATH"
    exit 1
fi

# 运行命令 - 注意 --model 参数必须在子命令之前
python run_inference_wsi.py \
    --model "$MODEL_PATH" \
    --batch_size 20 \
    --magnification 20 \
    --gpu 0 \
    --geojson \
    process_dataset \
    --wsi_paths "$WSI_PATH" \
    --patch_dataset_path "$OUTPUT_PATH" \
    --wsi_extension svs