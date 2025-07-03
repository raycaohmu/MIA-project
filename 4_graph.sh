python construct_graph_clean.py \
    --slide_csv_path ./data/slide_ov_response.csv \
    --output_dir ./output/graph_output/ \
    --min_tumor_cells_per_patch 20 \
    --num_workers 1 \
    --worker_type basic

echo "=== 处理完成 ==="