export CUDA_VISIBLE_DEVICES= "your device GPU id."

nohup python src/Prefix_alignment.py > output/log/prefix-alignment.log 2>&1 &
