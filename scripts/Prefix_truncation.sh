export CUDA_VISIBLE_DEVICES=" your device GPU i.d."

nohup python src/binary_select.py > output/log/prefix.log 2>&1 &
