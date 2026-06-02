train_config_name=$1
model_name=$2
gpu_use=$3
fast_tokenizer_path=$4

export CUDA_VISIBLE_DEVICES=$gpu_use
if [ -n "$fast_tokenizer_path" ]; then
    export OPENPI_FAST_TOKENIZER_PATH=$fast_tokenizer_path
fi
#export http_proxy="http://127.0.0.1:1081"
#export https_proxy="http://127.0.0.1:1081"

echo $CUDA_VISIBLE_DEVICES
if [ -n "$OPENPI_FAST_TOKENIZER_PATH" ]; then
    echo "OPENPI_FAST_TOKENIZER_PATH=$OPENPI_FAST_TOKENIZER_PATH"
fi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONDONTWRITEBYTECODE=1 uv run scripts/train.py $train_config_name --exp-name=$model_name --overwrite
