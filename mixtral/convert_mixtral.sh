TOKENIZER_MODEL=/kimchou/Megatron-LM/mixtral/tokenizer.model
MEGATRON_PATH="/kimchou/Megatron-LM/"
export PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

TARGET_TP_SIZE=1
TARGET_EP_SIZE=8
TARGET_PP_SIZE=4

HF_FORMAT_DIR=/kimchou/Megatron-LM/mixtral
MEGATRON_FORMAT_DIR=/kimchou/Megatron-LM/mixtral/mixtral-mcore-TP${TARGET_TP_SIZE}PP${TARGET_PP_SIZE}EP${TARGET_EP_SIZE}

python tools/checkpoint/convert.py \
--model-type GPT \
--loader loader_mixtral_hf \
--saver mcore \
--target-tensor-parallel-size ${TARGET_TP_SIZE} \
--target-pipeline-parallel-size ${TARGET_PP_SIZE} \
--target-expert-parallel-size ${TARGET_EP_SIZE} \
--load-dir ${HF_FORMAT_DIR} \
--save-dir ${MEGATRON_FORMAT_DIR} \
--tokenizer-model ${TOKENIZER_MODEL}
