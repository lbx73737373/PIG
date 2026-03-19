export CUDA_VISIBLE_DEVICES=0
export HOROVOD_CACHE_CAPACITY=0
export TOKENIZERS_PARALLELISM=false
horovodrun -np 1 python src/tasks/run_video_retrieval.py \
--config src/configs/msrvtt_retrieval/msrvtt_retrieval_vip_base_32.json

echo "Training Completed"

