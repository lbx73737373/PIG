export CUDA_VISIBLE_DEVICES=0,1,2,3
export HOROVOD_CACHE_CAPACITY=0
export TOKENIZERS_PARALLELISM=false
horovodrun -np 4 python src/tasks/run_video_retrieval_pig.py \
--config src/configs/msrvtt_retrieval/msrvtt_retrieval_pig_base_32.json

echo "Training Completed"
