DATA_DIR='/'

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
   CUDA_VISIBLE_DEVICES='all'
fi

docker run --gpus "device=$CUDA_VISIBLE_DEVICES" --ipc=host --rm -it \
   --mount src="$(pwd)",dst=/workspace/PIG,type=bind \
   --mount src="$DATA_DIR",dst=/blob_mount,type=bind \
   -e NVIDIA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
   -w /workspace/PIG lbx73737373/fk:torch111-cu113 \
   bash -c "source /workspace/PIG/setup.sh && export OMPI_MCA_btl_vader_single_copy_mechanism=none && bash"


