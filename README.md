# PIG

<p align="center">
  <b>Hybrid-Tower: Fine-grained Pseudo-query Interaction and Generation for Text-to-Video Retrieval</b><br/>
  ICCV 2025
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2509.04773">Paper</a> |
  <a href="https://arxiv.org/pdf/2509.04773">PDF</a> |
  <a href="https://lbx73737373.github.io/PIG-ProjectPage/">Project Page</a>
</p>

This repository releases the official training code for the PIG method on MSRVTT retrieval.
This first release supports exactly two training entrypoints:

- `scripts/train_pig_b32_msrvtt.sh`
- `scripts/train_clip-vip_b32_msrvtt.sh`

## Environment

Use `launch_container.sh` as the unified runtime entrypoint.

```bash
# At repo root
source launch_container.sh
```

- Host root path `/` is mounted to `/blob_mount` in container.
- Docker image is fixed to `lbx73737373/fk:torch111-cu113`.

## Data Layout

Place MSRVTT files under your storage root with this layout:

```text
/path/to/storage/
  datasets/
    msrvtt/
      annotations/
        train9k.jsonl
        test1ka.jsonl
      videos/
        all/
          video0.mp4
          ...
```

Inside container these paths become:

- `/blob_mount/datasets/msrvtt/annotations/train9k.jsonl`
- `/blob_mount/datasets/msrvtt/annotations/test1ka.jsonl`
- `/blob_mount/datasets/msrvtt/videos/all`

## Training

Enter the container first, then run one of the following:

### 1) Train CLIP-ViP baseline (B/32, MSRVTT)

```bash
bash scripts/train_clip-vip_b32_msrvtt.sh
```

Config:

- `src/configs/msrvtt_retrieval/msrvtt_retrieval_vip_base_32.json`

### 2) Train PIG (B/32, MSRVTT)

```bash
bash scripts/train_pig_b32_msrvtt.sh
```

Config:

- `src/configs/msrvtt_retrieval/msrvtt_retrieval_pig_base_32.json`

Fixed GPU settings in scripts:

- CLIP-ViP script: `CUDA_VISIBLE_DEVICES=0`, `horovodrun -np 1`
- PIG script: `CUDA_VISIBLE_DEVICES=0,1,2,3`, `horovodrun -np 4`

## Model Testing

### Test during training

Both training pipelines report MSRVTT retrieval metrics (`R@1`, `R@5`, `R@10`) in logs.
You can check:

- console output
- `${output_dir}/log/log.txt`

### PIG test-only run with a saved checkpoint

After finishing PIG training once, you can run test-only evaluation on `ckpt/model_best.pt`:

```bash
horovodrun -np 1 python src/tasks/run_video_retrieval_pig.py \
  --config src/configs/msrvtt_retrieval/msrvtt_retrieval_pig_base_32.json \
  --do_eval 1
```

This command loads the best checkpoint from:

- `outputs/msrvtt/pig_b32/ckpt/model_best.pt`

## Citation

If you find this project useful, please cite:

```bibtex
@inproceedings{lan2025hybridtower,
  title={Hybrid-Tower: Fine-grained Pseudo-query Interaction and Generation for Text-to-Video Retrieval},
  author={Lan, Bangxiang and Xie, Ruobing and Zhao, Ruixiang and Sun, Xingwu and Kang, Zhanhui and Yang, Gang and Li, Xirong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```

## Acknowledgement

This codebase is built upon CLIP, XPool, CLIP-ViP and HD-VILA. We thank the original authors for their excellent contributions and for making their work publicly available.
