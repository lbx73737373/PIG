import torch
from torch.utils.data import Dataset, get_worker_info
import random
import os
import json
from typing import List
import ipdb
from tqdm import tqdm
from torch.utils.data.dataloader import default_collate
from src.utils.logger import LOGGER
from src.utils.basic_utils import flat_list_of_lists
from src.datasets.data_utils import mask_batch_text_tokens, img_collate
from src.datasets.dataloader import init_transform_dict, init_transform_dict_simple
import decord
from decord import VideoReader
from decord import cpu, gpu

decord.bridge.set_bridge("torch")
import math
import torch.nn.functional as F
import numpy as np
import cv2
import lmdb
import glob
import src.utils.stop_words as stop_words
from PIL import Image
from src.datasets.sample_frames import SampleFrames
from typing import List

class HDVILAVideoRetrievalDataset(Dataset):
    """
    datalist
    """

    def __init__(self, cfg, vis_dir, anno_path, vis_format='video', mode="train"):
        assert vis_format in ["video", "frame"]
        self.cfg = cfg
        self.vis_dir = vis_dir
        self.anno_path = anno_path
        self.mode = mode
        self.vis_format = vis_format
        self.n_clips = cfg.train_n_clips if mode == "train" else cfg.test_n_clips
        self.num_frm = cfg.train_num_frms if mode == "train" else cfg.test_num_frms
        self.sample_rate = cfg.sample_rate
        if hasattr(cfg, "text_pos_num"):
            self.pos_num = cfg.pos_num
        else:
            self.pos_num = 1
        self.transform = init_transform_dict_simple(video_res=cfg.video_res,
                                                    input_res=cfg.input_res)[mode]
        self.frame_sampler = SampleFrames(clip_len=self.num_frm,
                                          frame_interval=self.sample_rate,
                                          num_clips=self.n_clips,
                                          temporal_jitter=True)
        self.init_dataset_process()

    def init_dataset_process(self):
        json_type = os.path.splitext(self.anno_path)[-1]
        assert json_type in ['.json', '.jsonl']

        if json_type == '.jsonl':
            data = []
            with open(self.anno_path) as f:
                for line in f:
                    data.append(json.loads(line))
        else:
            data = json.load(open(self.anno_path))
        self.datalist = data


    def id2path(self, id):
        clip_name = id
        if self.vis_format == 'video':
            name = os.path.join(self.vis_dir, clip_name.split('/')[-1] + ".mp4")
            if "lsmdc" in self.vis_dir:
                name = os.path.join(self.vis_dir, clip_name + ".avi")
        else:
            name = os.path.join(self.vis_dir, clip_name)
        return name

    def __len__(self):
        return len(self.datalist)

    # def get_sample_idx(self, total_frame_num, vis_path=None):
    #     """
    #     sample rate > 0: use SampleFrames, loop default
    #     sample rate = 0: uniform sampling, temporal jittering
    #     """
    #     if self.sample_rate > 0:
    #         results = {"total_frames": total_frame_num,
    #                    "start_index": 0}
    #         results = self.frame_sampler(results)
    #         return results["frame_inds"]
    #     elif self.sample_rate == 0:
    #         if hasattr(self.cfg, "sample_jitter") and self.cfg.sample_jitter and self.mode == "train":
    #             interval = int(total_frame_num / (self.n_clips * self.num_frm - 1))
    #             start = np.random.randint(0, interval + 1)
    #             end = np.random.randint(total_frame_num - 1 - interval, total_frame_num)
    #             return np.linspace(start, end, self.n_clips * self.num_frm).astype(int)
    #         else:
    #             # if hasattr(self.cfg, 'limited_frame_sampling') and self.cfg.limited_frame_sampling:
    #             if vis_path is not None and ('VATEX' in vis_path or 'vatex' in vis_path):
    #                 image_names = [x for x in os.listdir(vis_path) if x.endswith('.jpg')]
    #                 frame_ids = [int(x.split('.')[0].split('_')[-1]) for x in image_names]
    #                 frame_ids.sort()
    #                 idx = np.linspace(0, len(frame_ids) - 1, self.n_clips * self.num_frm).astype(int)
    #                 return np.array(frame_ids, dtype=int)[idx]
    #             else:
    #                 return np.linspace(0, total_frame_num - 1, self.n_clips * self.num_frm).astype(int)

    def get_sample_idx(self, total_frame_num, vis_path=None):
        """
        sample rate > 0: use SampleFrames, loop default
        sample rate = 0: uniform sampling, temporal jittering
        """
        if self.sample_rate > 0:
            results = {"total_frames": total_frame_num,
                       "start_index": 0}
            results = self.frame_sampler(results)
            return results["frame_inds"]
        elif self.sample_rate == 0:
            if hasattr(self.cfg, "sample_jitter") and self.cfg.sample_jitter and self.mode == "train":
                if vis_path is not None and ('VATEX' in vis_path or 'vatex' in vis_path):
                    image_names = [x for x in os.listdir(vis_path) if x.endswith('.jpg')]
                    frame_ids = [int(x.split('.')[0].split('_')[-1]) for x in image_names]
                    frame_ids.sort()
                    interval = int(len(frame_ids) / (self.n_clips * self.num_frm - 1))
                    start = np.random.randint(0, interval + 1)
                    end = np.random.randint(len(frame_ids) - 1 - interval, len(frame_ids))
                    idx = np.linspace(start, end, self.n_clips * self.num_frm).astype(int)
                    # idx = np.linspace(0, len(frame_ids) - 1, self.n_clips * self.num_frm).astype(int)
                    return np.array(frame_ids, dtype=int)[idx]
                else:
                    interval = int(total_frame_num / (self.n_clips * self.num_frm - 1))
                    start = np.random.randint(0, interval + 1)
                    end = np.random.randint(total_frame_num - 1 - interval, total_frame_num)
                    return np.linspace(start, end, self.n_clips * self.num_frm).astype(int)
            else:
                if vis_path is not None and ('VATEX' in vis_path or 'vatex' in vis_path):
                    image_names = [x for x in os.listdir(vis_path) if x.endswith('.jpg')]
                    frame_ids = [int(x.split('.')[0].split('_')[-1]) for x in image_names]
                    frame_ids.sort()
                    idx = np.linspace(0, len(frame_ids) - 1, self.n_clips * self.num_frm).astype(int)
                    return np.array(frame_ids, dtype=int)[idx]
                else:
                    return np.linspace(0, total_frame_num - 1, self.n_clips * self.num_frm).astype(int)

    def load_video(self, vis_path):
        vr = VideoReader(vis_path, ctx=cpu(0))
        total_frame_num = len(vr)

        frame_idx = self.get_sample_idx(total_frame_num)
        img_array = vr.get_batch(frame_idx)  # (n_clips*num_frm, H, W, 3)

        img_array = img_array.permute(0, 3, 1, 2).float() / 255.
        img_array = self.transform(img_array)

        return img_array

    def load_frames(self, vis_path, total_frame_num):
        # print('total_frame_num',total_frame_num)
        frame_idx = self.get_sample_idx(total_frame_num, vis_path)

        img_array = []
        for i in frame_idx:
            if 'VATEX' in vis_path or 'vatex' in vis_path:
                img = Image.open(os.path.join(vis_path, \
                                              vis_path.split('/')[-1] + '_{}.jpg'.format(i))).convert("RGB")

            else:
                img = Image.open(os.path.join(vis_path, \
                                              vis_path.split('/')[-1] + '_{0:03d}.jpg'.format(i))).convert("RGB")
            img_array.append(np.array(img))
        img_array = torch.from_numpy(np.array(img_array))  # (n_clips*num_frm, H, W, 3)

        img_array = img_array.permute(0, 3, 1, 2).float() / 255.
        img_array = self.transform(img_array)

        return img_array

    def __getitem__(self, index):
        if self.cfg.dummy_data:
            return dict(
                video=torch.randn(self.n_clips * self.num_frm, 3, self.cfg.input_res[0], self.cfg.input_res[1]),
                # [clips, num_frm, C, H_crop, W_crop]
                texts=["This is a dummy sentence, which contains nothing meaningful."]
            )

        vis_id = self.datalist[index]['clip_id']
        texts = self.datalist[index]['text']

        if isinstance(texts, list):
            texts = random.sample(self.datalist[index]['text'], self.pos_num)

            if 'didemo' in self.anno_path:
                texts = [' '.join(self.datalist[index]['text'])]
        else:
            texts = [texts]

        vis_path = self.id2path(vis_id)
        video = self.load_video(vis_path) if self.vis_format == 'video' else self.load_frames(vis_path,
                                                                                              self.datalist[index][
                                                                                                  'num_frame'])

        return dict(
            video=video,  # [clips*num_frm, C, H_crop, W_crop]
            texts=texts,
        )

class SentenceDataset(Dataset):
    """
    never set shuffle=True in DataLoader for this dataset
    """

    def __init__(self, cfg, vis_dir, anno_path, vis_format='video', mode="train", multi_sentence_per_video=False):
        assert vis_format in ["video", "frame"]
        self.cfg = cfg
        self.anno_path = anno_path
        self.mode = mode
        self.multi_sentence_per_video =  multi_sentence_per_video
        if hasattr(cfg, "text_pos_num"):
            self.pos_num = cfg.pos_num
        else:
            self.pos_num = 1

        self.init_dataset_process()

    def init_dataset_process(self):
        json_type = os.path.splitext(self.anno_path)[-1]
        assert json_type in ['.json', '.jsonl']

        if json_type == '.jsonl':
            data = []
            with open(self.anno_path) as f:
                for line in f:
                    data.append(json.loads(line))
        else:
            data = json.load(open(self.anno_path))
        self.datalist = data

        # expand all sentences in order
        self.sentences : List[str] = []
        self.cut_off_points : List[int] = []
        for data in self.datalist:
            sentences = data['text']
            if isinstance(sentences, list):
                self.sentences.extend(sentences)
            else:
                self.sentences.append(sentences)

            self.cut_off_points.append(len(self.sentences))

    def __len__(self):
        if 'didemo' in self.anno_path:
            return len(self.datalist)
        else:
            return len(self.sentences)

    def __getitem__(self, index):
        if 'didemo' in self.anno_path:
            texts = [' '.join(self.datalist[index]['text'])]
        else:
            texts = self.sentences[index]
            texts = [texts]

        return dict(
            index=index,
            texts=texts,
        )


class VideoDataset(Dataset):
    """
    never set shuffle=True in DataLoader for this dataset
    """

    def __init__(self, cfg, vis_dir, anno_path, vis_format='video', mode="train"):
        assert vis_format in ["video", "frame"]
        self.cfg = cfg
        self.vis_dir = vis_dir
        self.anno_path = anno_path
        self.mode = mode
        self.vis_format = vis_format
        self.n_clips = cfg.train_n_clips if mode == "train" else cfg.test_n_clips
        self.num_frm = cfg.train_num_frms if mode == "train" else cfg.test_num_frms
        self.sample_rate = cfg.sample_rate
        if hasattr(cfg, "text_pos_num"):
            self.pos_num = cfg.pos_num
        else:
            self.pos_num = 1
        self.transform = init_transform_dict_simple(video_res=cfg.video_res,
                                                    input_res=cfg.input_res)[mode]
        self.frame_sampler = SampleFrames(clip_len=self.num_frm,
                                          frame_interval=self.sample_rate,
                                          num_clips=self.n_clips,
                                          temporal_jitter=True)
        self.init_dataset_process()

    def init_dataset_process(self):
        json_type = os.path.splitext(self.anno_path)[-1]
        assert json_type in ['.json', '.jsonl']

        if json_type == '.jsonl':
            data = []
            with open(self.anno_path) as f:
                for line in f:
                    data.append(json.loads(line))
        else:
            data = json.load(open(self.anno_path))
        self.datalist = data


    def id2path(self, id):
        clip_name = id
        if self.vis_format == 'video':
            name = os.path.join(self.vis_dir, clip_name.split('/')[-1] + ".mp4")
            if "lsmdc" in self.vis_dir:
                name = os.path.join(self.vis_dir, clip_name + ".avi")
        else:
            name = os.path.join(self.vis_dir, clip_name)
        return name

    def __len__(self):
        return len(self.datalist)

    def get_sample_idx(self, total_frame_num, vis_path=None):
        """
        sample rate > 0: use SampleFrames, loop default
        sample rate = 0: uniform sampling, temporal jittering
        """
        if self.sample_rate > 0:
            results = {"total_frames": total_frame_num,
                       "start_index": 0}
            results = self.frame_sampler(results)
            return results["frame_inds"]
        elif self.sample_rate == 0:
            if hasattr(self.cfg, "sample_jitter") and self.cfg.sample_jitter and self.mode == "train":
                if vis_path is not None and ('VATEX' in vis_path or 'vatex' in vis_path):
                    image_names = [x for x in os.listdir(vis_path) if x.endswith('.jpg')]
                    frame_ids = [int(x.split('.')[0].split('_')[-1]) for x in image_names]
                    frame_ids.sort()
                    interval = int(len(frame_ids) / (self.n_clips * self.num_frm - 1))
                    start = np.random.randint(0, interval + 1)
                    end = np.random.randint(len(frame_ids) - 1 - interval, len(frame_ids))
                    idx = np.linspace(start, end, self.n_clips * self.num_frm).astype(int)
                    # idx = np.linspace(0, len(frame_ids) - 1, self.n_clips * self.num_frm).astype(int)
                    return np.array(frame_ids, dtype=int)[idx]
                else:
                    interval = int(total_frame_num / (self.n_clips * self.num_frm - 1))
                    start = np.random.randint(0, interval + 1)
                    end = np.random.randint(total_frame_num - 1 - interval, total_frame_num)
                    return np.linspace(start, end, self.n_clips * self.num_frm).astype(int)
            else:
                if vis_path is not None and ('VATEX' in vis_path or 'vatex' in vis_path):
                    image_names = [x for x in os.listdir(vis_path) if x.endswith('.jpg')]
                    frame_ids = [int(x.split('.')[0].split('_')[-1]) for x in image_names]
                    frame_ids.sort()
                    idx = np.linspace(0, len(frame_ids) - 1, self.n_clips * self.num_frm).astype(int)
                    return np.array(frame_ids, dtype=int)[idx]
                else:
                    return np.linspace(0, total_frame_num - 1, self.n_clips * self.num_frm).astype(int)

    def load_video(self, vis_path):
        vr = VideoReader(vis_path, ctx=cpu(0))
        total_frame_num = len(vr)

        frame_idx = self.get_sample_idx(total_frame_num)
        img_array = vr.get_batch(frame_idx)  # (n_clips*num_frm, H, W, 3)

        img_array = img_array.permute(0, 3, 1, 2).float() / 255.
        img_array = self.transform(img_array)

        return img_array

    def load_frames(self, vis_path, total_frame_num):
        # print('total_frame_num',total_frame_num)
        frame_idx = self.get_sample_idx(total_frame_num, vis_path)

        img_array = []
        for i in frame_idx:
            if 'VATEX' in vis_path or 'vatex' in vis_path:
                img = Image.open(os.path.join(vis_path, \
                                              vis_path.split('/')[-1] + '_{}.jpg'.format(i))).convert("RGB")
            else:
                img = Image.open(os.path.join(vis_path, \
                                              vis_path.split('/')[-1] + '_{0:03d}.jpg'.format(i))).convert("RGB")
            img_array.append(np.array(img))
        img_array = torch.from_numpy(np.array(img_array))  # (n_clips*num_frm, H, W, 3)

        img_array = img_array.permute(0, 3, 1, 2).float() / 255.
        img_array = self.transform(img_array)

        return img_array

    def __getitem__(self, index):
        if self.cfg.dummy_data:
            return dict(
                video=torch.randn(self.n_clips * self.num_frm, 3, self.cfg.input_res[0], self.cfg.input_res[1]),
                # [clips, num_frm, C, H_crop, W_crop]
                texts=["This is a dummy sentence, which contains nothing meaningful."]
            )

        vis_id = self.datalist[index]['clip_id']

        vis_path = self.id2path(vis_id)
        video = self.load_video(vis_path) if self.vis_format == 'video' else self.load_frames(vis_path,
                                                                                              self.datalist[index][
                                                                                                  'num_frame'])

        return dict(
            index=index,
            video=video,  # [clips*num_frm, C, H_crop, W_crop]
        )

class HDVILAVideoRetrievalWithAuxiliaryTextsDataset(Dataset):
    """
    datalist
    """

    def __init__(self, cfg, vis_dir, anno_path, auxiliary_anno_path, vis_format='video', mode="train"):
        assert vis_format in ["video", "frame"]
        self.cfg = cfg
        self.vis_dir = vis_dir
        self.anno_path = anno_path
        self.auxiliary_anno_path = auxiliary_anno_path
        self.mode = mode
        self.vis_format = vis_format
        self.n_clips = cfg.train_n_clips if mode == "train" else cfg.test_n_clips
        self.num_frm = cfg.train_num_frms if mode == "train" else cfg.test_num_frms
        self.sample_rate = cfg.sample_rate
        # in this way, we can sample different text for each video in A BATCH
        self.sample_text_idx = cfg.sample_text_idx
        if hasattr(cfg, "text_pos_num"):
            self.pos_num = cfg.pos_num
        else:
            self.pos_num = 1
        self.transform = init_transform_dict_simple(video_res=cfg.video_res,
                                                    input_res=cfg.input_res)[mode]
        self.frame_sampler = SampleFrames(clip_len=self.num_frm,
                                          frame_interval=self.sample_rate,
                                          num_clips=self.n_clips,
                                          temporal_jitter=True)
        self.init_dataset_process()

        # 初始化不重复采样所需的数据结构
        if self.cfg.thread_not_repeated_sampling:
            self.texts_remaining = {item['clip_id']: list(item['text']) if isinstance(item['text'], list) else [item['text']]
                                    for item in self.datalist}

    def init_dataset_process(self):
        json_type = os.path.splitext(self.anno_path)[-1]
        assert json_type in ['.json', '.jsonl']

        aux_json_type = os.path.splitext(self.auxiliary_anno_path)[-1]
        assert aux_json_type in ['.json', '.jsonl']

        aux_data = json.load(open(self.auxiliary_anno_path))
        # self.auxiliary_datalist = {x['video_id']: x['pred_caption'] for x in aux_data}
        # MSRVTT test split
        if len(aux_data) == 1000:
            self.auxiliary_datalist = {vis_id: v['titles'] for vis_id, v in aux_data.items()}
        else:
            self.auxiliary_datalist = {vis_id: [v] for vis_id, v in aux_data['title'].items()}

        if json_type == '.jsonl':
            data = []
            with open(self.anno_path) as f:
                for line in f:
                    data.append(json.loads(line))
        else:
            data = json.load(open(self.anno_path))
        self.datalist = data

        if self.sample_text_idx:
            pairlist = []
            for x in self.datalist:
                if isinstance(x['text'], list):
                    for text in x['text']:
                        pairlist.append({'clip_id': x['clip_id'], 'text': text})

            # overwrite datalist
            self.datalist = pairlist


    def id2path(self, id):
        clip_name = id
        if self.vis_format == 'video':
            name = os.path.join(self.vis_dir, clip_name.split('/')[-1] + ".mp4")
            if "lsmdc" in self.vis_dir:
                name = os.path.join(self.vis_dir, clip_name + ".avi")
        else:
            name = os.path.join(self.vis_dir, clip_name)
        return name

    def id2aux_text(self, id):
        aux_text = self.auxiliary_datalist[id]
        return aux_text

    def __len__(self):
        return len(self.datalist)

    def get_sample_idx(self, total_frame_num, vis_path=None):
        """
        sample rate > 0: use SampleFrames, loop default
        sample rate = 0: uniform sampling, temporal jittering
        """
        if self.sample_rate > 0:
            results = {"total_frames": total_frame_num,
                       "start_index": 0}
            results = self.frame_sampler(results)
            return results["frame_inds"]
        elif self.sample_rate == 0:
            if hasattr(self.cfg, "sample_jitter") and self.cfg.sample_jitter and self.mode == "train":
                if vis_path is not None and ('VATEX' in vis_path or 'vatex' in vis_path):
                    image_names = [x for x in os.listdir(vis_path) if x.endswith('.jpg')]
                    frame_ids = [int(x.split('.')[0].split('_')[-1]) for x in image_names]
                    frame_ids.sort()
                    interval = int(len(frame_ids) / (self.n_clips * self.num_frm - 1))
                    start = np.random.randint(0, interval + 1)
                    end = np.random.randint(len(frame_ids) - 1 - interval, len(frame_ids))
                    idx = np.linspace(start, end, self.n_clips * self.num_frm).astype(int)
                    # idx = np.linspace(0, len(frame_ids) - 1, self.n_clips * self.num_frm).astype(int)
                    return np.array(frame_ids, dtype=int)[idx]
                else:
                    interval = int(total_frame_num / (self.n_clips * self.num_frm - 1))
                    start = np.random.randint(0, interval + 1)
                    end = np.random.randint(total_frame_num - 1 - interval, total_frame_num)
                    return np.linspace(start, end, self.n_clips * self.num_frm).astype(int)
            else:
                if vis_path is not None and ('VATEX' in vis_path or 'vatex' in vis_path):
                    image_names = [x for x in os.listdir(vis_path) if x.endswith('.jpg')]
                    frame_ids = [int(x.split('.')[0].split('_')[-1]) for x in image_names]
                    frame_ids.sort()
                    idx = np.linspace(0, len(frame_ids) - 1, self.n_clips * self.num_frm).astype(int)
                    return np.array(frame_ids, dtype=int)[idx]
                else:
                    return np.linspace(0, total_frame_num - 1, self.n_clips * self.num_frm).astype(int)

    def load_video(self, vis_path):
        vr = VideoReader(vis_path, ctx=cpu(0))
        total_frame_num = len(vr)

        frame_idx = self.get_sample_idx(total_frame_num)
        img_array = vr.get_batch(frame_idx)  # (n_clips*num_frm, H, W, 3)

        img_array = img_array.permute(0, 3, 1, 2).float() / 255.
        img_array = self.transform(img_array)

        return img_array


    def load_frames(self, vis_path, total_frame_num):
        # print('total_frame_num',total_frame_num)
        frame_idx = self.get_sample_idx(total_frame_num, vis_path)

        img_array = []
        for i in frame_idx:
            if 'VATEX' in vis_path or 'vatex' in vis_path:
                img = Image.open(os.path.join(vis_path, \
                                              vis_path.split('/')[-1] + '_{}.jpg'.format(i))).convert("RGB")
            else:
                img = Image.open(os.path.join(vis_path, \
                                              vis_path.split('/')[-1] + '_{0:03d}.jpg'.format(i))).convert("RGB")
            img_array.append(np.array(img))
        img_array = torch.from_numpy(np.array(img_array))  # (n_clips*num_frm, H, W, 3)

        img_array = img_array.permute(0, 3, 1, 2).float() / 255.
        img_array = self.transform(img_array)

        return img_array

    def __getitem__(self, index):
        if self.cfg.dummy_data:
            return dict(
                video=torch.randn(self.n_clips * self.num_frm, 3, self.cfg.input_res[0], self.cfg.input_res[1]),
                # [clips, num_frm, C, H_crop, W_crop]
                texts=["This is a dummy sentence, which contains nothing meaningful."],
                aux_texts=["This is a dummy auxiliary sentence."]
            )

        vis_id = self.datalist[index]['clip_id']
        texts = self.datalist[index]['text']
        aux_texts = self.id2aux_text(vis_id)
        aux_texts = [aux_texts]

        if isinstance(texts, list):
            if self.cfg.thread_not_repeated_sampling:
                remaining_texts = self.texts_remaining.get(vis_id, []).copy()

                if len(remaining_texts) < self.pos_num:
                    # 重置未采样的文本列表
                    remaining_texts = list(self.datalist[index]['text'])
                    self.texts_remaining[vis_id] = remaining_texts.copy()

                sampled_texts = random.sample(remaining_texts, self.pos_num)
                for text in sampled_texts:
                    self.texts_remaining[vis_id].remove(text)

                texts = sampled_texts
            else:
                texts = random.sample(self.datalist[index]['text'], self.pos_num)

            if 'didemo' in self.anno_path:
                texts = [' '.join(self.datalist[index]['text'])]
        else:
            texts = [texts]

        vis_path = self.id2path(vis_id)
        video = self.load_video(vis_path) if self.vis_format == 'video' else self.load_frames(vis_path,
                                                                                              self.datalist[index][
                                                                                                  'num_frame'])
        # print('vis_id: ', vis_id)
        # print('texts: ', texts)
        # print('aux_texts: ', aux_texts)

        return dict(
            video=video,  # [clips*num_frm, C, H_crop, W_crop]
            texts=texts,
            aux_texts=aux_texts
        )


class ImageTextPretrainingDataset(Dataset):
    def __init__(self, cfg, vis_dir, anno_path, mode='train'):
        self.cfg = cfg
        self.vis_dir = vis_dir
        self.anno_path = anno_path
        self.mode = mode
        self.transform = init_transform_dict_simple(video_res=cfg.image_res,
                                                    input_res=cfg.input_res)[mode]
        self.init_dataset_process()

    def init_dataset_process(self):
        json_type = os.path.splitext(self.anno_path)[-1]
        assert json_type == '.json'

        chat_data = json.load(open(self.anno_path))
        val_split = self.cfg.val_split
        if self.mode == 'train':
            chat_data = chat_data[:-val_split]
        else:
            # using the rest of the data for validation
            chat_data = chat_data[-val_split:]
        self.datalist = []
        for x in chat_data:
            caption = x['conversations'][-1]['value']
            self.datalist.append({'id': x['id'], 'image': x['image'], 'caption': caption})

    def __len__(self):
        return len(self.datalist)

    def id2path(self, id):
        return os.path.join(self.vis_dir, self.datalist[id]['image'])

    def load_image(self, vis_path):
        img = Image.open(vis_path).convert("RGB")
        # (3, H, W)
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.
        img = self.transform(img)
        return img

    def __getitem__(self, index):
        vis_id = self.datalist[index]['id']
        caption = self.datalist[index]['caption']
        vis_path = self.id2path(index)
        image = self.load_image(vis_path)

        return dict(
            vis_id=vis_id,
            image=image,
            caption=caption
        )


class VideoRetrievalCollator(object):
    def __init__(self, tokenizer, max_length=40, is_train=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train

    def collate_batch(self, batch):
        if isinstance(batch[0]["video"], torch.Tensor):
            v_collate = default_collate
        else:
            v_collate = img_collate
        video = v_collate([d["video"] for d in batch])

        text_examples = flat_list_of_lists([d["texts"] for d in batch])
        text_str_list = [d for d in text_examples]  # (B, )
        batch_enc = self.tokenizer.batch_encode_plus(
            text_str_list,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        text_input_ids = batch_enc.input_ids  # (B, L)
        text_input_mask = batch_enc.attention_mask  # (B, L)

        collated_batch = dict(
            video=video,  # [B, clips, num_frm, C, H_crop, W_crop]
            text_input_ids=text_input_ids,
            text_input_mask=text_input_mask
        )

        return collated_batch

    def collate_batch_with_auxiliary_text(self, batch):
        if isinstance(batch[0]["video"], torch.Tensor):
            v_collate = default_collate
        else:
            v_collate = img_collate
        video = v_collate([d["video"] for d in batch])

        text_examples = flat_list_of_lists([d["texts"] for d in batch])
        text_str_list = [d for d in text_examples]  # (B, )
        batch_enc = self.tokenizer.batch_encode_plus(
            text_str_list,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        text_input_ids = batch_enc.input_ids  # (B, L)
        text_input_mask = batch_enc.attention_mask  # (B, L)

        # auxiliary text
        # text_examples = flat_list_of_lists([d["aux_texts"] for d in batch])
        # text_str_list = [d for d in text_examples]  # (B, )
        bacth_enc = self._multi_encode([flat_list_of_lists(d['aux_texts']) for d in batch])
        # (B, number of captions, max_length) -> (B*number of captions, max_length)
        aux_text_input_ids = bacth_enc['input_ids'].reshape(-1, bacth_enc['input_ids'].shape[-1])
        aux_text_input_mask = bacth_enc['attention_mask'].reshape(-1, bacth_enc['attention_mask'].shape[-1])

        collated_batch = dict(
            video=video,  # [B, clips, num_frm, C, H_crop, W_crop]
            text_input_ids=text_input_ids,
            text_input_mask=text_input_mask,
            aux_text_input_ids=aux_text_input_ids,
            aux_text_input_mask=aux_text_input_mask,
        )

        return collated_batch

    def _multi_encode(self, data: List[List[str]]):
        list_of_ids = []
        list_of_attention_mask = []
        for sentences in data:
            ret = self.tokenizer.batch_encode_plus(
                sentences,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt')
            list_of_ids.append(ret['input_ids'].unsqueeze(0))
            list_of_attention_mask.append(ret['attention_mask'].unsqueeze(0))

        input_ids = torch.cat(list_of_ids, dim=0)
        attention_mask = torch.cat(list_of_attention_mask, dim=0)

        return {'input_ids': input_ids, 'attention_mask': attention_mask}


class ImageRetrievalCollator(object):
    def __init__(self, tokenizer, max_length=40, is_train=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train

    def collate_batch(self, batch):
        if isinstance(batch[0]["image"], torch.Tensor):
            v_collate = default_collate
        else:
            v_collate = img_collate
        image = v_collate([d["image"] for d in batch])
        vis_id = default_collate([d["vis_id"] for d in batch])

        # text_examples = flat_list_of_lists([d["caption"] for d in batch])
        # text_str_list = [d for d in text_examples]  # (B, )
        text_str_list = [d['caption'] for d in batch]
        batch_enc = self.tokenizer.batch_encode_plus(
            text_str_list,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        text_input_ids = batch_enc.input_ids  # (B, L)
        text_input_mask = batch_enc.attention_mask  # (B, L)

        collated_batch = dict(
            vis_id=vis_id,
            image=image,  # [B, C, H_crop, W_crop]
            text_input_ids=text_input_ids,
            text_input_mask=text_input_mask
        )

        return collated_batch


class OnlySentenceCollator(object):
    def __init__(self, tokenizer, max_length=40, is_train=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train

    def collate_batch(self, batch):

        text_examples = flat_list_of_lists([d["texts"] for d in batch])
        text_str_list = [d for d in text_examples]  # (B, )
        batch_enc = self.tokenizer.batch_encode_plus(
            text_str_list,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        text_input_ids = batch_enc.input_ids  # (B, L)
        text_input_mask = batch_enc.attention_mask  # (B, L)


        indices=default_collate([d['index'] for d in batch])

        collated_batch = dict(
            indices=indices,
            text_input_ids=text_input_ids,
            text_input_mask=text_input_mask
        )

        return collated_batch

class OnlyVideoCollator(object):
    def __init__(self, is_train=False):
        self.is_train = is_train
        pass

    def collate_batch(self, batch):
        if isinstance(batch[0]["video"], torch.Tensor):
            v_collate = default_collate
        else:
            v_collate = img_collate

        indices=default_collate([d['index'] for d in batch])
        video = v_collate([d["video"] for d in batch])

        collated_batch = dict(
            indices=indices,
            video=video,  # [B, clips, num_frm, C, H_crop, W_crop]
        )

        return collated_batch
