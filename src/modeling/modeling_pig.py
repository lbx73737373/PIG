import json
from typing import List
from collections import OrderedDict
import math

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
import transformers
from transformers import CLIPTokenizerFast
from transformers.models.clip.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
import torch.distributed as dist
import ipdb
import yaml
import numpy as np
import horovod.torch as hvd

from src.modeling.VidCLIP import VidCLIP
from src.modeling.CLIP import CLIPModel as CLIP
from src.modeling.co_attention_module import Co_attention_block
# from src.modeling.med import BertConfig, BertForMaskedLM, BertEncoder
# from transformers import BlipProcessor, BlipForConditionalGeneration

DEFAULT_TOPK_TEXT_TOKENS = 3
DEFAULT_BERT_QFORMER_CONFIG = {
    "hidden_size": 512,
    "hidden_act": "gelu",
    "initializer_range": 0.02,
    "vocab_size": 30522,
    "hidden_dropout_prob": 0.1,
    "num_attention_heads": 8,
    "type_vocab_size": 2,
    "max_position_embeddings": 512,
    "num_hidden_layers": 8,
    "intermediate_size": 2048,
    "attention_probs_dropout_prob": 0.1,
}
DEFAULT_SELF_ATTENTION_CONFIG = {
    "hidden_size": 512,
    "hidden_act": "gelu",
    "initializer_range": 0.02,
    "vocab_size": 30522,
    "hidden_dropout_prob": 0.1,
    "num_attention_heads": 8,
    "type_vocab_size": 2,
    "max_position_embeddings": 512,
    "num_hidden_layers": 4,
    "intermediate_size": 2048,
    "attention_probs_dropout_prob": 0.1,
}



class PIG(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.encoder = VidCLIP(self.cfg)

        if cfg.loss_config.loss_name == 'NCELearnableTempLoss_two_matrix':
            self.alpha = nn.Parameter(torch.tensor(0.5))

        if hasattr(cfg, 'fusioner_config') and cfg.fusioner_config.text_embeds_type == 'word-level':
            # (B, M, D) --> (B, M, 1)
            self.AFA = nn.Sequential(nn.Linear(cfg.fusioner_config.embed_dim, cfg.fusioner_config.embed_dim, bias=True),
                                     nn.SiLU(),
                                     nn.Linear(cfg.fusioner_config.embed_dim, 1, bias=True),
                                     nn.Softmax(dim=-1))

        if hasattr(cfg, 'pseudo_final_alpha_add'):
            self.final_pseudo_final_alpha_add = nn.Parameter(torch.tensor(0.1))

        # self.aux_feature_scale = nn.Parameter(torch.tensor(1.0))
        if hasattr(self.cfg, 'generator_config'):
            self.generator = FeatureReconstructor(self.cfg, self.cfg.generator_config)

        if hasattr(self.cfg, 'fusioner_config'):
            self.fusioner_type = cfg.fusioner_config.fusioner_type
            if self.fusioner_type == 'mean_pooling':
                self.fusioner = MeanPooler()
            elif self.fusioner_type == 'self_attention':
                self.fusioner = AttentionFusioner(cfg.fusioner_config)
            elif self.fusioner_type == 'XPoolCrossAttention':
                self.fusioner = XPoolCrossAttentionPooler(cfg.fusioner_config)
            elif self.fusioner_type == 'wide_cross_attention':
                self.fusioner = WideCrossAttentionPooler(cfg.fusioner_config)
            elif self.fusioner_type == 'wo_residual_wide_cross_attention':
                self.fusioner = WoResidualWideCrossAttentionPooler(cfg.fusioner_config)
            elif self.fusioner_type == 'ts2-XPoolCrossAttention':
                self.fusioner = TS2XPoolCrossAttentionPooler(cfg.fusioner_config)
            elif self.fusioner_type == 'CoAttention':
                self.fusioner = CoAttentionPooler(cfg.fusioner_config)
            else:
                raise NotImplementedError

        # self.fusioner_vision_proj = nn.Linear(cfg.generator_config.Qformer_hidden_dim, cfg.fusioner_config.embed_dim, bias=True)

    def get_clip_embeds(self, video, text_input_ids, text_input_mask, image=None, caption_ids=None, caption_masks=None):
        inputs = {"input_ids": text_input_ids,
                  "attention_mask": text_input_mask,
                  "pixel_values": video,
                  "output_attentions": True,
                  "output_hidden_states": True,
                  "return_loss": False}
        outputs = self.encoder.clipmodel.forward_wo_norm(**inputs)
        # embeds are global features: [cls] and [sep] tokens
        text_feats = outputs["text_embeds"]
        video_feats = outputs["image_embeds"]


        # 0 is all feats(last_hidden_state)
        text_all_feats = outputs["text_model_output"][0]
        video_all_feats = outputs["vision_model_output"][0]
        if hasattr(self.cfg, 'generation_loss_config') and hasattr(self.cfg.generation_loss_config, 'num_layers'):
            num_layers = self.cfg.generation_loss_config.num_layers
            # do not select last hidden state
            text_hidden_states = outputs["text_model_output"]["hidden_states"][-num_layers:-1]
            # (bs, num_layers, len_seq, hidden_dim)
            text_hidden_states = torch.stack(text_hidden_states, dim=1)
            text_hidden_states = torch.cat([text_hidden_states, text_all_feats.unsqueeze(1)], dim=1)
        else:
            text_hidden_states = text_all_feats

        vision_global_cls_attentions = outputs['vision_model_output']['attentions']
        return text_feats, video_feats, text_all_feats, video_all_feats, text_hidden_states, vision_global_cls_attentions

    def select_video_tokens(self, cfg, select_type, video_feats, attention_scores=None):
        """
        video_feats:
        [bs, 1+num_global_prompts+num_frames*(num_local_prompts+num_patches), hidden_dim]
        we fix num_local_prompts == 1
        """
        if select_type == 'patch':
            return video_feats
        elif select_type == 'added_cls+frame_cls':
            assert cfg.clip_vision_additional_config.keep_frame_cls == 1, 'dont have frame cls tokens'
            num_global_prompts = cfg.clip_vision_additional_config.add_cls_num
            num_frames = cfg.clip_vision_additional_config.temporal_size
            global_prompt_tokens = video_feats[:, 1:num_global_prompts + 1, :]
            # [bs, num_frames, num_local_prompts+num_patches, hidden_dim]
            video_frame_feats = video_feats[:, num_global_prompts + 1:, :].reshape(video_feats.shape[0], num_frames, -1,
                                                                                   video_feats.shape[-1])
            local_prompt_tokens = video_frame_feats[:, :, 0, :].reshape(video_frame_feats.shape[0], -1,
                                                                        video_frame_feats.shape[-1])
            # [bs, num_global_prompts+num_frames*num_local_prompts, hidden_dim]
            selected_video_feats = torch.cat([global_prompt_tokens, local_prompt_tokens], dim=1)

            return selected_video_feats
        elif select_type == 'cls+added_cls+frame_cls':
            assert cfg.clip_vision_additional_config.keep_frame_cls == 1, 'dont have frame cls tokens'
            num_global_prompts = cfg.clip_vision_additional_config.add_cls_num
            num_frames = cfg.clip_vision_additional_config.temporal_size
            cls_prompt_tokens = video_feats[:, 0, :]
            cls_prompt_tokens = cls_prompt_tokens.unsqueeze(1)
            global_prompt_tokens = video_feats[:, 1:num_global_prompts + 1, :]
            # [bs, num_frames, num_local_prompts+num_patches, hidden_dim]
            video_frame_feats = video_feats[:, num_global_prompts + 1:, :].reshape(video_feats.shape[0], num_frames, -1,
                                                                                   video_feats.shape[-1])
            local_prompt_tokens = video_frame_feats[:, :, 0, :].reshape(video_frame_feats.shape[0], -1,
                                                                        video_frame_feats.shape[-1])
            # [bs, num_global_prompts+num_frames*num_local_prompts, hidden_dim]
            selected_video_feats = torch.cat([cls_prompt_tokens, global_prompt_tokens, local_prompt_tokens], dim=1)

            return selected_video_feats
        elif select_type == 'cls+frame_cls':
            assert cfg.clip_vision_additional_config.keep_frame_cls == 1, 'dont have frame cls tokens'
            num_global_prompts = cfg.clip_vision_additional_config.add_cls_num
            num_frames = cfg.clip_vision_additional_config.temporal_size
            cls_prompt_tokens = video_feats[:, 0, :]
            cls_prompt_tokens = cls_prompt_tokens.unsqueeze(1)
            # [bs, num_frames, num_local_prompts+num_patches, hidden_dim]
            video_frame_feats = video_feats[:, num_global_prompts + 1:, :].reshape(video_feats.shape[0], num_frames, -1,
                                                                                   video_feats.shape[-1])
            local_prompt_tokens = video_frame_feats[:, :, 0, :].reshape(video_frame_feats.shape[0], -1,
                                                                        video_frame_feats.shape[-1])
            # [bs, num_global_prompts+num_frames*num_local_prompts, hidden_dim]
            selected_video_feats = torch.cat([cls_prompt_tokens, local_prompt_tokens], dim=1)

            return selected_video_feats
        elif select_type == 'frame_cls':
            assert cfg.clip_vision_additional_config.keep_frame_cls == 1, 'dont have frame cls tokens'
            num_global_prompts = cfg.clip_vision_additional_config.add_cls_num
            num_frames = cfg.clip_vision_additional_config.temporal_size
            # [bs, num_frames, num_local_prompts+num_patches, hidden_dim]
            video_frame_feats = video_feats[:, num_global_prompts + 1:, :].reshape(video_feats.shape[0], num_frames, -1,
                                                                                   video_feats.shape[-1])
            local_prompt_tokens = video_frame_feats[:, :, 0, :].reshape(video_frame_feats.shape[0], -1,
                                                                        video_frame_feats.shape[-1])
            # [bs, num_global_prompts+num_frames*num_local_prompts, hidden_dim]
            selected_video_feats = local_prompt_tokens

            return selected_video_feats
        elif select_type == 'cls+cls_attention_scores':
            assert attention_scores is not None, 'attention scores is None'
            # attention_scores shape in (bs, num_heads, num_queries, num_keys)
            num_global_prompts = cfg.clip_vision_additional_config.add_cls_num
            cls_prompt_tokens = video_feats[:, 0, :]
            cls_prompt_tokens = cls_prompt_tokens.unsqueeze(1)
            # reduce head dim
            if self.cfg.fusioner_config.select_attention_scores_type == 'max':
                attention_scores, _ = attention_scores.max(dim=1)
            elif self.cfg.fusioner_config.select_attention_scores_type == 'mean':
                attention_scores = attention_scores.mean(dim=1)

            # select cls tokens attention distribution at position 0
            attention_scores = attention_scores[:, 0, :]
            k = self.cfg.fusioner_config.num_selected_tokens

            # topk_indices shape in (bs, k)
            topk_values, topk_indices = torch.topk(attention_scores, k, dim=-1, largest=True, sorted=True)
            batch_indices = torch.arange(attention_scores.size(0)).unsqueeze(1).expand(-1, k).to(attention_scores.device)
            # [bs, k, hidden_dim]
            selected_video_feats = video_feats[batch_indices, topk_indices, :]

            # [bs, k+1, hidden_dim]
            selected_video_feats = torch.cat([cls_prompt_tokens, selected_video_feats], dim=1)

            return selected_video_feats
        elif select_type == 'cls+added_cls+frame_cls+cls_attention_scores':
            assert cfg.clip_vision_additional_config.keep_frame_cls == 1, 'dont have frame cls tokens'
            num_global_prompts = cfg.clip_vision_additional_config.add_cls_num
            num_frames = cfg.clip_vision_additional_config.temporal_size
            cls_prompt_tokens = video_feats[:, 0, :]
            cls_prompt_tokens = cls_prompt_tokens.unsqueeze(1)
            global_prompt_tokens = video_feats[:, 1:num_global_prompts + 1, :]
            # [bs, num_frames, num_local_prompts+num_patches, hidden_dim]
            video_frame_feats = video_feats[:, num_global_prompts + 1:, :].reshape(video_feats.shape[0], num_frames, -1,
                                                                                   video_feats.shape[-1])
            local_prompt_tokens = video_frame_feats[:, :, 0, :].reshape(video_frame_feats.shape[0], -1,
                                                                        video_frame_feats.shape[-1])


            # attention_scores shape in (bs, num_heads, num_queries, num_keys)
            assert attention_scores is not None, 'attention scores is None'

            # reduce head dim
            if self.cfg.fusioner_config.select_attention_scores_type == 'max':
                attention_scores, _ = attention_scores.max(dim=1)
            elif self.cfg.fusioner_config.select_attention_scores_type == 'mean':
                attention_scores = attention_scores.mean(dim=1)

            # select cls tokens attention distribution at position 0
            attention_scores = attention_scores[:, 0, :]
            k = self.cfg.fusioner_config.num_selected_tokens

            # topk_indices shape in (bs, k)
            topk_values, topk_indices = torch.topk(attention_scores, k, dim=-1, largest=True, sorted=True)
            batch_indices = torch.arange(attention_scores.size(0)).unsqueeze(1).expand(-1, k).to(attention_scores.device)
            # [bs, k, hidden_dim]
            attention_tokens = video_feats[batch_indices, topk_indices, :]

            selected_video_feats = torch.cat([cls_prompt_tokens, global_prompt_tokens, local_prompt_tokens, attention_tokens], dim=1)
            return selected_video_feats
        elif select_type == 'cls+added_cls+frame_cls+cls_attention_scores_no_overlap':
            assert cfg.clip_vision_additional_config.keep_frame_cls == 1, 'dont have frame cls tokens'
            num_global_prompts = cfg.clip_vision_additional_config.add_cls_num
            num_frames = cfg.clip_vision_additional_config.temporal_size

            cls_prompt_tokens = video_feats[:, 0, :].unsqueeze(1)
            global_prompt_tokens = video_feats[:, 1:num_global_prompts + 1, :]
            # [bs, num_frames, num_local_prompts+num_patches, hidden_dim]
            video_frame_feats = video_feats[:, num_global_prompts + 1:, :].reshape(
                video_feats.shape[0], num_frames, -1, video_feats.shape[-1]
            )
            local_prompt_tokens = video_frame_feats[:, :, 0, :].reshape(video_frame_feats.shape[0], -1,
                                                                        video_frame_feats.shape[-1])

            # attention_scores: shape = (bs, num_heads, num_queries, num_keys)
            assert attention_scores is not None, 'attention scores is None'

            if self.cfg.fusioner_config.select_attention_scores_type == 'max':
                attention_scores, _ = attention_scores.max(dim=1)  # [bs, num_queries, num_keys]
            elif self.cfg.fusioner_config.select_attention_scores_type == 'mean':
                attention_scores = attention_scores.mean(dim=1)  # [bs, num_queries, num_keys]

            # select cls tokens attention distribution at position 0
            attention_scores = attention_scores[:, 0, :]
            k = self.cfg.fusioner_config.num_selected_tokens
            bs, total_token_num = attention_scores.shape  # total_token_num == video_feats.shape[1]

            # num_local_prompts+num_patches
            num_tokens_per_frame = video_frame_feats.shape[2]
            # first part skipped indexes: cls_prompt_tokens + global_prompt_tokens
            skip_indices_list = list(range(num_global_prompts + 1))  # 0 到 num_global_prompts 这部分

            # second part skipped indexes: local_prompt_tokens
            for i in range(num_frames):
                # TODO: make an assumption that num_local_prompts == 1
                local_prompt_index = num_global_prompts + 1 + i * num_tokens_per_frame
                skip_indices_list.append(local_prompt_index)

            skip_indices = torch.tensor(skip_indices_list, device=attention_scores.device)

            # set attention_scores to -inf
            attention_scores.scatter_(
                dim=1,
                index=skip_indices.unsqueeze(0).expand(bs, -1),
                value=float('-inf')
            )

            # topk_indices.shape = (bs, k)
            topk_values, topk_indices = torch.topk(attention_scores, k, dim=-1, largest=True, sorted=True)
            batch_indices = torch.arange(bs, device=attention_scores.device).unsqueeze(1).expand(-1, k)

            # [bs, k, hidden_dim]
            attention_tokens = video_feats[batch_indices, topk_indices, :]

            selected_video_feats = torch.cat([
                cls_prompt_tokens,  # [bs, 1, hidden_dim]
                global_prompt_tokens,  # [bs, num_global_prompts, hidden_dim]
                local_prompt_tokens,  # [bs, num_frames, hidden_dim] (已flatten或未flatten)
                attention_tokens  # [bs, k, hidden_dim]
            ], dim=1)


            return selected_video_feats

        else:
            raise NotImplementedError

    def select_text_tokens(self, cfg, text_all_feats, text_input_mask, video_feats):
        # text_all_feats: [bs, num_words, hidden_dim]
        # text_input_mask: [bs, num_words]
        # video_feats: [bs, hidden_dim]
        k = min(DEFAULT_TOPK_TEXT_TOKENS, text_all_feats.shape[1])
        video_embeds = video_feats / video_feats.norm(dim=-1, keepdim=True)
        text_all_embeds = text_all_feats / text_all_feats.norm(dim=-1, keepdim=True)
        # [bs, num_words]
        sim_matrix = torch.bmm(text_all_embeds, video_embeds.unsqueeze(-1)).squeeze(-1)
        text_input_mask = text_input_mask.bool()
        sim_matrix[~text_input_mask] = -float('inf')

        scores, indices = torch.topk(sim_matrix, k, dim=-1)
        selected_text_feats = text_all_feats[torch.arange(text_all_feats.shape[0]).unsqueeze(-1), indices, :]
        return selected_text_feats





    def forward(self, video, text_input_ids, text_input_mask, aux_text_input_ids=None, aux_text_input_mask=None, image=None, caption_ids=None, caption_masks=None):
        text_feats, video_feats, text_all_feats, video_all_feats, text_hidden_states, vision_global_cls_attentions = self.get_clip_embeds(video, text_input_ids,
                                                                                      text_input_mask, image,
                                                                                      caption_ids, caption_masks)
        # get auxiliary text features
        if aux_text_input_ids is not None:
            inputs = {"input_ids": aux_text_input_ids,
                      "attention_mask": aux_text_input_mask}
            aux_text_feats = self.encoder.clipmodel.get_text_features(**inputs)
            aux_group_size = max(1, aux_text_feats.shape[0] // max(1, text_feats.shape[0]))
            aux_text_feats = torch.mean(aux_text_feats.reshape(-1, aux_group_size, aux_text_feats.shape[-1]), dim=1)
            # aux_text_feats = self.aux_feature_scale * aux_text_feats
        else:
            aux_text_feats = None

        # fc of clipmodel, text_feats and video_feats are also go through fc inside of self.encoder.clipmodel.forward_wo_norm call
        # only video_feats need to go through a post_layernorm
        video_all_feats = self.encoder.clipmodel.vision_model.post_layernorm(video_all_feats)
        video_all_feats = self.encoder.clipmodel.visual_projection(video_all_feats)
        text_all_feats = self.encoder.clipmodel.text_projection(text_all_feats)

        # get topk most informative text tokens
        if hasattr(self.cfg, 'generator_config'):
            selected_text_feats = self.select_text_tokens(self.cfg, text_all_feats, text_input_mask, video_feats)

            # TODO: fix this hard code, eot_feat is the last one
            selected_text_feats = torch.cat([selected_text_feats, text_feats.unsqueeze(1)], dim=1)
        else:
            selected_text_feats = text_all_feats

        pred_logits = None
        out_tokens = None

        if aux_text_feats is None:
            if hasattr(self.cfg, 'pretraining'):
                generator_selected_video_feats = self.select_video_tokens(self.cfg, self.cfg.generator_config.input_video_tokens_type, video_all_feats)
                generated_text_feats, generator_hidden_states = self.generator(generator_selected_video_feats)
                generated_global_text_feats = self.generator.get_overall_text_repr(generated_text_feats)
                final_video_feats = None
                fused_video_feats = None
                pred_logits = None
                out_tokens = None
                final_video_feats = None
                fusioner_selected_video_feats = None
            elif self.cfg.clip_vision_additional_config.type == 'CLIP4clip':
                assert self.cfg.train_num_frms == self.cfg.test_num_frms
                T = self.cfg.train_num_frms
                _, L, D = video_all_feats.shape
                video_feats = video_feats.reshape(-1, T, D)
                video_all_feats = video_all_feats.reshape(-1, T, L, D)
                # mean-pooling at time dim
                video_feats = torch.mean(video_feats, dim=1)
                video_all_feats = torch.mean(video_all_feats, dim=1)
                generator_selected_video_feats = video_all_feats
                generated_text_feats = video_all_feats
                generated_global_text_feats = video_feats
                fusioner_selected_video_feats = video_all_feats
                fused_video_feats = video_feats
                final_video_feats = video_feats
                generator_hidden_states = generated_text_feats
                pred_logits = None
                out_tokens = None
            elif self.cfg.clip_vision_additional_config.type == 'CLIP4clip+pig':
                assert self.cfg.train_num_frms == self.cfg.test_num_frms
                T = self.cfg.train_num_frms
                _, L, D = video_all_feats.shape
                video_feats_meanpooling = torch.mean(video_feats.reshape(-1, T, D), dim=1)
                video_all_feats_meanpooling = torch.mean(video_all_feats.reshape(-1, T, L, D), dim=1)
                # we use the mean-pooling of video_all_feats as the generated_selected_text_feats
                generator_selected_video_feats = video_all_feats_meanpooling
                generated_text_feats, generator_hidden_states = self.generator(generator_selected_video_feats)
                generated_global_text_feats = self.generator.get_overall_text_repr(generated_text_feats)

                if hasattr(self.cfg, 'fusioner_config'):
                    # get frame cls tokens
                    fusioner_selected_video_feats = video_feats.reshape(-1, T, D)
                    if self.cfg.fusioner_config.text_embeds_type == 'word-level':
                        _, num_words, _ = generated_text_feats.shape
                        fused_video_feats_list = []
                        for i in range(num_words):
                            generated_word_feats = generated_text_feats[:, i, :]
                            fused_video_feats, _ = self.fusioner(generated_word_feats, fusioner_selected_video_feats,
                                                                 video_mask=None)
                            fused_video_feats_list.append(fused_video_feats)
                        # fused_video_feats = torch.mean(torch.stack(fused_video_feats_list, dim=1), dim=1)
                        # B x num_words x D
                        fused_video_feats = torch.stack(fused_video_feats_list, dim=1)
                        # B x num_words x 1
                        weights = self.AFA(fused_video_feats)
                        weighted_sum_video_feats = torch.sum(weights * fused_video_feats, dim=1)
                        fused_video_feats = weighted_sum_video_feats
                    elif self.cfg.fusioner_config.text_embeds_type == 'co-attn':
                        fused_video_feats, _ = self.fusioner(generated_text_feats, fusioner_selected_video_feats,
                                                             video_mask=None)
                    else:
                        fused_video_feats, place_holder = self.fusioner(generated_global_text_feats, fusioner_selected_video_feats, video_mask=None)

                else:
                    fused_video_feats = generated_global_text_feats

                video_feats = video_feats_meanpooling
                final_video_feats = fused_video_feats

            else:
                if hasattr(self.cfg, 'generator_config'):
                    # select the attention score of the last layer
                    attention_scores = vision_global_cls_attentions[-1]
                    generator_selected_video_feats = self.select_video_tokens(self.cfg, self.cfg.generator_config.input_video_tokens_type, video_all_feats, attention_scores=attention_scores)
                    generated_text_feats, generator_hidden_states = self.generator(generator_selected_video_feats)
                    generated_global_text_feats = self.generator.get_overall_text_repr(generated_text_feats)
                else:
                    generator_hidden_states = video_all_feats
                    generator_selected_video_feats = video_all_feats
                    generated_text_feats = video_all_feats
                    generated_global_text_feats = video_feats

                if hasattr(self.cfg, 'fusioner_config'):
                    # select the attention score of the last layer
                    attention_scores = vision_global_cls_attentions[-1]
                    fusioner_selected_video_feats = self.select_video_tokens(self.cfg,
                                                                                 self.cfg.fusioner_config.input_video_tokens_type,
                                                                                 video_all_feats, attention_scores=attention_scores)

                    if self.cfg.fusioner_config.text_embeds_type == 'word-level':
                        _, num_words, _ = generated_text_feats.shape
                        fused_video_feats_list = []
                        for i in range(num_words):
                            generated_word_feats = generated_text_feats[:, i, :]
                            fused_video_feats, _ = self.fusioner(generated_word_feats, fusioner_selected_video_feats,
                                                                 video_mask=None)
                            fused_video_feats_list.append(fused_video_feats)
                        # fused_video_feats = torch.mean(torch.stack(fused_video_feats_list, dim=1), dim=1)
                        # B x num_words x D
                        fused_video_feats = torch.stack(fused_video_feats_list, dim=1)
                        # B x num_words x 1
                        weights = self.AFA(fused_video_feats)
                        weighted_sum_video_feats = torch.sum(weights * fused_video_feats, dim=1)
                        fused_video_feats = weighted_sum_video_feats
                    elif self.cfg.fusioner_config.text_embeds_type == 'co-attn':
                        fused_video_feats, _ = self.fusioner(generated_text_feats, fusioner_selected_video_feats,
                                                             video_mask=None)
                    else:
                        fused_video_feats, place_holder = self.fusioner(generated_global_text_feats, fusioner_selected_video_feats, video_mask=None)

                else:
                    fusioner_selected_video_feats = self.select_video_tokens(self.cfg,
                                                                             self.cfg.fusioner_select_video_tokens_type,
                                                                             video_all_feats)
                    fused_video_feats = generated_global_text_feats
                final_video_feats = fused_video_feats

        else:
            generator_selected_video_feats = self.select_video_tokens(self.cfg, self.cfg.generator_config.input_video_tokens_type,
                                                            video_all_feats)
            generated_text_feats = aux_text_feats
            generated_global_text_feats = aux_text_feats
            fusioner_selected_video_feats = self.select_video_tokens(self.cfg,
                                                                  self.cfg.fusioner_config.input_video_tokens_type,
                                                                  video_all_feats)
            if self.cfg.fusioner_config.text_embeds_type == 'word-level':
                _, num_words, _ = generated_text_feats.shape
                fused_video_feats_list = []
                for i in range(num_words):
                    generated_word_feats = generated_text_feats[:, i, :]
                    fused_video_feats, _ = self.fusioner(generated_word_feats, fusioner_selected_video_feats,
                                                         video_mask=None)
                    fused_video_feats_list.append(fused_video_feats)
                # fused_video_feats = torch.mean(torch.stack(fused_video_feats_list, dim=1), dim=1)
                # B x num_words x D
                fused_video_feats = torch.stack(fused_video_feats_list, dim=1)
                # B x num_words x 1
                weights = self.AFA(fused_video_feats)
                weighted_sum_video_feats = torch.sum(weights * fused_video_feats, dim=1)
                fused_video_feats = weighted_sum_video_feats
            elif self.cfg.fusioner_config.text_embeds_type == 'co-attn':
                fused_video_feats, _ = self.fusioner(generated_text_feats, fusioner_selected_video_feats, video_mask=None)
            else:
                fused_video_feats, place_holder = self.fusioner(generated_global_text_feats, fusioner_selected_video_feats,
                                                   video_mask=None)

            final_video_feats = fused_video_feats

        outputs = {
            'video_feats': video_feats,
            'video_all_feats': video_all_feats,
            'final_video_feats': final_video_feats,
            'fused_video_feats': fused_video_feats,
            'text_feats': text_feats,
            'text_all_feats': text_all_feats,
            'text_hidden_states': text_hidden_states,
            'selected_text_feats': selected_text_feats,
            'generated_text_feats': generated_text_feats,
            'generated_global_text_feats': generated_global_text_feats,
            'generator_selected_video_feats': generator_selected_video_feats,
            'generator_hidden_states': generator_hidden_states,
            'fusioner_selected_video_feats': fusioner_selected_video_feats ,
            'pred_logits': pred_logits,
            'out_tokens': out_tokens,
        }

        return outputs



class FeatureReconstructor(nn.Module):
    def __init__(self, cfg, fr_config):
        super().__init__()
        self.cfg = cfg
        self.fr_config = fr_config
        self.fr_model_type = self.fr_config.fr_model_type
        if self.fr_model_type == 'CLIP-text-encoder':
            self.num_query_token = self.fr_config.num_query_token
            self.clip_reconstructor, self.bos_tokens, self.query_tokens = FeatureReconstructor.init_CLIP_reconstructor(self.cfg)

        elif self.fr_model_type == 'Bert-Qformer':
            self.num_query_token = self.fr_config.num_query_token
            self.vision_hidden_dim = self.fr_config.vision_hidden_dim
            self.model_hidden_dim = self.fr_config.model_hidden_dim
            self.Qformer_hidden_dim = self.fr_config.Qformer_hidden_dim
            self.huggingface_model_name = getattr(self.fr_config, "huggingface_model_name", "prajjwal1/bert-medium")
            self.huggingface_model_config = DEFAULT_BERT_QFORMER_CONFIG
            self.overall_text_repr_type = self.fr_config.overall_text_repr_type
            self.input_video_tokens_type = self.fr_config.input_video_tokens_type
            # hard code for checking
            # TODO: find a better way to check
            encoder_config = BertConfig.from_dict(self.huggingface_model_config)
            assert self.Qformer_hidden_dim == encoder_config.hidden_size
            self.Qformer, self.query_tokens = FeatureReconstructor.init_Qformer(self.num_query_token,
                                                                                self.vision_hidden_dim,
                                                                                bert_config=self.huggingface_model_config,
                                                                                huggingface_model_name=self.huggingface_model_name)
            self.Qformer_proj = nn.Linear(self.Qformer_hidden_dim, self.model_hidden_dim, bias=True)
        else:
            raise NotImplementedError

    @classmethod
    # we only need a standard transformer decoder, so fix cross_attention_freq to 1
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=1, from_pretrained=True,
                     bert_config=None, huggingface_model_name=None):
        # encoder_config = BertConfig.from_pretrained(bert_config)
        encoder_config = BertConfig.from_dict(bert_config)
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        if from_pretrained:
            Qformer, msg = BertLMHeadModelForQformer.from_pretrained(
                pretrained_model_name_or_path=huggingface_model_name, config=encoder_config, output_loading_info=True
            )
            if hvd.rank() == 0:
                print('When initialize Qformer form {}, got msg: \n {}'.format(huggingface_model_name, msg))
            query_tokens = nn.Parameter(
                torch.zeros(1, num_query_token, encoder_config.hidden_size)
            )
            query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        else:
            raise NotImplementedError
        return Qformer, query_tokens

    @classmethod
    def init_CLIP_reconstructor(cls, cfg):
        clip_reconstructor = CLIPFeatureReconstructor(cfg)

        with torch.no_grad():
            tokenizer = CLIPTokenizerFast.from_pretrained(cfg.clip_config)
            bos_token_id = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.bos_token), dtype=int, device='cpu')
            eos_token_id = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.eos_token), dtype=int, device='cpu')
            bos_embeds = clip_reconstructor.embeddings.token_embedding(bos_token_id)
            eos_embeds = clip_reconstructor.embeddings.token_embedding(eos_token_id)

        return clip_reconstructor, bos_embeds, eos_embeds

    def get_overall_text_repr(self, x):
        if self.num_query_token > 1:
            if self.overall_text_repr_type == 'pick_eot':
                # pick the last token
                x = x[:, -1, :]
                x = x.squeeze(1)
            else:
                raise NotImplementedError
        else:
            x = x.squeeze(1)
        return x



    def forward(self, video_embeds):
        if self.fr_model_type == 'Bert-Qformer':
            # expand batch_size
            query_tokens = self.query_tokens.expand(video_embeds.shape[0], -1, -1)

            video_atts = torch.ones(video_embeds.size()[:-1], dtype=torch.long).to(
                video_embeds.device
            )

            # position embeds are added inside qformer
            query_outputs = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=video_embeds,
                encoder_attention_mask=video_atts,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )

            x = self.Qformer_proj(query_outputs.last_hidden_state)
            hidden_states = query_outputs.hidden_states
            if hasattr(self.cfg.generation_loss_config, 'num_layers'):
                num_layers = self.cfg.generation_loss_config.num_layers
                # (bs, num_layers, len_seq, hidden_dim)
                hidden_states = torch.stack(hidden_states[-num_layers:-1], dim=1)
                hidden_states = torch.cat([hidden_states, x.unsqueeze(1)], dim=1)
            else:
                hidden_states = x


            return x, hidden_states
        elif self.fr_model_type == 'CLIP-text-encoder':
            # [bs, hidden_dim]
            bos_tokens = self.bos_tokens.expand(video_embeds.shape[0], -1).to(video_embeds.device)
            query_tokens = self.query_tokens.expand(video_embeds.shape[0], -1).to(video_embeds.device)
            # [bs, hidden_dim]
            query_outputs = self.clip_reconstructor.generate_features(video_embeds, bos_embeds=bos_tokens, eos_embeds=query_tokens)

            hidden_states = query_outputs.hidden_states
            x = query_outputs.pooler_output

            return x, hidden_states
        else:
            raise NotImplementedError


class CLIPFeatureReconstructor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        clipconfig = CLIPConfig.from_pretrained(cfg.clip_config)
        clipmodel = CLIP.from_pretrained(cfg.clip_weights, config=clipconfig)
        self.embeddings = clipmodel.text_model.embeddings
        self.encoder = clipmodel.text_model.encoder
        self.final_layer_norm = clipmodel.text_model.final_layer_norm

        if getattr(self.cfg.generator_config, 'video_mapper_type') == 'Linear':
            self.video_mapper = nn.Linear(clipmodel.text_model.config.hidden_size, clipmodel.text_model.config.hidden_size, bias=True)
        elif getattr(self.cfg.generator_config, 'video_mapper_type') == 'MLP':
            self.video_mapper = nn.Sequential(
                nn.Linear(clipmodel.text_model.config.hidden_size, clipmodel.text_model.config.hidden_size, bias=True),
                nn.GELU(),
                nn.Linear(clipmodel.text_model.config.hidden_size, clipmodel.text_model.config.hidden_size, bias=True)
            )
        else:
            raise NotImplementedError

        del clipmodel

    def generate_features(self, video_feats, bos_embeds, eos_embeds):
        video_feats = self.video_mapper(video_feats)

        if hasattr(self.cfg, 'generator_config') and getattr(self.cfg.generator_config, 'use_bos_embeds', 0):
                inputs_embeds = torch.cat((bos_embeds.unsqueeze(1), video_feats, eos_embeds.unsqueeze(1)), dim=1)
        else:
            inputs_embeds = torch.cat((video_feats, eos_embeds.unsqueeze(1)), dim=1)

        bsz, seq_len, hidden_dim = inputs_embeds.shape
        position_ids = self.embeddings.position_ids[:, :seq_len]
        position_embeddings = self.embeddings.position_embedding(position_ids)
        hidden_states = inputs_embeds + position_embeddings

        attention_mask = None
        output_attentions = None
        output_attentions = None
        output_hidden_states = None
        return_dict = True
        if_fp16 = hidden_states.dtype == torch.float16
        causal_attention_mask = self._build_causal_attention_mask(bsz, seq_len, fp16=if_fp16).to(hidden_states.device)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # pick the last one
        pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0]), -1, :]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def _build_causal_attention_mask(self, bsz, seq_len, fp16=False):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(bsz, seq_len, seq_len)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask
        if fp16:
            mask = mask.half()
        return mask


class MeanPooler(nn.Module):
    def __init__(self):
        super().__init__()

    def mean_pooling_visual(self, visual_output, video_mask):
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out, None

    def forward(self, generated_feat, visual_output, video_mask):
        return self.mean_pooling_visual(visual_output, video_mask)


class AttentionFusioner(nn.Module):
    def __init__(self,
                 fusioner_config='./attention_fusioner.json',
                 set_position_embeddings: bool = False,
                 input_dim=None,
                 ):
        super().__init__()


        encoder_config = BertConfig.from_dict(DEFAULT_SELF_ATTENTION_CONFIG)
        self.encoder = BertEncoder(encoder_config)

        self.hidden_dim = encoder_config.hidden_size
        scale = self.hidden_dim ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(self.hidden_dim))
        self.input_dim = fusioner_config.embed_dim
        self.output_dim = encoder_config.hidden_size
        self.fc = nn.Linear(self.input_dim, self.output_dim, bias=True)
        self.LayerNorm = nn.LayerNorm(encoder_config.hidden_size, eps=encoder_config.layer_norm_eps)
        if set_position_embeddings:
            raise NotImplementedError

    def forward(self, generated_text_feat, visual_feat, video_mask):
        # visual_feat shape is (batch_size, len_seq, hidden_dim)
        # generated_text_feat shape is (batch_size, len_seq, hidden_dim)
        visual_feat = self.fc(self.LayerNorm(visual_feat))
        # data = torch.cat((self.class_embedding + torch.zeros(visual_feat.shape[0], 1, visual_feat.shape[-1]).to(
        #     self.class_embedding.device),
        #                   visual_feat, generated_text_feat), dim=1)
        data = torch.cat((self.class_embedding + torch.zeros(visual_feat.shape[0], 1, visual_feat.shape[-1]).to(
            self.class_embedding.device), visual_feat), dim=1)
        # data = visual_feat

        # encoder_output = self.encoder(visual_feat, mode=None)
        encoder_output = self.encoder(data, mode=None)
        # get cls token
        x = encoder_output.last_hidden_state
        x = x[:, 0, :]
        # x = torch.mean(x, dim=1)
        # x = encoder_output.last_hidden_state[torch.arange(x.shape[0]), torch.zeros(x.shape[0])]

        return x, encoder_output.last_hidden_state


# Do not support video mask
# class SelfAttentionMeanPooler(nn.Module):
#     def __init__(self, fusioner_config):
#         super().__init__()
#         self.fusioner_config = json.load(open(fusioner_config, 'r'))
#         self.width = self.fusioner_config['width']
#         self.layers = self.fusioner_config['layers']
#         self.heads = self.fusioner_config['heads']
#         # fixed positional_embedding length
#         self.positional_embeddings = nn.Embedding(1000, self.width)
#         self.encoder = Transformer_CLIP(self.width, self.layers, self.heads, attn_mask=None)
#
#     def forward(self, generated_text_feat, visual_output, video_mask):
#         # Sequential type: Transformer Encoder
#         if generated_text_feat is not None:
#             generated_text_feat = generated_text_feat.unsqueeze(1)
#             # concat on time dim
#             visual_output = torch.cat((generated_text_feat, visual_output), dim=1)
#
#         visual_output_original = visual_output
#
#         seq_length = visual_output.size(1)
#         position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
#         position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
#         positional_embeddings = self.positional_embeddings(position_ids)
#         visual_output = visual_output + positional_embeddings
#
#         visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND
#         visual_output = self.encoder(visual_output)
#         visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD
#         # residual add at end
#         visual_output = visual_output + visual_output_original
#
#         # mean at sequence_length dim
#         visual_output = torch.mean(visual_output, dim=1)
#         return visual_output, None

class WideMutliHeadedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, head_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.proj_dim = self.num_heads * self.head_dim

        self.q_proj = nn.Linear(self.embed_dim, self.proj_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.proj_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.proj_dim)
        self.out_proj = nn.Linear(self.proj_dim, self.embed_dim)

    def forward(self, text_embeds, video_embeds):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            o: num_vids x embed_dim
        """
        num_texts, embed_dim = text_embeds.shape
        num_vids, num_frames, embed_dim = video_embeds.shape
        # num_vids == num_texts

        q = self.q_proj(text_embeds)
        q = q.reshape(num_texts, self.num_heads, self.head_dim)
        q = q.unsqueeze(-1)

        k = self.k_proj(video_embeds)
        k = k.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        # num_vids x num_heads x num_frames x head_dim
        k = k.permute(0, 2, 1, 3)

        v = self.v_proj(video_embeds)
        v = v.reshape(num_vids, num_frames, self.num_heads, self.head_dim)

        # num_vids x num_heads x num_frames x 1
        attention_logits = k @ q
        attention_logits = attention_logits.squeeze(-1) / math.sqrt(self.head_dim)
        # num_vids x num_heads x num_frames
        attention_weights = F.softmax(attention_logits, dim=-1)

        # num_vids x num_heads x head_dim x num_frames
        v = v.permute(0, 2, 3, 1)
        # num_vids x num_heads x num_frames x 1
        attention_weights = attention_weights.unsqueeze(-1)
        # num_vids x num_heads x head_dim x 1
        attention = v @ attention_weights
        attention = attention.squeeze(-1).reshape(num_vids, self.proj_dim)

        # num_vids x embed_dim
        o = self.out_proj(attention)
        return o


class ResidualAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, head_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.attn = WideMutliHeadedAttention(embed_dim, num_heads, head_dim)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(embed_dim, embed_dim * 4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(embed_dim * 4, embed_dim))
        ]))
        self.ln_text = nn.LayerNorm(embed_dim)
        self.ln_video = nn.LayerNorm(embed_dim)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, text_embeds, video_embeds):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            o: num_vids x embed_dim
        """
        text_embeds = self.ln_text(text_embeds)
        video_embeds = self.ln_video(video_embeds)
        x = text_embeds + self.attn(text_embeds, video_embeds)
        x = x + self.mlp(self.ln(x))

        return x

class WoResidualAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, head_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.attn = WideMutliHeadedAttention(embed_dim, num_heads, head_dim)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(embed_dim, embed_dim * 4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(embed_dim * 4, embed_dim))
        ]))
        self.ln_text = nn.LayerNorm(embed_dim)
        self.ln_video = nn.LayerNorm(embed_dim)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, text_embeds, video_embeds):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            o: num_vids x embed_dim
        """
        text_embeds = self.ln_text(text_embeds)
        video_embeds = self.ln_video(video_embeds)
        x = self.attn(text_embeds, video_embeds)
        # x = text_embeds + self.attn(text_embeds, video_embeds)
        x = x + self.mlp(self.ln(x))

        return x

class WideCrossAttentionPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_layers = self.config['num_layers']
        self.embed_dim = self.config['embed_dim']
        self.num_heads = self.config['num_mha_heads']
        self.head_dim = self.config['head_dim']
        self.dropout = self.config['transformer_dropout']
        self.text_embeds_type = self.config['text_embeds_type']

        self.ln = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(self.dropout)
        self.blocks = nn.ModuleList([ResidualAttentionBlock(self.embed_dim, self.num_heads, self.head_dim)
                                     for _ in range(self.num_layers)])

        self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)

    def _init_identical_map_parameters(self):
        # Initialize q_proj weights and biases to zero so that q = 0
        for name, param in self.named_parameters():
            if 'q_proj.weight' in name or 'q_proj.bias' in name:
                nn.init.constant_(param, 0.)
            # Initialize k_proj weights and biases to zero so that k = 0
            elif 'k_proj.weight' in name or 'k_proj.bias' in name:
                nn.init.constant_(param, 0.)
            # Initialize v_proj weights to identity and biases to zero so that v = video_embeds
            elif 'v_proj.weight' in name:
                param.data.copy_(torch.zeros_like(param.data))
                for i in range(self.num_heads):
                    start = i * self.head_dim
                    end = start + self.head_dim
                    param.data[start:end, :] = torch.eye(self.head_dim, self.embed_dim)
            elif 'v_proj.bias' in name:
                nn.init.constant_(param, 0.)
            # Initialize out_proj weights to combine the heads and biases to zero
            elif 'out_proj.weight' in name:
                param.data.copy_(torch.zeros_like(param.data))
                for i in range(self.num_heads):
                    start = i * self.head_dim
                    end = start + self.head_dim
                    param.data[:, start:end] = (1 / self.num_heads) * torch.eye(self.embed_dim, self.head_dim)
            elif 'out_proj.bias' in name:
                nn.init.constant_(param, 0.)
            # Initialize MLP weights and biases to zero so that MLP outputs zero
            elif 'mlp' in name:
                nn.init.constant_(param, 0.)
            # Initialize LayerNorm weights to one and biases to zero so that LayerNorm is identity
            elif 'ln' in name:
                if 'weight' in name:
                    nn.init.constant_(param, 1.)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.)

    def forward(self, text_embeds, video_embeds, video_mask):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            out: num_vids x embed_dim
        """
        x = text_embeds
        for block in self.blocks:
            x = block(x, video_embeds)

        x = self.dropout(x)
        x = self.ln(x + text_embeds)
        return x, None


class WoResidualWideCrossAttentionPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_layers = self.config['num_layers']
        self.embed_dim = self.config['embed_dim']
        self.num_heads = self.config['num_mha_heads']
        self.head_dim = self.config['head_dim']
        self.dropout = self.config['transformer_dropout']
        self.text_embeds_type = self.config['text_embeds_type']

        self.ln = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(self.dropout)
        self.final_mlp = nn.Linear(self.embed_dim, self.embed_dim)
        self.blocks = nn.ModuleList([WoResidualAttentionBlock(self.embed_dim, self.num_heads, self.head_dim)
                                     for _ in range(self.num_layers)])

        self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)

    def _init_zero_map_parameters(self):
        for name, param in self.named_parameters():
            if 'final_mlp.weight' in name:
                nn.init.constant_(param, 0.0)
            elif 'final_mlp.bias' in name:
                param.data.fill_(0.0)

    def forward(self, text_embeds, video_embeds, video_mask):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            out: num_vids x embed_dim
        """
        x = text_embeds
        for block in self.blocks:
            x = block(x, video_embeds)

        x = self.dropout(x)
        x = self.ln(x)
        # x = self.ln(x + text_embeds)
        x = self.final_mlp(x)
        return x, None


class XPoolMultiHeadedAttention(nn.Module):
    def __init__(self, config):
        super(XPoolMultiHeadedAttention, self).__init__()
        self.embed_dim = config['embed_dim']
        self.num_heads = config['num_mha_heads']
        self.attn_temp = config['attn_temp']
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, text_embeds, video_embeds):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            o: num_vids x num_texts x embed_dim
        """
        num_texts, _ = text_embeds.shape
        # num_texts x embed_dim
        q = self.q_proj(text_embeds)
        q = q.reshape(num_texts, self.num_heads, self.head_dim)
        # num_heads x head_dim x num_texts
        q = q.permute(1, 2, 0)

        num_vids, num_frames, _ = video_embeds.shape
        # num_vids x num_frames x embed_dim
        k = self.k_proj(video_embeds)
        k = k.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        # num_vids x num_heads x num_frames x head_dim
        k = k.permute(0, 2, 1, 3)

        # num_vids x num_frames x embed_dim
        v = self.v_proj(video_embeds)
        v = v.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        # num_vids x num_heads x head_dim x num_frames
        v = v.permute(0, 2, 3, 1)

        # num_vids x num_heads x num_frames x num_texts
        attention_logits = k @ q
        attention_logits = attention_logits / math.sqrt(self.head_dim)
        attention_logits = attention_logits / self.attn_temp
        attention_weights = F.softmax(attention_logits, dim=2)

        # num_vids x num_heads x head_dim x num_texts
        attention = v @ attention_weights
        # num_vids x num_texts x num_heads x head_dim
        attention = attention.permute(0, 3, 1, 2)
        attention = attention.reshape(num_vids, num_texts, self.embed_dim)

        # num_vids x num_texts x embed_dim
        o = self.out_proj(attention)
        return o


class XPoolCrossAttentionPooler(nn.Module):
    def __init__(self, config):
        super(XPoolCrossAttentionPooler, self).__init__()
        self.config = config
        self.embed_dim = self.config.embed_dim
        dropout = self.config.transformer_dropout

        self.cross_attn = XPoolMultiHeadedAttention(self.config)

        self.linear_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.layer_norm3 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)

    def _init_zero_map_parameters(self):
        for name, param in self.named_parameters():
            if 'linear_proj.weight' or 'out_proj.weight' in name:
                nn.init.constant_(param, 0.0)
            elif 'linear_proj.bias' or 'out_proj.bias' in name:
                param.data.fill_(0.0)

    def forward(self, text_embeds, video_embeds, video_mask):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            out: num_vids x num_texts x embed_dim
        """
        text_embeds = self.layer_norm1(text_embeds)
        video_embeds = self.layer_norm1(video_embeds)

        # num_vids x num_texts x embed_dim
        attn_out = self.cross_attn(text_embeds, video_embeds)
        attn_out = self.layer_norm2(attn_out)

        linear_out = self.linear_proj(attn_out)
        out = attn_out + self.dropout(linear_out)
        out = self.layer_norm3(out)

        # only get positive pairs
        diagonal_out = torch.diagonal(out, dim1=0, dim2=1).permute(1, 0)

        return diagonal_out, out

class TS2XPoolCrossAttentionPooler(nn.Module):
    def __init__(self, config):
        super(TS2XPoolCrossAttentionPooler, self).__init__()
        self.config = config
        self.embed_dim = self.config.embed_dim
        dropout = self.config.transformer_dropout

        self.pre_mlp = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim, bias=True), nn.ReLU(), nn.Linear(self.embed_dim, self.embed_dim, bias=True))
        self.post_mlp = nn.Sequential(nn.Linear(self.embed_dim * 2, self.embed_dim, bias=True), nn.ReLU(), nn.Linear(self.embed_dim, self.embed_dim, bias=True))
        self.cross_attn = XPoolMultiHeadedAttention(self.config)

        self.linear_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.layer_norm3 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)

    def _init_zero_map_parameters(self):
        for name, param in self.named_parameters():
            if 'linear_proj.weight' or 'out_proj.weight' in name:
                nn.init.constant_(param, 0.0)
            elif 'linear_proj.bias' or 'out_proj.bias' in name:
                param.data.fill_(0.0)

    def forward(self, text_embeds, video_embeds, video_mask):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            out: num_vids x num_texts x embed_dim
        """
        video_embeds = self.pre_mlp(video_embeds)
        video_cls = video_embeds[:, 0, :]
        video_embeds = video_embeds[:, 1:, :]
        video_cls = video_cls.unsqueeze(1).expand(-1, video_embeds.shape[1], -1)
        video_embeds = torch.cat([video_cls, video_embeds], dim=-1)
        video_embeds = self.post_mlp(video_embeds)

        text_embeds = self.layer_norm1(text_embeds)
        video_embeds = self.layer_norm1(video_embeds)

        # num_vids x num_texts x embed_dim
        attn_out = self.cross_attn(text_embeds, video_embeds)
        attn_out = self.layer_norm2(attn_out)

        linear_out = self.linear_proj(attn_out)
        out = attn_out + self.dropout(linear_out)
        out = self.layer_norm3(out)

        # only get positive pairs
        diagonal_out = torch.diagonal(out, dim1=0, dim2=1).permute(1, 0)

        return diagonal_out, out

class CoAttentionPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = self.config.embed_dim
        self.num_heads = self.config.num_mha_heads
        self.dropout = 0.1
        self.num_layers = self.config.num_layers
        self.final_pooler_type = "MeanPooling"
        self.final_pooler_input = "video"

        self.co_attn_blocks = nn.ModuleList([Co_attention_block(hidden_size=self.hidden_dim, num_attention_heads=self.num_heads, dropout_rate=self.dropout) for _ in range(self.num_layers)])

        # Keep standard initialization path for stable checkpoint compatibility.


    def init_identity_weights(self):
        """
        对内部权重进行初始化，使得在初始状态下 video branch 尽量接近恒等映射。
        简要策略：
        1. 将和 video 特征处理相关的线性层初始化为接近恒等或零变换，不改变输入。
        2. 将和 text 特征处理相关的权重初始化为零，使 cross-attention 在初始时不起作用。
        """

        for block in self.co_attn_blocks:
            # BertBiAttention 部分
            # 对于 video 侧的线性层 (query1, key1, value1)，尽量初始化为近似恒等映射
            # 对于 text 侧线性层 (query2, key2, value2)，将其初始化为零，削弱text对video的影响
            # Linear层是 (out_features, in_features)
            attn = block.biattention

            # video侧Q,K,V初始化为近似恒等：权重为单位矩阵，偏置为0
            nn.init.eye_(attn.query1.weight)
            nn.init.zeros_(attn.query1.bias)
            nn.init.eye_(attn.key1.weight)
            nn.init.zeros_(attn.key1.bias)
            nn.init.eye_(attn.value1.weight)
            nn.init.zeros_(attn.value1.bias)

            # text侧Q,K,V初始化为0，使得起初 cross-attention 对video无影响
            nn.init.zeros_(attn.query2.weight)
            nn.init.zeros_(attn.query2.bias)
            nn.init.zeros_(attn.key2.weight)
            nn.init.zeros_(attn.key2.bias)
            nn.init.zeros_(attn.value2.weight)
            nn.init.zeros_(attn.value2.bias)

            # BertBiOutput 部分
            # 将 dense1, dense2 初始化为零映射，使不改变输入
            # LayerNorm默认初始化为 weight=1, bias=0，这里不用改变，虽然会有归一化
            out = block.biOutput
            nn.init.zeros_(out.dense1.weight)
            nn.init.zeros_(out.dense1.bias)
            nn.init.zeros_(out.dense2.weight)
            nn.init.zeros_(out.dense2.bias)

            # v_intermediate 和 v_output 部分对video有两层线性+激活
            # 为了减小其影响，将 v_intermediate.dense 初始化为近零，使激活输出接近0
            # 然后 v_output.dense 也初始化为0，这样最终 v_output输出 ~ LN(vision_attention_output)
            vi = block.v_intermediate
            nn.init.zeros_(vi.dense.weight)
            nn.init.zeros_(vi.dense.bias)

            vo = block.v_output
            nn.init.zeros_(vo.dense.weight)
            nn.init.zeros_(vo.dense.bias)

            # text侧同理全设为0，防止 text 对 video 输出有影响（尽管text最终输出可能无关）
            ti = block.t_intermediate
            nn.init.zeros_(ti.dense.weight)
            nn.init.zeros_(ti.dense.bias)

            to = block.t_output
            nn.init.zeros_(to.dense.weight)
            nn.init.zeros_(to.dense.bias)


    def forward(self, video_feats, text_feats, video_mask=None, text_mask=None):
        # video_feats: [bs, num_tokens, hidden_dim]
        # text_feats: [bs, num_words, hidden_dim]
        # video_masks: [bs, num_tokens]
        # text_masks: [bs, num_words]
        if video_mask is None:
            video_mask = torch.ones(video_feats.size()[:-1], dtype=torch.long).to(video_feats.device)
            video_mask = video_mask.reshape(video_mask.size(0), 1, 1, video_mask.size(-1))
        if text_mask is None:
            text_mask = torch.ones(text_feats.size()[:-1], dtype=torch.long).to(text_feats.device)
            text_mask = text_mask.reshape(text_mask.size(0), 1, 1, text_mask.size(-1))


        for layer in self.co_attn_blocks:
            video_feats, text_feats, co_attn_probs = layer(video_feats, video_mask, text_feats, text_mask)

        if self.final_pooler_input == 'video':
            x = video_feats
        elif self.final_pooler_input == 'text':
            x = text_feats
        else:
            raise NotImplementedError

        if self.final_pooler_type == 'MeanPooling':
            x = torch.mean(x, dim=1)
        else:
            raise NotImplementedError

        return x, None


def compute_token_overlap(
        topk_indices: torch.Tensor,
        video_feats: torch.Tensor,
        num_global_prompts: int,
        num_frames: int,
):
    """
    计算在一个 batch 内, 基于 attention_scores 选出的 topk token (即 topk_indices),
    与下列几种 token 索引集合的重合情况:
      1) cls_prompt_tokens
      2) global_prompt_tokens
      3) local_prompt_tokens (每个 frame 的第一个 token)
      4) 所有 frame token

    Args:
        topk_indices: [batch_size, k]
                     根据 attention_scores.topk(...) 得到的索引。
        video_feats: [batch_size, total_tokens, hidden_dim]
                     原始的所有 token 特征，主要用它的 shape 来推断各种索引边界。
        num_global_prompts: int
                     global_prompt_tokens 的数量。
        num_frames: int
                     视频帧数，即 temporal_size。

    Returns:
        overlap_dict: dict
            {
              'cls_overlap':          Tensor(shape=[batch_size]),
              'global_overlap':       Tensor(shape=[batch_size]),
              'local_overlap':        Tensor(shape=[batch_size]),
              'frame_token_overlap':  Tensor(shape=[batch_size])
            }
            每个部分的返回值都是 [batch_size]，表示对每个样本，有多少个 topk 索引落在对应的集合里。
    """

    """
    ------------------------------------------------------------
    根据你的代码逻辑：
      video_feats 的维度是 [BS, total_tokens, HDIM]，其中:
        - 第 0 号索引: cls_prompt_tokens
        - 第 [1 ... num_global_prompts] 索引: global_prompt_tokens
        - 第 [num_global_prompts+1 ... ] 索引: frame token
          将其 reshape 之后才得到 (num_frames, per_frame_tokens)，
          而本示例中我们只需要知道原始 flatten 后的索引范围。

      local_prompt_tokens:
        - 在你的代码里: 
          video_frame_feats = video_feats[:, num_global_prompts + 1:, :].reshape(
                                BS, num_frames, -1, HDIM
                             )
          local_prompt_tokens = video_frame_feats[:, :, 0, :]
          表示每个 frame 的第一个 token。

        因此，原始索引可以这样计算:
          per_frame_tokens = (total_tokens - (num_global_prompts + 1)) // num_frames
          local_prompt_idx[i] = num_global_prompts + 1 + i * per_frame_tokens
          (其中 i = 0..(num_frames-1))

    ------------------------------------------------------------
    """

    batch_size, total_tokens, _ = video_feats.shape
    device = topk_indices.device

    # ---- 1) cls_prompt_tokens 索引集合 (只有一个: index=0) ----
    cls_indices = torch.tensor([0], device=device)  # [1]

    # ---- 2) global_prompt_tokens 索引集合 [1..num_global_prompts] ----
    global_indices = torch.arange(1, num_global_prompts + 1, device=device)  # [num_global_prompts]

    # ---- 3) local_prompt_tokens: 每个 frame 的第一个 token ----
    per_frame_tokens = (total_tokens - (num_global_prompts + 1)) // num_frames
    local_indices = []
    for frame_i in range(num_frames):
        # 每个 frame 的第一个 token 在原始 flatten 里的位置
        idx = num_global_prompts + 1 + frame_i * per_frame_tokens
        local_indices.append(idx)
    local_indices = torch.tensor(local_indices, device=device)  # [num_frames]

    # ---- 4) frame token (包括所有 frame 的所有 token) ----
    frame_indices = torch.arange(
        num_global_prompts + 1, total_tokens, device=device
    )

    # ============= 计算重合度 ==============
    # 注意：topk_indices 形状是 [BS, k]，
    # 我们要判断 topk_indices 与这些集合是否有交集，然后进行计数。
    # 做法：把每个集合都视作要对比的“白名单”索引 set，然后比较 topk_indices 的元素是否在这其中。

    # (a) CLS overlap
    #   topk_indices.unsqueeze(-1).eq(cls_indices) => [BS, k, 1] -> bool
    #   any(-1) => [BS, k] -> 这一行表示 k 个位置里是否和 cls_indices 相等
    #   sum(dim=1) => [BS], 表示每条样本 topk 内与 cls_indices 相等的总数
    cls_overlap = topk_indices.unsqueeze(-1).eq(cls_indices).any(-1).sum(dim=1)

    # (b) GLOBAL overlap
    global_overlap = topk_indices.unsqueeze(-1).eq(global_indices).any(-1).sum(dim=1)

    # (c) LOCAL overlap
    local_overlap = topk_indices.unsqueeze(-1).eq(local_indices).any(-1).sum(dim=1)

    # (d) FRAME overlap
    frame_token_overlap = topk_indices.unsqueeze(-1).eq(frame_indices).any(-1).sum(dim=1)

    overlap_dict = {
        'cls_overlap': cls_overlap,  # [BS]
        'global_overlap': global_overlap,  # [BS]
        'local_overlap': local_overlap,  # [BS]
        'frame_token_overlap': frame_token_overlap,  # [BS]
    }

    return overlap_dict
