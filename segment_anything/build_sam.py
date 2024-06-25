# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial
from pathlib import Path
import urllib.request
import torch
import torch.nn.init as init
from torch.nn import functional as F
from .modeling import (
    ImageEncoderViT,
    MaskDecoder,
    PromptEncoder,
    Sam,
    TwoWayTransformer,
)


def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 256#****************
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    flag_eval=True   
    if not flag_eval:
        if checkpoint is not None:
            checkpoint = Path(checkpoint)
            
            #把除了expert以外的其他权重加载medsam的权重，expert分别加载4个权重
            print('*******interpolate')
            with open(checkpoint, "rb") as f:
                state_dict  = torch.load(f, map_location=torch.device('cpu'))
            new_state_dict = {}
            for k,v in state_dict.items(): 
                if k.find('mlp')==-1 and k.find('iou_prediction_head')==-1:
                    new_state_dict[k] = v
                
            new_state_dict = load_from(sam, new_state_dict, image_size, vit_patch_size)   
            #分别加载4个expert的权重
            experts_weights_list=[]
            experts_weights_list.append('/notebooks/MedSAM/work_dir/BraTS2021_endos_der_CTSpine1K-20240306-0843/epoch.pth')
            experts_weights_list.append('/notebooks/MedSAM/work_dir/BraTS2021_endos_der_mr-20240223-0703/epoch.pth')
            experts_weights_list.append('/notebooks/MedSAM/work_dir/BraTS2021_endos_der_endoscopy-20240223-0900/epoch.pth')
            experts_weights_list.append('/notebooks/MedSAM/work_dir/BraTS2021_endos_der_dermoscopy-20240223-1348/epoch.pth')
            # experts_weights_list.append('/notebooks//MedSAM/work_dir/MedSAM/medsam_vit_b.pth')
            # experts_weights_list.append('/notebooks//MedSAM/work_dir/MedSAM/medsam_vit_b.pth')
            # experts_weights_list.append('/notebooks//MedSAM/work_dir/MedSAM/medsam_vit_b.pth')
            # experts_weights_list.append('/notebooks//MedSAM/work_dir/MedSAM/medsam_vit_b.pth')
            for i,experts_weights in enumerate(experts_weights_list):
                with open(experts_weights, "rb") as f:
                    state_dict  = torch.load(f, map_location=torch.device('cpu'))
                for k,v in state_dict["model"].items(): #["model"]
                    if k.find('encoder')!=-1 and k.find('mlp')!=-1:
                        key=k.replace('module.', '').replace('mlp', 'mlp.experts.'+str(i)).replace('lin1.', '0.').replace('lin2.', '2.')
                        new_state_dict[key] = v
                    if k.find('mask_decoder.transformer')!=-1 and k.find('mlp')!=-1:
                        key=k.replace('module.', '').replace('mlp', 'mlp.experts.'+str(i)).replace('lin1.', '0.').replace('lin2.', '2.')
                        new_state_dict[key] = v
                    if k.find('mask_decoder.output_hypernetworks_mlps')!=-1 and k.find('mlp')!=-1:
                        key=k.replace('module.', '').replace('layers', 'experts.'+str(i)+'.layers')
                        new_state_dict[key] = v
                    if k.find('iou_prediction_head')!=-1 :
                        key=k.replace('module.', '').replace('layers', 'experts.'+str(i)+'.layers')
                        new_state_dict[key] = v
            sam.load_state_dict(new_state_dict, strict=False) #
            print("loading finish!")
        else:
            print("No weight!")
    else:
        print("eval_weight!")
            # 在加载权重前打印当前模型的参数名称
            # print("Before loading weights:", list(sam.state_dict().keys()))
        print(checkpoint)
        with open(checkpoint, "rb") as f:
                state_dict = torch.load(f, map_location=torch.device('cpu'))
        print("pre_loading weights:\n")
        new_state_dict = {}
        for k,v in state_dict["model"].items(): #
            key=k.replace('module.', '')
            new_state_dict[key] = v
        print("weights transformation success!!")
        sam.load_state_dict(new_state_dict)#, strict=False
        print("loading finish!")
    return sam

def load_from(sam, state_dicts, image_size, vit_patch_size):

    sam_dict = sam.state_dict()
    except_keys = ['mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head']
    new_state_dict = {k: v for k, v in state_dicts.items() if
                      k in sam_dict.keys() }#and except_keys[0] not in k and except_keys[1] not in k and except_keys[2] not in k
    pos_embed = new_state_dict['image_encoder.pos_embed']
    token_size = int(image_size // vit_patch_size)
    if pos_embed.shape[1] != token_size:
        # print(pos_embed.shape[1],token_size)
        # resize pos embedding, which may sacrifice the performance, but I have no better idea
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
        new_state_dict['image_encoder.pos_embed'] = pos_embed
        rel_pos_keys = [k for k in sam_dict.keys() if 'rel_pos' in k]
        for k in rel_pos_keys:
            h_check, w_check = sam_dict[k].shape
            rel_pos_params = new_state_dict[k]
            h, w = rel_pos_params.shape
            rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
            if h != h_check or w != w_check:
                rel_pos_params = F.interpolate(rel_pos_params, (h_check, w_check), mode='bilinear', align_corners=False)

            new_state_dict[k] = rel_pos_params[0, 0, ...]
    return new_state_dict