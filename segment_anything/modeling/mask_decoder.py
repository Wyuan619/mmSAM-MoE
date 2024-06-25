# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor, nn
from typing import List, Tuple, Type
import math
from .common import LayerNorm2d




class MoEFFN(nn.Module):
    def __init__(self, dim, hidden_dim,out_dim,num_experts):
        super(MoEFFN, self).__init__()
        self.num_experts=num_experts
        self.out_dim=out_dim #nn.Linear(dim, out_dim*num_experts)
        self.gating_network =nn.Sequential(
            # nn.Linear(dim, dim),
            # nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, out_dim*num_experts)
        )
        self.experts = nn.ModuleList([MLP(dim, hidden_dim, out_dim, 3) for _ in range(num_experts)])
        

    def forward(self, x,modality):
        # #gate第一部分gating_network
        gate_weights = self.gating_network(x)#,x,x
        batch_size,d=gate_weights.shape
        gate_weights = gate_weights.reshape (self.num_experts,batch_size,self.out_dim)
        gate_weights = torch.nn.functional.softmax(gate_weights, dim=-1)
        #gate第二部分 先验（1000,0100..)
        mask_weights=[]
        for i in range(self.num_experts):
            # 创建一个形状与 x 相同的掩码张量
            mask = torch.zeros(batch_size,self.out_dim)
            # 将 modality 中为 i 的位置对应的张量的那几维设置为 1
            for j in range(len(modality)):
                if modality[j] == i:
                    mask[j] = 1
            mask_weights.append(mask)
        mask_weights = torch.stack(mask_weights, dim=0).cuda()
        #合并gate
        c=torch.tensor(0.1).cuda() 
        final_weights=c*mask_weights+(1-c)*gate_weights
        # Get outputs from all experts
        outputs = [expert(x,modality) for expert in self.experts]
        outputs = torch.stack(outputs, dim=0)
        outputs = (final_weights * outputs).sum(dim=0)
        return outputs
    
class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        # self.output_hypernetworks_mlps = nn.ModuleList(
        #     [
        #         MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        #         for i in range(self.num_mask_tokens)
        #     ]
        # )
        # self.iou_prediction_head = MLP(
        #     transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        # )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MoEFFN(dim=transformer_dim,hidden_dim=transformer_dim,out_dim=transformer_dim // 8,num_experts=4)
                for i in range(self.num_mask_tokens)
            ]
        )
        self.iou_prediction_head=MoEFFN(
            dim=transformer_dim, 
            hidden_dim=iou_head_hidden_dim, 
            out_dim=self.num_mask_tokens,
            num_experts=4
        )

        

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        modality: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            modality=modality,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        modality: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        # print("sparse_prompt_embeddings:"+str(sparse_prompt_embeddings.shape))
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        # Expand per-image data in batch direction to be per-mask
        if image_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        src = src + dense_prompt_embeddings
        
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape
        # print("src.shape:"+str(src.shape))
        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens,modality)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        # print("upscaled_embedding.shape:"+str(upscaled_embedding.shape))
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :],modality)
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        # print(masks)
        # self.act_layer = nn.Tanh()
        # probabilities = self.act_layer(masks)
        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out,modality)
        
        return masks, iou_pred

        # print("&&&&&&&")
        # print(probabilities)
        # file_path1 = "123450.txt"
        # file_path2 = "345670.txt"
        # # 使用 torch.save 保存张量到文件
        # # 打开文件以写入
        # with open(file_path1, 'w') as file:
        #     for item in masks.tolist():
        #         file.write("%s\n" % item)
        # # 打开文件以写入
        # with open(file_path2, 'w') as file:
        #     for item in probabilities.tolist():
        #         file.write("%s\n" % item)


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x,modality):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
