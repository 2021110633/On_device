#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from functools import partial
from types import SimpleNamespace
from typing import Dict

import torch
import torch.nn as nn

# 导入自定义模块
from models.helpers import (EinOpsRearrange, LearnableLogitScaling, Normalize,
                            SelectElement, SelectEOSAndProject)
from models.multimodal_preprocessors import (AudioPreprocessor,
                                             IMUPreprocessor, PadIm2Video,
                                             PatchEmbedGeneric,
                                             RGBDTPreprocessor,
                                             SpatioTemporalPosEmbeddingHelper,
                                             TextPreprocessor,
                                             ThermalPreprocessor)
from models.transformer import MultiheadAttention, SimpleTransformer

# 定义模态类型常量
ModalityType = SimpleNamespace(
    VISION="vision",  # 视觉模态
    TEXT="text",  # 文本模态
    AUDIO="audio",  # 音频模态
    THERMAL="thermal",  # 热成像模态
    DEPTH="depth",  # 深度图模态
    IMU="imu",  # 惯性测量单元模态
)


class ImageBindModel(nn.Module):
    """ImageBind模型：支持多种模态的多模态融合模型"""

    def __init__(
            self,
            video_frames=2,  # 视频帧数
            kernel_size=(2, 14, 14),  # 视觉卷积核大小
            audio_kernel_size=16,  # 音频卷积核大小
            audio_stride=10,  # 音频卷积步长
            out_embed_dim=768,  # 输出嵌入维度
            vision_embed_dim=1024,  # 视觉嵌入维度
            vision_num_blocks=24,  # 视觉Transformer块数
            vision_num_heads=16,  # 视觉注意力头数
            audio_embed_dim=768,  # 音频嵌入维度
            audio_num_blocks=12,  # 音频Transformer块数
            audio_num_heads=12,  # 音频注意力头数
            audio_num_mel_bins=128,  # 音频梅尔频谱bin数
            audio_target_len=204,  # 音频目标长度
            audio_drop_path=0.1,  # 音频drop path率
            text_embed_dim=768,  # 文本嵌入维度
            text_num_blocks=12,  # 文本Transformer块数
            text_num_heads=12,  # 文本注意力头数
            depth_embed_dim=384,  # 深度图嵌入维度
            depth_kernel_size=16,  # 深度图卷积核大小
            depth_num_blocks=12,  # 深度图Transformer块数
            depth_num_heads=8,  # 深度图注意力头数
            depth_drop_path=0.0,  # 深度图drop path率
            thermal_embed_dim=768,  # 热成像嵌入维度
            thermal_kernel_size=16,  # 热成像卷积核大小
            thermal_num_blocks=12,  # 热成像Transformer块数
            thermal_num_heads=12,  # 热成像注意力头数
            thermal_drop_path=0.0,  # 热成像drop path率
            imu_embed_dim=512,  # IMU嵌入维度
            imu_kernel_size=8,  # IMU卷积核大小
            imu_num_blocks=0,  # IMU Transformer块数
            imu_num_heads=8,  # IMU注意力头数
            imu_drop_path=0.7,  # IMU drop path率
    ):
        super().__init__()

        # 创建模态预处理器
        self.modality_preprocessors = self._create_modality_preprocessors(
            video_frames,
            vision_embed_dim,
            kernel_size,
            text_embed_dim,
            audio_embed_dim,
            audio_kernel_size,
            audio_stride,
            audio_num_mel_bins,
            audio_target_len,
            depth_embed_dim,
            depth_kernel_size,
            thermal_embed_dim,
            thermal_kernel_size,
            imu_embed_dim,
        )

        # 创建模态主干网络（Transformer）
        self.modality_trunks = self._create_modality_trunks(
            vision_embed_dim,
            vision_num_blocks,
            vision_num_heads,
            text_embed_dim,
            text_num_blocks,
            text_num_heads,
            audio_embed_dim,
            audio_num_blocks,
            audio_num_heads,
            audio_drop_path,
            depth_embed_dim,
            depth_num_blocks,
            depth_num_heads,
            depth_drop_path,
            thermal_embed_dim,
            thermal_num_blocks,
            thermal_num_heads,
            thermal_drop_path,
            imu_embed_dim,
            imu_num_blocks,
            imu_num_heads,
            imu_drop_path,
        )

        # 创建模态头部网络
        self.modality_heads = self._create_modality_heads(
            out_embed_dim,
            vision_embed_dim,
            text_embed_dim,
            audio_embed_dim,
            depth_embed_dim,
            thermal_embed_dim,
            imu_embed_dim,
        )

        # 创建模态后处理器
        self.modality_postprocessors = self._create_modality_postprocessors(
            out_embed_dim
        )

    def _create_modality_preprocessors(
            self,
            video_frames=2,
            vision_embed_dim=1024,
            kernel_size=(2, 14, 14),
            text_embed_dim=768,
            audio_embed_dim=768,
            audio_kernel_size=16,
            audio_stride=10,
            audio_num_mel_bins=128,
            audio_target_len=204,
            depth_embed_dim=768,
            depth_kernel_size=16,
            thermal_embed_dim=768,
            thermal_kernel_size=16,
            imu_embed_dim=512,
    ):
        """创建各种模态的预处理器"""

        # 视觉（RGB-T）预处理器
        rgbt_stem = PatchEmbedGeneric(
            proj_stem=[
                PadIm2Video(pad_type="repeat", ntimes=2),  # 将图像填充为视频格式
                nn.Conv3d(  # 3D卷积处理视频
                    in_channels=3,
                    kernel_size=kernel_size,
                    out_channels=vision_embed_dim,
                    stride=kernel_size,
                    bias=False,
                ),
            ]
        )
        rgbt_preprocessor = RGBDTPreprocessor(
            img_size=[3, video_frames, 224, 224],  # 输入图像尺寸
            num_cls_tokens=1,  # CLS token数量
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),  # 时空位置编码
            rgbt_stem=rgbt_stem,
            depth_stem=None,
        )

        # 文本预处理器
        text_preprocessor = TextPreprocessor(
            context_length=77,  # 上下文长度
            vocab_size=49408,  # 词汇表大小
            embed_dim=text_embed_dim,  # 嵌入维度
            causal_masking=True,  # 是否使用因果掩码
        )

        # 音频预处理器
        audio_stem = PatchEmbedGeneric(
            proj_stem=[
                nn.Conv2d(  # 2D卷积处理音频频谱图
                    in_channels=1,
                    kernel_size=audio_kernel_size,
                    stride=audio_stride,
                    out_channels=audio_embed_dim,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=audio_embed_dim),  # 层归一化
        )
        audio_preprocessor = AudioPreprocessor(
            img_size=[1, audio_num_mel_bins, audio_target_len],  # 音频频谱图尺寸
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            audio_stem=audio_stem,
        )

        # 深度图预处理器
        depth_stem = PatchEmbedGeneric(
            [
                nn.Conv2d(  # 2D卷积处理深度图
                    kernel_size=depth_kernel_size,
                    in_channels=1,
                    out_channels=depth_embed_dim,
                    stride=depth_kernel_size,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=depth_embed_dim),
        )

        depth_preprocessor = RGBDTPreprocessor(
            img_size=[1, 224, 224],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            rgbt_stem=None,
            depth_stem=depth_stem,
        )

        # 热成像预处理器
        thermal_stem = PatchEmbedGeneric(
            [
                nn.Conv2d(  # 2D卷积处理热成像图
                    kernel_size=thermal_kernel_size,
                    in_channels=1,
                    out_channels=thermal_embed_dim,
                    stride=thermal_kernel_size,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=thermal_embed_dim),
        )
        thermal_preprocessor = ThermalPreprocessor(
            img_size=[1, 224, 224],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            thermal_stem=thermal_stem,
        )

        # IMU预处理器
        imu_stem = PatchEmbedGeneric(
            [
                nn.Linear(  # 线性层处理IMU数据
                    in_features=48,
                    out_features=imu_embed_dim,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=imu_embed_dim),
        )

        imu_preprocessor = IMUPreprocessor(
            img_size=[6, 2000],  # IMU数据维度
            num_cls_tokens=1,
            kernel_size=8,
            embed_dim=imu_embed_dim,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            imu_stem=imu_stem,
        )

        # 将所有预处理器组织成ModuleDict
        modality_preprocessors = {
            ModalityType.VISION: rgbt_preprocessor,
            ModalityType.TEXT: text_preprocessor,
            ModalityType.AUDIO: audio_preprocessor,
            ModalityType.DEPTH: depth_preprocessor,
            ModalityType.THERMAL: thermal_preprocessor,
            ModalityType.IMU: imu_preprocessor,
        }

        return nn.ModuleDict(modality_preprocessors)

    def _create_modality_trunks(
            self,
            vision_embed_dim=1024,
            vision_num_blocks=24,
            vision_num_heads=16,
            text_embed_dim=768,
            text_num_blocks=12,
            text_num_heads=12,
            audio_embed_dim=768,
            audio_num_blocks=12,
            audio_num_heads=12,
            audio_drop_path=0.0,
            depth_embed_dim=768,
            depth_num_blocks=12,
            depth_num_heads=12,
            depth_drop_path=0.0,
            thermal_embed_dim=768,
            thermal_num_blocks=12,
            thermal_num_heads=12,
            thermal_drop_path=0.0,
            imu_embed_dim=512,
            imu_num_blocks=6,
            imu_num_heads=8,
            imu_drop_path=0.7,
    ):
        """创建各种模态的Transformer主干网络"""

        def instantiate_trunk(
                embed_dim, num_blocks, num_heads, pre_transformer_ln, add_bias_kv, drop_path
        ):
            """实例化Transformer主干网络"""
            return SimpleTransformer(
                embed_dim=embed_dim,  # 嵌入维度
                num_blocks=num_blocks,  # Transformer块数
                ffn_dropout_rate=0.0,  # FFN dropout率
                drop_path_rate=drop_path,  # drop path率
                attn_target=partial(  # 注意力机制配置
                    MultiheadAttention,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    bias=True,
                    add_bias_kv=add_bias_kv,
                ),
                pre_transformer_layer=nn.Sequential(  # Transformer前的预处理层
                    nn.LayerNorm(embed_dim, eps=1e-6)
                    if pre_transformer_ln
                    else nn.Identity(),
                    EinOpsRearrange("b l d -> l b d"),  # 重排维度
                ),
                post_transformer_layer=EinOpsRearrange("l b d -> b l d"),  # Transformer后的后处理层
            )

        # 为每种模态创建Transformer主干
        modality_trunks = {}
        modality_trunks[ModalityType.VISION] = instantiate_trunk(
            vision_embed_dim,
            vision_num_blocks,
            vision_num_heads,
            pre_transformer_ln=True,  # 视觉模态使用LayerNorm
            add_bias_kv=False,  # 视觉模态不使用bias_kv
            drop_path=0.0,  # 视觉模态drop path率
        )
        modality_trunks[ModalityType.TEXT] = instantiate_trunk(
            text_embed_dim,
            text_num_blocks,
            text_num_heads,
            pre_transformer_ln=False,  # 文本模态不使用LayerNorm
            add_bias_kv=False,  # 文本模态不使用bias_kv
            drop_path=0.0,  # 文本模态drop path率
        )
        modality_trunks[ModalityType.AUDIO] = instantiate_trunk(
            audio_embed_dim,
            audio_num_blocks,
            audio_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,  # 音频模态使用bias_kv
            drop_path=audio_drop_path,  # 音频模态drop path率
        )
        modality_trunks[ModalityType.DEPTH] = instantiate_trunk(
            depth_embed_dim,
            depth_num_blocks,
            depth_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=depth_drop_path,
        )
        modality_trunks[ModalityType.THERMAL] = instantiate_trunk(
            thermal_embed_dim,
            thermal_num_blocks,
            thermal_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=thermal_drop_path,
        )
        modality_trunks[ModalityType.IMU] = instantiate_trunk(
            imu_embed_dim,
            imu_num_blocks,
            imu_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=imu_drop_path,
        )

        return nn.ModuleDict(modality_trunks)

    def _create_modality_heads(
            self,
            out_embed_dim,  # 输出嵌入维度
            vision_embed_dim,  # 视觉嵌入维度
            text_embed_dim,  # 文本嵌入维度
            audio_embed_dim,  # 音频嵌入维度
            depth_embed_dim,  # 深度图嵌入维度
            thermal_embed_dim,  # 热成像嵌入维度
            imu_embed_dim,  # IMU嵌入维度
    ):
        """创建各种模态的头部网络，将不同模态映射到统一空间"""

        modality_heads = {}

        # 视觉模态头部：LayerNorm + 选择CLS token + 线性投影
        modality_heads[ModalityType.VISION] = nn.Sequential(
            nn.LayerNorm(normalized_shape=vision_embed_dim, eps=1e-6),
            SelectElement(index=0),  # 选择CLS token
            nn.Linear(vision_embed_dim, out_embed_dim, bias=False),
        )

        # 文本模态头部：选择EOS token并投影
        modality_heads[ModalityType.TEXT] = SelectEOSAndProject(
            proj=nn.Sequential(
                nn.LayerNorm(normalized_shape=text_embed_dim, eps=1e-6),
                nn.Linear(text_embed_dim, out_embed_dim, bias=False),
            )
        )

        # 音频模态头部
        modality_heads[ModalityType.AUDIO] = nn.Sequential(
            nn.LayerNorm(normalized_shape=audio_embed_dim, eps=1e-6),
            SelectElement(index=0),  # 选择CLS token
            nn.Linear(audio_embed_dim, out_embed_dim, bias=False),
        )

        # 深度图模态头部
        modality_heads[ModalityType.DEPTH] = nn.Sequential(
            nn.LayerNorm(normalized_shape=depth_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(depth_embed_dim, out_embed_dim, bias=False),
        )

        # 热成像模态头部
        modality_heads[ModalityType.THERMAL] = nn.Sequential(
            nn.LayerNorm(normalized_shape=thermal_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(thermal_embed_dim, out_embed_dim, bias=False),
        )

        # IMU模态头部：额外包含Dropout层
        modality_heads[ModalityType.IMU] = nn.Sequential(
            nn.LayerNorm(normalized_shape=imu_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Dropout(p=0.5),  # 50%的dropout率
            nn.Linear(imu_embed_dim, out_embed_dim, bias=False),
        )

        return nn.ModuleDict(modality_heads)

    def _create_modality_postprocessors(self, out_embed_dim):
        """创建各种模态的后处理器，进行归一化和logit缩放"""

        modality_postprocessors = {}

        # 视觉模态后处理：仅归一化
        modality_postprocessors[ModalityType.VISION] = Normalize(dim=-1)

        # 文本模态后处理：归一化 + 可学习的logit缩放
        modality_postprocessors[ModalityType.TEXT] = nn.Sequential(
            Normalize(dim=-1), LearnableLogitScaling(learnable=True)
        )

        # 音频模态后处理：归一化 + 固定logit缩放
        modality_postprocessors[ModalityType.AUDIO] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=20.0, learnable=False),
        )

        # 深度图模态后处理
        modality_postprocessors[ModalityType.DEPTH] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=5.0, learnable=False),
        )

        # 热成像模态后处理
        modality_postprocessors[ModalityType.THERMAL] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=10.0, learnable=False),
        )

        # IMU模态后处理
        modality_postprocessors[ModalityType.IMU] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=5.0, learnable=False),
        )

        return nn.ModuleDict(modality_postprocessors)

    def forward(self, inputs):
        """前向传播"""
        outputs = {}
        for modality_key, modality_value in inputs.items():
            # 检查是否为多clip输入（音频和视频）
            reduce_list = (
                    modality_value.ndim >= 5
            )
            if reduce_list:
                # 重塑多clip输入为批量形式
                B, S = modality_value.shape[:2]
                modality_value = modality_value.reshape(
                    B * S, *modality_value.shape[2:]
                )

            if modality_value is not None:
                # 1. 模态预处理
                modality_value = self.modality_preprocessors[modality_key](
                    **{modality_key: modality_value}
                )
                trunk_inputs = modality_value["trunk"]  # 主干网络输入
                head_inputs = modality_value["head"]  # 头部网络输入

                # 2. 通过Transformer主干
                modality_value = self.modality_trunks[modality_key](**trunk_inputs)

                # 3. 通过模态头部
                modality_value = self.modality_heads[modality_key](
                    modality_value, **head_inputs
                )

                # 4. 后处理
                modality_value = self.modality_postprocessors[modality_key](
                    modality_value
                )

                # 如果是多clip输入，计算平均值
                if reduce_list:
                    modality_value = modality_value.reshape(B, S, -1)
                    modality_value = modality_value.mean(dim=1)

                outputs[modality_key] = modality_value

        return outputs


class ElasticImageBindModel(ImageBindModel):
    """支持早期退出和重入的弹性ImageBind模型"""

    def forward(self, inputs, ee=100, re_enter=0):
        """
        弹性ImageBindModel的前向传播

        参数:
            inputs (dict): 包含不同模态输入数据的字典
            ee (int): 早期退出层数
            re_enter (int): 重入次数

        返回:
            dict: 包含每个模态输出的字典
        """
        outputs = {}
        for modality_key, modality_value in inputs.items():
            reduce_list = (
                    modality_value.ndim >= 5
            )  # 音频和视频输入包含多个clip
            if reduce_list:
                B, S = modality_value.shape[:2]
                modality_value = modality_value.reshape(
                    B * S, *modality_value.shape[2:]
                )

            if modality_value is not None:
                # 模态预处理
                modality_value = self.modality_preprocessors[modality_key](
                    **{modality_key: modality_value}
                )
                trunk_inputs = modality_value["trunk"]
                head_inputs = modality_value["head"]

                # 通过Transformer主干（支持早期退出和重入）
                modality_value = self.modality_trunks[modality_key](**trunk_inputs, ee=ee, re_enter=re_enter)

                # 通过模态头部
                modality_value = self.modality_heads[modality_key](
                    modality_value, **head_inputs
                )

                # 后处理
                modality_value = self.modality_postprocessors[modality_key](
                    modality_value
                )

                # CDQ: 为避免bug，在重入时不进行多clip平均，这不影响延迟
                if not re_enter:
                    if reduce_list:
                        modality_value = modality_value.reshape(B, S, -1)
                        modality_value = modality_value.mean(dim=1)

                outputs[modality_key] = modality_value

        return outputs


def imagebind_huge(pretrained=False, elastic=False):
    """创建huge版本的ImageBind模型"""
    if elastic:
        # 创建弹性版本模型
        model = ElasticImageBindModel(
            vision_embed_dim=1280,  # 更大的视觉嵌入维度
            vision_num_blocks=32,  # 更多的Transformer块
            vision_num_heads=16,  # 更多的注意力头
            text_embed_dim=1024,  # 更大的文本嵌入维度
            text_num_blocks=24,  # 更多的文本Transformer块
            text_num_heads=16,  # 更多的文本注意力头
            out_embed_dim=1024,  # 更大的输出嵌入维度
            audio_drop_path=0.1,  # 音频drop path率
            imu_drop_path=0.7,  # IMU drop path率
        )
    else:
        # 创建标准版本模型
        model = ImageBindModel(
            vision_embed_dim=1280,
            vision_num_blocks=32,
            vision_num_heads=16,
            text_embed_dim=1024,
            text_num_blocks=24,
            text_num_heads=16,
            out_embed_dim=1024,
            audio_drop_path=0.1,
            imu_drop_path=0.7,
        )

    # 加载预训练权重
    if pretrained:
        if not os.path.exists(".checkpoints/imagebind_huge.pth"):
            print("正在下载imagebind权重到 .checkpoints/imagebind_huge.pth ...")
            os.makedirs(".checkpoints", exist_ok=True)
            torch.hub.download_url_to_file(
                "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth",
                ".checkpoints/imagebind_huge.pth",
                progress=True,
            )

        # 加载权重（在CPU上加载以避免GPU内存问题）
        state_dict = torch.load(".checkpoints/imagebind_huge.pth", map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=False)  # 非严格加载以兼容不同配置

    return model


def save_module(module_dict: nn.ModuleDict, module_name: str = "",
                checkpoint_dir: str = "./.checkpoints/full", postfix: str = "_last",
                extension: str = "pth"):
    """保存模块参数到文件"""
    try:
        torch.save(module_dict.state_dict(),
                   os.path.join(checkpoint_dir, f"imagebind-{module_name}{postfix}.{extension}"))
        logging.info(f"已保存模块 {module_name} 的参数到 {checkpoint_dir}。")
    except FileNotFoundError:
        logging.warning(f"无法保存模块 {module_name} 的参数到 {checkpoint_dir}。")


def load_module(module_dict: nn.ModuleDict, module_name: str = "",
                checkpoint_dir: str = "./.checkpoints/full", postfix: str = "_last",
                extension: str = "pth"):
    """从文件加载模块参数"""
    try:
        module_dict.load_state_dict(torch.load(
            os.path.join(checkpoint_dir, f"imagebind-{module_name}{postfix}.{extension}")), strict=False)
        logging.info(f"已从 {checkpoint_dir} 加载模块 {module_name} 的参数。")
    except FileNotFoundError:
        logging.warning(f"无法从 {checkpoint_dir} 加载模块 {module_name} 的参数。")