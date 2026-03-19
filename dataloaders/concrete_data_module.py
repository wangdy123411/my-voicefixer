# -*- coding: utf-8 -*-
# @Time    : 2026/2/25
# @Author  : Concrete Eavesdrop System 3022234317@tju.edu.cn
# @FileName: concrete_data_module.py
#
# 混凝土穿透语音恢复专用数据模块
# 继承并扩展原始 SrRandSampleRate，集成双规制增强 + 自适应 Collator

import os
import warnings
from typing import Optional, Dict, Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataloaders.data_module import SrRandSampleRate
from dataloaders.augmentation.base import AudioAug
from dataloaders.collators.concrete_collator import ConcreteEavesdropCollator, ConcreteEavesdropCollatorGPU
from dataloaders.concrete_dataset import ConcreteAugDataset, worker_init_fn

class ConcreteDataModule(pl.LightningDataModule):
    """
    混凝土穿透场景数据模块。

    与原始 SrRandSampleRate 的区别：
    1. 集成双规制增强（Phase 1 声学 + Phase 2 物理链路）
    2. 使用 ConcreteEavesdropCollator 施加自适应低通
    3. 支持 Curriculum Learning 动态更新增强参数

    DataLoader 性能优化：
    - persistent_workers=True：避免每 epoch 重新 fork
    - prefetch_factor=3：流水线化 CPU 预处理和 GPU 训练
    - pin_memory=True：加速 CPU→GPU 传输
    - IR 预加载到内存：避免 worker 内磁盘 IO
    """

    def __init__(self, hp: dict, distributed: bool = False):
        super().__init__()
        self.hp = hp
        self.distributed = distributed
        self.concrete_cfg = hp.get("concrete", {})

        # ---- 数据配置 ----
        self.sample_rate = hp["data"]["sampling_rate"]
        self.batch_size = hp["train"].get("batch_size", 8)
        self.num_workers = hp["train"].get("num_workers", 4)

        # ---- Phase 2 强度（Curriculum 动态调整）----
        self.phase2_intensity = self.concrete_cfg.get("default_phase2_intensity", 0.5)

        # ---- 数据增强引擎 ----
        self.audio_aug = self._build_audio_aug()

        # ---- 独立的 Collator (训练与验证隔离) ----
        self.train_collator, self.val_collator = self._build_collators()

        # ---- 数据集引用（setup 时初始化）----
        self.train_dataset = None
        self.val_dataset = None

    def _build_audio_aug(self) -> AudioAug:
        """构建双规制数据增强引擎"""
        aug_cfg = self.hp.get("augment", {})

        # Phase 1 效果配置
        p_effects = aug_cfg.get("effects", {})

        # IR 目录
        room_rir_dir = aug_cfg.get("params", {}).get("rir_root", None)
        concrete_rir_dir = self.concrete_cfg.get("concrete_rir_dir", None)

        if concrete_rir_dir and self.hp.get("root"):
            concrete_rir_dir = os.path.join(self.hp["root"], concrete_rir_dir)

        # Phase 2 物理配置
        physics_config = self.concrete_cfg.get("physics_config", {})

        return AudioAug(
            p_effects=p_effects,
            target_sr=self.sample_rate,
            room_rir_dir=room_rir_dir,
            concrete_rir_dir=concrete_rir_dir,
            physics_config=physics_config,
        )

    def _build_collators(self):
        """分别构建训练与验证的 GPU Collator"""
        col_cfg = self.concrete_cfg.get("collator", {})
        from dataloaders.collators.concrete_collator import ConcreteEavesdropCollatorGPU
        
        # 训练集：保持原有的随机 Curriculum 难度
        train_collator = ConcreteEavesdropCollatorGPU(
            sample_rate=self.sample_rate,
            cutoff_range=tuple(col_cfg.get("cutoff_range", [1500, 4500])),
            filter_order=col_cfg.get("filter_order", 8),
            apply_prob=col_cfg.get("apply_prob", 0.9),
        )
        
        # 👇 修复 1：验证集也必须用 GPU Collator，并强制锁死在 Stage 4 满级难度！
        stage4_cfg = self.hp["concrete"]["curriculum_stages"][-1]
        cutoff_mean = sum(stage4_cfg["collator_config"]["cutoff_range"]) / 2.0
        
        val_collator = ConcreteEavesdropCollatorGPU(
            sample_rate=self.sample_rate,
            cutoff_range=(cutoff_mean, cutoff_mean),  # 上下界一致，强制固定截止频率 (平均值)
            filter_order=col_cfg.get("filter_order", 8),
            apply_prob=1.0,  # 100%触发，不留任何随机后门
        )
        
        return train_collator, val_collator

    def setup(self, stage: Optional[str] = None):
        """初始化数据集"""
        if stage == "fit" or stage is None:
            from dataloaders.concrete_dataset import ConcreteAugDataset

            self.train_dataset = ConcreteAugDataset(
                hp=self.hp,
                split="train",
                audio_aug=self.audio_aug,
                phase2_intensity=self.phase2_intensity,
            )
            
            stage4_cfg = self.hp["concrete"]["curriculum_stages"][-1]
            self.val_dataset = ConcreteAugDataset(
                hp=self.hp,
                split="val",
                audio_aug=self.audio_aug,  
                phase2_intensity=stage4_cfg["phase2_intensity"]
                # 👈 删除了这里报错的 val_paired_mode=False
            )

            if len(self.train_dataset) == 0:
                raise RuntimeError("训练集为空，请检查 data.train_dataset 配置路径")
            if len(self.val_dataset) == 0:
                warnings.warn("验证集为空，将跳过验证步骤")

            print(f"[ConcreteDataModule] 数据集就绪: train={len(self.train_dataset)}, val={len(self.val_dataset)}")

    def train_dataloader(self) -> DataLoader:
        sampler = None
        if self.distributed:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(self.train_dataset, shuffle=True)

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self.train_collator,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=3 if self.num_workers > 0 else None,
            drop_last=True,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self):
        from torch.utils.data import DataLoader

        print("\n🚀 [警告] 正在启动 100% 全动态验证管道！物理参数与 FFT 截断已对齐满级 (Stage 4)！")

        val_batch_size = max(1, self.hp["train"]["batch_size"] // 2)

        return DataLoader(
            self.val_dataset,  # 👈 直接使用 setup 中初始化好的数据集
            batch_size=val_batch_size,
            shuffle=False, 
            num_workers=self.hp["train"]["num_workers"],
            collate_fn=self.val_collator,  
            pin_memory=True,
            drop_last=False
        )
    # ---- Curriculum Learning 动态更新接口 ----

    def update_physics_config(self, new_config: dict):
        """更新 Phase 2 物理链路参数"""
        self.audio_aug.update_physics_config(new_config)

    def update_collator_config(self, new_config: dict):
        """更新训练 Collator 参数 (验证集保持不变以对比收敛)"""
        if "cutoff_range" in new_config:
            self.train_collator.cutoff_range = tuple(new_config["cutoff_range"])
            self.train_collator._precompute_filters()
        if "apply_prob" in new_config:
            self.train_collator.apply_prob = new_config["apply_prob"]