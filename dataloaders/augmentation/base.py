# -*- coding: utf-8 -*-
# @Time    : 2026/2/24
# @Author  : Concrete Eavesdrop System 3022234317@tju.edu.cn
# @FileName: base.py

import numpy as np
import torch
import os.path as op
from os import listdir
from dataloaders.augmentation.magical_effects import MagicalEffects
from dataloaders.augmentation.concrete_physics import ConcretePhysicsChain
import sys
sys.path.append("../../tools")
from tools.pytorch.random_ import *
from tools.others.audio_op import *


class AudioAug:
    """
    双规制数据增强基类。
    Phase 1: 环境声学做旧（原有 MagicalEffects）
    Phase 2: 物理链路降级（混凝土穿透链路模拟）
    """

    def __init__(
        self,
        p_effects: dict,
        room_rir_dir: str = None,
        concrete_rir_dir: str = None,
        target_sr: int = 44100,
        physics_config: dict = None,
    ):
        """
        Args:
            p_effects: 原有环境声学效果概率配置
            room_rir_dir: 室内 RIR 目录（Phase 1）
            concrete_rir_dir: 混凝土 IR 目录（Phase 2，40kHz 频段脉冲响应）
            target_sr: 最终目标采样率
            physics_config: 物理链路参数配置，控制失真强度等
        """
        self.target_sr = target_sr

        # ---- Phase 1: 环境声学做旧（原有效果链）----
        self.magical_effects = MagicalEffects(
            p_effects=p_effects, rir_dir=room_rir_dir
        )
        # ---- Phase 2: 物理链路降级 ----
        self.concrete_rir_dir = concrete_rir_dir
        self.concrete_ir_list = []
        if concrete_rir_dir is not None and op.isdir(concrete_rir_dir):
            self.concrete_ir_list = [
                op.join(concrete_rir_dir, f)
                for f in listdir(concrete_rir_dir)
                if f.endswith(('.npy', '.wav'))
            ]
            if len(self.concrete_ir_list) == 0:
                raise RuntimeError(f"Error: 混凝土 IR 目录 {concrete_rir_dir} 中无有效文件")

        # 预加载混凝土 IR 到内存，避免 DataLoader 中反复磁盘 IO
        self._concrete_ir_cache = {}
        for ir_path in self.concrete_ir_list:
            if ir_path.endswith('.npy'):
                self._concrete_ir_cache[ir_path] = np.load(ir_path, allow_pickle=True).astype(np.float32)
            elif ir_path.endswith('.wav'):
                import soundfile as sf
                data, _ = sf.read(ir_path, dtype='float32')
                self._concrete_ir_cache[ir_path] = data

        # 默认物理链路配置
        default_physics = {
            "enable": True,
            "high_sr": 250000,          # 高频域采样率
            "carrier_freq": 40000,      # AM 载波频率
            "jfet_gain": 1.0,           # JFET 失真增益
            "jfet_harmonic": 0.1,       # 二次谐波系数
            "emi_snr_db": (10, 30),     # EMI 信噪比范围
            "emi_freqs": [50, 1000, 2048],  # EMI 啸叫频率候选
            "demod_lpf_cutoff": 8000,   # 解调低通截止频率
        }
        if physics_config:
            default_physics.update(physics_config)

        self.physics_chain = ConcretePhysicsChain(
            config=default_physics,
            concrete_ir_cache=self._concrete_ir_cache,
            target_sr=target_sr,
        )
    @property
    def concrete_physics(self):
        return self.physics_chain

    def update_physics_config(self, new_config: dict):
        """
        动态更新物理链路参数（Curriculum Learning 接口）。
        
        Args:
            new_config: 新的物理链路参数字典，将与现有配置合并
        """
        if hasattr(self, 'physics_chain') and self.physics_chain is not None:
            if hasattr(self.physics_chain, 'config'):
                self.physics_chain.config.update(new_config)
            elif hasattr(self.physics_chain, 'update_config'):
                self.physics_chain.update_config(new_config)
            else:
                # 直接更新属性
                for k, v in new_config.items():
                    if hasattr(self.physics_chain, k):
                        setattr(self.physics_chain, k, v)

    def augment(
        self,
        frames: np.ndarray,
        effects=None,
        sample_rate: int = 44100,
        apply_phase2: bool = True,
        phase2_intensity: float = 0.5,
    ):
        """
        双规制数据增强入口。

        Args:
            frames: 干净音频 numpy 数组
            effects: 效果名称列表。None 表示使用 random_server 内置默认全部效果
            sample_rate: 采样率
            apply_phase2: 是否应用 Phase 2 物理链路
            phase2_intensity: Phase 2 强度 [0, 1]

        Returns:
            (degraded_audio, metadata_dict)
        """
        metadata = {"phase1_effects": [], "phase2_applied": False}

        # ---- Phase 1: 传统声学增强 ----
        augmented = frames.copy()
        if hasattr(self, 'magical_effects') and self.magical_effects is not None:
            if effects is None:
                effects = list(self.magical_effects.ps.p_effects.keys())

            # 过滤掉 random_server 中不被 magical_effects 支持的效果名
            supported = set(self.magical_effects.ps.p_effects.keys()) if hasattr(self.magical_effects, 'ps') else set()
            effects = [e for e in effects if e in supported] if supported else effects

            if len(effects) > 0:
                try:
                    augmented, applied_effects = self.magical_effects.effect(
                        augmented,
                        effects,
                        sample_rate,
                        None,
                        True,
                    )
                    metadata["phase1_effects"] = applied_effects
                except Exception as e:
                    import warnings
                    warnings.warn(f"[Phase1] 增强失败，使用原始音频: {e}")
        
        # ====================================================================
        # 【核心新增】：在 Phase1 结束，Phase2 开始前，把当前的半成品音频抽出来保存
        # ====================================================================
        metadata["phase1_audio"] = augmented.copy()
        
        # ---- Phase 2: 混凝土物理链路 ----
        if apply_phase2 and hasattr(self, 'physics_chain') and self.physics_chain is not None:
            try:
                augmented = self.physics_chain.apply(
                    augmented,
                    intensity=phase2_intensity,
                    input_sr=sample_rate
                )
                metadata["phase2_applied"] = True
                metadata["phase2_intensity"] = phase2_intensity
            except Exception as e:
                import warnings
                warnings.warn(f"[Phase2] 物理链路失败，跳过: {e}")

        return augmented, metadata

def add_noise_and_scale(front, noise, snr_l=-5, snr_h=35, scale_lower=0.6, scale_upper=1.0):
    snr = None
    noise, front = normalize_energy_torch(noise), normalize_energy_torch(front)  
    if(snr_l is not None and snr_h is not None):
        front, noise, snr = _random_noise(front, noise, snr_l=snr_l, snr_h=snr_h)  
    noisy, noise, front = unify_energy_torch(noise + front, noise, front)  
    scale = _random_scale(scale_lower, scale_upper)  
    noisy, noise, front = noisy * scale, noise * scale, front * scale  
    return front, noise, snr, scale

def add_noise_and_scale_with_HQ_with_Aug(HQ, front, augfront, noise, snr_l=-5, snr_h=35, scale_lower=0.6, scale_upper=1.0):
    snr = None
    noise = normalize_energy_torch(noise)  
    HQ, front, augfront = unify_energy_torch(HQ, front, augfront)
    front_level = torch.mean(torch.abs(augfront))
    if(front_level > 0.02):
        noise_front_energy_ratio = torch.mean(torch.abs(noise)) / front_level
        noise = noise / noise_front_energy_ratio
    if(snr_l is not None and snr_h is not None):
        augfront, noise, snr = _random_noise(augfront, noise, snr_l=snr_l, snr_h=snr_h)  
    _, augfront, noise, front, HQ = unify_energy_torch(noise + augfront, augfront,  noise, front, HQ)  
    scale = _random_scale(scale_lower, scale_upper)  
    noise, front, augfront, HQ = noise * scale, front * scale, augfront*scale, HQ*scale  
    return HQ, front, augfront, noise, snr, scale

def add_noise_and_scale_with_HQ(HQ, front, noise, snr_l=-5, snr_h=35, scale_lower=0.6, scale_upper=1.0):
    snr = None
    noise = normalize_energy_torch(noise)  
    HQ, front = unify_energy_torch(HQ, front)
    front_level = torch.mean(torch.abs(front))
    if(front_level > 0.02):
        noise_front_energy_ratio = torch.mean(torch.abs(noise)) / front_level
        noise = noise / noise_front_energy_ratio
    if(snr_l is not None and snr_h is not None):
        front, noise, snr = _random_noise(front, noise, snr_l=snr_l, snr_h=snr_h)  
    _, noise, front, HQ = unify_energy_torch(noise + front, noise, front, HQ)  
    scale = _random_scale(scale_lower, scale_upper)  
    noise, front, HQ = noise * scale, front * scale, HQ*scale  
    return HQ, front, noise, snr, scale

def _random_scale(lower = 0.3, upper=0.9):
    return float(uniform_torch(lower,upper))

def _random_noise(clean, noise, snr_l = None, snr_h = None):
    snr = uniform_torch(snr_l,snr_h)
    clean_weight = 10 ** (float(snr) / 20)
    return clean,noise/clean_weight, snr