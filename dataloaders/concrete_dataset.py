# -*- coding: utf-8 -*-
# @Time    : 2026/2/25
# @Author  : Concrete Eavesdrop System 3022234317@tju.edu.cn
# @FileName: concrete_dataset.py
#
# 混凝土穿透语音恢复专用数据集
# 读取干净语音 → 双规制增强 → 返回 (degraded, clean) 训练对

import os
import warnings
import random
from typing import Optional, Tuple, Dict, Any

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from dataloaders.augmentation.base import AudioAug


class ConcreteAugDataset(Dataset):
    """
    混凝土穿透场景数据集。

    数据流：
    ┌──────────┐    ┌──────────────────────┐    ┌───────────────────┐
    │ 干净语音  │ ──→ │ Phase 1: 环境声学做旧 │ ──→ │ Phase 2: 物理链路  │ ──→ (degraded, clean)
    │ .wav文件  │    │ (混响/EQ/变速...)     │    │ (JFET/AM/混凝土IR) │
    └──────────┘    └──────────────────────┘    └───────────────────┘

    与原始 VoiceFixer 数据集的关系：
    - 原始数据集在 __getitem__ 中调用 AudioAug.perform() 做单阶段增强
    - 本类扩展为双阶段增强，并新增 phase2_intensity 控制
    - 保持与原始 collate 兼容的输出格式
    """

    def __init__(
        self,
        hp: dict,
        split: str = "train",
        audio_aug: Optional[AudioAug] = None,
        phase2_intensity: float = 0.5,
    ):
        """
        Args:
            hp: 超参配置字典
            split: "train" 或 "val"
            audio_aug: 双规制数据增强引擎（外部注入，避免每个 worker 重复初始化）
            phase2_intensity: Phase 2 物理链路强度 [0, 1]
        """
        super().__init__()
        self.hp = hp
        self.split = split
        self.audio_aug = audio_aug
        
        # 【核心修复 1】：如果是验证集，强制将强度锁定为 0.5（或 1.0），
        # 保证验证基准的恒定，并且让终端打印的日志完全准确！
        if self.split == "val":
            self.phase2_intensity = 0.5
        else:
            self.phase2_intensity = phase2_intensity

        # ---- 音频参数 ----
        self.sample_rate = hp["data"]["sampling_rate"]
        self.segment_length = hp["data"].get("segment_length", self.sample_rate)  # 默认 1 秒

        # ---- 加载文件列表 ----
        dataset_key = "train_dataset" if split == "train" else "val_dataset"
        dataset_cfg = hp["data"].get(dataset_key, {})

        self.vocal_files = self._scan_audio_files(
            dataset_cfg.get("speech", {}).get("vocal", "")
        )
        self.noise_files = self._scan_audio_files(
            dataset_cfg.get("speech", {}).get("noise", "")
        )

        # ---- 增强效果列表（从配置中提取）----
        self.effect_names = list(hp.get("augment", {}).get("effects", {}).keys())

        # ---- 验证 ----
        if len(self.vocal_files) == 0:
            raise RuntimeError(
                f"[{split}] 未找到语音文件，请检查配置路径: "
                f"{dataset_cfg.get('speech', {}).get('vocal', 'N/A')}"
            )

        # 此时打印出来的验证集 Phase 2 强度就会是正确的 0.5 了
        print(
            f"[ConcreteAugDataset/{split}] "
            f"语音: {len(self.vocal_files)} 文件, "
            f"噪声: {len(self.noise_files)} 文件, "
            f"片段长度: {self.segment_length} 采样点, "
            f"Phase 2 强度: {self.phase2_intensity}"
        )
        
        # ---- 增强效果列表（从配置中提取）----
        # 如果为空列表，augment 会使用 random_server 内置默认效果
        effects_cfg = hp.get("augment", {}).get("effects", {})
        self.effect_names = list(effects_cfg.keys()) if effects_cfg else None

    def __len__(self) -> int:
        return len(self.vocal_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        返回一个训练样本。
        """
        # ---- 1. 读取干净语音 (Target) ----
        clean = self._load_audio_segment(self.vocal_files[idx])

        # ---- 2. 混合环境噪声 (Phase 0) ----
        input_frames = clean.copy()
        
        # 【核心修复】：移除只在 train 加噪的限制，验证集也必须包含环境背景音！
        if self.noise_files:
            if self.split == "train":
                # 训练时：85% 概率加噪，完全随机的信噪比和音量
                if random.random() < 0.85:
                    add_noise = True
                    snr_db = random.uniform(-5.0, 35.0)
                    scale = random.uniform(0.6, 1.0)
                else:
                    add_noise = False
            else:
                # 验证集：100% 加噪，但使用固定的中等难度参数
                # 保证每次 validation 的评估基准完全一致，且画图能清晰看到 Phase 1
                add_noise = True
                snr_db = 15.0  # 固定 15dB 信噪比 (中等环境噪音)
                scale = 0.8    # 固定 0.8 音量缩放

            if add_noise:
                noise = self._load_random_noise(clean.shape[0])
                
                # 计算 RMS 能量
                clean_rms = np.sqrt(np.mean(clean**2) + 1e-10)
                noise_rms = np.sqrt(np.mean(noise**2) + 1e-10)
                
                # 根据设定的 SNR 缩放噪声并与干净语音混合
                target_noise_rms = clean_rms / (10 ** (snr_db / 20.0))
                noise_scaled = noise * (target_noise_rms / noise_rms)
                input_frames = clean + noise_scaled
                # 【新增】同步缩放干净目标，保证输入和目标的音量基准一致！
                clean = clean * scale
                # 整体音量缩放
                input_frames = input_frames * scale
                
                # 兜底防爆音裁剪
                peak = np.max(np.abs(input_frames))
                if peak > 0.99:
                    input_frames = (input_frames / peak) * 0.95

        # ---- 3. 送入双规制增强管线 (Phase 1 & Phase 2) ----
        if self.audio_aug is not None:
            if self.split == "train":
                apply_phase2 = self.phase2_intensity > 0
                current_intensity = self.phase2_intensity
            else:
                # 【核心修复 2】：验证集/画图时强制开启 Phase 2！
                # 保证每次 Epoch 评估的标准一致，且能让 TensorBoard 画出清晰的穿墙/电流特征图。
                apply_phase2 = True
                # 这里直接使用 __init__ 里修正好的类属性，避免硬编码
                current_intensity = self.phase2_intensity 

            degraded, metadata = self.audio_aug.augment(
                frames=input_frames,
                effects=self.effect_names,
                sample_rate=self.sample_rate,
                apply_phase2=apply_phase2,           # 传入修复后的布尔值
                phase2_intensity=current_intensity,  # 传入修复后的强度值
            )
        else:
            degraded = input_frames.copy()
            metadata = {"phase1_audio": input_frames.copy(), "phase1_effects": None, "phase2_applied": False}
            
        # ---- 4. 格式化并返回结果 ----
        if isinstance(degraded, torch.Tensor):
            degraded = degraded.numpy()
        if isinstance(clean, torch.Tensor):
            clean = clean.numpy()
        phase1 = metadata.get("phase1_audio", input_frames.copy())
        if isinstance(phase1, torch.Tensor): 
            phase1 = phase1.numpy()

        # 获取当前实际长度
        len_d, len_c, len_p = len(degraded), len(clean), len(phase1)
        max_len = max(len_d, len_c, len_p)
        
        # 额外保护：为每个音频独立计算微淡出 (Fade-out)，防短信号硬截断
        def apply_fadeout(audio_arr):
            arr_len = len(audio_arr)
            # 自适应淡出长度：最长 256 点，若信号极短则取其 5%
            f_len = min(256, arr_len // 20)
            if f_len > 1:
                f_curve = np.linspace(1.0, 0.0, f_len, dtype=np.float32)
                audio_arr[-f_len:] = audio_arr[-f_len:] * f_curve
            return audio_arr

        # 【必须在 padding 补零之前执行】，确保淡出的是真实的信号边缘
        degraded = apply_fadeout(degraded)
        clean = apply_fadeout(clean)
        phase1 = apply_fadeout(phase1)

        # 强制对齐维度：使用 zero-padding 补齐到 max_len
        if len_d < max_len:
            degraded = np.pad(degraded, (0, max_len - len_d), mode='constant', constant_values=0.0)
        if len_c < max_len:
            clean = np.pad(clean, (0, max_len - len_c), mode='constant', constant_values=0.0)
        if len_p < max_len:
            phase1 = np.pad(phase1, (0, max_len - len_p), mode='constant', constant_values=0.0)

        result = {
            "input_wave": degraded.astype(np.float32),    # 给模型输入的残缺音频 (Phase2)
            "target_wave": clean.astype(np.float32),      # 纯净的拟合目标 (Clean)
            "phase1_wave": phase1.astype(np.float32),     # 记录中间状态，用于 TensorBoard 听感
        }

        return result

    # ================================================================
    #  文件扫描与音频读取
    # ================================================================

    @staticmethod
    def _scan_audio_files(directory: str) -> list:
        """
        递归扫描目录下的音频文件。
        支持 .wav, .flac, .ogg, .mp3 格式。
        """
        if not directory or not os.path.isdir(directory):
            return []

        valid_ext = {".wav", ".flac", ".ogg", ".mp3"}
        files = []
        for root, _, fnames in os.walk(directory):
            for fname in sorted(fnames):
                if os.path.splitext(fname)[1].lower() in valid_ext:
                    files.append(os.path.join(root, fname))
        return files

    def _load_audio_segment(self, filepath: str) -> np.ndarray:
        """
        读取音频文件的随机片段。

        策略：
        - 训练时：随机起点截取 segment_length
        - 验证时：从头截取 segment_length（可复现）
        - 文件短于 segment_length：零填充

        使用 soundfile 而非 librosa，避免 resampy 的额外开销。
        """
        try:
            info = sf.info(filepath)
            total_frames = info.frames
            file_sr = info.samplerate
        except Exception as e:
            warnings.warn(f"无法读取音频信息 {filepath}: {e}")
            return np.zeros(self.segment_length, dtype=np.float32)

        # 计算需要读取的采样点数（考虑采样率差异）
        sr_ratio = file_sr / self.sample_rate
        needed_frames = int(self.segment_length * sr_ratio)

        # 确定起始位置
        if total_frames <= needed_frames:
            start = 0
            frames_to_read = total_frames
        elif self.split == "train":
            start = random.randint(0, total_frames - needed_frames)
            frames_to_read = needed_frames
        else:
            start = 0
            frames_to_read = needed_frames

        # 读取
        try:
            audio, sr = sf.read(
                filepath,
                start=start,
                frames=frames_to_read,
                dtype="float32",
                always_2d=False,
            )
        except Exception as e:
            warnings.warn(f"读取音频失败 {filepath}: {e}")
            return np.zeros(self.segment_length, dtype=np.float32)

        # 多声道转单声道
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # 重采样（如需要）
        if sr != self.sample_rate:
            audio = self._simple_resample(audio, sr, self.sample_rate)

        return audio.astype(np.float32)

    def _load_random_noise(self, length: int) -> np.ndarray:
        """随机选取一段噪声"""
        noise_path = random.choice(self.noise_files)
        noise = self._load_audio_segment(noise_path)

        # 确保噪声长度与信号匹配
        if len(noise) < length:
            # 循环填充
            repeats = (length // len(noise)) + 1
            noise = np.tile(noise, repeats)
        return noise[:length]

    @staticmethod
    def _simple_resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        简单重采样（线性插值）。
        
        注意：对训练数据来说线性插值精度足够，
        且比 librosa.resample 快约 10 倍。
        如需高精度重采样，可替换为 scipy.signal.resample_poly。
        """
        if orig_sr == target_sr:
            return audio
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    @staticmethod
    def _pad_or_trim(tensor: torch.Tensor, target_length: int) -> torch.Tensor:
        """填充或截断到固定长度"""
        current = tensor.shape[-1]
        if current >= target_length:
            return tensor[..., :target_length]
        else:
            pad_size = target_length - current
            return torch.nn.functional.pad(tensor, (0, pad_size), mode="constant", value=0.0)