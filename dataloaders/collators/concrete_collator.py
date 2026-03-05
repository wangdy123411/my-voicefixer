# -*- coding: utf-8 -*-
# @Time    : 2026/2/24
# @Author  : Concrete Eavesdrop System
# @FileName: concrete_collator.py
#
# 混凝土穿透场景专用 Collator：
# 对输入数据施加随机低通滤波，模拟 PZT 换能器的极窄带宽特性，
# 逼迫网络学习极限频带扩展 (Bandwidth Extension, BWE)。

import torch
import numpy as np
from scipy.signal import butter, sosfiltfilt
from typing import List, Dict, Optional, Tuple
from functools import lru_cache


class ConcreteEavesdropCollator:
    """
    专用 Collator，在 Batch 打包阶段对输入施加强制低通滤波。
    
    物理依据：
    PZT 压电陶瓷换能器在混凝土表面的接收带宽通常仅有 1-5 kHz，
    经过解调后的基带信号高频成分严重缺失。
    网络必须学会从 1.5-4.5 kHz 带宽的信号中恢复全带宽语音。
    
    用法：
        collator = ConcreteEavesdropCollator(
            sample_rate=44100,
            cutoff_range=(1500, 4500),
            filter_order=8,
        )
        dataloader = DataLoader(dataset, collate_fn=collator, ...)
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        cutoff_range: Tuple[float, float] = (1500.0, 4500.0),
        filter_order: int = 8,
        apply_prob: float = 1.0,
        curriculum_cutoff_min: Optional[float] = None,
    ):
        """
        Args:
            sample_rate: 音频采样率
            cutoff_range: 低通截止频率的随机范围 [Hz]
            filter_order: Butterworth 滤波器阶数（越高截止越陡峭）
            apply_prob: 对每个 batch 施加低通的概率（可用于 Curriculum）
            curriculum_cutoff_min: 如果设置，会覆盖 cutoff_range 的下界
                                   用于 Curriculum Learning 动态调整
        """
        self.sample_rate = sample_rate
        self.cutoff_range = cutoff_range
        self.filter_order = filter_order
        self.apply_prob = apply_prob
        self.curriculum_cutoff_min = curriculum_cutoff_min

        # 预缓存常用截止频率的滤波器系数（量化到 100Hz 步长）
        self._sos_cache = {}
        self._precompute_filters()

    def _precompute_filters(self):
        """
        预计算所有可能截止频率对应的 Butterworth 系数。
        量化到 100Hz 步长，减少运行时计算。
        """
        low = int(self.cutoff_range[0] / 100) * 100
        high = int(self.cutoff_range[1] / 100) * 100 + 100
        nyq = self.sample_rate / 2.0

        for cutoff in range(low, high + 1, 100):
            norm_cutoff = min(cutoff / nyq, 0.99)
            if norm_cutoff > 0.01:
                sos = butter(self.filter_order, norm_cutoff, btype='low', output='sos')
                self._sos_cache[cutoff] = sos

    def _get_nearest_sos(self, cutoff_hz: float):
        """获取最近的预缓存滤波器系数"""
        quantized = int(round(cutoff_hz / 100)) * 100
        quantized = max(quantized, min(self._sos_cache.keys()))
        quantized = min(quantized, max(self._sos_cache.keys()))
        return self._sos_cache[quantized]

    def _apply_lowpass(self, waveform: np.ndarray, cutoff_hz: float) -> np.ndarray:
        """
        对单个波形施加低通滤波。
        使用 sosfiltfilt（零相位滤波），避免引入相位失真。
        """
        sos = self._get_nearest_sos(cutoff_hz)
        # sosfiltfilt 在短信号上可能不稳定，设置 padlen
        padlen = min(len(waveform) - 1, 3 * max(len(sos), 1))
        if padlen < 1:
            return waveform
        filtered = sosfiltfilt(sos, waveform, padlen=padlen).astype(np.float32)
        return filtered

    def update_curriculum(self, cutoff_min: float = None, apply_prob: float = None):
        """
        Curriculum Learning 阶段切换时更新参数。
        例如：Stage 1 cutoff 3000-4500Hz, Stage 3 cutoff 1500-2500Hz
        """
        if cutoff_min is not None:
            self.curriculum_cutoff_min = cutoff_min
            self.cutoff_range = (cutoff_min, self.cutoff_range[1])
            self._precompute_filters()
        if apply_prob is not None:
            self.apply_prob = apply_prob

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate 函数入口。
        
        期望每个 sample 是 dict，包含:
        - 'input_wave': np.ndarray, 降级后的音频波形
        - 'target_wave': np.ndarray, 干净目标波形
        - 'input_spec': np.ndarray (optional), 降级后的频谱
        - 'target_spec': np.ndarray (optional), 干净目标频谱
        
        输出:
        - 'input_wave': torch.Tensor [B, T] (经过额外低通)
        - 'target_wave': torch.Tensor [B, T]
        - 'input_spec': torch.Tensor [B, F, T] (经过额外低通后重新计算)
        - 'target_spec': torch.Tensor [B, F, T]
        - 'cutoff_hz': torch.Tensor [B] (记录每个样本的截止频率)
        """
        # 确定本 batch 是否施加低通
        do_lowpass = np.random.random() < self.apply_prob

        # 为 batch 中每个样本独立采样截止频率
        batch_size = len(batch)
        cutoff_low = self.curriculum_cutoff_min or self.cutoff_range[0]
        cutoff_high = self.cutoff_range[1]
        cutoff_hz_array = np.random.uniform(cutoff_low, cutoff_high, size=batch_size)

        # ---- 处理波形 ----
        input_waves = []
        target_waves = []
        phase1_waves = []
        cutoffs = []

        # 找到 batch 中的最大长度，用于 padding
        max_input_len = max(len(s['input_wave']) for s in batch)
        max_target_len = max(len(s['target_wave']) for s in batch)
        max_phase1_len = max(len(s.get('phase1_wave', s['input_wave'])) for s in batch)

        for i, sample in enumerate(batch):
            input_wave = sample['input_wave'].astype(np.float32)
            target_wave = sample['target_wave'].astype(np.float32)
            phase1_wave = sample.get('phase1_wave', input_wave).astype(np.float32)

            # 对输入施加低通滤波（目标不施加）
            if do_lowpass and len(input_wave) > self.filter_order * 3:
                cutoff = cutoff_hz_array[i]
                input_wave = self._apply_lowpass(input_wave, cutoff)
                cutoffs.append(cutoff)
            else:
                cutoffs.append(cutoff_high)  # 记录实际截止频率

            # Zero-padding 到统一长度
            input_padded = np.zeros(max_input_len, dtype=np.float32)
            input_padded[:len(input_wave)] = input_wave

            target_padded = np.zeros(max_target_len, dtype=np.float32)
            target_padded[:len(target_wave)] = target_wave

            # 【新增】对 phase1_wave 进行 Zero-padding
            phase1_padded = np.zeros(max_phase1_len, dtype=np.float32)
            phase1_padded[:len(phase1_wave)] = phase1_wave

            input_waves.append(input_padded)
            target_waves.append(target_padded)
            phase1_waves.append(phase1_padded)  # 【新增】添加到列表

        input_tensor = torch.from_numpy(np.stack(input_waves, axis=0))   # [B, T]
        target_tensor = torch.from_numpy(np.stack(target_waves, axis=0)) # [B, T]
        phase1_tensor = torch.from_numpy(np.stack(phase1_waves, axis=0)) # 【新增】转换为 Tensor [B, T]

        result = {
            # ConcreteCollator 原始键名
            'input_wave': input_tensor,
            'target_wave': target_tensor,
            'phase1_wave': phase1_tensor,
            # 父类 VoiceFixer.preprocess(batch) 兼容键名
            'noisy': input_tensor,          # gsr_voicefixer.py L227
            'vocal': target_tensor,         # gsr_voicefixer.py L227
            'fname': [''] * len(batch),     # gsr_voicefixer.py L266
            'cutoff_hz': torch.tensor(cutoffs, dtype=torch.float32),         # [B]
        }

        # ---- 处理频谱（如果提供）----
        if 'input_spec' in batch[0] and batch[0]['input_spec'] is not None:
            # 注意：如果施加了低通，频谱需要从滤波后的波形重新计算
            # 这里假设 Dataset 返回的是波形，频谱在 Collator 中按需计算
            # 或者直接使用 Dataset 提供的频谱（但低通效果不会体现在频谱上）
            input_specs = []
            target_specs = []
            for sample in batch:
                input_specs.append(sample['input_spec'])
                target_specs.append(sample['target_spec'])
            result['input_spec'] = torch.from_numpy(np.stack(input_specs, axis=0))
            result['target_spec'] = torch.from_numpy(np.stack(target_specs, axis=0))

        return result


import torch.fft

class ConcreteEavesdropCollatorGPU(ConcreteEavesdropCollator):
    """
    GPU 加速版本的 Collator。
    取消 CPU 端的 scipy 滤波，只采样截止频率并打包。
    将滤波压力转移给 GPU 进行 FFT 频域软掩码。
    """
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        do_lowpass = np.random.random() < self.apply_prob
        batch_size = len(batch)
        
        cutoff_low = self.curriculum_cutoff_min or self.cutoff_range[0]
        cutoff_high = self.cutoff_range[1]
        
        # 为每个样本记录截止频率
        cutoffs = []
        if do_lowpass:
            cutoff_hz_array = np.random.uniform(cutoff_low, cutoff_high, size=batch_size)
        else:
            # 【修复】：设为远大于奈奎斯特频率的值（例如采样率的 2 倍）
            # 确保 freqs - cutoffs 永远是极大的负数，Sigmoid 掩码全频段输出 1.0
            cutoff_hz_array = [self.sample_rate * 2.0] * batch_size

        input_waves = []
        target_waves = []
        phase1_waves = []

        max_input_len = max(len(s['input_wave']) for s in batch)
        max_target_len = max(len(s['target_wave']) for s in batch)
        max_phase1_len = max(len(s.get('phase1_wave', s['input_wave'])) for s in batch)

        for i, sample in enumerate(batch):
            # 完全跳过 scipy 滤波，直接提取波形
            input_wave = sample['input_wave'].astype(np.float32)
            target_wave = sample['target_wave'].astype(np.float32)
            phase1_wave = sample.get('phase1_wave', input_wave).astype(np.float32)

            cutoffs.append(cutoff_hz_array[i])

            # Zero-padding 到统一长度
            input_padded = np.zeros(max_input_len, dtype=np.float32)
            input_padded[:len(input_wave)] = input_wave

            target_padded = np.zeros(max_target_len, dtype=np.float32)
            target_padded[:len(target_wave)] = target_wave

            phase1_padded = np.zeros(max_phase1_len, dtype=np.float32)
            phase1_padded[:len(phase1_wave)] = phase1_wave

            input_waves.append(input_padded)
            target_waves.append(target_padded)
            phase1_waves.append(phase1_padded)

        input_tensor = torch.from_numpy(np.stack(input_waves, axis=0))   # [B, T]
        target_tensor = torch.from_numpy(np.stack(target_waves, axis=0)) # [B, T]
        phase1_tensor = torch.from_numpy(np.stack(phase1_waves, axis=0)) # [B, T]

        result = {
            'input_wave': input_tensor,
            'target_wave': target_tensor,
            'phase1_wave': phase1_tensor,
            'noisy': input_tensor,          
            'vocal': target_tensor,         
            'fname': [''] * len(batch),     
            # 核心：将采样好的 cutoff_hz 作为 Tensor 送入 GPU
            'cutoff_hz': torch.tensor(cutoffs, dtype=torch.float32),  
        }
        return result

    @staticmethod
    def gpu_fft_lowpass(waveform: torch.Tensor, cutoff_hz: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        利用 GPU 进行高速频域低通滤波 (Zero-phase)。
        使用 Sigmoid 软掩码平滑过渡，避免硬截断(Brick-wall)引起的吉布斯振铃伪影。
        
        Args:
            waveform: [B, 1, T] 或 [B, T] 的音频张量
            cutoff_hz: [B] 的截止频率张量
            sample_rate: 音频采样率
        """
        is_3d = waveform.dim() == 3
        if is_3d:
            B, C, T = waveform.shape
            waveform = waveform.squeeze(1) # 压平为 [B, T] 以便做 FFT
        else:
            B, T = waveform.shape

        # 1. 一键将整个 Batch 的波形转换到频域 [B, N]
        X = torch.fft.rfft(waveform, dim=-1)
        N = X.shape[-1]
        
        # 2. 构造物理频率坐标轴 [1, N]
        freqs = torch.linspace(0, sample_rate / 2, N, device=waveform.device).unsqueeze(0)
        cutoffs = cutoff_hz.view(B, 1).to(waveform.device)
        
        # 3. 构造 Sigmoid 平滑过渡掩码 (Transition Bandwidth 约设为 200Hz)
        # 频率小于 cutoff 时趋近于 1，大于 cutoff 时迅速衰减为 0
        transition_bw = 200.0
        k = 10.0 / transition_bw
        mask = torch.sigmoid(-k * (freqs - cutoffs)) # 自动广播为 [B, N]
        
        # 4. 频域掩码相乘，然后逆变换回时域
        X_filtered = X * mask
        filtered_waveform = torch.fft.irfft(X_filtered, n=T, dim=-1) # 强行指定 n=T 防止奇偶长度丢失
        
        if is_3d:
            filtered_waveform = filtered_waveform.unsqueeze(1)
            
        return filtered_waveform