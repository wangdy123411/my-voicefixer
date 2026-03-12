# -*- coding: utf-8 -*-
# @Time    : 2026/2/24
# @Author  : Concrete Eavesdrop System 3022234317@tju.deu.cn
# @FileName: concrete_physics.py
# 
# 物理链路降级模拟：
# 干净语音 → JFET非线性失真 → 40kHz AM调制 → 混凝土IR卷积 → 包络解调 → EMI叠加

import random
import numpy as np
from scipy.signal import (
    resample_poly, fftconvolve, butter, sosfiltfilt, hilbert
)
from math import gcd
from functools import lru_cache
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


class ConcretePhysicsChain:
    """
    混凝土穿透物理链路完整模拟。
    
    设计要点（性能优先）：
    1. 使用 resample_poly 替代 librosa.resample，避免高阶 polyphase 开销
    2. 使用 fftconvolve（OA 法），对长 IR 最优
    3. 预缓存 butter 滤波器系数，避免每次重新设计
    4. 包络解调使用 |x| + LPF 替代 Hilbert（后者对长信号 FFT 开销大）
    """

    def __init__(self, config: dict, concrete_ir_cache: dict, target_sr: int):
        self.config = config
        self.concrete_ir_cache = concrete_ir_cache
        self.ir_paths = list(concrete_ir_cache.keys())
        self.target_sr = target_sr

        # 预缓存常用的滤波器系数
        self._filter_cache = {}

    def _get_butter_sos(self, cutoff: float, fs: float, order: int = 5, btype: str = 'low'):
        """缓存 Butterworth 滤波器系数，避免重复计算"""
        cache_key = (cutoff, fs, order, btype)
        if cache_key not in self._filter_cache:
            nyq = fs / 2.0
            # 防止截止频率超过 Nyquist
            norm_cutoff = min(cutoff / nyq, 0.99)
            self._filter_cache[cache_key] = butter(order, norm_cutoff, btype=btype, output='sos')
        return self._filter_cache[cache_key]

    def _fast_resample(self, signal: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
        """
        高效整数比重采样。
        resample_poly 使用 polyphase 滤波，比 FFT-based resample 快得多。
        """
        if from_sr == to_sr:
            return signal
        # 化简分数，减少 polyphase 分支数
        g = gcd(from_sr, to_sr)
        up = to_sr // g
        down = from_sr // g

        # 对于极端比率（如 44100→250000），限制最大 polyphase 分支
        # 如果比率过大，分段重采样
        if up > 500 or down > 500:
            # 使用中间采样率分两步
            mid_sr = from_sr * 4  # 先上采样4倍
            if mid_sr >= to_sr:
                g2 = gcd(from_sr, to_sr)
                return resample_poly(signal, to_sr // g2, from_sr // g2)
            else:
                signal = resample_poly(signal, 4, 1)
                g2 = gcd(mid_sr, to_sr)
                return resample_poly(signal, to_sr // g2, mid_sr // g2)

        return resample_poly(signal, up, down)

    def _jfet_distortion(self, signal: np.ndarray) -> np.ndarray:
        """
        重构版：随机动态 JFET 前置放大器非线性失真
        模拟不同电池电量、不同偏置电压下的电路谐波特征
        """
        # 1. 随机化输入驱动力度 (模拟前级增益旋钮)
        # 驱动越大，失真越严重；驱动小，声音相对干净
        drive = np.random.uniform(1.0, 6.0)
        x = signal * drive
        
        # 2. 引入动态直流偏置 (DC Bias Shift)
        # 真实的模拟电路因为电容充放电，大信号会推移工作点，导致失真极度不对称
        bias = np.random.uniform(-0.3, 0.3)
        x_biased = x + bias
        
        # 3. 核心非对称多项式失真 (模拟 JFET 的转移特性曲线)
        # y = x - alpha * x^2 - beta * x^3
        # alpha 决定偶次谐波 (温暖感/浑浊感)，beta 决定奇次谐波 (刺耳感)
        alpha = np.random.uniform(0.05, 0.25) # 强偶次谐波
        beta = np.random.uniform(0.02, 0.1)   # 弱奇次谐波
        
        distorted = x_biased - alpha * (x_biased ** 2) - beta * (x_biased ** 3)
        
        # 4. 模拟电源电压硬截断 (Clipping)
        # 模拟 9V 电池电量不足时的上下轨截断差异
        clip_pos = np.random.uniform(0.7, 0.95)
        clip_neg = np.random.uniform(-0.95, -0.7)
        distorted = np.clip(distorted, clip_neg, clip_pos)
        
        # 移除人为引入的偏置
        distorted = distorted - np.mean(distorted)
        
        # 增益补偿
        distorted = distorted / drive
        
        return distorted.astype(np.float32)

    def _am_modulate(self, signal: np.ndarray) -> np.ndarray:
        """
        重构版：高物理还原度的随机 AM 调制与包络畸变
        模拟 PZT/微波传感器 接收到的不稳定反射波及其解调瑕疵
        """
        # 1. 随机化调制深度 m (真实场景下，距离越远 m 越小)
        # 避免模型死记硬背固定 m 带来的特定信噪比
        m = np.random.uniform(0.3, 0.95)
        
        # 2. 模拟环境气流或人员走动带来的低频包络波动 (Fading)
        # 产生 0.1Hz 到 2Hz 的极其缓慢的波动
        fading_freq = np.random.uniform(0.1, 2.0) 
        t = np.arange(len(signal)) / self.target_sr
        # 波动幅度控制在 5%~15% 之间
        fading_amp = np.random.uniform(0.05, 0.15)
        fading_envelope = 1.0 + fading_amp * np.sin(2 * np.pi * fading_freq * t)
        
        # 3. 执行调幅模拟
        # 基础包络
        envelope = (1.0 + m * signal) * fading_envelope
        
        # 4. 模拟廉价包络检波器（二极管）的非线性畸变 (Square-law distortion)
        # 当信号幅度很小时，二极管处于平方律区，产生二次谐波
        if np.random.random() < 0.5: # 50% 概率触发检波器劣质化
            distortion_factor = np.random.uniform(0.05, 0.2)
            envelope = envelope + distortion_factor * (envelope ** 2)
            
        # 移除直流偏置 (DC Block)，只保留交流音频成分
        modulated = envelope - np.mean(envelope)
        
        # 能量归一化保护
        peak = np.max(np.abs(modulated))
        if peak > 0:
            modulated = (modulated / peak) * 0.95
            
        return modulated.astype(np.float32)

    # ...existing code...

    def _concrete_ir_convolve(self, signal: np.ndarray) -> np.ndarray:
        """
        升级版：频域随机扰动替代时域拉伸。
        
        物理依据：
        - 混凝土墙体的传递函数 = IR 的频率响应
        - 不同墙厚/配筋/含水率 → 不同的频率衰减曲线
        - 直接在频域随机化 IR 的幅度/相位，等价于模拟不同墙体条件
        - 比 resample_poly 快 10-50 倍（只需一次 FFT）
        """
        if len(self.ir_paths) == 0:
            return signal

        # 1. 随机选择基础 IR
        ir_idx = np.random.randint(0, len(self.ir_paths))
        ir = self.concrete_ir_cache[self.ir_paths[ir_idx]].copy()

        # =======================================================
        # 核心增强 A：频域随机扰动（替代 resample 拉伸）
        # =======================================================
        ir = self._randomize_ir_spectral(ir)

        # =======================================================
        # 核心增强 B：随机截断/延伸 IR 长度（模拟不同墙厚）
        # =======================================================
        # 厚墙 → IR 更长（能量衰减慢）；薄墙 → IR 更短
        length_factor = np.random.uniform(0.6, 1.4)
        new_len = max(int(len(ir) * length_factor), 64)
        if new_len < len(ir):
            # 截断 + 淡出（避免截断噪声）
            ir = ir[:new_len]
            fade_len = min(32, new_len // 4)
            if fade_len > 1:
                ir[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
        elif new_len > len(ir):
            # 指数衰减尾部延伸
            tail_len = new_len - len(ir)
            tail_amp = ir[-1] if len(ir) > 0 else 0.0
            decay = np.exp(-np.linspace(0, 5, tail_len)) * tail_amp
            noise_tail = np.random.randn(tail_len).astype(np.float32) * 0.001
            ir = np.concatenate([ir, (decay + noise_tail).astype(np.float32)])

        # =======================================================
        # 核心增强 C：频率依赖吸收（墙体材质滤波，不变）
        # =======================================================
        if np.random.random() < 0.7:
            cutoff_freq = np.random.uniform(1000, 8000)
            sos = self._get_butter_sos(
                cutoff_freq, self.target_sr,
                order=random.choice([2, 4]), btype='low'
            )
            ir = sosfiltfilt(sos, ir)

        # =======================================================
        # 核心增强 D：微观噪声扰动（不变）
        # =======================================================
        ir_noise_level = np.random.uniform(0.0001, 0.002)
        ir_peak = np.max(np.abs(ir))
        if ir_peak > 0:
            ir_noise = np.random.randn(len(ir)).astype(np.float32) * ir_noise_level * ir_peak
            ir = ir + ir_noise

        # =======================================================
        # 核心增强 E：指数包络衰减（不变）
        # =======================================================
        damp_factor = np.random.uniform(0.0, 2.0)
        if damp_factor > 0.1:
            envelope = np.exp(-damp_factor * np.linspace(0, 1, len(ir)))
            ir = (ir * envelope).astype(np.float32)

        # =======================================================
        # 物理卷积与防爆音
        # =======================================================
        convolved = fftconvolve(signal, ir, mode='full')[:len(signal)]

        peak = np.max(np.abs(convolved))
        if peak > 0.99:
            convolved = (convolved / peak) * 0.95

        return convolved.astype(np.float32)

    def _randomize_ir_spectral(self, ir: np.ndarray) -> np.ndarray:
        """
        频域随机扰动 IR，生成物理上合理的新 IR 变体。
        
        三种独立的频域变换，每种概率独立触发：
        
        1. 幅度随机调制：模拟不同配筋密度导致的频率选择性衰减
           - 用低阶随机多项式调制幅度谱
           - 物理含义：钢筋间距不同 → 不同频率的衍射/散射特性不同
           
        2. 相位随机扰动：模拟墙体内部微观不均匀性
           - 对相位加小幅高斯噪声
           - 物理含义：骨料分布随机 → 不同频率的传播速度微扰
           
        3. 频带随机衰减：模拟墙体共振/反共振
           - 随机选 1-3 个频带做 notch 衰减
           - 物理含义：墙体固有模态在特定频率产生驻波节点
        """
        n = len(ir)
        if n < 16:
            return ir

        # FFT
        IR_fft = np.fft.rfft(ir)
        magnitude = np.abs(IR_fft)
        phase = np.angle(IR_fft)
        n_bins = len(IR_fft)

        # ---- 变换 1：低阶幅度调制 (80% 概率) ----
        if np.random.random() < 0.8:
            # 生成 4-8 个控制点的随机曲线
            n_ctrl = np.random.randint(4, 9)
            ctrl_points = np.random.uniform(0.5, 1.5, size=n_ctrl)
            # 确保首尾接近 1.0（低频和高频变化不应太剧烈）
            ctrl_points[0] = np.random.uniform(0.8, 1.2)
            ctrl_points[-1] = np.random.uniform(0.6, 1.0)
            # 插值到所有频率 bin
            x_ctrl = np.linspace(0, n_bins - 1, n_ctrl)
            x_full = np.arange(n_bins)
            mod_curve = np.interp(x_full, x_ctrl, ctrl_points).astype(np.float32)
            magnitude = magnitude * mod_curve

        # ---- 变换 2：相位微扰 (60% 概率) ----
        if np.random.random() < 0.6:
            # 相位噪声强度：0.05-0.3 弧度（约 3°-17°）
            phase_noise_std = np.random.uniform(0.05, 0.3)
            phase_noise = np.random.randn(n_bins).astype(np.float32) * phase_noise_std
            # DC 分量不扰动
            phase_noise[0] = 0.0
            phase = phase + phase_noise

        # ---- 变换 3：频带 Notch 衰减 (50% 概率) ----
        if np.random.random() < 0.5:
            n_notches = np.random.randint(1, 4)
            freq_resolution = self.target_sr / (2.0 * n_bins)  # Hz per bin
            for _ in range(n_notches):
                # 随机选择中心频率 (200Hz ~ 6000Hz)
                center_hz = np.random.uniform(200, 6000)
                center_bin = int(center_hz / freq_resolution)
                center_bin = np.clip(center_bin, 1, n_bins - 2)
                # 随机带宽 (50Hz ~ 500Hz)
                bw_hz = np.random.uniform(50, 500)
                bw_bins = max(int(bw_hz / freq_resolution), 1)
                # 随机衰减深度 (3dB ~ 15dB)
                depth_db = np.random.uniform(3, 15)
                attenuation = 10 ** (-depth_db / 20.0)
                # 高斯形状的 notch
                bin_lo = max(center_bin - bw_bins * 2, 0)
                bin_hi = min(center_bin + bw_bins * 2, n_bins)
                for b in range(bin_lo, bin_hi):
                    dist = abs(b - center_bin) / max(bw_bins, 1)
                    # 高斯衰减：中心最深，两侧渐变
                    notch_gain = 1.0 - (1.0 - attenuation) * np.exp(-0.5 * dist ** 2)
                    magnitude[b] *= notch_gain

        # IFFT 重建
        IR_fft_new = magnitude * np.exp(1j * phase)
        ir_new = np.fft.irfft(IR_fft_new, n=n)

        return ir_new.astype(np.float32)

    # ...existing code...

    def _envelope_demodulate(self, signal: np.ndarray, fs: float, lpf_cutoff: float) -> np.ndarray:
        """
        包络检波解调。
        
        方案对比：
        - Hilbert 变换：精确但对长信号 O(N log N) 且内存翻倍
        - |x| + LPF：经典无线电检波，计算量小，适合 DataLoader
        
        这里选用 |x| + LPF（整流检波），符合真实硬件。
        """
        # 全波整流
        envelope = np.abs(signal)

        # 低通滤波恢复基带信号
        sos = self._get_butter_sos(lpf_cutoff, fs, order=5, btype='low')
        demodulated = sosfiltfilt(sos, envelope).astype(np.float32)

        # 去除直流分量
        demodulated -= np.mean(demodulated)

        # 【修复】：仅做防溢出保护，保留真实的包络能量比例
        peak = np.max(np.abs(demodulated))
        if peak > 0.99:
            demodulated = (demodulated / peak) * 0.95
        return demodulated.astype(np.float32)

    def _add_emi_noise(self, signal: np.ndarray, fs: float, snr_db: float,
                       emi_freqs: list) -> np.ndarray:
        """
        添加 EMI 干扰：工频 (50Hz) + 高频啸叫 + 热噪声。
        
        真实场景中：
        - 50Hz 来自电源线耦合
        - 1kHz/2048Hz 来自开关电源或数字电路辐射
        - 宽带热噪声来自高增益放大器
        """
        signal_power = np.mean(signal ** 2)
        if signal_power < 1e-10:
            return signal

        t = np.arange(len(signal), dtype=np.float32) / fs
        emi = np.zeros_like(signal)

        # 随机选择 1-3 个 EMI 频率 (增加空列表安全保护)
        if emi_freqs:
            n_emi = np.random.randint(1, min(4, len(emi_freqs) + 1))
            selected_freqs = np.random.choice(emi_freqs, size=n_emi, replace=False)

            for freq in selected_freqs:
                # 每个 EMI 分量有随机幅度和初始相位
                amplitude = np.random.uniform(0.3, 1.0)
                phase = np.random.uniform(0, 2 * np.pi)
                emi += amplitude * np.sin(2 * np.pi * freq * t + phase)

        # 加入宽带高斯热噪声
        thermal_noise = np.random.randn(len(signal)).astype(np.float32) * 0.5
        emi += thermal_noise

        # 按 SNR 调整 EMI 功率
        emi_power = np.mean(emi ** 2)
        if emi_power > 0:
            target_emi_power = signal_power / (10 ** (snr_db / 10))
            emi *= np.sqrt(target_emi_power / emi_power)

        noisy = signal + emi

        # 防削峰
        peak = np.max(np.abs(noisy))
        if peak > 0.99:
            noisy = noisy / peak * 0.98

        return noisy.astype(np.float32)

    def apply(self, signal: np.ndarray, input_sr: int, intensity: float = 1.0) -> np.ndarray:
        """
        内外 PZT 协同的反向散射窃听链路 (无源接收端特化版)
        
        硬件物理链：
        [室内有源端] 声音 + 室内电磁干扰(EMI) → 一起被放大器拾取 → JFET(产生交调与非线性阻抗)
        [穿墙] 墙外PZT发射40kHz → 穿墙 → 打在JFET上被阻抗调制 → 携带基带包络穿墙返回
        [墙外无源端] 墙外无源PZT接收 → 纯物理包络解调
        """
        cfg = self.config
        if not cfg.get("enable", True):
            return signal

        signal = signal.astype(np.float32)

        # 归一化输入
        orig_peak = np.max(np.abs(signal))
        if orig_peak > 0:
            signal = signal / orig_peak

        # ============ Step 1: 室内有源电路拾取与 EMI 底噪 (44.1kHz 基带) ============
        # 【核心修正】：接收端是无源的，所以 50Hz 工频和放大器热噪声必然来自室内的有源发射电路！
        # 这些噪音会被麦克风引线拾取，与人声混合。
        snr_range = cfg["emi_snr_db"]
        low_bound = min(snr_range[0] + (1 - intensity) * 20, snr_range[1])
        snr_db = np.random.uniform(low_bound, snr_range[1])
        signal = self._add_emi_noise(
            signal, input_sr, snr_db, cfg["emi_freqs"]
        )

        # ============ Step 2: JFET 放大器非线性与交调失真 (44.1kHz 基带) ============
        # 混入了 EMI 的人声共同驱动 JFET。
        # 此时会产生极度逼真的“交调失真(IMD)”，即人声的包络会被 50Hz 进一步撕裂。
        jfet_gain = cfg["jfet_gain"] * intensity
        harmonic = cfg["jfet_harmonic"] * intensity
        if jfet_gain > 0.01:
            signal = self._jfet_distortion(signal, jfet_gain, harmonic)

        # ============ Step 3: 混凝土双向穿透的基带等效 (44.1kHz 基带) ============
        # 载波在墙壁里一来一回穿透了两次，高频超声波受到的多径衰减，
        # 等效于将墙体的基带 IR 直接卷积在低频音频包络上。
        if intensity > 0.3 and len(self.ir_paths) > 0:
            signal = self._concrete_ir_convolve(signal)

        # ============ Step 4: 升采样至超声波/射频域 (250kHz) ============
        high_sr = cfg["high_sr"]
        signal_hf = self._fast_resample(signal, input_sr, high_sr)

        # ============ Step 5: 墙外 PZT 40kHz 载波调制与反向散射 ============
        # 墙外 PZT 发出的 40kHz 载波，被室内 JFET 的阻抗变化进行了幅度调制 (AM)
        signal_am = self._am_modulate(signal_hf, cfg["carrier_freq"], high_sr)

        # ============ Step 6: 墙外无源 PZT 接收与包络检波 ============
        # 墙外 PZT 接收到含有音频包络的 40kHz 反射波，进行检波解调
        demod_cutoff = cfg["demod_lpf_cutoff"]
        signal_demod = self._envelope_demodulate(signal_am, high_sr, demod_cutoff)

        # ============ Step 7: 降采样回声学基带 ============
        # 无源 PZT 直接输出基带信号，进入采集卡 ADC
        signal_out = self._fast_resample(signal_demod, high_sr, self.target_sr)

        # 恢复原始能量尺度
        signal_out = signal_out * orig_peak

        return signal_out