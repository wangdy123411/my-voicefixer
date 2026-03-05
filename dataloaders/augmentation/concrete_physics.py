# -*- coding: utf-8 -*-
# @Time    : 2026/2/24
# @Author  : Concrete Eavesdrop System 3022234317@tju.deu.cn
# @FileName: concrete_physics.py
# 
# 物理链路降级模拟：
# 干净语音 → JFET非线性失真 → 40kHz AM调制 → 混凝土IR卷积 → 包络解调 → EMI叠加

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

    def _jfet_distortion(self, signal: np.ndarray, gain: float, harmonic_coeff: float) -> np.ndarray:
        """
        JFET 前置放大器非线性失真模拟。
        模型：soft clipping (tanh) + 二次谐波失真
        
        物理依据：JFET 工作在饱和区时，传输特性近似:
        I_ds = I_dss * (1 - V_gs/V_p)^2
        简化为 tanh 软削峰 + 偶次谐波
        """
        x = signal * gain
        # 软削峰 + 二次谐波
        distorted = np.tanh(x) + harmonic_coeff * (x ** 2)
       # 去除直流偏移（二次谐波引入）
        distorted -= np.mean(distorted)
        # 【修复】：取消绝对归一化，改为峰值防爆音限制
        peak = np.max(np.abs(distorted))
        if peak > 0.99:
            distorted = (distorted / peak) * 0.95
        return distorted.astype(np.float32)

    def _am_modulate(self, signal: np.ndarray, carrier_freq: float, fs: float) -> np.ndarray:
        """
        标准 AM 调制（双边带全载波 DSB-FC）。
        s_am(t) = [1 + m * x(t)] * cos(2π * f_c * t)
        调制度 m 设为 0.8，避免过调制
        """
        m = 0.8  # 调制度
        t = np.arange(len(signal), dtype=np.float32) / fs
        carrier = np.cos(2 * np.pi * carrier_freq * t, dtype=np.float32)
        modulated = (1.0 + m * signal) * carrier
        return modulated

    def _concrete_ir_convolve(self, signal: np.ndarray) -> np.ndarray:
        """
        使用混凝土 IR 进行卷积。
        【进阶版 IR 数据增强】：
          1. 随机时间轴拉伸：等效模拟不同厚度和密度的墙体。
          2. 指数包络衰减：等效模拟不同材质（如空心砖、吸音棉）的吸音率。
        fftconvolve 使用 overlap-add，对长信号 + 短 IR 最优。
        """
        if len(self.ir_paths) == 0:
            return signal

        # 1. 随机选择一个基础 IR
        ir_idx = np.random.randint(0, len(self.ir_paths))
        ir = self.concrete_ir_cache[self.ir_paths[ir_idx]]

        # =======================================================
        # 2. 核心增强 A：IR 随机时间拉伸 (Time-Stretch)
        # =======================================================
        # 设定拉伸因子：0.8(墙变薄/声速变快) ~ 1.2(墙变厚/声速变慢)
        stretch_factor = np.random.uniform(0.8, 1.2)
        
        if abs(stretch_factor - 1.0) > 0.01:
            orig_len = len(ir)
            new_len = int(orig_len * stretch_factor)
            # 使用 numpy 极速线性插值进行重采样（耗时极低）
            indices = np.linspace(0, orig_len - 1, new_len)
            ir = np.interp(indices, np.arange(orig_len), ir).astype(np.float32)

        # =======================================================
        # 3. 核心增强 B：指数包络衰减 (Damping Envelope)
        # =======================================================
        # damp_factor 越大，尾部余响消失得越快 (模拟松软、吸音的材质，如石膏板/木板)
        # damp_factor 接近 0，则保留原 IR 尾部 (模拟坚硬反射，如纯混凝土/钢板)
        damp_factor = np.random.uniform(0.0, 2.0)
        
        if damp_factor > 0.1:
            # 生成一个从 1 衰减到靠近 0 的指数包络曲线
            envelope = np.exp(-damp_factor * np.linspace(0, 1, len(ir)))
            ir = ir * envelope

        # =======================================================
        # 4. 物理卷积
        # =======================================================
        # fftconvolve 'full' 模式，然后严格截断到原始信号长度
        convolved = fftconvolve(signal, ir, mode='full')[:len(signal)]

        # 【修复】：保留真实的物理衰减，仅做防溢出限制
        peak = np.max(np.abs(convolved))
        if peak > 0.99:
            convolved = (convolved / peak) * 0.95
            
        return convolved.astype(np.float32)

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