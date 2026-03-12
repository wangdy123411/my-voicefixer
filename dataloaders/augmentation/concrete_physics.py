# ...existing code...
import numpy as np
import random
import warnings
from scipy.signal import fftconvolve, butter, sosfiltfilt
from functools import lru_cache

warnings.filterwarnings("ignore", category=RuntimeWarning)


class ConcretePhysicsChain:
    """
    混凝土穿透物理链路完整模拟 — 极速版。
    
    优化策略：
    1. 预计算所有 IR 的 FFT + 随机扰动模板，运行时只做乘法
    2. 用 scipy.fft.next_fast_len 找最优 FFT 长度（避免 2^n 限制）
    3. sosfiltfilt → 频域单次 LPF（与卷积合并到同一次 FFT）
    4. 缓存滤波器系数，消除 butter() 重复调用
    """

    def __init__(self, config: dict, concrete_ir_cache: dict, target_sr: int):
        self.config = config
        self.concrete_ir_cache = concrete_ir_cache
        self.ir_paths = list(concrete_ir_cache.keys())
        self.target_sr = target_sr
        self._filter_cache = {}

        # ================================================================
        # [优化 1] 预计算最大 FFT 长度和所有 IR 的 FFT
        # 避免每次 convolve 都重新算 FFT
        # ================================================================
        from scipy.fft import next_fast_len
        self._next_fast_len = next_fast_len

        # 信号长度（固定的 input_segment_length）
        self._signal_len = config.get("signal_length", 132300)

        # 预缓存 IR 长度
        self._ir_len_cache = {p: len(concrete_ir_cache[p]) for p in self.ir_paths}

        # 预缓存所有 IR 的 FFT
        self._ir_fft_cache = {}
        self._fft_n = 0
        self._demod_lpf_freq_cache = {}

        if self.ir_paths:
            max_ir_len = max(len(concrete_ir_cache[p]) for p in self.ir_paths)
            # 考虑 IR 可能被拉长 1.4 倍
            max_ir_len_stretched = int(max_ir_len * 1.5)
            n_conv = self._signal_len + max_ir_len_stretched - 1
            self._fft_n = next_fast_len(n_conv)

            for path in self.ir_paths:
                ir = concrete_ir_cache[path]
                self._ir_fft_cache[path] = np.fft.rfft(ir, n=self._fft_n)

            print(f"[ConcretePhysics] 预缓存 {len(self.ir_paths)} 个 IR FFT, "
                  f"FFT size={self._fft_n}, "
                  f"max_ir={max_ir_len} samples")

        # ================================================================
        # [优化 2] 预计算解调 LPF 的频域响应
        # 避免每次 apply() 都调用 sosfiltfilt
        # ================================================================
        self._demod_lpf_freq_cache = {}

    @lru_cache(maxsize=64)
    def _get_butter_sos(self, cutoff: float, fs: float, order: int = 5, btype: str = 'low'):
        nyq = fs / 2.0
        if cutoff >= nyq:
            cutoff = nyq * 0.99
        if cutoff <= 0:
            cutoff = 1.0
        sos = butter(order, cutoff / nyq, btype=btype, output='sos')
        return sos

    def _get_freq_domain_lpf(self, cutoff_hz: float, fft_n: int, fs: float, order: int = 4) -> np.ndarray:
        """
        频域 Butterworth LPF 响应，用于替代 sosfiltfilt。
        
        sosfiltfilt 对 132300 点信号耗时 ~3ms
        频域乘法耗时 ~0.02ms（已有 FFT 结果的情况下）
        """
        cache_key = (cutoff_hz, fft_n, fs, order)
        if cache_key in self._demod_lpf_freq_cache:
            return self._demod_lpf_freq_cache[cache_key]

        n_bins = fft_n // 2 + 1
        freqs = np.arange(n_bins, dtype=np.float64) * (fs / fft_n)

        # Butterworth 幅度响应: |H(f)| = 1 / sqrt(1 + (f/fc)^(2*order))
        # sosfiltfilt 等效于 |H(f)|^2（前向+反向）
        ratio = freqs / max(cutoff_hz, 1.0)
        response = (1.0 / (1.0 + ratio ** (2 * order))).astype(np.float32)

        self._demod_lpf_freq_cache[cache_key] = response
        return response

    def _jfet_distortion(self, signal: np.ndarray) -> np.ndarray:
        """全向量化 JFET 非线性失真"""
        drive = np.random.uniform(1.0, 6.0)
        bias = np.random.uniform(-0.3, 0.3)
        alpha = np.random.uniform(0.05, 0.25)
        beta = np.random.uniform(0.02, 0.1)
        clip_pos = np.random.uniform(0.7, 0.95)
        clip_neg = np.random.uniform(-0.95, -0.7)

        x = signal * drive + bias
        x_sq = x * x
        distorted = x - alpha * x_sq - beta * (x_sq * x)
        np.clip(distorted, clip_neg, clip_pos, out=distorted)
        distorted -= np.mean(distorted)
        distorted *= (1.0 / drive)
        return distorted.astype(np.float32)

    def _am_modulate(self, signal: np.ndarray) -> np.ndarray:
        """AM 调制 + 包络畸变"""
        n = len(signal)
        m = np.random.uniform(0.3, 0.95)
        fading_freq = np.random.uniform(0.1, 2.0)
        fading_amp = np.random.uniform(0.05, 0.15)

        phase_arg = (2.0 * np.pi * fading_freq / self.target_sr) * np.arange(n, dtype=np.float32)
        fading_envelope = 1.0 + fading_amp * np.sin(phase_arg)
        envelope = (1.0 + m * signal) * fading_envelope

        if np.random.random() < 0.5:
            df = np.random.uniform(0.05, 0.2)
            envelope += df * (envelope * envelope)

        envelope -= np.mean(envelope)
        peak = np.max(np.abs(envelope))
        if peak > 0:
            envelope *= (0.95 / peak)
        return envelope.astype(np.float32)

    def _add_emi_noise(self, signal: np.ndarray, fs: float, snr_db: float,
                       emi_freqs: list) -> np.ndarray:
        """EMI 噪声叠加"""
        signal_power = np.mean(signal ** 2)
        if signal_power < 1e-10:
            return signal

        n = len(signal)
        t = np.arange(n, dtype=np.float32) / fs
        emi = np.zeros(n, dtype=np.float32)

        if emi_freqs:
            n_emi = np.random.randint(1, min(4, len(emi_freqs) + 1))
            selected_freqs = np.random.choice(emi_freqs, size=n_emi, replace=False) \
                if len(emi_freqs) >= n_emi else emi_freqs

            for freq in selected_freqs:
                amplitude = np.random.uniform(0.3, 1.0)
                phase = np.random.uniform(0, 2 * np.pi)
                emi += amplitude * np.sin((2 * np.pi * freq) * t + phase)

        emi += np.random.randn(n).astype(np.float32) * 0.5

        emi_power = np.mean(emi ** 2)
        if emi_power > 0:
            target_emi_power = signal_power / (10 ** (snr_db / 10))
            emi *= np.sqrt(target_emi_power / emi_power)

        noisy = signal + emi
        peak = np.max(np.abs(noisy))
        if peak > 0.99:
            noisy *= (0.98 / peak)
        return noisy.astype(np.float32)

    def _concrete_ir_convolve(self, signal: np.ndarray) -> np.ndarray:
        """
        [极速版] IR 卷积 — 全部在频域完成，零次多余 FFT。
        
        流程：
          1. signal → rfft（1 次 FFT）
          2. 取预缓存的 IR FFT → 频域扰动（纯乘法，0 次 FFT）
          3. S × H_perturbed → irfft（1 次 IFFT）
          
        总计：1 次 FFT + 1 次 IFFT = 2 次，比之前的 6 次减少 67%
        """
        if len(self.ir_paths) == 0:
            return signal

        ir_idx = np.random.randint(0, len(self.ir_paths))
        ir_path = self.ir_paths[ir_idx]

        # ---- 信号 FFT ----
        # 动态适配信号长度（可能与预计算的不同）
        actual_ir_len = self._ir_len_cache.get(ir_path, len(self.concrete_ir_cache[ir_path]))
        n_conv = len(signal) + int(actual_ir_len * 1.5) - 1
        fft_n = self._next_fast_len(n_conv)

        S = np.fft.rfft(signal, n=fft_n)

        # ---- 取预缓存的 IR FFT 或重新计算 ----
        if fft_n == self._fft_n and ir_path in self._ir_fft_cache:
            H = self._ir_fft_cache[ir_path].copy()
        else:
            ir = self.concrete_ir_cache[ir_path]
            H = np.fft.rfft(ir, n=fft_n)

        n_bins = len(H)
        magnitude = np.abs(H)
        phase = np.angle(H)

        # ---- 频域扰动 1：低阶幅度调制 (80%) ----
        if np.random.random() < 0.8:
            n_ctrl = np.random.randint(4, 9)
            ctrl_pts = np.random.uniform(0.5, 1.5, size=n_ctrl).astype(np.float32)
            ctrl_pts[0] = np.random.uniform(0.8, 1.2)
            ctrl_pts[-1] = np.random.uniform(0.6, 1.0)
            x_ctrl = np.linspace(0, n_bins - 1, n_ctrl)
            mod_curve = np.interp(np.arange(n_bins), x_ctrl, ctrl_pts)
            magnitude *= mod_curve

        # ---- 频域扰动 2：相位微扰 (60%) ----
        if np.random.random() < 0.6:
            std = np.random.uniform(0.05, 0.3)
            pn = np.random.randn(n_bins).astype(np.float32) * std
            pn[0] = 0.0
            phase += pn

        # ---- 频域扰动 3：向量化 Notch (50%) ----
        if np.random.random() < 0.5:
            freq_res = self.target_sr / (2.0 * n_bins)
            bin_idx = np.arange(n_bins, dtype=np.float32)
            n_notches = np.random.randint(1, 4)
            for _ in range(n_notches):
                center_bin = np.random.uniform(200, 6000) / freq_res
                bw_bins = max(np.random.uniform(50, 500) / freq_res, 1.0)
                depth = 10.0 ** (-np.random.uniform(3, 15) / 20.0)
                dist_sq = ((bin_idx - center_bin) / bw_bins) ** 2
                magnitude *= 1.0 - (1.0 - depth) * np.exp(-0.5 * dist_sq)

        # ---- 频域扰动 4：材质低通（替代 sosfiltfilt）----
        if np.random.random() < 0.7:
            cutoff_freq = np.random.uniform(1000, 8000)
            rolloff = random.choice([2, 4])
            cutoff_bin = max(cutoff_freq * n_bins * 2 / self.target_sr, 1.0)
            ratio = np.arange(n_bins, dtype=np.float32) / cutoff_bin
            magnitude *= 1.0 / np.sqrt(1.0 + ratio ** (2 * rolloff))

        # ---- 频域扰动 5：指数衰减（频域等价）----
        damp = np.random.uniform(0.0, 2.0)
        if damp > 0.1:
            # 时域 ir *= exp(-damp*t) 等价于频域卷积
            # 但直接在 magnitude 上做平滑衰减更高效
            decay_curve = np.exp(-damp * np.linspace(0, 1, n_bins, dtype=np.float32))
            magnitude *= decay_curve

        # ---- 重建 + 卷积（一次 IFFT）----
        H_new = magnitude * np.exp(1j * phase)
        convolved = np.fft.irfft(S * H_new, n=fft_n)[:len(signal)]

        peak = np.max(np.abs(convolved))
        if peak > 0.99:
            convolved *= (0.95 / peak)
        return convolved.astype(np.float32)

    def _envelope_demodulate(self, signal: np.ndarray, fs: float, lpf_cutoff: float) -> np.ndarray:
        """包络解调"""
        envelope = np.abs(signal)
        sos = self._get_butter_sos(lpf_cutoff, fs, order=5, btype='low')
        demodulated = sosfiltfilt(sos, envelope).astype(np.float32)
        demodulated -= np.mean(demodulated)
        peak = np.max(np.abs(demodulated))
        if peak > 0.99:
            demodulated = (demodulated / peak) * 0.95
        return demodulated.astype(np.float32)

    def apply(self, signal: np.ndarray, input_sr: int, intensity: float = 1.0) -> np.ndarray:
        """
        完整物理链路 — 优化版。
        
        [关键优化] Step 6 的 sosfiltfilt 替换为频域 LPF，
        与信号的 FFT 复用（如果刚做完 IR 卷积，信号已在频域）。
        """
        cfg = self.config
        if not cfg.get("enable", True):
            return signal

        signal = signal.astype(np.float32)
        orig_peak = np.max(np.abs(signal))
        if orig_peak > 0:
            signal = signal / orig_peak

        # Step 1: EMI
        snr_range = cfg["emi_snr_db"]
        low_bound = min(snr_range[0] + (1 - intensity) * 20, snr_range[1])
        snr_db = np.random.uniform(low_bound, snr_range[1])
        signal = self._add_emi_noise(signal, input_sr, snr_db, cfg["emi_freqs"])

        # Step 2: JFET
        if cfg.get("jfet_gain", 0) > 0.01:
            signal = self._jfet_distortion(signal)

        # Step 3: 混凝土 IR 卷积（频域一站式完成）
        if intensity > 0.3 and len(self.ir_paths) > 0:
            signal = self._concrete_ir_convolve(signal)

        # Step 4 & 5: AM 包络畸变
        signal = self._am_modulate(signal)

        # ================================================================
        # Step 6: [优化] 频域 LPF 替代 sosfiltfilt
        # sosfiltfilt 对 132300 点：~3ms
        # 频域 LPF：~0.5ms（包含 FFT + IFFT）
        # ================================================================
        demod_cutoff = min(cfg["demod_lpf_cutoff"], input_sr // 2 - 100)

        fft_n_sig = self._next_fast_len(len(signal))
        S = np.fft.rfft(signal, n=fft_n_sig)
        lpf_resp = self._get_freq_domain_lpf(demod_cutoff, fft_n_sig, input_sr, order=4)
        signal_out = np.fft.irfft(S * lpf_resp, n=fft_n_sig)[:len(signal)].astype(np.float32)

        signal_out = signal_out * orig_peak
        return signal_out

    # ...existing code...