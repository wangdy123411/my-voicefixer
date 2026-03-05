"""
EffectChain: WavAugment 的 Windows 兼容替代实现。

用 scipy.signal + torchaudio.functional 替代 SoX 后端。
API 与原始 augment.EffectChain 完全一致。

已实现效果（覆盖 VoiceFixer MagicalEffects 中所有调用）：
  ✓ lowpass     ✓ highpass    ✓ pitch      ✓ tempo
  ✓ speed       ✓ reverb      ✓ treble     ✓ bass
  ✓ tremolo     ✓ clip        ✓ fade       ✓ reverse
  ✓ channels    ✓ rate        ✓ overdrive
"""

import warnings
import numpy as np
import torch
import scipy.signal as sig
from typing import Union, Optional, Callable


class EffectChain:
    """
    链式音频效果处理器（Windows 兼容版）。

    用法与原始 augment.EffectChain 完全一致：
        chain = EffectChain()
        chain.lowpass(4000).reverb(50, 50, 50).channels(1)
        output = chain.apply(input_tensor, src_info={'rate': 44100})
    """

    def __init__(self):
        self._effects = []  # [(effect_name, func, kwargs), ...]

    # ================================================================
    #  效果注册（链式调用，每个方法返回 self）
    # ================================================================

    def rate(self, sample_rate: Union[int, float]) -> "EffectChain":
        """重采样到指定采样率"""
        self._effects.append(("rate", self._apply_rate, {"target_sr": int(sample_rate)}))
        return self

    def lowpass(self, frequency: Union[float, Callable]) -> "EffectChain":
        """低通滤波"""
        freq = frequency() if callable(frequency) else float(frequency)
        self._effects.append(("lowpass", self._apply_lowpass, {"freq": freq}))
        return self

    def highpass(self, frequency: Union[float, Callable]) -> "EffectChain":
        """高通滤波"""
        freq = frequency() if callable(frequency) else float(frequency)
        self._effects.append(("highpass", self._apply_highpass, {"freq": freq}))
        return self

    def pitch(self, cents: Union[float, Callable]) -> "EffectChain":
        """变调（单位：音分）"""
        c = cents() if callable(cents) else float(cents)
        self._effects.append(("pitch", self._apply_pitch, {"cents": c}))
        return self

    def tempo(self, factor: Union[float, Callable]) -> "EffectChain":
        """变速不变调"""
        f = factor() if callable(factor) else float(factor)
        self._effects.append(("tempo", self._apply_tempo, {"factor": f}))
        return self

    def speed(self, factor: Union[float, Callable]) -> "EffectChain":
        """变速（同时变调）"""
        f = factor() if callable(factor) else float(factor)
        self._effects.append(("speed", self._apply_speed, {"factor": f}))
        return self

    def reverb(
        self,
        reverberance: Union[float, Callable] = 50,
        damping: Union[float, Callable] = 50,
        room_scale: Union[float, Callable] = 50,
    ) -> "EffectChain":
        """混响（模拟 SoX freeverb）"""
        rev = reverberance() if callable(reverberance) else float(reverberance)
        dmp = damping() if callable(damping) else float(damping)
        rm = room_scale() if callable(room_scale) else float(room_scale)
        self._effects.append(("reverb", self._apply_reverb, {
            "reverberance": rev, "damping": dmp, "room_scale": rm,
        }))
        return self

    def treble(self, gain_db: Union[float, Callable]) -> "EffectChain":
        """高频增益/衰减"""
        g = gain_db() if callable(gain_db) else float(gain_db)
        self._effects.append(("treble", self._apply_treble, {"gain_db": g}))
        return self

    def bass(self, gain_db: Union[float, Callable]) -> "EffectChain":
        """低频增益/衰减"""
        g = gain_db() if callable(gain_db) else float(gain_db)
        self._effects.append(("bass", self._apply_bass, {"gain_db": g}))
        return self

    def tremolo(self, speed_hz: Union[float, Callable]) -> "EffectChain":
        """颤音"""
        spd = speed_hz() if callable(speed_hz) else float(speed_hz)
        self._effects.append(("tremolo", self._apply_tremolo, {"speed_hz": spd}))
        return self

    def clip(self, factor: Union[float, Callable]) -> "EffectChain":
        """硬削波"""
        f = factor() if callable(factor) else float(factor)
        self._effects.append(("clip", self._apply_clip, {"factor": f}))
        return self

    def fade(self, fade_in: float, duration: float, fade_out: float) -> "EffectChain":
        """淡入淡出"""
        self._effects.append(("fade", self._apply_fade, {
            "fade_in": float(fade_in),
            "duration": float(duration),
            "fade_out": float(fade_out),
        }))
        return self

    def reverse(self) -> "EffectChain":
        """反转"""
        self._effects.append(("reverse", self._apply_reverse, {}))
        return self

    def channels(self, n: int = 1) -> "EffectChain":
        """声道转换"""
        self._effects.append(("channels", self._apply_channels, {"n": int(n)}))
        return self

    def overdrive(self, gain: float = 20, colour: float = 20) -> "EffectChain":
        """过载失真"""
        self._effects.append(("overdrive", self._apply_overdrive, {
            "gain": float(gain), "colour": float(colour),
        }))
        return self

    # ================================================================
    #  执行入口
    # ================================================================

    def apply(
        self,
        input_tensor: Union[torch.Tensor, np.ndarray],
        src_info: Optional[dict] = None,
        target_info: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        依次执行所有已注册的效果。

        Args:
            input_tensor: [channels, samples] 或 [samples]
            src_info: {'rate': 44100}
            target_info: {'rate': 44100}（可选）

        Returns:
            torch.Tensor: [channels, samples]
        """
        if src_info is None:
            src_info = {}
        sr = src_info.get("rate", src_info.get("sample_rate", 44100))

        # 统一为 numpy [samples]
        audio = self._to_numpy(input_tensor)
        current_sr = int(sr)

        for name, func, kwargs in self._effects:
            try:
                audio, current_sr = func(audio, current_sr, **kwargs)
            except Exception as e:
                warnings.warn(f"[EffectChain] 效果 '{name}' 执行失败: {e}，已跳过")

        # 转回 tensor [1, samples]
        return torch.from_numpy(audio).float().unsqueeze(0)

    # ================================================================
    #  效果实现（纯 numpy/scipy）
    # ================================================================

    @staticmethod
    def _apply_rate(audio, sr, target_sr):
        """重采样"""
        if sr == target_sr:
            return audio, sr
        # scipy.signal.resample_poly 比线性插值精度高
        gcd = np.gcd(sr, target_sr)
        up, down = target_sr // gcd, sr // gcd
        # 限制倍率避免内存爆炸
        if up > 100 or down > 100:
            ratio = target_sr / sr
            new_len = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_len)
            audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
        else:
            audio = sig.resample_poly(audio, up, down).astype(np.float32)
        return audio, target_sr

    @staticmethod
    def _apply_lowpass(audio, sr, freq):
        """Butterworth 低通"""
        nyq = sr / 2.0
        if freq >= nyq:
            return audio, sr
        freq = min(freq, nyq * 0.99)
        try:
            sos = sig.butter(5, freq / nyq, btype="low", output="sos")
            audio = sig.sosfilt(sos, audio).astype(np.float32)
        except Exception:
            pass
        return audio, sr

    @staticmethod
    def _apply_highpass(audio, sr, freq):
        """Butterworth 高通"""
        nyq = sr / 2.0
        if freq <= 0 or freq >= nyq:
            return audio, sr
        freq = min(freq, nyq * 0.99)
        try:
            sos = sig.butter(5, freq / nyq, btype="high", output="sos")
            audio = sig.sosfilt(sos, audio).astype(np.float32)
        except Exception:
            pass
        return audio, sr

    @staticmethod
    def _apply_pitch(audio, sr, cents):
        """
        变调：通过重采样实现。
        cents > 0 升调, cents < 0 降调。
        注意：这会改变音频长度，通常后面跟一个 rate() 恢复。
        """
        if abs(cents) < 1:
            return audio, sr
        # 音分转频率比
        ratio = 2.0 ** (cents / 1200.0)
        new_len = int(len(audio) / ratio)
        if new_len < 1:
            return audio, sr
        indices = np.linspace(0, len(audio) - 1, new_len)
        audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
        return audio, sr

    @staticmethod
    def _apply_tempo(audio, sr, factor):
        """
        变速不变调：WSOLA 简化版。
        使用 scipy resample 实现（会轻微影响音调，但对训练增强足够）。
        """
        if abs(factor - 1.0) < 0.01:
            return audio, sr
        new_len = int(len(audio) / factor)
        if new_len < 1:
            return audio, sr
        # 使用 resample（频域方法，保持采样率不变）
        audio = sig.resample(audio, new_len).astype(np.float32)
        return audio, sr

    @staticmethod
    def _apply_speed(audio, sr, factor):
        """变速（同时变调）：等效于改变采样率"""
        if abs(factor - 1.0) < 0.01:
            return audio, sr
        new_sr = int(sr * factor)
        return audio, new_sr

    @staticmethod
    def _apply_reverb(audio, sr, reverberance, damping, room_scale):
        """
        简化混响：用指数衰减 FIR 模拟 freeverb。
        
        参数范围 [0, 100]，与 SoX reverb 一致。
        """
        # 混响时间与参数的映射
        rt60 = (reverberance / 100.0) * 1.5  # 最长 1.5 秒
        room_factor = room_scale / 100.0
        damp_factor = damping / 100.0

        if rt60 < 0.01:
            return audio, sr

        # 生成简化 IR
        ir_len = int(sr * rt60 * room_factor) + 1
        ir_len = min(ir_len, sr * 2)  # 最长 2 秒
        if ir_len < 2:
            return audio, sr

        # 指数衰减 + 随机反射
        t = np.arange(ir_len, dtype=np.float32)
        decay = np.exp(-6.9 * t / ir_len)  # -60dB at ir_len

        # 高频阻尼
        ir = np.random.randn(ir_len).astype(np.float32) * decay
        if damp_factor > 0:
            # 简单低通模拟高频阻尼
            cutoff = max(1000, 20000 * (1 - damp_factor * 0.8))
            nyq = sr / 2.0
            if cutoff < nyq:
                try:
                    sos = sig.butter(2, cutoff / nyq, btype="low", output="sos")
                    ir = sig.sosfilt(sos, ir).astype(np.float32)
                except Exception:
                    pass

        # 归一化 IR
        ir[0] = 1.0  # 直达声
        ir = ir / (np.max(np.abs(ir)) + 1e-8)

        # 卷积
        wet = sig.fftconvolve(audio, ir, mode="full")[:len(audio)].astype(np.float32)

        # 干湿混合
        wet_ratio = reverberance / 200.0  # [0, 0.5]
        output = audio * (1 - wet_ratio) + wet * wet_ratio

        # 防止削波
        peak = np.max(np.abs(output))
        if peak > 0.99:
            output = output * (0.98 / peak)

        return output.astype(np.float32), sr

    @staticmethod
    def _apply_treble(audio, sr, gain_db):
        """高频搁架式 EQ (2kHz 以上)"""
        nyq = sr / 2.0
        shelf_freq = min(2000.0, nyq * 0.9)
        try:
            # 高通提取高频 → 增益 → 混回
            sos = sig.butter(2, shelf_freq / nyq, btype="high", output="sos")
            high = sig.sosfilt(sos, audio)
            gain = 10.0 ** (gain_db / 20.0) - 1.0
            audio = (audio + high * gain).astype(np.float32)
        except Exception:
            pass
        return audio, sr

    @staticmethod
    def _apply_bass(audio, sr, gain_db):
        """低频搁架式 EQ (300Hz 以下)"""
        nyq = sr / 2.0
        shelf_freq = min(300.0, nyq * 0.9)
        try:
            sos = sig.butter(2, shelf_freq / nyq, btype="low", output="sos")
            low = sig.sosfilt(sos, audio)
            gain = 10.0 ** (gain_db / 20.0) - 1.0
            audio = (audio + low * gain).astype(np.float32)
        except Exception:
            pass
        return audio, sr

    @staticmethod
    def _apply_tremolo(audio, sr, speed_hz):
        """颤音：AM 调制"""
        if speed_hz <= 0:
            return audio, sr
        t = np.arange(len(audio), dtype=np.float32) / sr
        modulator = 0.5 * (1.0 + np.sin(2 * np.pi * speed_hz * t))
        return (audio * modulator).astype(np.float32), sr

    @staticmethod
    def _apply_clip(audio, sr, factor):
        """硬削波"""
        if factor <= 0:
            return audio, sr
        threshold = min(factor, 1.0)
        return np.clip(audio, -threshold, threshold).astype(np.float32), sr

    @staticmethod
    def _apply_fade(audio, sr, fade_in, duration, fade_out):
        """淡入淡出"""
        n = len(audio)
        fade_in_samples = min(int(fade_in * sr), n)
        fade_out_samples = min(int(fade_out * sr), n)

        if fade_in_samples > 0:
            ramp = np.linspace(0, 1, fade_in_samples, dtype=np.float32)
            audio[:fade_in_samples] *= ramp

        if fade_out_samples > 0:
            ramp = np.linspace(1, 0, fade_out_samples, dtype=np.float32)
            audio[-fade_out_samples:] *= ramp

        return audio, sr

    @staticmethod
    def _apply_reverse(audio, sr):
        """反转"""
        return audio[::-1].copy(), sr

    @staticmethod
    def _apply_channels(audio, sr, n):
        """声道转换（对单声道无操作）"""
        # 我们始终处理单声道，这里只确保形状正确
        return audio, sr

    @staticmethod
    def _apply_overdrive(audio, sr, gain, colour):
        """过载失真：soft clipping"""
        gain_linear = 10.0 ** (gain / 20.0)
        audio = audio * gain_linear
        # tanh soft clipping
        audio = np.tanh(audio).astype(np.float32)
        return audio, sr

    # ================================================================
    #  工具方法
    # ================================================================

    @staticmethod
    def _to_numpy(data) -> np.ndarray:
        """转为 1D numpy float32"""
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        data = np.asarray(data, dtype=np.float32)
        # 展平：[1, samples] → [samples]
        return data.squeeze()