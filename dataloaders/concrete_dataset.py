# -*- coding: utf-8 -*-
import os
import warnings
import random
from typing import Optional, Tuple, Dict, Any

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from dataloaders.augmentation.base import AudioAug

try:
    from dataloaders.augmentation.base import add_noise_and_scale_with_HQ_with_Aug
    _HAS_ORIG_AUG = True
except ImportError:
    _HAS_ORIG_AUG = False
    warnings.warn(
        "[ConcreteAugDataset] 无法导入 add_noise_and_scale_with_HQ_with_Aug，"
        "30% 通用增强分支将退化为仅 Phase 1"
    )


def worker_init_fn(worker_id: int):
    seed = torch.initial_seed() % (2**31)
    np.random.seed(seed)
    random.seed(seed)

    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"

    try:
        import psutil
        p = psutil.Process()
        cpu_count = psutil.cpu_count(logical=True)
        core_start = (worker_id * 2) % cpu_count
        cores = [core_start, (core_start + 1) % cpu_count]
        p.cpu_affinity(cores)
    except Exception:
        pass


class ConcreteAugDataset(Dataset):

    # ================================================================
    # [优化核心] Worker 级别的共享缓存
    # persistent_workers=True 时，每个 worker 进程在整个训练过程中
    # 只初始化一次 → 缓存只加载一次，之后全部从内存读取
    # ================================================================
    _worker_noise_cache: Optional[np.ndarray] = None   # 所有噪声拼成一个大数组
    _worker_noise_lengths: Optional[list] = None        # 每段噪声的长度边界
    _worker_bandpass_sos_cache: dict = {}               # butter sos 缓存
    _worker_vocal_cache: Optional[dict] = None          # 热门vocal文件缓存

    def __init__(
        self,
        hp: dict,
        split: str = "train",
        audio_aug: Optional[AudioAug] = None,
        phase2_intensity: float = 0.5,
    ):
        super().__init__()
        self.hp = hp
        self.split = split
        self.audio_aug = audio_aug

        concrete_cfg = hp.get("concrete", {})
        if self.split == "val":
            self.phase2_intensity = 0.5
            self.concrete_ratio = 0.5
        else:
            self.phase2_intensity = phase2_intensity
            self.concrete_ratio = concrete_cfg.get("concrete_ratio", 0.7)

        self.sample_rate = hp["data"]["sampling_rate"]
        self.segment_length = hp["data"].get("segment_length", self.sample_rate)

        dataset_key = "train_dataset" if split == "train" else "val_dataset"
        dataset_cfg = hp["data"].get(dataset_key, {})

        self.vocal_files = self._scan_audio_files(
            dataset_cfg.get("speech", {}).get("vocal", "")
        )
        self.noise_files = self._scan_audio_files(
            dataset_cfg.get("speech", {}).get("noise", "")
        )
        # ================================================================
        # [新增] 验证集模式检测：
        # 如果 val 的 noise 目录存在预生成的退化音频，
        # 则建立 clean↔degraded 配对，跳过所有增强
        # ================================================================
        self.val_paired_mode = False
        self._val_pairs = []  # [(clean_path, degraded_path), ...]

        if split == "val" and self.noise_files:
            self._build_val_pairs()
        
        effects_cfg = hp.get("augment", {}).get("effects", {})
        self.effect_names = list(effects_cfg.keys()) if effects_cfg else None
        self._orig_aug_available = _HAS_ORIG_AUG and len(self.noise_files) > 0

        if len(self.vocal_files) == 0:
            raise RuntimeError(
                f"[{split}] 未找到语音文件，请检查配置路径: "
                f"{dataset_cfg.get('speech', {}).get('vocal', 'N/A')}"
            )

        # 训练集才需要预加载噪声池
        if split == "train":
            self._preload_all_noise()
        else:
            self._noise_pool = np.zeros(self.segment_length * 2, dtype=np.float32)
            self._noise_pool_len = self.segment_length * 2

        if self.val_paired_mode:
            print(
                f"[ConcreteAugDataset/val] ★ 配对模式: "
                f"{len(self._val_pairs)} 对 (clean↔degraded), "
                f"跳过所有增强，极速验证"
            )
        else:
            print(
                f"[ConcreteAugDataset/{split}] "
                f"语音: {len(self.vocal_files)} 文件, "
                f"噪声: {len(self.noise_files)} 文件, "
                f"片段: {self.segment_length}, "
                f"Phase2强度: {self.phase2_intensity}, "
                f"混凝土占比: {self.concrete_ratio:.0%}"
            )
    def _build_val_pairs(self):
        """
        建立验证集 clean↔degraded 配对。
        
        匹配策略：按文件名（去掉扩展名）匹配。
        例如：
          vocal/p001_001.wav  ↔  noise/p001_001.wav
          vocal/p001_002.wav  ↔  noise/p001_002.wav
        
        如果文件名带前缀/后缀差异（如 degraded_p001_001.wav），
        则按包含关系模糊匹配。
        """
        # 建立 noise 文件的 basename → path 索引
        noise_index = {}
        for npath in self.noise_files:
            stem = os.path.splitext(os.path.basename(npath))[0]
            noise_index[stem] = npath
            # 同时存一个去掉常见前缀的版本
            for prefix in ("degraded_", "damaged_", "concrete_", "noisy_"):
                if stem.startswith(prefix):
                    noise_index[stem[len(prefix):]] = npath

        pairs = []
        unmatched_clean = []

        for vpath in self.vocal_files:
            stem = os.path.splitext(os.path.basename(vpath))[0]
            if stem in noise_index:
                pairs.append((vpath, noise_index[stem]))
            else:
                unmatched_clean.append(stem)
        if len(pairs) >= len(self.vocal_files) * 0.5:
            # 超过一半能配对，启用配对模式
            self._val_pairs = pairs
            self.val_paired_mode = True
            if unmatched_clean:
                print(f"  [Val配对] ⚠ {len(unmatched_clean)} 个 clean 文件未匹配到 degraded")
                for s in unmatched_clean[:5]:
                    print(f"    - {s}")
        else:
            # 配对率太低，可能目录结构不对，回退到按索引配对
            # 假设 vocal 和 noise 按排序后一一对应
            min_len = min(len(self.vocal_files), len(self.noise_files))
            if min_len > 0:
                self._val_pairs = list(zip(
                    sorted(self.vocal_files)[:min_len],
                    sorted(self.noise_files)[:min_len],
                ))
                self.val_paired_mode = True
                print(
                    f"  [Val配对] 文件名匹配率低 ({len(pairs)}/{len(self.vocal_files)})，"
                    f"回退为排序索引配对: {min_len} 对"
                )
        
    def _preload_all_noise(self):
        """
        将所有噪声文件拼接成一个连续大数组。
        
        为什么拼成一个数组而不是 list of arrays：
        - list of arrays：random.choice → 各自在不连续内存，cache miss 多
        - 单个连续数组：随机偏移切片，CPU cache 友好，且避免 Python 对象开销
        """
        if not self.noise_files:
            self._noise_pool = np.zeros(self.segment_length * 2, dtype=np.float32)
            self._noise_pool_len = self.segment_length * 2
            return

        chunks = []
        loaded = 0
        failed = 0

        # 最多加载 400 个噪声文件（约 400 × 132300 × 4B ≈ 211MB）
        files_to_load = self.noise_files[:400]
        if self.split == "train":
            random.shuffle(files_to_load)

        for path in files_to_load:
            try:
                info = sf.info(path)
                sr = info.samplerate
                # 直接读取，避免 _load_audio_segment 的额外开销
                audio, _ = sf.read(path, dtype='float32', always_2d=False)
                if audio.ndim > 1:
                    audio = audio[:, 0]
                # resample 如果需要
                if sr != self.sample_rate:
                    audio = self._simple_resample(audio, sr, self.sample_rate)
                # 归一化
                peak = np.max(np.abs(audio))
                if peak > 1e-6:
                    audio = audio / peak
                chunks.append(audio.astype(np.float32))
                loaded += 1
            except Exception:
                failed += 1
                continue

        if not chunks:
            self._noise_pool = np.random.randn(self.segment_length * 4).astype(np.float32) * 0.01
            self._noise_pool_len = len(self._noise_pool)
            return

        # 拼成一个连续大数组
        self._noise_pool = np.concatenate(chunks, axis=0)
        self._noise_pool_len = len(self._noise_pool)

        mem_mb = self._noise_pool_len * 4 / (1024 ** 2)
        print(f"  [噪声预加载] {loaded}/{len(files_to_load)} 文件 → "
              f"{self._noise_pool_len:,} 采样点 ({mem_mb:.0f} MB)")

    def _load_random_noise_fast(self, length: int) -> np.ndarray:
        """
        [极速版] 从预加载的连续数组中随机切片。
        原来每次 ~6ms，现在 ~0.01ms。
        """
        if self._noise_pool_len <= length:
            # 噪声池比所需片段短，循环填充
            repeats = (length // self._noise_pool_len) + 2
            pool_tiled = np.tile(self._noise_pool, repeats)
            return pool_tiled[:length].copy()

        start = random.randint(0, self._noise_pool_len - length)
        return self._noise_pool[start: start + length].copy()

    def __len__(self) -> int:
        if self.val_paired_mode:
            return len(self._val_pairs)
        return len(self.vocal_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.val_paired_mode:
            return self._getitem_val_paired(idx)
        clean = self._load_audio_segment(self.vocal_files[idx])
        use_concrete = random.random() < self.concrete_ratio

        if use_concrete:
            degraded, clean, phase1 = self._augment_concrete(clean)
        else:
            degraded, clean, phase1 = self._augment_voicefixer_original(clean)

        if isinstance(degraded, torch.Tensor):
            degraded = degraded.numpy()
        if isinstance(clean, torch.Tensor):
            clean = clean.numpy()
        if isinstance(phase1, torch.Tensor):
            phase1 = phase1.numpy()

        degraded = np.asarray(degraded, dtype=np.float32)
        clean    = np.asarray(clean,    dtype=np.float32)
        phase1   = np.asarray(phase1,   dtype=np.float32)

        max_len = max(len(degraded), len(clean), len(phase1))

        def apply_fadeout(arr):
            f_len = min(256, len(arr) // 20)
            if f_len > 1:
                arr[-f_len:] *= np.linspace(1.0, 0.0, f_len, dtype=np.float32)
            return arr

        degraded = apply_fadeout(degraded)
        clean    = apply_fadeout(clean)
        phase1   = apply_fadeout(phase1)

        if len(degraded) < max_len:
            degraded = np.pad(degraded, (0, max_len - len(degraded)))
        if len(clean) < max_len:
            clean    = np.pad(clean,    (0, max_len - len(clean)))
        if len(phase1) < max_len:
            phase1   = np.pad(phase1,   (0, max_len - len(phase1)))

        return {
            "input_wave":  degraded,
            "target_wave": clean,
            "phase1_wave": phase1,
        }

    
    def _getitem_val_paired(self, idx: int) -> Dict[str, Any]:
        """
        验证集配对模式：直接读取预生成的 clean 和 degraded。
        零增强、零卷积、零噪声混合 → 极速。
        """
        clean_path, degraded_path = self._val_pairs[idx]

        clean = self._load_audio_segment(clean_path)
        degraded = self._load_audio_segment(degraded_path)

        # 长度对齐
        min_len = min(len(clean), len(degraded), self.segment_length)
        clean = clean[:min_len]
        degraded = degraded[:min_len]

        if min_len < self.segment_length:
            pad = self.segment_length - min_len
            clean = np.pad(clean, (0, pad))
            degraded = np.pad(degraded, (0, pad))

        # phase1 在验证集中不需要，直接用 degraded 占位
        return {
            "input_wave":  degraded,
            "target_wave": clean,
            "phase1_wave": degraded.copy(),
        }
    # ================================================================
    #  路径 A：混凝土增强（高速版）
    # ================================================================
    # ...existing code...

    # ================================================================
    #  路径 A：混凝土增强（高速版）
    # ================================================================
    def _augment_concrete(self, clean: np.ndarray):
        input_frames = clean.copy()

        # Step 1: 加背景噪声
        if self._noise_pool_len > 0:
            if self.split == "train":
                if random.random() < 0.85:
                    input_frames, clean = self._mix_noise(
                        input_frames, clean,
                        snr_range=(-5.0, 35.0),
                        scale_range=(0.6, 1.0),
                    )
            else:
                input_frames, clean = self._mix_noise(
                    input_frames, clean,
                    snr_range=(15.0, 15.0),
                    scale_range=(0.8, 0.8),
                )

        phase1 = input_frames.copy()

        # Step 2: 混凝土 IR 卷积
        # ================================================================
        # [BUG修复] 属性名 concrete_physics → physics_chain
        # base.py 第 84 行: self.physics_chain = ConcretePhysicsChain(...)
        # ================================================================
        use_fast_path = random.random() < 0.7

        if use_fast_path and self.audio_aug is not None and \
           hasattr(self.audio_aug, 'physics_chain') and \
           self.audio_aug.physics_chain is not None and \
           len(getattr(self.audio_aug.physics_chain, 'ir_list', [])) > 0:
            # 快速路径：直接调 ConcretePhysicsChain.apply()
            degraded = self.audio_aug.physics_chain.apply(
                signal=input_frames,
                input_sr=self.sample_rate,
                intensity=self.phase2_intensity,
            )
        elif self.audio_aug is not None:
            # 慢速路径：走完整 audio_aug.augment()
            apply_phase2 = self.phase2_intensity > 0 and not use_fast_path
            degraded, metadata = self.audio_aug.augment(
                frames=input_frames,
                effects=self.effect_names,
                sample_rate=self.sample_rate,
                apply_phase2=apply_phase2,
                phase2_intensity=self.phase2_intensity,
            )
            phase1 = metadata.get("phase1_audio", phase1)
        else:
            degraded = input_frames.copy()

        return degraded, clean, phase1

# ...existing code...

    # ================================================================
    #  路径 B：原始 VoiceFixer 通用增强
    # ================================================================

    def _augment_voicefixer_original(self, clean: np.ndarray):
        if self._orig_aug_available:
            try:
                augfront_np = clean.copy()

                if random.random() < 0.5:
                    augfront_np = self._random_resample_degrade(augfront_np)

                if random.random() < 0.4:
                    augfront_np = self._random_bandpass_fast(augfront_np)

                if self.audio_aug is not None:
                    try:
                        degraded_p1, metadata = self.audio_aug.augment(
                            frames=augfront_np,
                            effects=self.effect_names,
                            sample_rate=self.sample_rate,
                            apply_phase2=False,
                            phase2_intensity=0.0,
                        )
                        augfront_np = degraded_p1
                    except Exception:
                        pass

                # ================================================================
                # [优化 2] 用预加载的噪声池替代磁盘读取
                # ================================================================
                noise_np = self._load_random_noise_fast(len(clean))

                HQ       = torch.from_numpy(clean.copy()).float()
                front    = torch.from_numpy(clean.copy()).float()
                augfront = torch.from_numpy(augfront_np).float()
                noise    = torch.from_numpy(noise_np).float()

                HQ_out, front_out, augfront_out, noise_out, snr, scale = \
                    add_noise_and_scale_with_HQ_with_Aug(
                        HQ, front, augfront, noise,
                        snr_l=-5, snr_h=35,
                        scale_lower=0.6, scale_upper=1.0,
                    )

                degraded    = (augfront_out + noise_out).numpy()
                clean_final = HQ_out.numpy()
                phase1      = augfront_out.numpy()
                return degraded, clean_final, phase1

            except Exception as e:
                warnings.warn(f"[ConcreteAugDataset] 原始增强失败: {e}")
                return self._augment_phase1_only(clean)
        else:
            return self._augment_phase1_only(clean)

    def _random_resample_degrade(self, audio: np.ndarray) -> np.ndarray:
        low_sr = random.choice([8000, 11025, 16000, 22050, 24000, 32000])
        if low_sr >= self.sample_rate:
            return audio
        ratio_down = low_sr / self.sample_rate
        n_down = int(len(audio) * ratio_down)
        if n_down < 100:
            return audio
        indices_down = np.linspace(0, len(audio) - 1, n_down)
        downsampled  = np.interp(indices_down, np.arange(len(audio)), audio)
        indices_up   = np.linspace(0, len(downsampled) - 1, len(audio))
        return np.interp(indices_up, np.arange(len(downsampled)), downsampled).astype(np.float32)

    def _random_bandpass_fast(self, audio: np.ndarray) -> np.ndarray:
        """
        [优化 3] 缓存 butter sos，避免每次重新设计滤波器。
        
        原来：每次调用 butter() ~0.5ms + sosfiltfilt ~1ms = ~1.5ms
        现在：butter() 结果缓存，只剩 sosfiltfilt ~1ms
        
        进一步：用频域 LPF + HPF 替代 sosfiltfilt，降到 ~0.3ms
        """
        try:
            low_cut  = random.choice([100, 200, 300, 500, 800])
            high_cut = random.choice([3400, 4000, 5000, 6000, 8000])

            if high_cut <= low_cut:
                return audio

            nyq  = self.sample_rate / 2.0
            low  = min(low_cut  / nyq, 0.99)
            high = min(high_cut / nyq, 0.99)
            if low >= high:
                return audio

            # ================================================================
            # [优化] 缓存 sos，同参数只计算一次
            # ================================================================
            cache_key = (low_cut, high_cut, self.sample_rate)
            if cache_key not in ConcreteAugDataset._worker_bandpass_sos_cache:
                from scipy.signal import butter
                sos = butter(4, [low, high], btype='band', output='sos')
                ConcreteAugDataset._worker_bandpass_sos_cache[cache_key] = sos
            sos = ConcreteAugDataset._worker_bandpass_sos_cache[cache_key]

            # ================================================================
            # [优化] 频域滤波替代 sosfiltfilt（对 132300 点快 3-5x）
            # sosfiltfilt = 双向IIR，对长信号非常慢
            # 频域 = 一次 FFT + 乘法 + IFFT，与信号长度线性相关
            # ================================================================
            return self._freq_domain_bandpass(audio, low_cut, high_cut)

        except Exception:
            return audio

    def _freq_domain_bandpass(self, audio: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
        """
        频域带通滤波。
        Butterworth 幅度响应直接在频域相乘，等效于零相位滤波。
        """
        n = len(audio)
        # 找最优 FFT 长度
        from scipy.fft import next_fast_len
        fft_n = next_fast_len(n)

        S = np.fft.rfft(audio, n=fft_n)
        n_bins = len(S)
        freqs  = np.arange(n_bins, dtype=np.float32) * (self.sample_rate / fft_n)

        # 高通部分：1 / sqrt(1 + (fc_low/f)^8)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_hp = np.where(freqs > 0, low_hz / freqs, 1e6)
        ratio_hp_safe = np.clip(ratio_hp, 0, 20.0) # 限制最大为 20 倍，20^8 足够大了
        hp_resp = 1.0 / np.sqrt(1.0 + ratio_hp_safe ** 8)

        # 低通部分：1 / sqrt(1 + (f/fc_high)^8)
        ratio_lp = freqs / max(high_hz, 1.0)
        lp_resp  = 1.0 / np.sqrt(1.0 + ratio_lp ** 8)

        # 带通 = 高通 × 低通（sosfiltfilt 等效于幅度平方，这里用单向）
        bp_resp = (hp_resp * lp_resp).astype(np.float32)

        filtered = np.fft.irfft(S * bp_resp, n=fft_n)[:n]
        return filtered.astype(np.float32)

    # ================================================================
    #  回退 + 噪声混合
    # ================================================================

    def _augment_phase1_only(self, clean: np.ndarray):
        input_frames = clean.copy()

        if self._noise_pool_len > 0:
            if self.split == "train":
                if random.random() < 0.85:
                    input_frames, clean = self._mix_noise(
                        input_frames, clean,
                        snr_range=(-5.0, 35.0),
                        scale_range=(0.6, 1.0),
                    )
            else:
                input_frames, clean = self._mix_noise(
                    input_frames, clean,
                    snr_range=(15.0, 15.0),
                    scale_range=(0.8, 0.8),
                )

        if self.audio_aug is not None:
            degraded, metadata = self.audio_aug.augment(
                frames=input_frames,
                effects=self.effect_names,
                sample_rate=self.sample_rate,
                apply_phase2=False,
                phase2_intensity=0.0,
            )
            phase1 = metadata.get("phase1_audio", input_frames.copy())
        else:
            degraded = input_frames.copy()
            phase1   = input_frames.copy()

        return degraded, clean, phase1

    def _mix_noise(
        self,
        input_frames: np.ndarray,
        clean: np.ndarray,
        snr_range: tuple = (-5.0, 35.0),
        scale_range: tuple = (0.6, 1.0),
    ):
        # ================================================================
        # [优化 4] 用快速噪声池替代磁盘读取
        # ================================================================
        noise = self._load_random_noise_fast(clean.shape[0])

        snr_db = random.uniform(*snr_range)
        scale  = random.uniform(*scale_range)

        clean_rms = np.sqrt(np.mean(clean ** 2) + 1e-10)
        noise_rms = np.sqrt(np.mean(noise ** 2) + 1e-10)

        target_noise_rms = clean_rms / (10 ** (snr_db / 20.0))
        noise_scaled     = noise * (target_noise_rms / noise_rms)

        mixed       = (input_frames + noise_scaled) * scale
        clean_scaled = clean * scale

        peak = np.max(np.abs(mixed))
        if peak > 0.99:
            mixed = (mixed / peak) * 0.95

        return mixed, clean_scaled

    # ================================================================
    #  文件扫描与音频读取
    # ================================================================

    @staticmethod
    def _scan_audio_files(directory: str) -> list:
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
        try:
            info = sf.info(filepath)
            total_frames = info.frames
            file_sr      = info.samplerate
        except Exception as e:
            warnings.warn(f"无法读取音频信息 {filepath}: {e}")
            return np.zeros(self.segment_length, dtype=np.float32)

        sr_ratio      = file_sr / self.sample_rate
        needed_frames = int(self.segment_length * sr_ratio)

        if total_frames <= needed_frames:
            start          = 0
            frames_to_read = total_frames
        elif self.split == "train":
            start          = random.randint(0, total_frames - needed_frames)
            frames_to_read = needed_frames
        else:
            start          = 0
            frames_to_read = needed_frames

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

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if sr != self.sample_rate:
            audio = self._simple_resample(audio, sr, self.sample_rate)

        # 长度对齐
        if len(audio) >= self.segment_length:
            audio = audio[:self.segment_length]
        else:
            repeats = (self.segment_length // len(audio)) + 1
            audio   = np.tile(audio, repeats)[:self.segment_length]

        peak = np.max(np.abs(audio))
        if peak > 1e-6:
            audio = audio / peak

        return audio.astype(np.float32)

    def _load_random_noise(self, length: int) -> np.ndarray:
        """兼容旧接口，内部走快速路径"""
        return self._load_random_noise_fast(length)

    @staticmethod
    def _simple_resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio
        new_length = int(len(audio) * target_sr / orig_sr)
        indices    = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    @staticmethod
    def _pad_or_trim(tensor: torch.Tensor, target_length: int) -> torch.Tensor:
        current = tensor.shape[-1]
        if current >= target_length:
            return tensor[..., :target_length]
        pad_size = target_length - current
        return torch.nn.functional.pad(tensor, (0, pad_size), mode="constant", value=0.0)