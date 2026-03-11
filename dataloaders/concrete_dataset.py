# ...existing code...

import os
import warnings
import random
from typing import Optional, Tuple, Dict, Any

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from dataloaders.augmentation.base import AudioAug

# ================================================================
# [抗过拟合] 导入原始 VoiceFixer 增强函数
# ================================================================
try:
    from dataloaders.augmentation.base import add_noise_and_scale_with_HQ_with_Aug
    _HAS_ORIG_AUG = True
except ImportError:
    _HAS_ORIG_AUG = False
    warnings.warn(
        "[ConcreteAugDataset] 无法导入原始 VoiceFixer 增强函数 "
        "add_noise_and_scale_with_HQ_with_Aug，30% 通用增强分支将退化为仅 Phase 1"
    )


class ConcreteAugDataset(Dataset):
    """
    混凝土穿透场景数据集。

    数据流（抗过拟合版）：
    ┌──────────┐
    │ 干净语音  │
    │ .wav文件  │
    └────┬─────┘
         │
         ├─── 70% ──→ Phase 0(加噪) → Phase 1(环境声学) → Phase 2(混凝土物理链路)  → (degraded, clean)
         │
         └─── 30% ──→ 原始 VoiceFixer 增强（add_noise_and_scale_with_HQ_with_Aug） → (degraded, clean)
    
    设计原理：
    - 70% 混凝土链路：保留核心任务的学习能力
    - 30% 通用增强：恢复 VoiceFixer 的通用降噪/去混响/超分辨先验，防止过拟合
    - 验证集：固定 50% / 50% 分流，保证评估基准稳定
    """

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

        # ================================================================
        # [抗过拟合] 混凝土链路占比，从配置读取，默认 0.7
        # ================================================================
        concrete_cfg = hp.get("concrete", {})
        if self.split == "val":
            self.phase2_intensity = 0.5
            # 验证集固定 50/50 分流，保证评估基准不随训练阶段变化
            self.concrete_ratio = 0.5
        else:
            self.phase2_intensity = phase2_intensity
            self.concrete_ratio = concrete_cfg.get("concrete_ratio", 0.7)

        # ---- 音频参数 ----
        self.sample_rate = hp["data"]["sampling_rate"]
        self.segment_length = hp["data"].get("segment_length", self.sample_rate)

        # ---- 加载文件列表 ----
        dataset_key = "train_dataset" if split == "train" else "val_dataset"
        dataset_cfg = hp["data"].get(dataset_key, {})

        self.vocal_files = self._scan_audio_files(
            dataset_cfg.get("speech", {}).get("vocal", "")
        )
        self.noise_files = self._scan_audio_files(
            dataset_cfg.get("speech", {}).get("noise", "")
        )

        # ---- 增强效果列表 ----
        effects_cfg = hp.get("augment", {}).get("effects", {})
        self.effect_names = list(effects_cfg.keys()) if effects_cfg else None

        # ================================================================
        # [抗过拟合] 原始 VoiceFixer 增强所需的噪声文件列表
        # add_noise_and_scale_with_HQ_with_Aug 需要 noise_files 参数
        # ================================================================
        self._orig_aug_available = _HAS_ORIG_AUG and len(self.noise_files) > 0

        # ---- 验证 ----
        if len(self.vocal_files) == 0:
            raise RuntimeError(
                f"[{split}] 未找到语音文件，请检查配置路径: "
                f"{dataset_cfg.get('speech', {}).get('vocal', 'N/A')}"
            )

        print(
            f"[ConcreteAugDataset/{split}] "
            f"语音: {len(self.vocal_files)} 文件, "
            f"噪声: {len(self.noise_files)} 文件, "
            f"片段长度: {self.segment_length} 采样点, "
            f"Phase 2 强度: {self.phase2_intensity}, "
            f"混凝土占比: {self.concrete_ratio:.0%}, "
            f"原始增强占比: {1 - self.concrete_ratio:.0%}"
        )

    def __len__(self) -> int:
        return len(self.vocal_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        返回一个训练样本。
        
        [抗过拟合] 按 concrete_ratio 概率分流：
        - concrete_ratio (70%) → 混凝土双规制增强链路
        - 1 - concrete_ratio (30%) → 原始 VoiceFixer 增强逻辑
        """
        # ---- 1. 读取干净语音 (Target) ----
        clean = self._load_audio_segment(self.vocal_files[idx])

        # ---- 2. 分流决策 ----
        use_concrete = random.random() < self.concrete_ratio

        if use_concrete:
            # ============================================================
            # 路径 A：混凝土双规制增强（70%）
            # Phase 0(加噪) → Phase 1(环境声学) → Phase 2(混凝土物理链路)
            # ============================================================
            degraded, clean, phase1 = self._augment_concrete(clean)
            aug_type = "concrete"
        else:
            # ============================================================
            # 路径 B：原始 VoiceFixer 通用增强（30%）
            # 使用 add_noise_and_scale_with_HQ_with_Aug
            # ============================================================
            degraded, clean, phase1 = self._augment_voicefixer_original(clean)
            aug_type = "voicefixer"

        # ---- 3. 格式化并返回结果 ----
        if isinstance(degraded, torch.Tensor):
            degraded = degraded.numpy()
        if isinstance(clean, torch.Tensor):
            clean = clean.numpy()
        if isinstance(phase1, torch.Tensor):
            phase1 = phase1.numpy()

        # 确保都是 float32 numpy
        degraded = np.asarray(degraded, dtype=np.float32)
        clean = np.asarray(clean, dtype=np.float32)
        phase1 = np.asarray(phase1, dtype=np.float32)

        # 长度对齐
        len_d, len_c, len_p = len(degraded), len(clean), len(phase1)
        max_len = max(len_d, len_c, len_p)

        # 淡出防截断
        def apply_fadeout(audio_arr):
            arr_len = len(audio_arr)
            f_len = min(256, arr_len // 20)
            if f_len > 1:
                f_curve = np.linspace(1.0, 0.0, f_len, dtype=np.float32)
                audio_arr[-f_len:] = audio_arr[-f_len:] * f_curve
            return audio_arr

        degraded = apply_fadeout(degraded)
        clean = apply_fadeout(clean)
        phase1 = apply_fadeout(phase1)

        # Zero-padding 对齐
        if len_d < max_len:
            degraded = np.pad(degraded, (0, max_len - len_d), mode='constant')
        if len_c < max_len:
            clean = np.pad(clean, (0, max_len - len_c), mode='constant')
        if len_p < max_len:
            phase1 = np.pad(phase1, (0, max_len - len_p), mode='constant')

        result = {
            "input_wave": degraded,
            "target_wave": clean,
            "phase1_wave": phase1,
        }

        return result

    # ================================================================
    #  路径 A：混凝土双规制增强
    # ================================================================

    def _augment_concrete(self, clean: np.ndarray):
        """
        原始混凝土链路：Phase 0(加噪) → Phase 1 → Phase 2。
        返回 (degraded, clean, phase1)
        """
        input_frames = clean.copy()

        # Phase 0: 环境噪声混合
        if self.noise_files:
            if self.split == "train":
                if random.random() < 0.85:
                    input_frames, clean = self._mix_noise(
                        input_frames, clean,
                        snr_range=(-5.0, 35.0),
                        scale_range=(0.6, 1.0),
                    )
            else:
                # 验证集：固定参数
                input_frames, clean = self._mix_noise(
                    input_frames, clean,
                    snr_range=(15.0, 15.0),
                    scale_range=(0.8, 0.8),
                )

        # Phase 1 + Phase 2
        if self.audio_aug is not None:
            if self.split == "train":
                apply_phase2 = self.phase2_intensity > 0
                current_intensity = self.phase2_intensity
            else:
                apply_phase2 = True
                current_intensity = self.phase2_intensity

            degraded, metadata = self.audio_aug.augment(
                frames=input_frames,
                effects=self.effect_names,
                sample_rate=self.sample_rate,
                apply_phase2=apply_phase2,
                phase2_intensity=current_intensity,
            )
            phase1 = metadata.get("phase1_audio", input_frames.copy())
        else:
            degraded = input_frames.copy()
            phase1 = input_frames.copy()

        return degraded, clean, phase1

    # ================================================================
    #  路径 B：原始 VoiceFixer 通用增强
    # ================================================================

    # ...existing code...

    def _augment_voicefixer_original(self, clean: np.ndarray):
        """
        调用原始 VoiceFixer 的 add_noise_and_scale_with_HQ_with_Aug。

        真实签名：
            add_noise_and_scale_with_HQ_with_Aug(HQ, front, augfront, noise, ...)
            → (HQ, front, augfront, noise, snr, scale)

        参数含义：
            HQ       = 高质量目标（干净语音）
            front    = 未增强的输入（和 HQ 相同或略有差异）
            augfront = Phase 1 增强后的音频
            noise    = 噪声信号

        返回 (degraded, clean, phase1)
        """
        if self._orig_aug_available:
            try:
                # ---- Step 1: 先执行 Phase 1 环境声学增强（得到 augfront）----
                augfront_np = clean.copy()
                if self.audio_aug is not None and hasattr(self.audio_aug, 'magical_effects'):
                    try:
                        effects = self.effect_names or []
                        if effects:
                            augfront_np, _ = self.audio_aug.magical_effects.effect(
                                augfront_np,
                                effects,
                                self.sample_rate,
                                None,
                                True,
                            )
                    except Exception:
                        pass  # Phase 1 失败则 augfront = clean

                # ---- Step 2: 加载噪声 ----
                noise_np = self._load_random_noise(len(clean))

                # ---- Step 3: numpy → torch.Tensor（原始函数要求 Tensor）----
                HQ       = torch.from_numpy(clean.copy()).float()
                front    = torch.from_numpy(clean.copy()).float()
                augfront = torch.from_numpy(augfront_np).float()
                noise    = torch.from_numpy(noise_np).float()

                # ---- Step 4: 调用原始增强函数 ----
                HQ_out, front_out, augfront_out, noise_out, snr, scale = \
                    add_noise_and_scale_with_HQ_with_Aug(
                        HQ, front, augfront, noise,
                        snr_l=-5, snr_h=35,
                        scale_lower=0.6, scale_upper=1.0,
                    )

                # ---- Step 5: torch → numpy 返回 ----
                # degraded = augfront_out + noise_out（增强后的前端 + 噪声）
                degraded = (augfront_out + noise_out).numpy()
                clean_final = HQ_out.numpy()
                phase1 = augfront_out.numpy()  # Phase 1 结果（增强后，未加噪）

                return degraded, clean_final, phase1

            except Exception as e:
                import warnings
                warnings.warn(
                    f"[ConcreteAugDataset] 原始 VoiceFixer 增强失败: {e}，"
                    f"回退到仅 Phase 1 增强"
                )
                return self._augment_phase1_only(clean)
        else:
            return self._augment_phase1_only(clean)

    # ...existing code...
    def _augment_phase1_only(self, clean: np.ndarray):
        """
        回退方案：只执行 Phase 1（环境声学做旧），跳过 Phase 2。
        当原始 VoiceFixer 增强函数不可用时使用。
        """
        input_frames = clean.copy()

        # 加噪
        if self.noise_files:
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

        # 仅 Phase 1，不执行 Phase 2
        if self.audio_aug is not None:
            degraded, metadata = self.audio_aug.augment(
                frames=input_frames,
                effects=self.effect_names,
                sample_rate=self.sample_rate,
                apply_phase2=False,          # ← 关键：跳过混凝土链路
                phase2_intensity=0.0,
            )
            phase1 = metadata.get("phase1_audio", input_frames.copy())
        else:
            degraded = input_frames.copy()
            phase1 = input_frames.copy()

        return degraded, clean, phase1

    # ================================================================
    #  噪声混合工具
    # ================================================================

    def _mix_noise(
        self,
        input_frames: np.ndarray,
        clean: np.ndarray,
        snr_range: tuple = (-5.0, 35.0),
        scale_range: tuple = (0.6, 1.0),
    ):
        """
        噪声混合。抽取为独立方法，路径 A/B 共用。
        
        返回 (mixed_input, scaled_clean)
        """
        noise = self._load_random_noise(clean.shape[0])

        snr_db = random.uniform(*snr_range)
        scale = random.uniform(*scale_range)

        clean_rms = np.sqrt(np.mean(clean ** 2) + 1e-10)
        noise_rms = np.sqrt(np.mean(noise ** 2) + 1e-10)

        target_noise_rms = clean_rms / (10 ** (snr_db / 20.0))
        noise_scaled = noise * (target_noise_rms / noise_rms)

        mixed = (input_frames + noise_scaled) * scale
        clean_scaled = clean * scale

        # 防爆音
        peak = np.max(np.abs(mixed))
        if peak > 0.99:
            mixed = (mixed / peak) * 0.95

        return mixed, clean_scaled

    # ================================================================
    #  文件扫描与音频读取（不变）
    # ================================================================

    @staticmethod
    def _scan_audio_files(directory: str) -> list:
        # ...existing code...
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
        # ...existing code...
        try:
            info = sf.info(filepath)
            total_frames = info.frames
            file_sr = info.samplerate
        except Exception as e:
            warnings.warn(f"无法读取音频信息 {filepath}: {e}")
            return np.zeros(self.segment_length, dtype=np.float32)

        sr_ratio = file_sr / self.sample_rate
        needed_frames = int(self.segment_length * sr_ratio)

        if total_frames <= needed_frames:
            start = 0
            frames_to_read = total_frames
        elif self.split == "train":
            start = random.randint(0, total_frames - needed_frames)
            frames_to_read = needed_frames
        else:
            start = 0
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

        return audio.astype(np.float32)

    def _load_random_noise(self, length: int) -> np.ndarray:
        # ...existing code...
        noise_path = random.choice(self.noise_files)
        noise = self._load_audio_segment(noise_path)

        if len(noise) < length:
            repeats = (length // len(noise)) + 1
            noise = np.tile(noise, repeats)
        return noise[:length]

    @staticmethod
    def _simple_resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        # ...existing code...
        if orig_sr == target_sr:
            return audio
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    @staticmethod
    def _pad_or_trim(tensor: torch.Tensor, target_length: int) -> torch.Tensor:
        # ...existing code...
        current = tensor.shape[-1]
        if current >= target_length:
            return tensor[..., :target_length]
        else:
            pad_size = target_length - current
            return torch.nn.functional.pad(tensor, (0, pad_size), mode="constant", value=0.0)