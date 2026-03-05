# -*- coding: utf-8 -*-
# @Time    : 2026/2/25
# @Author  : Concrete Eavesdrop System 3022234317@tju.edu.cn
# @FileName: train_concrete.py
#
# 穿透混凝土墙体窃听语音还原 - 迁移学习训练脚本
# 基于 VoiceFixer 原始训练架构（PyTorch Lightning）重构
#
# 修复列表：
# [核心修复 1] 修复 TensorBoard 绘图导致的严重内存泄漏 (Memory Leak)
# [核心修复 2] 修复 AMP 混合精度下的假梯度范数 (False Grad Norm)
# [核心修复 5] 兼容 VoiceFixer 父类的 training_step 返回值类型
# [Linux修复 1] 移除 Windows 专用 monkey-patch，保留多平台兼容判断
# [Linux修复 2] 注册 SIGTERM/SIGINT 优雅退出，防止 DDP 僵尸进程
# [Linux修复 3] 多 GPU NCCL 后端显式配置
# [Linux修复 4] 修复 training_step / validation_step 中的 device mismatch
# [代码修复 1] 解除对 warnings.warn 的全局替换（保留真实错误警告）
# [代码修复 2] START_STAGE 可通过命令行 --start_stage 配置
# [代码修复 3] 修复 SNR 公式分母混淆 bug
# [代码修复 4] 向量化 Mel FilterBank 计算

import os
# 只屏蔽 Python 级别的用户警告，不屏蔽 C++/CUDA 级别
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore', message='.*pynvml.*')
warnings.filterwarnings('ignore', message='.*is_pos_inf.*')
warnings.filterwarnings('ignore', message='.*is_neg_inf.*')
warnings.filterwarnings('ignore', message='.*isfinite.*')
warnings.filterwarnings('ignore', message='.*has_inf_or_nan.*')
warnings.filterwarnings('ignore', message='.*SoX.*')
# [代码修复 1] 不再全局替换 warnings.warn，保留 PyTorch 真实错误警告

import sys
import json
import argparse
import math
import signal        # [Linux修复 2]
import platform
import multiprocessing
from typing import Optional, Dict, Any, List, Tuple
from collections import OrderedDict

from tools.pytorch.pytorch_util import to_log

import git
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
    Callback,
)
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from dataloaders.collators.concrete_collator import ConcreteEavesdropCollatorGPU

# ============================================================================
#  PL 版本兼容层
# ============================================================================

def _parse_pl_version() -> Tuple[int, int]:
    """安全解析 PL 版本号，处理 rc/dev/post 后缀"""
    version_str = pl.__version__
    parts = []
    for part in version_str.split(".")[:2]:
        digits = ""
        for ch in part:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits) if digits else 0)
    while len(parts) < 2:
        parts.append(0)
    return tuple(parts)


_pl_version = _parse_pl_version()
_USE_PL2 = _pl_version >= (2, 0)

if _USE_PL2:
    DDPStrategy = "ddp_find_unused_parameters_true"
else:
    try:
        from pytorch_lightning.plugins import DDPPlugin as DDPStrategy
    except ImportError:
        DDPStrategy = None

try:
    from pytorch_lightning.callbacks.progress import TQDMProgressBar
except ImportError:
    try:
        from pytorch_lightning.callbacks import TQDMProgressBar
    except ImportError:
        TQDMProgressBar = None


# ============================================================================
#  项目路径初始化
# ============================================================================

git_root = git.Repo("", search_parent_directories=True).git.rev_parse("--show-toplevel")
if git_root not in sys.path:
    sys.path.insert(0, git_root)

third_party_dir = os.path.join(git_root, "third_party")
if third_party_dir not in sys.path:
    sys.path.insert(0, third_party_dir)

import tools.utils
from tools.callbacks.base import initLogDir
from dataloaders.concrete_data_module import ConcreteDataModule
from models.gsr_voicefixer import VoiceFixer


# ============================================================================
#  [Linux修复 2] 优雅退出处理：防止 DDP/NCCL 僵尸进程
# ============================================================================

_trainer_ref: Optional["Trainer"] = None  # 用于信号处理器中访问 Trainer


def _graceful_exit_handler(signum, frame):
    """
    捕获 SIGTERM（SLURM/kill）和 SIGINT（Ctrl+C），
    在退出前保存最后一个 checkpoint 并清理 DDP 进程组。
    """
    sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
    print(f"\n[SIGNAL] 收到 {sig_name}，正在执行优雅退出...")

    global _trainer_ref
    if _trainer_ref is not None:
        try:
            ckpt_path = os.path.join(
                _trainer_ref.logger.log_dir,
                "checkpoints",
                "emergency_exit.ckpt",
            )
            _trainer_ref.save_checkpoint(ckpt_path)
            print(f"[SIGNAL] 紧急 checkpoint 已保存: {ckpt_path}")
        except Exception as e:
            print(f"[SIGNAL] 保存 checkpoint 失败: {e}")

        try:
            # 清理 DDP 进程组，防止 NCCL 挂起
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
                print("[SIGNAL] DDP 进程组已清理")
        except Exception:
            pass

    sys.exit(0)


def _register_signal_handlers():
    """仅在主进程（rank 0）注册信号处理器"""
    if os.environ.get("LOCAL_RANK", "0") == "0":
        signal.signal(signal.SIGTERM, _graceful_exit_handler)
        signal.signal(signal.SIGINT, _graceful_exit_handler)
        print("[SIGNAL] 已注册 SIGTERM/SIGINT 优雅退出处理器")


# ============================================================================
#  Task 4: Audio/Mel TensorBoard 可视化 Callback
# ============================================================================


class AudioVisualLoggingCallback(Callback):
    """
    训练过程中的音频与频谱可视化（升级版）。
    包含：干净原声 -> Phase1环境音 -> Phase2穿墙音 -> AI修复音 的全链路展示。
    """
    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        max_samples: int = 3,
        log_every_n_epochs: int = 1,
    ):
        super().__init__()
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.max_samples = max_samples
        self.log_every_n_epochs = log_every_n_epochs
        self._val_cache = None
        self._cache_captured = False
        # [代码修复 4] 预计算 Mel FilterBank（向量化，只算一次）
        self._mel_fb: Optional[torch.Tensor] = None

    def _get_mel_fb(self) -> torch.Tensor:
        if self._mel_fb is None:
            self._mel_fb = self._mel_filterbank_vectorized(
                self.sr, self.n_fft, self.n_mels
            )
        return self._mel_fb

    def on_validation_epoch_start(self, trainer, pl_module):
        self._val_cache = None
        self._cache_captured = False

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if self._cache_captured or batch_idx > 0:
            return

        try:
            if not isinstance(batch, dict):
                return

            dirty = batch.get("input_wave")
            clean = batch.get("target_wave")
            phase1 = batch.get("phase1_wave", dirty)

            if dirty is None or clean is None:
                return

            # 扩展维度 (B, T) -> (B, 1, T)
            if dirty.dim() == 2:
                dirty = dirty.unsqueeze(1)
            if clean.dim() == 2:
                clean = clean.unsqueeze(1)
            if phase1.dim() == 2:
                phase1 = phase1.unsqueeze(1)

            # [Linux修复 4] 显式将所有张量移到模型所在设备
            device = pl_module.device
            dirty = dirty.to(device)
            clean = clean.to(device)
            phase1 = phase1.to(device)

            cutoff_hz = batch.get("cutoff_hz")
            if cutoff_hz is not None:
                cutoff_hz = cutoff_hz.to(device)
                dirty = ConcreteEavesdropCollatorGPU.gpu_fft_lowpass(
                    waveform=dirty,
                    cutoff_hz=cutoff_hz,
                    sample_rate=self.sr,
                )

            with torch.no_grad():
                _, mel_low_quality = pl_module.pre(dirty)
                pred_mel_dict = pl_module(mel_low_quality)
                pred_log_mel = pred_mel_dict['mel']

                # 钳位防止数值爆炸
                pred_log_mel_safe = pred_log_mel.clamp(-10.0, 5.0)
                pred_linear_mel = 10 ** pred_log_mel_safe

                # 使 vocoder 保持 eval 状态进行推理
                vocoder_module, _ = pl_module._find_vocoder_module()
                if vocoder_module is not None:
                    with torch.cuda.amp.autocast(enabled=False):
                        prediction = pl_module.vocoder(
                            pred_linear_mel.float()
                        )
                else:
                    prediction = pl_module.vocoder(pred_linear_mel)

                # 强制对齐所有波形的时间维度
                min_len = min(
                    dirty.shape[-1],
                    prediction.shape[-1],
                    clean.shape[-1],
                )
                dirty = dirty[..., :min_len]
                clean = clean[..., :min_len]
                phase1 = phase1[..., :min_len]
                prediction = prediction[..., :min_len]

            self._val_cache = {
                "dirty": dirty.detach().cpu(),
                "clean": clean.detach().cpu(),
                "phase1": phase1.detach().cpu(),
                "prediction": prediction.detach().cpu(),
            }
            self._cache_captured = True

        except Exception as e:
            print(f"\n[AudioVisual] 验证集绘图数据捕获失败: {e}\n")

    def on_validation_epoch_end(self, trainer, pl_module):
        if (
            self._val_cache is None
            or trainer.current_epoch % self.log_every_n_epochs != 0
        ):
            return

        # 仅在 rank 0 执行 TensorBoard 写入，避免多卡重复写入
        if trainer.global_rank != 0:
            self._val_cache = None
            return

        logger_exp = (
            trainer.logger.experiment
            if hasattr(trainer, 'logger') and trainer.logger is not None
            else None
        )
        if logger_exp is None:
            return

        dirty = self._val_cache["dirty"]
        clean = self._val_cache["clean"]
        phase1 = self._val_cache["phase1"]
        pred = self._val_cache["prediction"]

        n_samples = min(self.max_samples, dirty.shape[0])
        import matplotlib.pyplot as plt

        for i in range(n_samples):
            tag_prefix = f"val_sample_{i}"
            step = trainer.current_epoch

            d = self._normalize_audio(self._to_1d(dirty[i]))
            c = self._normalize_audio(self._to_1d(clean[i]))
            p1 = self._normalize_audio(self._to_1d(phase1[i]))
            p = self._normalize_audio(self._to_1d(pred[i]))

            if None in (d, c, p1, p):
                continue

            # 1. 记录音频试听
            try:
                logger_exp.add_audio(
                    f"{tag_prefix}/1_Clean", c.unsqueeze(0), step, sample_rate=self.sr
                )
                logger_exp.add_audio(
                    f"{tag_prefix}/2_Phase1_Env", p1.unsqueeze(0), step, sample_rate=self.sr
                )
                logger_exp.add_audio(
                    f"{tag_prefix}/3_Phase2_Wall", d.unsqueeze(0), step, sample_rate=self.sr
                )
                logger_exp.add_audio(
                    f"{tag_prefix}/4_AI_Restored", p.unsqueeze(0), step, sample_rate=self.sr
                )
            except Exception:
                pass

            # 2. 绘制 4 行 Mel 频谱图全链路
            try:
                fig = self._plot_mel_comparison(c, p1, d, p, step, i)
                if fig is not None:
                    logger_exp.add_figure(f"{tag_prefix}/mel_pipeline", fig, step)
                    fig.clear()
                    plt.close(fig)
            except Exception:
                pass

        # 兜底：清空所有残留画布，防止内存泄漏
        plt.close('all')
        self._val_cache = None

        import gc
        gc.collect()

    def _plot_mel_comparison(self, clean, phase1, dirty, pred, epoch, sample_idx):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        fig, axes = plt.subplots(4, 1, figsize=(12, 13), constrained_layout=True)
        titles = [
            "1. Clean Target",
            "2. Phase 1 (Reverb & Env Noise)",
            "3. Phase 2 (Wall Decay & Circuit EMI)",
            "4. AI Restored",
        ]
        signals = [clean, phase1, dirty, pred]

        for ax, title, signal in zip(axes, titles, signals):
            mel = self._compute_mel_spectrogram(signal)
            if mel is not None:
                im = ax.imshow(
                    mel.numpy(),
                    aspect='auto',
                    origin='lower',
                    interpolation='nearest',
                    cmap='magma',
                )
                fig.colorbar(im, ax=ax, format='%+2.0f dB', pad=0.01)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_ylabel('Mel Bin')

        axes[-1].set_xlabel('Time Frame')

        snr_p1 = self._estimate_snr(phase1, clean)
        snr_p2 = self._estimate_snr(dirty, clean)
        snr_ai = self._estimate_snr(pred, clean)

        fig.suptitle(
            f"Sample {sample_idx} - Epoch {epoch}\n"
            f"Env SNR: {snr_p1:.1f}dB  |  Wall SNR: {snr_p2:.1f}dB"
            f"  |  AI Restored SNR: {snr_ai:.1f}dB",
            fontsize=13,
            fontweight='bold',
        )
        return fig

    def _compute_mel_spectrogram(
        self, audio: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if audio.dim() == 0 or len(audio) < self.n_fft:
            return None
        try:
            window = torch.hann_window(self.n_fft)
            stft = torch.stft(
                audio, self.n_fft, self.hop_length,
                window=window, return_complex=True,
            )
            power = stft.abs().pow(2)
            # [代码修复 4] 使用预计算的向量化 FilterBank
            mel_fb = self._get_mel_fb()
            mel_spec = torch.mm(mel_fb, power)
            mel_db = 10.0 * torch.log10(mel_spec.clamp(min=1e-10))
            return torch.clamp(mel_db, min=mel_db.max() - 80)
        except Exception:
            return None

    @staticmethod
    def _mel_filterbank_vectorized(
        sr: int, n_fft: int, n_mels: int
    ) -> torch.Tensor:
        """
        [代码修复 4] 全向量化 Mel FilterBank，替换原来的 Python 循环实现。
        速度提升约 10-30x（n_mels=128 时）。
        """
        fmax = sr / 2.0
        fmin = 0.0
        mel_min = 2595.0 * math.log10(1.0 + fmin / 700.0)
        mel_max = 2595.0 * math.log10(1.0 + fmax / 700.0)

        # shape: (n_mels + 2,)
        mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

        freq_bins = n_fft // 2 + 1
        # shape: (freq_bins,)
        fft_freqs = torch.linspace(0, fmax, freq_bins)

        # shape: (freq_bins, 1) broadcast with (1, n_mels+2)
        # up_slope[f, i]   = (fft_freqs[f] - hz_points[i])   / (hz_points[i+1] - hz_points[i])
        # down_slope[f, i] = (hz_points[i+2] - fft_freqs[f]) / (hz_points[i+2] - hz_points[i+1])
        f_left   = hz_points[:-2]   # (n_mels,)
        f_center = hz_points[1:-1]  # (n_mels,)
        f_right  = hz_points[2:]    # (n_mels,)

        # 广播：(freq_bins, n_mels)
        up_slope   = (fft_freqs[:, None] - f_left[None, :]) / (
            (f_center - f_left)[None, :].clamp(min=1e-10)
        )
        down_slope = (f_right[None, :] - fft_freqs[:, None]) / (
            (f_right - f_center)[None, :].clamp(min=1e-10)
        )

        # shape: (freq_bins, n_mels) -> transpose -> (n_mels, freq_bins)
        fb = torch.clamp(torch.min(up_slope, down_slope), min=0.0).T
        return fb

    @staticmethod
    def _to_1d(tensor: torch.Tensor) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        t = tensor.detach().cpu().float()
        while t.dim() > 1:
            t = t.squeeze(0) if t.size(0) == 1 else t[0]
        return t

    @staticmethod
    def _normalize_audio(audio: torch.Tensor) -> torch.Tensor:
        if audio is None:
            return audio
        peak = audio.abs().max()
        return audio / peak * 0.95 if peak > 1e-8 else audio

    @staticmethod
    def _estimate_snr(signal: torch.Tensor, reference: torch.Tensor) -> float:
        """
        [代码修复 3] 修复 SNR 公式：SNR = 10 * log10(signal_power / noise_power)
        原代码将分子分母写反（ref_power 与 noise_power 位置混淆）。
        正确定义：signal = reference (干净音)，noise = reference - signal (误差)
        """
        min_len = min(len(signal), len(reference))
        sig = signal[:min_len].float()
        ref = reference[:min_len].float()
        noise = ref - sig
        signal_power = (ref ** 2).mean()
        noise_power  = (noise ** 2).mean()
        if noise_power < 1e-10:
            return 100.0
        if signal_power < 1e-10:
            return -100.0
        return 10.0 * math.log10(signal_power / noise_power)


# ============================================================================
#  核心：混凝土穿透语音恢复 LightningModule
# ============================================================================

class ConcreteVoiceFixer(VoiceFixer):
    """
    继承原始 VoiceFixer LightningModule，覆写迁移学习相关逻辑。
    """

    def __init__(self, hp: dict, channels: int = 1, type_target: str = "vocals"):
        if "task" not in hp:
            hp["task"] = {}
        hp["task"].setdefault("inspect_training_data", False)
        hp["task"].setdefault("gsr", {})
        hp["task"]["gsr"].setdefault("gsr_model", {})
        hp["task"]["gsr"]["gsr_model"].setdefault("voicefixer", {})
        vf = hp["task"]["gsr"]["gsr_model"]["voicefixer"]
        vf.setdefault("unet", True)
        vf.setdefault("unet_small", False)
        vf.setdefault("bi_lstm", False)
        vf.setdefault("dnn", False)

        if "train" not in hp:
            hp["train"] = {}
        hp["train"].setdefault("learning_rate", 1e-4)
        hp["train"].setdefault("lr_decay", 0.999)
        hp["train"].setdefault("batch_size", 2)
        hp["train"].setdefault("input_segment_length", 44100)
        hp["train"].setdefault("warmup_steps", 1000)
        hp["train"].setdefault("reduce_lr_every_n_steps", 5000)
        hp["train"].setdefault("check_val_every_n_epoch", 1)
        hp["train"].setdefault("max_epoches", 200)
        hp["train"].setdefault("num_workers", 4)

        if "data" not in hp:
            hp["data"] = {}
        hp["data"].setdefault("sampling_rate", 44100)
        hp["data"].setdefault("segment_length", 44100)

        if "model" not in hp:
            hp["model"] = {}
        hp["model"].setdefault("channels_in", 1)
        hp["model"].setdefault("channels", channels)
        hp["model"].setdefault("type_target", type_target)
        hp["model"].setdefault("nsrc", 1)
        hp["model"].setdefault("loss_type", "l1")
        hp["model"].setdefault("mel_freq_bins", 128)
        hp["model"].setdefault("window_size", 2048)
        hp["model"].setdefault("hop_size", 512)
        hp["model"].setdefault("pad_mode", "reflect")
        hp["model"].setdefault("window", "hann")

        if "mel" not in hp:
            hp["mel"] = {}
        hp["mel"].setdefault("n_fft", 2048)
        hp["mel"].setdefault("hop_length", 512)
        hp["mel"].setdefault("win_length", 2048)
        hp["mel"].setdefault("n_mels", 128)
        hp["mel"].setdefault("fmin", 0)
        hp["mel"].setdefault("fmax", 22050)

        hp.setdefault("model_dir", "exp/concrete_v1")

        super().__init__(hp, channels=channels, type_target=type_target)

        self.hp = hp
        self.concrete_cfg = hp.get("concrete", {})

        self._load_pretrained_weights()
        self._apply_freeze_strategy()
        self._print_model_summary()

    # ----------------------------------------------------------------
    #  预训练权重加载
    # ----------------------------------------------------------------

    def _load_pretrained_weights(self):
        weights_cfg = self.concrete_cfg.get("pretrained_weights", {})

        if not weights_cfg:
            legacy_path = self.concrete_cfg.get("pretrained_checkpoint", "")
            if legacy_path and os.path.isfile(legacy_path):
                print(f"[PRETRAIN] 加载单一 checkpoint: {legacy_path}")
                ckpt = torch.load(legacy_path, map_location="cpu", weights_only=False)
                raw_sd = self._extract_state_dict(ckpt)
                cleaned_sd = self._clean_state_dict(raw_sd)
                info = self.load_state_dict(cleaned_sd, strict=False)
                print(
                    f"  ✓ missing: {len(info.missing_keys)}, "
                    f"unexpected: {len(info.unexpected_keys)}"
                )
            elif legacy_path:
                warnings.warn(f"[PRETRAIN] checkpoint 不存在: {legacy_path}")
            else:
                print("[PRETRAIN] 未指定预训练路径，从随机初始化开始")
            return

        analysis_cfg = weights_cfg.get("analysis_module", {})
        analysis_path = analysis_cfg.get("path", "")
        if analysis_path and os.path.isfile(analysis_path):
            self._load_analysis_module(analysis_path)
        elif analysis_path:
            warnings.warn(f"[PRETRAIN] Analysis Module 权重不存在: {analysis_path}")

        synthesis_cfg = weights_cfg.get("synthesis_module", {})
        synthesis_path = synthesis_cfg.get("path", "")
        if synthesis_path and os.path.isfile(synthesis_path):
            self._load_synthesis_module(synthesis_path)
        elif synthesis_path:
            warnings.warn(f"[PRETRAIN] Synthesis Module 权重不存在: {synthesis_path}")

    def _load_analysis_module(self, ckpt_path: str):
        print(f"[PRETRAIN] 加载 Analysis Module: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        raw_sd = self._extract_state_dict(ckpt)
        cleaned_sd = self._clean_state_dict(raw_sd)

        model_sd = self.state_dict()
        loadable = OrderedDict()
        skipped = []

        for k, v in cleaned_sd.items():
            if k in model_sd and v.shape == model_sd[k].shape:
                loadable[k] = v
                continue
            matched = False
            for prefix in ("model.", "analysis_module.", "resunet."):
                candidate = prefix + k
                if candidate in model_sd and v.shape == model_sd[candidate].shape:
                    loadable[candidate] = v
                    matched = True
                    break
            if not matched:
                skipped.append(k)

        self.load_state_dict(loadable, strict=False)
        print(f"  ✓ 成功加载: {len(loadable)}/{len(model_sd)} 个参数")
        if skipped:
            print(f"  ⚠ 跳过 {len(skipped)} 个不匹配的 key")
            for s in skipped[:5]:
                print(f"    - {s}")
            if len(skipped) > 5:
                print(f"    ... 及其余 {len(skipped) - 5} 个")

    def _load_synthesis_module(self, ckpt_path: str):
        print(f"[PRETRAIN] 加载 Synthesis Module: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        raw_sd = self._extract_state_dict(ckpt)
        cleaned_sd = self._clean_state_dict(raw_sd)

        vocoder_module, vocoder_name = self._find_vocoder_module()

        if vocoder_module is not None:
            load_success = False
            for strip_prefixes in ([], ["generator."], ["model."], ["vocoder."]):
                try:
                    sd_to_load = cleaned_sd
                    if strip_prefixes:
                        sd_to_load = OrderedDict()
                        for k, v in cleaned_sd.items():
                            new_k = k
                            for pfx in strip_prefixes:
                                if new_k.startswith(pfx):
                                    new_k = new_k[len(pfx):]
                                    break
                            sd_to_load[new_k] = v
                    vocoder_module.load_state_dict(sd_to_load, strict=False)
                    load_success = True
                    n_params = sum(p.numel() for p in vocoder_module.parameters())
                    pfx_desc = f"(去前缀 {strip_prefixes})" if strip_prefixes else ""
                    print(
                        f"  ✓ Vocoder ({vocoder_name}) 加载成功{pfx_desc}: "
                        f"{n_params:,} 个参数"
                    )
                    break
                except Exception:
                    continue
            if not load_success:
                warnings.warn(f"  ✗ Vocoder ({vocoder_name}) 所有加载策略均失败")
        else:
            warnings.warn("[PRETRAIN] 未找到 Vocoder 子模块")

    def _find_vocoder_module(self) -> Tuple[Optional[nn.Module], Optional[str]]:
        vocoder_module = None
        vocoder_name = None

        for name, module in self.named_modules():
            if name == "":
                continue
            if "vocoder" in name.lower() or "voc" in name.lower():
                if vocoder_module is None or len(name) < len(vocoder_name):
                    vocoder_module = module
                    vocoder_name = name

        if vocoder_module is None:
            for attr_name in ("vocoder", "voc", "synthesis_module"):
                if hasattr(self, attr_name):
                    candidate = getattr(self, attr_name)
                    if isinstance(candidate, nn.Module) and \
                       sum(1 for _ in candidate.parameters()) > 0:
                        vocoder_module = candidate
                        vocoder_name = attr_name
                        break

        return vocoder_module, vocoder_name

    @staticmethod
    def _extract_state_dict(ckpt) -> dict:
        if isinstance(ckpt, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in ckpt:
                    return ckpt[key]
        return ckpt if isinstance(ckpt, dict) else {}

    @staticmethod
    def _clean_state_dict(state_dict: dict) -> OrderedDict:
        cleaned = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            cleaned[name] = v
        return cleaned

    def _apply_freeze_strategy(self):
        freeze_cfg = self.concrete_cfg.get("freeze_strategy", {})

        if freeze_cfg.get("freeze_vocoder", True):
            vocoder_frozen = 0
            vocoder_module, vocoder_name = self._find_vocoder_module()
            if vocoder_module is not None:
                for param in vocoder_module.parameters():
                    param.requires_grad = False
                    vocoder_frozen += param.numel()
                vocoder_module.eval()
                print(
                    f"[FREEZE] Vocoder ({vocoder_name}): "
                    f"冻结 {vocoder_frozen:,} 个参数, 切换为 eval() 模式"
                )
            else:
                print("[FREEZE] 未找到 Vocoder 模块，跳过冻结")

        n_freeze = freeze_cfg.get("freeze_encoder_layers", 0)
        if n_freeze > 0:
            frozen_count = self._freeze_encoder_layers(n_freeze)
            print(
                f"[FREEZE] ResUNet 编码器: 冻结前 {n_freeze} 层 "
                f"({frozen_count:,} 个参数)"
            )

    def _get_encoder_modules(self) -> List[Tuple[str, nn.Module]]:
        encoder_modules = []
        for name, module in self.named_modules():
            if name == "":
                continue
            name_lower = name.lower()
            if ("encoder" in name_lower or "down" in name_lower) and \
               not any(skip in name_lower for skip in ("decoder", "up")):
                if not any(
                    name.startswith(existing + ".") for existing, _ in encoder_modules
                ):
                    encoder_modules.append((name, module))
        return sorted(encoder_modules, key=lambda x: x[0])

    def _freeze_encoder_layers(self, n_layers: int) -> int:
        encoder_modules = self._get_encoder_modules()
        frozen_params = 0
        for i, (name, module) in enumerate(encoder_modules):
            if i >= n_layers:
                break
            for param in module.parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    frozen_params += param.numel()
        return frozen_params

    def _print_model_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"\n[MODEL] 参数统计:")
        print(f"  总参数:     {total:>12,}")
        print(f"  可训练:     {trainable:>12,}  ({100 * trainable / max(total, 1):.1f}%)")
        print(f"  已冻结:     {frozen:>12,}  ({100 * frozen / max(total, 1):.1f}%)")

        trainable_mb = trainable * 4 / (1024 ** 2)
        optimizer_mb = trainable_mb * 3
        total_model_mb = total * 2 / (1024 ** 2)
        print(f"\n[MEMORY] 显存估算 (FP16 混合精度):")
        print(f"  模型参数:   {total_model_mb:>8.0f} MB")
        print(f"  优化器状态: {optimizer_mb:>8.0f} MB")
        print(f"  合计 (不含激活): {total_model_mb + optimizer_mb:>8.0f} MB")

    def on_train_epoch_start(self):
        """确保每个 epoch 开始时 Vocoder 保持 eval 模式"""
        freeze_cfg = self.concrete_cfg.get("freeze_strategy", {})
        if freeze_cfg.get("freeze_vocoder", True):
            vocoder_module, _ = self._find_vocoder_module()
            if vocoder_module is not None:
                vocoder_module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            freeze_cfg = self.concrete_cfg.get("freeze_strategy", {})
            if freeze_cfg.get("freeze_vocoder", True):
                vocoder_module, _ = self._find_vocoder_module()
                if vocoder_module is not None:
                    vocoder_module.eval()
        return self

    def configure_optimizers(self):
        opt_cfg = self.concrete_cfg.get("optimizer", {})
        base_lr = opt_cfg.get("base_lr", 1e-4)
        weight_decay = opt_cfg.get("weight_decay", 1e-5)

        lr_multipliers = opt_cfg.get("lr_multipliers", {
            "encoder": 0.1,
            "bottleneck": 1.0,
            "decoder": 0.5,
            "other": 1.0,
        })

        groups: Dict[str, List[nn.Parameter]] = {
            "encoder": [],
            "bottleneck": [],
            "decoder": [],
            "other": [],
        }

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            name_lower = name.lower()
            if "encoder" in name_lower or "down" in name_lower:
                groups["encoder"].append(param)
            elif any(kw in name_lower for kw in ("bottleneck", "bridge", "middle")):
                groups["bottleneck"].append(param)
            elif "decoder" in name_lower or "up" in name_lower:
                groups["decoder"].append(param)
            else:
                groups["other"].append(param)

        param_groups = []
        for group_name, params in groups.items():
            if not params:
                continue
            lr = base_lr * lr_multipliers.get(group_name, 1.0)
            n_params = sum(p.numel() for p in params)
            param_groups.append({
                "params": params,
                "lr": lr,
                "name": group_name,
            })
            print(f"[OPT] {group_name:>12s}: {n_params:>10,} params, lr={lr:.2e}")

        if not param_groups:
            warnings.warn("[OPT] 无可训练参数！")
            param_groups = [
                {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": base_lr}
            ]

        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

        sched_cfg = self.concrete_cfg.get("scheduler", {})
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=sched_cfg.get("T_0", 10),
            T_mult=sched_cfg.get("T_mult", 2),
            eta_min=sched_cfg.get("eta_min", 1e-7),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # ----------------------------------------------------------------
    #  训练步骤 + 梯度监控
    # ----------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        dirty_audio = batch["input_wave"]
        clean_audio = batch["target_wave"]
        cutoff_hz = batch["cutoff_hz"]

        # [Linux修复 4] 显式确保张量在正确设备上（DDP 下必须）
        device = self.device
        dirty_audio = dirty_audio.to(device)
        clean_audio = clean_audio.to(device)
        cutoff_hz = cutoff_hz.to(device)

        # GPU 频域低通滤波
        dirty_audio = ConcreteEavesdropCollatorGPU.gpu_fft_lowpass(
            waveform=dirty_audio,
            cutoff_hz=cutoff_hz,
            sample_rate=self.hp["data"]["sampling_rate"],
        )

        if dirty_audio.dim() == 2:
            dirty_audio = dirty_audio.unsqueeze(1)
        if clean_audio.dim() == 2:
            clean_audio = clean_audio.unsqueeze(1)

        _, mel_target = self.pre(clean_audio)
        _, mel_low_quality = self.pre(dirty_audio)

        generated = self(mel_low_quality)
        target_log_mel = to_log(mel_target)
        gen_mel = generated['mel']

        # 强制对齐时间维度
        min_frames = min(gen_mel.size(2), target_log_mel.size(2))
        gen_mel = gen_mel[:, :, :min_frames, :]
        target_log_mel = target_log_mel[:, :, :min_frames, :]

        # 混合损失：L1 + SiMelSpec
        loss_l1 = self.l1loss(gen_mel, target_log_mel)
        loss_simel = self.simelspecloss(gen_mel, target_log_mel)
        loss = loss_l1 + 0.5 * loss_simel

        self.log(
            "train/loss_l1", loss_l1,
            on_step=True, on_epoch=True, prog_bar=False, sync_dist=True,
        )
        self.log(
            "train/loss_simel", loss_simel,
            on_step=True, on_epoch=True, prog_bar=False, sync_dist=True,
        )
        self.log(
            "train_loss", loss,
            on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
        )

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        dirty_audio = batch["input_wave"]
        clean_audio = batch["target_wave"]
        cutoff_hz = batch["cutoff_hz"]

        # [Linux修复 4] 同步设备
        device = self.device
        dirty_audio = dirty_audio.to(device)
        clean_audio = clean_audio.to(device)
        cutoff_hz = cutoff_hz.to(device)

        dirty_audio = ConcreteEavesdropCollatorGPU.gpu_fft_lowpass(
            waveform=dirty_audio,
            cutoff_hz=cutoff_hz,
            sample_rate=self.hp["data"]["sampling_rate"],
        )

        if dirty_audio.dim() == 2:
            dirty_audio = dirty_audio.unsqueeze(1)
        if clean_audio.dim() == 2:
            clean_audio = clean_audio.unsqueeze(1)

        _, mel_target = self.pre(clean_audio)
        _, mel_low_quality = self.pre(dirty_audio)
        estimation = self(mel_low_quality)['mel']
        target_log_mel = to_log(mel_target)

        min_frames = min(estimation.size(2), target_log_mel.size(2))
        estimation = estimation[:, :, :min_frames, :]
        target_log_mel = target_log_mel[:, :, :min_frames, :]

        val_loss_l1 = self.l1loss(estimation, target_log_mel)
        val_loss_simel = self.simelspecloss(estimation, target_log_mel)
        val_loss = val_loss_l1 + 0.5 * val_loss_simel

        self.log(
            "val/loss_l1", val_loss_l1,
            on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,
        )
        self.log(
            "val/loss_simel", val_loss_simel,
            on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,
        )
        self.log(
            "val_loss", val_loss,
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
        )

        return {"val_loss": val_loss}

    def on_before_optimizer_step(self, optimizer, optimizer_idx=0):
        """计算梯度范数（每 100 步采样，降低开销）"""
        if self.trainer.global_step % 100 == 0:
            grad_norm = self._compute_grad_norm()
            self.log("train/grad_norm", grad_norm, prog_bar=False)

            for group_name in ("encoder", "decoder", "bottleneck"):
                group_norm = self._compute_group_grad_norm(group_name)
                if group_norm > 0:
                    self.log(
                        f"train/grad_norm_{group_name}", group_norm, prog_bar=False
                    )

    def _compute_grad_norm(self) -> float:
        total_norm = 0.0
        for p in self.parameters():
            if p.requires_grad and p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def _compute_group_grad_norm(self, group_keyword: str) -> float:
        total_norm = 0.0
        for name, p in self.named_parameters():
            if (
                group_keyword in name.lower()
                and p.requires_grad
                and p.grad is not None
            ):
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5


# ============================================================================
#  Curriculum Learning 管理器
# ============================================================================

class CurriculumManager:
    """
    管理由易到难的训练阶段切换。
    适配 config/train_concrete.json 中的 curriculum_stages 结构。
    """

    def __init__(self, hp: dict, model: ConcreteVoiceFixer, dm: ConcreteDataModule):
        self.hp = hp
        self.model = model
        self.dm = dm
        self.stages = hp.get("concrete", {}).get("curriculum_stages", [])

        if not self.stages:
            print("[CURRICULUM] 未配置 curriculum_stages，使用默认单阶段训练")
            self.stages = [{
                "name": "default",
                "epochs": hp["train"]["max_epoches"],
            }]

    def apply_stage(self, stage_idx: int):
        if stage_idx >= len(self.stages):
            return

        stage = self.stages[stage_idx]
        print(f"\n{'═' * 60}")
        print(
            f"  CURRICULUM Stage {stage_idx + 1}/{len(self.stages)}: "
            f"{stage.get('name', 'unnamed')}"
        )
        print(f"{'═' * 60}")

        if "physics_config" in stage:
            if hasattr(self.dm, 'update_physics_config'):
                self.dm.update_physics_config(stage["physics_config"])
            print(f"  [DATA] 物理链路配置已更新")

        if "collator_config" in stage:
            if hasattr(self.dm, 'update_collator_config'):
                self.dm.update_collator_config(stage["collator_config"])
            print(f"  [DATA] Collator 配置已更新: {stage['collator_config']}")

        if "phase2_intensity" in stage:
            self.dm.phase2_intensity = stage["phase2_intensity"]
            print(f"  [DATA] Phase 2 降级强度已更新为: {stage['phase2_intensity']}")

        if "unfreeze_encoder_layers" in stage:
            n = stage["unfreeze_encoder_layers"]
            unfrozen = self._unfreeze_layers(n)
            print(f"  [MODEL] 解冻编码器前 {n} 层 ({unfrozen:,} 个参数)")

        trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"  [MODEL] 当前可训练参数: {trainable:,}\n")

    def _unfreeze_layers(self, n_layers: int) -> int:
        encoder_modules = self.model._get_encoder_modules()
        unfrozen = 0
        for i, (name, module) in enumerate(encoder_modules):
            if i >= n_layers:
                break
            for param in module.parameters():
                if not param.requires_grad:
                    param.requires_grad = True
                    unfrozen += param.numel()
        return unfrozen

    def get_stage_epochs(self, stage_idx: int) -> int:
        if stage_idx < len(self.stages):
            return self.stages[stage_idx].get("epochs", 50)
        return 0


# ============================================================================
#  验证工具
# ============================================================================

def validate_config(hp: dict):
    """训练前配置完整性校验"""
    errors = []
    warnings_list = []

    required_keys = [
        ("data.sampling_rate", lambda: hp["data"]["sampling_rate"]),
        ("train.max_epoches", lambda: hp["train"]["max_epoches"]),
    ]
    for key_path, accessor in required_keys:
        try:
            val = accessor()
            if val is None:
                errors.append(f"配置缺失: {key_path}")
        except (KeyError, TypeError):
            errors.append(f"配置缺失: {key_path}")

    try:
        sr = hp["data"]["sampling_rate"]
        if sr != 44100:
            warnings_list.append(
                f"采样率为 {sr}，VoiceFixer 预训练模型使用 44100Hz"
            )
    except KeyError:
        pass

    concrete = hp.get("concrete", {})
    if not concrete:
        warnings_list.append("未找到 'concrete' 配置块，将使用默认参数")

    weights_cfg = concrete.get("pretrained_weights", {})
    if weights_cfg:
        for module_name in ("analysis_module", "synthesis_module"):
            mod_cfg = weights_cfg.get(module_name, {})
            path = mod_cfg.get("path", "")
            if path and not os.path.isfile(path):
                warnings_list.append(f"预训练权重不存在: {module_name} → {path}")
    else:
        legacy = concrete.get("pretrained_checkpoint", "")
        if legacy and not os.path.isfile(legacy):
            warnings_list.append(f"预训练 checkpoint 不存在: {legacy}")

    stages = concrete.get("curriculum_stages", [])
    total_epochs = 0
    for i, stage in enumerate(stages):
        if "name" not in stage:
            errors.append(f"curriculum_stages[{i}] 缺少 'name'")
        if "epochs" not in stage:
            warnings_list.append(
                f"curriculum_stages[{i}] 缺少 'epochs'，将使用默认值 50"
            )
        total_epochs += stage.get("epochs", 50)

        phys = stage.get("physics_config", {})
        carrier = phys.get("carrier_freq", 40000)
        high_sr = phys.get("high_sr", 250000)
        if carrier >= high_sr / 2:
            errors.append(
                f"curriculum_stages[{i}]: carrier_freq ({carrier}) "
                f">= high_sr/2 ({high_sr // 2})，违反 Nyquist 定理"
            )

    if stages:
        print(f"[CONFIG] Curriculum: {len(stages)} 个阶段，总计 {total_epochs} epochs")

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        batch_size = hp.get("train", {}).get("batch_size", 8)
        if gpu_mem <= 10 and batch_size > 2:
            warnings_list.append(
                f"GPU 仅 {gpu_mem:.0f}GB 但 batch_size={batch_size}，"
                f"建议 batch_size=1~2 + accumulate_grad_batches=4~8"
            )

    if warnings_list:
        print("[CONFIG 警告]")
        for w in warnings_list:
            print(f"  ⚠ {w}")

    if errors:
        print("[CONFIG 错误]")
        for e in errors:
            print(f"  ✗ {e}")
        raise ValueError(f"配置校验失败，共 {len(errors)} 个错误")

    print("[CONFIG] ✓ 配置校验通过")


def validate_environment():
    """训练环境检查（Linux 增强版）"""
    issues = []

    if not torch.cuda.is_available():
        issues.append("未检测到 CUDA GPU")
    else:
        gpu_nums = torch.cuda.device_count()
        for i in range(gpu_nums):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"[ENV] GPU {i}: {name}, {mem:.1f}GB")

        # [Linux修复 3] 检查 NCCL 多卡环境
        if gpu_nums > 1:
            nccl_ver = torch.cuda.nccl.version() if hasattr(torch.cuda, 'nccl') else "unknown"
            print(f"[ENV] NCCL version: {nccl_ver}")
            # 检查网络接口配置
            if "NCCL_SOCKET_IFNAME" not in os.environ:
                print("[ENV] ⚠ 未设置 NCCL_SOCKET_IFNAME，多卡通信可能使用错误网口")
                print("      建议：export NCCL_SOCKET_IFNAME=eth0  (或实际网卡名)")

    print(f"[ENV] PyTorch Lightning: {pl.__version__} "
          f"({'PL2' if _USE_PL2 else 'PL1'})")
    print(f"[ENV] PyTorch: {torch.__version__}")
    print(f"[ENV] Python: {sys.version.split()[0]}")
    print(f"[ENV] OS: {platform.system()} {platform.release()}")

    # Linux 下检查文件描述符限制（DataLoader 多进程需要大量 fd）
    if platform.system() == "Linux":
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"[ENV] 文件描述符限制: soft={soft}, hard={hard}")
        if soft < 65536:
            print(
                f"[ENV] ⚠ 文件描述符 soft limit={soft} 较低，"
                "DataLoader 多进程可能报 'Too many open files'\n"
                "      建议：ulimit -n 65536"
            )

    try:
        tp_augment = os.path.join(git_root, "third_party", "augment")
        if os.path.isdir(tp_augment) and tp_augment not in sys.path:
            sys.path.insert(0, tp_augment)
        import augment  # type: ignore
        print(f"[ENV] augment: {augment.__file__}")
    except ImportError:
        tp_augment = os.path.join(git_root, "third_party", "augment")
        if os.path.isdir(tp_augment):
            print(f"[ENV] augment: 本地版本 ({tp_augment})")
        else:
            issues.append("未安装 augment 且 third_party/augment 不存在")

    try:
        import torchaudio
        print(f"[ENV] torchaudio: {torchaudio.__version__}")
    except ImportError:
        issues.append("未安装 torchaudio")

    try:
        import scipy
        print(f"[ENV] scipy: {scipy.__version__}")
    except ImportError:
        issues.append("未安装 scipy")

    try:
        import matplotlib
        print(f"[ENV] matplotlib: {matplotlib.__version__}")
    except ImportError:
        issues.append("未安装 matplotlib（AudioVisualLoggingCallback 依赖）")

    if issues:
        for issue in issues:
            print(f"[ENV] ⚠ {issue}")
    else:
        print("[ENV] ✓ 环境检查通过")

    return len(issues) == 0


def _setup_linux_nccl_env(gpu_nums: int):
    """
    [Linux修复 3] 配置 Linux 多 GPU NCCL 环境变量。
    在 `main()` 中调用，早于 Trainer 初始化。
    """
    if platform.system() != "Linux" or gpu_nums <= 1:
        return

    # 禁用 InfiniBand（AutoDL / 云主机通常无 IB，强制使用以太网）
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    # P2P 传输：云环境下建议关闭（避免 PCIe 拓扑不匹配导致挂起）
    os.environ.setdefault("NCCL_P2P_DISABLE", "0")
    # 等待超时（默认 30min，在慢节点上可适当延长）
    os.environ.setdefault("NCCL_TIMEOUT", "1800")
    # 关闭 NCCL 调试（生产模式）
    os.environ.setdefault("NCCL_DEBUG", "WARN")

    print("[NCCL] 多 GPU 环境变量已配置:")
    for k in ("NCCL_IB_DISABLE", "NCCL_P2P_DISABLE", "NCCL_TIMEOUT",
              "NCCL_DEBUG", "NCCL_SOCKET_IFNAME"):
        if k in os.environ:
            print(f"  {k}={os.environ[k]}")


# ============================================================================
#  主入口
# ============================================================================


def _parse_args():
    """
    [代码修复 2] 提取命令行参数解析，支持 --start_stage 覆盖硬编码值。
    其余参数由 tools.utils.get_hparams() 处理。
    """
    parser = argparse.ArgumentParser(
        description="混凝土穿墙语音恢复 Curriculum 训练",
        add_help=False,  # 避免与 tools.utils 的 argparse 冲突
    )
    parser.add_argument(
        "--start_stage",
        type=int,
        default=0,
        help="从第几个 Curriculum Stage 开始训练（0-indexed，默认 0）",
    )
    # 仅解析已知参数，未知参数留给 tools.utils
    args, _ = parser.parse_known_args()
    return args


def _hparams_to_dict(obj):
    if isinstance(obj, dict):
        return {k: _hparams_to_dict(v) for k, v in obj.items()}
    if hasattr(obj, 'keys') and hasattr(obj, '__getitem__'):
        return {k: _hparams_to_dict(obj[k]) for k in obj.keys()}
    return obj


def main():
    """主入口：Curriculum Learning 训练管线（Linux 优化版）。"""

    # [代码修复 2] 通过命令行控制起始 Stage，不再硬编码
    cli_args = _parse_args()
    START_STAGE = cli_args.start_stage

    # Linux 下 DataLoader 默认使用 fork，无需强制 spawn
    # 但在极少数情况下（使用 CUDA 初始化后 fork）需要改为 forkserver
    if platform.system() == "Linux":
        # 仅在 CUDA 已初始化的子进程场景下才需要 forkserver
        # 正常主进程启动时保持 fork（最快）
        pass
    elif platform.system() == "Windows":
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

    # [Linux修复 2] 注册优雅退出信号
    _register_signal_handlers()

    env_ok = validate_environment()
    if not env_ok:
        print(
            "[ENV] 环境检查未通过，建议安装缺失的依赖并确保至少有一块 CUDA GPU"
        )
        return

    gpu_nums = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # [Linux修复 3] 配置 NCCL 多 GPU 环境变量
    _setup_linux_nccl_env(gpu_nums)

    hp_result = tools.utils.get_hparams()
    hp = hp_result[0] if isinstance(hp_result, tuple) else hp_result

    if hasattr(hp, 'to_dict'):
        hp_dict = hp.to_dict()
    elif isinstance(hp, dict):
        hp_dict = hp
    else:
        hp_dict = {}
        for k in hp.keys():
            v = hp[k]
            hp_dict[k] = _hparams_to_dict(v) if hasattr(v, 'keys') else v
    hp = hp_dict

    assert hp["data"]["sampling_rate"] == 44100, (
        f"采样率必须为 44100，当前: {hp['data']['sampling_rate']}"
    )

    hp["root"] = git_root

    # 将相对路径转换为绝对路径
    for split in ("train_dataset", "val_dataset"):
        if split in hp.get("data", {}):
            for category in hp["data"][split]:
                for sub_key in hp["data"][split][category]:
                    path = hp["data"][split][category][sub_key]
                    if isinstance(path, str) and not os.path.isabs(path):
                        hp["data"][split][category][sub_key] = os.path.join(
                            hp["root"], path
                        )

    if "rir_root" in hp.get("augment", {}).get("params", {}):
        rir_path = hp["augment"]["params"]["rir_root"]
        if not os.path.isabs(rir_path):
            hp["augment"]["params"]["rir_root"] = os.path.join(
                hp["root"], rir_path
            )

    validate_config(hp)

    model_dir = hp.get("model_dir", "exp/concrete_v1")
    logger = TensorBoardLogger(
        os.path.dirname(model_dir) or ".",
        name=os.path.basename(model_dir),
    )
    hp["log_dir"] = logger.log_dir

    model = ConcreteVoiceFixer(hp, channels=1, type_target="vocals")

    distributed = gpu_nums > 1
    dm = ConcreteDataModule(hp, distributed=distributed)

    curriculum = CurriculumManager(hp, model, dm)
    stages = curriculum.stages

    resume_ckpt = hp["train"].get("resume_from_checkpoint", "") or None

    # [代码修复 2] START_STAGE 来自命令行，可覆盖
    print(f"[TRAIN] 从 Stage {START_STAGE} 开始训练 "
          f"(共 {len(stages)} 个 Stage)")

    # 累计之前跳过的 epochs，保证 max_epochs 语义正确
    cumulative_epochs = sum(
        curriculum.get_stage_epochs(i) for i in range(START_STAGE)
    )

    for stage_idx in range(START_STAGE, len(stages)):
        stage = stages[stage_idx]
        curriculum.apply_stage(stage_idx)

        stage_epochs = curriculum.get_stage_epochs(stage_idx)
        cumulative_epochs += stage_epochs
        print(
            f"[TRAIN] Stage {stage_idx + 1}: 训练 {stage_epochs} epochs "
            f"(全局 epoch 上限: {cumulative_epochs})"
        )

        callbacks = [
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                filename=f"stage{stage_idx}_" + "{epoch}-{step}-{val_loss:.4f}",
                dirpath=os.path.join(logger.log_dir, "checkpoints"),
                save_top_k=hp["train"].get("save_top_k", 3),
                monitor="val_loss",
                mode="min",
                save_last=True,
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=hp["train"].get("early_stop_patience", 15),
                mode="min",
                verbose=True,
            ),
            initLogDir(hp, current_dir=os.getcwd()),
            AudioVisualLoggingCallback(
                sample_rate=hp["data"]["sampling_rate"],
                n_fft=hp.get("mel", {}).get("n_fft", 2048),
                hop_length=hp.get("mel", {}).get("hop_length", 512),
                n_mels=hp.get("mel", {}).get("n_mels", 128),
                max_samples=hp.get("log", {}).get("visual_max_samples", 3),
                log_every_n_epochs=hp.get("log", {}).get(
                    "visual_every_n_epochs", 1
                ),
            ),
        ]

        if TQDMProgressBar is not None:
            callbacks.append(
                TQDMProgressBar(
                    refresh_rate=hp.get("log", {}).get(
                        "progress_bar_refresh_rate", 10
                    )
                )
            )

        trainer_kwargs = dict(
            max_epochs=cumulative_epochs,
            detect_anomaly=hp["train"].get("detect_anomaly", False),
            num_sanity_val_steps=2,
            callbacks=callbacks,
            check_val_every_n_epoch=hp["train"].get("check_val_every_n_epoch", 1),
            logger=logger,
            log_every_n_steps=hp.get("log", {}).get("log_every_n_steps", 50),
            gradient_clip_val=hp.get("concrete", {}).get("grad_clip_norm", 5.0),
            gradient_clip_algorithm="norm",
            accumulate_grad_batches=hp["train"].get("accumulate_grad_batches", 4),
        )

        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_mem <= 10:
                trainer_kwargs["precision"] = "16-mixed" if _USE_PL2 else 16
                print(f"[TRAIN] GPU {gpu_mem:.1f}GB → FP16 混合精度")

        if _USE_PL2:
            if gpu_nums > 1:
                trainer_kwargs["strategy"] = "ddp_find_unused_parameters_true"
                trainer_kwargs["devices"] = gpu_nums
                trainer_kwargs["accelerator"] = "gpu"
                trainer_kwargs["sync_batchnorm"] = True
            elif gpu_nums == 1:
                trainer_kwargs["devices"] = 1
                trainer_kwargs["accelerator"] = "gpu"
            else:
                trainer_kwargs["accelerator"] = "cpu"
        else:
            if gpu_nums > 1:
                trainer_kwargs["gpus"] = gpu_nums
                if DDPStrategy is not None:
                    trainer_kwargs["strategy"] = DDPStrategy(
                        find_unused_parameters=True
                    )
                trainer_kwargs["sync_batchnorm"] = True
            elif gpu_nums == 1:
                trainer_kwargs["gpus"] = 1

        if _USE_PL2:
            ckpt_path = (
                resume_ckpt if (stage_idx == START_STAGE and resume_ckpt) else None
            )
        else:
            if stage_idx == START_STAGE and resume_ckpt:
                trainer_kwargs["resume_from_checkpoint"] = resume_ckpt

        trainer = Trainer(**trainer_kwargs)

        # [Linux修复 2] 保存 Trainer 引用供信号处理器使用
        global _trainer_ref
        _trainer_ref = trainer

        lr_scale = stage.get("lr_scale", 1.0)
        if lr_scale != 1.0 and lr_scale > 0 and stage_idx > 0:
            opt_cfg = hp.get("concrete", {}).get("optimizer", {})
            original_lr = opt_cfg.get("base_lr", 1e-4)
            new_lr = original_lr * lr_scale
            opt_cfg["base_lr"] = new_lr
            print(f"[TRAIN] LR 缩放: {original_lr:.2e} × {lr_scale} = {new_lr:.2e}")

        dm.setup("fit")

        if _USE_PL2 and ckpt_path:
            trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
        else:
            trainer.fit(model, datamodule=dm)

        stage_ckpt = os.path.join(
            logger.log_dir, "checkpoints", f"stage_{stage_idx}_final.ckpt"
        )
        trainer.save_checkpoint(stage_ckpt)
        print(f"[STAGE {stage_idx}] 完成 → {stage_ckpt}")

        resume_ckpt = None
        # Stage 结束后恢复原始 LR，防止影响下一 Stage
        if lr_scale != 1.0 and lr_scale > 0 and stage_idx > 0:
            opt_cfg = hp.get("concrete", {}).get("optimizer", {})
            opt_cfg["base_lr"] = opt_cfg["base_lr"] / lr_scale

    _trainer_ref = None
    print("\n" + "=" * 60)
    print("  全部训练阶段完成")
    print("=" * 60)


if __name__ == "__main__":
    main()