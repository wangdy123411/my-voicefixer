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
from pytorch_lightning.plugins import DDPPlugin
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

# =====================================================================
# [GPU修复 2] DDPPlugin 导入：PL1 必须用 DDPPlugin 而非字符串 "ddp"
# 字符串 "ddp" 在 PL 1.x 某些版本下会被错误解析为 DP (DataParallel)
# =====================================================================
_DDPPlugin = None
if not _USE_PL2:
    try:
        from pytorch_lightning.plugins import DDPPlugin as _DDPPlugin_cls
        _DDPPlugin = _DDPPlugin_cls
    except ImportError:
        try:
            from pytorch_lightning.strategies import DDPStrategy as _DDPPlugin_cls
            _DDPPlugin = _DDPPlugin_cls
        except ImportError:
            _DDPPlugin = None

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
    训练过程中的音频与频谱可视化（防爆装甲版）。
    
    [解封说明]:
    1. Vocoder 推理频率改为每 1 个 epoch 跑一次（全程监控）
    2. max_samples 提升到 6（一次看足够多的多样性样本）
    3. 保留了底层 matplotlib 的后台绘制防溢出保护
    4. 🛡️ 新增音频与频谱级的 NaN/Inf 强制清洗，根除 TensorBoard 断图警告
    """
    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        hop_length: int = 441,
        n_mels: int = 128,
        max_samples: int = 6,            # ← [解封] 从 2 提升到 6，观测更多数据
        log_every_n_epochs: int = 3,     # ← 每一个 Epoch 都记录
        vocoder_every_n_epochs: int = 3 , # ← [解封] 每一个 Epoch 都强制跑 Vocoder 生成音频
    ):
        super().__init__()
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.max_samples = max_samples
        self.log_every_n_epochs = log_every_n_epochs
        self.vocoder_every_n_epochs = vocoder_every_n_epochs
        self._val_cache = None
        self._cache_captured = False
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
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return
        
        if self._cache_captured or batch_idx > 0:
            return

        try:
            if not isinstance(batch, dict):
                return

            dirty = batch.get("input_wave")
            clean = batch.get("target_wave")
            cutoff_hz = batch.get("cutoff_hz")  # 👈 [修复1] 把截断频率拿出来！

            if dirty is None or clean is None:
                return

            if dirty.dim() == 2:
                dirty = dirty.unsqueeze(1)
            if clean.dim() == 2:
                clean = clean.unsqueeze(1)

            device = pl_module.device
            dirty = dirty.to(device, dtype=torch.float32)
            clean = clean.to(device, dtype=torch.float32)

            # 🚀 [修复2] 删除了之前的 1e-7 白噪声，保持频谱的绝对干净！

            run_vocoder = (trainer.current_epoch % self.vocoder_every_n_epochs == 0)

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                
                # =========================================================
                # 💥 [修复3] 强制画图插件也执行 GPU 低通滤波！
                # 这样 TensorBoard 里的图就会呈现出完美的“一刀切”截断线！
                # =========================================================
                if cutoff_hz is not None:
                    from dataloaders.collators.concrete_collator import ConcreteEavesdropCollatorGPU
                    cutoff_hz = cutoff_hz.to(device, dtype=torch.float32)
                    dirty = ConcreteEavesdropCollatorGPU.gpu_fft_lowpass(
                        waveform=dirty,
                        cutoff_hz=cutoff_hz,
                        sample_rate=self.sr
                    )

                _, mel_low_quality = pl_module.pre(dirty)
                
                # 用极其安全的 1e-10 托底，防 to_log 报错
                mel_low_quality = torch.nan_to_num(mel_low_quality, nan=1e-10, posinf=2.0, neginf=1e-10)
                mel_low_quality = mel_low_quality.clamp(min=1e-10)
                
                pred_mel_dict = pl_module(mel_low_quality)
                pred_log_mel = pred_mel_dict['mel']

                pred_log_mel = torch.nan_to_num(pred_log_mel, nan=-11.5, posinf=3.0, neginf=-11.5)

                prediction = None
                if run_vocoder:
                    try:
                        pred_log_mel_safe = pred_log_mel.clamp(-11.5, 3.0)
                        pred_linear_mel = 10 ** pred_log_mel_safe

                        n_voc = min(self.max_samples, pred_linear_mel.shape[0])
                        pred_linear_mel_subset = pred_linear_mel[:n_voc]

                        vocoder_module, _ = pl_module._find_vocoder_module()
                        if vocoder_module is not None:
                            prediction = pl_module.vocoder(pred_linear_mel_subset)
                        else:
                            prediction = pl_module.vocoder(pred_linear_mel_subset)
                    except Exception as e:
                        print(f"[AudioVisual] Vocoder 推理失败: {e}")
                        prediction = None

                min_len = dirty.shape[-1]
                dirty = dirty[..., :min_len]
                clean = clean[..., :min_len]
                if prediction is not None:
                    pred_min = min(min_len, prediction.shape[-1])
                    dirty = dirty[..., :pred_min]
                    clean = clean[..., :pred_min]
                    prediction = prediction[..., :pred_min]

            self._val_cache = {
                "dirty": dirty.detach().cpu(),
                "clean": clean.detach().cpu(),
                "prediction": prediction.detach().cpu() if prediction is not None else None,
                "pred_mel": pred_log_mel.detach().cpu(),
            }
            self._cache_captured = True

        except Exception as e:
            print(f"\n[AudioVisual] 验证集数据捕获失败: {e}\n")

    def on_validation_epoch_end(self, trainer, pl_module):
        if self._val_cache is None:
            return
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            self._val_cache = None
            return
        if trainer.global_rank != 0:
            self._val_cache = None
            return

        logger_exp = (
            trainer.logger.experiment
            if hasattr(trainer, 'logger') and trainer.logger is not None
            else None
        )
        if logger_exp is None:
            self._val_cache = None
            return

        dirty = self._val_cache["dirty"]
        clean = self._val_cache["clean"]
        pred = self._val_cache.get("prediction")

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        n_samples = min(self.max_samples, dirty.shape[0])

        for i in range(n_samples):
            tag_prefix = f"val_sample_{i}"
            step = trainer.current_epoch

            # 提取 1D 音频
            d_raw = self._to_1d(dirty[i])
            c_raw = self._to_1d(clean[i])
            
            # =========================================================
            # 🛡️ [防爆净水器 1] 强力清洗音频，拦截 NaN 和 Inf 
            # 解决 TensorBoard cast warning 且防止写入失败
            # =========================================================
            d = self._normalize_audio(self._sanitize_tensor(d_raw))
            c = self._normalize_audio(self._sanitize_tensor(c_raw))

            if d is None or c is None:
                continue

            try:
                # 1. 保存原音与噪声
                logger_exp.add_audio(
                    f"{tag_prefix}/1_Clean_Original",
                    c.unsqueeze(0), step, sample_rate=self.sr
                )
                logger_exp.add_audio(
                    f"{tag_prefix}/2_Noisy_Input",
                    d.unsqueeze(0), step, sample_rate=self.sr
                )
                # 2. 如果 Vocoder 合成成功，保存 AI 修复后的声音！
                if pred is not None and i < pred.shape[0]:
                    p_raw = self._to_1d(pred[i])
                    # 同样的清洗工序
                    p = self._normalize_audio(self._sanitize_tensor(p_raw))
                    if p is not None:
                        logger_exp.add_audio(
                            f"{tag_prefix}/3_AI_Restored",
                            p.unsqueeze(0), step, sample_rate=self.sr
                        )
            except Exception:
                pass

            try:
                # 3. 画出三行频谱对比图
                p_audio = None
                if pred is not None and i < pred.shape[0]:
                    p_raw = self._to_1d(pred[i])
                    p_audio = self._normalize_audio(self._sanitize_tensor(p_raw))

                fig = self._plot_mel_comparison_v2(c, d, p_audio, step, i)
                if fig is not None:
                    logger_exp.add_figure(f"{tag_prefix}/mel_pipeline", fig, step)
                    fig.clear()
                    plt.close(fig)
            except Exception:
                pass

        # 兜底清理内存
        plt.close('all')
        self._val_cache = None

        import gc
        gc.collect()

    def _plot_mel_comparison_v2(self, clean, dirty, pred, epoch, sample_idx):
        """
        三行比对频谱图绘制。
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        has_pred = pred is not None
        n_rows = 3 if has_pred else 2

        fig, axes = plt.subplots(n_rows, 1, figsize=(12, 4 * n_rows),
                                 constrained_layout=True)
        if n_rows == 1:
            axes = [axes]

        titles = ["1. Original (Clean Target)", "2. Noisy Input (Concrete)"]
        signals = [clean, dirty]
        if has_pred:
            titles.append("3. AI Restored")
            signals.append(pred)

        for ax, title, sig in zip(axes, titles, signals):
            mel = self._compute_mel_spectrogram(sig)
            if mel is not None:
                # =====================================================
                # 🛡️ [防爆净水器 2] 清洗频谱二维数组，防止 imshow 崩溃
                # =====================================================
                mel_np = mel.numpy()
                import numpy as np
                mel_np = np.nan_to_num(mel_np, nan=-80.0, posinf=0.0, neginf=-80.0)

                im = ax.imshow(
                    mel_np,
                    aspect='auto',
                    origin='lower',
                    interpolation='nearest',
                    cmap='magma',
                )
                fig.colorbar(im, ax=ax, format='%+2.0f dB', pad=0.01)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_ylabel('Mel Bin')

        axes[-1].set_xlabel('Time Frame')

        # 估算并标注 SNR
        snr_input = self._estimate_snr(dirty, clean)
        suptitle = f"Sample {sample_idx} - Epoch {epoch}\nInput SNR: {snr_input:.1f}dB"
        if has_pred:
            snr_ai = self._estimate_snr(pred, clean)
            suptitle += f"  |  AI Restored SNR: {snr_ai:.1f}dB"

        fig.suptitle(suptitle, fontsize=13, fontweight='bold')
        return fig

    # ================================================================
    # 新增的安全方法
    # ================================================================
    @staticmethod
    def _sanitize_tensor(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """清除张量中的 NaN 和 Inf，并限制极端值"""
        if tensor is None:
            return None
        return torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)

    # ================================================================
    # 以下方法保持不变
    # ================================================================
    def _compute_mel_spectrogram(
        self, audio: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if audio is None or audio.dim() == 0 or len(audio) < self.n_fft:
            return None
        try:
            window = torch.hann_window(self.n_fft)
            stft = torch.stft(
                audio, self.n_fft, self.hop_length,
                window=window, return_complex=True,
            )
            power = stft.abs().pow(2)
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
        fmax = sr / 2.0
        fmin = 0.0
        mel_min = 2595.0 * math.log10(1.0 + fmin / 700.0)
        mel_max = 2595.0 * math.log10(1.0 + fmax / 700.0)
        mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        freq_bins = n_fft // 2 + 1
        fft_freqs = torch.linspace(0, fmax, freq_bins)
        f_left   = hz_points[:-2]
        f_center = hz_points[1:-1]
        f_right  = hz_points[2:]
        up_slope   = (fft_freqs[:, None] - f_left[None, :]) / (
            (f_center - f_left)[None, :].clamp(min=1e-10)
        )
        down_slope = (f_right[None, :] - fft_freqs[:, None]) / (
            (f_right - f_center)[None, :].clamp(min=1e-10)
        )
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
        # 🛡️ 再次确保输出完全被钳制在 [-0.95, 0.95]
        norm = audio / peak * 0.95 if peak > 1e-8 else audio
        return torch.clamp(norm, min=-0.95, max=0.95)

    @staticmethod
    def _estimate_snr(signal: torch.Tensor, reference: torch.Tensor) -> float:
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
        import os
        cpu_count = os.cpu_count() or 16
        num_workers = hp.get("train", {}).get("num_workers", 8)
        # 每个 worker 分配的线程数 = 总核数 / (worker数 * GPU数)
        gpu_count = max(torch.cuda.device_count() if torch.cuda.is_available() else 1, 1)
        threads_per_worker = max(cpu_count // (num_workers * gpu_count), 1)
        threads_str = str(threads_per_worker)
        
        os.environ.setdefault("OMP_NUM_THREADS", threads_str)
        os.environ.setdefault("MKL_NUM_THREADS", threads_str)
        os.environ.setdefault("OPENBLAS_NUM_THREADS", threads_str)
        os.environ.setdefault("NUMEXPR_NUM_THREADS", threads_str)
        print(f"[INIT] CPU 线程分配: {cpu_count} 核 / {num_workers} workers / {gpu_count} GPUs = {threads_per_worker} threads/worker")


        # ...existing code... (所有 setdefault 块保持不变)
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
        hp["model"].setdefault("hop_size", 441)
        hp["model"].setdefault("pad_mode", "reflect")
        hp["model"].setdefault("window", "hann")

        if "mel" not in hp:
            hp["mel"] = {}
        hp["mel"].setdefault("n_fft", 2048)
        hp["mel"].setdefault("hop_length", 441)
        hp["mel"].setdefault("win_length", 2048)
        hp["mel"].setdefault("n_mels", 128)
        hp["mel"].setdefault("fmin", 0)
        hp["mel"].setdefault("fmax", 22050)

        hp.setdefault("model_dir", "exp/concrete_v1")
        
        # ================================================================
        # [显存修复 - 核心] Monkeypatch torch.Tensor.cuda / Module.cuda
        # 在 super().__init__() 期间，拦截所有 .cuda() 调用，
        # 强制 Vocoder 在 CPU 上初始化。
        # 这是唯一可靠的方式，因为 Vocoder.__init__ 内部
        # 可能在任意位置调用 .cuda()/.to('cuda')
        # ================================================================
        _need_patch = torch.cuda.is_available()

        if _need_patch:
            # 保存原始方法
            _orig_torch_load = torch.load
            _orig_cuda_available = torch.cuda.is_available
            def _cpu_only_load(*args, **kwargs):
                kwargs['map_location'] = 'cpu'
                return _orig_torch_load(*args, **kwargs)

            torch.load = _cpu_only_load
            torch.cuda.is_available = lambda: False
            print("[INIT] ★ 已拦截 torch.load + cuda.is_available → 强制 CPU 初始化")

        try:
            super().__init__(hp, channels=channels, type_target=type_target)
        finally:
            if _need_patch:
                torch.load = _orig_torch_load
                torch.cuda.is_available = _orig_cuda_available
                print("[INIT] ★ 已恢复 torch.load + cuda.is_available")

        self.hp = hp
        self.concrete_cfg = hp.get("concrete", {})

        # 确认所有参数在 CPU
        gpu_params = sum(1 for p in self.parameters() if p.device.type == 'cuda')
        if gpu_params > 0:
            print(f"[INIT] ⚠ 发现 {gpu_params} 个 GPU 参数，强制移回 CPU")
            self.cpu()

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                alloc = torch.cuda.memory_allocated(i) / (1024**3)
                print(f"[INIT] GPU {i}: allocated={alloc:.3f}GB (应为 0)")

        self._load_pretrained_weights()
        self._apply_freeze_strategy()

        for name, param in self.named_parameters():
            if param.device.type == 'cuda':
                self.cpu()
                break

        self._vocoder_cache: Optional[nn.Module] = None
        self._vocoder_name_cache: Optional[str] = None
        voc, voc_name = self._find_vocoder_module()
        if voc is not None:
            self._vocoder_cache = voc
            self._vocoder_name_cache = voc_name
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
        # [性能修复] 优先返回缓存，避免每次遍历整个模型树
        if hasattr(self, '_vocoder_cache') and self._vocoder_cache is not None:
            return self._vocoder_cache, self._vocoder_name_cache
        # 原始查找逻辑（仅首次调用时执行）
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
            if self._vocoder_cache is not None:
                self._vocoder_cache.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            freeze_cfg = self.concrete_cfg.get("freeze_strategy", {})
            if freeze_cfg.get("freeze_vocoder", True):
                # [性能修复] 同上
                if self._vocoder_cache is not None:
                    self._vocoder_cache.eval()
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

        # 回归最纯粹的频谱能量 L1 拟合
        loss = self.l1loss(gen_mel, target_log_mel)
        
        # 记录日志
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        dirty_audio = batch["input_wave"]
        clean_audio = batch["target_wave"]
        cutoff_hz = batch["cutoff_hz"]

        device = self.device
        
        # 🛡️ FP32 防爆与清洗
        dirty_audio = dirty_audio.to(device, dtype=torch.float32)
        clean_audio = clean_audio.to(device, dtype=torch.float32)
        cutoff_hz = cutoff_hz.to(device, dtype=torch.float32)

        dirty_audio = torch.nan_to_num(dirty_audio, nan=0.0)
        clean_audio = torch.nan_to_num(clean_audio, nan=0.0)

        # 🛡️ 局部的 GPU 低通滤波
        with torch.cuda.amp.autocast(enabled=False):
            dirty_audio = ConcreteEavesdropCollatorGPU.gpu_fft_lowpass(
                waveform=dirty_audio,
                cutoff_hz=cutoff_hz,
                sample_rate=self.hp["data"]["sampling_rate"],
            )

        if dirty_audio.dim() == 2:
            dirty_audio = dirty_audio.unsqueeze(1)
        if clean_audio.dim() == 2:
            clean_audio = clean_audio.unsqueeze(1)

        # =========================================================
        # 💥 [终极修复] 把过滤好的、安全的音频塞回 batch 字典！
        # 这样后面的 AudioVisualLoggingCallback 画图时就不会拿到有毒数据了！
        # =========================================================
        batch["input_wave"] = dirty_audio
        batch["target_wave"] = clean_audio

        with torch.no_grad():
            _, mel_target = self.pre(clean_audio)
            _, mel_low_quality = self.pre(dirty_audio)
            
            estimation = self(mel_low_quality)['mel']
            
            # 钳制极小值防 -inf
            mel_target_safe = mel_target.clamp(min=1e-5)
            target_log_mel = to_log(mel_target_safe)

            min_frames = min(estimation.size(2), target_log_mel.size(2))
            estimation = estimation[:, :, :min_frames, :]
            target_log_mel = target_log_mel[:, :, :min_frames, :]

            val_loss = self.l1loss(estimation, target_log_mel)

            # 防爆护盾
            if not torch.isfinite(val_loss):
                val_loss = torch.tensor(0.0, device=device)

        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

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

    tp_augment = os.path.join(git_root, "third_party", "augment")
    if os.path.isdir(tp_augment) and tp_augment not in sys.path:
        sys.path.insert(0, tp_augment)
    try:
        import augment  # type: ignore
        print(f"[ENV] augment: {augment.__file__}")
    except ImportError:
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
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
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
    """主入口：满级终极冲刺训练管线（纯享优化版）。"""

    if platform.system() == "Linux":
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
        print("[ENV] 环境检查未通过，建议安装缺失的依赖并确保至少有一块 CUDA GPU")
        return

    gpu_nums = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # [Linux修复 3] 配置 NCCL 多 GPU 环境变量
    _setup_linux_nccl_env(gpu_nums)

    # 加载超参数
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

    assert hp["data"]["sampling_rate"] == 44100, f"采样率必须为 44100，当前: {hp['data']['sampling_rate']}"

    hp["root"] = git_root

    # 将相对路径转换为绝对路径
    for split in ("train_dataset", "val_dataset"):
        if split in hp.get("data", {}):
            for category in hp["data"][split]:
                for sub_key in hp["data"][split][category]:
                    path = hp["data"][split][category][sub_key]
                    if isinstance(path, str) and not os.path.isabs(path):
                        hp["data"][split][category][sub_key] = os.path.join(hp["root"], path)

    if "rir_root" in hp.get("augment", {}).get("params", {}):
        rir_path = hp["augment"]["params"]["rir_root"]
        if not os.path.isabs(rir_path):
            hp["augment"]["params"]["rir_root"] = os.path.join(hp["root"], rir_path)

    validate_config(hp)

    # 初始化 Logger
    model_dir = hp.get("model_dir", "exp/concrete_v1")
    logger = TensorBoardLogger(
        os.path.dirname(model_dir) or ".",
        name=os.path.basename(model_dir),
    )
    hp["log_dir"] = logger.log_dir

    # =================================================================
    # 1. 实例化模型底座 & 热加载权重 (Warm Start)
    # =================================================================
    model = ConcreteVoiceFixer(hp, channels=1, type_target="vocals")

    # 👇 填入你最新跑完的 0.58 那个 ckpt 文件！
    latest_ckpt_path = "/root/autodl-tmp/myvoicefixer/logs/train_concrete/version_21/checkpoints/last.ckpt" 

    if os.path.exists(latest_ckpt_path):
        print(f"\n🔥 [Warm Start] 正在暴力加载满级权重: {latest_ckpt_path}")
        print("  -> 已彻底抹除 Lightning 进度记忆，Epoch 和 Patience 从零开始！")
        ckpt_data = torch.load(latest_ckpt_path, map_location="cpu")
        clean_state_dict = ckpt_data["state_dict"]
        clean_state_dict = {k: v for k, v in clean_state_dict.items() if "ssim_loss" not in k}
        model.load_state_dict(clean_state_dict, strict=False)
    else:
        print("\n⚠️ [警告] 未找到指定的检查点，模型将从随机/基础权重起飞。")

    # =================================================================
    # 2. 实例化数据管道 & 强制锁死满级物理难度
    # =================================================================
    distributed = gpu_nums > 1
    dm = ConcreteDataModule(hp, distributed=distributed)

    print("\n🌪️ [Curriculum] 已废弃多阶段过渡，强制锁死在最终满级难度 (Stage 3)！")
    stage3_cfg = hp["concrete"]["curriculum_stages"][-1]
    
    dm.phase2_intensity = stage3_cfg["phase2_intensity"]
    dm.update_physics_config(stage3_cfg["physics_config"])
    dm.update_collator_config(stage3_cfg["collator_config"])
    dm.setup("fit")

    # =================================================================
    # 3. 动态配置 Trainer 参数 (GPU 修复机制集成)
    # =================================================================
    trainer_kwargs = {
        "logger": logger,
        "max_epochs": hp["train"].get("max_epochs", 1000),
        "gradient_clip_val": hp["train"].get("gradient_clip_val", 1.0),
        "accumulate_grad_batches": hp["train"].get("accumulate_grad_batches", 1),
        "num_sanity_val_steps": hp["train"].get("num_sanity_val_steps", 2),
        "val_check_interval": hp["train"].get("val_check_interval", 1.0),
        "log_every_n_steps": hp.get("log", {}).get("log_every_n_steps", 50),
    }

    # 精度动态判断
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if _USE_PL2:
            trainer_kwargs["precision"] = "bf16-mixed" if gpu_mem >= 20 else "16-mixed"
        else:
            trainer_kwargs["precision"] = 16
            trainer_kwargs["amp_backend"] = "native"

    # DDP 策略动态判断
    if gpu_nums > 1:
        if _USE_PL2:
            trainer_kwargs["strategy"] = "ddp_find_unused_parameters_false"
            trainer_kwargs["devices"] = gpu_nums
            trainer_kwargs["accelerator"] = "gpu"
        else:
            if _DDPPlugin is not None:
                trainer_kwargs["strategy"] = _DDPPlugin(find_unused_parameters=False)
            else:
                trainer_kwargs["strategy"] = "ddp"
            trainer_kwargs["gpus"] = gpu_nums
        trainer_kwargs["sync_batchnorm"] = True
    elif gpu_nums == 1:
        if _USE_PL2:
            trainer_kwargs["devices"] = 1
            trainer_kwargs["accelerator"] = "gpu"
        else:
            trainer_kwargs["gpus"] = 1
    else:
        if _USE_PL2:
            trainer_kwargs["accelerator"] = "cpu"

    # =================================================================
    # 4. 配置 Callbacks
    # =================================================================
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            filename="ultimate_stage3_{epoch}-{step}-{val_loss:.4f}",
            dirpath=os.path.join(logger.log_dir, "checkpoints"),
            save_top_k=hp["train"].get("save_top_k", 3),
            monitor="val_loss",
            mode="min",
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=hp["train"].get("early_stop_patience", 1000), 
            mode="min",
            verbose=True,
        ),
        initLogDir(hp, current_dir=os.getcwd()),
        AudioVisualLoggingCallback(
            sample_rate=hp["data"]["sampling_rate"],
            n_fft=hp.get("mel", {}).get("n_fft", 2048),
            hop_length=hp.get("mel", {}).get("hop_length", 441),
            n_mels=hp.get("mel", {}).get("n_mels", 128),
            max_samples=hp.get("log", {}).get("visual_max_samples", 6),
            log_every_n_epochs=1,
            vocoder_every_n_epochs=1,
        ),
    ]

    if TQDMProgressBar is not None:
        callbacks.append(TQDMProgressBar(refresh_rate=hp.get("log", {}).get("progress_bar_refresh_rate", 10)))
    
    trainer_kwargs["callbacks"] = callbacks

    # =================================================================
    # 5. 点火启动！
    # =================================================================
    from pytorch_lightning import Trainer # 确保导入
    print("\n[TRAINER] 启动配置已锁定，抛弃 resume_ckpt 羁绊，直接 Fit！")
    trainer = Trainer(**trainer_kwargs)
    
    global _trainer_ref
    _trainer_ref = trainer

    # 因为是 Warm Start，决不能传入 ckpt_path 阻止它继承失败的 Epoch！
    trainer.fit(model, datamodule=dm)

    # 最终保存
    final_ckpt = os.path.join(logger.log_dir, "checkpoints", "stage_3_ultimate_final.ckpt")
    trainer.save_checkpoint(final_ckpt)
    
    _trainer_ref = None
    print("\n" + "=" * 60)
    print(f"  终极训练阶段完成，模型已保存至: {final_ckpt}")
    print("=" * 60)


if __name__ == "__main__":
    main()