"""
WavAugment 的 Windows 兼容替代层。

原始 augment (WavAugment) 依赖 torchaudio.sox_effects，
而 SoX 后端在 Windows 上不可用。

本模块用 torchaudio.transforms + scipy.signal 实现等效功能，
对外保持 EffectChain API 完全一致。
"""

import sys
import warnings

# 尝试加载原始 augment，如果失败则使用本地替代
try:
    # 检查 SoX 是否真的可用
    import torchaudio
    torchaudio.sox_effects.effect_names()
    # SoX 可用（Linux/macOS），使用原始库
    from augment.effects import EffectChain  # noqa: F401
except (RuntimeError, OSError, ImportError):
    # SoX 不可用（Windows），使用本地替代
    from .effect_chain import EffectChain  # noqa: F401
    warnings.warn(
        "[augment] SoX 后端不可用 (Windows)，已切换到纯 PyTorch/scipy 替代实现",
        stacklevel=2,
    )

__all__ = ["EffectChain"]