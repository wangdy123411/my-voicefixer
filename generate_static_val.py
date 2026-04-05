# -*- coding: utf-8 -*-
import os
import glob
import random
import json
import torch
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# =========================================================================
# 1. 直接复用训练管道的【原生组件】
# =========================================================================
from dataloaders.augmentation.concrete_physics import ConcretePhysicsChain
from dataloaders.concrete_dataset import ConcreteAugDataset

# 临时继承 Dataset 以调用它原生的环境噪音混合逻辑（无需真的初始化 Dataloader）
class DiagnosticMixer(ConcreteAugDataset):
    def __init__(self, hp, target_sr):
        # 欺骗性初始化，只需要它的 _mix_noise 方法和底噪加载功能
        self.sample_rate = target_sr
        self.segment_length = hp["data"]["segment_length"]
        self.concrete_noise_prob = hp["concrete"]["concrete_noise_prob"]
        
        # 加载通用底噪池 (模拟 _preload_all_noise)
        noise_dir = hp["data"]["train_dataset"]["speech"]["noise"]
        noise_files = glob.glob(os.path.join(noise_dir, "**", "*.*"), recursive=True)
        noise_files = [f for f in noise_files if f.endswith(('.wav', '.flac'))][:100]
        self._noise_pool = np.concatenate([librosa.load(f, sr=target_sr)[0] for f in noise_files])
        self._noise_pool_len = len(self._noise_pool)
        
        # 加载真实物理底噪 (模拟 _preload_concrete_noise)
        concrete_dir = hp["data"]["train_dataset"]["speech"]["concrete_noise"]
        self._concrete_noise_list = []
        for f in glob.glob(os.path.join(concrete_dir, "*.wav")):
            audio, _ = librosa.load(f, sr=target_sr)
            peak = np.max(np.abs(audio))
            if peak > 1e-6: audio = audio / peak
            self._concrete_noise_list.append(audio.astype(np.float32))
            
    # 【核心：覆盖原生的真实底噪抽取，强制相位锁死 0！】
    def _load_concrete_noise(self, length: int) -> np.ndarray:
        if not self._concrete_noise_list:
            return np.zeros(length, dtype=np.float32)
        noise = random.choice(self._concrete_noise_list)
        if len(noise) < length:
            noise = np.tile(noise, (length // len(noise)) + 2)
        # 👈 核心诊断点：强行返回 0 起点的数据！
        return noise[0 : length].copy()

def apply_training_lpf(audio_np, target_sr, cutoff_hz):
    """
    复用训练中 gpu_fft_lowpass 的数学逻辑，直接对 np 数组做截断
    以确保低通效果 100% 镜像
    """
    audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0) # [1, 1, T]
    fft_result = torch.fft.rfft(audio_tensor)
    freqs = torch.fft.rfftfreq(audio_tensor.size(-1), 1 / target_sr)
    
    mask = freqs <= cutoff_hz
    fft_result = fft_result * mask.unsqueeze(0).unsqueeze(0).to(fft_result.device)
    
    filtered_audio = torch.fft.irfft(fft_result, n=audio_tensor.size(-1))
    return filtered_audio.squeeze().numpy()


def generate_full_val_set():
    print("[1] 正在读取并对齐训练 JSON 配置...")
    with open("config/train_concrete.json", "r", encoding="utf-8") as f:
        hp = json.load(f)
        
    target_sr = hp["data"]["sampling_rate"]
    signal_length = hp["data"]["segment_length"]
    
    # 获取 Stage 4 的终极物理参数
    stage4_cfg = hp["concrete"]["curriculum_stages"][-1]
    physics_cfg = stage4_cfg["physics_config"]
    physics_cfg["enable"] = True
    physics_cfg["signal_length"] = signal_length
    val_cutoff_hz = int(np.mean(stage4_cfg["collator_config"]["cutoff_range"]))

    # 路径准备
    src_vocal_dir = "/root/autodl-tmp/Train_set_speech/wav48/test"
    ir_dir = hp["concrete"]["concrete_rir_dir"]
    
    out_base_dir = "/root/autodl-tmp/Val_set_concrete_final"
    out_degraded_dir = os.path.join(out_base_dir, "degraded")
    out_clean_dir = os.path.join(out_base_dir, "clean_target")
    os.makedirs(out_degraded_dir, exist_ok=True)
    os.makedirs(out_clean_dir, exist_ok=True)

    print(f"[2] 实例化原生环境混合器 (DiagnosticMixer) 与 物理管线 (ConcretePhysicsChain)...")
    mixer = DiagnosticMixer(hp, target_sr)
    ir_cache = {f: librosa.load(f, sr=target_sr)[0] for f in glob.glob(os.path.join(ir_dir, "*.wav"))}
    chain = ConcretePhysicsChain(config=physics_cfg, concrete_ir_cache=ir_cache, target_sr=target_sr)

    # 找验证语音
    vocal_files = glob.glob(os.path.join(src_vocal_dir, "**", "*.flac"), recursive=True) + \
                  glob.glob(os.path.join(src_vocal_dir, "**", "*.wav"), recursive=True)
    vocal_files.sort()
    
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    
    processed_degraded = []
    processed_clean = []

    print(f"[3] 准备生成 {len(vocal_files)} 条数据，严格调用原生管道...")
    for fpath in tqdm(vocal_files, desc="生成终极诊断验证集"):
        try:
            # 1. 读原始干净人声
            clean, _ = librosa.load(fpath, sr=target_sr)
            if len(clean) > signal_length:
                clean = clean[:signal_length]
            else:
                clean = np.pad(clean, (0, signal_length - len(clean)))
            
            # 2. 原生方法：混合底噪 (已内嵌 lock_phase=0 逻辑)
            # 调用 mixer 继承来的 _mix_noise 方法 (包含同步防爆音逻辑)
            mixed, clean_scaled = mixer._mix_noise(
                clean.copy(), clean.copy(), 
                snr_range=(10.0, 20.0), scale_range=(0.7, 0.9)
            )
            
            # 3. 原生方法：施加物理失真
            degraded = chain.apply(mixed, input_sr=target_sr, intensity=stage4_cfg["phase2_intensity"])
            
            # 4. 原生方法：施加 LPF 截断
            degraded = apply_training_lpf(degraded, target_sr, val_cutoff_hz)

            # 5. 保存
            rel_path = os.path.relpath(fpath, src_vocal_dir).replace('.flac.flac', '.wav').replace('.flac', '.wav')
            deg_save_path = os.path.join(out_degraded_dir, rel_path)
            clean_save_path = os.path.join(out_clean_dir, rel_path)
            
            os.makedirs(os.path.dirname(deg_save_path), exist_ok=True)
            os.makedirs(os.path.dirname(clean_save_path), exist_ok=True)
            
            sf.write(deg_save_path, degraded, target_sr, subtype='FLOAT')
            sf.write(clean_save_path, clean_scaled, target_sr, subtype='FLOAT')
            
            processed_degraded.append(deg_save_path)
            processed_clean.append(clean_save_path)
        except Exception as e:
            pass

    with open(os.path.join(out_base_dir, "val_degraded.lst"), 'w') as f: f.write('\n'.join(processed_degraded))
    with open(os.path.join(out_base_dir, "val_clean.lst"), 'w') as f: f.write('\n'.join(processed_clean))

    print("\n[DONE] ✅ 正规训练管道同款验证集已生成！(已锁定相位，专门用于排查鸿沟 Bug)")

if __name__ == "__main__":
    generate_full_val_set()