# -*- coding: utf-8 -*-
import os
import glob
import random
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# 导入你的物理链路类
from dataloaders.augmentation.concrete_physics import ConcretePhysicsChain

def apply_fft_lpf(audio, sr=44100, cutoff=3000):
    """完美模拟训练时的 GPU 暴力频域截断"""
    fft_n = len(audio)
    S = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(fft_n, 1/sr)
    S[freqs > cutoff] = 0
    return np.fft.irfft(S, n=fft_n).astype(np.float32)

def mix_noise_fast(clean, noise_pool, snr_db=15.0, scale=0.8):
    """环境噪声混合"""
    req_len = len(clean)
    if len(noise_pool) > req_len:
        start = random.randint(0, len(noise_pool) - req_len)
        noise = noise_pool[start:start+req_len]
    else:
        noise = np.tile(noise_pool, (req_len // len(noise_pool)) + 2)[:req_len]
        
    clean_rms = np.sqrt(np.mean(clean ** 2) + 1e-10)
    noise_rms = np.sqrt(np.mean(noise ** 2) + 1e-10)
    
    target_noise_rms = clean_rms / (10 ** (snr_db / 20.0))
    noise_scaled = noise * (target_noise_rms / noise_rms)
    
    mixed = (clean + noise_scaled) * scale
    clean_scaled = clean * scale
    return mixed, clean_scaled

def generate_full_val_set():
    # ======================== 配置区 ========================
    src_vocal_dir = "/root/autodl-tmp/Train_set_speech/wav48/test"
    src_noise_dir = "/root/autodl-tmp/Train_set_noise/vd_noise"
    ir_dir = "/root/autodl-tmp/CIRwav"
    
    # 静态数据集输出路径
    out_base_dir = "/root/autodl-tmp/Val_set_concrete_final"
    out_degraded_dir = os.path.join(out_base_dir, "degraded")
    out_clean_dir = os.path.join(out_base_dir, "clean_target")
    os.makedirs(out_degraded_dir, exist_ok=True)
    os.makedirs(out_clean_dir, exist_ok=True)
    
    physics_cfg = {
        "enable": True, "emi_snr_db": [10, 30], "emi_freqs": [50, 1000, 2048],
        "jfet_gain": 0.8, "jfet_harmonic": 0.1, "demod_lpf_cutoff": 4000,
        "signal_length": 132300 # 3秒 @ 44.1k
    }
    target_sr = 44100
    # ========================================================

    print("[1] 正在加载 IR 脉冲与底噪池...")
    ir_cache = {f: librosa.load(f, sr=target_sr)[0] for f in glob.glob(os.path.join(ir_dir, "*.wav"))}
    chain = ConcretePhysicsChain(config=physics_cfg, concrete_ir_cache=ir_cache, target_sr=target_sr)

    # ✅ 替换为：
    noise_files = []
    for ext in ['*.flac.flac', '*.flac', '*.wav', '*.WAV']:
        noise_files.extend(glob.glob(os.path.join(src_noise_dir, "**", ext), recursive=True))
    if not noise_files:
        print("❌ 未找到噪声文件！请检查路径。")
        return
    # 为了速度，抽取前 100 个长噪声拼接成噪声池
    noise_pool = np.concatenate([librosa.load(f, sr=target_sr)[0] for f in noise_files[:100]])

    print("[2] 正在搜索语音文件...")
    vocal_files = []
    for ext in ['*.flac.flac', '*.flac', '*.wav', '*.WAV']:
        vocal_files.extend(glob.glob(os.path.join(src_vocal_dir, "**", ext), recursive=True))
    vocal_files.sort()
    
    print(f"[3] 准备处理 {len(vocal_files)} 条音频，请稍候...")
    np.random.seed(42)
    random.seed(42)
    
    processed_degraded = []
    processed_clean = []

    for fpath in tqdm(vocal_files, desc="生成终极静态测试集"):
        try:
            clean, _ = librosa.load(fpath, sr=target_sr)
            if len(clean) > physics_cfg["signal_length"]:
                clean = clean[:physics_cfg["signal_length"]]
            else:
                clean = np.pad(clean, (0, max(0, physics_cfg["signal_length"] - len(clean))))
                
            # 物理管线三步走
            mixed, clean_scaled = mix_noise_fast(clean, noise_pool, snr_db=15.0, scale=0.8)
            degraded = chain.apply(mixed, input_sr=target_sr, intensity=0.7)
            degraded = apply_fft_lpf(degraded, sr=target_sr, cutoff=3000)

            # 保存路径构造
            rel_path = os.path.relpath(fpath, src_vocal_dir).replace('.flac.flac', '.wav').replace('.flac', '.wav')
            deg_save_path = os.path.join(out_degraded_dir, rel_path)
            clean_save_path = os.path.join(out_clean_dir, rel_path)
            
            os.makedirs(os.path.dirname(deg_save_path), exist_ok=True)
            os.makedirs(os.path.dirname(clean_save_path), exist_ok=True)
            
            # 使用 FLOAT 格式保存，绝不丢失物理衰减特性
            sf.write(deg_save_path, degraded, target_sr, subtype='FLOAT')
            sf.write(clean_save_path, clean_scaled, target_sr, subtype='FLOAT')
            
            processed_degraded.append(deg_save_path)
            processed_clean.append(clean_save_path)
        except Exception as e:
            pass

    # 生成 LST 文件供 DataLoader 读取
    deg_lst_path = os.path.join(out_base_dir, "val_degraded.lst")
    clean_lst_path = os.path.join(out_base_dir, "val_clean.lst")
    
    with open(deg_lst_path, 'w') as f: f.write('\n'.join(processed_degraded))
    with open(clean_lst_path, 'w') as f: f.write('\n'.join(processed_clean))

    print(f"\n[DONE] 完美静态验证集已生成！")
    print(f"请将 config.json 中的 val_dataset 指向这两个 .lst 文件。")

if __name__ == "__main__":
    generate_full_val_set()