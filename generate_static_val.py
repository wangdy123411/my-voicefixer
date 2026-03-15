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
    # 👇 新增：真实物理底噪池路径
    src_concrete_noise_dir = "/root/autodl-tmp/Train_set_noise/Real_Noise_3s_Pool" 
    ir_dir = "/root/autodl-tmp/CIRwav"
    
    # 静态数据集输出路径
    out_base_dir = "/root/autodl-tmp/Val_set_concrete_final"
    out_degraded_dir = os.path.join(out_base_dir, "degraded")
    out_clean_dir = os.path.join(out_base_dir, "clean_target")
    os.makedirs(out_degraded_dir, exist_ok=True)
    os.makedirs(out_clean_dir, exist_ok=True)
    
    # 👇 更新：完全对齐 Stage 4 的最终打磨参数
    physics_cfg = {
        "enable": True, 
        "emi_snr_db": [10, 20], 
        "emi_freqs": [50, 1000, 1950, 2000, 2052], # 写入致命频点
        "jfet_gain": 0.4,           # 降低失真，防止吞声
        "jfet_harmonic": 0.05, 
        "demod_lpf_cutoff": 5000,
        "signal_length": 132300     # 3秒 @ 44.1k
    }
    target_sr = 44100
    concrete_noise_prob = 0.50      # 👈 50% 概率：一半验证通用，一半验证物理毒药
    val_cutoff_hz = 4000            # 验证集统一低通截断频率 (Stage4 范围是 3000-6000)
    # ========================================================

    print("[1] 正在加载 IR 脉冲...")
    ir_cache = {f: librosa.load(f, sr=target_sr)[0] for f in glob.glob(os.path.join(ir_dir, "*.wav"))}
    chain = ConcretePhysicsChain(config=physics_cfg, concrete_ir_cache=ir_cache, target_sr=target_sr)

    print("[2] 正在加载 通用底噪池...")
    noise_files = []
    for ext in ['*.flac.flac', '*.flac', '*.wav', '*.WAV']:
        noise_files.extend(glob.glob(os.path.join(src_noise_dir, "**", ext), recursive=True))
    if not noise_files:
        print("❌ 未找到通用噪声文件！")
        return
    # 抽取前 100 个长噪声拼接成通用池
    generic_noise_pool = np.concatenate([librosa.load(f, sr=target_sr)[0] for f in noise_files[:100]])

    print("[3] 正在加载 真实物理底噪池...")
    concrete_noise_files = glob.glob(os.path.join(src_concrete_noise_dir, "*.wav"))
    if not concrete_noise_files:
        print("⚠️ 未找到真实物理底噪！将 100% 退化使用通用底噪。")
        concrete_noise_pool = generic_noise_pool
        concrete_noise_prob = 0.0
    else:
        concrete_noise_pool = np.concatenate([librosa.load(f, sr=target_sr)[0] for f in concrete_noise_files])
        print(f"  ✅ 成功加载 {len(concrete_noise_files)} 个真实物理底噪文件，验证命中率设定为 {concrete_noise_prob:.0%}")

    print("[4] 正在搜索语音文件...")
    vocal_files = []
    for ext in ['*.flac.flac', '*.flac', '*.wav', '*.WAV']:
        vocal_files.extend(glob.glob(os.path.join(src_vocal_dir, "**", ext), recursive=True))
    vocal_files.sort()
    
    print(f"[5] 准备处理 {len(vocal_files)} 条音频，请稍候...")
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
                
            # 🎲 核心机制：50% 概率给验证集上物理毒药
            if random.random() < concrete_noise_prob:
                current_noise_pool = concrete_noise_pool
            else:
                current_noise_pool = generic_noise_pool

            # 物理管线三步走
            mixed, clean_scaled = mix_noise_fast(clean, current_noise_pool, snr_db=15.0, scale=0.8)
            degraded = chain.apply(mixed, input_sr=target_sr, intensity=0.7)
            degraded = apply_fft_lpf(degraded, sr=target_sr, cutoff=val_cutoff_hz)

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

    print(f"\n[DONE] 完美静态验证集已生成！(已融合真实硬件底噪指纹)")
    print(f"可以直接去启动 train_concrete.py 炼丹了！")

if __name__ == "__main__":
    generate_full_val_set()