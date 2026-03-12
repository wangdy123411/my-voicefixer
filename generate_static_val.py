# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import glob
import warnings

# 忽略不必要的警告
warnings.filterwarnings("ignore")

# 导入你现有的物理链路类
# 请确保脚本在项目根目录下运行，以便正确导入
from dataloaders.augmentation.concrete_physics import ConcretePhysicsChain

def generate_static_dataset():
    # ======================== 配置区 ========================
    # 1. 验证集源文件目录 (包含 p360 等子目录)
    src_vocal_dir = "/root/autodl-tmp/Train_set_speech/wav48/test"
    
    # 2. 静态受损音频保存目录
    out_base_dir = "/root/autodl-tmp/Val_set_concrete_static"
    out_vocal_dir = os.path.join(out_base_dir, "vocal_damaged")
    os.makedirs(out_vocal_dir, exist_ok=True)
    
    # 3. 混凝土 IR 目录
    ir_dir = "/root/autodl-tmp/CIRwav"
    
    # 4. 物理降级参数 (模拟 Stage 4 难度)
    physics_cfg = {
        "enable": True,
        "emi_snr_db": [10, 30],
        "emi_freqs": [50, 1000, 2048],
        "jfet_gain": 0.8,
        "jfet_harmonic": 0.1,
        "demod_lpf_cutoff": 4000,
        "signal_length": 132300 # 3秒 @ 44.1k
    }
    target_sr = 44100
    intensity = 0.7 
    
    # 支持的后缀名列表
    exts = ['*.flac.flac', '*.flac', '*.wav', '*.WAV']
    # ========================================================

    # 1. 加载 IR Cache
    print(f"[INIT] 正在加载 IR 脉冲响应...")
    ir_files = glob.glob(os.path.join(ir_dir, "*.wav"))
    if not ir_files:
        print(f"❌ 错误：在 {ir_dir} 未找到 IR 文件！")
        return
        
    ir_cache = {}
    for f in ir_files:
        wav, _ = librosa.load(f, sr=target_sr)
        ir_cache[f] = wav
    
    # 2. 初始化物理链
    chain = ConcretePhysicsChain(config=physics_cfg, concrete_ir_cache=ir_cache, target_sr=target_sr)

    # 3. 深度递归获取待处理文件列表
    vocal_files = []
    print(f"[SEARCH] 正在搜索音频文件...")
    for ext in exts:
        pattern = os.path.join(src_vocal_dir, "**", ext)
        found = glob.glob(pattern, recursive=True)
        vocal_files.extend(found)
        if found:
            print(f"  - 发现 {ext} 文件: {len(found)} 个")
    
    if len(vocal_files) == 0:
        print(f"❌ 错误：在 {src_vocal_dir} 下未找到任何匹配的音频文件！")
        return

    vocal_files.sort() # 保证顺序一致
    print(f"[START] 准备处理 {len(vocal_files)} 条验证集音频...")

    # 4. 循环处理并保存
    np.random.seed(42) 
    processed_list = []

    for fpath in tqdm(vocal_files, desc="生成进度"):
        try:
            # 读取音频
            wav, _ = librosa.load(fpath, sr=target_sr)
            
            # 统一长度裁剪/填充
            if len(wav) > physics_cfg["signal_length"]:
                wav = wav[:physics_cfg["signal_length"]]
            else:
                wav = np.pad(wav, (0, max(0, physics_cfg["signal_length"] - len(wav))))
            
            # 核心物理变换 (JFET + IR卷积 + AM调制 + LPF)
            damaged_wav = chain.apply(wav, input_sr=target_sr, intensity=intensity)
            
            # 构造保存路径：保持原有的文件夹结构，但后缀统一改为 .wav
            rel_path = os.path.relpath(fpath, src_vocal_dir)
            # 处理特殊的 .flac.flac 或 .flac 后缀
            clean_rel_path = rel_path.replace('.flac.flac', '.wav').replace('.flac', '.wav')
            
            save_path = os.path.join(out_vocal_dir, clean_rel_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 保存为标准 WAV
            sf.write(save_path, damaged_wav, target_sr)
            processed_list.append(save_path)
            
        except Exception as e:
            print(f"\n⚠️ 跳过文件 {fpath}，错误: {e}")

    # 5. 生成对应的索引 .lst 文件
    lst_path = os.path.join(out_base_dir, "vocal_damaged.lst")
    with open(lst_path, 'w') as f:
        for item in processed_list:
            f.write(item + "\n")
            
    print(f"\n[DONE] 静态验证集已成功生成！")
    print(f"总计处理: {len(processed_list)} 条音频")
    print(f"保存索引: {lst_path}")
    print(f"请在 config.json 中将验证集路径指向此 .lst 文件。")

if __name__ == "__main__":
    generate_static_dataset()