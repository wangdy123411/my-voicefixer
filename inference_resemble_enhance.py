# -*- coding: utf-8 -*-
import os
import glob
import torch
import torchaudio
import numpy as np
import librosa
import warnings

warnings.filterwarnings("ignore")

try:
    from resemble_enhance.enhancer.inference import enhance
except ImportError:
    raise ImportError("❌ 未检测到 resemble-enhance！请先执行: pip install resemble-enhance")

try:
    from speechmos import dnsmos
except ImportError:
    raise ImportError("❌ 未检测到 speechmos！请先执行: pip install speechmos")


def evaluate_audio_dnsmos(audio_path):
    """调用 DNSMOS 评估单条音频"""
    if not os.path.exists(audio_path):
        return None
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
        peak = np.max(np.abs(audio))
        if peak > 1.0: audio = audio / peak
        audio = np.clip(audio, -1.0, 1.0)
        
        raw_scores = dnsmos.run(audio, sr)
        return {
            'ovrl': raw_scores.get('ovrl_mos', raw_scores.get('ovrl', 0.0)),
            'sig':  raw_scores.get('sig_mos',  raw_scores.get('sig', 0.0)),
            'bak':  raw_scores.get('bak_mos',  raw_scores.get('bak', 0.0))
        }
    except Exception as e:
        print(f"❌ DNSMOS 评估失败: {e}")
        return None


def batch_process_resemble_enhance(input_dir, out_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(out_dir, exist_ok=True)
    
    wav_files = glob.glob(os.path.join(input_dir, "*.wav"))
    if not wav_files:
        print(f"❌ 在 {input_dir} 目录下没找到 .wav 文件！")
        return

    print(f"🚀 找到 {len(wav_files)} 个音频，开始启动 Resemble-Enhance 扩散抛光...")
    
    report_path = os.path.join(out_dir, "Resemble_Enhance_Report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Stage 2: Resemble-Enhance 终极抛光评估报告 ===\n")
        f.write(f"输入目录: {input_dir}\n")
        f.write(f"输出目录: {out_dir}\n")
        f.write("=" * 60 + "\n\n")

    for idx, input_wav in enumerate(wav_files, 1):
        basename = os.path.splitext(os.path.basename(input_wav))[0]
        print(f"\n[{idx}/{len(wav_files)}] 正在抛光: {basename}")
        
        out_wav_path = os.path.join(out_dir, f"{basename}_ResembleEnhanced.wav")

        # ==========================================
        # 核心：Resemble-Enhance 推理
        # ==========================================
        try:
            # 1. 加载音频 (返回 2D: [C, T])
            dwav, sr = torchaudio.load(input_wav)
            
            # 🚀【修复点 1】强制转为单声道，并“降维打击”成 1D 张量 [T]
            dwav = dwav.mean(dim=0) 
                
            # 2. 核心抛光运算 (输入必须是 1D)
            hwav, new_sr = enhance(dwav, sr, device, nfe=64, solver="midpoint", lambd=0.7, tau=0.5)
            
            # 3. 结果防爆并保存
            hwav = hwav.cpu()
            
            # 🚀【修复点 2】给抛光好的音频重新套上声道维度，变回 2D [1, T]，用来保存
            hwav = hwav.unsqueeze(0) 
            
            peak = torch.max(torch.abs(hwav))
            if peak > 0.99:
                hwav = hwav * (0.95 / peak)
                
            torchaudio.save(out_wav_path, hwav, new_sr)
            
        except Exception as e:
            print(f"❌ 抛光失败: {e}")
            continue

        # ==========================================
        # 对比评测：抛光前 vs 抛光后
        # ==========================================
        scores_in = evaluate_audio_dnsmos(input_wav)
        scores_out = evaluate_audio_dnsmos(out_wav_path)
        
        log_str = f"[{basename}]\n"
        if scores_in:
            log_str += f"  - [抛光前 Stage1] OVRL: {scores_in['ovrl']:.2f} | SIG: {scores_in['sig']:.2f} | BAK: {scores_in['bak']:.2f}\n"
        if scores_out:
            log_str += f"  - [抛光后 Stage2] OVRL: {scores_out['ovrl']:.2f} | SIG: {scores_out['sig']:.2f} | BAK: {scores_out['bak']:.2f}\n"
            
            if scores_in and scores_out['ovrl'] > scores_in['ovrl']:
                log_str += f"  👉 提升: +{scores_out['ovrl'] - scores_in['ovrl']:.2f} 分\n"
            elif scores_in:
                log_str += f"  📉 下降: {scores_out['ovrl'] - scores_in['ovrl']:.2f} 分 (可能是扩散模型过于激进，可调低 lambd)\n"

        print(log_str.strip())
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(log_str + "\n")

    print("\n" + "=" * 60)
    print(f"🎉 抛光完毕！去听听看是否有惊艳的变化吧！输出目录: {out_dir}")

if __name__ == "__main__":
    # ================== 核心配置区 ==================
    # 1. 你的第一阶段(Ours)输出音频文件夹
    # 建议填入你之前跑出来的 V25 或 V26 带有 "Ours" 的结果目录
    INPUT_DIR = "/root/autodl-tmp/results_batch/Epoch29_v24" 
    
    # 2. Resemble-Enhance 抛光后的结果输出文件夹
    OUT_DIR = "/root/autodl-tmp/results_batch/Stage2_Resemble_Enhance"
    # ==============================================
    
    batch_process_resemble_enhance(INPUT_DIR, OUT_DIR)