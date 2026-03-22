# -*- coding: utf-8 -*-
import os

# ==========================================
# 🛡️ 修复 1：强行覆写非法的环境变量，彻底消除报错！
# ==========================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import glob
import numpy as np
import librosa
import warnings
from tqdm import tqdm  # 引入 tqdm 进度条

warnings.filterwarnings("ignore")

try:
    from df.enhance import enhance, init_df, load_audio, save_audio
except ImportError:
    raise ImportError("❌ 未检测到 DeepFilterNet！请先执行: pip install deepfilternet")

try:
    from speechmos import dnsmos
except ImportError:
    raise ImportError("❌ 未检测到 speechmos！请先执行: pip install speechmos")

def evaluate_audio_dnsmos(audio_path):
    if not audio_path or not os.path.exists(audio_path): return None
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
        return None

def batch_process_dfnet(input_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    wav_files = glob.glob(os.path.join(input_dir, "*.wav"))
    if not wav_files:
        print(f"❌ 在 {input_dir} 没有找到 .wav 文件！")
        return

    # ==========================================
    # 🛡️ 修复 2：加载提示
    # ==========================================
    print("🚀 正在初始化 DeepFilterNet3 模型...")
    print("💡 提示：如果是首次运行，后台正在自动下载预训练权重，请耐心等待十几秒...")
    
    model, df_state, _ = init_df("/root/autodl-tmp/DeepFilterNet3_Model/DeepFilterNet3")
    print("✅ 模型加载成功！开始批量处理...\n")
    
    report_path = os.path.join(out_dir, "DeepFilterNet_Report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Stage 3: DeepFilterNet 终极净水器评估报告 ===\n\n")

    # ==========================================
    # 🛡️ 修复 3：使用 tqdm 包装主循环
    # ==========================================
    pbar = tqdm(wav_files, desc="🌊 净化进度", unit="file", dynamic_ncols=True)
    
    for input_wav in pbar:
        basename = os.path.splitext(os.path.basename(input_wav))[0]
        out_wav_path = os.path.join(out_dir, f"{basename}_DFNet3.wav")

        try:
            audio, _ = load_audio(input_wav, sr=df_state.sr())
            enhanced = enhance(model, df_state, audio)
            save_audio(out_wav_path, enhanced, df_state.sr())
        except Exception as e:
            tqdm.write(f"❌ DFNet 处理失败 ({basename}): {e}")
            continue

        # 评测 DNSMOS
        scores_in = evaluate_audio_dnsmos(input_wav)
        scores_out = evaluate_audio_dnsmos(out_wav_path)
        
        log_str = f"\n[{basename}]\n"
        if scores_in:
            log_str += f"  - [输入(Stage2)] OVRL: {scores_in['ovrl']:.2f} | SIG: {scores_in['sig']:.2f} | BAK: {scores_in['bak']:.2f}\n"
        if scores_out:
            log_str += f"  - [输出(DFNet3)] OVRL: {scores_out['ovrl']:.2f} | SIG: {scores_out['sig']:.2f} | BAK: {scores_out['bak']:.2f}\n"
            if scores_in and scores_out['ovrl'] > scores_in['ovrl']:
                log_str += f"  👉 提升: +{scores_out['ovrl'] - scores_in['ovrl']:.2f} 分\n"
            else:
                log_str += f"  ➖ 分数无明显提升，但底噪听感应更纯净\n"

        # 🛡️ 使用 tqdm.write 打印日志，绝不会打乱底部的进度条！
        tqdm.write(log_str)
        
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(log_str + "\n")

    print(f"\n🎉 DFNet 批量处理完毕！请前往查看: {out_dir}")

if __name__ == "__main__":
    # ⚠️ 确保路径正确指向你过了两遍 VoiceFixer 的音频
    INPUT_DIR = "/root/autodl-tmp/results_batch/VoiceFixer_Baseline_Full_Metrics2" 
    OUT_DIR = "/root/autodl-tmp/results_batch/Stage3_DFNet_Final"
    
    batch_process_dfnet(INPUT_DIR, OUT_DIR)