# -*- coding: utf-8 -*-
import os
import glob
import librosa
import soundfile as sf
from speechmos import dnsmos

import numpy as np

def evaluate_audio_dnsmos(audio_path):
    """
    调用 DNSMOS 评估单条音频（带防爆与兼容层）。
    """
    if not os.path.exists(audio_path):
        return None
    
    try:
        # 1. 读取并重采样为 16kHz
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # 2. 🛡️ [修复1] 强制波形归一化并钳制，绝对保证在 [-1, 1] 之间！
        peak = np.max(np.abs(audio))
        if peak > 1.0:
            audio = audio / peak
        audio = np.clip(audio, -1.0, 1.0)  # 防微小浮点溢出
        
        # 3. 运行评估
        raw_scores = dnsmos.run(audio, sr)
        
        # 4. 🛡️ [修复2] 鲁棒地提取分数，兼容不同的键名版本
        result = {
            'ovrl': raw_scores.get('ovrl_mos', raw_scores.get('ovrl', 0.0)),
            'sig':  raw_scores.get('sig_mos',  raw_scores.get('sig', 0.0)),
            'bak':  raw_scores.get('bak_mos',  raw_scores.get('bak', 0.0))
        }
        return result
        
    except Exception as e:
        print(f"❌ 评估 {audio_path} 失败: {e}")
        return None

def batch_evaluate_dnsmos(input_dir, results_dir, weight_label):
    print("=" * 60)
    print("🎙️ DNSMOS 盲测音质评估启动 (满分 5.0，越高越好)")
    print("=" * 60)
    print("指标说明:")
    print(" - [SIG] Signal Quality: 人声质量 (是否有吞音、机械音失真)")
    print(" - [BAK] Background Quality: 底噪抑制质量 (环境噪音是否被抹除)")
    print(" - [OVRL] Overall Quality: 综合听感质量")
    print("=" * 60)

    # 找到所有的输入原始降级音频
    input_wavs = glob.glob(os.path.join(input_dir, "*.wav"))
    if not input_wavs:
        print(f"未在 {input_dir} 找到输入音频！")
        return

    # 准备日志文件
    report_path = os.path.join(results_dir, f"DNSMOS_Report_{weight_label}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"=== DNSMOS 盲测评估报告 (权重: {weight_label}) ===\n\n")

    for input_wav in input_wavs:
        basename = os.path.splitext(os.path.basename(input_wav))[0]
        
        # 构建对比文件的路径 (根据上一轮脚本的命名规则)
        ours_wav = os.path.join(results_dir, f"{basename}_Ours_{weight_label}.wav")
        baseline_wav = os.path.join(results_dir, f"{basename}_Baseline.wav")

        print(f"\n🎧 正在评估样本: {basename}")
        
        # 1. 评估原始输入
        scores_in = evaluate_audio_dnsmos(input_wav)
        # 2. 评估 Ours
        scores_ours = evaluate_audio_dnsmos(ours_wav)
        # 3. 评估 Baseline
        scores_base = evaluate_audio_dnsmos(baseline_wav)

        # 格式化输出字符串
        log_str = f"[{basename}]\n"
        
        if scores_in:
            log_str += f"  [Input]    OVRL: {scores_in['ovrl']:.2f} | SIG(人声): {scores_in['sig']:.2f} | BAK(降噪): {scores_in['bak']:.2f}\n"
        
        if scores_base:
            log_str += f"  [Baseline] OVRL: {scores_base['ovrl']:.2f} | SIG(人声): {scores_base['sig']:.2f} | BAK(降噪): {scores_base['bak']:.2f}\n"
        elif os.path.exists(baseline_wav):
            log_str += f"  [Baseline] 评估失败\n"

        if scores_ours:
            log_str += f"  [Ours]     OVRL: {scores_ours['ovrl']:.2f} | SIG(人声): {scores_ours['sig']:.2f} | BAK(降噪): {scores_ours['bak']:.2f}\n"
            
            # 简单对比逻辑
            if scores_in and scores_ours['ovrl'] > scores_in['ovrl']:
                log_str += f"  👉 结论: 相比输入，综合音质提升了 +{scores_ours['ovrl'] - scores_in['ovrl']:.2f} 分\n"
        elif os.path.exists(ours_wav):
             log_str += f"  [Ours]     评估失败\n"

        print(log_str.strip())
        
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(log_str + "\n")

    print("\n" + "=" * 60)
    print(f"🎉 DNSMOS 评估完成！详细报告已保存至: {report_path}")

if __name__ == "__main__":
    # ================== 配置区 ==================
    # 1. 你存放实验采集 wav 文件的文件夹 (未处理的源文件)
    INPUT_DIR = "/root/autodl-tmp/Ground" 
    
    # 2. 上一个脚本输出修复后音频的文件夹
    RESULTS_DIR = "/root/autodl-tmp/results_batch"
    
    # 3. 你上一轮跑批量脚本时设置的 WEIGHT_LABEL，必须保持一致才能找到对应的文件！
    WEIGHT_LABEL = "Epoch20_V22" 
    # ============================================
    
    batch_evaluate_dnsmos(INPUT_DIR, RESULTS_DIR, WEIGHT_LABEL)