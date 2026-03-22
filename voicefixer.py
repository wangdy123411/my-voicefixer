# -*- coding: utf-8 -*-
import numpy as np
import torch
import os
import glob
import matplotlib
matplotlib.use('Agg') # 防止在无界面的 Linux 上画图报错
import matplotlib.pyplot as plt
import librosa
import librosa.display
import warnings

warnings.filterwarnings("ignore")

try:
    from voicefixer import VoiceFixer as OfficialVF
except ImportError:
    raise ImportError("❌ 未检测到官方 voicefixer 库！请先执行: pip install voicefixer")

try:
    from speechmos import dnsmos
except ImportError:
    raise ImportError("❌ 未检测到 speechmos 库！请先执行: pip install speechmos")


def calculate_lsd(y_ref, y_deg, sr=44100, n_fft=2048, hop_length=512):
    """计算对数谱距离 (Log-Spectral Distance)"""
    min_len = min(len(y_ref), len(y_deg))
    y_ref = y_ref[:min_len]
    y_deg = y_deg[:min_len]
    S_ref = np.abs(librosa.stft(y_ref, n_fft=n_fft, hop_length=hop_length))**2
    S_deg = np.abs(librosa.stft(y_deg, n_fft=n_fft, hop_length=hop_length))**2
    log_S_ref = np.log10(S_ref + 1e-10)
    log_S_deg = np.log10(S_deg + 1e-10)
    lsd = np.mean(np.sqrt(np.mean((log_S_ref - log_S_deg)**2, axis=0)))
    return float(lsd)


def calculate_hf_energy(y, sr=44100, cutoff_hz=3000):
    """计算高频能量占比 (High-Frequency Energy Ratio)"""
    S = np.abs(librosa.stft(y, n_fft=2048))**2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    
    hf_idx = np.where(freqs >= cutoff_hz)[0]
    total_energy = np.sum(S) + 1e-10
    hf_energy = np.sum(S[hf_idx, :])
    
    return float(hf_energy / total_energy) * 100  


def evaluate_audio_dnsmos(audio_path):
    """调用 DNSMOS 评估单条音频（带防爆与兼容层）"""
    if not audio_path or not os.path.exists(audio_path):
        return None
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
        peak = np.max(np.abs(audio))
        if peak > 1.0:
            audio = audio / peak
        audio = np.clip(audio, -1.0, 1.0)
        
        raw_scores = dnsmos.run(audio, sr)
        result = {
            'ovrl': raw_scores.get('ovrl_mos', raw_scores.get('ovrl', 0.0)),
            'sig':  raw_scores.get('sig_mos',  raw_scores.get('sig', 0.0)),
            'bak':  raw_scores.get('bak_mos',  raw_scores.get('bak', 0.0))
        }
        return result
    except Exception as e:
        print(f"❌ 评估 {audio_path} DNSMOS 失败: {e}")
        return None


def plot_mel_comparison(wav_paths, titles, out_img_path):
    """画出原始音频与 VoiceFixer 修复后的 Mel 频谱对比图"""
    plt.figure(figsize=(14, 5 * len(wav_paths)))
    
    for i, (wav_p, title) in enumerate(zip(wav_paths, titles)):
        y, sr = librosa.load(wav_p, sr=44100)
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=2048, hop_length=441, n_mels=128, fmin=0, fmax=22050
        )
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        plt.subplot(len(wav_paths), 1, i + 1)
        librosa.display.specshow(
            S_dB, sr=sr, hop_length=441, x_axis='time', y_axis='mel', 
            fmin=0, fmax=22050, cmap='magma'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(title, fontsize=15, fontweight='bold')
        
    plt.suptitle("Baseline: VoiceFixer Recovery & Metrics", fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_img_path, dpi=300, bbox_inches='tight')
    plt.close()


def batch_process_voicefixer_only(input_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    wav_files = glob.glob(os.path.join(input_dir, "*.wav"))
    if not wav_files:
        print(f"❌ 在 {input_dir} 目录下没有找到任何 .wav 文件！")
        return

    print(f"🔍 找到 {len(wav_files)} 个音频文件，准备加载 VoiceFixer 模型与 DNSMOS...")

    official_vf = OfficialVF()
    use_cuda = torch.cuda.is_available()

    report_path = os.path.join(out_dir, "Evaluation_Report_Baseline_VoiceFixer.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"=== VoiceFixer 官方模型独立评估报告 (包含 DNSMOS) ===\n")
        f.write(f"输入目录: {input_dir}\n")
        f.write(f"输出目录: {out_dir}\n")
        f.write("=" * 60 + "\n\n")

    # ==========================================
    # 遍历处理每个文件
    # ==========================================
    for idx, input_wav in enumerate(wav_files, 1):
        basename = os.path.splitext(os.path.basename(input_wav))[0]
        print(f"\n[{idx}/{len(wav_files)}] 正在处理: {basename}")
        
        official_out_wav = os.path.join(out_dir, f"{basename}_VoiceFixer.wav")
        img_out_path = os.path.join(out_dir, f"{basename}_Mel_Comparison.png")

        # --- 1. 推理由官方 VF 完成 ---
        try:
            official_vf.restore(input=input_wav, output=official_out_wav, cuda=use_cuda, mode=0)
        except Exception as e:
            print(f"❌ 官方 VF 修复失败 ({basename}): {e}")
            continue

        # --- 2. 计算频域物理指标 (LSD & HFE) ---
        y_in, _ = librosa.load(input_wav, sr=44100)
        y_base, _ = librosa.load(official_out_wav, sr=44100)
        
        lsd_val = calculate_lsd(y_in, y_base)
        hfe_in = calculate_hf_energy(y_in, cutoff_hz=3000)
        hfe_base = calculate_hf_energy(y_base, cutoff_hz=3000)

        # --- 3. 计算听感指标 (DNSMOS) ---
        scores_in = evaluate_audio_dnsmos(input_wav)
        scores_base = evaluate_audio_dnsmos(official_out_wav)
        
        # --- 4. 汇总日志 ---
        log_str = f"[{basename}]\n"
        log_str += f"  - [物理指标] LSD (VoiceFixer vs Input): {lsd_val:.4f}\n"
        log_str += f"  - [物理指标] 高频能量(>3kHz): Input {hfe_in:.2f}% -> VF {hfe_base:.2f}%\n"
        
        ovrl_in_str = "N/A"
        ovrl_base_str = "N/A"

        if scores_in:
            ovrl_in_str = f"{scores_in['ovrl']:.2f}"
            log_str += f"  - [DNSMOS Input]      OVRL: {scores_in['ovrl']:.2f} | SIG: {scores_in['sig']:.2f} | BAK: {scores_in['bak']:.2f}\n"
        if scores_base:
            ovrl_base_str = f"{scores_base['ovrl']:.2f}"
            log_str += f"  - [DNSMOS VoiceFixer] OVRL: {scores_base['ovrl']:.2f} | SIG: {scores_base['sig']:.2f} | BAK: {scores_base['bak']:.2f}\n"

        print(log_str.strip())
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(log_str + "\n")

        # --- 5. 画图 (将 DNSMOS 融合进标题) ---
        paths_to_plot = [input_wav, official_out_wav]
        titles = [
            f"1. Input: {basename} (HFE: {hfe_in:.2f}%, OVRL: {ovrl_in_str})",
            f"2. VoiceFixer Baseline (HFE: {hfe_base:.2f}%, OVRL: {ovrl_base_str})"
        ]
        plot_mel_comparison(paths_to_plot, titles, img_out_path)

    print("\n" + "=" * 60)
    print(f"🎉 批量处理与全方位评估完成！所有数据已保存在: {out_dir}")

if __name__ == "__main__":
    # ================== 核心配置区 ==================
    # 1. 你的输入音频文件夹
    # 【进阶用法】：如果你想做“两阶段修复”，把这里的 INPUT_DIR 改成你 Stage 1 模型生成的音频文件夹！
    INPUT_DIR = "/root/autodl-tmp/results_batch/VoiceFixer_Baseline_Full_Metrics2" 
    
    # 2. 结果输出文件夹
    OUT_DIR = "/root/autodl-tmp/results_batch/VoiceFixer_Baseline_Full_Metrics3"
    # ==============================================
    
    batch_process_voicefixer_only(INPUT_DIR, OUT_DIR)