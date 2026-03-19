# -*- coding: utf-8 -*-
import numpy as np
import torch
import soundfile as sf
import os
import glob
import matplotlib
matplotlib.use('Agg') # 防止在无界面的 Linux 上画图报错
import matplotlib.pyplot as plt
import librosa
import librosa.display
import warnings

warnings.filterwarnings("ignore")

from tools.file.wav import read_wave
from train_concrete import ConcreteVoiceFixer as Model
from tools.utils import get_hparams_from_file

try:
    from voicefixer import VoiceFixer as OfficialVF
except ImportError:
    print("⚠️ 警告: 未检测到官方 voicefixer 库，Baseline 对比将被跳过。")
    OfficialVF = None

def hparams_to_dict(obj):
    if isinstance(obj, dict):
        return {k: hparams_to_dict(v) for k, v in obj.items()}
    if hasattr(obj, 'keys') and hasattr(obj, '__getitem__'):
        return {k: hparams_to_dict(obj[k]) for k in obj.keys()}
    return obj

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
    """
    [新增指标] 计算高频能量占比 (High-Frequency Energy Ratio)
    用于盲测模型是否真的恢复了被混凝土吃掉的高频。值越大说明高频越丰富。
    """
    S = np.abs(librosa.stft(y, n_fft=2048))**2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    
    # 找到大于 cutoff_hz 的频点索引
    hf_idx = np.where(freqs >= cutoff_hz)[0]
    
    total_energy = np.sum(S) + 1e-10
    hf_energy = np.sum(S[hf_idx, :])
    
    return float(hf_energy / total_energy) * 100  # 返回百分比

def plot_mel_comparison(wav_paths, titles, out_img_path, weight_label):
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
        
    plt.suptitle(f"Concrete Wall Penetration Recovery (Weight: {weight_label})", fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_img_path, dpi=300, bbox_inches='tight')
    plt.close()

def batch_process_folder(input_dir, out_dir, config_path, ckpt_path, weight_label):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(out_dir, exist_ok=True)
    
    # 获取所有 wav 文件
    wav_files = glob.glob(os.path.join(input_dir, "*.wav"))
    if not wav_files:
        print(f"❌ 在 {input_dir} 目录下没有找到任何 .wav 文件！")
        return

    print(f"🔍 找到 {len(wav_files)} 个音频文件，准备加载模型...")

    # ==========================================
    # 1. 统一加载模型 (只加载一次，极大提升批量速度)
    # ==========================================
    hp = hparams_to_dict(get_hparams_from_file(config_path))
    model = Model(hp, channels=1, type_target="vocals")
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
    
    model.eval().to(device)
    for param in model.parameters(): param.data = param.data.to(device)
    for buf in model.buffers(): buf.data = buf.data.to(device)
    if hasattr(model, 'f_helper') and hasattr(model.f_helper, 'stft'):
        model.f_helper.stft.to(device)
        if hasattr(model.f_helper.stft, 'device'): model.f_helper.stft.device = device 
    if hasattr(model, 'mel'):
        if hasattr(model.mel, 'to'): model.mel.to(device)
        elif hasattr(model.mel, 'mel_basis'): model.mel.mel_basis = model.mel.mel_basis.to(device)

    official_vf = OfficialVF() if OfficialVF is not None else None

    # 打开一个日志文件记录所有指标
    report_path = os.path.join(out_dir, f"Evaluation_Report_{weight_label}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"=== 混凝土穿透语音修复评估报告 ===\n")
        f.write(f"权重版本: {weight_label}\n")
        f.write(f"模型路径: {ckpt_path}\n\n")

    # ==========================================
    # 2. 遍历处理每个文件
    # ==========================================
    for idx, input_wav in enumerate(wav_files, 1):
        basename = os.path.splitext(os.path.basename(input_wav))[0]
        print(f"\n[{idx}/{len(wav_files)}] 正在处理: {basename}")
        
        concrete_out_wav = os.path.join(out_dir, f"{basename}_Ours_{weight_label}.wav")
        official_out_wav = os.path.join(out_dir, f"{basename}_Baseline.wav") if official_vf else None
        img_out_path = os.path.join(out_dir, f"{basename}_Mel_{weight_label}.png")

        # --- 推理 Ours ---
        wav_10k = read_wave(input_wav, sample_rate=44100)
        with torch.no_grad():
            audio_np = np.asarray(wav_10k, dtype=np.float32)
            if audio_np.ndim == 2: audio_np = audio_np[:, 0] if audio_np.shape[0] > audio_np.shape[1] else audio_np[0, :]
            elif audio_np.ndim > 2: audio_np = audio_np.ravel()
                
            audio_tensor = torch.from_numpy(audio_np).float().reshape(1, 1, -1).to(device)
            sp, mel_noisy = model.pre(audio_tensor)
            out_model = model(mel_noisy)
            denoised_mel = 10 ** out_model['mel'].clamp(-10.0, 5.0)
            
            out_wav = model.vocoder(denoised_mel)
            if torch.max(torch.abs(out_wav)) > 1.0: out_wav /= torch.max(torch.abs(out_wav))
            out_wav = out_wav.squeeze().cpu().numpy()
            
        sf.write(concrete_out_wav, out_wav, 44100)

        # --- 推理 Baseline ---
        if official_vf:
            try:
                official_vf.restore(input=input_wav, output=official_out_wav, cuda=torch.cuda.is_available(), mode=0)
            except Exception as e:
                print(f"❌ 官方 VF 失败: {e}")
                official_out_wav = None

        # --- 计算指标 ---
        y_in, _ = librosa.load(input_wav, sr=44100)
        y_ours, _ = librosa.load(concrete_out_wav, sr=44100)
        
        lsd_ours = calculate_lsd(y_in, y_ours)
        hfe_in = calculate_hf_energy(y_in, cutoff_hz=3000)
        hfe_ours = calculate_hf_energy(y_ours, cutoff_hz=3000)
        
        log_str = f"[{basename}]\n"
        log_str += f"  - [LSD] Ours vs Input: {lsd_ours:.4f}\n"
        log_str += f"  - [高频能量占比(>3kHz)] Input: {hfe_in:.2f}% -> Ours: {hfe_ours:.2f}%\n"

        paths_to_plot = [input_wav]
        titles = [f"1. Input: {basename} (HFE: {hfe_in:.2f}%)"]

        if official_out_wav and os.path.exists(official_out_wav):
            y_base, _ = librosa.load(official_out_wav, sr=44100)
            lsd_base = calculate_lsd(y_in, y_base)
            hfe_base = calculate_hf_energy(y_base, cutoff_hz=3000)
            log_str += f"  - [LSD] Baseline vs Input: {lsd_base:.4f}\n"
            log_str += f"  - [高频能量占比(>3kHz)] Baseline: {hfe_base:.2f}%\n"
            
            paths_to_plot.append(official_out_wav)
            titles.append(f"2. Baseline: VoiceFixer (HFE: {hfe_base:.2f}%)")
            
        paths_to_plot.append(concrete_out_wav)
        titles.append(f"3. Ours: ({weight_label}) (HFE: {hfe_ours:.2f}%)")
        
        # 打印并写入日志
        print(log_str.strip())
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(log_str + "\n")

        # 画图
        plot_mel_comparison(paths_to_plot, titles, img_out_path, weight_label)

    print(f"\n🎉 批量处理完成！所有结果和评估报告已保存在: {out_dir}")

if __name__ == "__main__":
    # ================== 核心配置区 ==================
    # 1. 你存放实验采集 wav 文件的文件夹
    INPUT_DIR = "/root/autodl-tmp/Ground" 
    
    # 2. 结果输出文件夹
    OUT_DIR = "/root/autodl-tmp/results_batch"
    
    # 3. 为当前使用的权重起个名字（会显示在图表和文件名中）
    WEIGHT_LABEL = "Epoch20_V22" 
    
    # 4. 模型配置与权重路径
    CONFIG_FILE = "config/train_concrete.json"
    CKPT_FILE = "/root/autodl-tmp/myvoicefixer/logs/train_concrete/version_22/checkpoints/ultimate_stage3_epoch=20-step=5963-val_loss=1.1960.ckpt" # 替换为你最新的权重
    # ==============================================
    
    batch_process_folder(INPUT_DIR, OUT_DIR, CONFIG_FILE, CKPT_FILE, WEIGHT_LABEL)