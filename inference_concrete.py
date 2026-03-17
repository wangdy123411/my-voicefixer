# -*- coding: utf-8 -*-
import numpy as np
import torch
import soundfile as sf
import os
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

# 导入官方原始的 VoiceFixer (基于你环境里的 voicefixer 包)
try:
    from voicefixer import VoiceFixer as OfficialVF
except ImportError:
    print("⚠️ 警告: 未检测到官方 voicefixer 库，Baseline 对比将被跳过。")
    OfficialVF = None

def hparams_to_dict(obj):
    """递归将 HParams 对象转换为原生 Python 字典，防止 setdefault 报错"""
    if isinstance(obj, dict):
        return {k: hparams_to_dict(v) for k, v in obj.items()}
    if hasattr(obj, 'keys') and hasattr(obj, '__getitem__'):
        return {k: hparams_to_dict(obj[k]) for k in obj.keys()}
    return obj

def calculate_lsd(y_ref, y_deg, sr=44100, n_fft=2048, hop_length=512):
    """计算对数谱距离 (Log-Spectral Distance)，越小越好"""
    # 确保长度一致
    min_len = min(len(y_ref), len(y_deg))
    y_ref = y_ref[:min_len]
    y_deg = y_deg[:min_len]

    S_ref = np.abs(librosa.stft(y_ref, n_fft=n_fft, hop_length=hop_length))**2
    S_deg = np.abs(librosa.stft(y_deg, n_fft=n_fft, hop_length=hop_length))**2

    # 加上微小偏移量防止 log(0)
    log_S_ref = np.log10(S_ref + 1e-10)
    log_S_deg = np.log10(S_deg + 1e-10)

    lsd = np.mean(np.sqrt(np.mean((log_S_ref - log_S_deg)**2, axis=0)))
    return float(lsd)

def plot_mel_comparison(wav_paths, titles, out_img_path):
    """读取音频列表，计算 Mel 频谱并画出堆叠的对比图"""
    plt.figure(figsize=(14, 5 * len(wav_paths)))
    
    for i, (wav_p, title) in enumerate(zip(wav_paths, titles)):
        y, sr = librosa.load(wav_p, sr=44100)
        
        # 为了展示极其细微的高频恢复效果，我们使用 441 步长
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
        
    plt.tight_layout()
    plt.savefig(out_img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✨ 大功告成！Mel 频谱对比图已保存至: {out_img_path}")

def infer_and_compare(input_wav, concrete_out_wav, official_out_wav, config_path, ckpt_path, img_out_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.dirname(concrete_out_wav), exist_ok=True)
    
    # ==========================================
    # 1. 运行你的 ConcreteVoiceFixer (穿墙特化版)
    # ==========================================
    print("\n---> [1/3] 正在运行 ConcreteVoiceFixer (Ours) 推理...")
    
    hp_raw = get_hparams_from_file(config_path)
    hp = hparams_to_dict(hp_raw)
    
    model = Model(hp, channels=1, type_target="vocals")
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    
    model.eval()
    model.to(device)
    
    for child in model.modules(): child.to(device)
    for param in model.parameters(): param.data = param.data.to(device)
    for buf in model.buffers(): buf.data = buf.data.to(device)

    if hasattr(model, 'f_helper') and hasattr(model.f_helper, 'stft'):
        model.f_helper.stft.to(device)
        if hasattr(model.f_helper.stft, 'device'):
            model.f_helper.stft.device = device 
            
    if hasattr(model, 'mel'):
        if hasattr(model.mel, 'to'): model.mel.to(device)
        elif hasattr(model.mel, 'mel_basis'): model.mel.mel_basis = model.mel.mel_basis.to(device)
    
    wav_10k = read_wave(input_wav, sample_rate=44100)
    with torch.no_grad():
        audio_np = np.asarray(wav_10k, dtype=np.float32)
        if audio_np.ndim == 2:
            audio_np = audio_np[:, 0] if audio_np.shape[0] > audio_np.shape[1] else audio_np[0, :]
        elif audio_np.ndim > 2:
            audio_np = audio_np.ravel()
            
        audio_tensor = torch.from_numpy(audio_np).float().reshape(1, 1, -1).to(device)

        sp, mel_noisy = model.pre(audio_tensor)
        out_model = model(mel_noisy)
        pred_log_mel = out_model['mel']
        
        pred_log_mel_safe = pred_log_mel.clamp(-10.0, 5.0)
        denoised_mel = 10 ** pred_log_mel_safe
        
        out_wav = model.vocoder(denoised_mel)
        if torch.max(torch.abs(out_wav)) > 1.0:
            out_wav = out_wav / torch.max(torch.abs(out_wav))
            
        out_wav = out_wav.squeeze().cpu().numpy()
        
    sf.write(concrete_out_wav, out_wav, 44100)
    
    # ==========================================
    # 2. 运行官方原始 VoiceFixer (Baseline)
    # ==========================================
    if OfficialVF is not None:
        print("\n---> [2/3] 正在运行官方原始 VoiceFixer 推理...")
        try:
            official_vf = OfficialVF()
            official_vf.restore(
                input=input_wav, 
                output=official_out_wav, 
                cuda=torch.cuda.is_available(), 
                mode=0
            )
        except Exception as e:
            print(f"❌ 官方 VoiceFixer 运行失败: {e}")
            official_out_wav = None
    else:
        official_out_wav = None

    # ==========================================
    # 3. 计算评价指标与画图
    # ==========================================
    print("\n---> [3/3] 正在计算评价指标并生成可视化图表...")
    
    paths_to_plot = [input_wav]
    titles = ["1. Input: Degraded Audio (Truncated High-Freq & Noisy)"]
    
    if official_out_wav and os.path.exists(official_out_wav):
        paths_to_plot.append(official_out_wav)
        titles.append("2. Baseline: Original VoiceFixer Restored")
        
    paths_to_plot.append(concrete_out_wav)
    titles.append("3. Ours: ConcreteVoiceFixer Restored")
    
    plot_mel_comparison(paths_to_plot, titles, img_out_path)
    
    # === 计算盲测 LSD 指标 ===
    # 因为没有干净的 Ground Truth，我们计算“输入降级语音”与“修复语音”的谱距离。
    # 对于 Baseline 来说，面对这种未知降级，它通常会越修越差（引入伪影），导致 LSD 极大。
    # 你的模型应该能平稳地扩展高频，LSD 会处于一个合理区间。
    print("\n" + "="*40)
    print("📊 盲推断评价指标 (参考基准：输入降级语音)")
    print("="*40)
    
    y_in, _ = librosa.load(input_wav, sr=44100)
    y_ours, _ = librosa.load(concrete_out_wav, sr=44100)
    
    lsd_ours = calculate_lsd(y_in, y_ours)
    print(f"Ours (ConcreteVoiceFixer) 与输入的 LSD: {lsd_ours:.4f}")
    
    if official_out_wav and os.path.exists(official_out_wav):
        y_base, _ = librosa.load(official_out_wav, sr=44100)
        lsd_base = calculate_lsd(y_in, y_base)
        print(f"Baseline (Original VF) 与输入的 LSD: {lsd_base:.4f}")
    print("="*40)

if __name__ == "__main__":
    # ================== AutoDL 路径配置 ==================
    # 1. 输入的音频路径
    INPUT_WAV = "/root/autodl-tmp/Rtalk0.50_CH2_decoded.wav"
    
    # 2. 输出路径 (我们将结果存在 /root/autodl-tmp/results 目录下方便查看)
    OUT_DIR = "/root/autodl-tmp/results"
    os.makedirs(OUT_DIR, exist_ok=True)
    
    CONCRETE_OUT_WAV = os.path.join(OUT_DIR, "Rtalk0.50_CH2_Ours19.wav")
    OFFICIAL_OUT_WAV = os.path.join(OUT_DIR, "Rtalk0.50_CH2_Baseline19.wav")
    IMG_OUT_PATH = os.path.join(OUT_DIR, "Comparison_Mel_Spectrogram0.5019.png")
    
    # 3. 模型配置与权重
    CONFIG_FILE = "config/train_concrete.json"
    
    # ⚠️ 请确保这里的 version_X 对应你最新停掉的那次训练日志
    # 如果它是自动停止的，通常会有 last.ckpt 或者最好的 val_loss.ckpt
    CKPT_FILE = "/root/autodl-tmp/myvoicefixer/logs/train_concrete/version_19/checkpoints/stage3_epoch=103-step=29535-val_loss=0.5946.ckpt" 
    # ==============================================
    
    infer_and_compare(INPUT_WAV, CONCRETE_OUT_WAV, OFFICIAL_OUT_WAV, CONFIG_FILE, CKPT_FILE, IMG_OUT_PATH)