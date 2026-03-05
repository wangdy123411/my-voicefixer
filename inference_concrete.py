import numpy as np
import torch
import soundfile as sf
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display

from tools.file.wav import read_wave
from train_concrete import ConcreteVoiceFixer as Model
from tools.utils import get_hparams_from_file

# 导入官方原始的 VoiceFixer (基于你环境里的 voicefixer 包)
from voicefixer import VoiceFixer as OfficialVF

def hparams_to_dict(obj):
    """递归将 HParams 对象转换为原生 Python 字典，防止 setdefault 报错"""
    if isinstance(obj, dict):
        return {k: hparams_to_dict(v) for k, v in obj.items()}
    if hasattr(obj, 'keys') and hasattr(obj, '__getitem__'):
        return {k: hparams_to_dict(obj[k]) for k in obj.keys()}
    return obj

def plot_mel_comparison(wav_paths, titles, out_img_path):
    """读取音频列表，计算 Mel 频谱并画出堆叠的对比图"""
    plt.figure(figsize=(12, 4 * len(wav_paths)))
    
    for i, (wav_p, title) in enumerate(zip(wav_paths, titles)):
        # 读取音频
        y, sr = librosa.load(wav_p, sr=44100)
        
        # 计算梅尔频谱
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128, fmin=0, fmax=22050
        )
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # 绘制频谱图
        plt.subplot(len(wav_paths), 1, i + 1)
        librosa.display.specshow(
            S_dB, sr=sr, hop_length=512, x_axis='time', y_axis='mel', 
            fmin=0, fmax=22050, cmap='magma'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(title, fontsize=14, fontweight='bold')
        
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
    print("\n---> [1/2] 正在运行你的 ConcreteVoiceFixer 推理...")
    
    hp_raw = get_hparams_from_file(config_path)
    hp = hparams_to_dict(hp_raw)
    
    model = Model(hp, channels=1, type_target="vocals")
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    
    # 【核心修复】：PyTorch Lightning 1.5 纯推理设备漂移强力补丁
    model.eval()
    model.to(device)
    
    # 1. 递归强制遍历所有层级子模块，全部推到 GPU
    for child in model.modules():
        child.to(device)
        
    # 2. 暴力遍历所有参数和缓冲，底层直接修改数据指针 (专治各种漏网之鱼)
    for param in model.parameters():
        param.data = param.data.to(device)
    for buf in model.buffers():
        buf.data = buf.data.to(device)

    # 3. 处理原作者未继承 nn.Module 的外挂特征算子
    if hasattr(model, 'f_helper') and hasattr(model.f_helper, 'stft'):
        model.f_helper.stft.to(device)
        if hasattr(model.f_helper.stft, 'device'):
            model.f_helper.stft.device = device 
            
    if hasattr(model, 'mel'):
        if hasattr(model.mel, 'to'):
            model.mel.to(device)
        elif hasattr(model.mel, 'mel_basis'):
            model.mel.mel_basis = model.mel.mel_basis.to(device)
    
    # 开始推理
    wav_10k = read_wave(input_wav, sample_rate=44100)
    with torch.no_grad():
        audio_np = np.asarray(wav_10k, dtype=np.float32)
        if audio_np.ndim == 2:
            audio_np = audio_np[:, 0] if audio_np.shape[0] > audio_np.shape[1] else audio_np[0, :]
        elif audio_np.ndim > 2:
            audio_np = audio_np.ravel()
            
        audio_tensor = torch.from_numpy(audio_np).float().reshape(1, 1, -1).to(device)

        # 提取频谱
        sp, mel_noisy = model.pre(audio_tensor)
        
        # UNet 修复
        out_model = model(mel_noisy)
        pred_log_mel = out_model['mel']
        
        # 【钳位保护防爆音】
        pred_log_mel_safe = pred_log_mel.clamp(-10.0, 5.0)
        denoised_mel = 10 ** pred_log_mel_safe
        
        # Vocoder 还原波形
        out_wav = model.vocoder(denoised_mel)
        if torch.max(torch.abs(out_wav)) > 1.0:
            out_wav = out_wav / torch.max(torch.abs(out_wav))
            
        out_wav = out_wav.squeeze().cpu().numpy()
        
    sf.write(concrete_out_wav, out_wav, 44100)
    
    # ==========================================
    # 2. 运行官方原始 VoiceFixer (Baseline)
    # ==========================================
    print("\n---> [2/2] 正在运行官方原始 VoiceFixer 推理...")
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

    # ==========================================
    # 3. 提取特征并画出三联对比图
    # ==========================================
    print("\n---> 正在生成 Mel 频谱对比图...")
    paths_to_plot = [input_wav, official_out_wav, concrete_out_wav]
    titles = [
        "1. Input: Concrete Degraded Audio (Truncated High-Freq & Noisy)",
        "2. Baseline: Original VoiceFixer Restored",
        "3. Ours: ConcreteVoiceFixer Restored"
    ]
    
    # 过滤掉因各种原因没生成成功的文件
    valid_paths, valid_titles = [], []
    for p, t in zip(paths_to_plot, titles):
        if os.path.exists(p):
            valid_paths.append(p)
            valid_titles.append(t)
            
    if len(valid_paths) > 0:
        plot_mel_comparison(valid_paths, valid_titles, img_out_path)
    else:
        print("❌ 找不到音频文件，画图失败。")

if __name__ == "__main__":
    # ================== 路径配置 ==================
    # 1. 输入的音频路径
    INPUT_WAV = "C:/Users/Defa/Desktop/Data/Data1/Intermediate_Decoded/Rtalk0.50_CH2_decoded.wav"
    
    # 2. 分别保存两个模型的输出
    CONCRETE_OUT_WAV = "C:/Users/Defa/Desktop/Data/Results/Restored/Rtalk0.50_CH2_Ours.wav"
    OFFICIAL_OUT_WAV = "C:/Users/Defa/Desktop/Data/Results/Restored/Rtalk0.50_CH2_Baseline.wav"
    
    # 3. 最终的对比图保存路径
    IMG_OUT_PATH = "C:/Users/Defa/Desktop/Data/Results/Restored/Comparison_Mel_Spectrogram.png"
    
    # 4. 模型配置与权重
    CONFIG_FILE = "config/train_concrete.json"
    CKPT_FILE = "logs/train_concrete/version_4/checkpoints/last.ckpt"
    # ==============================================
    
    infer_and_compare(INPUT_WAV, CONCRETE_OUT_WAV, OFFICIAL_OUT_WAV, CONFIG_FILE, CKPT_FILE, IMG_OUT_PATH)