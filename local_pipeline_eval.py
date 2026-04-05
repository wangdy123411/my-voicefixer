# -*- coding: utf-8 -*-
import os
import glob
import torch
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib
matplotlib.use('Agg') # 防止批量处理时弹出大量图片窗口
import matplotlib.pyplot as plt
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

# ==========================================
# 🔧 1. 评估指标工具箱
# ==========================================
try:
    from speechmos import dnsmos
except Exception as e:
    print(f"\n❌ 致命错误: 无法加载 speechmos，DNSMOS 分数将被跳过！")
    print(f"👉 真实的报错原因是: {e}")
    print("💡 解决方案: 请务必在终端运行 `pip install onnxruntime`\n")
    dnsmos = None

def calculate_lsd(y_ref, y_deg, sr=44100, n_fft=2048, hop_length=512):
    min_len = min(len(y_ref), len(y_deg))
    S_ref = np.abs(librosa.stft(y_ref[:min_len], n_fft=n_fft, hop_length=hop_length))**2
    S_deg = np.abs(librosa.stft(y_deg[:min_len], n_fft=n_fft, hop_length=hop_length))**2
    log_S_ref = np.log10(S_ref + 1e-10)
    log_S_deg = np.log10(S_deg + 1e-10)
    return float(np.mean(np.sqrt(np.mean((log_S_ref - log_S_deg)**2, axis=0))))

def calculate_hf_energy(y, sr=44100, cutoff_hz=3000):
    S = np.abs(librosa.stft(y, n_fft=2048))**2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    hf_idx = np.where(freqs >= cutoff_hz)[0]
    return float(np.sum(S[hf_idx, :]) / (np.sum(S) + 1e-10)) * 100  

def evaluate_dnsmos(audio_path):
    if not dnsmos or not os.path.exists(audio_path): return None
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
        audio = np.clip(audio / max(np.max(np.abs(audio)), 1.0), -1.0, 1.0)
        raw_scores = dnsmos.run(audio, sr)
        return {
            'ovrl': raw_scores.get('ovrl_mos', raw_scores.get('ovrl', 0.0)),
            'sig':  raw_scores.get('sig_mos',  raw_scores.get('sig', 0.0)),
            'bak':  raw_scores.get('bak_mos',  raw_scores.get('bak', 0.0))
        }
    except Exception as e:
        print(f"⚠️ 评估 {os.path.basename(audio_path)} 时 DNSMOS 失败: {e}")
        return None


# ==========================================
# 🤖 2. 模型包装器 (懒加载)
# ==========================================
class ModelRegistry:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._vf_model = None
        self._df_model = None
        self._df_state = None
        self._myvf_model = None
        self._audiosr_model = None  

    def get_voicefixer(self):
        if self._vf_model is None:
            from voicefixer import VoiceFixer
            print("⏳ 正在加载官方 VoiceFixer...")
            self._vf_model = VoiceFixer()
        return self._vf_model

    def get_dfnet(self):
        if self._df_model is None:
            from df.enhance import init_df
            print("⏳ 正在加载 DeepFilterNet3 (本地权重)...")
            LOCAL_DF_PATH = r"C:\Users\Defa\Desktop\DeepFilterNet3" 
            self._df_model, self._df_state, _ = init_df(LOCAL_DF_PATH)
        return self._df_model, self._df_state

    def get_myvoicefixer(self, config_path, ckpt_path):
        if self._myvf_model is None:
            print("⏳ 正在加载 MyVoiceFixer (本地微调权重)...")
            from train_concrete import ConcreteVoiceFixer
            from tools.utils import get_hparams_from_file
            
            def hparams_to_dict(obj):
                if isinstance(obj, dict): return {k: hparams_to_dict(v) for k, v in obj.items()}
                if hasattr(obj, 'keys'): return {k: hparams_to_dict(obj[k]) for k in obj.keys()}
                return obj

            hp = hparams_to_dict(get_hparams_from_file(config_path))
            self._myvf_model = ConcreteVoiceFixer(hp, channels=1, type_target="vocals")
            ckpt = torch.load(ckpt_path, map_location='cpu')
            self._myvf_model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
            self._myvf_model.eval().to(self.device)
        return self._myvf_model

    def get_audiosr(self):
        if self._audiosr_model is None:
            print("⏳ 正在加载 AudioSR 大模型...")
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            from audiosr import build_model
            self._audiosr_model = build_model(model_name="basic", device=self.device.type)
        return self._audiosr_model
    
    def get_demucs(self):
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        print("⏳ 正在加载 HTDemucs 大模型...")
        # 加载目前最强的 htdemucs 模型
        model = get_model('htdemucs')
        model.cuda() if torch.cuda.is_available() else model.cpu()
        model.eval()
        return model, apply_model

    def get_resemble_enhance(self):
        from resemble_enhance.enhancer.inference import load_enhancer
        print("⏳ 正在加载 Resemble Enhance 大模型...")
        # 默认自动下载并加载模型
        enhancer = load_enhancer(None, self.device)
        return enhancer


# ==========================================
# 🎨 3. 动态可视化画图
# ==========================================
def plot_pipeline_mels(history, out_img_path):
    plt.figure(figsize=(15, 5 * len(history)))
    
    for i, step in enumerate(history):
        y, sr = librosa.load(step['path'], sr=44100)
        S = librosa.power_to_db(librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=2048, hop_length=441, n_mels=128, fmin=0, fmax=22050
        ), ref=np.max)
        
        plt.subplot(len(history), 1, i + 1)
        librosa.display.specshow(S, sr=sr, hop_length=441, x_axis='time', y_axis='mel', fmin=0, fmax=22050, cmap='magma')
        plt.colorbar(format='%+2.0f dB')
        
        title = f"{step['stage_id']} | HFE: {step['hfe']:.2f}%"
        if step['dnsmos']:
            title += f" | OVRL: {step['dnsmos']['ovrl']:.2f} (SIG: {step['dnsmos']['sig']:.2f}, BAK: {step['dnsmos']['bak']:.2f})"
        if 'lsd' in step:
            title += f" | LSD: {step['lsd']:.4f}"
            
        plt.title(title, fontsize=14, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(out_img_path, dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================
# ⚙️ 4. 核心流水线执行器 (加入全局统计)
# ==========================================
def run_pipeline(input_dir, out_dir, pipeline_config, my_vf_cfg=None, my_vf_ckpt=None):
    os.makedirs(out_dir, exist_ok=True)
    wav_files = glob.glob(os.path.join(input_dir, "*.wav"))
    
    if not wav_files:
        print(f"❌ 找不到输入音频：{input_dir}")
        return

    pipeline_name_str = " -> ".join(pipeline_config)
    print(f"\n🚀 开始执行流水线: [Input] -> {pipeline_name_str}")
    
    registry = ModelRegistry()
    report_path = os.path.join(out_dir, f"Pipeline_Report_{'-'.join(pipeline_config)}.txt")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"=== 流水线全指标评估报告 ===\n配置: Input -> {pipeline_name_str}\n\n")

    # 📊 全局统计存储字典
    global_stats = defaultdict(lambda: {'OVRL': [], 'SIG': [], 'BAK': [], 'HFE': [], 'LSD': []})

    for idx, input_wav in enumerate(wav_files, 1):
        basename = os.path.splitext(os.path.basename(input_wav))[0]
        print(f"\n[{idx}/{len(wav_files)}] 处理中: {basename}")
        
        y_orig, _ = librosa.load(input_wav, sr=44100)
        history = [{
            'stage_id': 'Step 0: ORIGINAL_INPUT',
            'name': 'Original Input', 
            'path': input_wav,
            'dnsmos': evaluate_dnsmos(input_wav),
            'hfe': calculate_hf_energy(y_orig)
        }]
        
        current_audio = input_wav

        for step_idx, stage_name in enumerate(pipeline_config, 1):
            stage_name = stage_name.lower()
            out_name = f"{basename}_step{step_idx}_{stage_name}.wav"
            out_path = os.path.join(out_dir, out_name)
            
            try:
                if stage_name == "voicefixer":
                    vf = registry.get_voicefixer()
                    vf.restore(input=current_audio, output=out_path, cuda=torch.cuda.is_available(), mode=0)
                    
                elif stage_name == "dfnet":
                    from df.enhance import load_audio, save_audio, enhance
                    model, state = registry.get_dfnet()
                    audio, _ = load_audio(current_audio, sr=state.sr())
                    enhanced = enhance(model, state, audio)
                    save_audio(out_path, enhanced, state.sr())
                    
                elif stage_name == "audiosr":
                    from audiosr import super_resolution
                    model = registry.get_audiosr()
                    y_in, sr_in = librosa.load(current_audio, sr=16000)
                    chunk_samples = int(5.12 * sr_in)
                    out_waveforms = []
                    
                    for i in range(0, len(y_in), chunk_samples):
                        chunk = y_in[i : i + chunk_samples]
                        original_chunk_len = len(chunk)
                        if original_chunk_len < 16000:  
                            chunk = np.pad(chunk, (0, 16000 - original_chunk_len), mode='constant')

                        temp_chunk_path = "temp_chunk_for_audiosr.wav"
                        sf.write(temp_chunk_path, chunk, sr_in)
                        
                        chunk_out = super_resolution(model, temp_chunk_path, seed=42, guidance_scale=3.5, ddim_steps=50)
                        target_out_len = original_chunk_len * (48000 // sr_in)
                        out_waveforms.append(np.squeeze(chunk_out)[:target_out_len])
                        
                    final_waveform = np.concatenate(out_waveforms)
                    if os.path.exists("temp_chunk_for_audiosr.wav"): os.remove("temp_chunk_for_audiosr.wav")
                    sf.write(out_path, final_waveform, 48000)
                
                elif stage_name == "demucs":
                    # HTDemucs 人声提取降噪
                    model, apply_model = registry.get_demucs()
                    wav, sr = librosa.load(current_audio, sr=44100, mono=False)
                    if wav.ndim == 1: wav = np.stack([wav, wav]) # Demucs需要双声道输入
                    wav_tensor = torch.tensor(wav).unsqueeze(0).to(next(model.parameters()).device)
                    
                    with torch.no_grad():
                        sources = apply_model(model, wav_tensor)[0]
                    
                    # 取出 'vocals' (人声) 轨道，索引通常为 3 (drums, bass, other, vocals)
                    vocals_idx = model.sources.index('vocals')
                    vocals_wav = sources[vocals_idx].mean(dim=0).cpu().numpy() # 转回单声道
                    sf.write(out_path, vocals_wav, 44100)

                elif stage_name == "resemble":
                    # Resemble Enhance 生成式增强
                    from resemble_enhance.enhancer.inference import enhance
                    enhancer = registry.get_resemble_enhance()
                    wav, sr = librosa.load(current_audio, sr=44100)
                    wav_tensor = torch.tensor(wav).to(registry.device)
                    
                    # 执行增强 (nfe=64是生成步数, solver='midpoint'是默认常微分求解器)
                    with torch.no_grad():
                        enhanced_wav, _ = enhance(wav_tensor, sr, enhancer.device, nfe=64, solver="midpoint", tau=0.5)
                    
                    sf.write(out_path, enhanced_wav.cpu().numpy(), 44100)

                elif stage_name == "myvoicefixer":
                    from tools.file.wav import read_wave
                    model = registry.get_myvoicefixer(my_vf_cfg, my_vf_ckpt)
                    wav_10k = read_wave(current_audio, sample_rate=44100)
                    with torch.no_grad():
                        audio_tensor = torch.from_numpy(np.asarray(wav_10k, dtype=np.float32)).float().reshape(1, 1, -1).to(registry.device)
                        sp, mel_noisy = model.pre(audio_tensor)
                        out_model = model(mel_noisy)
                        denoised_mel = 10 ** out_model['mel'].clamp(-10.0, 5.0)
                        out_wav = model.vocoder(denoised_mel)
                        if torch.max(torch.abs(out_wav)) > 1.0: out_wav /= torch.max(torch.abs(out_wav))
                    sf.write(out_path, out_wav.squeeze().cpu().numpy(), 44100)
                else:
                    print(f"⚠️ 未知步骤: {stage_name}")
                    continue
                
                y_curr, _ = librosa.load(out_path, sr=44100)
                step_data = {
                    'stage_id': f"Step {step_idx}: {stage_name.upper()}",
                    'name': stage_name.upper(),
                    'path': out_path,
                    'dnsmos': evaluate_dnsmos(out_path),
                    'hfe': calculate_hf_energy(y_curr),
                    'lsd': calculate_lsd(y_orig, y_curr)
                }
                history.append(step_data)
                current_audio = out_path
                
            except Exception as e:
                print(f"❌ 步骤 {stage_name} 失败: {e}")
                break

        # 📊 收集这一条音频的数据用于全局统计
        log_str = f"[{basename}]\n"
        for step in history:
            s_id = step['stage_id']
            log_str += f"  - [{s_id}] "
            
            if step['dnsmos']: 
                ovrl, sig, bak = step['dnsmos']['ovrl'], step['dnsmos']['sig'], step['dnsmos']['bak']
                log_str += f"OVRL: {ovrl:.2f} (SIG: {sig:.2f}, BAK: {bak:.2f}) | "
                global_stats[s_id]['OVRL'].append(ovrl)
                global_stats[s_id]['SIG'].append(sig)
                global_stats[s_id]['BAK'].append(bak)
                
            hfe = step['hfe']
            log_str += f"HFE: {hfe:.2f}%"
            global_stats[s_id]['HFE'].append(hfe)
            
            if 'lsd' in step: 
                lsd = step['lsd']
                log_str += f" | LSD: {lsd:.4f}"
                global_stats[s_id]['LSD'].append(lsd)
                
            log_str += "\n"
            
        print(log_str.strip())
        with open(report_path, "a", encoding="utf-8") as f: f.write(log_str + "\n")
            
        img_path = os.path.join(out_dir, f"{basename}_Pipeline_Visual.png")
        plot_pipeline_mels(history, img_path)

    # ==========================================
    # 📈 生成全局统计数据摘要 (均值与方差)
    # ==========================================
    summary_str = "\n" + "="*70 + "\n"
    summary_str += "📊 全局实验统计摘要 (Global Statistical Summary)\n"
    summary_str += "格式说明: 均值 ± 标准差 (方差)\n"
    summary_str += "="*70 + "\n"
    
    for stage_id, metrics in global_stats.items():
        summary_str += f"🔹 {stage_id}:\n"
        if metrics['OVRL']:
            m_o, s_o, v_o = np.mean(metrics['OVRL']), np.std(metrics['OVRL']), np.var(metrics['OVRL'])
            m_s, s_s, v_s = np.mean(metrics['SIG']), np.std(metrics['SIG']), np.var(metrics['SIG'])
            m_b, s_b, v_b = np.mean(metrics['BAK']), np.std(metrics['BAK']), np.var(metrics['BAK'])
            summary_str += f"   - OVRL: {m_o:.2f} ± {s_o:.2f} (Var: {v_o:.4f})\n"
            summary_str += f"   - SIG : {m_s:.2f} ± {s_s:.2f} (Var: {v_s:.4f})\n"
            summary_str += f"   - BAK : {m_b:.2f} ± {s_b:.2f} (Var: {v_b:.4f})\n"
        if metrics['HFE']:
            m_h, s_h, v_h = np.mean(metrics['HFE']), np.std(metrics['HFE']), np.var(metrics['HFE'])
            summary_str += f"   - HFE : {m_h:.2f}% ± {s_h:.2f}% (Var: {v_h:.4f})\n"
        if metrics['LSD']:
            m_l, s_l, v_l = np.mean(metrics['LSD']), np.std(metrics['LSD']), np.var(metrics['LSD'])
            summary_str += f"   - LSD : {m_l:.4f} ± {s_l:.4f} (Var: {v_l:.4f})\n"
        summary_str += "-"*70 + "\n"

    print(summary_str)
    with open(report_path, "a", encoding="utf-8") as f: f.write(summary_str)

    print(f"\n🎉 流水线任务完成！详细指标报告和统计摘要已保存在: {out_dir}")


if __name__ == "__main__":
    # ==============================================================
    # 🎯 本地核心配置区 
    # ==============================================================
    INPUT_DIR = r"D:\Data\realdata\Output_Audio"  
    OUT_DIR   = r"D:\Data\realdata\Evaluation_Results-audiosr"
    
    MY_VF_CONFIG = r"C:\Users\Defa\Desktop\my-voicefixer-main\config\train_concrete.json" 
    MY_VF_CKPT   = r"C:\Users\Defa\Desktop\Data\Data1\ultimate_stage3_epoch=29-step=8519.ckpt" 

    # ==============================================================
    # 🧩 魔法组合区：在这里切换你想跑的对照组！
    # ==============================================================
    
    # 🏆 终极版冠军配置 (用来验证你们的最强结果)
    #IPELINE = ["myvoicefixer", "voicefixer", "dfnet"]
    # 🧪 测试最新判别式大模型
    PIPELINE = ["audiosr"]
    
    # 🧪 测试最新生成式大模型
    #PIPELINE = ["resemble"]
    
    # 🏆 你们的冠军组合依然是：
    # PIPELINE = ["myvoicefixer", "voicefixer", "dfnet"]
    # 🧪 其他消融实验 (取消注释即可单独运行)：
    # PIPELINE = ["voicefixer", "voicefixer"]
    # PIPELINE = ["myvoicefixer"]
    # PIPELINE = ["myvoicefixer", "voicefixer"]
    
    run_pipeline(
        input_dir=INPUT_DIR, 
        out_dir=OUT_DIR, 
        pipeline_config=PIPELINE,
        my_vf_cfg=MY_VF_CONFIG,
        my_vf_ckpt=MY_VF_CKPT
    )