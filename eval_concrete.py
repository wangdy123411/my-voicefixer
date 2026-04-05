import os
import git
import sys
import torch

# 获取根目录并加入环境变量
git_root = git.Repo("", search_parent_directories=True).git.rev_parse("--show-toplevel")
sys.path.append(git_root)

from tools.file.wav import *
from tools.pytorch.pytorch_util import from_log, to_log
from evaluation_proc.config import Config
from evaluation_proc.eval import evaluation
from evaluation_proc.metrics import AudioMetrics
from tools.utils import *

# =====================================================================
# [关键修改 1] 导入我们最新的 Concrete 模型，而不是旧的 VoiceFixer
# =====================================================================
# 注意：这里假设你的 ConcreteVoiceFixer 写在 train_concrete.py 里
# 如果路径不对，请自行修改 import 路径
from train_concrete import ConcreteVoiceFixer as Model 

EPS = 1e-9

def pre(input_wav, device):
    """音频预处理：转频域"""
    input_wav = input_wav[None, None, ...]
    input_wav = torch.tensor(input_wav).to(device)
    # 调用模型内部的 f_helper 进行 STFT
    sp, _, _ = model.f_helper.wav_to_spectrogram_phase(input_wav)
    mel_orig = model.mel(sp.permute(0,1,3,2)).permute(0,1,3,2)
    return sp, mel_orig

# 全局变量
model = None
am = None
hp = None

def refresh_model(ckpt):
    """重新加载模型权重"""
    global model, am
    
    # =====================================================================
    # [关键修改 2] 适配 Concrete 模型的参数，并开启 strict=False
    # 因为我们刚才修改了网络架构（加入了 DilatedTimeBottleneck），
    # 如果你跑旧的 Checkpoint，必须开启 strict=False 防止维度报错！
    # 并且把 channels 改成了 1 (单声道)，匹配我们在 train_concrete 里的设定。
    # =====================================================================
    print(f"[EVAL] 正在加载模型权重: {ckpt}")
    model = Model.load_from_checkpoint(
        ckpt, 
        strict=False,   # 极其重要！兼容新旧架构的过渡
        hp=hp, 
        channels=1,     # 修改为 1
        type_target="vocals"
    )
    am = AudioMetrics(rate=44100)
    
    # 切换到评估模式，关闭 Dropout 和 BatchNorm 的更新
    model.eval()

def handler(input_path, output_path, target_path, ckpt, device, needrefresh=False, meta={}):
    """单条音频的处理与评估管线"""
    if needrefresh: 
        refresh_model(ckpt)
        
    global model
    model = model.to(device)
    metrics = {}
    
    with torch.no_grad(): # 评估阶段全程关闭梯度，省显存提速
        # 读取待处理的带噪/混响音频
        wav_noisy = load_wav(input_path, sample_rate=44100)
        
        if target_path is not None:
            target_wav = load_wav(target_path, sample_rate=44100)
            
        res = []
        seg_length = 44100 * 60 # 60秒切片处理
        break_point = seg_length
        
        while break_point < wav_noisy.shape[0] + seg_length:
            # 截取音频切片
            segment = wav_noisy[break_point - seg_length : break_point]
            _, mel_noisy = pre(segment, device)
            
            # =====================================================================
            # [关键修改 3] 确保前向传播逻辑匹配 ConcreteVoiceFixer
            # 如果你的 forward 直接返回的是 mel 张量，请把 out_model['mel'] 改成 out_model
            # 这里按照通常返回字典的形式保留
            # =====================================================================
            out_model = model(mel_noisy)
            
            # 取出预测的 log-mel 并转换回线性振幅
            gen_log_mel = out_model['mel'] if isinstance(out_model, dict) else out_model
            denoised_mel = from_log(gen_log_mel)
            
            if meta.get("unify_energy", False):
                denoised_mel, mel_noisy = amp_to_original_f(mel_sp_est=denoised_mel, mel_sp_target=mel_noisy)
                
            # 计算客观指标
            if target_path is not None:
                target_segment = target_wav[break_point - seg_length : break_point]
                _, target_mel = pre(target_segment, device)
                
                # 计算各种频域指标
                metrics = {
                    "mel-lsd": float(am.lsd(denoised_mel, target_mel)),
                    "mel-sispec": float(am.sispec(gen_log_mel, to_log(target_mel))), 
                    "mel-non-log-sispec": float(am.sispec(from_log(gen_log_mel), target_mel)),
                    "mel-ssim": float(am.ssim(denoised_mel, target_mel)),
                }

            # 声码器重建时域波形
            # 声码器重建时域波形
            out_wav = model.vocoder(denoised_mel)
            
            # 能量归一化保护
            if torch.max(torch.abs(out_wav)) > 1.0:
                out_wav = out_wav / torch.max(torch.abs(out_wav))
                print(f"[Warning] 输出音频能量越界，已自动截断: {input_path}")
                
            # 帧对齐补偿
            out_wav, _ = trim_center(out_wav, segment)
            res.append(out_wav)
            
            # 👇👇👇 [新增代码] 在生成波形后立刻计算 PESQ 👇👇👇
            if target_path is not None:
                # 把张量转成 numpy 给 PESQ 算
                est_np = tensor2numpy(out_wav[0, ...])
                # 注意：此时 target_segment 是 44100Hz 的 numpy 数组
                pesq_score = am.calculate_pesq(est_np, target_segment)
                metrics["pesq"] = pesq_score
            # 👆👆👆 ===================================== 👆👆👆
            
            break_point += seg_length
            
        # 拼接切片并保存波形
        out_final = torch.cat(res, -1)
        save_wave(tensor2numpy(out_final[0, ...]), fname=output_path, sample_rate=44100)
        
    return metrics

if __name__ == '__main__':
    from argparse import ArgumentParser
    from tools.utils import get_hparams_from_file
    
    parser = ArgumentParser()
    parser.add_argument("--config", default="", help="训练配置文件路径 (如 config/train_concrete.json)")
    parser.add_argument("--ckpt", default="", help="要评估的权重路径 (.ckpt 文件)")
    parser.add_argument("--limit_numbers", default=None, help="限制评估条数，用于快速测试")
    parser.add_argument("--description", default="concrete_eval", help="评估任务描述")
    parser.add_argument("--testset", default="base", help="测试集名称")
    args = parser.parse_args()

    # 读取超参数
    hp = get_hparams_from_file(args.config)
    
    print("=" * 60)
    print(f"🚀 开始跑分评估：{args.ckpt}")
    print("=" * 60)

    # 启动评估管线
    evaluation(
        output_path=Config.EVAL_RESULT,
        handler=handler,
        ckpt=args.ckpt,
        description=os.path.splitext(os.path.basename(args.config))[0] + "_" + args.description.strip(),
        limit_testset_to=Config.get_testsets(args.testset),
        limit_phrase_number=int(args.limit_numbers) if args.limit_numbers is not None else None
    )