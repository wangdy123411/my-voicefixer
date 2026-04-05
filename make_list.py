import os
import glob

# 1. 你存放音频的文件夹路径 (修改为你的实际路径)
audio_folder = "C:/Users/Defa/Desktop/Data/Dataset/clean" 

# 2. 你想保存的 .lst 文件路径 (就填入 json 里的那个路径)
output_lst_file = "dataIndex/vctk/train/speech.lst"

# 确保输出目录存在
os.makedirs(os.path.dirname(output_lst_file), exist_ok=True)

# 搜索文件夹下所有的 wav 文件
wav_files = glob.glob(os.path.join(audio_folder, "**", "*.wav"), recursive=True)

# 写入 lst 文件
with open(output_lst_file, 'w', encoding='utf-8') as f:
    for wav_path in wav_files:
        # 统一使用正斜杠，并写入绝对路径
        normalized_path = wav_path.replace('\\', '/')
        f.write(normalized_path + '\n')

print(f"✅ 成功生成 lst 文件！共包含 {len(wav_files)} 个音频路径。")
print(f"保存在: {output_lst_file}")