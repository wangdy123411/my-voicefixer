import sys
import os

# 1. 修复 ModuleNotFoundError: 自动定位到项目根目录并加入环境变量
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.append(root_dir)

from tools.file.path import find_and_build
from tools.file.io import write_list
from tools.file.wav import *

# 2. 修改为你本地的纯净人声绝对路径 (前面加 r 防止 \ 被转义)
ROOT = r"D:\Data\Train_set_speech\wav48"
DATA = os.path.join(root_dir, "dataIndex")

# 转换格式 (如果你的音频已经是 .wav，这步会自动跳过或处理)
convert_flac_to_wav(ROOT)

find_and_build("", ROOT)
find_and_build("", DATA)

SOFTLINKSAVEDIR = os.path.join(DATA, "vctk")
find_and_build(SOFTLINKSAVEDIR, "")

data = {
    "test":{
        "fname":[], "speech":[]
    },
    "train":{
        "fname":[], "speech":[]
    }
}

print("正在扫描测试集 (test) ...")
SubDir_test = os.path.join(ROOT, "test")
test = find_and_build(SOFTLINKSAVEDIR, "test")

if os.path.exists(SubDir_test):
    for each in os.listdir(SubDir_test):
        if(".DS_Store" in each or ".pkf" in each): continue
        speaker = os.path.join(SubDir_test, each)
        if not os.path.isdir(speaker): continue # 确保是文件夹
        for audio in os.listdir(speaker):
            if (".DS_Store" in audio or ".pkf" in audio): continue
            # 3. 【核心修改】将 Windows 的反斜杠 \ 强制替换为正斜杠 /
            audio_path = os.path.join(speaker, audio).replace("\\", "/")
            data['test']['speech'].append(audio_path)
else:
    print(f"⚠️ 警告: 未找到 {SubDir_test} 文件夹！")

print("正在扫描训练集 (train) ...")
SubDir_train = os.path.join(ROOT, "train")
train = find_and_build(SOFTLINKSAVEDIR, "train")

if os.path.exists(SubDir_train):
    for each in os.listdir(SubDir_train):
        if(".DS_Store" in each or ".pkf" in each): continue
        speaker = os.path.join(SubDir_train, each)
        if not os.path.isdir(speaker): continue # 确保是文件夹
        for audio in os.listdir(speaker):
            if (".DS_Store" in audio or ".pkf" in audio): continue
            # 3. 【核心修改】将 Windows 的反斜杠 \ 强制替换为正斜杠 /
            audio_path = os.path.join(speaker, audio).replace("\\", "/")
            data['train']['speech'].append(audio_path)
else:
    print(f"⚠️ 警告: 未找到 {SubDir_train} 文件夹！")


# 写入 .lst 文件
write_list(data['test']['speech'], os.path.join(test, "speech.lst"))
write_list(data['train']['speech'], os.path.join(train, "speech.lst"))

print("\n✅ 处理完成！")
print(f"📂 测试集 (Test) 找到并写入: {len(data['test']['speech'])} 个音频路径。")
print(f"📂 训练集 (Train) 找到并写入: {len(data['train']['speech'])} 个音频路径。")