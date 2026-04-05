import git
import sys
import os
import glob  # 新增：用于在 Windows 下安全地遍历文件


# 1. 自动获取当前脚本所在目录 (.../datasetParser)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. 往上退一级 (.../datasets)
parent_dir = os.path.dirname(current_dir)
# 3. 再往上退一级，找到项目根目录 (.../voicefixer_main-main)
root_dir = os.path.dirname(parent_dir)

# 4. 把根目录强行加入 Python 的环境变量中
sys.path.append(root_dir)

# ===== 在这之后再导入 tools 就绝对不会报错了 =====
from tools.file.path import find_and_build
from tools.file.wav import *

# (保留你原来的其他代码往下写...)
git_root = git.Repo("", search_parent_directories=True).git.rev_parse("--show-toplevel")
sys.path.append(git_root)
r = os.path.dirname(git_root) 

# 1. 修改为你本地的 Windows 绝对路径（前面加 r 防止 \ 转义）
ROOT = r"D:\Data\Train_set_noise\vd_noise"

# 如果你当前的工程名叫 my-voicefixer，建议把这里的 voicefixer_main 改成你实际的文件夹名
# 或者直接用 git_root： DATA = os.path.join(git_root, "dataIndex")
DATA = os.path.join(git_root, "dataIndex")

convert_flac_to_wav(ROOT)

find_and_build("", ROOT)
find_and_build("", DATA)

SOFTLINKSAVEDIR = os.path.join(DATA, "vd_noise")

find_and_build(SOFTLINKSAVEDIR, "")

SubDir = ROOT
train = SOFTLINKSAVEDIR
lst_out_path = os.path.join(train, "vd_noise.lst")

print(f"正在扫描文件夹: {SubDir} ...")

# 2. 【核心修改】将 Linux 的 find 命令替换为 Windows 兼容的 Python 原生遍历逻辑
wav_files = glob.glob(os.path.join(SubDir, "**", "*.wav"), recursive=True)

with open(lst_out_path, "w", encoding="utf-8") as f:
    for wav_path in wav_files:
        # 必须将 Windows 默认的反斜杠 \ 替换为正斜杠 /，否则后续模型训练读取会报错
        normalized_path = wav_path.replace("\\", "/")
        f.write(normalized_path + "\n")

print(f"✅ 处理完成！共找到 {len(wav_files)} 个音频，已保存至: {lst_out_path}")