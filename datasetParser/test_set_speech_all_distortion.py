import sys
import os

# 1. 修复根目录定位，消除对 git 的依赖 (向上退两级找到 tools 所在的根目录)
current_dir = os.path.dirname(os.path.abspath(__file__)) # 当前在 datasetParser
parent_dir = os.path.dirname(current_dir)                # 退一级到 datasets
root_dir = os.path.dirname(parent_dir)                   # 再退一级到 voicefixer_main-main

sys.path.append(root_dir)

# 这样导入 tools 就绝对不会报错了
from tools.file.wav import *
from tools.file.path import find_and_build
from evaluation_proc.config import Config

# 规避 Windows 下没有装 progressbar 库导致的报错
try:
    from progressbar import *
except ImportError:
    pass

# 2. 指定你的本地测试集目录 (注意不要拼错你的文件夹名字)
ROOT = r"D:\Data\GSR_and_SSR_testsets\TestSets"

find_and_build("", ROOT)
# 如果你的测试集里都是 wav，这一步会直接跳过；如果有 flac，它会自动帮你转成 wav
convert_flac_to_wav(ROOT)

print("正在刷新测试集索引文件 (.lst)...")
Config.refresh_lists()

print("正在校验生成的列表与音频长度 (可能需要一点时间)...")
Config.checklst()

print("\n🎉 You have the following test set successfully loaded:")
for each in Config.get_all_testsets(): 
    print(each, end=", ")
print("\n")