import numpy as np
import matplotlib.pyplot as plt
from math import pi
from matplotlib.patches import Rectangle
import matplotlib.patheffects as patheffects

# ==========================================
# 📊 0. 数据录入 
# ==========================================
models = ['Original', 'AudioSR', 'Demucs', 'DFNet', 'VoiceFixer', 'MyVoiceFixer', 'Combined Pipeline']

# DNSMOS 综合得分 (OVRL)
ovrl_mean = [1.71, 1.44, 2.74, 3.09, 3.20, 3.17, 3.28]
ovrl_std = [0.25, 0.31, 0.26, 0.13, 0.08, 0.15, 0.14]

# DNSMOS 信号质量 (SIG) 和 背景抑制 (BAK)
sig_mean = [2.76, 2.01, 3.41, 3.37, 3.50, 3.47, 3.51]
bak_mean = [1.59, 1.38, 3.24, 4.01, 4.01, 4.00, 4.12]

# 全局学术绘图风格设置
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial'] # 国际学术界通用无衬线字体
plt.rcParams['axes.linewidth'] = 1.2

# ==========================================
# 📈 图表 1: 画中画局部放大柱状图 (纯净版)
# ==========================================
bar_colors = ['#ecf0f1', '#bdc3c7', '#a9dfbf', '#aed6f1', '#85929e', '#d2b4de', '#f5b7b1']
edge_colors = ['#bdc3c7', '#7f8c8d', '#27ae60', '#2980b9', '#2c3e50', '#8e44ad', '#c0392b']

fig1, ax1 = plt.subplots(figsize=(12, 6.5))
x_pos = np.arange(len(models))

# 主图
bars = ax1.bar(x_pos, ovrl_mean, yerr=ovrl_std, capsize=4, color=bar_colors, edgecolor=edge_colors, linewidth=1.5, width=0.6, zorder=3)
ax1.set_ylim(1.0, 4.8)
ax1.set_ylabel('DNSMOS OVRL Score', fontsize=14, fontweight='bold')
ax1.set_title('Overall Perceptual Quality (with Highlighted SOTA Region)', fontsize=16, fontweight='bold', pad=15)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(models, rotation=20, ha='right', fontsize=12, fontweight='bold')
ax1.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

# 分数标注
for i, bar in enumerate(bars):
    yval = bar.get_height()
    font_weight = 'bold' if i >= 4 else 'normal'
    y_text = yval + ovrl_std[i] + 0.05
    ax1.text(bar.get_x() + bar.get_width()/2.0, y_text, f'{yval:.2f}', ha='center', va='bottom', fontsize=10, fontweight=font_weight, color='black', zorder=5)

# 画中画
axins = ax1.inset_axes([0.02, 0.55, 0.48, 0.40])
inset_indices = [3, 4, 5, 6]
bars_in = axins.bar(inset_indices, [ovrl_mean[i] for i in inset_indices], 
          yerr=[ovrl_std[i] for i in inset_indices], capsize=3, 
          color=[bar_colors[i] for i in inset_indices], 
          edgecolor=[edge_colors[i] for i in inset_indices], linewidth=1.2, width=0.55)
          
axins.set_xlim(min(inset_indices)-0.5, max(inset_indices)+0.5)
axins.set_xticks(inset_indices)
axins.set_xticklabels([models[i] for i in inset_indices], rotation=15, ha='right', fontsize=10, fontweight='bold')
axins.set_ylim(3.05, 3.48) # 极端放大分值
axins.grid(axis='y', linestyle='--', alpha=0.5)
axins.set_title("Zoomed View: Top 4 Models", fontsize=12, fontweight='bold', pad=10, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

for i in inset_indices:
    val = ovrl_mean[i]
    font_weight = 'bold' if i == 6 else 'bold'
    axins.text(i, val - 0.035, f'{val:.2f}', ha='center', va='top', fontsize=12, fontweight=font_weight, color='white', zorder=5, path_effects=[patheffects.withStroke(linewidth=2.5, foreground='black')])

# 纯手工作图框线和单指引线
rect = Rectangle((2.5, 3.0), 4, 0.48, fill=False, edgecolor='gray', linestyle='--', linewidth=1.5, zorder=4)
ax1.add_patch(rect)
ax1.annotate('', xy=(2.5, 3.48), xycoords='data', xytext=(max(inset_indices)+0.5, 3.48), textcoords=axins.transData, arrowprops=dict(arrowstyle="-", color="gray", linestyle="--", linewidth=1.5))

fig1.tight_layout()
fig1.savefig('paper_bar_chart_final.pdf', dpi=300, bbox_inches='tight')
plt.close(fig1)
print("✅ 局部放大柱状图已生成：paper_bar_chart_final.pdf")


# ==========================================
# 🕸️ 图表 2: 终极学术雷达图 (极致截断版: 2.2)
# ==========================================
categories = ['OVRL', 'SIG', 'BAK']
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

fig2, ax2 = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
ax2.set_theta_offset(pi / 2)
ax2.set_theta_direction(-1)

# 🚀 极距空间折叠魔法：圆心强制从 2.2 扩展
RADAR_MIN = 2.2 
RADAR_MAX = 4.15
ax2.set_ylim(RADAR_MIN, RADAR_MAX)
ax2.set_rorigin(RADAR_MIN) 

# 设置可视刻度（优化网格线，更柔和）
plt.yticks([2.5, 3.0, 3.5, 4.0], ["2.5", "3.0", "3.5", "4.0"], color="#888888", size=10)
plt.xticks(angles[:-1], categories, size=14, fontweight='bold', color='black')
ax2.grid(color='#E5E5E5', linestyle='--', linewidth=1) # 调整内圈蜘蛛网为高级浅灰色虚线

colors = ['#bdc3c7', '#95a5a6', '#2ecc71', '#3498db', '#34495e', '#8e44ad', '#e74c3c']
linestyles = [':', ':', '--', '--', '-', '-', '-']
linewidths = [1.5, 1.5, 2.0, 2.0, 2.5, 2.5, 3.5] 
markers = ['', '', '', '', 's', '^', 'o']
markersizes = [0, 0, 0, 0, 6, 6, 9]
zorders = [1, 1, 2, 2, 3, 4, 5]

for i in range(len(models)):
    # 数据截流：防止 2.2 以下分数引起计算崩溃，全部归于圆点
    values = [max(ovrl_mean[i], RADAR_MIN), max(sig_mean[i], RADAR_MIN), max(bak_mean[i], RADAR_MIN)]
    values += values[:1]
    
    alpha_val = 0.85 if i < 2 else 0.95 
    
    ax2.plot(angles, values, linewidth=linewidths[i], linestyle=linestyles[i], 
            label=models[i], color=colors[i], marker=markers[i], 
            markersize=markersizes[i], zorder=zorders[i], alpha=alpha_val)
            
    if models[i] == 'Combined Pipeline':
        ax2.fill(angles, values, color=colors[i], alpha=0.15, zorder=0)

plt.title('DNSMOS Multi-dimensional Analysis\n(Ultra-Zoomed inner radius at 2.2)', size=16, fontweight='bold', y=1.1)
plt.legend(loc='center left', bbox_to_anchor=(1.15, 0.5), fontsize=11, frameon=True, edgecolor='#cccccc', shadow=False)

fig2.tight_layout()
fig2.savefig('paper_radar_extreme_zoom_2.2.pdf', dpi=300, bbox_inches='tight')
plt.close(fig2)
print("✅ 终极 2.2 雷达图已生成：paper_radar_extreme_zoom_2.2.pdf")

print("\n🎉 全部排版无瑕疵，建议直接插入您的论文使用！")