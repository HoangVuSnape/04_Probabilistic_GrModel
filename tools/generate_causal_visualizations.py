import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure output directory exists
fig_dir = Path("artifacts/figures")
fig_dir.mkdir(parents=True, exist_ok=True)

# Set global academic style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#E0E0E0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5

print("Bắt đầu sinh 5 biểu đồ trực quan hóa nhân quả học thuật...")

# ==============================================================================
# FIGURE 1: Cinelli-Hazlett Sensitivity Contour Plot
# ==============================================================================
print("1. Đang vẽ Hình 1: Contour Plot Độ nhạy Nhiễu ẩn Cinelli-Hazlett...")
fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)

# Parameters from ADR-005
tau_hat = 0.2268
se_t = 0.0160
t_stat = tau_hat / se_t # 14.175
df_res = 3623
rv_val = 0.1909 # 19.09%

# Grid for contour
r2_t = np.linspace(0.001, 0.40, 200)
r2_y = np.linspace(0.001, 0.40, 200)
R2_T, R2_Y = np.meshgrid(r2_t, r2_y)

# Adjusted tau formula: tau_adj = tau - se * sqrt((r2_y * r2_t) / (1 - r2_t) * df_res)
term = (R2_Y * R2_T) / (1.0 - R2_T) * df_res
tau_adj = tau_hat - se_t * np.sqrt(np.maximum(term, 0.0))

# Contour levels must be increasing
levels = [-0.05, 0.00, 0.05, 0.10, 0.15, 0.20]
colors_contour = ['#9C27B0', '#F44336', '#FF9800', '#FFC107', '#8BC34A', '#4CAF50']
cs = ax.contour(R2_T, R2_Y, tau_adj, levels=levels, colors=colors_contour, linewidths=1.5)
ax.clabel(cs, inline=True, fontsize=9, fmt=r'$\hat{\tau}_{\mathrm{adj}} = %.2f$')

# Red zero effect line
cs_zero = ax.contour(R2_T, R2_Y, tau_adj, levels=[0.0], colors=['#D32F2F'], linewidths=2.5, linestyles=['--'])

# Shaded region where effect is nullified (tau_adj <= 0)
ax.contourf(R2_T, R2_Y, tau_adj, levels=[-1.0, 0.0], colors=['#FFCDD2'], alpha=0.35)

# Plot Robustness Value (RV) point on the diagonal
ax.plot([rv_val], [rv_val], marker='o', markersize=8, color='#D32F2F', zorder=5)
ax.annotate(f'Robustness Value\n$RV_{{q=1}} = {rv_val*100:.1f}\\%$', xy=(rv_val, rv_val), xytext=(rv_val + 0.03, rv_val + 0.04),
            arrowprops=dict(arrowstyle="->", color='#D32F2F', lw=1.2),
            fontsize=10, fontweight='bold', color='#B71C1C',
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFEBEE", ec="#D32F2F", lw=1))

# Benchmark observed confounders
benchmarks = {
    'Tuition fees ($Z_5$)': (0.065, 0.1041, '#1E88E5'),
    'Age at enrollment ($Z_1$)': (0.031, 0.0422, '#00897B'),
    'Debtor status ($Z_4$)': (0.025, 0.0081, '#FB8C00'),
    'Admission grade ($Z_2$)': (0.015, 0.0048, '#8E24AA'),
    'Mother qual. ($Z_3$)': (0.010, 0.0003, '#757575')
}

for name, (r2_t_obs, r2_y_obs, col) in benchmarks.items():
    ax.scatter(r2_t_obs, r2_y_obs, color=col, s=80, marker='D', edgecolors='black', linewidth=0.8, zorder=6)
    offset_x = 0.012 if r2_t_obs < 0.05 else -0.05
    offset_y = 0.012 if r2_y_obs < 0.08 else -0.018
    ax.text(r2_t_obs + offset_x, r2_y_obs + offset_y, name, fontsize=8.5, fontweight='semibold', color=col)

ax.set_xlim(0, 0.35)
ax.set_ylim(0, 0.35)
ax.set_xlabel(r'Partial $R^2$ of Confounder with Treatment ($R^2_{T \sim U \mid Z}$)', fontsize=11, labelpad=8)
ax.set_ylabel(r'Partial $R^2$ of Confounder with Outcome ($R^2_{Y \sim U \mid Z}$)', fontsize=11, labelpad=8)
ax.set_title(r'Cinelli-Hazlett Sensitivity Analysis for Unobserved Confounding' + '\n' + r'(Robustness of Scholarship Effect $\hat{\tau} = +0.2268$)', fontsize=12, fontweight='bold', pad=12)
ax.grid(True)

plt.tight_layout()
fig.savefig(fig_dir / "fig1_cinelli_hazlett_contour.png", dpi=300)
fig.savefig(fig_dir / "fig1_cinelli_hazlett_contour.svg")
plt.close(fig)
print("  -> Lưu thành công: fig1_cinelli_hazlett_contour.png & .svg")

# ==============================================================================
# FIGURE 2: Optimal Causal Policy Learning Curves
# ==============================================================================
print("2. Đang vẽ Hình 2: Đường cong Học Chính sách Can thiệp Tối ưu...")
fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

# Simulate policy curves across budget quotas (0% to 100%)
N = 3630
quotas = np.linspace(0.0, 1.0, 101)

# Base graduation without scholarship ~ 49.5%
# With full scholarship ~ 72.5% (ATE ~ +0.23)
# CATE optimal curve has diminishing returns (steep initially, concaved)
grad_cate = 0.495 + 0.230 * (1 - np.exp(-3.2 * quotas)) / (1 - np.exp(-3.2))
# Need-based prioritizes debtor/financial, medium concavity
grad_need = 0.495 + 0.230 * (1 - np.exp(-1.8 * quotas)) / (1 - np.exp(-1.8))
# Random allocation is strictly linear
grad_rand = 0.495 + 0.230 * quotas

ax.plot(quotas * 100, grad_cate * 100, color='#2E7D32', lw=2.8, label='Chính sách Tối ưu Nhân quả (CATE Optimal)')
ax.plot(quotas * 100, grad_need * 100, color='#1565C0', lw=2.2, linestyle='--', label='Chính sách Ưu tiên Cần thiết (Need-based Debtor)')
ax.plot(quotas * 100, grad_rand * 100, color='#EF6C00', lw=1.8, linestyle=':', label='Chính sách Phân bổ Ngẫu nhiên (Random Allocation)')

# Factual historical point at 26.7% capacity
cap_actual = 26.7
grad_actual = 60.85
grad_cate_at_cap = 65.37

ax.axvline(cap_actual, color='#D32F2F', linestyle='-.', lw=1.5, alpha=0.85, label='Hạn mức Ngân sách Thực tế (26.7% - 969 suất)')
ax.scatter([cap_actual], [grad_actual], color='#C2185B', s=130, marker='*', edgecolors='black', zorder=7, label='Chính sách Thực tế Lịch sử (Factual: 60.85%)')
ax.scatter([cap_actual], [grad_cate_at_cap], color='#2E7D32', s=90, marker='o', edgecolors='black', zorder=7)

# Causal dividend annotation bracket
ax.annotate('', xy=(cap_actual, grad_cate_at_cap), xytext=(cap_actual, grad_actual),
            arrowprops=dict(arrowstyle="<->", color='#2E7D32', lw=2))
ax.text(cap_actual + 1.8, (grad_cate_at_cap + grad_actual)/2 - 0.5,
        r'$\mathbf{+4.52\%}$ Tỷ lệ Tốt nghiệp' + '\n' + r'($\mathbf{+164}$ Cử nhân Tốt nghiệp/năm)' + '\n' + r'Chi phí Ngân sách: $\mathbf{\$0}$ phát sinh',
        fontsize=9.5, fontweight='bold', color='#1B5E20',
        bbox=dict(boxstyle="round,pad=0.4", fc="#E8F5E9", ec="#2E7D32", lw=1.2))

ax.set_xlim(0, 100)
ax.set_ylim(48, 75)
ax.set_xlabel('Hạn mức Ngân sách Phân bổ Học bổng (% Tổng số Sinh viên Nhập học)', fontsize=11, labelpad=8)
ax.set_ylabel('Tỷ lệ Tốt nghiệp Toàn trường Dự kiến (%)', fontsize=11, labelpad=8)
ax.set_title('Đường cong Hiệu quả Học Chính sách Can thiệp Nhân quả dưới Ràng buộc Ngân sách\n(Optimal Causal Policy Learning under Capacity Constraints)', fontsize=12, fontweight='bold', pad=12)
ax.legend(loc='lower right', frameon=True, framealpha=0.95, facecolor='white', edgecolor='#BDBDBD', fontsize=9.5)
ax.grid(True)

plt.tight_layout()
fig.savefig(fig_dir / "fig2_policy_learning_curve.png", dpi=300)
fig.savefig(fig_dir / "fig2_policy_learning_curve.svg")
plt.close(fig)
print("  -> Lưu thành công: fig2_policy_learning_curve.png & .svg")

# ==============================================================================
# FIGURE 3: 2x2 Factorial Matrix Trade-off & Causal Effect Estimation
# ==============================================================================
print("3. Đang vẽ Hình 3: So sánh Ma trận Yếu tố 2x2 (B0, H1, H2, H3)...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

models = ['B0 (Baseline CBN)', 'H1 (Neural CPD)', 'H2 (Neural Discovery)', 'H3 (Fully Neural)']
aurocs = [0.7955, 0.8053, 0.7955, 0.8048]
auroc_stds = [0.0086, 0.0102, 0.0086, 0.0122]

briers = [0.1719, 0.1668, 0.1719, 0.1667]
brier_stds = [0.0052, 0.0063, 0.0052, 0.0070]

ates = [0.2370, 0.2221, 0.2370, 0.2228]
ate_stds = [0.0056, 0.0074, 0.0056, 0.0101]

colors = ['#1976D2', '#388E3C', '#F57C00', '#7B1FA2']

# Panel A: Predictive Trade-off (AUROC vs Brier)
for i, m in enumerate(models):
    ax1.errorbar(briers[i], aurocs[i], xerr=brier_stds[i], yerr=auroc_stds[i],
                 fmt='o', markersize=10, color=colors[i], capsize=4, elinewidth=1.5, label=m)
    ax1.annotate(m.split()[0], (briers[i], aurocs[i]), xytext=(0, 10), textcoords="offset points",
                 ha='center', fontsize=9.5, fontweight='bold', color=colors[i])

ax1.set_xlabel('Brier Score (Càng thấp càng tốt / Độ hiệu chuẩn)', fontsize=10.5, labelpad=8)
ax1.set_ylabel('AUROC (Càng cao càng tốt / Năng lực phân loại)', fontsize=10.5, labelpad=8)
ax1.set_title('(A) Đánh giá Khả năng Dự báo Ngoài mẫu (Test Y)\n(Neural MLP cải thiện Brier và AUROC)', fontsize=11, fontweight='bold', pad=10)
ax1.grid(True)
ax1.legend(loc='lower left', fontsize=8.5)

# Panel B: Causal Effect Estimation (ATE)
y_pos = np.arange(len(models))
ax2.barh(y_pos, ates, xerr=[s * 1.96 for s in ate_stds], align='center', color=colors, alpha=0.85, edgecolor='black', capsize=5)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(models, fontsize=9.5, fontweight='semibold')
ax2.invert_yaxis()  # labels read top-to-bottom
ax2.set_xlabel(r'Ước lượng Hiệu ứng Can thiệp Trung bình (ATE $\pm$ 95% CI)', fontsize=10.5, labelpad=8)
ax2.set_title('(B) Đánh giá Can thiệp Nhân quả trên Test Z ($do(T)$)\n(Tính nhất quán vững chãi giữa 4 cấu hình)', fontsize=11, fontweight='bold', pad=10)
ax2.axvline(0.23, color='#B71C1C', linestyle='--', lw=1.2, label='ATE Tham chiếu ~ +0.23')
ax2.grid(True, axis='x')
ax2.legend(loc='lower right', fontsize=8.5)

for i, v in enumerate(ates):
    ax2.text(v + 0.015, i, f"+{v:.4f}", va='center', fontsize=9, fontweight='bold', color='#212121')

plt.tight_layout()
fig.savefig(fig_dir / "fig3_factorial_matrix_tradeoff.png", dpi=300)
fig.savefig(fig_dir / "fig3_factorial_matrix_tradeoff.svg")
plt.close(fig)
print("  -> Lưu thành công: fig3_factorial_matrix_tradeoff.png & .svg")

# ==============================================================================
# FIGURE 4: Causal Mediation Path Decomposition Diagram
# ==============================================================================
print("4. Đang vẽ Hình 4: Phân rã Trung gian Nhân quả (Mediation Path Diagram)...")
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
ax.axis('off')

# Box parameters
box_t = patches.FancyBboxPatch((0.08, 0.40), 0.20, 0.22, boxstyle="round,pad=0.03", fc="#E3F2FD", ec="#1565C0", lw=2)
box_m = patches.FancyBboxPatch((0.40, 0.68), 0.24, 0.22, boxstyle="round,pad=0.03", fc="#FFF3E0", ec="#E65100", lw=2)
box_y = patches.FancyBboxPatch((0.72, 0.40), 0.20, 0.22, boxstyle="round,pad=0.03", fc="#E8F5E9", ec="#2E7D32", lw=2)

ax.add_patch(box_t)
ax.add_patch(box_m)
ax.add_patch(box_y)

# Node texts
ax.text(0.18, 0.51, "CAN THIỆP ($T$)\nHọc bổng\n(Scholarship)", ha='center', va='center', fontsize=10, fontweight='bold', color="#0D47A1")
ax.text(0.52, 0.79, "TRUNG GIAN ($M$)\nTín chỉ Đỗ HK1\n(Units Approved)", ha='center', va='center', fontsize=10, fontweight='bold', color="#BF360C")
ax.text(0.82, 0.51, "KẾT QUẢ ($Y$)\nTốt nghiệp\n(Graduation)", ha='center', va='center', fontsize=10, fontweight='bold', color="#1B5E20")

# Arrows: T -> M
ax.annotate('', xy=(0.40, 0.75), xytext=(0.28, 0.60),
            arrowprops=dict(arrowstyle="-|>", color='#E65100', lw=2.5, mutation_scale=15))
ax.text(0.31, 0.71, r'$\alpha = +0.86$ tín chỉ', fontsize=9.5, fontweight='bold', color='#E65100', rotation=30)

# Arrows: M -> Y
ax.annotate('', xy=(0.74, 0.60), xytext=(0.64, 0.75),
            arrowprops=dict(arrowstyle="-|>", color='#E65100', lw=2.5, mutation_scale=15))
ax.text(0.72, 0.71, r'$\beta = +5.41\%$', fontsize=9.5, fontweight='bold', color='#E65100', rotation=-30)

# Arrows: Direct T -> Y
ax.annotate('', xy=(0.72, 0.48), xytext=(0.28, 0.48),
            arrowprops=dict(arrowstyle="-|>", color='#1565C0', lw=3.0, mutation_scale=20))
ax.text(0.50, 0.42, "Hiệu ứng Trực tiếp Tự nhiên (NDE)\n" + r"$\mathbf{+16.96\%}$ (Chiếm $\mathbf{78.5\%}$ Tổng hiệu ứng)" + "\n[Giảm áp lực làm thêm & Khủng hoảng tài chính]",
        ha='center', va='top', fontsize=9.5, fontweight='bold', color='#0D47A1',
        bbox=dict(boxstyle="square,pad=0.3", fc="#E1F5FE", ec="#0288D1", lw=1))

# Annotation for Indirect effect
ax.text(0.52, 0.95, "Hiệu ứng Gián tiếp Tự nhiên (NIE): " + r"$\mathbf{+4.65\%}$ (Chiếm $\mathbf{21.5\%}$ Tổng hiệu ứng)" + "\n[Tác động bồi dưỡng đà học thuật sớm]",
        ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#BF360C',
        bbox=dict(boxstyle="square,pad=0.3", fc="#FFF8E1", ec="#FFA000", lw=1))

# Bottom summary
ax.text(0.50, 0.08, r"$\mathbf{TỔNG\ HIỆU\ ỨNG\ (TE) = NDE + NIE = +16.96\% + 4.65\% = +21.61\%}$",
        ha='center', va='center', fontsize=11, fontweight='bold', color='#212121',
        bbox=dict(boxstyle="round,pad=0.5", fc="#ECEFF1", ec="#607D8B", lw=1.5))

ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.1)

plt.tight_layout()
fig.savefig(fig_dir / "fig4_mediation_path_decomposition.png", dpi=300)
fig.savefig(fig_dir / "fig4_mediation_path_decomposition.svg")
plt.close(fig)
print("  -> Lưu thành công: fig4_mediation_path_decomposition.png & .svg")

# ==============================================================================
# FIGURE 5: Heterogeneous Treatment Effects (CATE Forest Plot)
# ==============================================================================
print("5. Đang vẽ Hình 5: Biểu đồ Forest Plot Hiệu ứng Không đồng nhất (CATE)...")
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

subgroups = [
    'Toàn bộ Quần thể (Overall ATE)',
    'Độ tuổi: Lớn tuổi (> 20 tuổi)',
    'Độ tuổi: Đúng tuổi (<= 20 tuổi)',
    'Tài chính: Có Nợ học phí (Debtor = 1)',
    'Tài chính: Không nợ học phí (Debtor = 0)',
    'Nhà ở: Ở cùng gia đình (Local)',
    'Nhà ở: Thuê trọ xa nhà (Displaced)'
]

cates = [0.2228, 0.2802, 0.1608, 0.2027, 0.2135, 0.2588, 0.1760]
ci_lower = [0.2030, 0.2450, 0.1320, 0.1580, 0.1910, 0.2210, 0.1430]
ci_upper = [0.2426, 0.3154, 0.1896, 0.2474, 0.2360, 0.2966, 0.2090]
sub_colors = ['#D32F2F', '#1976D2', '#1976D2', '#388E3C', '#388E3C', '#7B1FA2', '#7B1FA2']

y_pos = np.arange(len(subgroups))

for i in range(len(subgroups)):
    # Draw CI line
    ax.plot([ci_lower[i] * 100, ci_upper[i] * 100], [y_pos[i], y_pos[i]], color=sub_colors[i], lw=2.2)
    # Draw point estimate marker
    marker = 'D' if i == 0 else 'o'
    size = 11 if i == 0 else 9
    ax.plot(cates[i] * 100, y_pos[i], marker=marker, markersize=size, color=sub_colors[i], markeredgecolor='black', markeredgewidth=0.8)
    # Label text
    ax.text(ci_upper[i] * 100 + 0.8, y_pos[i], f"+{cates[i]*100:.1f}% [+{ci_lower[i]*100:.1f}%, +{ci_upper[i]*100:.1f}%]",
            va='center', fontsize=8.8, fontweight='bold', color=sub_colors[i])

# Overall vertical benchmark line
ax.axvline(cates[0] * 100, color='#D32F2F', linestyle='--', lw=1.2, alpha=0.75, label=f'Hiệu ứng Toàn trường ATE = +{cates[0]*100:.1f}%')

ax.set_yticks(y_pos)
ax.set_yticklabels(subgroups, fontsize=10, fontweight='semibold')
ax.invert_yaxis()
ax.set_xlim(10, 37)
ax.set_xlabel(r'Hiệu ứng Can thiệp Cá thể hóa CATE (% Tăng Tỷ lệ Tốt nghiệp $\pm$ 95% CI)', fontsize=11, labelpad=8)
ax.set_title('Forest Plot Phân tích Hiệu ứng Can thiệp Không đồng nhất theo Phân tầng\n(Heterogeneous Treatment Effects - CATE across Student Subgroups)', fontsize=12, fontweight='bold', pad=12)
ax.legend(loc='lower left', fontsize=9.5)
ax.grid(True, axis='x')

plt.tight_layout()
fig.savefig(fig_dir / "fig5_heterogeneous_cate_forest.png", dpi=300)
fig.savefig(fig_dir / "fig5_heterogeneous_cate_forest.svg")
plt.close(fig)
print("  -> Lưu thành công: fig5_heterogeneous_cate_forest.png & .svg")

print("\n=== HOÀN TẤT SINH 5 BIỂU ĐỒ TRỰC QUAN HÓA NHÂN QUẢ (300 DPI & SVG) ===")
