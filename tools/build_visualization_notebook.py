import json
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def create_cell(cell_type: str, source: list, execution_count: int = None, outputs: list = None):
    source_lines = [line if line.endswith('\n') else line + '\n' for line in source]
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source_lines
    }
    if cell_type == "code":
        cell["execution_count"] = execution_count
        cell["outputs"] = outputs if outputs is not None else []
    return cell

def build_visualization_notebook():
    cells = []

    # =========================================================================
    # Header Cell
    # =========================================================================
    cells.append(create_cell("markdown", [
        "# NOTEBOOK 5: CAUSAL VISUALIZATION SUITE & EXECUTIVE SCIENTIFIC SYNTHESIS",
        "## Trực quan hóa Học thuật Chuẩn Quốc tế (300 DPI / SVG) & Tổng hợp Kết luận Nhân quả Toàn diện",
        "### Học phần: Mô hình đồ thị có xác suất (IT701100) — HK1 2026–2027",
        "> **Tài liệu tham chiếu:** Slide 70–73, 80, 86 (`HK1_2026_IT701100_Mô hình đồ thị có xác suất_N01.pdf`).  ",
        "> **File Quyết định Kiến trúc:** [ADR-006](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/report/decisions/ADR_006_causal_visualization_and_final_synthesis.md).  ",
        "> **Bảng Checklist Nghiệm thu:** [CHECKLIST_ADR_006.md](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/report/decisions/CHECKLIST_ADR_006.md).  ",
        "",
        "---",
        "",
        "### Mục tiêu Khoa học của Notebook 5:",
        "Notebook này tích hợp toàn bộ hệ thống biểu đồ học thuật chất lượng cao (Vector SVG & 300 DPI PNG) phục vụ báo cáo khoa học, trình chiếu luận văn và ấn phẩm học thuật:",
        "1. **Hình 1: Contour Plot Độ nhạy Nhiễu ẩn Cinelli-Hazlett ($RV = 19.09\%$)** — Bác bỏ hoài nghi về thiên kiến confounder chưa quan sát $U$.",
        "2. **Hình 2: Đường cong Học Chính sách Can thiệp Tối ưu (CATE Policy Curve)** — Chứng minh mức tăng thặng dư **+164 sinh viên tốt nghiệp/năm** ở hạn mức ngân sách 26.7%.",
        "3. **Hình 3: Đánh giá Đa chiều Ma trận Yếu tố 2x2 (B0, H1, H2, H3)** — So sánh trade-off giữa Brier Score, AUROC và tính nhất quán của hiệu ứng can thiệp $ATE$.",
        "4. **Hình 4: Sơ đồ Đường dẫn Phân rã Trung gian Nhân quả ($T \\to M \\to Y$)** — Phân lập Tác động Trực tiếp NDE (78.5%) và Tác động Gián tiếp NIE qua kết quả HK1 (21.5%).",
        "5. **Hình 5: Forest Plot Hiệu ứng Can thiệp Không đồng nhất (CATE Subgroups)** — Phân tầng hiệu ứng theo Độ tuổi, Tình trạng Nợ học phí và Nơi cư trú."
    ]))

    # =========================================================================
    # Cell 1: Environment & Setup
    # =========================================================================
    cells.append(create_cell("code", [
        "# ====================================================================",
        "# Cấu hình Môi trường Trực quan hóa Học thuật",
        "# ====================================================================",
        "import sys",
        "from pathlib import Path",
        "import pandas as pd",
        "try:",
        "    from IPython.display import display, Image, Markdown",
        "except ImportError:",
        "    class Image:",
        "        def __init__(self, filename=None, **kwargs): self.filename = filename",
        "        def __repr__(self): return f'<Image: {self.filename}>'",
        "    def display(*args, **kwargs):",
        "        for a in args: print(a)",
        "",
        "if hasattr(sys.stdout, 'reconfigure'):",
        "    sys.stdout.reconfigure(encoding='utf-8')",
        "",
        "fig_dir = Path('../artifacts/figures') if Path('../artifacts/figures').exists() else Path('artifacts/figures')",
        "table_dir = Path('../artifacts/tables') if Path('../artifacts/tables').exists() else Path('artifacts/tables')",
        "print(f'Thư mục Biểu đồ: {fig_dir.resolve()}')",
        "print(f'Thư mục Bảng Dữ liệu: {table_dir.resolve()}')",
        "print('Môi trường Trực quan hóa Notebook 5 sẵn sàng!')"
    ], execution_count=1))

    # =========================================================================
    # Cell 2: Figure 1 - Cinelli-Hazlett Contour Plot
    # =========================================================================
    cells.append(create_cell("markdown", [
        "---",
        "### 1. Phân tích Độ nhạy Nhiễu ẩn Cinelli & Hazlett (Sensitivity Contour Plot)",
        "* **Cơ sở Lý thuyết (Cinelli & Hazlett, JRSS-B 2020):** Đồ thị biểu diễn đường mức giá trị can thiệp điều chỉnh $\\hat{\\tau}_{\\mathrm{adj}}$ khi có sự xuất hiện của biến nhiễu ẩn $U$.",
        "* **Giá trị Bền vững (Robustness Value $RV_{q=1}$):** Đạt mức **$19.09\\%$**.",
        "* **Ý nghĩa Khoa học:** Biến quan sát mạnh nhất trong việc dự báo bỏ học ($Z_5$: Nợ học phí) chỉ giải thích $10.41\\%$ phương sai. Một biến nhiễu ẩn cần mạnh gấp đôi $Z_5$ và gấp 4.5 lần $Z_1$ (Tuổi) thì mới có thể xóa bỏ hiệu ứng học bổng."
    ]))

    cells.append(create_cell("code", [
        "fig1_path = fig_dir / 'fig1_cinelli_hazlett_contour.png'",
        "if fig1_path.exists():",
        "    display(Image(filename=str(fig1_path), width=750))",
        "else:",
        "    print('Vui lòng chạy script sinh hình ảnh tools/generate_causal_visualizations.py trước!')"
    ], execution_count=2))

    # =========================================================================
    # Cell 3: Figure 2 - Optimal Policy Learning Curve
    # =========================================================================
    cells.append(create_cell("markdown", [
        "---",
        "### 2. Đường cong Hiệu quả Học Chính sách Nhân quả Dưới Ràng buộc Ngân sách (CATE Policy)",
        "* **Thiết lập:** So sánh tỷ lệ tốt nghiệp toàn trường theo các tỷ lệ ngân sách khác nhau ($0\\%$ đến $100\\%$).",
        "* **Hạn mức Thực tế:** Tại mốc ngân sách thực tế **$26.7\\%$** ($969$ suất học bổng trên $3,630$ sinh viên):",
        "  - Chính sách Tối ưu Nhân quả (CATE Optimal): Tỷ lệ tốt nghiệp đạt **$65.37\\%$** ($2,372$ cử nhân).",
        "  - Chính sách Thực tế Lịch sử (Factual): Tỷ lệ tốt nghiệp đạt **$60.85\\%$** ($2,208$ cử nhân).",
        "  - **Thặng dư Nhân quả (Causal Dividend):** Tăng thêm **$+164$ sinh viên tốt nghiệp/năm** với **chi phí bằng $0$**."
    ]))

    cells.append(create_cell("code", [
        "fig2_path = fig_dir / 'fig2_policy_learning_curve.png'",
        "if fig2_path.exists():",
        "    display(Image(filename=str(fig2_path), width=850))",
        "    df_policy = pd.read_csv(table_dir / 'Table_Policy_Learning.csv')",
        "    display(df_policy)",
        "else:",
        "    print('Không tìm thấy fig2_policy_learning_curve.png!')"
    ], execution_count=3))

    # =========================================================================
    # Cell 4: Figure 3 - 2x2 Factorial Matrix Trade-off
    # =========================================================================
    cells.append(create_cell("markdown", [
        "---",
        "### 3. Đánh giá Ma trận Yếu tố 2x2 & Mô hình H3 (Slide 80)",
        "* **Panel A (Predictive Trade-off):** Mạng nơ-ron đa tầng MLP ($H_1, H_3$) giúp giảm mạnh Brier Score từ $0.1719$ xuống $0.1667$ và nâng AUROC từ $0.7955$ lên $0.8053$.",
        "* **Panel B (Causal Effect Robustness):** Cả 4 mô hình đều cho ước lượng $ATE$ ổn định chặt chẽ quanh mức $+0.222$ đến $+0.237$, chứng minh cấu trúc học máy NOTEARS không làm biến dạng hiệu ứng can thiệp."
    ]))

    cells.append(create_cell("code", [
        "fig3_path = fig_dir / 'fig3_factorial_matrix_tradeoff.png'",
        "if fig3_path.exists():",
        "    display(Image(filename=str(fig3_path), width=900))",
        "    df_fact = pd.read_csv(table_dir / 'Table_Factorial_H3.csv')",
        "    display(df_fact)",
        "else:",
        "    print('Không tìm thấy fig3_factorial_matrix_tradeoff.png!')"
    ], execution_count=4))

    # =========================================================================
    # Cell 5: Figure 4 - Causal Mediation Path Decomposition
    # =========================================================================
    cells.append(create_cell("markdown", [
        "---",
        "### 4. Sơ đồ Đường dẫn Phân rã Trung gian Nhân quả ($T \\to M \\to Y$)",
        "* **Tổng hiệu ứng can thiệp (Total Effect):** $\\text{TE} = +21.61\\%$.",
        "* **Hiệu ứng Trực tiếp Tự nhiên (Natural Direct Effect - NDE):** $\\mathbf{+16.96\\%}$ ($78.5\\%$ tổng hiệu ứng) — phản ánh cơ chế giải tỏa áp lực làm thêm và căng thẳng kinh tế gia đình.",
        "* **Hiệu ứng Gián tiếp Tự nhiên (Natural Indirect Effect - NIE):** $\\mathbf{+4.65\\%}$ ($21.5\\%$ tổng hiệu ứng) — phản ánh cơ chế bồi dưỡng đà học tập sớm (tích lũy tín chỉ đỗ HK1)."
    ]))

    cells.append(create_cell("code", [
        "fig4_path = fig_dir / 'fig4_mediation_path_decomposition.png'",
        "if fig4_path.exists():",
        "    display(Image(filename=str(fig4_path), width=850))",
        "else:",
        "    print('Không tìm thấy fig4_mediation_path_decomposition.png!')"
    ], execution_count=5))

    # =========================================================================
    # Cell 6: Figure 5 - Heterogeneous Treatment Effects (CATE Forest Plot)
    # =========================================================================
    cells.append(create_cell("markdown", [
        "---",
        "### 5. Biểu đồ Forest Plot Phân tích Hiệu ứng Không đồng nhất (CATE)",
        "* **Độ tuổi:** Hiệu ứng trên sinh viên lớn tuổi ($> 20$ tuổi) đạt **$+28.02\\%$**, mạnh gấp **$1.72$ lần** so với sinh viên đúng tuổi ($+16.08\\%$).",
        "* **Khủng hoảng Nợ học phí:** Với nhóm nợ học phí (`Debtor = 1`), học bổng giúp **nhân đôi tỷ lệ tốt nghiệp** từ $19.74\\%$ lên $40.01\\%$ (Relative Risk $2.03\\times$).",
        "* **Cư trú:** Sinh viên ở cùng gia đình hưởng lợi $+25.88\\%$, cao hơn sinh viên trọ xa nhà ($+17.60\\%$)."
    ]))

    cells.append(create_cell("code", [
        "fig5_path = fig_dir / 'fig5_heterogeneous_cate_forest.png'",
        "if fig5_path.exists():",
        "    display(Image(filename=str(fig5_path), width=850))",
        "    df_cate = pd.read_csv(table_dir / 'Table_Heterogeneous_CATE.csv')",
        "    display(df_cate)",
        "else:",
        "    print('Không tìm thấy fig5_heterogeneous_cate_forest.png!')"
    ], execution_count=6))

    # =========================================================================
    # Cell 7: Summary & Verification
    # =========================================================================
    cells.append(create_cell("code", [
        "# ====================================================================",
        "# Kiểm tra Nghiệm thu Tất cả Biểu đồ và Bảng Dữ liệu",
        "# ====================================================================",
        "required_figs = [",
        "    'fig1_cinelli_hazlett_contour.png',",
        "    'fig2_policy_learning_curve.png',",
        "    'fig3_factorial_matrix_tradeoff.png',",
        "    'fig4_mediation_path_decomposition.png',",
        "    'fig5_heterogeneous_cate_forest.png'",
        "]",
        "all_present = True",
        "for f in required_figs:",
        "    p = fig_dir / f",
        "    status = 'OK' if p.exists() else 'MISSING'",
        "    size_kb = f'{p.stat().st_size / 1024:.1f} KB' if p.exists() else '0 KB'",
        "    print(f'  [Biểu đồ] {f:38s}: {status:7s} ({size_kb})')",
        "    if not p.exists(): all_present = False",
        "",
        "assert all_present, 'Thiếu biểu đồ trực quan hóa!'",
        "print('\\n=== TOÀN BỘ 5 BIỂU ĐỒ HỌC THUẬT ĐÃ ĐƯỢC XÁC THỰC THÀNH CÔNG 100% ===')"
    ], execution_count=7))

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.10.12"}
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    target_path = Path("notebook/causal_visualization_and_synthesis.ipynb")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=1, ensure_ascii=False)

    print(f"Thành công! Đã tạo Notebook 5 tại: {target_path}")
    print(f"Tổng số cells: {len(cells)}")

if __name__ == "__main__":
    build_visualization_notebook()
