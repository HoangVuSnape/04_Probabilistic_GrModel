import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Color Palette: Clean Academic Navy & Emerald
    PRIMARY = RGBColor(15, 34, 64)       # Deep Navy
    SECONDARY = RGBColor(0, 128, 128)    # Teal / Emerald
    DARK_TEXT = RGBColor(30, 41, 59)     # Slate 800
    LIGHT_BG = RGBColor(248, 250, 252)   # Slate 50
    CARD_BG = RGBColor(255, 255, 255)
    ACCENT_RED = RGBColor(220, 38, 38)
    ACCENT_GREEN = RGBColor(22, 163, 74)

    blank_layout = prs.slide_layouts[6]

    def add_header(slide, title_text, category_text="MÔ HÌNH ĐỒ THỊ CÓ XÁC SUẤT (IT701100) — BÁO CÁO GIỮA KỲ"):
        # Header banner
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.733), Inches(1.1))
        tf = header_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        p_cat = tf.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = SECONDARY
        
        p_title = tf.add_paragraph()
        p_title.text = title_text
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = PRIMARY
        p_title.space_before = Pt(4)

    # -------------------------------------------------------------
    # SLIDE 1: TITLE SLIDE
    # -------------------------------------------------------------
    s1 = prs.slides.add_slide(blank_layout)
    tb1 = s1.shapes.add_textbox(Inches(1.0), Inches(1.2), Inches(11.333), Inches(5.0))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.text = "TRƯỜNG ĐẠI HỌC BÁCH KHOA HÀ NỘI — CHƯƠNG TRÌNH THẠC SĨ CNTT"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = SECONDARY

    p = tf1.add_paragraph()
    p.text = "HỌC PHẦN: MÔ HÌNH ĐỒ THỊ CÓ XÁC SUẤT (IT701100)"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = PRIMARY
    p.space_before = Pt(8)

    p = tf1.add_paragraph()
    p.text = "SUY LUẬN NHÂN QUẢ VÀ ĐỐI CHUẨN MÔ HÌNH ĐỒ THỊ LAI (HYBRID CBN & DEEP LEARNING)"
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = PRIMARY
    p.space_before = Pt(16)

    p = tf1.add_paragraph()
    p.text = "Đánh giá Chính sách Cấp Học bổng lên Tỷ lệ Tốt nghiệp trên Dữ liệu Bảng UCI ID 697"
    p.font.size = Pt(18)
    p.font.color.rgb = SECONDARY
    p.space_before = Pt(6)

    p = tf1.add_paragraph()
    p.text = "• Giảng viên hướng dẫn: TS. Nguyễn Chí Thiện (Nguyen Chi Thien, Ph.D.)\n" \
             "• Nhóm học viên: Nhóm Nghiên cứu Causal ML\n" \
             "• Đối chiếu yêu cầu: Slide 39–44, 53–75, 76–79 của học phần (Tables A–C, 5 paired runs)"
    p.font.size = Pt(13)
    p.font.color.rgb = DARK_TEXT
    p.space_before = Pt(28)

    # -------------------------------------------------------------
    # SLIDE 2: RESEARCH QUESTION & CAUSAL VOCABULARY
    # -------------------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "1. Câu hỏi Nghiên cứu Can thiệp & Không gian Biến số")
    
    tb2 = s2.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.2))
    tf2 = tb2.text_frame
    tf2.word_wrap = True

    p = tf2.paragraphs[0]
    p.text = "CÂU HỎI NHÂN QUẢ MỤC TIÊU (CAUSAL ESTIMAND - SLIDE 54):"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = PRIMARY

    p = tf2.add_paragraph()
    p.text = '“Dưới các giả định Backdoor, việc can thiệp áp đặt chính sách cấp học bổng do(T=1) làm thay đổi xác suất tốt nghiệp Y=1 như thế nào so với tình huống không cấp do(T=0) trên cùng một quần thể sinh viên tham chiếu?”'
    p.font.size = Pt(14)
    p.font.italic = True
    p.font.color.rgb = SECONDARY
    p.space_before = Pt(4)

    p = tf2.add_paragraph()
    p.text = "TỪ ĐIỂN 7 BIẾN SỐ TRONG PRIMARY DOMAIN DAG (UCI ID 697):"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = PRIMARY
    p.space_before = Pt(16)

    p = tf2.add_paragraph()
    p.text = "• Can thiệp (T): Scholarship holder (0 = Không, 1 = Có học bổng) — Đo lường tại thời điểm nhập học (t1)\n" \
             "• Kết quả (Y): Target (1 = Graduate: 60.85%, 0 = Dropout: 39.15%) — Đo lường tại kết thúc khóa học (t2)\n" \
             "• Biến gây nhiễu Confounders (Z) xác lập trước khi nhập học (t0 < t1):\n" \
             "    - Z1 (Điểm xét tuyển): 3 nhóm {<120, 120–140, >=140}\n" \
             "    - Z2 (Debtor — nợ khi nhập học): 0/1\n" \
             "    - Z3 (Đóng học phí đúng hạn): 0/1 — thời điểm đo mơ hồ, có thể là hậu quả của bỏ học (giả định)\n" \
             "    - Z4 (Học vấn của mẹ): {0: dưới đại học, 1: đại học trở lên (mã >= 13)}\n" \
             "    - Z5 (Tuổi nhập học): {0: <= 20, 1: > 20}\n" \
             "• Chống rò rỉ: các biến tín chỉ HK1/HK2 xảy ra sau T nên KHÔNG đưa vào Z (M1 chỉ dùng cho phân tích trung gian)."
    p.font.size = Pt(12)
    p.font.color.rgb = DARK_TEXT
    p.space_before = Pt(6)

    # -------------------------------------------------------------
    # SLIDE 3: DOMAIN DAG & BACKDOOR IDENTIFICATION
    # -------------------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "2. Cấu trúc Đồ thị Primary Domain DAG & Công thức Nhận diện")

    tb3 = s3.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.2))
    tf3 = tb3.text_frame
    tf3.word_wrap = True

    p = tf3.paragraphs[0]
    p.text = "DOMAIN DAG THAM CHIẾU 7 NÚT & 11 CẠNH (B0 HỌC DAG RIÊNG TRÊN TRAIN — XEM SLIDE SAU):"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = PRIMARY

    p = tf3.add_paragraph()
    p.text = "• 5 cạnh từ Confounders sang Can thiệp: Z1 -> T, Z2 -> T, Z3 -> T, Z4 -> T, Z5 -> T\n" \
             "• 5 cạnh từ Confounders sang Kết quả: Z1 -> Y, Z2 -> Y, Z3 -> Y, Z4 -> Y, Z5 -> Y\n" \
             "• 1 cạnh can thiệp trực tiếp: T -> Y (Hiệu ứng nhân quả mục tiêu)\n" \
             "• Phân rã xác suất đồng thời: P(Z, T, Y) = [∏ P(Zi)] * P(T | Z1..Z5) * P(Y | T, Z1..Z5)"
    p.font.size = Pt(12)
    p.font.color.rgb = DARK_TEXT
    p.space_before = Pt(4)

    p = tf3.add_paragraph()
    p.text = "PHÂN BIỆT BẢN CHẤT: QUAN SÁT P(Y|T) vs CAN THIỆP P(Y|do(T)):"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = PRIMARY
    p.space_before = Pt(14)

    p = tf3.add_paragraph()
    p.text = "• Xác suất Quan sát P(Y=1 | T=t): Bị ô nhiễm bởi các đường dẫn cửa sau T <- Z -> Y (chọn lọc tự nhiên)\n" \
             "• Xác suất Can thiệp P(Y=1 | do(T=t)): Phẫu thuật đồ thị (Graph Surgery) cắt bỏ các cạnh đi vào T:\n" \
             "    P(Y=1 | do(T=t)) = ∑_z P(Y=1 | T=t, Z=z) * P(Z=z)\n" \
             "• Chuẩn hóa trên tập Test: q_m(t, C_i) = P_m(Y=1 | T=t, Pa(Y)\\T = C_i), risk_m(t) = (1/n_test) * ∑_i q_m(t, C_i)\n" \
             "• ATE_m = risk_m(1) - risk_m(0). Y_test chỉ dùng chấm điểm dự báo, KHÔNG dùng để tính hiệu ứng can thiệp."
    p.font.size = Pt(12)
    p.font.color.rgb = DARK_TEXT
    p.space_before = Pt(4)

    # -------------------------------------------------------------
    # SLIDE 4: THREE CORE MODELS B0, H1, H2
    # -------------------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "3. Thiết kế Kiểm soát 3 Mô hình: B0, H1 và H2")

    tb4 = s4.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.2))
    tf4 = tb4.text_frame
    tf4.word_wrap = True

    p = tf4.paragraphs[0]
    p.text = "MA TRẬN ĐỐI CHỨNG CÓ KIỂM SOÁT (SLIDE 32 & 53):"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = PRIMARY

    p = tf4.add_paragraph()
    p.text = "Ràng buộc chung (khai báo trước khi học): cấm Y -> {T, Z}, cấm T -> Z; bắt buộc T -> Y (giả thuyết cần ước lượng).\n\n" \
             "1. B0 (CBN truyền thống):\n" \
             "   - Đồ thị: pgmpy HillClimbSearch trên TRAIN; hàm điểm {BIC, BDeu, K2} chọn theo log-likelihood VALIDATION (K2 ở cả 5 seed).\n" \
             "   - Tham số: CPD rời rạc BDeu (ESS = 5). DAG học được: Pa(Y) = {T, Debtor, Học phí, Tuổi}.\n\n" \
             "2. H1 (Neural CPD trên DAG B0):\n" \
             "   - Giữ nguyên DAG B0 của từng run; thay CPD của Y bằng MLP PyTorch chỉ nhận Pa(Y) (4 đầu vào, 769 tham số).\n" \
             "   - BCE loss, Adam, dropout 0.2, dừng sớm trên VALIDATION; liệt kê mọi cấu hình cha -> TabularCPD -> check_model().\n\n" \
             "3. H2 (Neural discovery — NOTEARS-MLP phi tuyến):\n" \
             "   - Mỗi nút một MLP; W_ij = chuẩn trọng số tầng 1 từ i vào j; ràng buộc h(W) = tr(exp(W∘W)) - d = 0 (Augmented Lagrangian).\n" \
             "   - Cùng mặt nạ M như B0 (trọng số cấm = 0, có assert); hội tụ khi h < 1e-6 (đạt ~1e-7 ở mọi seed).\n" \
             "   - λ1 và ngưỡng τ chọn trên VALIDATION; CPD BDeu như B0; DoWhy identify trên DAG học được."
    p.font.size = Pt(12)
    p.font.color.rgb = DARK_TEXT
    p.space_before = Pt(4)

    # -------------------------------------------------------------
    # SLIDE 5: EXPERIMENTAL PROTOCOL & 5 PAIRED RUNS
    # -------------------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "4. Quy trình Thực nghiệm Ghép cặp Cố định (5 Paired Runs)")

    tb5 = s5.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.2))
    tf5 = tb5.text_frame
    tf5.word_wrap = True

    p = tf5.paragraphs[0]
    p.text = "THIẾT KẾ ĐỐI CHỨNG NGHIÊM NGẶT (SLIDE 28, 56, 68):"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = PRIMARY

    p = tf5.add_paragraph()
    p.text = "• Quy mô mẫu sạch: Loại bỏ 794 sinh viên 'Enrolled' (chưa có nhãn tốt nghiệp/bỏ học) -> N = 3,630 sinh viên.\n" \
             "• Tỷ lệ phân chia mẫu: 70% Train (2,541 SV) / 15% Validation (544 SV) / 15% Test (545 SV).\n" \
             "• 5 Hạt giống Cố định (Paired Seeds = [42, 43, 44, 45, 46]):\n" \
             "   - Đóng băng toàn bộ chỉ số dòng thành các file JSON (artifacts/splits/split_seed_{seed}.json).\n" \
             "   - Trong mỗi seed: B0, H1 và H2 chạy trên CÙNG MỘT TẬP TRAIN, VAL và TEST.\n" \
             "• Nguyên tắc Đánh giá Tách bạch:\n" \
             "   - Đánh giá Dự báo: So sánh P(Y=1 | T, Z) với nhãn thực tế Y_test (đo AUROC, F1, Brier Score).\n" \
             "   - Đánh giá Can thiệp: Chuẩn hóa P(Y=1 | do(T)) trên tập hiệp biến tham chiếu C_test (n = 545 SV).\n" \
             "• Chọn thiết lập trên VALIDATION (hàm điểm B0, dừng sớm H1, λ1/τ của H2), đóng băng trước khi chạm TEST.\n" \
             "• Lưu artifact: split JSON, DAG từng seed (artifacts/graphs), checkpoint H1 (artifacts/checkpoints), Tables A–C (CSV)."
    p.font.size = Pt(13)
    p.font.color.rgb = DARK_TEXT
    p.space_before = Pt(6)

    # -------------------------------------------------------------
    # SLIDE 6: TABLE A - PREDICTION RESULTS
    # -------------------------------------------------------------
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "5. Bảng A: Đánh giá Năng lực Dự báo Ngoài Mẫu (Held-Out Prediction)")

    tb6 = s6.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.2))
    tf6 = tb6.text_frame
    tf6.word_wrap = True

    p = tf6.paragraphs[0]
    p.text = "KẾT QUẢ ĐỐI SÁNH TRÊN NHÃN KIỂM THỬ Y_test (POOLED 5 PAIRED RUNS - SLIDE 74):"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = PRIMARY

    p = tf6.add_paragraph()
    p.text = "======================================================================================================\n" \
             "Model    Cấu trúc Đồ thị        AUROC (Mean±Std)     Macro-F1 (Mean±Std)   Brier Score (Mean±Std)   Δ_AUROC\n" \
             "------------------------------------------------------------------------------------------------------\n" \
             "B0       HillClimb-K2 DAG       0.7996 ± 0.0119      0.7351 ± 0.0077       0.1659 ± 0.0058          Baseline\n" \
             "H1       B0 DAG + MLP CPD(Y)    0.8012 ± 0.0130      0.7351 ± 0.0077       0.1656 ± 0.0062          +0.0016 ± 0.0025\n" \
             "H2       NOTEARS-MLP DAG        0.8034 ± 0.0155      0.7261 ± 0.0277       0.1666 ± 0.0060          +0.0038 ± 0.0038\n" \
             "======================================================================================================\n" \
             "Test n = 545 mỗi seed; ngưỡng F1 chọn trên VALIDATION; Δ là hiệu ghép cặp cùng seed (mean ± std qua 5 run)."
    p.font.size = Pt(11)
    p.font.name = "Consolas"
    p.font.color.rgb = DARK_TEXT
    p.space_before = Pt(4)

    p = tf6.add_paragraph()
    p.text = "NHẬN XÉT BẢNG A (NULL FINDING):\n" \
             "1. Hiệu ghép cặp H1−B0 và H2−B0 nhỏ hơn hoặc ngang độ tản giữa các run; Brier gần như không đổi -> không có bằng chứng phần nơ-ron cải thiện dự báo.\n" \
             "2. Macro-F1 của B0 và H1 trùng nhau vì Pa(Y) chỉ có 16 cấu hình: hai mô hình gán nhãn giống nhau tại ngưỡng chọn trên VALIDATION; khác biệt chỉ ở xác suất."
    p.font.size = Pt(12)
    p.font.color.rgb = DARK_TEXT
    p.space_before = Pt(10)

    # -------------------------------------------------------------
    # SLIDE 7: TABLE B - CAUSAL INTERVENTION RESULTS
    # -------------------------------------------------------------
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "6. Bảng B: Đánh giá Can thiệp Nhân quả Chuẩn hóa trên C_test")

    tb7 = s7.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.2))
    tf7 = tb7.text_frame
    tf7.word_wrap = True

    p = tf7.paragraphs[0]
    p.text = "KẾT QUẢ SUY LUẬN CAN THIỆP CHUẨN HÓA TRÊN HIỆP BIẾN THAM CHIẾU C_test (SLIDE 75):"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = PRIMARY

    p = tf7.add_paragraph()
    p.text = "======================================================================================================\n" \
             "Model    Risk(do(T=0))        Risk(do(T=1))        Causal ATE (Mean±Std)   95% Bootstrap CI       Adjustment\n" \
             "------------------------------------------------------------------------------------------------------\n" \
             "B0       0.5495 ± 0.0114      0.7914 ± 0.0092      +0.2418 ± 0.0046        [+0.2341, +0.2496]     {Z2,Z3,Z5}\n" \
             "H1       0.5494 ± 0.0112      0.7869 ± 0.0183      +0.2374 ± 0.0114        [+0.2293, +0.2454]     {Z2,Z3,Z5}\n" \
             "H2       0.5507 ± 0.0098      0.7865 ± 0.0104      +0.2358 ± 0.0067        [+0.2261, +0.2454]     {Z1?,Z2,Z3,Z5}\n" \
             "======================================================================================================\n" \
             "CI: percentile bootstrap 1000 lần trên hồ sơ C_test, mô hình cố định (trung bình cận theo seed). Z1? = có ở 4/5 seed."
    p.font.size = Pt(11)
    p.font.name = "Consolas"
    p.font.color.rgb = DARK_TEXT
    p.space_before = Pt(4)

    p = tf7.add_paragraph()
    p.text = "NHẬN XÉT BẢNG B:\n" \
             "1. Kết luận can thiệp không đổi giữa B0, H1, H2: ATE ≈ +0.24 (hiệu H1−B0 = -0.0044 ± 0.0106, H2−B0 = -0.0060 ± 0.0042).\n" \
             "2. So với chênh lệch thô +0.3454, khoảng 0.10 (≈ 30%) là thiên lệch do nhiễu QUAN SÁT được; nhiễu ẩn đánh giá bằng RV = 20.97%.\n" \
             "3. Overlap: H2 thêm Z1 (3 trạng thái) nên 1–2% hồ sơ test thiếu hỗ trợ ở T=0 hoặc T=1 (báo trong Table B). CI hẹp không chứng minh ước lượng đúng."
    p.font.size = Pt(12)
    p.font.color.rgb = DARK_TEXT
    p.space_before = Pt(10)

    # -------------------------------------------------------------
    # SLIDE 8: TABLE C - REFUTATION & COMPUTATIONAL COST
    # -------------------------------------------------------------
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "7. Bảng C: Kiểm định Độ bền vững DoWhy & Chi phí Thực thi")

    tb8 = s8.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.2))
    tf8 = tb8.text_frame
    tf8.word_wrap = True

    p = tf8.paragraphs[0]
    p.text = "KIỂM ĐỊNH PHẢN BÁC DOWHY REFUTATIONS & TÀI NGUYÊN (SLIDE 70, 71, 75):"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = PRIMARY

    p = tf8.add_paragraph()
    p.text = "======================================================================================================\n" \
             "Model  Đối tượng kiểm định                 Placebo     Subset 80%   Spread 5 run   Fit / run (CPU)\n" \
             "------------------------------------------------------------------------------------------------------\n" \
             "B0     DoWhy LR trên DAG B0 (seed 42)      +0.0016     +0.2265      0.0087         0.43 s\n" \
             "H1     Huấn luyện lại MLP (tự cài)         +0.0177     +0.2365      0.0277         1.72 s\n" \
             "H2     DoWhy LR trên DAG H2 (seed 42)      +0.0016     +0.2233      0.0150         17.64 s\n" \
             "======================================================================================================\n" \
             "Ước lượng gốc: B0 +0.2309, H1 +0.2453, H2 +0.2262. Placebo DoWhy 30 mô phỏng; H1 3 lần huấn luyện lại. Truy vấn ~0.2 ms."
    p.font.size = Pt(11)
    p.font.name = "Consolas"
    p.font.color.rgb = DARK_TEXT
    p.space_before = Pt(4)

    p = tf8.add_paragraph()
    p.text = "DIỄN GIẢI (SLIDE 70–71):\n" \
             "• Refuter DoWhy kiểm tra (đồ thị + ước lượng tuyến tính), không kiểm tra trực tiếp CPD BDeu; refuter tuyến tính không kiểm tra được H1,\n" \
             "  nên H1 có kiểm định huấn luyện lại riêng, ghi nhãn rõ.\n" \
             "• Placebo về ~0 và subset ổn định: ước lượng không phụ thuộc vào một mẫu con cụ thể.\n" \
             "• Vượt qua refuter KHÔNG chứng minh DAG đúng hay không có nhiễu ẩn."
    p.font.size = Pt(12)
    p.font.color.rgb = DARK_TEXT
    p.space_before = Pt(10)

    # -------------------------------------------------------------
    # SLIDE 9: ABLATION EXPERIMENTS A1 - A3
    # -------------------------------------------------------------
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "8. Phụ lục tùy chọn: Ablations A1–A3 & Bộ ước lượng khác")

    tb9 = s9.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.2))
    tf9 = tb9.text_frame
    tf9.word_wrap = True

    p = tf9.paragraphs[0]
    p.text = "NGHIÊN CỨU BÓC TÁCH TỪNG THÀNH PHẦN (SLIDE 33–35, 64–67):"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = PRIMARY

    p = tf9.add_paragraph()
    p.text = "A1–A3 là phần TÙY CHỌN (Slide 53) và được chạy bằng script riêng tools/run_ablations_and_extensions.py với\n" \
             "Domain DAG cố định (pipeline trước khi B0 chuyển sang học DAG) — chỉ đọc như phụ lục, không so trực tiếp với Tables A–C.\n\n" \
             "1. A1 (bỏ dropout của H1), A2 (bỏ mặt nạ M của H2), A3 (MLE thay BDeu): kết quả chi tiết trong artifacts/tables/Table_Extended_Ablations.csv.\n" \
             "2. Ví dụ trực tiếp trong notebook chính (mục 3.3): khi KHÔNG có ràng buộc, HillClimb học ra Target -> Admission_grade\n" \
             "   (kết cục gây ra điểm đầu vào) — minh họa vì sao cần ràng buộc thời gian cho cả B0 và H2.\n\n" \
             "ĐỐI CHIẾU CÁC BỘ ƯỚC LƯỢNG KHÁC CHO T = HỌC BỔNG (Table Multi-Treatment, C_test):\n" \
             "• LPM +0.218, G-computation +0.218, Outcome Regression (T-learner) +0.231 — cùng bậc với B0/H1/H2 (≈ +0.24)."
    p.font.size = Pt(12)
    p.font.color.rgb = DARK_TEXT
    p.space_before = Pt(6)

    # -------------------------------------------------------------
    # SLIDE 10: HONEST FAILURE MODES & LIMITATIONS
    # -------------------------------------------------------------
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "9. Thảo luận Trung thực về Giới hạn & Ca Thất bại (Slide 79)")

    tb10 = s10.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.2))
    tf10 = tb10.text_frame
    tf10.word_wrap = True

    p = tf10.paragraphs[0]
    p.text = "BA GIỚI HẠN VÀ CA THẤT BẠI CẦN GHI NHẬN THỰC TẾ:"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = PRIMARY

    p = tf10.add_paragraph()
    p.text = "1. Null finding về dự báo: CPD nơ-ron (H1) và học cấu trúc nơ-ron (H2) không cải thiện AUROC/Brier vượt độ tản giữa các run.\n\n" \
             "2. Chiều cạnh giữa các biến nền không xác định được: Z_i -> Z_j đổi chiều giữa B0/H2 và giữa các seed\n" \
             "   (không ảnh hưởng do(T) vì mọi Z đều là tiền can thiệp). Cạnh T -> Y của B0 là cạnh BẮT BUỘC (giả thuyết), không phải phát hiện.\n\n" \
             "3. Overlap: 1–2% hồ sơ test của H2 thiếu hỗ trợ ở T=0 hoặc T=1 trong TRAIN -> ước lượng dựa vào làm mịn BDeu.\n\n" \
             "4. Z3 'Học phí đúng hạn' có thể là hậu quả của bỏ học (nhân quả ngược); điều chỉnh theo Z3 có thể gây chệch.\n\n" \
             "5. Nhiễu ẩn: RV_{q=1} = 20.97% — kết luận vững với nhiễu yếu hơn biến quan sát mạnh nhất, nhưng không miễn nhiễm; refuter không thay thế RCT."
    p.font.size = Pt(12)
    p.font.color.rgb = DARK_TEXT
    p.space_before = Pt(6)

    # -------------------------------------------------------------
    # SLIDE 11: ADVANCED EXTENSIONS (MEDIATION & CATE)
    # -------------------------------------------------------------
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "10. Mở rộng Nâng cao: Phân rã Trung gian (Mediation) & CATE")

    tb11 = s11.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.2))
    tf11 = tb11.text_frame
    tf11.word_wrap = True

    p = tf11.paragraphs[0]
    p.text = "PHÂN RÃ CƠ CHẾ TÁC ĐỘNG (CAUSAL MEDIATION ANALYSIS T -> M -> Y):\n" \
             "• Tổng hiệu ứng (Total Effect): TE = +21.61%\n" \
             "• Hiệu ứng Trực tiếp Tự nhiên (NDE): +16.96% (78.5% tổng hiệu ứng) -> Trực tiếp giải tỏa áp lực làm thêm và căng thẳng tài chính.\n" \
             "• Hiệu ứng Gián tiếp Tự nhiên (NIE): +4.65% (21.5% tổng hiệu ứng) -> Học bổng giúp đỗ thêm +0.86 tín chỉ HK1 (M), từ đó tăng xác suất tốt nghiệp."
    p.font.size = Pt(12)
    p.font.color.rgb = DARK_TEXT

    p = tf11.add_paragraph()
    p.text = "HIỆU ỨNG CAN THIỆP KHÔNG ĐỒNG NHẤT (CATE SUBGROUPS):\n" \
             "• Phương pháp: G-computation logistic theo NHÓM định trước (không phải Causal Forest); chưa có CI cho từng nhóm.\n" \
             "• Theo tuổi: >20 tuổi CATE = +28.02% so với +16.08% ở nhóm <=20.\n" \
             "• Theo nợ: Debtor = 1 có hiệu ứng tuyệt đối +20.27%, KHÔNG lớn hơn Debtor = 0 (+21.35%); chỉ tỷ số rủi ro cao hơn (2.03x) vì mức nền thấp.\n\n" \
             "Ý NGHĨA: gợi ý ưu tiên nhóm nhập học lớn tuổi, cần thí điểm có kiểm soát trước khi áp dụng."
    p.font.size = Pt(12)
    p.font.color.rgb = DARK_TEXT
    p.space_before = Pt(8)

    # -------------------------------------------------------------
    # SLIDE 12: 6-STEP LIVE DEMO PROTOCOL (SLIDE 77)
    # -------------------------------------------------------------
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "11. Kịch bản Diễn tập Demo Trực tiếp 6 Bước trên Colab (Slide 77)")

    tb12 = s12.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.2))
    tf12 = tb12.text_frame
    tf12.word_wrap = True

    p = tf12.paragraphs[0]
    p.text = "KỊCH BẢN THAO TÁC TRỰC TIẾP TRƯỚC HỘI ĐỒNG (6 BƯỚC CHUẨN MỰC):"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = PRIMARY

    p = tf12.add_paragraph()
    p.text = "• BƯỚC 1: Mục 1.5 (câu hỏi, từ điển biến, giả định) -> Hình 6.1 (Domain DAG + graph surgery) -> giá trị T=0/1 và C_test.\n" \
             "• BƯỚC 2: Nêu TRAIN (fit) / VALIDATION (chọn hàm điểm B0, dừng sớm H1, λ1/τ H2) / TEST (chấm dự báo + chuẩn hóa).\n" \
             "• BƯỚC 3: Hình 8.1 + bảng tóm tắt đồ thị: DAG B0 học được, DAG H2, mặt nạ M, |W|, h(W), kết quả identify.\n" \
             "• BƯỚC 4: Bảng q_m(0, C_i), q_m(1, C_i) của B0 và H1 cho vài hồ sơ test -> trung bình ra risk và ATE.\n" \
             "• BƯỚC 5: Tables A–C + Hình 8.3: AUROC (null finding), ATE ≈ +0.24 ở cả 3 mô hình, CI, refuter, runtime.\n" \
             "• BƯỚC 6: Ca hạn chế: overlap của H2, Z3 có thể là hậu quả, RV = 20.97%; truy mọi số về file trong artifacts/."
    p.font.size = Pt(13)
    p.font.color.rgb = DARK_TEXT
    p.space_before = Pt(8)

    # -------------------------------------------------------------
    # SLIDE 13: CONCLUSION & CHECKLIST VERIFICATION
    # -------------------------------------------------------------
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "12. Tổng kết & Đối soát Bảng Kiểm tra Nghiệm thu (10/10 Điểm)")

    tb13 = s13.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.2))
    tf13 = tb13.text_frame
    tf13.word_wrap = True

    p = tf13.paragraphs[0]
    p.text = "ĐỐI CHIẾU BẰNG CHỨNG VỚI RUBRIC (SLIDE 42 & 43) — tự đánh giá, điểm do giảng viên quyết định:"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = PRIMARY

    p = tf13.add_paragraph()
    p.text = "• Biểu diễn nhân quả: biến, Domain DAG, factorization, giả định (mục 1.5, 6)\n" \
             "• Học & vai trò DL: B0 HillClimb + BDeu, H1 MLP CPD, H2 NOTEARS-MLP; giới hạn định danh (mục 8.1–8.2c)\n" \
             "• Suy luận: P(Y|T) vs P(Y|do(T)), DoWhy identify trên từng DAG, q_m(t, C_i) (mục 2, 7, 8)\n" \
             "• Bài toán & dữ liệu UCI 697: nguồn, checksum, luồng loại trừ, trùng lặp, overlap (mục 1)\n" \
             "• Cài đặt chạy được: notebook chuyên sâu (≈ 3–4 phút CPU) + notebook chính; phiên bản thư viện & git revision\n" \
             "• Table A (dự báo, hiệu ghép cặp), Table B (risk, ATE, CI, tập điều chỉnh, overlap), Table C (refuter, spread, runtime, phần cứng)\n" \
             "• Kết luận: ATE ≈ +0.24 ổn định giữa B0/H1/H2; phần nơ-ron không cải thiện dự báo (null finding); hạn chế đã nêu ở slide 9\n\n" \
             "Số liệu trong slide lấy từ artifacts/tables/*.csv sinh bởi notebook chính 00_MASTER_Causal_PGM_Educational_Intervention.ipynb (mục 8–10)."
    p.font.size = Pt(12)
    p.font.color.rgb = DARK_TEXT
    p.space_before = Pt(6)

    # Save presentation
    output_path = os.path.abspath("report/midterm/Presentation_Midterm_Group_CausalML.pptx")
    prs.save(output_path)
    print(f"Successfully generated PowerPoint presentation at: {output_path}")

if __name__ == "__main__":
    build_presentation()
