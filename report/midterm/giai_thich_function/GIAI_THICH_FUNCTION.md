# Giải thích các hàm trong notebook chính

Notebook: `notebook/00_MASTER_Causal_PGM_Educational_Intervention.ipynb`. Các hàm được chia thành 13 nhóm theo chức năng; mỗi nhóm là một file `GroupNN_TenNhom.py` trong thư mục này, chứa **code nguyên văn** của notebook kèm chú thích chi tiết (mục đích, đầu vào/ra, từng bước, lý thuyết, câu hỏi hay gặp). File `.md` này là bảng tra cứu nhanh.

> Code trong các file `.py` không được sửa so với notebook; mọi giải thích nằm trong comment. Notebook không bị thay đổi.

## Bảng nhóm

| Nhóm | File | Nội dung | Mục notebook |
| :--- | :--- | :--- | :--- |
| 01 | [Group01_NapDuLieu_KiemToan.py](Group01_NapDuLieu_KiemToan.py) | Nạp dữ liệu UCI 697 và kiểm toán trước khi mô hình hóa (mục 1) | Cell 4–14 |
| 02 | [Group02_ChenhLechTho_TrucQuan.py](Group02_ChenhLechTho_TrucQuan.py) | Chênh lệch quan sát thô và các hàm vẽ hình (mục 2, 3, 6, 8) | Cell 17–64 |
| 03 | [Group03_HocCauTruc_HocThamSo.py](Group03_HocCauTruc_HocThamSo.py) | Minh họa học cấu trúc và học tham số (mục 3, 4) | Cell 21–25 |
| 04 | [Group04_KiemTraDAG_DoWhy.py](Group04_KiemTraDAG_DoWhy.py) | Bốn kiểm tra hợp lệ của DAG và quy trình DoWhy trên Domain DAG (mục 5, 6, 7) | Cell 28–35 |
| 05 | [Group05_CauHinh_DuLieu_Split.py](Group05_CauHinh_DuLieu_Split.py) | Cấu hình chung, mã hóa 7 biến và 5 split dùng chung (mục 8.A) | Cell 41–43 |
| 06 | [Group06_MoHinh_B0.py](Group06_MoHinh_B0.py) | Mô hình B0: học DAG truyền thống + CPD BDeu + truy vấn can thiệp (mục 8.B) | Cell 45 |
| 07 | [Group07_MoHinh_H1.py](Group07_MoHinh_H1.py) | Mô hình H1: thay CPD của Y bằng mạng nơ-ron trên DAG của B0 (mục 8.C) | Cell 47 |
| 08 | [Group08_MoHinh_H2.py](Group08_MoHinh_H2.py) | Mô hình H2: học DAG bằng NOTEARS-MLP có mặt nạ M (mục 8.D) | Cell 49 |
| 09 | [Group09_DanhGia_5Run_TablesAB.py](Group09_DanhGia_5Run_TablesAB.py) | Đánh giá dùng chung, vòng lặp 5 run ghép cặp, xuất Table A và B (mục 8.E) | Cell 51–55 |
| 10 | [Group10_HuanLuyen_Overfit.py](Group10_HuanLuyen_Overfit.py) | Quá trình huấn luyện, kiểm tra overfit, có / không early stopping (mục 8.E2) | Cell 53 |
| 11 | [Group11_Refuter_BangKetQua.py](Group11_Refuter_BangKetQua.py) | Refuter (Table C), hiển thị bảng, positivity, đọc lại đồ thị và q_m minh bạch (mục 8.F, 8.2b, 8.2c) | Cell 57–62 |
| 12 | [Group12_PhanTichMoRong_Muc9.py](Group12_PhanTichMoRong_Muc9.py) | Phân tích mở rộng: kết cục khác, AIPW, trung gian, CATE, độ nhạy RV (mục 9) | Cell 69–83 |
| 13 | [Group13_PhanBoChinhSach_Muc10.py](Group13_PhanBoChinhSach_Muc10.py) | Mô phỏng phân bổ học bổng theo CATE (mục 10) | Cell 86 |

## Tra cứu theo hàm

### Nhóm 01 — Nạp dữ liệu UCI 697 và kiểm toán trước khi mô hình hóa (mục 1)
File: [Group01_NapDuLieu_KiemToan.py](Group01_NapDuLieu_KiemToan.py). Đọc dữ liệu gốc, lọc quần thể Graduate/Dropout, ghi môi trường chạy, báo cáo luồng loại trừ, phân loại biến theo thời điểm đo, vẽ EDA (Hình 1.1, 1.2) và kiểm tra overlap thô (Bảng 1.6b).

- **`_dinh_dang_so`** · Cell 4 · def — Định dạng số khi HIỂN THỊ bảng pandas: 4 chữ số thập phân, số nguyên in gọn, số rất nhỏ (p-value) dạng khoa học.
- **`ThongKeTapDuLieu`** · Cell 4 · class — Dataclass gom các con số tóm tắt quần thể: số mẫu, số tốt nghiệp/bỏ học, số nhận học bổng và tỷ lệ.
- **`doc_va_kiem_toan_du_lieu_uci`** · Cell 4 · def — Đọc file UCI 697, chuẩn hóa tên cột, lọc quần thể Graduate/Dropout và tạo nhãn nhị phân Target_Graduate.
  - *Hỏi:* Vì sao loại Enrolled? → *Đáp:* Họ chưa có kết cục lúc chốt dữ liệu; gán 0 hay 1 đều là đoán. Mục 9.D kiểm tra lại trên cả 4,424 SV bằng kết cục giữ chân: +16.87%.
- **`in_thong_tin_moi_truong`** · Cell 6 · def — Ghi lại phiên bản Python, thư viện, CPU và revision git của lần chạy (yêu cầu tái lập, Slide 41, 76).
- **`luong_loai_tru`** · Cell 9 · def — Bảng luồng loại trừ (eligibility flow, Slide 55): 4,424 -> loại 794 Enrolled -> 3,630, kèm kiểm tra chất lượng.
  - *Hỏi:* Có xử lý missing/outlier không? → *Đáp:* Không cần: 0 dòng trùng, 0 giá trị thiếu, 0 điểm ngoài miền hợp lệ; chỉ loại Enrolled.
- **`kiem_toan_thoi_diem_bien`** · Cell 11 · def — Bảng 1.6a: phân loại mọi cột theo thời điểm đo so với học bổng: biến nền (được làm Z) và biến sau nhập học (cấm điều chỉnh).
  - *Hỏi:* Sao không dùng điểm HK1/HK2 làm biến điều chỉnh, dù chúng dự báo rất tốt? → *Đáp:* Chúng xảy ra sau học bổng, là trung gian T -> M -> Y. Điều chỉnh theo chúng sẽ "ăn mất" hiệu ứng.
- **`ve_phan_bo_diem_vao`** · Cell 12 · def — Hình 1.1: phân bố điểm đầu vào với hai ngưỡng 120/140 của Z1, và tỷ lệ tốt nghiệp trong từng nhóm.
  - *Hỏi:* Ngưỡng 120/140 chọn thế nào? → *Đáp:* Chốt trước theo ý nghĩa thực tế, không dò trên dữ liệu; hình kiểm tra lại là 3 nhóm đều đủ mẫu (1,075 / 1,869 / 686).
- **`ve_mutual_information`** · Cell 13 · def — Hình 1.2: xếp hạng biến nền theo Mutual Information với Y; tô xanh biến đang dùng, đỏ biến học phí, xám biến không dùng.
  - *Hỏi:* Có chọn Z theo MI không? → *Đáp:* Không. Z chọn theo tiêu chuẩn cửa sau (thời điểm + ảnh hưởng cả T và Y); MI chỉ để kiểm tra lại.
- **`bang_overlap_theo_tang`** · Cell 14 · def — Bảng 1.6b: đếm số SV có/không học bổng trong 6 tầng (3 nhóm điểm vào x 2 nhóm tuổi) để kiểm tra overlap thô.
  - *Hỏi:* Overlap có đạt không? → *Đáp:* Ở mức 6 tầng thì đạt (tầng nhỏ nhất vẫn có 44 SV có học bổng). Khi chia đủ 5 biến thì có vài hồ sơ thiếu hỗ trợ, ghi ở cột Overlap_Status của Table B.

### Nhóm 02 — Chênh lệch quan sát thô và các hàm vẽ hình (mục 2, 3, 6, 8)
File: [Group02_ChenhLechTho_TrucQuan.py](Group02_ChenhLechTho_TrucQuan.py). Tính P(Y|T) thô (+34.54 điểm %), vẽ Hình 2.1, và các hàm vẽ đồ thị có hướng dùng chung cho Hình 3.1, 6.1, 8.1 cùng forest plot ATE (Hình 8.3).

- **`tinh_toan_do_chenh_lech_quan_sat_tho`** · Cell 17 · def — Tính chênh lệch quan sát thô: P(Y=1|T=1) - P(Y=1|T=0) = 86.17% - 51.63% = +34.54 điểm %.
  - *Hỏi:* Vậy cấp học bổng tăng tốt nghiệp 35% à? → *Đáp:* Không. Sau điều chỉnh nhiễu, ATE khoảng +24%; khoảng 10 điểm % là thiên lệch do nhiễu quan sát được.
- **`ve_ty_le_tot_nghiep_theo_nhom`** · Cell 18 · def — Hình 2.1: tỷ lệ tốt nghiệp theo học bổng (trái) và theo học bổng x học phí đúng hạn (phải), tô đỏ ô thưa n < 100.
  - *Hỏi:* 24.3% ở cột đỏ có tin được không? → *Đáp:* Là 9/37; CI 95% Wilson khoảng [13%, 40%]. Ô này cho thấy positivity yếu, không dùng làm ước lượng chính xác.
- **`ve_do_thi_co_huong`** · Cell 22 · def — Hàm vẽ đồ thị có hướng dùng chung cho Hình 3.1, 6.1, 8.1: vị trí nút cố định, tô đỏ / nét đứt các cạnh được chỉ định.
- **`vi_tri_theo_tang`** · Cell 31 · def — Tính tọa độ nút cho Hình 6.1 theo tầng thời gian: Z cột trái, T và M1 hạ xuống hàng dưới, Y bên phải.
- **`do_cong_canh_z_z`** · Cell 61 · def — Gán độ cong cho cạnh nối hai biến Z cùng cột ở Hình 8.1: cạnh kề nhau vẽ thẳng, cạnh cách xa vòng ra bên trái.
- **`ve_forest_ate_table_b`** · Cell 64 · def — Hình 8.3: forest plot ATE và CI 95% theo từng seed cho B0, H1, H2, kèm vạch chênh lệch thô.
  - *Hỏi:* Hình nói lên điều gì? → *Đáp:* Cả ba mô hình đều quanh +0.24, cách xa +0.345 thô: kết luận can thiệp không phụ thuộc mô hình.

### Nhóm 03 — Minh họa học cấu trúc và học tham số (mục 3, 4)
File: [Group03_HocCauTruc_HocThamSo.py](Group03_HocCauTruc_HocThamSo.py). HillClimb + BIC có / không ràng buộc (Hình 3.1) và so sánh CPD MLE với CPD BDeu.

- **`hoc_cau_truc_hill_climb_bic`** · Cell 21 · def — Minh họa (mục 3) HillClimb + BIC trên 4 biến, có hoặc không có ExpertKnowledge cấm cạnh.
  - *Hỏi:* Kết quả minh họa là gì? → *Đáp:* Học tự do ra cạnh ngược Y -> Z; cấm cạnh ra từ Y thì hết nhưng còn T -> Debtor vì chưa cấm T -> Z. Bài học: ràng buộc phải đủ Z -> T -> Y, như ở B0 mục 8.B.
- **`so_sanh_hoc_tham_so_mle_va_bayesian`** · Cell 25 · def — So sánh CPD của Y ước lượng bằng MLE (đếm tần suất) và bằng Bayes với tiên nghiệm BDeu (ESS = 10).
  - *Hỏi:* Vì sao dùng BDeu thay MLE? → *Đáp:* Ô thưa (ví dụ Debtor=1 và có học bổng) thì MLE dễ ra 0% hoặc 100%; BDeu làm mịn, tránh xác suất cực đoan.

### Nhóm 04 — Bốn kiểm tra hợp lệ của DAG và quy trình DoWhy trên Domain DAG (mục 5, 6, 7)
File: [Group04_KiemTraDAG_DoWhy.py](Group04_KiemTraDAG_DoWhy.py). Kiểm tra phi chu trình, phân tầng thời gian, danh sách cạnh cấm/bắt buộc, tiêu chuẩn cửa sau; chạy DoWhy 4 bước (model, identify, estimate, refute) trên Domain DAG.

- **`KetQuaKiemTraRangBuoc`** · Cell 28 · class — Dataclass kết quả của một bài kiểm tra: có hợp lệ không, tên bài kiểm tra, danh sách vi phạm.
- **`kiem_tra_tinh_khong_chu_trinh`** · Cell 28 · def — Kiểm tra đồ thị có phải DAG (không có chu trình) không; nếu không thì liệt kê chu trình.
- **`kiem_tra_phan_tang_thoi_gian`** · Cell 28 · def — Kiểm tra không có cạnh đi ngược thời gian: với mỗi cạnh u -> v, tầng(u) phải <= tầng(v).
  - *Hỏi:* Cạnh cùng tầng có bị cấm không? → *Đáp:* Không (điều kiện là >, không phải >=); chiều giữa các Z cùng tầng để dữ liệu/chuyên gia quyết.
- **`kiem_tra_danh_sach_chuyen_gia`** · Cell 28 · def — Kiểm tra blacklist (cạnh cấm không được có) và whitelist (cạnh bắt buộc phải có).
- **`kiem_tra_d_separation_backdoor`** · Cell 28 · def — Kiểm tra tập Z có thỏa tiêu chuẩn cửa sau không: chặn mọi đường cửa sau giữa T và Y.
  - *Hỏi:* Vì sao xóa cạnh đi ra từ T? → *Đáp:* Để chỉ còn lại các đường không nhân quả (cửa sau); đường nhân quả T -> ... -> Y không được chặn.
- **`KetQuaUocLuongDoWhy`** · Cell 35 · class — Dataclass kết quả ước lượng DoWhy: tên phương pháp, ATE, tập biến cửa sau.
- **`KetQuaKiemDinhRefute`** · Cell 35 · class — Dataclass kết quả một refuter: tên, hiệu ứng gốc, hiệu ứng sau bác bỏ, kết luận.
- **`thuc_thi_quy_trinh_can_thiep_dowhy`** · Cell 35 · def — Chạy đủ 4 bước DoWhy trên Domain DAG (mục 7): model -> identify -> estimate (2 cách) -> refute (3 cách).
  - *Hỏi:* Refuter qua thì chứng minh không có nhiễu ẩn à? → *Đáp:* Không. Refuter chỉ kiểm tra độ ổn định của bộ ước lượng; nhiễu ẩn được đánh giá bằng RV ở mục 9.H.

### Nhóm 05 — Cấu hình chung, mã hóa 7 biến và 5 split dùng chung (mục 8.A)
File: [Group05_CauHinh_DuLieu_Split.py](Group05_CauHinh_DuLieu_Split.py). Mọi siêu tham số của B0/H1/H2 nằm trong Config; dữ liệu được mã hóa thành Z1..Z5, T, Y; 5 split 70/15/15 phân tầng được lưu ra JSON và đóng băng.

- **`Config`** · Cell 41 · class — Dataclass "đóng băng" chứa MỌI cấu hình của mục 8: seed, tỷ lệ split, tên nút, thư mục, siêu tham số B0/H1/H2, tham số đánh giá.
  - *Hỏi:* Siêu tham số chọn thế nào? → *Đáp:* Hàm điểm B0, tau và lambda của H2, ngưỡng F1 đều chọn trên VALIDATION; phần còn lại cố định trước, không dò theo TEST.
- **`init_directories`** · Cell 41 · def — Tạo các thư mục artifacts/ (splits, tables, checkpoints, graphs) nếu chưa có.
- **`set_seed`** · Cell 41 · def — Cố định mọi nguồn ngẫu nhiên: random, numpy, torch (và CUDA nếu có), PYTHONHASHSEED.
- **`load_and_preprocess_uci697`** · Cell 42 · def — Nạp UCI 697 và mã hóa đúng 7 biến của mô hình chính: Z1..Z5, T, Y (mã hóa dùng cho Tables A–C).
- **`get_or_create_splits`** · Cell 43 · def — Tạo (hoặc nạp lại) 5 phân chia TRAIN/VAL/TEST 70/15/15 cho 5 seed, phân tầng theo tổ hợp (Y, T), lưu JSON.
  - *Hỏi:* Vì sao 5 seed? → *Đáp:* Slide 39 yêu cầu ít nhất 5 run ghép cặp; B0/H1/H2 dùng cùng split trong mỗi seed nên so sánh được theo cặp.

### Nhóm 06 — Mô hình B0: học DAG truyền thống + CPD BDeu + truy vấn can thiệp (mục 8.B)
File: [Group06_MoHinh_B0.py](Group06_MoHinh_B0.py). Ràng buộc thời gian, HillClimb với 3 hàm điểm chọn trên VALIDATION, khớp CPD BDeu, tính log-likelihood, và hàm truy vấn q_m(t, C_i) dùng chung cho B0/H1/H2.

- **`build_b0_dag`** · Cell 45 · def — Dựng Domain DAG tham chiếu 7 nút, 11 cạnh (mỗi Z -> T, mỗi Z -> Y, T -> Y).
- **`temporal_forbidden_edges`** · Cell 45 · def — Danh sách cạnh cấm theo thời gian, dùng chung cho B0 (ExpertKnowledge) và H2 (mặt nạ M).
  - *Hỏi:* Có cấm Z -> Z không? → *Đáp:* Không. Các Z cùng tầng; chiều giữa chúng không ảnh hưởng truy vấn do(T) vì đều là tiền can thiệp.
- **`fit_discrete_bayesian_model`** · Cell 45 · def — Khớp mọi CPD của một mạng Bayes rời rạc bằng ước lượng Bayes với tiên nghiệm BDeu (ESS mặc định 5).
- **`fit_bn_on_dag`** · Cell 45 · def — Từ một DAG (networkx) tạo DiscreteBayesianNetwork đủ 7 nút rồi khớp CPD BDeu trên TRAIN.
- **`validation_log_likelihood`** · Cell 45 · def — Log-likelihood của toàn mạng trên một tập dữ liệu: sum_i sum_v log P(x_v | pa(x_v)). Dùng để CHỌN trên VALIDATION.
  - *Hỏi:* Vì sao chọn hàm điểm bằng log-likelihood VAL chứ không phải bằng chính hàm điểm? → *Đáp:* Ba hàm điểm có thang khác nhau nên không so trực tiếp được; log-likelihood trên dữ liệu chưa thấy là thước đo chung, khách quan.
- **`learn_b0_dag`** · Cell 45 · def — Tìm sơ đồ mũi tên (DAG) cho B0 từ dữ liệu TRAIN. Vì có nhiều cách "chấm điểm" sơ đồ cho ra kết quả khác nhau, hàm thử 3 cách rồi để dữ liệu chưa thấy (VALIDATION) quyết định cách nào đúng hơn.
  - *Hỏi:* Vì sao không dùng luôn BIC cho đơn giản? → *Đáp:* Vì BIC bỏ mất Debtor -> Y, và trên dữ liệu chưa thấy (VAL) sơ đồ đó đoán kém hơn ở cả 5 seed. Chọn bằng VAL là để dữ liệu quyết định, không phải nhóm chọn theo ý thích.
  - *Hỏi:* Chọn trên VAL có làm rò rỉ TEST không? → *Đáp:* Không. Chỉ dùng TRAIN để học và VAL để chọn; TEST chỉ dùng một lần ở cuối cho Table A, B.
  - *Hỏi:* B0 có "train" không? → *Đáp:* Có, nhưng không bằng gradient: tìm sơ đồ bằng leo đồi, điền bảng bằng đếm có làm mịn BDeu; lựa chọn duy nhất là giám khảo, chọn trên VAL.
  - *Hỏi:* Sao phải ép T -> Y? Như vậy là áp đặt kết quả à? → *Đáp:* Không áp đặt độ lớn: cạnh bắt buộc chỉ để câu hỏi do(T) có nghĩa, còn tác dụng lớn hay nhỏ vẫn do dữ liệu. H2 không bị ép mà vẫn tự học ra cạnh này.
- **`save_graph_artifacts`** · Cell 45 · def — Lưu danh sách nút, cạnh và thông tin phụ (hàm điểm, tau, log H2...) ra artifacts/graphs/<tên>.json.
- **`query_model_effects`** · Cell 45 · def — Hàm truy vấn CHUNG cho B0/H1/H2: với mỗi hồ sơ C_i tính q(0, C_i), q(1, C_i) và xác suất dự báo P(Y=1 | T thật, Pa(Y)).
  - *Hỏi:* Sao không dùng Variable Elimination của pgmpy? → *Đáp:* Kết quả như nhau vì chỉ cần CPD của Y; tra bảng trực tiếp nhanh và minh bạch hơn (mục 8.2c tính tay lại để đối chiếu).

### Nhóm 07 — Mô hình H1: thay CPD của Y bằng mạng nơ-ron trên DAG của B0 (mục 8.C)
File: [Group07_MoHinh_H1.py](Group07_MoHinh_H1.py). MLP PyTorch ước lượng P(Y=1 | Pa(Y)), huấn luyện có early stopping, lưu checkpoint, rồi chuyển MLP thành bảng CPD để gắn lại vào mạng Bayes.

- **`NeuralCPDModule`** · Cell 47 · class — Mạng MLP ước lượng P(Y=1 | Pa(Y)) cho H1. Đầu ra là logit (chưa qua sigmoid).
  - *Hỏi:* Vì sao cần BatchNorm và Dropout? → *Đáp:* BatchNorm ổn định huấn luyện; Dropout và weight decay chống overfit. Thực tế đầu vào chỉ 16 tổ hợp nên mạng hội tụ rất nhanh.
- **`encode_h1_features`** · Cell 47 · def — Mã hóa ĐÚNG các nút cha của Y làm đầu vào MLP (Slide 60): biến 3 trạng thái -> one-hot 3 chiều, biến nhị phân giữ nguyên.
- **`h1_input_dim`** · Cell 47 · def — Số chiều đầu vào của MLP: cộng 3 cho mỗi biến 3 trạng thái, 1 cho mỗi biến nhị phân.
- **`train_h1_mlp`** · Cell 47 · def — Huấn luyện MLP của H1 trên TRAIN bằng BCE, theo dõi loss VAL sau mỗi epoch; mặc định có early stopping.
  - *Hỏi:* Bao nhiêu epoch? → *Đáp:* Tối đa 150; thực tế dừng sau 17–39 epoch, tốt nhất ở epoch 2–4 (seed 43 là 24) vì đầu vào chỉ có 16 tổ hợp.
  - *Hỏi:* Loss TRAIN cao hơn VAL là sao? → *Đáp:* Loss TRAIN đo khi dropout bật nên bị đội lên; loss VAL đo ở chế độ eval.
- **`save_h1_checkpoint`** · Cell 47 · def — Lưu trọng số và nhật ký H1 ra artifacts/checkpoints/h1_seed<seed><hậu tố>.pt.
- **`build_h1_bayesian_network`** · Cell 47 · def — Biến MLP thành bảng CPD của Y rồi gắn vào bản sao của B0: H1 = B0 với CPD(Y) do mạng nơ-ron sinh ra.
  - *Hỏi:* H1 khác B0 ở đâu? → *Đáp:* Cùng DAG, cùng CPD các nút khác; chỉ CPD của Y được học bằng MLP thay vì đếm BDeu.

### Nhóm 08 — Mô hình H2: học DAG bằng NOTEARS-MLP có mặt nạ M (mục 8.D)
File: [Group08_MoHinh_H2.py](Group08_MoHinh_H2.py). Mặt nạ cạnh cấm, mạng NOTEARS phi tuyến, tối ưu augmented Lagrangian, lọc ngưỡng tau thành DAG và chọn tau trên VALIDATION.

- **`build_domain_mask`** · Cell 49 · def — Tạo mặt nạ M (7x7) cho H2: M[i, j] = 0 nếu cạnh i -> j bị cấm (cùng danh sách với B0) và cấm tự nối.
- **`NotearsMLP`** · Cell 49 · class — Mạng NOTEARS phi tuyến: mỗi nút j có một MLP riêng f_j dự đoán X_j từ các nút khác; độ mạnh cạnh i -> j suy ra từ trọng số tầng 1.
- **`train_notears_mlp`** · Cell 49 · def — Tối ưu NOTEARS-MLP bằng augmented Lagrangian để vừa khớp dữ liệu vừa ép đồ thị không có chu trình.
  - *Hỏi:* H2 có epoch không? → *Đáp:* Không. Mỗi vòng ngoài gồm 300 bước Adam; cả 5 seed hội tụ sau 7–8 vòng với h khoảng 1e-7.
  - *Hỏi:* Vì sao sai số tái tạo tăng qua các vòng? → *Đáp:* Phạt chu trình càng nặng thì mô hình càng phải bỏ cạnh tạo vòng, chấp nhận khớp kém hơn: đánh đổi giữa khớp dữ liệu và là DAG hợp lệ.
- **`graph_from_weights`** · Cell 49 · def — Chuyển ma trận W thành DAG: giữ cạnh có |W| >= tau, cắt chu trình còn sót, rồi thêm cạnh bắt buộc T -> Y nếu thiếu.
- **`select_h2_graph`** · Cell 49 · def — Chọn ngưỡng tau của H2 trên VALIDATION: với mỗi tau, khớp CPD BDeu trên TRAIN và tính log-likelihood VAL, giữ tau tốt nhất.

### Nhóm 09 — Đánh giá dùng chung, vòng lặp 5 run ghép cặp, xuất Table A và B (mục 8.E)
File: [Group09_DanhGia_5Run_TablesAB.py](Group09_DanhGia_5Run_TablesAB.py). Chỉ số dự báo (ngưỡng chọn trên VAL), CI bootstrap của ATE, định danh DoWhy trên DAG của từng mô hình, kiểm tra overlap, vòng lặp 5 seed và tổng hợp Table A, Table B.

- **`evaluate_prediction_metrics`** · Cell 51 · def — Tính chỉ số dự báo trên TEST (Slide 74): AUROC, Macro-F1, Brier, log-loss, precision, recall.
  - *Hỏi:* Vì sao Macro-F1 của B0 và H1 trùng nhau? → *Đáp:* Pa(Y) chỉ có 16 cấu hình, hai mô hình xếp hạng giống nhau tại ngưỡng chọn, nên gán nhãn giống hệt; khác biệt chỉ ở xác suất (AUROC, Brier).
- **`compute_bootstrap_ate_ci`** · Cell 51 · def — CI 95% của ATE bằng percentile bootstrap: lấy mẫu lại có hoàn lại các hồ sơ C_test, GIỮ CỐ ĐỊNH mô hình.
  - *Hỏi:* CI hẹp (khoảng ±0.008) có nghĩa ATE rất chính xác? → *Đáp:* Không hẳn: mô hình được giữ cố định. Bất định do học lại mô hình thể hiện qua độ tản giữa 5 seed (Table C).
- **`identify_with_dowhy`** · Cell 51 · def — Chạy DoWhy identify_effect trên CHÍNH DAG của từng mô hình (Slide 57, 62) để xác nhận hiệu ứng định danh được.
- **`overlap_report`** · Cell 51 · def — Đếm hồ sơ C_test "không được hỗ trợ": trong TRAIN không có ai cùng cấu hình biến điều chỉnh ở CẢ T=0 và T=1.
  - *Hỏi:* Hồ sơ không được hỗ trợ thì xử lý sao? → *Đáp:* Được ghi rõ ở cột Overlap_Status (1–2% ở H2); ước lượng ở các hồ sơ đó dựa vào làm mịn BDeu.
- **`record_model_run`** · Cell 52 · def — Đánh giá một mô hình trong một seed: dự báo trên TEST, can thiệp chuẩn hóa trên C_test, CI, định danh, overlap; ghi vào hai danh sách kết quả.
- **`doan_ma_cell_52_vong_lap_5_run_ghep_cap`** · Cell 52 · đoạn mã — Vòng lặp chính: với mỗi seed 42..46, khớp B0, H1, H2 trên cùng split rồi đánh giá bằng record_model_run.
- **`mean_std`** · Cell 55 · def — Định dạng chuỗi "trung bình ± độ lệch chuẩn" cho dòng tổng hợp 5 seed của Tables A–C.
- **`doan_ma_cell_55_xuat_table_a_b`** · Cell 55 · đoạn mã — Xuất Table A (dự báo) và Table B (can thiệp): tính hiệu theo cặp so với B0 trong cùng seed, thêm dòng tổng hợp trung bình ± độ lệch chuẩn, ghi CSV.

### Nhóm 10 — Quá trình huấn luyện, kiểm tra overfit, có / không early stopping (mục 8.E2)
File: [Group10_HuanLuyen_Overfit.py](Group10_HuanLuyen_Overfit.py). Vẽ đường cong loss của H1 và hội tụ của H2 (Hình 8.2), bảng AUROC/log-loss trên TRAIN/VAL/TEST (Bảng 8.E2), huấn luyện H1 không dừng sớm để so sánh và lưu 2 checkpoint (Hình 8.2b, Bảng 8.E2b).

- **`ve_qua_trinh_huan_luyen`** · Cell 53 · def — Hình 8.2: (a) loss TRAIN/VAL của H1 theo epoch, chấm tại epoch tốt nhất; (b) h(W) của H2 theo vòng ngoài (thang log); (c) sai số tái tạo của H2.
- **`bang_overfit_theo_tap`** · Cell 53 · def — Bảng 8.E2: AUROC, accuracy (ngưỡng 0.5) và log-loss của B0/H1/H2 trên TRAIN, VAL, TEST cho cả 5 seed.
  - *Hỏi:* Có overfit không? → *Đáp:* Chênh AUROC TRAIN - TEST: B0 +0.002, H1 khoảng 0, H2 +0.008 (nhỏ hơn độ dao động giữa seed ±0.016) => không overfit đáng kể.
- **`so_sanh_early_stopping`** · Cell 53 · def — Huấn luyện thêm bản H1 KHÔNG early stopping (đủ 150 epoch) cho mỗi seed, lưu 2 checkpoint riêng và so sánh với bản có early stopping.
  - *Hỏi:* Bỏ early stopping thì sao? → *Đáp:* Loss VAL tăng (0.504 so với 0.500), AUROC TEST giảm nhẹ (0.799 so với 0.801): dấu hiệu bắt đầu overfit. ATE trung bình gần như không đổi nhưng từng seed lệch tới ±0.02.
- **`ve_early_stopping`** · Cell 53 · def — Hình 8.2b: loss TRAIN/VAL của bản H1 không dừng sớm qua 150 epoch, vạch dọc ở epoch bản có early stopping dừng, chấm ở trọng số được giữ.

### Nhóm 11 — Refuter (Table C), hiển thị bảng, positivity, đọc lại đồ thị và q_m minh bạch (mục 8.F, 8.2b, 8.2c)
File: [Group11_Refuter_BangKetQua.py](Group11_Refuter_BangKetQua.py). Refuter DoWhy trên DAG của B0/H2, kiểm định huấn luyện lại cho H1, đọc và hiển thị các bảng, bảng 4 ô học bổng x học phí, đọc DAG đã lưu cho Hình 8.1 và tính tay q_m(t, C_i).

- **`dowhy_refute_on_graph`** · Cell 57 · def — Hai refuter DoWhy (Slide 69–71) trên ĐÚNG DAG của B0 hoặc H2, dùng ước lượng hỗ trợ hồi quy tuyến tính, cho Table C.
- **`h1_retraining_refuter`** · Cell 57 · def — Kiểm định tương tự refuter cho H1 nơ-ron (vì refuter tuyến tính không áp dụng được): huấn luyện lại MLP trong 3 lần.
- **`doan_ma_cell_57_chay_refuter_va_table_c`** · Cell 57 · đoạn mã — Chạy 4 việc cho seed 42 rồi xuất Table C: (1) lấy sơ đồ B0, H2 và mạng B0 của seed 42; (2) refuter DoWhy trên B0 và H2; (3) refuter huấn luyện lại cho H1; (4) gom thành bảng Table C (độ bền vững + chi phí).
- **`doc_bang_csv`** · Cell 58 · def — Đọc một bảng kết quả từ CSV trong artifacts/tables/.
- **`bang_hien_thi`** · Cell 58 · def — Tạo BẢN SAO bảng để hiển thị, làm tròn 4 chữ số các ô số nằm trong cột kiểu chữ (cột trộn số với "mean ± std").
- **`bang_positivity_T1_T2`** · Cell 59 · def — Bảng 4 ô (học bổng x học phí đúng hạn) kèm tỷ lệ tốt nghiệp, để kiểm tra positivity của T3 và dấu hiệu nhân quả ngược của T2.
  - *Hỏi:* Bảng cho thấy gì? → *Đáp:* Ô có học bổng nhưng chưa đóng đúng hạn chỉ 37 SV (positivity yếu); nhóm không học bổng và chưa đóng đúng hạn chỉ 4.5% tốt nghiệp (gợi ý bỏ học -> ngừng đóng phí).
- **`doc_do_thi_da_luu`** · Cell 61 · def — Đọc file JSON đồ thị do vòng lặp 5 run đã lưu (artifacts/graphs/<tên>.json).
- **`mat_na_thoi_gian`** · Cell 61 · def — Dựng lại mặt nạ M để vẽ heatmap Hình 8.1(a): cấm tự nối, cấm Y -> mọi nút, cấm T -> Z.
- **`bang_tom_tat_do_thi`** · Cell 61 · def — Bảng tóm tắt đồ thị 5 seed: hàm điểm B0 được chọn, số cạnh, Pa(Y); tau, số cạnh, Pa(Y), h(W), hội tụ, có ép T -> Y không của H2.
- **`ma_hoa_7_bien`** · Cell 62 · def — Mã hóa lại 7 biến đúng như mục 8.A (từ dữ liệu cột gốc) để mục 8.2c dùng lại split và checkpoint đã lưu.
- **`tai_mlp_h1`** · Cell 62 · def — Dựng lại kiến trúc MLP của H1 và nạp trọng số từ checkpoint h1_seed42.pt (chế độ eval).
- **`ma_hoa_dau_vao_h1`** · Cell 62 · def — Mã hóa Pa(Y) như H1 (Z1 one-hot 3 chiều, biến nhị phân giữ nguyên) cho phần tính tay ở 8.2c.
- **`q_b0`** · Cell 62 · def — Tính tay q_B0(t, C_i): tra trực tiếp P(Y=1 | T=t, cha khác = giá trị của hồ sơ) trong bảng CPD của B0.
- **`q_h1`** · Cell 62 · def — Tính tay q_H1(t, C_i): đặt T = t, mã hóa Pa(Y), cho MLP dự đoán và qua sigmoid.

### Nhóm 12 — Phân tích mở rộng: kết cục khác, AIPW, trung gian, CATE, độ nhạy RV (mục 9)
File: [Group12_PhanTichMoRong_Muc9.py](Group12_PhanTichMoRong_Muc9.py). Mục 9 viết dạng tuần tự (không bọc hàm) cho 9.B–9.G; ở file này mỗi đoạn được bọc thành một hàm chỉ để đọc. 9.H có hai hàm thật: tính Robustness Value và partial R^2 làm mốc.

- **`AdvancedConfig`** · Cell 69 · class — Cấu hình riêng cho mục 9: đường dẫn dữ liệu, seed 42, 1000 bootstrap và 7 hiệp biến (thêm Displaced, Gender so với mục 8).
- **`doan_ma_cell_69_nap_hai_quan_the`** · Cell 69 · đoạn mã — Mục 9.A: tạo hai quần thể: df_completed (3,630 SV có kết cục) và df_full (4,424 SV, gồm Enrolled) với kết cục giữ chân Y_retain = 1 nếu không bỏ học.
  - *Hỏi:* Vì sao cần quần thể 4,424? → *Đáp:* Để kiểm tra kết luận không phụ thuộc việc loại Enrolled (mục 9.D).
- **`doan_ma_cell_71_diem_hk2_ols`** · Cell 71 · đoạn mã — Mục 9.B: hiệu ứng của học bổng lên điểm trung bình HK2 bằng hồi quy OLS Y_grade ~ T + 7 hiệp biến.
- **`doan_ma_cell_73_ty_le_dat_tin_chi_ols`** · Cell 73 · đoạn mã — Mục 9.C: hiệu ứng lên tỷ lệ đạt tín chỉ HK2 = số tín chỉ đạt / số đăng ký (mẫu số tối thiểu 1 để tránh chia 0), OLS với 7 hiệp biến.
- **`doan_ma_cell_75_giu_chan_aipw`** · Cell 75 · đoạn mã — Mục 9.D: hiệu ứng lên tỷ lệ giữ chân trên 4,424 SV bằng AIPW (doubly robust).
  - *Hỏi:* Vì sao +16.87% nhỏ hơn +24%? → *Đáp:* Kết cục khác (giữ chân gồm cả đang học) và quần thể khác; điều quan trọng là vẫn dương rõ.
- **`doan_ma_cell_77_trung_gian_nde_nie`** · Cell 77 · đoạn mã — Mục 9.E: tách hiệu ứng thành trực tiếp (NDE) và gián tiếp qua tín chỉ HK1 (NIE) bằng phương pháp tích hệ số.
  - *Hỏi:* 21.5% nghĩa là gì? → *Đáp:* Khoảng một phần năm tác động của học bổng đi qua việc học tốt hơn ở HK1; phần lớn còn lại là trực tiếp (tài chính, động lực...).
- **`doan_ma_cell_79_cate_theo_nhom`** · Cell 79 · đoạn mã — Mục 9.F: CATE theo 6 nhóm định trước (tuổi, Debtor, nơi ở) bằng G-computation logistic khớp riêng trong từng nhóm.
- **`KetQuaDoNhayCinelli`** · Cell 83 · class — Dataclass kết quả độ nhạy: hệ số của T, t-stat, bậc tự do, partial R^2 của T với Y, RV và RV ở mức alpha 0.05.
- **`tinh_rv_cinelli_hazlett`** · Cell 83 · def — Tính Robustness Value (Cinelli & Hazlett 2020) từ hồi quy OLS Y ~ T + Z1..Z5: một nhiễu ẩn phải mạnh cỡ nào mới xóa được hiệu ứng.
  - *Hỏi:* 21% là lớn hay nhỏ? → *Đáp:* So với biến quan sát mạnh nhất là học phí (partial R^2 với Y 10.4%): kết luận vững trước nhiễu ẩn yếu hơn học phí, nhưng không miễn nhiễm với nhiễu mạnh gấp đôi.
- **`benchmark_partial_r2`** · Cell 83 · def — Partial R^2 của từng biến nền với Y (khi đã có T và các biến nền khác), làm mốc so sánh cho RV.

### Nhóm 13 — Mô phỏng phân bổ học bổng theo CATE (mục 10)
File: [Group13_PhanBoChinhSach_Muc10.py](Group13_PhanBoChinhSach_Muc10.py). Mã hóa biến nền và so sánh 4 cách phân bổ cùng số suất học bổng bằng T-learner logistic (in-sample).

- **`ma_hoa_z_chinh_sach`** · Cell 86 · def — Mã hóa 5 biến nền cho mô phỏng phân bổ (cùng mã hóa với mục 9.H), kèm T và Y.
- **`danh_gia_chinh_sach`** · Cell 86 · def — So sánh 4 cách phân bổ CÙNG số suất học bổng (969): theo CATE, thực tế, ưu tiên Debtor, ngẫu nhiên; ước lượng tỷ lệ tốt nghiệp dự kiến.
  - *Hỏi:* +164 SV có đáng tin không? → *Đáp:* Là ước lượng mô hình, không phải quan sát; cần cross-fitting và đánh giá off-policy (IPW/DR) mới chắc.

## Luồng gọi chính của mục 8 (B0 / H1 / H2)

```text
load_and_preprocess_uci697 -> get_or_create_splits            (Group05)
cho mỗi seed 42..46 (doan_ma_cell_52_vong_lap_5_run_ghep_cap, Group09):
  B0: learn_b0_dag -> fit_bn_on_dag                          (Group06)
  H1: train_h1_mlp -> build_h1_bayesian_network              (Group07, dùng DAG của B0)
  H2: train_notears_mlp -> select_h2_graph -> fit_bn_on_dag  (Group08)
  mỗi mô hình: record_model_run -> query_model_effects       (Group09, Group06)
               -> evaluate_prediction_metrics / compute_bootstrap_ate_ci / identify_with_dowhy / overlap_report
xuất Table A, B (Group09) -> 8.E2 overfit (Group10) -> refuter, Table C (Group11)
```
