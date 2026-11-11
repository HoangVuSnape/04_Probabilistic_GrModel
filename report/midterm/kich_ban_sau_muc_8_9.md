### Cell 37 · markdown · 8.1 Thiết kế B0 / H1 / H2 + 8.2 Multi-treatment

**🧩 Cell này là gì:** cell "bản đồ" của cả mục 8: mô tả ba mô hình bắt buộc và phần khám phá 8.2. Chưa có code chạy.

**🎤 Nói:**
"Mục 8 là phần chính của đồ án. Một mạng Bayes gồm hai phần: **sơ đồ mũi tên** và **bảng xác suất** cho từng nút. Ba mô hình khác nhau ở chỗ ai làm phần nào:
* **B0** — mạng Bayes truyền thống: HillClimb học sơ đồ trên TRAIN, BDeu điền bảng.
* **H1** — giữ nguyên sơ đồ của B0, chỉ thay **bảng của tốt nghiệp** bằng một mạng nơ-ron.
* **H2** — để mạng nơ-ron NOTEARS học **sơ đồ**, bảng vẫn điền bằng BDeu như B0.

Mỗi lần so sánh chỉ đổi đúng một thứ: H1 so với B0 là đổi cách tính xác suất; H2 so với B0 là đổi cách vẽ sơ đồ. Cả ba dùng cùng 7 biến, cùng dữ liệu, cùng luật cấm trong mỗi lần chạy.

Mục 8.2 là phần khám phá: ngoài học bổng, em thử thêm can thiệp *học phí đúng hạn* và *kết hợp cả hai*. Em giới thiệu ở đây, còn kết quả trình bày sau Tables A–C."

### Cell 41 · code · 8.A Config — mọi thiết lập ở một chỗ

**🧩 Cell này làm gì:** tạo một đối tượng `Config` "đóng băng" (không sửa được sau khi tạo) chứa **mọi** thiết lập của mục 8, và hàm `set_seed` cố định mọi nguồn ngẫu nhiên.

**🎤 Nói:**
"Toàn bộ thiết lập nằm ở một chỗ để không có con số nào 'ẩn' trong code. 5 seed 42–46, chia 70/15/15. B0 dùng ESS = 5 và thử ba hàm điểm. H1: 32 nơ-ron, dropout 0.2, tối đa 150 epoch, dừng sớm sau 15 epoch không tiến bộ. H2: lưới thử $\lambda_1$ và ngưỡng $\tau$, dừng khi ràng buộc vòng tròn dưới $10^{-6}$. Những gì có lưới thử đều được chọn trên VALIDATION."

**📊 Đọc output:** chỉ một dòng "Khởi tạo thành công…" — cell này không tính gì.

### Cell 43 · code · 8.A Năm cách chia dữ liệu dùng chung

**🧩 Cell này làm gì:**
1. Tạo "tầng" = tổ hợp (tốt nghiệp, học bổng), để mỗi phần có tỷ lệ giống nhau.
2. Chia 70% TRAIN / 30% còn lại, rồi chia đôi 30% thành VAL và TEST — mỗi lần đều giữ đúng tỷ lệ theo tầng.
3. Lưu chỉ số ra `artifacts/splits/split_seed_<s>.json`; lần sau đọc lại file thay vì chia mới (đóng băng).
4. Kiểm tra ba phần không trùng nhau và phủ đủ 3,630 dòng.

**🎤 Nói:**
"Mỗi seed cho một cách chia: 2,541 TRAIN, 544 VAL, 545 TEST. Chia phân tầng theo cả tốt nghiệp lẫn học bổng nên ba phần có tỷ lệ giống nhau. Trong mỗi seed, B0, H1, H2 dùng **đúng cùng** ba phần này — gọi là chạy ghép cặp."

**📊 Đọc output:** 5 dòng giống nhau (2541 / 544 / 545) — đúng, vì cùng tỷ lệ chia; chỉ khác *người nào* rơi vào phần nào.

### Cell 45 · code · 8.B B0 — học sơ đồ bằng HillClimb, chọn giám khảo trên VALIDATION

**🧩 Cell này làm gì (từng bước):**
1. **Đặt luật trước khi học:** cấm mọi mũi tên đi ra từ tốt nghiệp và mọi mũi tên học bổng → biến nền; bắt buộc học bổng → tốt nghiệp.
2. **Ba "giám khảo" chấm sơ đồ:** BIC, BDeu (ESS = 5), K2. Với mỗi giám khảo, HillClimb thử thêm / bớt / đảo từng mũi tên, giữ thay đổi làm điểm tăng → được 3 sơ đồ.
3. **Thi lại trên đề mới:** mỗi sơ đồ được điền bảng trên TRAIN, rồi đo xem đoán 544 sinh viên VAL tốt cỡ nào (log-likelihood, càng gần 0 càng tốt).
4. Chọn sơ đồ đoán VAL tốt nhất; kiểm tra không có vòng tròn và không có mũi tên cấm.
5. Định nghĩa hàm `query_model_effects`: tra bảng xác suất của tốt nghiệp để tính $q(0, C_i)$, $q(1, C_i)$ cho từng hồ sơ TEST. Hàm này dùng chung cho cả B0, H1, H2.
6. Chạy thử trên seed 42.

**🎤 Nói:**
"Ba giám khảo khác nhau ở mức khó tính với mũi tên mới: BIC phạt nặng, K2 dễ hơn. Không có lý thuyết nào nói trước giám khảo nào đúng, nên em để VALIDATION quyết định. Ở seed 42, BIC và BDeu cùng ra 9 mũi tên và bỏ cạnh Debtor → tốt nghiệp; K2 ra 11 mũi tên, giữ cạnh đó, và đoán VAL tốt hơn: −2208.37 so với −2211.11. K2 thắng ở cả 5 seed, nghĩa là Debtor thật sự giúp dự đoán. Cha của tốt nghiệp là học bổng, Debtor, học phí, tuổi. ATE trên TEST seed 42: 0.7863 − 0.5397 = **+0.2467**."

**📊 Đọc output:**
| Dòng | Nghĩa |
| :--- | :--- |
| Bảng 3 ứng viên | Score = giám khảo; Edges = số mũi tên; Val_LogLik = điểm thi lại (gần 0 hơn là tốt hơn); Pa(Y) = cha của tốt nghiệp |
| "Chọn hàm điểm = K2 \| Cạnh học được …" | Sơ đồ cuối cùng của B0, seed 42 |
| Risk(do(T=0)) = 0.5397 | Nếu **không ai** có học bổng: 54% tốt nghiệp |
| Risk(do(T=1)) = 0.7863 | Nếu **ai cũng** có học bổng: 79% tốt nghiệp |

### Cell 47 · code · 8.C H1 — thay bảng xác suất của tốt nghiệp bằng mạng nơ-ron

**🧩 Cell này làm gì (từng bước):**
1. Lấy **đúng sơ đồ của B0**; đầu vào của mạng chỉ là **cha của tốt nghiệp** (học bổng, Debtor, học phí, tuổi — Slide 60).
2. Mạng: 4 đầu vào → 32 nơ-ron (BatchNorm, ReLU, Dropout 0.2) → 16 nơ-ron → 1 đầu ra; tổng **769 tham số**.
3. Huấn luyện trên TRAIN bằng BCE, Adam (lr 0.005), batch 64; sau mỗi epoch đo loss trên VAL; 15 epoch không tiến bộ thì dừng và lấy lại trọng số tốt nhất.
4. **Đổi mạng thành bảng:** liệt kê cả 16 tổ hợp cha, cho mạng đoán xác suất từng tổ hợp, ghi vào một bảng CPD.
5. Gắn bảng này vào bản sao của B0 (các bảng khác giữ nguyên), kiểm tra mô hình hợp lệ, rồi tính ATE bằng **cùng hàm** như B0.

**🎤 Nói:**
"H1 giữ nguyên sơ đồ của B0, chỉ thay bảng xác suất của tốt nghiệp bằng một mạng 769 tham số. Sau khi học, mạng được 'đổi ngược' thành bảng 16 ô và gắn lại vào mạng Bayes, nên cách tính hiệu ứng giống hệt B0 — chỉ khác cách ra con số trong bảng. Seed 42: ATE **+0.2453**, gần bằng B0."

**📊 Đọc output:**
| Dòng | Nghĩa |
| :--- | :--- |
| Pa(Y) = [T, Z2, Z3, Z5] | Đầu vào = học bổng, Debtor, học phí, tuổi (4 biến nhị phân) |
| params = 769 | $(4 \times 32 + 32) + 64 + (32 \times 16 + 16) + (16 + 1)$ |
| best epoch = 2 | Đếm **từ 0**, tức là **epoch thứ 3** |
| ATE +0.2453 | Gần bằng B0 (+0.2467) |

### Cell 49 · code · 8.D H2 — chọn $\lambda_1$, chọn ngưỡng $\tau$, kiểm tra trên seed 42

**🧩 Cell này làm gì (từng bước):**
1. Tạo **mặt nạ M** từ đúng danh sách cấm của B0, cộng thêm cấm tự nối.
2. Mạng NOTEARS-MLP: mỗi biến có một mạng nhỏ đoán biến đó từ các biến khác; độ mạnh mũi tên $W_{ij}$ lấy từ trọng số lớp đầu.
3. Tối ưu: đoán đúng + phạt nhiều mũi tên ($\lambda_1$) + phạt vòng tròn $h(W)$ tăng dần; mỗi vòng ngoài 300 bước, dừng khi $h < 10^{-6}$.
4. **Chọn $\lambda_1$** trong {$10^{-3}$, $10^{-4}$} trên VAL của seed 42, rồi giữ cố định.
5. **Chọn $\tau$** trong {0.1, 0.2, 0.3, 0.5, 0.8}: lọc mũi tên ≥ $\tau$, điền bảng BDeu trên TRAIN, chấm trên VAL.
6. Điền bảng BDeu như B0 và tính ATE.

**🎤 Nói:**
"Bảng đầu là chọn mức phạt số mũi tên: $10^{-4}$ thắng (−2251 so với −2277) nên giữ cho mọi seed. Bảng thứ hai là chọn ngưỡng: $\tau = 0.1$ giữ 8 mũi tên và đoán VAL tốt nhất. Thầy nhìn cột T->Y_forced: ở ngưỡng được chọn là False, nghĩa là mũi tên học bổng → tốt nghiệp được **tự học ra**, không phải ép; chỉ khi ngưỡng quá cao (0.8) chỉ còn 1 mũi tên mới phải ép. Ràng buộc vòng tròn $3.3 \times 10^{-7}$ — đã hội tụ; trọng số ở ô bị cấm đúng bằng 0. ATE **+0.2428**."

**📊 Đọc output:**
| Cột / dòng | Nghĩa |
| :--- | :--- |
| `lambda_l1`, `best_Val_LogLik` | Mức phạt số mũi tên và điểm thi lại tương ứng |
| `tau`, `Edges` | Ngưỡng và số mũi tên còn lại (0.1 → 8, 0.8 → 1) |
| `Cut_for_cycle` = [] | Không phải cắt mũi tên nào để phá vòng |
| `T->Y_forced` | True = phải ép thêm học bổng → tốt nghiệp |
| `Pa(Y)` ở τ = 0.1 | H2 thêm **điểm vào (Z1)** làm cha của tốt nghiệp, khác B0 |
| `max |W| ở ô bị cấm = 0.0` | Mặt nạ hoạt động đúng |

### Cell 52 · code · 8.E Vòng lặp 5 lần chạy ghép cặp

**🧩 Cell này làm gì:** với mỗi seed 42–46: lấy đúng split của seed đó → học B0 → học H1 trên sơ đồ B0 → học H2 → với mỗi mô hình gọi `record_model_run` để (a) chấm dự báo trên TEST bằng nhãn thật, (b) tính ATE trên hồ sơ TEST **không dùng nhãn**, (c) CI bootstrap, (d) chạy DoWhy identify trên sơ đồ của chính mô hình, (e) kiểm tra overlap. Mọi đồ thị, trọng số, checkpoint được lưu ra `artifacts/`.

**🎤 Nói:**
"Đây là vòng lặp chính. Mỗi dòng tóm tắt một seed: sơ đồ B0 luôn do K2 chọn, 10 đến 12 mũi tên; H2 luôn hội tụ, ngưỡng 0.1 hoặc 0.2. ATE của cả ba mô hình ở mọi seed đều nằm trong khoảng **0.22 đến 0.25**."

**📊 Đọc output:**
| Seed | B0 | H1 | H2 |
| :--- | :--- | :--- | :--- |
| 42 | +0.247 | +0.245 | +0.243 |
| 43 | +0.247 | +0.239 | +0.240 |
| 44 | +0.238 | +0.218 | +0.230 |
| 45 | +0.238 | +0.242 | +0.228 |
| 46 | +0.239 | +0.243 | +0.239 |

### Cell 53 · code · 8.E2 Quá trình huấn luyện, overfit, có / không early stopping

**🧩 Cell này làm gì:**
1. Vẽ **Hình 8.2**: (a) loss TRAIN/VAL của H1 qua từng epoch, chấm tại epoch tốt nhất; (b) $h(W)$ của H2 qua từng vòng (thang log); (c) sai số tái tạo của H2.
2. **Bảng 8.E2**: AUROC, accuracy, log-loss của B0/H1/H2 trên **cả** TRAIN, VAL, TEST.
3. Huấn luyện thêm bản H1 **không early stopping** (đủ 150 epoch), lưu 2 checkpoint riêng, so sánh (**Hình 8.2b, Bảng 8.E2b**).

**🎤 Nói:**
"Trước khi xem kết quả, em cho thầy thấy quá trình huấn luyện ổn. Loss VAL của H1 chạm đáy ở epoch 2–4 (seed 43 là 24) nên dừng sau 17–39 epoch. H2 đưa ràng buộc vòng tròn về khoảng $10^{-7}$ sau 7–8 vòng. AUROC trên TRAIN, VAL, TEST gần như bằng nhau — chênh TRAIN trừ TEST: B0 +0.002, H1 khoảng 0, H2 +0.008 — nên không overfit đáng kể. Bỏ early stopping thì loss VAL tăng (0.504 so với 0.500) và AUROC TEST giảm nhẹ — đúng dấu hiệu bắt đầu overfit."

**📊 Đọc output:**
| Chỗ | Nghĩa |
| :--- | :--- |
| Hình 8.2a, nét liền / nét đứt | Loss TRAIN / VAL; chấm tròn = trọng số được giữ |
| Hình 8.2b | $h(W)$ giảm đều theo thang log, xuống dưới vạch $10^{-6}$ |
| Hình 8.2c | Sai số tái tạo **tăng** nhẹ — cái giá của việc ép thành DAG |
| Cột `AUROC_gap_TRAIN−TEST` | Thước đo overfit; càng gần 0 càng tốt |
| Bảng 8.E2b | So hai bản H1; ATE trung bình gần như không đổi (0.237 / 0.236) |

### Cell 57 · code · 8.F Refuter và chi phí — tạo Table C

**🧩 Cell này làm gì:**
1. Trên **sơ đồ của B0** và **sơ đồ của H2** (seed 42, TRAIN): chạy DoWhy với ước lượng hồi quy tuyến tính, rồi 2 phép thử phá, mỗi phép 30 lần:
   * **Placebo:** xáo trộn ngẫu nhiên cột học bổng → hiệu ứng phải về khoảng 0.
   * **Subset:** chỉ dùng 80% dữ liệu → hiệu ứng phải gần như cũ.
2. **H1** không chạy được refuter tuyến tính, nên tự làm phép thử tương đương: xáo học bổng rồi **huấn luyện lại mạng**; và huấn luyện lại trên 80% TRAIN (3 lần).
3. Gom độ tản ATE giữa 5 seed và thời gian chạy thành Table C.

**🎤 Nói:**
"Phép placebo xáo trộn ngẫu nhiên ai có học bổng: nếu mô hình đúng, 'học bổng giả' không được có tác dụng — và thực tế về +0.002 cho B0, H2, +0.018 cho H1. Phép subset bỏ 20% dữ liệu: hiệu ứng vẫn khoảng 0.22–0.24. Cả ba mô hình đều qua."

**📊 Đọc output:** cell chỉ in "đã lưu"; kết quả nằm ở Table C (cell sau).

### Cell 58 · code · Hiển thị Tables A, B, C, H3, Multi-Treatment

**🧩 Cell này làm gì:** đọc các bảng CSV vừa lưu và hiển thị (số làm tròn 4 chữ số; CSV giữ đủ độ chính xác).

**🎤 Nói:**
"**Table A — dự báo trên TEST**: AUROC B0 0.800, H1 0.801, H2 0.803. Hiệu ghép cặp H1 trừ B0 là +0.002 ± 0.003, H2 trừ B0 là +0.004 ± 0.004 — nhỏ hơn độ dao động giữa các lần chạy, nên mạng nơ-ron **không cải thiện rõ** dự báo. Đây là null finding hợp lệ.

**Table B — can thiệp**: ATE B0 +0.242 ± 0.005, H1 +0.237 ± 0.011, H2 +0.236 ± 0.007. **Kết luận không đổi giữa ba mô hình.** So với 34.5 điểm thô, khoảng 10 điểm là thiên lệch do khác biệt sẵn có.

**Table C — độ vững**: độ tản ATE giữa 5 seed 0.009 (B0), 0.028 (H1), 0.015 (H2); thời gian B0 dưới 1 giây, H2 chậm nhất."

**📊 Đọc output:**
| Bảng / cột | Nghĩa |
| :--- | :--- |
| A · `Macro_F1` B0 = H1 | Bảng chỉ 16 tình huống nên hai mô hình gán nhãn giống hệt; khác nhau ở xác suất (AUROC, Brier) |
| A · `Delta_*_vs_B0` | Hiệu theo cặp trong cùng seed |
| B · `Adjustment_Set` | Biến dùng để điều chỉnh = cha của tốt nghiệp trừ học bổng: {Z2, Z3, Z5}; H2 thêm Z1 |
| B · `DoWhy_Backdoor_Set` = {Z3, Z5} | Tập **tối thiểu** DoWhy tìm được; Debtor không cần cho định danh vì đường qua nó đã bị chặn ở học phí, nhưng thêm vào giúp ước lượng chính xác hơn |
| B · `Overlap_Status` | H2: 5–12 hồ sơ TEST (1–2%) thiếu dữ liệu so sánh, ở đó dựa vào làm mịn BDeu |
| B · `CI_95` | Khoảng ±0.008 — chỉ phản ánh việc chọn 545 hồ sơ, mô hình giữ cố định |
| H3 | Phụ lục, script riêng, dùng Domain DAG cố định — không so số với A–C |
| Multi-Treatment | Học bổng +0.21 đến +0.23 qua 4 bộ ước lượng; học phí và kết hợp: khám phá (cell sau) |

### Cell 59 · code · 8.2b Positivity và dấu hiệu nhân quả ngược của học phí

**🧩 Cell này làm gì:** đếm số sinh viên và tỷ lệ tốt nghiệp trong 4 ô (học bổng × học phí đúng hạn).

**🎤 Nói:**
"Mục 8.2 thử can thiệp *học phí đúng hạn* và ra +45 đến +51% — quá lớn để tin. Bảng 4 ô giải thích: người không học bổng và chưa đóng đúng hạn chỉ **4.5%** tốt nghiệp — giống 'bỏ học rồi ngừng đóng phí' hơn là 'nợ phí nên bỏ học', tức có thể là nhân quả ngược. Còn ô vừa có học bổng vừa chưa đóng đúng hạn chỉ **37 sinh viên** — điều kiện dương yếu, ước lượng ở đó dựa nhiều vào ngoại suy. Vì vậy em chỉ kết luận về học bổng."

**📊 Đọc output:** 449 SV (4.5%) · 2,212 SV (61.2%) · **37 SV (24.3%)** · 932 SV (88.6%).

### Cell 61 · code · Hình 8.1 — B0 và H2 học được sơ đồ gì

**🧩 Cell này làm gì:** đọc lại sơ đồ đã lưu của 5 seed → bảng tóm tắt; vẽ Hình 8.1: (a) mặt nạ M, (b) ma trận $|W|$ của H2 (khung đỏ = mũi tên được giữ), (c) sơ đồ B0, (d) sơ đồ H2 cùng bố cục.

**🎤 Nói:**
"Bảng tóm tắt cho thấy B0 rất ổn định: K2 ở cả 5 seed, 10–12 mũi tên, cha của tốt nghiệp luôn là học bổng, Debtor, học phí, tuổi. H2 thưa hơn, 7–9 mũi tên, hội tụ mọi seed, không seed nào phải ép học bổng → tốt nghiệp, và ở 4/5 seed thêm điểm vào làm cha của tốt nghiệp.

Trên hình: B0 học thêm chuỗi mũi tên **giữa các biến nền**; H2 gần như chỉ giữ mũi tên vào học bổng và tốt nghiệp. Chiều giữa các biến nền đổi theo mô hình và seed — dữ liệu không xác định được chiều giữa các biến cùng thời điểm, nhưng điều đó không ảnh hưởng truy vấn về học bổng vì chúng đều có trước học bổng."

**📊 Đọc output:**
| Cột | Nghĩa |
| :--- | :--- |
| `B0_score(val)` | Giám khảo được chọn (K2 cả 5 seed) |
| `H2_tau(val)` | Ngưỡng được chọn (0.1 / 0.2) |
| `H2_T->Y_forced` | False cả 5 seed → cạnh được tự học |
| `H2_max|W|_forbidden` | 0 cả 5 seed → mặt nạ tuân thủ |

### Cell 62 · code · Tính tay $q_m(t, C_i)$ để thấy công thức minh bạch

**🧩 Cell này làm gì:** dựng lại B0 và H1 seed 42 từ file đã lưu, chọn vài hồ sơ TEST, tự tra bảng / tự cho mạng đoán: xác suất tốt nghiệp nếu đặt học bổng = 0 và = 1. Rồi lấy trung bình trên 545 hồ sơ.

**🎤 Nói:**
"Cell này mở 'hộp đen': với mỗi hồ sơ, em đặt học bổng bằng 0 rồi bằng 1, giữ nguyên hoàn cảnh, và đọc xác suất tốt nghiệp. Ví dụ hồ sơ không nợ, đóng đúng hạn, dưới 20 tuổi: B0 cho 73.6% nếu không học bổng và 91.3% nếu có. Lấy trung bình 545 hồ sơ ra đúng ATE của Table B: +0.2467 cho B0, +0.2453 cho H1. Không dùng nhãn TEST ở bước nào."

**📊 Đọc output:**
| Cột | Nghĩa |
| :--- | :--- |
| Z2, Z3, Z5 | Hoàn cảnh của hồ sơ (Debtor, học phí, tuổi) |
| `q_B0(0,C_i)`, `q_B0(1,C_i)` | Xác suất tốt nghiệp nếu không / có học bổng, theo B0 |
| `q_H1(…)` | Tương tự theo H1 — gần B0 vì cùng sơ đồ |
| Dòng risk / ATE | Trung bình trên 545 hồ sơ = công thức Slide 59, 61 |

### Cell 64 · code · Hình 8.3 — forest plot ATE theo seed

**🧩 Cell này làm gì:** vẽ ATE và CI 95% của từng seed cho B0, H1, H2, kèm vạch chênh lệch thô.

**🎤 Nói:**
"Mỗi chấm là ATE của một mô hình ở một seed, thanh ngang là CI. Mọi chấm đều quanh +0.24 và cách xa vạch xám +0.345 của chênh lệch thô. Nhìn một hình là thấy: dù mô hình nào, seed nào, kết luận vẫn như nhau."

### Cell 69 · code · 9.A Hai quần thể cho phần mở rộng

**🧩 Cell này làm gì:** tạo hai bảng dữ liệu riêng cho mục 9:
* **3,630 SV có kết cục** (như mục 8) — cho điểm HK2, tín chỉ, trung gian, CATE.
* **4,424 SV toàn trường** (kể cả đang học) với kết cục *giữ chân* = 1 nếu không bỏ học — cho mục 9.D.

Khác mục 8: dùng biến gốc (tuổi, điểm vào dạng số) và **7 hiệp biến** (thêm Displaced, Gender), ước lượng bằng hồi quy thay vì mạng Bayes.

**🎤 Nói:**
"Mục 9 hỏi rộng hơn: học bổng giúp ở những mặt nào, qua đường nào, giúp ai nhiều hơn, và kết luận có dễ bị lật không. Em dùng các công cụ thống kê nhân quả chuẩn, độc lập với mạng Bayes, để kiểm tra chéo."

### Cell 71 · code · 9.B Kết cục 1 — điểm trung bình HK2

**🧩 Cell này làm gì:** hồi quy tuyến tính: điểm HK2 ~ học bổng + 7 hiệp biến; hệ số của học bổng = ATE (thang 20).

**🎤 Nói:**
"Học bổng làm điểm HK2 tăng **+1.48 điểm** trên thang 20, CI 1.09 đến 1.86. Chênh lệch thô là 2.63, nên sau điều chỉnh còn khoảng một nửa — lại là dấu vết của khác biệt sẵn có."

**📊 Đọc output:** ATE +1.478; CI [1.093, 1.863]; p ≈ $10^{-13}$; $R^2$ = 0.17 (mô hình giải thích 17% độ biến thiên điểm — bình thường với dữ liệu người).

### Cell 73 · code · 9.C Kết cục 2 — tỷ lệ đạt tín chỉ HK2

**🧩 Cell này làm gì:** tỷ lệ = tín chỉ đạt / tín chỉ đăng ký ở HK2 (mẫu số tối thiểu 1 để tránh chia cho 0), rồi hồi quy như 9.B.

**🎤 Nói:**
"Tỷ lệ đạt tín chỉ tăng **+14.5 điểm %**, CI 11.8 đến 17.2. Tỷ lệ sinh viên đạt hết mọi môn đăng ký: 58% ở nhóm có học bổng so với 34% nhóm không — đây là số thô, chưa điều chỉnh."

### Cell 75 · code · 9.D Kết cục 3 — giữ chân toàn trường bằng AIPW

**🧩 Cell này làm gì (từng bước):**
1. **Mô hình xác suất được học bổng** $e(x)$: hồi quy logistic học bổng ~ hiệp biến; chặn trong [0.05, 0.95] để không có trọng số quá lớn.
2. **Hai mô hình kết cục:** logistic riêng cho nhóm có học bổng ($\mu_1$) và không có ($\mu_0$), đoán khả năng ở lại.
3. **Điểm AIPW** từng sinh viên: dự đoán + phần "sửa sai" bằng sai số thật, có trọng số theo $e(x)$.
4. ATE = trung bình điểm có học bổng − trung bình điểm không học bổng.
5. CI: bootstrap 1,000 lần trên các điểm AIPW.

**🎤 Nói:**
"Kết cục giữ chân tính trên **cả 4,424 sinh viên**, kể cả 794 người đang học — nên nó cũng trả lời câu hỏi việc loại Enrolled có làm chệch không. Không có học bổng: 64.3% ở lại; có học bổng: 81.2%; ATE **+16.87%**, CI 13.79 đến 19.73. AIPW 'chắc ăn gấp đôi': chỉ cần một trong hai mô hình — xác suất được học bổng hoặc kết cục — đúng là kết quả không chệch."

**📊 Đọc output:** Risk(do(T=0)) 64.28% · Risk(do(T=1)) 81.15% · ATE +16.87% · CI [13.79%, 19.73%].

### Cell 77 · code · 9.E Trung gian — học bổng giúp qua đường nào

**🧩 Cell này làm gì (phương pháp tích hệ số):**
1. Hồi quy **tín chỉ đạt HK1** ~ học bổng + hiệp biến → hệ số $a$: học bổng làm tăng bao nhiêu tín chỉ HK1.
2. Hồi quy **tốt nghiệp** ~ học bổng + tín chỉ HK1 + hiệp biến → hệ số của học bổng = **tác dụng trực tiếp (NDE)**; hệ số $b$ của tín chỉ.
3. **Tác dụng gián tiếp (NIE)** = $a \times b$; tỷ lệ trung gian = NIE / (NDE + NIE).

**🎤 Nói:**
"Học bổng làm sinh viên đạt thêm **0.73 tín chỉ** ở HK1; mỗi tín chỉ thêm lại tăng khả năng tốt nghiệp. Nhân lại: tác dụng đi qua HK1 là **+4.65%**, trực tiếp là **+16.96%**, tổng +21.61%. Vậy khoảng **21.5%** tác dụng của học bổng đi qua việc học tốt hơn ở học kỳ đầu; phần còn lại là trực tiếp — bớt lo tài chính, có thêm động lực."

**📊 Đọc output:** hệ số T → M +0.729 tín chỉ · TE +21.61% · NDE +16.96% (78.5%) · NIE +4.65% (21.5%).

### Cell 79 · code · 9.F CATE — học bổng giúp ai nhiều hơn

**🧩 Cell này làm gì:** với 6 nhóm định trước (tuổi, Debtor, nơi ở): trong từng nhóm, khớp logistic tốt nghiệp ~ học bổng + hiệp biến; đặt học bổng = 0 cho mọi người, đoán, lấy trung bình; đặt = 1, làm lại; hiệu = CATE (G-computation).

**🎤 Nói:**
"Sinh viên **trên 20 tuổi** được lợi nhiều nhất: +28.0 điểm, so với +16.1 ở nhóm đúng tuổi. Với nhóm có nợ, hiệu ứng tuyệt đối +20.3 **không** lớn hơn nhóm không nợ +21.4; cột '2.03x' chỉ là tỷ số rủi ro, lớn vì mức nền của họ thấp (19.7%). Các nhóm chưa có khoảng tin cậy nên em đọc thận trọng."

**📊 Đọc output:**
| Cột | Nghĩa |
| :--- | :--- |
| Risk(do(T=0)) / Risk(do(T=1)) | Tỷ lệ tốt nghiệp trong nhóm nếu không / có học bổng |
| Hiệu ứng CATE | Hiệu tuyệt đối (điểm %) — **dùng cột này để so nhóm** |
| Relative Risk "1.72x" | Tỷ số; **không** phải "hưởng lợi gấp 1.72 lần nhóm khác" |

### Cell 83 · code · 9.H Robustness Value — kết luận có dễ bị lật không

**🧩 Cell này làm gì:**
1. Hồi quy tốt nghiệp ~ học bổng + 5 biến nền; lấy hệ số và thống kê $t$ của học bổng ($t$ = 14.20, bậc tự do 3,623).
2. Tính $f = |t|/\sqrt{df}$ rồi **RV** theo công thức Cinelli & Hazlett (2020).
3. Bảng mốc: mỗi biến nền quan sát được giải thích bao nhiêu phần trăm phần dư của tốt nghiệp (partial $R^2$).

**🎤 Nói:**
"Câu hỏi: nếu có một biến ẩn em chưa đo — ví dụ động lực học tập — nó phải mạnh cỡ nào mới xóa được hiệu ứng? Đáp án: phải giải thích ít nhất **21%** phần dư của **cả** việc nhận học bổng lẫn việc tốt nghiệp. Để so sánh, biến quan sát mạnh nhất là học phí đúng hạn chỉ 10.4%, các biến khác dưới 5%. Vậy kết luận vững trước nhiễu ẩn yếu hơn học phí, nhưng không miễn nhiễm với một nhiễu mạnh gấp đôi học phí."

**📊 Đọc output:**
| Dòng | Nghĩa |
| :--- | :--- |
| Hệ số +0.2268, t = 14.20 | Học bổng tăng 22.7 điểm (hồi quy tuyến tính) |
| Partial $R^2$ 5.27% | Học bổng giải thích 5.3% phần dư của tốt nghiệp |
| $RV_{q=1}$ = 20.97% | Ngưỡng để **xóa** hiệu ứng |
| $RV_{q=1,\alpha=0.05}$ = 18.37% | Ngưỡng để hiệu ứng **mất ý nghĩa thống kê** |
| Bảng mốc | Tuổi 4.22% · điểm vào 0.48% · học vấn mẹ 0.03% · Debtor 0.81% · học phí 10.41% |

### Cell 86 · code · Table 10.1 phân bổ học bổng + Table 10.2 chuyển sang Kaggle

**🧩 Cell này làm gì:**
1. Hai mô hình logistic ($\mu_0$ không học bổng, $\mu_1$ có học bổng) trên toàn bộ dữ liệu → CATE dự đoán = $\mu_1 - \mu_0$ cho từng sinh viên.
2. Bốn cách phân bổ **cùng 969 suất**: theo CATE (người hưởng lợi nhiều nhất trước), thực tế, ưu tiên người có nợ, ngẫu nhiên.
3. Giá trị mỗi cách = trung bình [được cấp × $\mu_1$ + không được cấp × $\mu_0$].
4. Table 10.2: đọc kết quả chuyển giữa UCI và Kaggle từ script riêng.

**🎤 Nói:**
"Giữ nguyên 969 suất nhưng ưu tiên người được dự đoán hưởng lợi nhiều nhất: tỷ lệ tốt nghiệp dự kiến từ 60.85% lên 65.37%, khoảng **+164 sinh viên**; ưu tiên theo nợ chỉ +49, ngẫu nhiên +37. Con số này **lạc quan**: cùng mô hình vừa chọn người vừa chấm điểm, chưa cross-fit. Kaggle cho kết quả trái chiều và là dữ liệu tổng hợp, nên chưa chứng minh được kết quả đúng ở trường khác."
