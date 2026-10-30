# Thứ tự cell notebook

Notebook nên chạy từ đầu đến cuối bằng `Run all`, không phụ thuộc state của
phiên trước.

## Cell 1-4: Setup

1. Tiêu đề, tác giả, câu hỏi nhân quả, nguồn dữ liệu.
2. Cài/import phiên bản package (`numpy`, `pandas`, `scikit-learn`,
   `pgmpy`, `dowhy`, `networkx`, `torch`).
3. Khai báo seed list, cấu hình, thư mục artifacts.
4. Hàm `set_seed`, logger và hàm ghi JSON/CSV.

## Cell 5-9: Synthetic benchmark

5. Sinh `N=20_000` mẫu với `Z -> T`, `Z -> Y`, `T -> Y`.
6. In counts theo `Z,T` và kiểm tra positivity.
7. Fit B0 và tính observational/do probabilities.
8. DoWhy identify/estimate, GCM intervention.
9. Hai refutation; đối chiếu ATE với `0.30`, lưu bảng synthetic.

## Cell 10-15: UCI pipeline

10. Nạp UCI 697 và in shape/metadata.
11. Lọc `Enrolled`, mã hóa Y/T, tạo Z.
12. Kiểm tra missing, range, leakage và overlap.
13. Vẽ phân bố T/Y và các bảng tần số cần thiết.
14. Tạo 5 paired splits, lưu index.
15. Hiển thị một split mẫu và xác nhận preprocessing chỉ fit trên train.

## Cell 16-22: B0

16. Khai báo domain DAG và giải thích temporal ordering.
17. Fit B0 trên từng seed.
18. `check_model`, in CPD.
19. Query observation và intervention.
20. Tính predictive metrics trên test.
21. DoWhy identify/estimate/refutation.
22. Lưu kết quả B0.

## Cell 23-29: H1, H2

23. Khai báo H1 MLP, train/early stopping.
24. H1 do-query và test metrics.
25. Khai báo H2, mask, acyclicity loss.
26. Train H2, kiểm tra convergence/DAG.
27. H2 do-query và test metrics nếu identified.
28. Lưu checkpoint, learned adjacency, loss curves.
29. Lưu kết quả H1/H2, lỗi phải có `status`.

## Cell 30-34: Ablations

30. Chạy A1 cùng protocol H1.
31. Chạy A2 cùng protocol H2.
32. Chạy A3 cùng protocol B0.
33. Ghép paired deltas theo seed.
34. Xuất Bảng C.

## Cell 35-39: Tổng hợp và demo

35. Bảng A: UCI mean +/- std/CI.
36. Bảng B: synthetic truth/ATE/GED/runtime.
37. Bảng C: ablation deltas.
38. Hình DAG, mask, CPD, uncertainty và runtime.
39. Kết luận, limitations, và 6 bước demo Midterm.

