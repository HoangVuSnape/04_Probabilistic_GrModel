with open('tools/build_master_notebook_v2.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Fix HillClimb
old_hc = """    estimator = HillClimbSearch(du_lieu)
    scoring = BIC(du_lieu)
    
    # Thực hiện ước lượng cấu trúc với điểm BIC và danh sách cạnh cấm (nếu có)
    mo_hinh_uoc_luong = estimator.estimate(
        scoring_method=scoring,
        black_list=danh_sach_cam,
        max_iter=so_vong_lap_toi_da,
        show_progress=False
    )"""

new_hc = """    from pgmpy.causal_discovery import ExpertKnowledge
    estimator = HillClimbSearch(du_lieu)
    scoring = BIC(du_lieu)
    
    tri_thuc = None
    if danh_sach_cam is not None:
        tri_thuc = ExpertKnowledge(forbidden_edges=danh_sach_cam)
        
    mo_hinh_uoc_luong = estimator.estimate(
        scoring_method=scoring,
        expert_knowledge=tri_thuc,
        max_iter=so_vong_lap_toi_da,
        show_progress=False
    )"""

assert old_hc in code, "old_hc not found in code!"
code = code.replace(old_hc, new_hc)

# Fix MLE
old_mle = """    # 1. Ước lượng bằng MLE
    mo_hinh.fit(du_lieu, estimator=MaximumLikelihoodEstimator)
    cpd_mle = mo_hinh.get_cpds(bien_muc_tieu)"""

new_mle = """    # 1. Ước lượng bằng MLE
    estimator_mle = MaximumLikelihoodEstimator(mo_hinh, du_lieu)
    cpd_mle = estimator_mle.estimate_cpd(bien_muc_tieu)"""

assert old_mle in code, "old_mle not found in code!"
code = code.replace(old_mle, new_mle)

with open('tools/build_master_notebook_v2.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("Patch applied successfully!")
