with open('tools/build_master_notebook_v2.py', 'r', encoding='utf-8') as f:
    code = f.read()

old_dsep = """    # Kiểm tra d-separation trên đồ thị phẫu thuật
    la_d_separated = nx.d_separated(do_thi_underline, {nut_can_thiep}, {nut_ket_qua}, tap_dieu_kien)
    vi_pham = []
    if not la_d_separated:
        vi_pham.append(f"Tập điều kiện Z={tap_dieu_kien} KHÔNG chặn hết các đường dẫn backdoor nối {nut_can_thiep} và {nut_ket_qua}!")"""

new_dsep = """    from pgmpy.base import DAG
    # Kiểm tra d-separation trên đồ thị phẫu thuật thông qua pgmpy DAG
    p_dag = DAG(list(do_thi_underline.edges()))
    la_d_separated = not p_dag.is_dconnected(nut_can_thiep, nut_ket_qua, observed=tap_dieu_kien)
    vi_pham = []
    if not la_d_separated:
        vi_pham.append(f"Tập điều kiện Z={tap_dieu_kien} KHÔNG chặn hết các đường dẫn backdoor nối {nut_can_thiep} và {nut_ket_qua}!")"""

assert old_dsep in code, "old_dsep not found in code!"
code = code.replace(old_dsep, new_dsep)

with open('tools/build_master_notebook_v2.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("d-separation patched successfully!")
