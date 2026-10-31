import sys
import kagglehub
import shutil
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print("Đang tải dữ liệu Kaggle Playground Series s4e6...")
try:
    path = kagglehub.competition_download("playground-series-s4e6")
    print(f"Tải thành công về cache: {path}")
    
    dest_dir = Path("data/kaggle_s4e6")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    src = Path(path)
    for f in src.glob("*"):
        dest_file = dest_dir / f.name
        if not dest_file.exists():
            shutil.copy2(f, dest_file)
            print(f"Đã sao chép: {f.name} ({dest_file.stat().st_size / (1024*1024):.2f} MB)")
        else:
            print(f"Đã tồn tại: {f.name}")
    print("HOÀN TẤT CHUẨN BỊ DỮ LIỆU KAGGLE S4E6!")
except Exception as e:
    print(f"Lỗi tải Kaggle: {type(e).__name__}: {e}")
