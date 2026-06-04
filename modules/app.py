import engine
import os
import sys


# Cấu hình hệ thống
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "true"
import streamlit as st
st.set_page_config(page_title="Hệ thống Mô hình Ra quyết định Kinh tế VN", layout="wide")
project_root = "/mount/src/aideom-vn"
if project_root not in sys.path:
    sys.path.append(project_root)

# Nạp dữ liệu và các module
from utils import load_all_data
import bai1
import bai2
import bai3
import bai4
import bai5
import bai6
import bai7
import bai8
import bai9
import bai10
import bai11
import bai12
st.title("📊 HỆ THỐNG MÔ HÌNH RA QUYẾT ĐỊNH PHÁT TRIỂN KINH TẾ VIỆT NAM")
st.markdown("---")

# 1. Nạp dữ liệu
try:
    df_macro, df_sectors, df_regions = load_all_data()
except Exception as e:
    st.error(f"Lỗi khi nạp dữ liệu: {e}")
    df_macro, df_sectors, df_regions = None, None, None

# 2. Thanh điều hướng
st.sidebar.title("MENU 12 BÀI TẬP")
menu_options = [
    "Bài 1: Hàm sản xuất Cobb-Douglas mở rộng", "Bài 2: Phân bổ ngân sách 4 hạng mục số",
    "Bài 3: Chỉ số ưu tiên 10 ngành Việt Nam", "Bài 4: Quy hoạch tuyến tính Ngành - Vùng",
    "Bài 5: Bài toán MIP lựa chọn dự án số", "Bài 6: Xếp hạng vùng qua TOPSIS & Entropy",
    "Bài 7: Tối ưu đa mục tiêu Pareto NSGA-II", "Bài 8: Tối ưu động phân bổ liên thời gian 2026-2035",
    "Bài 9: Mô phỏng dịch chuyển lao động", "Bài 10: Quy hoạch ngẫu nhiên hai giai đoạn",
    "Bài 11: Học tăng cường (Q-Learning) thích nghi", "Bài 12: Đồ án tích hợp hệ thống AIDEOM-VN"
]

bai_tap = st.sidebar.radio("Chọn bài tập để xem và chạy lời giải:", menu_options)

# 3. Điều hướng bài tập
# Lưu ý: Các file module (bai1, bai2...) phải có hàm .run()
dispatch = {
    menu_options[0]: bai1.run,
    menu_options[1]: bai2.run,
    menu_options[2]: bai3.run,
    menu_options[3]: bai4.run,
    menu_options[4]: bai5.run,
    menu_options[5]: bai6.run,
    menu_options[6]: bai7.run,
    menu_options[7]: bai8.run,
    menu_options[8]: bai9.run,
    menu_options[9]: bai10.run,
    menu_options[10]: bai11.run,
    menu_options[11]: bai12.run,
}

# 4. Thực thi module tương ứng
if bai_tap in dispatch:
    func = dispatch[bai_tap]
    try:
        # Bài 11 chạy độc lập, các bài khác cần truyền dữ liệu
        if "Bài 11" in bai_tap:
            func() 
        else:
            func(df_macro, df_sectors, df_regions)
    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi chạy {bai_tap}: {e}")
        st.info("Hãy kiểm tra file __init__.py của module bài tập đó.")