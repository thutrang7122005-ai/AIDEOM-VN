import streamlit as st
import pandas as pd
import numpy as np

@st.cache_data
def load_all_data():
    """
    Hàm này nạp tất cả dữ liệu cần thiết cho 12 bài tập.
    Việc sử dụng @st.cache_data giúp web chạy nhanh hơn bằng cách
    lưu trữ dữ liệu vào bộ nhớ sau lần đọc đầu tiên.
    """
    
    # 1. Nạp dữ liệu Macro (GDP, kinh tế vĩ mô)
    try:
        macro = pd.read_csv("vietnam_macro_2020_2025.csv")
    except Exception:
        # Dữ liệu mặc định nếu không tìm thấy file
        macro = pd.DataFrame({
            "Nam": [2020, 2021, 2022, 2023, 2024, 2025], 
            "GDP": [342, 366, 408, 430, 465, 500]
        })
        
    # 2. Nạp dữ liệu Ngành (Sectors)
    try:
        sectors = pd.read_csv("vietnam_sectors_2024.csv")
    except Exception:
        sectors = pd.DataFrame({
            "Nganh": ["Nông nghiệp", "Công nghiệp", "Dịch vụ"], 
            "Ty_trong": [12.0, 38.0, 50.0]
        })

    # 3. Nạp dữ liệu Vùng (Regions)
    try:
        regions = pd.read_csv("vietnam_regions_2024.csv")
    except Exception:
        data_fallback = {
            "region_name_vi": [
                "Trung du và miền núi phía Bắc", "Đồng bằng sông Hồng", 
                "Bắc Trung Bộ và Duyên hải miền Trung", "Tây Nguyên", 
                "Đông Nam Bộ", "Đồng bằng sông Cửu Long"
            ],
            "grdp_trillion_VND": [450.0, 1600.0, 750.0, 320.0, 2100.0, 1050.0],
            "digital_index_0_100": [42.5, 78.2, 53.4, 38.0, 84.5, 59.1],
            "ai_readiness_0_100": [35.0, 72.1, 48.6, 32.4, 81.0, 52.3],
            "trained_labor_pct": [18.5, 32.4, 22.1, 15.2, 35.8, 17.9],
            "grdp_growth_pct": [6.2, 7.5, 5.8, 5.2, 6.8, 6.0]
        }
        regions = pd.DataFrame(data_fallback)
        
    return macro, sectors, regions

# Các hàm tính toán dùng chung (Ví dụ TOPSIS)
def topsis_score(X, w, is_benefit):
    # Chuẩn hóa dữ liệu
    R = X / np.sqrt((X**2).sum(axis=0))
    # Tính ma trận trọng số
    V = R * w
    # Tìm lý tưởng dương và âm
    A_star = np.where(is_benefit, V.max(axis=0), V.min(axis=0))
    A_neg = np.where(is_benefit, V.min(axis=0), V.max(axis=0))
    # Tính khoảng cách
    S_star = np.sqrt(((V - A_star)**2).sum(axis=1))
    S_neg = np.sqrt(((V - A_neg)**2).sum(axis=1))
    return S_neg / (S_star + S_neg)