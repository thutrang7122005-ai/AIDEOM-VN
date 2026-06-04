import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pulp

def run(df_macro, df_sectors, df_regions):
    st.header("🏢 BÀI 5: QUY HOẠCH NGUYÊN HỖN HỢP (MIP) - LỰA CHỌN DỰ ÁN TỐI ƯU")
        
    # ── 5.1 & 5.2 THIẾT LẬP DỮ LIỆU GỐC ────────────────────────────────
    P = list(range(1, 16))
    PROJECT_NAMES = {
        1: "Trung tâm dữ liệu Hòa Lạc", 2: "Trung tâm dữ liệu phía Nam", 3: "Hệ thống 5G toàn quốc",
        4: "VNeID 2.0", 5: "Cổng dịch vụ công v3", 6: "Y tế số quốc gia",
        7: "Giáo dục số K-12", 8: "Trung tâm AI quốc gia", 9: "Sandbox Fintech",
        10: "Logistics thông minh", 11: "Nông nghiệp số ĐBSCL", 12: "Đào tạo kỹ sư AI/Bán dẫn",
        13: "Khu CN bán dẫn BN-BG", 14: "An ninh mạng SOC", 15: "Open Data quốc gia"
    }
    SECTORS = {
        1: "Hạ tầng", 2: "Hạ tầng", 3: "Hạ tầng", 4: "Chính phủ số", 5: "Chính phủ số",
        6: "Y tế số", 7: "Giáo dục", 8: "AI", 9: "Tài chính số", 10: "Logistics",
        11: "Nông nghiệp", 12: "Nhân lực", 13: "Bán dẫn", 14: "An ninh", 15: "Dữ liệu"
    }
    C = {1:12000, 2:11500, 3:18000, 4:4500, 5:3200, 6:5800, 7:6500, 8:15000, 9:2500, 10:7200, 11:4800, 12:8500, 13:20000, 14:3800, 15:1500}
    C1 = {1:8500, 2:7500, 3:12000, 4:3500, 5:2500, 6:4000, 7:4500, 8:9000, 9:1800, 10:5000, 11:3500, 12:5500, 13:13000, 14:2800, 15:1200}
    B = {1:21500, 2:20800, 3:32500, 4:9200, 5:6800, 6:11400, 7:12200, 8:28500, 9:5800, 10:13800, 11:8500, 12:16200, 13:35000, 14:7500, 15:3800}
        
    # Xác suất rủi ro (Câu 5.4.4)
    PROB = {}
    for i in P:
        if SECTORS[i] == "Hạ tầng": PROB[i] = 0.85
        elif SECTORS[i] == "Chính phủ số": PROB[i] = 0.75
        elif SECTORS[i] in ["AI", "Bán dẫn"]: PROB[i] = 0.65
        else: PROB[i] = 0.80

    # ── HÀM GIẢI MÔ HÌNH MIP TỔNG QUÁT ─────────────────────────────────
    def solve_project_mip(total_budget=80000, force_redundancy=False, use_risk=False):
        model = pulp.LpProblem("VN_Digital_Selection", pulp.LpMaximize)
        y = pulp.LpVariable.dicts("y", P, cat="Binary")
            
        # Hàm mục tiêu
        if use_risk:
            model += pulp.lpSum(PROB[i] * B[i] * y[i] for i in P), "Expected_NPV"
        else:
            model += pulp.lpSum(B[i] * y[i] for i in P), "Total_NPV"
            
        # Ràng buộc ngân sách
        model += pulp.lpSum(C[i] * y[i] for i in P) <= total_budget, "C1_Total_Budget"
        model += pulp.lpSum(C1[i] * y[i] for i in P) <= 40000, "C2_Year_1_2_Budget"
            
        # Ràng buộc logic
        if not force_redundancy:
            model += y[1] + y[2] <= 1, "C3_Exclusion"
        else:
            model += y[1] == 1, "C3_Force_P1"
            model += y[2] == 1, "C3_Force_P2"
                
        model += y[8] <= y[12], "C4_Precedence_AI"
        model += y[13] <= y[12], "C5_Precedence_Semiconductor"
        model += y[4] + y[5] >= 1, "C6_Min_E_Gov"
        model += y[14] >= 1, "C6_Mandatory_Security"
        model += pulp.lpSum(y[i] for i in P) >= 7, "C7_Min_Projects"
        model += pulp.lpSum(y[i] for i in P) <= 11, "C7_Max_Projects"
            
        model.solve(pulp.PULP_CBC_CMD(msg=False))
        return model, y

    # ── GIAO DIỆN TABS ────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 5.4.1 Giải cơ sở", 
        "🚀 5.4.2 & 5.4.3 Kịch bản", 
        "🎲 5.4.4 Rủi ro", 
        "💬 5.5 Thảo luận"
    ])

    with tab1:
        st.subheader("Giải bài toán lựa chọn dự án tối ưu (Ngân sách 80.000 tỷ)")
        m_base, y_base = solve_project_mip()
        
        if pulp.LpStatus[m_base.status] == "Optimal":
            selected_base = [i for i in P if y_base[i].value() > 0.5]
            z_star = pulp.value(m_base.objective)
            cost_star = sum(C[i] for i in selected_base)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Tổng NPV (Z*)", f"{z_star:,.0f} tỷ")
            col2.metric("Ngân sách sử dụng", f"{cost_star:,.0f} tỷ")
            col3.metric("Hiệu suất (Z*/C)", f"{z_star/cost_star:.3f}")
            
            df_res = pd.DataFrame({
                "Mã": [f"P{i}" for i in selected_base],
                "Tên dự án": [PROJECT_NAMES[i] for i in selected_base],
                "Lĩnh vực": [SECTORS[i] for i in selected_base],
                "Chi phí (C)": [C[i] for i in selected_base],
                "Lợi ích (B)": [B[i] for i in selected_base],
                "B/C Ratio": [round(B[i]/C[i], 2) for i in selected_base]
            })
            st.dataframe(df_res, use_container_width=True)
        else:
            st.error("Không tìm thấy phương án tối ưu.")

    with tab2:
        st.subheader("Phân tích độ nhạy & Ràng buộc chiến lược")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**5.4.2 Nới ngân sách lên 100.000 tỷ**")
            m_100, y_100 = solve_project_mip(total_budget=100000)
            st.write(f"Z* mới: **{pulp.value(m_100.objective):,.0f} tỷ**")
            st.write(f"Số dự án chọn: **{sum(1 for i in P if y_100[i].value() > 0.5)}**")

        with col_b:
            st.markdown("**5.4.3 Yêu cầu cả P1 và P2 (Dự phòng)**")
            m_red, y_red = solve_project_mip(force_redundancy=True)
            if pulp.LpStatus[m_red.status] == "Optimal":
                st.success(f"Khả thi! Z* = {pulp.value(m_red.objective):,.0f} tỷ")
            else:
                st.error("BẤT KHẢ THI! Ràng buộc ngân sách và điều kiện loại trừ bị vi phạm.")

    with tab3:
        st.subheader("5.4.4 Tối ưu hóa lợi ích kỳ vọng (Có xét rủi ro)")
        m_risk, y_risk = solve_project_mip(use_risk=True)
        # Lưu ý: cần dùng m_base, y_base từ tab1 hoặc chạy lại
        df_comp = pd.DataFrame({
            "Dự án": [f"P{i}" for i in P],
            "Tên": [PROJECT_NAMES[i] for i in P],
            "Chọn (Cơ sở)": ["✅" if y_base[i].value() > 0.5 else "❌" for i in P],
            "Chọn (Có rủi ro)": ["✅" if y_risk[i].value() > 0.5 else "❌" for i in P],
            "Xác suất thành công": [PROB[i] for i in P]
        })
        st.dataframe(df_comp, use_container_width=True)

    with tab4:
        st.subheader("5.5 Luận giải chính sách")
        with st.expander("a) Tại sao mô hình bỏ qua P15 (Open Data)?"):
            st.write("Dự án P15 có tỉ lệ B/C cao nhưng NPV tuyệt đối nhỏ. Trong MIP, solver ưu tiên NPV lớn để tối ưu Z*.")
        with st.expander("b) Ràng buộc bắt buộc P14 (An ninh mạng) có hợp lý?"):
            st.write("An ninh mạng là hạ tầng nền tảng, việc bắt buộc này là hợp lý để quản trị rủi ro.")
        with st.expander("c) Cách mô hình hóa hiệu ứng cộng hưởng?"):
            st.write("Thêm biến nhị phân phụ và ràng buộc logic để ép nhận thêm lợi ích khi các dự án bổ trợ cùng chọn.")

    # Vẽ biểu đồ
    st.divider()
    st.subheader("📈 Phân bổ ngân sách theo lĩnh vực")
    sector_cost = {}
    selected_base = [i for i in P if y_base[i].value() > 0.5]
    for i in selected_base:
        sec = SECTORS[i]
        sector_cost[sec] = sector_cost.get(sec, 0) + C[i]
    
    fig = px.pie(names=list(sector_cost.keys()), values=list(sector_cost.values()), 
                 hole=0.4, title="Tỷ trọng vốn đầu tư theo lĩnh vực (Kịch bản cơ sở)")
    st.plotly_chart(fig, use_container_width=True)