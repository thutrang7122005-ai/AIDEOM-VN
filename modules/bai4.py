import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def run(df_macro, df_sectors, df_regions):
        try:
            import pulp
            PULP_OK = True
        except ImportError:
            PULP_OK = False

        st.header("📐 BÀI 4: QUY HOẠCH TUYẾN TÍNH PHÂN BỔ NGÂN SÁCH SỐ THEO NGÀNH - VÙNG")
        if not PULP_OK:
            st.error("⚠️ Thư viện `pulp` chưa được cài đặt! Vui lòng chạy lệnh: `pip install pulp` trong PowerShell.")
            return # Dừng hàm tại đây để không gây lỗi tiếp theo

        st.markdown(
            "Tối ưu hóa phân bổ **hạn ngạch tăng trưởng GRDP** cho 6 vùng kinh tế xã hội Việt Nam "
            "dưới ràng buộc ngân sách quỹ phát triển liên vùng, công bằng vùng miền và cân đối chiến lược."
        )

        # ── Thiết lập tham số mô hình từ dữ liệu vĩ mô ───────────────────────
        REGIONS      = df_regions["region_name_vi"].tolist()
        REGION_SHORT = ["TDMNPB", "ĐBSH", "BTB-DHMT", "TN", "ĐNB", "ĐBSCL"]
        N            = len(REGIONS)

        GRDP         = df_regions["grdp_trillion_VND"].values.astype(float)
        DIG          = df_regions["digital_index_0_100"].values.astype(float)
        AI_R         = df_regions["ai_readiness_0_100"].values.astype(float)

        # Hệ số đóng góp công nghệ số w_r (Tổng hợp từ Digital Index + AI Readiness)
        w_raw = 0.55 * (DIG / 100.0) + 0.45 * (AI_R / 100.0)
        W     = w_raw / w_raw.sum()

        # Chi phí đầu tư biên quy đổi c_r
        DIG_NORM  = DIG / 100.0
        GRDP_NORM = GRDP / GRDP.max()
        C_r       = 1.5 - 0.8 * DIG_NORM - 0.3 * GRDP_NORM
        C_r       = np.clip(C_r, 0.4, 1.5)

        NORTH_KEY   = 1 # ĐBSH
        SOUTH_KEY   = 4 # ĐNB

        # ── Thiết lập các Tabs cho Bài 4 ───────────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 4.1 – 4.3 Bài toán & Dữ liệu",
            "🧮 4.4.1 & 4.4.2 Tối ưu hóa PuLP",
            "📈 4.4.3 Trực quan hóa",
            "💬 4.5 Thảo luận Chính sách",
        ])

        with tab1:
            st.markdown("### 4.1 Bối cảnh chính sách")
            st.info(
                "Theo **Quyết định 411/QĐ-TTg (2022)** và **Nghị quyết 57-NQ/TW (2024)**, Việt Nam cần phân bổ "
                "ngân sách kinh tế số theo hướng **hiệu quả + công bằng**. Bài toán đặt ra: tối đa hóa GDP "
                "gia tăng toàn quốc trong khi bảo đảm không vùng nào bị bỏ lại phía sau và hai cực tăng trưởng "
                "phía Bắc – phía Nam vẫn duy trì vai trò đầu tàu."
            )

            st.markdown("### 4.2 Mô hình toán học")
            st.markdown(
                r"""
**Biến quyết định:** $\Delta Y_r \geq 0$ — sản lượng GRDP tăng thêm mục tiêu của vùng $r$ (nghìn tỷ VND), với $r \in \{1,\ldots,6\}$.

**Hàm mục tiêu:**
$$\max \; Z = \sum_{r=1}^{6} w_r \cdot \Delta Y_r$$

**Các ràng buộc chủ yếu:**
* **C1 (Trần ngân sách):** $\sum_{r=1}^{6} c_r \cdot \Delta Y_r \leq \text{Ngân sách Quỹ}$
* **C2 (Sàn tăng trưởng vùng miền):** $\Delta Y_r \geq \text{Sàn \%} \cdot \mathrm{GRDP}_r \quad \forall r$
* **C3 (Đầu tàu kinh tế):** $\Delta Y_{\text{ĐBSH}} + \Delta Y_{\text{ĐNB}} \geq \text{Tỷ trọng \%} \cdot \sum \Delta Y_r$
                """
            )

            st.markdown("### 4.3 Dữ liệu vĩ mô tích hợp hiện tại")
            df_display = df_regions[["region_name_vi", "grdp_trillion_VND", "grdp_growth_pct", "digital_index_0_100", "ai_readiness_0_100"]].copy()
            df_display.columns = ["Vùng", "GRDP (nT)", "Tăng trưởng (%)", "Digital Index", "AI Readiness"]
            df_display["w_r (Trọng số số)"] = W
            df_display["c_r (Hệ số chi phí)"] = C_r
            st.dataframe(df_display.set_index("Vùng").style.format("{:.4f}"), use_container_width=True)

        with tab2:
            st.markdown("### 4.4.1 Giải bài toán bằng PuLP (CBC Solver)")

            if not PULP_OK:
                st.error("⚠️ Thư viện `pulp` chưa được cài đặt hoặc PowerShell không nhận diện. Hãy chạy lệnh `python -m pip install pulp` ở Terminal.")
            else:
                with st.expander("⚙️ Tinh chỉnh tham số mô hình bằng Kịch bản động", expanded=True):
                    budget_input  = st.slider("Tổng quỹ phân bổ liên vùng (nghìn tỷ VND)", min_value=100, max_value=300, value=170, step=10)
                    floor_pct     = st.slider("Sàn tăng trưởng tối thiểu mỗi vùng (% GRDP)", min_value=1, max_value=6, value=2, step=1)
                    key_pct_input = st.slider("Tỷ trọng tối thiểu 2 vùng trọng điểm ĐBSH & ĐNB (%)", min_value=30, max_value=70, value=50, step=5)

                BUDGET_USE  = float(budget_input)
                FLOOR_USE   = float(floor_pct) / 100.0
                KEY_USE     = float(key_pct_input) / 100.0
                DELTA_MIN_U = FLOOR_USE * GRDP

                min_cost_required = np.dot(C_r, DELTA_MIN_U)

                if min_cost_required > BUDGET_USE:
                    st.error(f"❌ **KỊCH BẢN BẤT KHẢ THI (Infeasible Model):**")
                    st.markdown(f"""
                    > **Giải thích định lượng:** Với mức sàn tăng trưởng mục tiêu là **{floor_pct}%**, tổng chi phí tối thiểu bắt buộc để trợ cấp 
                    > nền tảng cho cả 6 vùng đã lên tới **{min_cost_required:.2f} nghìn tỷ VND**. 
                    > Trong khi đó, Ngân sách hiện tại bạn thiết lập chỉ có **{BUDGET_USE:.0f} nghìn tỷ VND** (Thiếu hụt **{min_cost_required - BUDGET_USE:.2f} nT**).
                    
                    💡 **Hướng xử lý đề xuất:**
                    1. **Kéo tăng** thanh trượt *'Tổng quỹ phân bổ liên vùng'* lên tối thiểu từ **{int(np.ceil(min_cost_required/10)*10)}** nT trở lên.
                    2. Hoặc **Kéo giảm** thanh trượt *'Sàn tăng trưởng tối thiểu mỗi vùng'* xuống mức thấp hơn.
                    """)
                else:
                    model = pulp.LpProblem("VN_Regional_Growth_LP", pulp.LpMaximize)
                    dY = [pulp.LpVariable(f"dY_{REGION_SHORT[r]}", lowBound=DELTA_MIN_U[r]) for r in range(N)]
                    
                    model += pulp.lpSum(W[r] * dY[r] for r in range(N)), "Tong_tang_truong"
                    model += pulp.lpSum(C_r[r] * dY[r] for r in range(N)) <= BUDGET_USE, "C1_Tran_ngan_sach"
                    
                    for r in range(N):
                        model += dY[r] >= DELTA_MIN_U[r], f"C2_San_{REGION_SHORT[r]}"

                    other_idx = [r for r in range(N) if r not in [NORTH_KEY, SOUTH_KEY]]
                    model += ((1 - KEY_USE) * (dY[NORTH_KEY] + dY[SOUTH_KEY]) >= KEY_USE * pulp.lpSum(dY[r] for r in other_idx)), "C3_Can_doi"

                    model.solve(pulp.PULP_CBC_CMD(msg=False))

                    if model.status == 1:
                        st.success(f"✅ Solver CBC: **Tìm thấy nghiệm tối ưu toàn cục** thỏa mãn hoàn hảo!")
                        dY_opt = np.array([pulp.value(dY[r]) for r in range(N)])
                        Z_star = pulp.value(model.objective)
                        cost_used = np.dot(C_r, dY_opt)

                        st.markdown("#### 📊 Kết quả tổng quan kịch bản")
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Z* (Hàm mục tiêu)", f"{Z_star:.4f}")
                        m2.metric("Tổng GRDP tăng thêm", f"{dY_opt.sum():.2f} nT")
                        m3.metric("Ngân sách sử dụng", f"{cost_used:.1f}/{BUDGET_USE:.0f} nT")
                        m4.metric("Hiệu suất vốn (ΔY/C)", f"{dY_opt.sum()/cost_used:.3f}")

                        st.markdown("#### 📋 Phân bổ tối ưu chi tiết")
                        result_df = pd.DataFrame({
                            "Vùng"                : REGIONS,
                            "GRDP hiện tại (nT)"  : GRDP,
                            "ΔY tối ưu (nT)"      : dY_opt.round(3),
                            "ΔY/GRDP (%)"         : (dY_opt / GRDP * 100).round(2),
                            "Chi phí quy đổi (nT)": (C_r * dY_opt).round(3),
                            "Trọng số w_r"        : W.round(4),
                        })
                        st.dataframe(result_df.set_index("Vùng"), use_container_width=True)

                        model_no_c3 = pulp.LpProblem("VN_No_Equity", pulp.LpMaximize)
                        dY2 = [pulp.LpVariable(f"dY2_{REGION_SHORT[r]}", lowBound=DELTA_MIN_U[r]) for r in range(N)]
                        model_no_c3 += pulp.lpSum(W[r] * dY2[r] for r in range(N))
                        model_no_c3 += pulp.lpSum(C_r[r] * dY2[r] for r in range(N)) <= BUDGET_USE
                        for r in range(N):
                            model_no_c3 += dY2[r] >= DELTA_MIN_U[r]
                        model_no_c3.solve(pulp.PULP_CBC_CMD(msg=False))
                        dY2_opt = np.array([pulp.value(dY2[r]) for r in range(N)])
                        Z2_star = pulp.value(model_no_c3.objective)

                        st.markdown("### 4.4.2 Đánh giá điều hòa (Trade-off) mục tiêu quốc gia")
                        col_l, col_r = st.columns(2)
                        col_l.metric("Z* có C3 (Cân đối vùng miền)", f"{Z_star:.4f}")
                        col_r.metric("Z* không C3 (Hiệu suất kinh tế tối đa)", f"{Z2_star:.4f}")

                        delta_z = (Z2_star - Z_star) / Z2_star * 100
                        if delta_z > 0.001:
                            st.warning(f"⚖️ **Chi phí kinh tế của sự công bằng:** Ràng buộc điều hòa vùng miền (C3) làm giảm hàm mục tiêu **{delta_z:.2f}%**. Đây là sự đánh đổi bắt buộc để tránh phân cực hóa dòng vốn tài khóa.")
                        else:
                            st.info("✅ Tại cấu hình này, mục tiêu công bằng hoàn toàn tương thích với hiệu suất quốc gia.")

                        st.session_state["bai4_dY_opt"]  = dY_opt
                        st.session_state["bai4_dY2_opt"] = dY2_opt

        with tab3:
            st.markdown("### 4.4.3 Trực quan hóa không gian phân bổ tối ưu")
            if "bai4_dY_opt" not in st.session_state:
                st.info("⏳ Vui lòng thiết lập cấu hình kịch bản hợp lý tại Tab 2 để kích hoạt đồ thị vẽ tự động.")
            else:
                dY_opt_v  = st.session_state["bai4_dY_opt"]
                dY2_opt_v = st.session_state["bai4_dY2_opt"]
                PALETTE   = ["#1a6b4a", "#2e86de", "#e84393", "#f39c12", "#8e44ad", "#16a085"]

                st.markdown("#### 1. So sánh GRDP tăng thêm mục tiêu: Có ràng buộc C3 vs. Không có C3")
                fig1 = go.Figure()
                fig1.add_trace(go.Bar(y=REGIONS, x=dY2_opt_v, orientation="h", name="Kịch bản Tự do (Hiệu suất)", marker_color="rgba(231,76,60,0.45)"))
                fig1.add_trace(go.Bar(y=REGIONS, x=dY_opt_v, orientation="h", name="Kịch bản Cân đối (Thực tế)", marker_color=PALETTE))
                fig1.update_layout(barmode="group", template="plotly_white", height=380)
                st.plotly_chart(fig1, use_container_width=True)

                st.markdown("#### 2. Tốc độ tăng trưởng GRDP tăng thêm bổ sung từ đầu tư số (%)")
                growth_add = dY_opt_v / GRDP * 100
                fig2 = px.bar(x=REGION_SHORT, y=growth_add, color=REGION_SHORT, color_discrete_sequence=PALETTE, text=[f"{v:.2f}%" for v in growth_add])
                fig2.update_layout(template="plotly_white", height=350, showlegend=False, xaxis_title="Vùng", yaxis_title="Tỷ lệ tăng trưởng thêm (%)")
                st.plotly_chart(fig2, use_container_width=True)

        with tab4:
            st.markdown("### 4.5 Luận cứ và Thảo luận chiến lược liên vùng")
            st.markdown("""
            * **a) Hệ quả phân bổ dòng vốn tự do kinh tế:** Nếu gạt bỏ hoàn toàn ràng buộc C3 (Cân đối vùng miền), thuật toán phân bổ sẽ lập tức dồn toàn bộ nguồn lực ngân sách còn dư về cho **Đồng bằng sông Hồng** và **Đông Nam Bộ**. Hai khu vực này có nền tảng hạ tầng số tốt, lực lượng lao động chất lượng cao và mức độ sẵn sàng hấp thụ AI tối ưu nên có chi phí vận hành biên rẻ hơn nhiều. Đầu tư tự do sẽ mang lại tăng trưởng kinh tế tổng thể rất cao trong ngắn hạn, nhưng hệ quả xã hội dài hạn là nới rộng hố sâu khoảng cách số, làm gia tăng chỉ số bất bình đẳng thu nhập giữa các vùng kinh tế xã hội.
            * **b) Ý nghĩa kinh tế vĩ mô của trần quỹ C1:** Bản chất ràng buộc C1 chính là công cụ thể hiện định mức giới hạn tổng trần chi tiêu tài khóa quốc gia. Kịch bản thực nghiệm khi kéo tăng hay giảm quy mô của biến trần này cho thấy rõ năng lực hấp thụ nguồn lực biên của các vùng kinh tế: khi dòng ngân sách công vượt qua một ngưỡng giới hạn vật lý nhất định, hiệu quả sinh lời biên của công nghệ số sẽ giảm dần do thiếu hụt đồng bộ các yếu tố phụ trợ đi kèm (như mạng lưới logistics, chuỗi cung ứng dịch vụ bản địa).
            * **c) Vai trò điều tiết của sàn tăng trưởng tối thiểu C2:** Ràng buộc C2 đóng vai trò là "lưới an sinh số", ngăn chặn tình trạng phân cực hóa dòng vốn đầu tư. Khi tăng giá trị sàn (floor_pct), mô hình buộc phải phân phối tài nguyên cho cả những vùng khó khăn (như Tây Nguyên hay Trung du miền núi phía Bắc), đảm bảo duy trì tốc độ phát triển tối thiểu để thu hẹp khoảng cách. Tuy nhiên, nếu đặt sàn quá cao vượt khỏi khả năng chịu tải của trần ngân sách C1, bài toán sẽ rơi vào trạng thái bất khả thi (Infeasible) — phản ánh thực tế chính sách: không thể dàn trải nguồn lực vượt quá giới hạn tài khóa cho phép.
            """)