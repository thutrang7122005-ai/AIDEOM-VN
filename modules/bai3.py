import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def run(df_macro, df_sectors, df_regions):
        st.header("📊 BÀI 3: CHỈ SỐ ƯU TIÊN ĐẦU TƯ SỐ CHO 10 NGÀNH KINH TẾ VIỆT NAM")
        st.markdown("---")
        
        # Thiết lập cấu trúc Tabs khoa học cho Bài 3
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 3.1 - 3.3 Cơ sở dữ liệu ngành", 
            "🧮 3.4.1 & 3.4.2 Chuẩn hóa & Xếp hạng", 
            "📈 3.4.3 Trực quan hóa Đa chiều (Radar)", 
            "💬 3.5 Thảo luận & Chiến lược ngành"
        ])
        
        with tab1:
            st.subheader("1. Cơ sở lý thuyết & Dữ liệu vĩ mô 10 Ngành Kinh tế (2024)")
            st.markdown("""
            Mô hình áp dụng phương pháp chuẩn hóa **Min-Max** nhằm đồng bộ các chỉ tiêu có đơn vị đo khác nhau (%, tỷ USD, triệu người) về cùng khoảng $[0, 1]$, sau đó tính điểm tổng hợp theo ma trận trọng số.
            """)
            
            # Hiển thị bảng dữ liệu gốc đọc từ file vietnam_sectors_2024.csv
            st.write("**Bảng dữ liệu gốc từ hệ thống cơ sở dữ liệu (`vietnam_sectors_2024.csv`):**")
            st.dataframe(df_sectors.style.format({
                "gdp_share_2024_pct": "{:.2f}%",
                "growth_rate_2024_pct": "{:.2f}%",
                "labor_million": "{:.2f}M",
                "export_billion_USD": "${:.1f}B",
                "automation_risk_pct": "{:.2f}%",
                "spillover_coef_0_1": "{:.2f}"
            }))
            
            st.info("""
            ⚙️ **Hệ thống cấu trúc chỉ tiêu đa tiêu chí (MCDA):**
            * **Chỉ tiêu tác động tích cực (Càng cao càng tốt):** Tốc độ tăng trưởng, Tỷ trọng GDP, Hệ số lan tỏa, Kim ngạch xuất khẩu, Quy mô lao động, Mức độ sẵn sàng AI. (Tổng trọng số: $85\%$)
            * **Chỉ tiêu tác động tiêu cực (Càng cao càng rủi ro - đảo ngược):** Rủi ro tự động hóa lao động ($15\%$).
            """)

        with tab2:
            st.subheader("2. Thuật toán chuẩn hóa Min-Max & Kết quả xếp hạng thứ tự ưu tiên")
            
            # Định nghĩa các tập chỉ tiêu tích cực và tiêu cực theo đúng đề bài
            cols_positive = ['growth_rate_2024_pct', 'gdp_share_2024_pct', 'spillover_coef_0_1', 'export_billion_USD', 'labor_million', 'ai_readiness_0_100']
            col_negative = 'automation_risk_pct'
            
            # Áp dụng công thức chuẩn hóa Min-Max
            df_pos_norm = df_sectors[cols_positive].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
            df_neg_norm = (df_sectors[col_negative].max() - df_sectors[col_negative]) / (df_sectors[col_negative].max() - df_sectors[col_negative].min())
            
            # Khai báo ma trận trọng số đa tiêu chí (MCDA weights)
            w_pos = np.array([0.15, 0.15, 0.20, 0.15, 0.10, 0.10])
            w_neg = 0.15
            
            # Tính toán chỉ số ưu tiên tổng hợp bằng tích vô hướng ma trận
            priority_scores = (df_pos_norm.values @ w_pos) + (df_neg_norm.values * w_neg)
            
            # Tạo bảng kết quả tổng hợp
            df_result_sectors = df_sectors.copy()
            df_result_sectors['Chỉ số Ưu tiên ngành'] = priority_scores
            
            # Sắp xếp thứ tự giảm dần
            df_ranked = df_result_sectors[['sector_id', 'sector_name_vi', 'sector_name_en', 'Chỉ số Ưu tiên ngành']].sort_values(by='Chỉ số Ưu tiên ngành', ascending=False).reset_index(drop=True)
            df_ranked['Thứ hạng'] = df_ranked.index + 1
            
            st.success("✅ Đã hoàn thành chuẩn hóa ma trận dữ liệu và tính toán điểm số đa tiêu chí!")
            
            col_t1, col_t2 = st.columns([1.2, 1])
            with col_t1:
                st.write("**Bảng xếp hạng ưu tiên đầu tư chuyển đổi số quốc gia:**")
                st.dataframe(df_ranked[['Thứ hạng', 'sector_name_vi', 'Chỉ số Ưu tiên ngành']].style.format({"Chỉ số Ưu tiên ngành": "{:.4f}"}))
            with col_t2:
                fig3 = px.bar(
                    df_ranked, 
                    x="Chỉ số Ưu tiên ngành", 
                    y="sector_name_vi", 
                    orientation='h', 
                    color="Chỉ số Ưu tiên ngành",
                    color_continuous_scale="Viridis",
                    title="Phân vị thứ tự ưu tiên số hóa giữa các ngành"
                )
                fig3.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
                st.plotly_chart(fig3, use_container_width=True)

        with tab3:
            st.subheader("3. Biểu đồ Radar so sánh cấu trúc hồ sơ các ngành top đầu")
            import plotly.graph_objects as go
            
            # Tạo ma trận chuẩn hóa toàn diện để vẽ biểu đồ mạng nhện (Radar Chart)
            df_all_norm = df_pos_norm.copy()
            df_all_norm['inverted_automation_risk'] = df_neg_norm
            categories = ['Tăng trưởng', 'Tỷ trọng GDP', 'Hệ số lan tỏa', 'Xuất khẩu', 'Lao động', 'Sẵn sàng AI', 'An toàn tự động hóa']
            
            fig_radar = go.Figure()
            
            # Lấy ra 3 ngành đứng đầu để vẽ đồ thị so sánh trực quan
            top_3_sectors = df_ranked.head(3)['sector_name_vi'].values
            
            for s_name in top_3_sectors:
                idx = df_sectors[df_sectors['sector_name_vi'] == s_name].index[0]
                values = df_all_norm.iloc[idx].values.tolist()
                values.append(values[0]) # Khép kín vòng tròn radar
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=s_name
                ))
                
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="Hồ sơ đa tiêu chí (Radar profile) của Top 3 ngành có chỉ số ưu tiên cao nhất"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            st.caption("💡 Biểu đồ mạng nhện thể hiện độ 'mạnh/yếu' của từng ngành trên mỗi khía cạnh cụ thể sau khi đã chuẩn hóa về thang [0, 1].")

        with tab4:
            st.subheader("💬 Câu 2.5: Báo cáo phân tích và Luận cứ tư vấn chiến lược ngành")
            
            # Tìm tên ngành đứng đầu và đứng cuối tự động từ dữ liệu để trả lời chính xác
            top_1_name = df_ranked.iloc[0]['sector_name_vi']
            bottom_1_name = df_ranked.iloc[-1]['sector_name_vi']
            
            st.markdown(f"""
            * **a) Biện giải kết quả Ngành xếp hạng cao nhất ({top_1_name}):**
                Ngành **{top_1_name}** chiếm vị trí quán quân với điểm số áp đảo nhờ sở hữu **hệ số lan tỏa kinh tế lớn nhất** ($0.78$) và **kim ngạch xuất khẩu dẫn đầu toàn quốc** ($290.9$ tỷ USD). Khi một đồng vốn công đổ vào số hóa ngành này, nó không chỉ kích thích riêng năng suất của nội tại doanh nghiệp sản xuất, mà còn kéo theo sự phát triển hạ tầng và luồng dữ liệu logistics, cung ứng phụ trợ của hàng loạt ngành dịch vụ đi kèm. Do đó, đây là điểm tựa chiến lược để tối ưu hóa sức lan tỏa của dòng vốn.
            * **b) Biện giải kết quả Ngành xếp hạng thấp nhất ({bottom_1_name}):**
                Ngành **{bottom_1_name}** đứng cuối bảng xếp hạng vì có tốc độ tăng trưởng âm, tỷ trọng đóng góp vào cấu trúc GDP thấp, đồng thời mức độ sẵn sàng công nghệ và năng lực hấp thụ AI ở thời điểm hiện tại rất hạn chế. Đầu tư công nghệ vào đây trong ngắn hạn sẽ gặp độ trễ rất lớn và hiệu quả biên thu về cho nền kinh tế tổng thể thấp hơn hẳn so với các khối ngành khác.
            * **c) Đề xuất chính sách phân bổ nguồn lực dựa trên kết quả định lượng:**
                Từ kết quả thực nghiệm đa tiêu chí, Chính phủ nên áp dụng **Chiến lược phân bổ dòng vốn số hóa theo 2 tầng**:
                1. *Tầng Chủ lực (Top 3 ngành đầu):* Tập trung phân bổ nguồn lực số hóa mạnh mẽ để nâng cao lợi thế cạnh tranh quốc tế, xây dựng các tổ hợp nhà máy thông minh, chuỗi cung ứng tự động hóa kết nối toàn cầu.
                2. *Tầng An sinh & Chuyển đổi số diện hẹp:* Đối với các ngành như Khai khoáng hay Nông nghiệp, không nên đầu tư dàn trải theo quy mô vốn lớn, mà tập trung vào các giải pháp số hóa mang tính thích ứng, ví dụ: phần mềm dự báo thời tiết nông nghiệp, số hóa chuỗi logistics phân phối nông sản, hoặc hệ thống cảnh báo an toàn tự động trong khai thác để giảm thiểu rủi ro con người.
            """)
