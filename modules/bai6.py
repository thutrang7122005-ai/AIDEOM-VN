import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def run(df_macro, df_sectors, df_regions):
    st.header("🤖 Bài 6: TOPSIS – Xếp hạng 6 Vùng Kinh tế theo Mức độ Ưu tiên Đầu tư AI")
    st.markdown(
        """
        Áp dụng phương pháp **TOPSIS** (Technique for Order of Preference by Similarity to Ideal Solution)
        để xếp hạng 6 vùng kinh tế – xã hội Việt Nam theo mức độ sẵn sàng triển khai AI,
        dựa trên Quyết định 127/QĐ-TTg ngày 26/01/2021.
        """
    )

    # ── Dữ liệu gốc ──────────────────────────────────────────────────────────
    data = {
        "region_id": [1, 2, 3, 4, 5, 6],
        "region_name_vi": [
            "Trung du & Miền núi phía Bắc",
            "Đồng bằng sông Hồng",
            "Bắc Trung Bộ & DH Trung Bộ",
            "Tây Nguyên",
            "Đông Nam Bộ",
            "Đồng bằng sông Cửu Long",
        ],
        "grdp_per_capita": [57.0,  152.3, 87.5,  68.9,  158.9, 80.5],
        "fdi":             [3.5,   20.0,  8.2,   0.8,   18.5,  2.1],
        "digital_index":   [38,    78,    55,    32,    82,    48],
        "ai_readiness":    [22,    68,    40,    18,    75,    30],
        "trained_labor":   [21.5,  36.8,  27.5,  18.2,  42.5,  16.8],
        "rd_intensity":    [0.18,  0.85,  0.32,  0.15,  0.78,  0.22],
        "internet":        [72,    92,    84,    68,    94,    78],
        "gini":            [0.405, 0.358, 0.372, 0.412, 0.385, 0.392],
    }
    df = pd.DataFrame(data)

    criteria_cols = [
        "grdp_per_capita", "fdi", "digital_index", "ai_readiness",
        "trained_labor", "rd_intensity", "internet", "gini",
    ]
    criteria_labels = [
        "GRDP/người (tr.VND)", "FDI (tỷ USD)", "Digital Index",
        "AI Readiness", "LĐ ĐT (%)", "R&D/GRDP (%)",
        "Internet (%)", "Gini",
    ]
    is_benefit = [True, True, True, True, True, True, True, False]
    w_expert = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10])

    # ── Hàm tiện ích ─────────────────────────────────────────────────────────
    def topsis(X, weights, benefit_flags):
        """Tính TOPSIS – trả về (C_star, S_star, S_neg, V, A_star, A_neg)."""
        R = X / np.sqrt((X ** 2).sum(axis=0))
        V = R * weights
        A_star = np.where(benefit_flags, V.max(axis=0), V.min(axis=0))
        A_neg  = np.where(benefit_flags, V.min(axis=0), V.max(axis=0))
        S_star = np.sqrt(((V - A_star) ** 2).sum(axis=1))
        S_neg  = np.sqrt(((V - A_neg)  ** 2).sum(axis=1))
        C_star = S_neg / (S_star + S_neg)
        return C_star, S_star, S_neg, V, A_star, A_neg

    def entropy_weights(X):
        """Tính trọng số khách quan bằng phương pháp Entropy."""
        P = X / X.sum(axis=0)
        k = 1.0 / np.log(len(X))
        E = -k * np.nansum(P * np.log(P + 1e-12), axis=0)
        d = 1 - E
        return d / d.sum()

    X = df[criteria_cols].values.astype(float)
    is_benefit_arr = np.array(is_benefit)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dữ liệu & Trọng số",
        "🏆 TOPSIS – Trọng số Chuyên gia",
        "⚖️ TOPSIS – Trọng số Entropy",
        "🔍 Phân tích Độ nhạy",
        "💬 Thảo luận Chính sách",
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 – Dữ liệu & Trọng số
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        st.subheader("📋 Ma trận quyết định ban đầu")
        display_df = df[["region_name_vi"] + criteria_cols].copy()
        display_df.columns = ["Vùng"] + criteria_labels
        st.dataframe(display_df.set_index("Vùng"), use_container_width=True)

        st.subheader("⚙️ Trọng số chuyên gia (Expert Weights)")
        w_df = pd.DataFrame({
            "Tiêu chí": criteria_labels,
            "Loại": ["Lợi ích" if b else "Chi phí" for b in is_benefit],
            "Trọng số": w_expert,
        })
        fig_w = px.bar(
            w_df, x="Tiêu chí", y="Trọng số", color="Loại",
            color_discrete_map={"Lợi ích": "#2196F3", "Chi phí": "#F44336"},
            text_auto=".2f", title="Trọng số chuyên gia cho 8 tiêu chí",
        )
        fig_w.update_layout(xaxis_tickangle=-30, height=380)
        st.plotly_chart(fig_w, use_container_width=True)

        st.subheader("🔢 Trọng số Entropy (Khách quan)")
        w_entropy = entropy_weights(X)
        w_ent_df = pd.DataFrame({
            "Tiêu chí": criteria_labels,
            "Trọng số Chuyên gia": w_expert,
            "Trọng số Entropy": w_entropy,
        })
        fig_wcomp = go.Figure()
        fig_wcomp.add_trace(go.Bar(
            name="Chuyên gia", x=criteria_labels, y=w_expert,
            marker_color="#2196F3", text=[f"{v:.3f}" for v in w_expert],
            textposition="outside",
        ))
        fig_wcomp.add_trace(go.Bar(
            name="Entropy", x=criteria_labels, y=w_entropy,
            marker_color="#FF9800", text=[f"{v:.3f}" for v in w_entropy],
            textposition="outside",
        ))
        fig_wcomp.update_layout(
            barmode="group", title="So sánh trọng số Chuyên gia vs Entropy",
            xaxis_tickangle=-30, height=400,
        )
        st.plotly_chart(fig_wcomp, use_container_width=True)
        st.dataframe(w_ent_df.set_index("Tiêu chí").style.format("{:.4f}"),
                     use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 – TOPSIS Trọng số Chuyên gia
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("🧮 Câu 6.4.1 – TOPSIS với Trọng số Chuyên gia")
        C_exp, S_star_exp, S_neg_exp, V_exp, A_star_exp, A_neg_exp = topsis(
            X, w_expert, is_benefit_arr
        )

        df_res_exp = df[["region_name_vi"]].copy()
        df_res_exp["S*  (kc đến lý tưởng+)"] = S_star_exp
        df_res_exp["S⁻  (kc đến lý tưởng-)"] = S_neg_exp
        df_res_exp["C*  (hệ số gần gũi)"]     = C_exp
        df_res_exp["Xếp hạng"] = df_res_exp["C*  (hệ số gần gũi)"].rank(
            ascending=False
        ).astype(int)
        df_res_exp = df_res_exp.sort_values("Xếp hạng")

        st.markdown("#### Kết quả TOPSIS")
        styled = df_res_exp.set_index("region_name_vi").style.format({
            "S*  (kc đến lý tưởng+)": "{:.4f}",
            "S⁻  (kc đến lý tưởng-)": "{:.4f}",
            "C*  (hệ số gần gũi)":     "{:.4f}",
        }).background_gradient(subset=["C*  (hệ số gần gũi)"], cmap="Blues")
        st.dataframe(styled, use_container_width=True)

        # Bar chart xếp hạng
        fig_bar = px.bar(
            df_res_exp.sort_values("C*  (hệ số gần gũi)", ascending=True),
            x="C*  (hệ số gần gũi)", y="region_name_vi",
            orientation="h", color="C*  (hệ số gần gũi)",
            color_continuous_scale="Blues",
            text=df_res_exp.sort_values("C*  (hệ số gần gũi)", ascending=True)[
                "C*  (hệ số gần gũi)"
            ].apply(lambda v: f"{v:.4f}"),
            title="Hệ số gần gũi C* – Trọng số Chuyên gia",
            labels={"region_name_vi": "Vùng", "C*  (hệ số gần gũi)": "C*"},
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(height=380, showlegend=False,
                               coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

        # Ma trận chuẩn hóa có trọng số
        with st.expander("🔬 Chi tiết ma trận chuẩn hóa có trọng số (V)"):
            V_df = pd.DataFrame(
                V_exp, columns=criteria_labels,
                index=df["region_name_vi"]
            )
            st.dataframe(V_df.style.format("{:.4f}").background_gradient(cmap="YlOrRd"),
                         use_container_width=True)
            st.markdown("**Lời giải lý tưởng A\*:**")
            st.dataframe(
                pd.DataFrame([A_star_exp, A_neg_exp],
                             index=["A* (tốt nhất)", "A⁻ (xấu nhất)"],
                             columns=criteria_labels).style.format("{:.4f}"),
                use_container_width=True,
            )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 – TOPSIS Trọng số Entropy
    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("⚖️ Câu 6.4.2 – TOPSIS với Trọng số Entropy")
        w_entropy = entropy_weights(X)
        C_ent, S_star_ent, S_neg_ent, V_ent, A_star_ent, A_neg_ent = topsis(
            X, w_entropy, is_benefit_arr
        )

        df_res_ent = df[["region_name_vi"]].copy()
        df_res_ent["C* Entropy"]     = C_ent
        df_res_ent["Hạng Entropy"]   = df_res_ent["C* Entropy"].rank(ascending=False).astype(int)
        df_res_ent["C* Chuyên gia"]  = C_exp
        df_res_ent["Hạng CG"]        = df_res_ent["C* Chuyên gia"].rank(ascending=False).astype(int)
        df_res_ent["Δ Hạng"]         = df_res_ent["Hạng CG"] - df_res_ent["Hạng Entropy"]
        df_res_ent = df_res_ent.sort_values("Hạng Entropy")

        st.markdown("#### So sánh xếp hạng: Chuyên gia vs Entropy")
        st.dataframe(
            df_res_ent.set_index("region_name_vi")[[
                "C* Chuyên gia", "Hạng CG", "C* Entropy", "Hạng Entropy", "Δ Hạng"
            ]].style.format({
                "C* Chuyên gia": "{:.4f}", "C* Entropy": "{:.4f}", "Δ Hạng": "{:+d}"
            }).background_gradient(subset=["Δ Hạng"], cmap="RdYlGn"),
            use_container_width=True,
        )

        # Scatter so sánh
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=df_res_ent["region_name_vi"], y=df_res_ent["C* Chuyên gia"],
            mode="lines+markers", name="Trọng số Chuyên gia",
            marker=dict(size=10, color="#2196F3"), line=dict(width=2),
        ))
        fig_sc.add_trace(go.Scatter(
            x=df_res_ent["region_name_vi"], y=df_res_ent["C* Entropy"],
            mode="lines+markers", name="Trọng số Entropy",
            marker=dict(size=10, color="#FF9800", symbol="diamond"),
            line=dict(width=2, dash="dash"),
        ))
        fig_sc.update_layout(
            title="So sánh hệ số C* theo hai phương pháp trọng số",
            xaxis_title="Vùng kinh tế", yaxis_title="Hệ số C*",
            height=420, xaxis_tickangle=-20,
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        # Trọng số Entropy
        st.markdown("#### Trọng số Entropy chi tiết")
        w_ent_detail = pd.DataFrame({
            "Tiêu chí": criteria_labels,
            "Entropy": entropy_weights(X),
        }).sort_values("Entropy", ascending=False)
        fig_ent = px.bar(
            w_ent_detail, x="Tiêu chí", y="Entropy",
            text_auto=".4f", color="Entropy",
            color_continuous_scale="Oranges",
            title="Trọng số Entropy (phản ánh mức độ phân tán dữ liệu)",
        )
        fig_ent.update_layout(xaxis_tickangle=-30, height=380,
                               coloraxis_showscale=False)
        st.plotly_chart(fig_ent, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 – Phân tích Độ nhạy
    # ════════════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("🔍 Câu 6.4.3 – Phân tích Độ nhạy: Thay đổi w_AI Readiness")
        st.markdown(
            "Tăng trọng số **AI Readiness** từ 0.10 → 0.40 và phân phối lại "
            "phần còn lại tỷ lệ cho các tiêu chí khác."
        )

        ai_idx = 3  # index của ai_readiness trong criteria_cols
        w_ai_range = np.linspace(0.10, 0.40, 25)
        records = []

        for w_ai in w_ai_range:
            # Giảm tỷ lệ phần còn lại (trừ AI index)
            remaining = w_expert.copy()
            remaining[ai_idx] = 0.0
            scale = (1.0 - w_ai) / remaining.sum()
            w_new = remaining * scale
            w_new[ai_idx] = w_ai
            C_sens, _, _, _, _, _ = topsis(X, w_new, is_benefit_arr)
            ranks = pd.Series(C_sens).rank(ascending=False).astype(int).values
            for i, name in enumerate(df["region_name_vi"]):
                records.append({
                    "w_AI": round(w_ai, 3),
                    "Vùng": name,
                    "C*": C_sens[i],
                    "Hạng": ranks[i],
                })

        df_sens = pd.DataFrame(records)

        # Line chart C*
        fig_sens = px.line(
            df_sens, x="w_AI", y="C*", color="Vùng",
            title="Hệ số C* theo trọng số w(AI Readiness)",
            labels={"w_AI": "w(AI Readiness)", "C*": "Hệ số C*"},
            markers=True,
        )
        fig_sens.update_layout(height=450)
        st.plotly_chart(fig_sens, use_container_width=True)

        # Xếp hạng theo w_AI
        fig_rank = px.line(
            df_sens, x="w_AI", y="Hạng", color="Vùng",
            title="Thứ hạng theo trọng số w(AI Readiness)",
            labels={"w_AI": "w(AI Readiness)", "Hạng": "Xếp hạng (1=tốt nhất)"},
            markers=True,
        )
        fig_rank.update_yaxes(autorange="reversed", dtick=1)
        fig_rank.update_layout(height=430)
        st.plotly_chart(fig_rank, use_container_width=True)

        # Kiểm tra ổn định Top-3
        st.markdown("#### 📌 Kiểm tra tính ổn định của Top-3")
        top3_counts = (
            df_sens[df_sens["Hạng"] <= 3]
            .groupby("Vùng")["w_AI"]
            .count()
            .reset_index()
            .rename(columns={"w_AI": "Số lần trong Top-3"})
            .sort_values("Số lần trong Top-3", ascending=False)
        )
        top3_counts["Tỷ lệ (%)"] = (top3_counts["Số lần trong Top-3"] / len(w_ai_range) * 100).round(1)
        st.dataframe(top3_counts.set_index("Vùng"), use_container_width=True)

        stable = top3_counts[top3_counts["Tỷ lệ (%)"] == 100.0]["Vùng"].tolist()
        if stable:
            st.success(f"✅ Các vùng **ổn định trong Top-3** suốt toàn bộ dải w_AI: {', '.join(stable)}")
        else:
            st.warning("⚠️ Không có vùng nào duy trì Top-3 ổn định hoàn toàn – cần xem xét thêm.")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 5 – Thảo luận Chính sách
    # ════════════════════════════════════════════════════════════════════════
    with tab5:
        st.subheader("💬 Thảo luận Chính sách (Câu 6.5)")

        C_exp_arr, _, _, _, _, _ = topsis(X, w_expert, is_benefit_arr)
        C_ent_arr, _, _, _, _, _ = topsis(X, entropy_weights(X), is_benefit_arr)
        rank_exp = pd.Series(C_exp_arr).rank(ascending=False).astype(int).values
        rank_ent = pd.Series(C_ent_arr).rank(ascending=False).astype(int).values

        region_names = df["region_name_vi"].tolist()
        top1_exp = region_names[int(np.argmax(C_exp_arr))]
        top1_ent = region_names[int(np.argmax(C_ent_arr))]

        max_shift_idx = int(np.argmax(np.abs(rank_ent - rank_exp)))
        max_shift_region = region_names[max_shift_idx]
        max_shift_val = int(rank_ent[max_shift_idx]) - int(rank_exp[max_shift_idx])

        top3_exp = sorted(range(6), key=lambda i: C_exp_arr[i], reverse=True)[:3]
        top3_names = [region_names[i] for i in top3_exp]

        st.markdown(f"""
**a) Vùng dẫn đầu theo TOPSIS (trọng số chuyên gia):**

Vùng **{top1_exp}** đứng đầu với hệ số C* cao nhất.
Đây là vùng hội tụ đủ các điều kiện hạ tầng số, nguồn lực FDI,
và lực lượng lao động qua đào tạo – là ứng viên hàng đầu cho
**Trung tâm AI Quốc gia đầu tiên**.

---

**b) Vùng thay đổi xếp hạng lớn nhất khi dùng trọng số Entropy:**

Vùng **{max_shift_region}** có sự thay đổi lớn nhất
(Δ = {max_shift_val:+d} bậc).
Điều này phản ánh vùng này có **mức độ phân tán dữ liệu cao**
trên các tiêu chí mà Entropy đánh giá nặng hơn (ví dụ: R&D/GRDP,
FDI) – ý nghĩa thực tế là tiêu chí đó *phân biệt* mạnh hơn giữa
các vùng.

---

**c) Giả định độc lập tuyến tính trong TOPSIS:**

TOPSIS giả định các tiêu chí **độc lập** với nhau. Tuy nhiên trong
thực tế:
- *AI Readiness* và *Internet Penetration* có thể tương quan rất
  cao (r > 0.9).
- *GRDP/người* và *FDI* cũng thường đồng biến.

**Hệ quả**: Các tiêu chí tương quan cao sẽ **nhân đôi ảnh hưởng**,
làm méo kết quả. **Đề xuất xử lý**:
1. Dùng **PCA** (Principal Component Analysis) để giảm chiều trước
   khi áp dụng TOPSIS.
2. Kiểm tra ma trận tương quan và **loại/gộp** tiêu chí trùng lặp.
3. Sử dụng **AHP** để xây dựng trọng số có xét đến sự phụ thuộc.

---

**d) Lựa chọn 3 vùng để xây dựng Trung tâm AI (theo Quyết định 127/QĐ-TTg):**

Dựa trên kết quả TOPSIS, đề xuất 3 vùng ưu tiên:
""")

        for i, name in enumerate(top3_names, 1):
            idx = region_names.index(name)
            st.markdown(
                f"- **{i}. {name}** — C* = {C_exp_arr[idx]:.4f}"
            )

        st.markdown("""
Ngoài chỉ số định lượng, cần **điều chỉnh thêm tiêu chí địa-chính trị**:
- **Vị trí chiến lược**: gần cửa ngõ quốc tế, cảng biển, sân bay lớn.
- **An ninh dữ liệu**: mức độ ổn định chính trị, cơ sở pháp lý sandbox.
- **Tính đại diện vùng miền**: đảm bảo không tập trung AI chỉ ở một
  cực tăng trưởng, tránh tạo ra *AI divide* giữa các vùng.
- **Hệ sinh thái startup & trường ĐH**: sự hiện diện của các trường
  kỹ thuật lớn (ĐHBK, ĐHQG…) là điều kiện quan trọng cho R&D dài hạn.
""")

        # Radar chart Top-3 vs All
        st.markdown("#### 🕸️ Radar Chart – So sánh đa tiêu chí Top-3 Vùng")
        radar_cols = criteria_labels[:7]  # bỏ Gini để đơn giản
        radar_indices = criteria_cols[:7]
        X_norm = (X[:, :7] - X[:, :7].min(axis=0)) / (
            X[:, :7].max(axis=0) - X[:, :7].min(axis=0) + 1e-9
        )

        fig_radar = go.Figure()
        colors_radar = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336", "#00BCD4"]
        for i, (name, color) in enumerate(zip(region_names, colors_radar)):
            vals = list(X_norm[i]) + [X_norm[i][0]]
            angles = radar_cols + [radar_cols[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=angles, fill="toself",
                name=name, line_color=color,
                opacity=0.7 if i in top3_exp else 0.3,
                line_width=3 if i in top3_exp else 1,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Radar Chart – Chuẩn hóa Min-Max (không tính Gini)",
            height=520,
        )
        st.plotly_chart(fig_radar, use_container_width=True)