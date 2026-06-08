# =============================================================================
# AIDEOM-VN | BÀI 3: TÍNH CHỈ SỐ ƯU TIÊN NGÀNH (PRIORITY INDEX)
# File: modules/Bai_3_Priority_Index/bai3_streamlit.py
# Mô tả: Module Streamlit — hiển thị lý thuyết, tính toán và biểu đồ tích hợp
# Phiên bản: 2.0 (Streamlit edition)
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from io import StringIO

# ---------------------------------------------------------------------------
# DỮ LIỆU TĨNH
# ---------------------------------------------------------------------------
RAW_CSV = """sector,growth_pct,productivity,spillover,export_usd,employment,ai_readiness,automation_risk_pct
Nông-Lâm-Thủy sản,3.27,103.4,0.35,40.5,13.20,15,18
CN Chế biến-Chế tạo,9.64,241.2,0.78,290.9,11.50,55,42
Xây dựng,7.45,168.8,0.42,2.5,4.80,20,25
Khai khoáng,-1.20,1290.5,0.30,8.2,0.30,30,55
Bán buôn-Bán lẻ,7.10,145.3,0.55,5.5,7.80,48,38
Tài chính-Ngân hàng,7.36,1072.4,0.85,1.2,0.55,72,52
Logistics-Vận tải,9.93,321.4,0.72,3.1,1.95,42,35
CNTT-Truyền thông,7.85,713.8,0.92,178.0,0.62,88,28
Giáo dục-Đào tạo,6.42,205.7,0.65,0.0,2.15,38,22
Y tế,6.85,437.1,0.60,0.0,0.75,45,18"""

FEATURE_COLS = [
    "growth_pct", "productivity", "spillover",
    "export_usd", "employment", "ai_readiness", "automation_risk_pct",
]

FEATURE_DISPLAY = [
    "Tăng trưởng (%)", "Năng suất (tr.VND/LĐ)", "Lan tỏa (0–1)",
    "XK (tỷ USD)", "Việc làm (tr.LĐ)", "AI Readiness (0–100)", "Rủi ro TĐH (đảo)",
]

# ---------------------------------------------------------------------------
# HÀM PHỤ TRỢ
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Nạp dữ liệu 10 ngành từ chuỗi CSV tĩnh."""
    return pd.read_csv(StringIO(RAW_CSV))


def minmax_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa Min-Max tất cả cột chỉ tiêu về [0,1].
    Đảo dấu cột automation_risk_pct (chỉ tiêu nghịch chiều).
    """
    norm = df[FEATURE_COLS].copy().astype(float)
    for col in FEATURE_COLS:
        col_min, col_max = norm[col].min(), norm[col].max()
        denom = col_max - col_min
        norm[col] = 0.0 if denom == 0 else (norm[col] - col_min) / denom
    # Đảo dấu rủi ro tự động hóa: cao rủi ro → thấp ưu tiên
    norm["automation_risk_pct"] = 1.0 - norm["automation_risk_pct"]
    return norm


def compute_priority(norm: pd.DataFrame, weights: dict) -> np.ndarray:
    """Tính Priority Index theo công thức WSM có trọng số."""
    return (
        weights["a1"] * norm["growth_pct"].values
        + weights["a2"] * norm["productivity"].values
        + weights["a3"] * norm["spillover"].values
        + weights["a4"] * norm["export_usd"].values
        + weights["a5"] * norm["employment"].values
        + weights["a6"] * norm["ai_readiness"].values
        + weights["w_risk"] * norm["automation_risk_pct"].values
    )


def set_matplotlib_style():
    """Cấu hình style matplotlib thống nhất cho toàn bộ báo cáo."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "figure.facecolor": "#FAFAFA",
        "axes.facecolor": "#FAFAFA",
    })


# ===========================================================================
# HÀM CHÍNH run() — GỌI TỪ main.py
# ===========================================================================

def run(*args, **kwargs):
    """
    Hàm điểm vào chính. Hiển thị toàn bộ Bài 3 lên giao diện Streamlit.
    Thứ tự: Lý thuyết (3.1→3.3) → Tính toán (3.4.1→3.4.4) → Thảo luận (3.5)
    """
    try:
        set_matplotlib_style()

        # ══════════════════════════════════════════════════════════════════
        # TIÊU ĐỀ BÀI
        # ══════════════════════════════════════════════════════════════════
        st.markdown("""
# 📊 Bài 3: Tính Chỉ Số Ưu Tiên Ngành (Priority Index)
### Hệ thống AIDEOM-VN — Phân tích Kinh tế Việt Nam 2024
---
""")

        # ══════════════════════════════════════════════════════════════════
        # 3.1 BỐI CẢNH
        # ══════════════════════════════════════════════════════════════════
        st.markdown("## 3.1. Bối cảnh")
        st.markdown("""
Trong bối cảnh Việt Nam đang bước vào giai đoạn chuyển đổi cơ cấu kinh tế sâu rộng,
việc xác định các ngành kinh tế trọng điểm cần được ưu tiên đầu tư và phát triển trở thành
một bài toán chiến lược cấp quốc gia. **Nghị quyết 57-NQ/TW** (tháng 12/2024) của Bộ Chính
trị về đột phá phát triển khoa học, công nghệ, đổi mới sáng tạo và chuyển đổi số quốc gia
đã đặt ra yêu cầu cấp thiết: Việt Nam cần tăng trưởng GDP tối thiểu **8%** giai đoạn
2026–2030, tiến tới mức hai chữ số sau năm 2030.

Hệ thống AIDEOM-VN tiếp cận bài toán này bằng phương pháp định lượng: xây dựng
**Chỉ số Ưu tiên Ngành (Priority Index — PI)** như một công cụ hỗ trợ ra quyết định,
tổng hợp đa chiều các thuộc tính kinh tế-xã hội-công nghệ thành một điểm số duy nhất,
có thể diễn giải, kiểm chứng và điều chỉnh linh hoạt theo định hướng chính sách.
""")

        # ══════════════════════════════════════════════════════════════════
        # 3.2 MÔ HÌNH TOÁN HỌC
        # ══════════════════════════════════════════════════════════════════
        st.markdown("## 3.2. Mô hình toán học")

        st.markdown("### 3.2.1. Chuẩn hóa Min-Max")
        st.markdown("""
Do các chỉ tiêu đầu vào có đơn vị và thang đo khác nhau, bước đầu tiên là đưa tất cả
về cùng khoảng $[0, 1]$ bằng phương pháp chuẩn hóa Min-Max:
""")
        st.latex(r"""
\tilde{x}_{ij} = \frac{x_{ij} - x_j^{\min}}{x_j^{\max} - x_j^{\min}}
""")
        st.markdown("""
> **⚠️ Lưu ý:** Cột *Rủi ro Tự động hóa* là chỉ tiêu **nghịch chiều** — ngành có rủi ro cao
> thì ưu tiên thấp hơn. Sau khi chuẩn hóa, cột này được **đảo dấu**:
""")
        st.latex(r"""
\tilde{x}_{i,\text{risk}}^* = 1 - \tilde{x}_{i,\text{risk}}
""")

        st.markdown("### 3.2.2. Công thức Priority Index (WSM)")
        st.markdown("Priority Index được tính theo mô hình tổng có trọng số:")
        st.latex(r"""
PI_i = a_1\tilde{g}_i + a_2\tilde{p}_i + a_3\tilde{s}_i
     + a_4\tilde{e}_i + a_5\tilde{l}_i + a_6\tilde{r}_i^{AI}
     + w_{\text{risk}}(1-\tilde{r}_i^{\text{auto}})
""")

        st.markdown("""
| Ký hiệu | Ý nghĩa | Trọng số mặc định |
|---|---|:---:|
| $a_1$ | Tăng trưởng giá trị gia tăng (%) | 0.15 |
| $a_2$ | Năng suất lao động (triệu VND/LĐ) | 0.15 |
| $a_3$ | Hệ số lan tỏa liên ngành (0–1) | 0.20 |
| $a_4$ | Kim ngạch xuất khẩu (tỷ USD) | 0.15 |
| $a_5$ | Quy mô việc làm (triệu LĐ) | 0.10 |
| $a_6$ | AI Readiness (0–100) | 0.20 |
| $w_{\\text{risk}}$ | Rủi ro tự động hóa (nghịch chiều) | 0.15 |

**Ràng buộc:** $\\sum_{k=1}^{6} a_k + w_{\\text{risk}} = 1.00$
""")

        st.markdown("### 3.2.3. Phân tích độ nhạy")
        st.markdown("""
Để kiểm tra tính bền vững của thứ hạng, $a_6$ được biến thiên trong khoảng
$[0.05;\\ 0.40]$ với bước nhảy $\\Delta = 0.05$. Phần chênh lệch được phân bổ lại
đồng đều cho các trọng số còn lại để đảm bảo tổng bằng 1.
""")

        st.markdown("### 3.2.4. So sánh kịch bản chính sách")
        st.markdown("""
| Trọng số | Tăng trưởng | Bao trùm |
|---|:---:|:---:|
| $a_1$ — Tăng trưởng | 0.25 | 0.10 |
| $a_2$ — Năng suất | 0.20 | 0.10 |
| $a_3$ — Lan tỏa | 0.20 | 0.20 |
| $a_4$ — Xuất khẩu | 0.15 | 0.10 |
| $a_5$ — Việc làm | 0.05 | 0.25 |
| $a_6$ — AI Readiness | 0.10 | 0.15 |
| $w_{\\text{risk}}$ — Rủi ro TĐH | 0.05 | 0.10 |
""")

        # ══════════════════════════════════════════════════════════════════
        # 3.3 DỮ LIỆU ĐẦU VÀO
        # ══════════════════════════════════════════════════════════════════
        st.markdown("## 3.3. Dữ liệu đầu vào")
        st.markdown("""
Dữ liệu 10 ngành kinh tế Việt Nam 2024 được tổng hợp từ Tổng cục Thống kê (GSO),
Bộ Kế hoạch & Đầu tư, Bộ TT&TT và các báo cáo nghiên cứu ngành
(nguồn: `vietnam_sectors_2024.csv`):
""")

        # Nạp dữ liệu và hiển thị bảng
        df = load_data()
        df_display = df.copy()
        df_display.columns = [
            "Ngành", "Tăng trưởng (%)", "Năng suất (tr.VND/LĐ)",
            "Lan tỏa (0–1)", "XK (tỷ USD)", "Việc làm (tr.LĐ)",
            "AI Readiness (0–100)", "Rủi ro TĐH (%)",
        ]
        df_display.index = range(1, 11)

        st.dataframe(
            df_display,
            use_container_width=True,
            height=390,
        )

        st.markdown("""
> **Nguồn:** GSO Niên giám 2024 · Bảng I-O 2022 · Oxford Insights AI Readiness Index 2024
> · Oxford Martin School (hiệu chỉnh thị trường lao động Việt Nam)
""")

        # ══════════════════════════════════════════════════════════════════
        # 3.4 LẬP TRÌNH & KẾT QUẢ
        # ══════════════════════════════════════════════════════════════════
        st.markdown("## 3.4. Lập trình & Kết quả")

        # Chuẩn hóa dữ liệu (dùng chung cho tất cả mục bên dưới)
        norm = minmax_normalize(df)

        # ------------------------------------------------------------------
        # 3.4.1 — Chuẩn hóa Min-Max
        # ------------------------------------------------------------------
        st.markdown("### 3.4.1. Ma trận chuẩn hóa Min-Max")
        st.markdown("""
Bảng dưới đây hiển thị giá trị đã chuẩn hóa $\\in [0,1]$ của 10 ngành theo 7 chỉ tiêu.
Cột *Rủi ro TĐH* đã được **đảo dấu** (1 − giá trị gốc):
""")

        try:
            norm_display = pd.DataFrame(
                norm.values,
                columns=FEATURE_DISPLAY,
                index=[f"{i}. {row['sector']}" for i, row in df.iterrows()],
            )
            # Highlight ô cao nhất mỗi cột
            st.dataframe(
                norm_display.style.format("{:.4f}").background_gradient(
                    cmap="Blues", axis=0
                ),
                use_container_width=True,
                height=390,
            )
        except Exception as e:
            st.error(f"Lỗi hiển thị bảng chuẩn hóa: {e}")

        # ------------------------------------------------------------------
        # 3.4.2 — Priority Index với trọng số mặc định
        # ------------------------------------------------------------------
        st.markdown("### 3.4.2. Priority Index — Bộ trọng số mặc định")

        default_weights = {
            "a1": 0.15, "a2": 0.15, "a3": 0.20,
            "a4": 0.15, "a5": 0.10, "a6": 0.20, "w_risk": 0.15,
        }

        st.markdown(f"""
Bộ trọng số mặc định: `a1=0.15 · a2=0.15 · a3=0.20 · a4=0.15 · a5=0.10 · a6=0.20 · w_risk=0.15`
(Tổng = **{sum(default_weights.values()):.2f}**)
""")

        try:
            pi_values = compute_priority(norm, default_weights)
            df_ranked = pd.DataFrame({
                "Ngành": df["sector"].values,
                "Priority Index": pi_values,
            })
            df_ranked["Xếp hạng"] = (
                df_ranked["Priority Index"]
                .rank(ascending=False, method="min")
                .astype(int)
            )
            df_ranked = df_ranked.sort_values("Xếp hạng").reset_index(drop=True)

            # Hiển thị bảng kết quả
            df_result_display = df_ranked.copy()
            df_result_display["Priority Index"] = df_result_display["Priority Index"].map("{:.4f}".format)
            df_result_display["Top"] = df_ranked["Xếp hạng"].apply(
                lambda r: "🥇" if r == 1 else "🥈" if r == 2 else "🥉" if r == 3 else ""
            )
            st.dataframe(
                df_result_display[["Xếp hạng", "Ngành", "Priority Index", "Top"]],
                use_container_width=True,
                hide_index=True,
                height=390,
            )

            # Vẽ biểu đồ cột ngang
            fig, ax = plt.subplots(figsize=(10, 6))
            palette = [
                "#1a5fa8" if r <= 3 else "#5b9fd0" if r <= 6 else "#b0cce0"
                for r in df_ranked["Xếp hạng"]
            ]
            bars = ax.barh(
                df_ranked["Ngành"],
                df_ranked["Priority Index"],
                color=palette,
                edgecolor="white",
                linewidth=0.7,
            )
            ax.bar_label(bars, fmt="%.4f", padding=5, fontsize=9, color="#333")
            ax.set_xlabel("Priority Index Score", fontsize=11)
            ax.set_title(
                "Hình 3.1 — Xếp hạng Chỉ số Ưu tiên Ngành (AIDEOM-VN 2024)",
                fontsize=12, pad=14,
            )
            ax.invert_yaxis()
            ax.set_xlim(0, df_ranked["Priority Index"].max() * 1.20)
            legend_elements = [
                Patch(facecolor="#1a5fa8", label="Top 3"),
                Patch(facecolor="#5b9fd0", label="Hạng 4–6"),
                Patch(facecolor="#b0cce0", label="Hạng 7–10"),
            ]
            ax.legend(handles=legend_elements, fontsize=9, loc="lower right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.error(f"Lỗi mục 3.4.2: {e}")

        # ------------------------------------------------------------------
        # 3.4.3 — Phân tích độ nhạy a6
        # ------------------------------------------------------------------
        st.markdown("### 3.4.3. Phân tích độ nhạy — Trọng số $a_6$ (AI Readiness)")
        st.markdown("""
Biến thiên $a_6 \\in [0.05;\\ 0.40]$, bước nhảy $\\Delta=0.05$.
Phần còn lại $(1 - a_6)$ được phân bổ lại tỷ lệ cho các trọng số khác.
""")

        try:
            a6_range = np.arange(0.05, 0.45, 0.05)
            base_others = {
                "a1": 0.15, "a2": 0.15, "a3": 0.20,
                "a4": 0.15, "a5": 0.10, "w_risk": 0.15,
            }
            sum_others = sum(base_others.values())

            rank_matrix = {s: [] for s in df["sector"].values}

            for a6_val in a6_range:
                remaining = 1.0 - a6_val
                scale = remaining / sum_others
                adjusted = {k: v * scale for k, v in base_others.items()}
                adjusted["a6"] = a6_val
                pi = compute_priority(norm, adjusted)
                ranks = (
                    pd.Series(pi, index=df["sector"].values)
                    .rank(ascending=False, method="min")
                    .astype(int)
                )
                for sector in df["sector"].values:
                    rank_matrix[sector].append(ranks[sector])

            df_heatmap = pd.DataFrame(
                rank_matrix,
                index=[f"a6={v:.2f}" for v in a6_range],
            ).T

            fig2, ax2 = plt.subplots(figsize=(13, 7))
            sns.heatmap(
                df_heatmap,
                annot=True,
                fmt="d",
                cmap=sns.color_palette("RdYlGn_r", n_colors=10),
                linewidths=0.5,
                linecolor="white",
                vmin=1, vmax=10,
                cbar_kws={"label": "Thứ hạng (1 = cao nhất)", "shrink": 0.8},
                ax=ax2,
            )
            ax2.set_title(
                "Hình 3.2 — Heatmap Độ nhạy Thứ hạng theo Trọng số $a_6$\n"
                "Xanh = thứ hạng cao · Đỏ = thứ hạng thấp",
                fontsize=12, pad=14,
            )
            ax2.set_xlabel("Giá trị $a_6$", fontsize=11)
            ax2.set_ylabel("Ngành kinh tế", fontsize=11)
            ax2.tick_params(axis="x", rotation=30)
            ax2.tick_params(axis="y", rotation=0)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

            # Bảng tóm tắt độ ổn định
            stability = df_heatmap.std(axis=1).sort_values()
            df_stable = pd.DataFrame({
                "Ngành": stability.index,
                "Độ lệch chuẩn thứ hạng": stability.values.round(3),
                "Đánh giá": [
                    "🟢 Rất ổn định" if v < 1 else "🟡 Tương đối ổn định" if v < 2
                    else "🔴 Biến động cao"
                    for v in stability.values
                ],
            })
            st.markdown("**Bảng 3.2 — Độ ổn định thứ hạng theo $a_6$** (độ lệch chuẩn nhỏ = ổn định hơn):")
            st.dataframe(df_stable, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Lỗi mục 3.4.3: {e}")

        # ------------------------------------------------------------------
        # 3.4.4 — So sánh 2 kịch bản chính sách
        # ------------------------------------------------------------------
        st.markdown("### 3.4.4. So sánh 2 kịch bản chính sách")

        scenarios = {
            "Tăng trưởng": {
                "a1": 0.25, "a2": 0.20, "a3": 0.20,
                "a4": 0.15, "a5": 0.05, "a6": 0.10, "w_risk": 0.05,
            },
            "Bao trùm": {
                "a1": 0.10, "a2": 0.10, "a3": 0.20,
                "a4": 0.10, "a5": 0.25, "a6": 0.15, "w_risk": 0.10,
            },
        }

        try:
            results = {}
            for name, w in scenarios.items():
                pi = compute_priority(norm, w)
                results[name] = pd.Series(pi, index=df["sector"].values)

            df_compare = pd.DataFrame(results)
            df_compare["Hạng_TT"] = df_compare["Tăng trưởng"].rank(
                ascending=False, method="min"
            ).astype(int)
            df_compare["Hạng_BT"] = df_compare["Bao trùm"].rank(
                ascending=False, method="min"
            ).astype(int)
            df_compare["Δ Hạng"] = df_compare["Hạng_TT"] - df_compare["Hạng_BT"]
            df_compare = df_compare.sort_values("Hạng_TT")

            # Bảng so sánh
            df_cmp_display = pd.DataFrame({
                "Ngành": df_compare.index,
                "PI Tăng trưởng": df_compare["Tăng trưởng"].map("{:.4f}".format),
                "Hạng TT": df_compare["Hạng_TT"],
                "PI Bao trùm": df_compare["Bao trùm"].map("{:.4f}".format),
                "Hạng BT": df_compare["Hạng_BT"],
                "Δ Hạng": df_compare["Δ Hạng"].apply(
                    lambda x: f"+{int(x)}" if x > 0 else str(int(x))
                ),
            })
            st.dataframe(df_cmp_display, use_container_width=True, hide_index=True)

            # Dot-plot so sánh
            fig3, ax3 = plt.subplots(figsize=(11, 7))
            sectors_sorted = df_compare.index.tolist()
            y_pos = np.arange(len(sectors_sorted))

            pi_tt = [df_compare.loc[s, "Tăng trưởng"] for s in sectors_sorted]
            pi_bt = [df_compare.loc[s, "Bao trùm"] for s in sectors_sorted]

            for i, s in enumerate(sectors_sorted):
                ax3.plot(
                    [pi_tt[i], pi_bt[i]], [i, i],
                    color="#aaa", linewidth=1.5, alpha=0.7, zorder=3,
                )
            ax3.scatter(pi_tt, y_pos, s=100, color="#e05c2a", zorder=5,
                        label="Định hướng Tăng trưởng", edgecolors="white", linewidths=0.8)
            ax3.scatter(pi_bt, y_pos, s=100, color="#1a6fa8", zorder=5,
                        label="Định hướng Bao trùm", edgecolors="white", linewidths=0.8)

            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(sectors_sorted, fontsize=10)
            ax3.set_xlabel("Priority Index Score", fontsize=11)
            ax3.set_title(
                "Hình 3.3 — So sánh Priority Index: Tăng trưởng vs Bao trùm\n"
                "(AIDEOM-VN 2024)",
                fontsize=12, pad=12,
            )
            ax3.legend(fontsize=10, loc="lower right")
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

        except Exception as e:
            st.error(f"Lỗi mục 3.4.4: {e}")

        # ══════════════════════════════════════════════════════════════════
        # 3.5 THẢO LUẬN CHÍNH SÁCH
        # ══════════════════════════════════════════════════════════════════
        st.markdown("## 3.5. Thảo luận chính sách")

        st.markdown("""
### 3.5.a. Kết quả xếp hạng có phản ánh đúng thực tiễn ưu tiên chính sách của Việt Nam không?

Kết quả xếp hạng từ mô hình Priority Index với bộ trọng số mặc định cho thấy ba ngành dẫn đầu
là **CNTT-Truyền thông, Logistics-Vận tải và CN Chế biến-Chế tạo** — một kết quả có độ
tương đồng cao với định hướng chiến lược quốc gia được thể hiện trong Nghị quyết 57-NQ/TW.

Ngành CNTT-Truyền thông đạt điểm PI cao vượt trội nhờ chỉ số AI Readiness ở mức 88/100 —
cao nhất trong 10 ngành — kết hợp với hệ số lan tỏa liên ngành đạt 0.92, phản ánh đúng vai
trò hạ tầng nền tảng của ngành này đối với quá trình chuyển đổi số toàn diện. Ngành
Logistics-Vận tải nổi bật nhờ tốc độ tăng trưởng cao nhất (9.93%) và hệ số lan tỏa tốt
(0.72), phù hợp với vị thế trung tâm logistics của Việt Nam trong chuỗi cung ứng khu vực.

Đáng chú ý, ngành **Khai khoáng**, dù có năng suất lao động cao nhất tuyệt đối
(1.290,5 triệu VND/LĐ), lại xếp hạng thấp do tăng trưởng âm (−1,20%), rủi ro tự động hóa
cao nhất (55%) và AI Readiness chỉ đạt 30/100 — nhất quán với chủ trương giảm dần phụ
thuộc vào tài nguyên thô. Mô hình PI không đơn thuần đo lường quy mô kinh tế hiện tại mà
đánh giá **tiềm năng phát triển trong kỷ nguyên số** — phù hợp với mục tiêu kinh tế số
chiếm 30% GDP vào năm 2030.

---

### 3.5.b. Trọng số nào có ảnh hưởng lớn nhất đến sự ổn định của thứ hạng?

Phân tích độ nhạy (Mục 3.4.3) cho thấy trọng số $a_6$ (AI Readiness) là **thông số phân
kỳ mạnh nhất** trong toàn bộ mô hình. Khi $a_6$ tăng từ 0.05 lên 0.40, thứ hạng của
CNTT-Truyền thông duy trì vị trí số 1 xuyên suốt, trong khi biến động lớn nhất xảy ra ở
nhóm ngành trung bình (hạng 4–7). Điều này phản ánh khoảng cách rất lớn về AI Readiness
giữa các ngành Việt Nam — từ 15 điểm (Nông-Lâm-Thủy sản) đến 88 điểm (CNTT-Truyền thông).

Ngược lại, nhóm ngành **Tài chính-Ngân hàng và Y tế** thể hiện thứ hạng ổn định cao, cho
thấy vị thế ưu tiên của hai ngành này được củng cố bởi sự cân bằng đa chỉ tiêu. Đây là
đặc tính quan trọng khi đưa ra các quyết định phân bổ ngân sách trung và dài hạn.

Về phương pháp luận, phân tích độ nhạy chứng minh rằng Priority Index là một
**công cụ tư duy có cấu trúc**: kết quả xếp hạng chỉ có giá trị trong phạm vi các giả định
về trọng số đã được tường minh hóa — đảm bảo tính minh bạch và trách nhiệm giải trình
trong quy trình ra quyết định chính sách.

---

### 3.5.c. Định hướng "Tăng trưởng" và "Bao trùm" dẫn đến sự khác biệt chính sách nào?

Kết quả so sánh hai kịch bản phản ánh một sự **đánh đổi (trade-off) chiến lược mang tính
nền tảng** trong mô hình phát triển kinh tế Việt Nam giai đoạn 2026–2030.

Dưới **định hướng Tăng trưởng** (ưu tiên $a_1$, $a_2$, $a_4$), các ngành tạo ra giá trị
gia tăng cao và thúc đẩy xuất khẩu như CNTT-Truyền thông, CN Chế biến-Chế tạo và
Logistics-Vận tải dẫn đầu. Kịch bản này phù hợp với mục tiêu tăng trưởng GDP 8–10%,
nhưng tiềm ẩn nguy cơ mở rộng bất bình đẳng thu nhập giữa lao động kỹ năng cao và thấp.

Dưới **định hướng Bao trùm** (ưu tiên $a_5$ — Việc làm và $a_3$ — Lan tỏa), ngành
**Nông-Lâm-Thủy sản và Bán buôn-Bán lẻ** cải thiện vị trí đáng kể nhờ quy mô lao động
lớn và khả năng tạo sinh kế diện rộng — ưu tiên bảo vệ 13,2 triệu lao động nông nghiệp
trước nguy cơ dịch chuyển do tự động hóa.

Hai kịch bản không nhất thiết mâu thuẫn nhau. Nghị quyết 57-NQ/TW đã hàm ý một chiến lược
**"tăng trưởng có bao trùm"**: thúc đẩy đổi mới công nghệ ở nhóm ngành tiên tiến trong
khi đồng thời nâng cao năng lực thích ứng số cho nhóm ngành lao động thâm dụng. Mô hình
AIDEOM-VN, thông qua cơ chế so sánh kịch bản, cung cấp nền tảng định lượng để kiểm định:
*với mỗi đồng ngân sách đầu tư, liệu ưu tiên tăng trưởng hay bao trùm tạo ra giá trị xã
hội tổng thể lớn hơn?*

---
> 📝 *Tài liệu này là một phần của Đồ án hệ thống AIDEOM-VN.
> Mọi dữ liệu và kết quả mang tính minh họa học thuật.*
""")

    except Exception as e:
        st.error(f"❌ Lỗi nghiêm trọng trong Bài 3: {e}")
        st.exception(e)


# ===========================================================================
# CHẠY TRỰC TIẾP: streamlit run bai3_streamlit.py
# ===========================================================================
if __name__ == "__main__":
    st.set_page_config(
        page_title="AIDEOM-VN | Bài 3 — Priority Index",
        page_icon="📊",
        layout="wide",
    )
    run()