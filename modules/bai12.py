"""
bai12.py – Module M6: Dashboard Tích hợp Hệ thống AIDEOM-VN
=============================================================
Đây là cổng giao tiếp duy nhất giữa app.py và AIDEOM_Engine.
app.py chỉ cần gọi: bai12.run(df_macro, df_sectors, df_regions)

Cấu trúc 4 tab:
    Tab 1 – Tổng quan       : KPI tóm tắt toàn hệ thống
    Tab 2 – Phân bổ         : Kết quả tối ưu M3 theo vùng
    Tab 3 – Kịch bản        : So sánh S1–S5 và biểu đồ GDP 2030
    Tab 4 – Cảnh báo rủi ro : Pareto front M5 + VaR stochastic
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# IMPORT ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _import_engine():
    """
    Import AIDEOM_Engine theo thứ tự ưu tiên:
        1. modules/engine.py  (cấu trúc package)
        2. engine.py          (cùng thư mục gốc)

    Parameters
    ----------
    Không có.

    Returns
    -------
    type | None
        Class AIDEOM_Engine nếu tìm thấy, None nếu không.
    """
    for module_path in ("modules.engine", "engine"):
        try:
            import importlib
            mod = importlib.import_module(module_path)
            return getattr(mod, "AIDEOM_Engine")
        except (ImportError, AttributeError):
            continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# HÀM TIỆN ÍCH NỘI BỘ
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(val, decimals: int = 2) -> str:
    """
    Format số thực thành chuỗi có dấu phẩy phân cách hàng nghìn.

    Parameters
    ----------
    val      : Giá trị số cần format (float hoặc int).
    decimals : Số chữ số thập phân (mặc định 2).

    Returns
    -------
    str
        Chuỗi đã format, ví dụ 12847.6 → '12,847.60'.
        Trả về str(val) nếu không convert được.
    """
    try:
        return f"{float(val):,.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)


def _short_region(name: str) -> str:
    """
    Rút gọn tên vùng kinh tế để vừa trục biểu đồ.

    Parameters
    ----------
    name : Tên vùng đầy đủ, ví dụ "Đồng bằng sông Hồng".

    Returns
    -------
    str
        Phần trước ký tự "&" (nếu có), đã strip khoảng trắng.
    """
    return name.split("&")[0].strip()


# ─────────────────────────────────────────────────────────────────────────────
# CÁC HÀM RENDER TỪNG TAB
# ─────────────────────────────────────────────────────────────────────────────

def _render_tab_overview(results) -> None:
    """
    Tab 1 – Tổng quan: KPI tóm tắt toàn hệ thống AIDEOM-VN.

    Parameters
    ----------
    results : AIDEOM_Results
        Đối tượng kết quả từ engine.run_all(). Truy cập các thuộc tính:
        results.summary_kpis (dict), results.m1.growth_decomp (dict).

    Returns
    -------
    None
        Render trực tiếp lên Streamlit.
    """
    st.subheader("📌 Chỉ số KPI Tổng hợp Hệ thống AIDEOM-VN")

    kpis = results.summary_kpis

    # ── Hàng 1: KPI vĩ mô ──────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        label   = "GDP Thực tế 2025 (nT VND)",
        value   = _fmt(kpis["gdp_2025_actual"]),
        delta   = "Điểm neo tham chiếu",
    )
    c2.metric(
        label   = "GDP Dự báo 2030 – Best Case (nT)",
        value   = _fmt(kpis["gdp_2030_best_case"]),
        delta   = f"Kịch bản {kpis['best_scenario_id']}",
    )
    c3.metric(
        label   = "TFP CAGR 2020–2025",
        value   = f"{_fmt(kpis['tfp_cagr_pct'], 3)}%/năm",
        delta   = "Tăng trưởng chiều sâu",
    )
    c4.metric(
        label         = "Sai số MAPE Mô hình M1",
        value         = f"{_fmt(kpis['mape_pct'], 2)}%",
        delta         = "< 5% = chính xác cao",
        delta_color   = "off",
    )

    st.markdown("---")

    # ── Hàng 2: KPI tối ưu & lao động ──────────────────────────────────────
    c5, c6, c7, c8 = st.columns(4)
    c5.metric(
        label   = "Z* Tối ưu LP Liên vùng (M3)",
        value   = _fmt(kpis["optimal_z_star"], 4),
        delta   = "Hàm mục tiêu tăng trưởng",
    )
    c6.metric(
        label   = "Vùng Ưu tiên AI hàng đầu (M2)",
        value   = str(kpis["top_region_name"]),
        delta   = f"C* = {_fmt(kpis['top_region_score'], 4)}",
    )
    c7.metric(
        label         = "Lao động bị dịch chuyển (M4)",
        value         = f"{_fmt(kpis['total_labor_displaced_mil'])} triệu",
        delta         = f"Tạo mới: {_fmt(kpis['total_new_jobs_mil'])} triệu",
        delta_color   = "inverse",
    )
    c8.metric(
        label         = "VaR 95% (Stochastic M5)",
        value         = _fmt(kpis["var_95_stochastic"]),
        delta         = "Ngưỡng tổn thất worst-case",
        delta_color   = "inverse",
    )

    st.markdown("---")

    # ── Bảng KPI đầy đủ ────────────────────────────────────────────────────
    st.subheader("📋 Bảng KPI Chi tiết")

    _KPI_LABELS = {
        "gdp_2025_actual":              "GDP Thực tế 2025 (nghìn tỷ VND)",
        "gdp_2030_best_case":           "GDP Dự báo 2030 Best-Case (nghìn tỷ VND)",
        "tfp_cagr_pct":                 "TFP CAGR 2020–2025 (%/năm)",
        "mape_pct":                     "Sai số MAPE Mô hình M1 (%)",
        "top_region_score":             "Điểm TOPSIS Vùng Dẫn đầu (C*)",
        "top_region_name":              "Vùng Ưu tiên AI Số 1 (M2)",
        "optimal_z_star":               "Hàm Mục tiêu Tối ưu Z* (M3)",
        "total_labor_displaced_mil":    "Tổng Lao động Dịch chuyển (triệu người)",
        "total_new_jobs_mil":           "Tổng Việc làm Mới từ AI (triệu người)",
        "var_95_stochastic":            "Ngưỡng VaR 95% – Stochastic (M5)",
        "best_scenario_id":             "Kịch bản Tối ưu Dự báo",
        "best_scenario_growth_avg_pct": "Tăng trưởng GDP TB Kịch bản Tốt nhất (%/năm)",
    }

    df_kpi = pd.DataFrame(
        [(_KPI_LABELS.get(k, k), str(v)) for k, v in kpis.items()],
        columns=["Chỉ số", "Giá trị"],
    )
    st.table(df_kpi.set_index("Chỉ số"))

    # ── Phân rã tăng trưởng GDP 2020-2025 ──────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Phân rã Đóng góp Tăng trưởng GDP 2020–2025 (M1)")

    decomp    = results.m1.growth_decomp
    df_decomp = pd.DataFrame(
        {"Nhân tố": list(decomp.keys()), "Đóng góp (%)": list(decomp.values())}
    )
    st.bar_chart(df_decomp.set_index("Nhân tố"))


def _render_tab_allocation(results) -> None:
    """
    Tab 2 – Phân bổ: Kết quả tối ưu M3 theo vùng kinh tế.

    Parameters
    ----------
    results : AIDEOM_Results
        Truy cập results.m3 (M3_AllocationResult) với các thuộc tính:
        region_names, dY_optimal, Z_star, budget_used,
        allocation_weights, cost_coefficients.

    Returns
    -------
    None
        Render trực tiếp lên Streamlit.
    """
    st.subheader("🗺️ Kết quả Tối ưu Phân bổ Ngân sách Liên vùng (M3 – LP)")

    m3 = results.m3

    # ── Tóm tắt ────────────────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Z* (Hàm mục tiêu)",     _fmt(m3.Z_star, 4))
    col_b.metric("Ngân sách đã dùng (nT)", _fmt(m3.budget_used))
    col_c.metric("Số vùng tham gia",       str(len(m3.region_names)))

    st.markdown("---")

    # ── Bảng phân bổ chi tiết ───────────────────────────────────────────────
    st.subheader("📋 Bảng Phân bổ Tối ưu Theo Vùng")

    df_alloc = pd.DataFrame(
        {
            "Vùng Kinh tế":        m3.region_names,
            "ΔY Tối ưu (nT VND)":  np.round(m3.dY_optimal, 3),
            "Trọng số w_r":        np.round(m3.allocation_weights, 4),
            "Hệ số chi phí c_r":   np.round(m3.cost_coefficients, 4),
            "Chi phí quy đổi (nT)":np.round(m3.cost_coefficients * m3.dY_optimal, 3),
        }
    )
    st.dataframe(df_alloc.set_index("Vùng Kinh tế"), use_container_width=True)

    # ── Biểu đồ ΔY theo vùng ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 GRDP Tăng thêm Tối ưu theo Vùng (ΔY*)")

    short_names = [_short_region(r) for r in m3.region_names]
    st.bar_chart(
        pd.Series(
            data  = np.round(m3.dY_optimal, 2),
            index = short_names,
            name  = "ΔY Tối ưu (nT)",
        )
    )

    # ── Trọng số và chi phí cạnh nhau ──────────────────────────────────────
    st.markdown("---")
    st.subheader("⚖️ Hệ số Trọng số w_r và Chi phí Biên c_r")

    col_w, col_c2 = st.columns(2)
    with col_w:
        st.caption("Trọng số số hóa tổng hợp (w_r)")
        st.bar_chart(
            pd.Series(np.round(m3.allocation_weights, 4), index=short_names)
        )
    with col_c2:
        st.caption("Hệ số chi phí biên (c_r) – thấp = đầu tư hiệu quả hơn")
        st.bar_chart(
            pd.Series(np.round(m3.cost_coefficients, 4), index=short_names)
        )

    st.info(
        "💡 **Đọc kết quả:** Vùng có w_r cao + c_r thấp là ứng viên ưu tiên "
        "phân bổ ngân sách số hóa. Ràng buộc C3 đảm bảo 2 cực tăng trưởng "
        "(ĐBSH + ĐNB) giữ ≥ 50% tổng ΔY."
    )


def _render_tab_scenarios(results) -> None:
    """
    Tab 3 – So sánh Kịch bản: Bảng tổng hợp, biểu đồ GDP 2030 và quỹ đạo S1–S5.

    Parameters
    ----------
    results : AIDEOM_Results
        Truy cập results.scenarios (Dict[str, ScenarioResult]) với mỗi phần tử
        chứa: label, allocation, gdp_2030, gdp_2035, growth_rate_avg,
        risk_score, inequality_delta, labor_displacement, gdp_trajectory,
        description.

    Returns
    -------
    None
        Render trực tiếp lên Streamlit.
    """
    st.subheader("🔀 So sánh 5 Kịch bản Chiến lược Phân bổ (S1–S5)")

    scenarios = results.scenarios

    # ── Bảng so sánh tổng hợp ──────────────────────────────────────────────
    rows = []
    for sid, s in scenarios.items():
        rows.append({
            "Kịch bản":               sid,
            "Tên chiến lược":          s.label,
            "Phân bổ [K, D, AI, H]":  str(np.round(s.allocation, 2).tolist()),
            "GDP 2030 (nT)":           round(s.gdp_2030, 1),
            "GDP 2035 (nT)":           round(s.gdp_2035, 1),
            "Tăng trưởng TB (%/năm)":  round(s.growth_rate_avg * 100, 2),
            "Rủi ro (0–1)":            round(s.risk_score, 3),
            "Δ Gini":                  round(s.inequality_delta, 3),
            "Dịch chuyển LĐ (%)":      round(s.labor_displacement * 100, 1),
        })
    df_compare = pd.DataFrame(rows)
    st.dataframe(df_compare.set_index("Kịch bản"), use_container_width=True)

    st.markdown("---")

    # ── Biểu đồ GDP 2030 ───────────────────────────────────────────────────
    st.subheader("📊 So sánh GDP Dự báo 2030 – 5 Kịch bản")
    st.bar_chart(
        pd.Series(
            {sid: round(s.gdp_2030, 1) for sid, s in scenarios.items()},
            name="GDP 2030 (nT VND)",
        )
    )

    # ── Quỹ đạo GDP 2026–2035 ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Quỹ đạo GDP 2026–2035 theo Kịch bản")
    traj_df = pd.DataFrame(
        {f"{sid} – {s.label[:20]}": s.gdp_trajectory for sid, s in scenarios.items()},
        index=list(range(2026, 2036)),
    )
    st.line_chart(traj_df)

    # ── Chi tiết từng kịch bản (expander) ──────────────────────────────────
    st.markdown("---")
    st.subheader("🔍 Mô tả Chi tiết Từng Kịch bản")

    _ALLOC_LABELS = ["Vốn vật chất (K)", "Hạ tầng số (D)", "AI", "Nhân lực (H)"]

    for sid, s in scenarios.items():
        with st.expander(f"{sid}: {s.label}", expanded=(sid == "S5")):
            st.markdown(s.description)

            col_1, col_2, col_3 = st.columns(3)
            col_1.metric("GDP 2030",           f"{_fmt(s.gdp_2030)} nT")
            col_2.metric("Rủi ro tổng hợp",    f"{s.risk_score:.1%}")
            col_3.metric("Dịch chuyển LĐ",     f"{s.labor_displacement:.1%}")

            st.bar_chart(
                pd.Series(
                    dict(zip(_ALLOC_LABELS, np.round(s.allocation, 2))),
                    name="Tỷ trọng phân bổ",
                )
            )

    # ── Bảng nổi bật S1 / S3 / S5 ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("📌 Bảng Tóm tắt Kịch bản Nổi bật: S1 / S3 / S5")

    _HIGHLIGHT_COLS = ["Tên chiến lược", "GDP 2030 (nT)", "Tăng trưởng TB (%/năm)", "Rủi ro (0–1)"]
    df_highlight = (
        df_compare[df_compare["Kịch bản"].isin(["S1", "S3", "S5"])]
        .set_index("Kịch bản")[_HIGHLIGHT_COLS]
    )
    st.table(df_highlight)


def _render_tab_risk(results) -> None:
    """
    Tab 4 – Cảnh báo Rủi ro: Pareto front M5 và phân tích stochastic.

    Parameters
    ----------
    results : AIDEOM_Results
        Truy cập results.m5 (M5_RiskResult) với các thuộc tính:
        stochastic_z_mean, stochastic_z_std, var_95, scenario_probs,
        pareto_gdp, pareto_inequality, pareto_emissions.

    Returns
    -------
    None
        Render trực tiếp lên Streamlit.
    """
    st.subheader("⚠️ Bảng Điều khiển Cảnh báo Rủi ro (M5: NSGA-II + Stochastic)")

    m5 = results.m5

    # ── KPI rủi ro ─────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric(
        label   = "E[Z] Stochastic (nT)",
        value   = _fmt(m5.stochastic_z_mean),
        delta   = f"±{_fmt(m5.stochastic_z_std, 2)} std",
    )
    c2.metric(
        label         = "VaR 95% (Worst-case)",
        value         = _fmt(m5.var_95),
        delta         = "Ngưỡng tổn thất 5% xấu nhất",
        delta_color   = "inverse",
    )
    c3.metric(
        label   = "Xác suất Kịch bản Cơ sở",
        value   = f"{m5.scenario_probs[1]:.0%}",
        delta   = "Base case: 50%",
    )

    st.markdown("---")

    # ── Pareto: GDP vs Bất bình đẳng ───────────────────────────────────────
    st.subheader("🎯 Pareto Front: Đánh đổi GDP vs Bất bình đẳng (NSGA-II)")
    st.line_chart(
        pd.DataFrame(
            {"Gini (Bất bình đẳng)": np.round(m5.pareto_inequality, 4)},
            index=np.round(m5.pareto_gdp, 1),
        )
    )
    st.caption(
        "💡 Mỗi điểm trên đường Pareto là một phương án tối ưu: không thể tăng GDP "
        "mà không làm tăng bất bình đẳng. Nhà hoạch định chính sách chọn điểm phù hợp "
        "với ưu tiên xã hội."
    )

    # ── Pareto: GDP vs Phát thải ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🌿 Pareto Front: GDP vs Phát thải Carbon")
    st.line_chart(
        pd.DataFrame(
            {"Chỉ số Phát thải": np.round(m5.pareto_emissions, 4)},
            index=np.round(m5.pareto_gdp, 1),
        )
    )

    # ── Bảng stochastic 2 giai đoạn ────────────────────────────────────────
    st.markdown("---")
    st.subheader("🎲 Phân tích Xác suất – Bài toán Stochastic 2 Giai đoạn (bai10)")

    _Z_BASE = np.array([100.0, 115.0, 128.0])
    _LABELS = [
        "Kịch bản Thấp (Suy thoái)",
        "Kịch bản Cơ sở (Trung bình)",
        "Kịch bản Cao (Bùng nổ)",
    ]
    df_stoch = pd.DataFrame({
        "Kịch bản":          _LABELS,
        "Xác suất":          m5.scenario_probs,
        "Z Kỳ vọng (nT)":    _Z_BASE,
        "Đóng góp vào E[Z]": m5.scenario_probs * _Z_BASE,
    })
    st.table(df_stoch.set_index("Kịch bản"))

    # ── Cảnh báo ngưỡng ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🚨 Bảng Cảnh báo Ngưỡng Rủi ro Hệ thống")

    _RISK_FLAGS = {
        "VaR 95% < 100 nT":                    m5.var_95 < 100,
        "Bất bình đẳng Pareto max > 0.43":     float(m5.pareto_inequality.max()) > 0.43,
        "Phát thải Pareto max > 1.4":          float(m5.pareto_emissions.max()) > 1.4,
        "std[Z] > 8 nT (biến động cao)":       m5.stochastic_z_std > 8,
    }

    for flag_name, triggered in _RISK_FLAGS.items():
        if triggered:
            st.error(f"🔴 CẢNH BÁO: {flag_name}")
        else:
            st.success(f"🟢 AN TOÀN: {flag_name}")

    st.info(
        "📖 **Hướng dẫn đọc:** Khi VaR 95% thấp hơn ngưỡng an toàn, "
        "cần xem xét bổ sung dự phòng tài khóa hoặc điều chỉnh kịch bản "
        "phân bổ sang chiến lược thận trọng hơn (S1 hoặc S4)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT – ĐƯỢC GỌI TỪ app.py
# ─────────────────────────────────────────────────────────────────────────────

_SESSION_KEY = "aideom_results"


def run(
    df_macro: Optional[pd.DataFrame]   = None,
    df_sectors: Optional[pd.DataFrame] = None,
    df_regions: Optional[pd.DataFrame] = None,
) -> None:
    """
    Hàm cổng giao tiếp duy nhất với app.py (Module M6 – Dashboard AIDEOM-VN).

    Thực hiện 3 bước:
        1. Import AIDEOM_Engine từ engine.py (hoặc modules/engine.py).
        2. Chạy pipeline M1–M5 một lần và cache vào st.session_state.
        3. Render Dashboard 4 tab trên Streamlit.

    Parameters
    ----------
    df_macro   : pd.DataFrame | None
        Dữ liệu vĩ mô Vietnam 2020–2025. None → engine dùng dữ liệu nội bộ.
    df_sectors : pd.DataFrame | None
        Dữ liệu 8 ngành kinh tế 2024. None → engine dùng dữ liệu nội bộ.
    df_regions : pd.DataFrame | None
        Dữ liệu 6 vùng kinh tế 2024. None → engine dùng dữ liệu nội bộ.

    Returns
    -------
    None
        Render trực tiếp lên Streamlit; không trả về giá trị.
    """
    st.header("🏛️ BÀI 12: ĐỒ ÁN TÍCH HỢP HỆ THỐNG AIDEOM-VN")
    st.caption(
        "AI-Driven Economic Optimization Model for Vietnam | "
        "M1 (Cobb-Douglas) · M2 (TOPSIS) · M3 (LP) · M4 (Lao động) · M5 (Rủi ro)"
    )
    st.markdown("---")

    # ── Bước 1: Import Engine ───────────────────────────────────────────────
    AIDEOM_Engine = _import_engine()
    if AIDEOM_Engine is None:
        st.error(
            "❌ Không tìm thấy `engine.py`. "
            "Đặt file trong cùng thư mục với `bai12.py` hoặc trong `modules/`."
        )
        return

    # ── Bước 2: Chạy pipeline (cache theo session) ─────────────────────────
    if _SESSION_KEY not in st.session_state:
        with st.spinner("⚙️ Đang chạy pipeline AIDEOM-VN (M1 → M5)…"):
            try:
                engine = AIDEOM_Engine(df_macro, df_sectors, df_regions)
                st.session_state[_SESSION_KEY] = engine.run_all()
            except Exception as exc:
                st.error(f"❌ Lỗi khi chạy AIDEOM Engine: {exc}")
                st.exception(exc)
                return

    results = st.session_state[_SESSION_KEY]
    st.success("✅ Pipeline AIDEOM-VN hoàn thành.")

    if st.button("🔄 Chạy lại / Làm mới kết quả"):
        del st.session_state[_SESSION_KEY]
        st.rerun()

    st.markdown("---")

    # ── Bước 3: Render 4 Tab ────────────────────────────────────────────────
    tab_overview, tab_alloc, tab_scenario, tab_risk = st.tabs([
        "📊 Tổng quan",
        "🗺️ Phân bổ Liên vùng",
        "🔀 So sánh Kịch bản",
        "⚠️ Cảnh báo Rủi ro",
    ])

    with tab_overview:
        _render_tab_overview(results)

    with tab_alloc:
        _render_tab_allocation(results)

    with tab_scenario:
        _render_tab_scenarios(results)

    with tab_risk:
        _render_tab_risk(results)