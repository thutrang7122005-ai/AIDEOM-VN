# =============================================================================
# modules/bai9.py
# Bài 9: Tác động AI tới thị trường lao động Việt Nam
# Entry point: def run(df_macro=None, df_sectors=None, df_regions=None)
# =============================================================================

import numpy as np
import streamlit as st

try:
    import cvxpy as cp
    CVXPY_OK = True
except ImportError:
    CVXPY_OK = False

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

import pandas as pd

# ── Màu sắc ──────────────────────────────────────────────────────────────────
C = {
    "primary":   "#00D4FF",
    "secondary": "#A855F7",
    "success":   "#00FF88",
    "danger":    "#FF4757",
    "warning":   "#FFD700",
    "grid":      "#2a3040",
    "card":      "#1a1f2e",
    "bg":        "#0f1117",
}

# ── Dữ liệu 8 ngành (Bảng 9.3) ───────────────────────────────────────────────
SECTORS = [
    "Nông-Lâm-Thủy sản",
    "CN chế biến chế tạo",
    "Xây dựng",
    "Bán buôn-bán lẻ",
    "Tài chính-Ngân hàng",
    "Logistics-Vận tải",
    "CNTT-Truyền thông",
    "Giáo dục-Đào tạo",
]
SHORT = ["NL-TS", "CN CB", "Xây dựng", "Bán lẻ", "Tài chính", "Logistics", "CNTT", "Giáo dục"]

N      = 8
BUDGET = 30_000  # tỷ VND

# Tham số — tất cả là numpy array thuần float64
L_i  = np.array([13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15], dtype=float)
risk = np.array([18,    42,    25,   38,   52,   35,   28,   22],    dtype=float) / 100.0
a1   = np.array([8.5,   32.5,  12.8, 22.4, 45.8, 28.5, 62.5, 18.5], dtype=float)
a2   = np.array([12.0,  18.5,   8.5, 15.2, 12.5, 16.8, 15.0, 22.0], dtype=float)
b1   = np.array([45.0,  28.0,  35.0, 32.0, 22.0, 30.0, 20.0, 55.0], dtype=float)
c1   = np.array([5.2,   62.4,  18.5, 48.2, 72.5, 42.8, 32.5, 12.5], dtype=float)
d1   = np.array([50.0,  32.0,  42.0, 38.0, 26.0, 36.0, 24.0, 62.0], dtype=float)

# Hệ số dẫn xuất (numpy thuần — KHÔNG phải CVXPY)
alpha           = a1 - c1 * risk          # hệ số ròng của xAI
displaced_coeff = c1 * risk               # = Displaced / xAI


# =============================================================================
# HELPER: ép về numpy 1-D float64 (tránh mọi vết CVXPY)
# =============================================================================
def _arr(x) -> np.ndarray:
    """Chuyển bất kỳ kiểu nào → numpy 1-D float64."""
    if x is None:
        return np.zeros(N)
    if CVXPY_OK and isinstance(x, (cp.Variable, cp.Expression)):
        v = x.value
        return np.zeros(N) if v is None else np.asarray(v, dtype=float).ravel()
    return np.asarray(x, dtype=float).ravel()


def _lst(x) -> list:
    """_arr → Python list (dùng cho pd.DataFrame column)."""
    return _arr(x).tolist()


# =============================================================================
# CSS
# =============================================================================
def _css():
    st.markdown("""
    <style>
    .stApp{background:#0f1117;color:#e0e0e0}
    div[data-testid="metric-container"]{
        background:#1a1f2e;border:1px solid #2a3040;
        border-radius:12px;padding:16px 20px}
    div[data-testid="metric-container"] label{color:#aaaacc!important;font-size:.78rem}
    div[data-testid="metric-container"] [data-testid="metric-value"]{
        color:#00D4FF!important;font-size:1.6rem;font-weight:700}
    .sh{background:linear-gradient(135deg,#1a1f2e,#0f1117);border-left:4px solid #00D4FF;
        padding:12px 20px;border-radius:0 10px 10px 0;margin:24px 0 16px}
    .sh h3{color:#00D4FF;margin:0;font-size:1.05rem}
    .sh p{color:#aaaacc;margin:4px 0 0;font-size:.82rem}
    .ib{background:rgba(0,212,255,.08);border:1px solid rgba(0,212,255,.3);
        border-radius:10px;padding:14px 18px;margin:10px 0}
    .wb{background:rgba(255,215,0,.08);border:1px solid rgba(255,215,0,.35);
        border-radius:10px;padding:14px 18px;margin:10px 0}
    .sb{background:rgba(0,255,136,.08);border:1px solid rgba(0,255,136,.35);
        border-radius:10px;padding:14px 18px;margin:10px 0}
    .db{background:rgba(255,71,87,.08);border:1px solid rgba(255,71,87,.35);
        border-radius:10px;padding:14px 18px;margin:10px 0}
    button[data-baseweb="tab"]{color:#aaaacc}
    button[data-baseweb="tab"][aria-selected="true"]{color:#00D4FF;border-bottom-color:#00D4FF}
    thead tr th{background:#0f1117!important;color:#00D4FF!important}
    </style>""", unsafe_allow_html=True)


def _sh(icon, title, sub=""):
    st.markdown(
        f'<div class="sh"><h3>{icon} {title}</h3>'
        + (f'<p>{sub}</p>' if sub else '') + '</div>',
        unsafe_allow_html=True)


# =============================================================================
# BÀI TOÁN TỐI ƯU — câu 9.4.1
# =============================================================================
def _solve(min_ai: float = 200.0, min_h: float = 300.0,
           cap5pct: bool = False) -> dict:
    """
    Giải LP bằng CVXPY (hoặc scipy fallback).
    Trả về dict với tất cả giá trị là numpy array thuần float64.
    KHÔNG bao giờ trả về cvxpy.Expression hay cvxpy.Variable.
    """
    if CVXPY_OK:
        return _solve_cvxpy(min_ai, min_h, cap5pct)
    return _solve_scipy(min_ai, min_h, cap5pct)


def _solve_cvxpy(min_ai, min_h, cap5pct):
    x_AI = cp.Variable(N, nonneg=True)
    x_H  = cp.Variable(N, nonneg=True)

    # Biểu thức CVXPY — CHỈ dùng trong prob.solve(), không bao giờ vào dict
    NJ_expr   = cp.multiply(a1, x_AI) + cp.multiply(b1, x_H) \
                - cp.multiply(displaced_coeff, x_AI)
    Disp_expr = cp.multiply(displaced_coeff, x_AI)
    Ret_expr  = cp.multiply(d1, x_H)

    cons = [
        cp.sum(x_AI + x_H) <= BUDGET,
        NJ_expr >= 0,
        Disp_expr <= Ret_expr,
    ]
    if min_ai > 0:
        cons.append(x_AI >= min_ai)
    if min_h > 0:
        cons.append(x_H >= min_h)
    if cap5pct:
        cons.append(Disp_expr <= 0.05 * L_i * 1000)

    prob = cp.Problem(cp.Maximize(cp.sum(NJ_expr)), cons)
    prob.solve(solver=cp.HIGHS, verbose=False)

    # ── Kiểm tra status TRƯỚC KHI lấy .value ─────────────────────────────────
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return {"ok": False, "status": prob.status}

    # ── Chuyển sang numpy thuần ngay tại đây ──────────────────────────────────
    xai = _arr(x_AI.value)   # x_AI.value là ndarray; _arr() để chắc chắn
    xh  = _arr(x_H.value)
    # Tính lại bằng numpy — KHÔNG dùng NJ_expr.value (tránh CVXPY artifacts)
    nj_val    = a1 * xai + b1 * xh - displaced_coeff * xai
    disp_val  = displaced_coeff * xai
    ret_val   = d1 * xh

    return {
        "ok":      True,
        "status":  prob.status,
        "xAI":     xai,
        "xH":      xh,
        "NewJob":  a1 * xai,
        "Upgrade": b1 * xh,
        "Disp":    disp_val,
        "Ret":     ret_val,
        "NetJob":  nj_val,
        "total":   float(np.sum(nj_val)),
    }


def _solve_scipy(min_ai, min_h, cap5pct):
    from scipy.optimize import linprog
    c_obj = np.concatenate([-alpha, -b1])
    A, b  = [], []

    # Ngân sách
    A.append(np.ones(2 * N)); b.append(BUDGET)
    # NetJob >= 0  →  -alpha·xai - b1·xh <= 0
    for i in range(N):
        r = np.zeros(2 * N); r[i] = -alpha[i]; r[N+i] = -b1[i]
        A.append(r); b.append(0.0)
    # Displaced <= RetrainCap
    for i in range(N):
        r = np.zeros(2 * N); r[i] = displaced_coeff[i]; r[N+i] = -d1[i]
        A.append(r); b.append(0.0)
    if cap5pct:
        thr = 0.05 * L_i * 1000
        for i in range(N):
            r = np.zeros(2 * N); r[i] = displaced_coeff[i]
            A.append(r); b.append(float(thr[i]))

    bounds = [(min_ai, None)] * N + [(min_h, None)] * N
    res = linprog(c_obj, A_ub=np.array(A), b_ub=np.array(b),
                  bounds=bounds, method="highs")

    if not res.success:
        return {"ok": False, "status": res.message}

    xai = _arr(res.x[:N]);  xh = _arr(res.x[N:])
    nj  = a1 * xai + b1 * xh - displaced_coeff * xai

    return {
        "ok":      True,
        "status":  "optimal",
        "xAI":     xai,
        "xH":      xh,
        "NewJob":  a1 * xai,
        "Upgrade": b1 * xh,
        "Disp":    displaced_coeff * xai,
        "Ret":     d1 * xh,
        "NetJob":  nj,
        "total":   float(np.sum(nj)),
    }


# =============================================================================
# CHARTS
# =============================================================================
def _chart_bubble():
    """Ma trận Rủi ro × Alpha, kích thước = quy mô lao động."""
    if not PLOTLY_OK:
        return None
    fig = go.Figure(go.Scatter(
        x=risk * 100, y=alpha,
        mode="markers+text",
        text=SHORT,
        textposition="top center",
        marker=dict(
            size=[max(L * 4, 10) for L in L_i],
            color=["#FF6B6B","#4ECDC4","#45B7D1","#96CEB4",
                   "#FFEAA7","#DDA0DD","#98D8C8","#F7DC6F"],
            line=dict(color="white", width=0.8), opacity=0.9,
        ),
        hovertemplate="<b>%{text}</b><br>Risk: %{x:.0f}%<br>α: %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=C["danger"], width=1.2, dash="dash"))
    fig.add_vline(x=35, line=dict(color=C["warning"], width=1, dash="dot"))
    fig.update_layout(
        title=dict(text="Ma trận Rủi ro × Hiệu quả AI  (kích thước ∝ lao động)",
                   font=dict(color=C["primary"], size=13)),
        xaxis=dict(title="Rủi ro tự động hóa (%)", gridcolor=C["grid"],
                   tickfont=dict(color="#e0e0e0")),
        yaxis=dict(title="α = a1 − c1·risk", gridcolor=C["grid"],
                   tickfont=dict(color="#e0e0e0")),
        paper_bgcolor=C["bg"], plot_bgcolor=C["card"],
        height=400, margin=dict(l=10,r=10,t=50,b=10),
    )
    return fig


def _chart_alloc(res):
    if not PLOTLY_OK:
        return None
    fig = go.Figure()
    fig.add_trace(go.Bar(name="xAI", x=SHORT, y=res["xAI"]/1000,
                         marker_color=C["primary"], opacity=.85,
                         hovertemplate="%{x}<br>xAI=%{y:.2f}k tỷ<extra></extra>"))
    fig.add_trace(go.Bar(name="xH",  x=SHORT, y=res["xH"]/1000,
                         marker_color=C["secondary"], opacity=.85,
                         hovertemplate="%{x}<br>xH=%{y:.2f}k tỷ<extra></extra>"))
    fig.update_layout(
        barmode="group",
        title=dict(text="Phân bổ ngân sách tối ưu (nghìn tỷ VND)",
                   font=dict(color=C["primary"], size=13)),
        xaxis=dict(gridcolor=C["grid"], tickfont=dict(color="#e0e0e0")),
        yaxis=dict(title="Nghìn tỷ VND", gridcolor=C["grid"],
                   tickfont=dict(color="#e0e0e0")),
        legend=dict(font=dict(color="#e0e0e0"), bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor=C["bg"], plot_bgcolor=C["card"],
        height=370, margin=dict(l=10,r=10,t=50,b=10),
    )
    return fig


def _chart_netjob(res):
    if not PLOTLY_OK:
        return None
    nj = res["NetJob"]
    cols = [C["success"] if v >= 0 else C["danger"] for v in nj]
    fig = go.Figure(go.Bar(
        x=SHORT, y=nj, marker_color=cols,
        hovertemplate="%{x}<br>NetJob=%{y:,.0f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=C["danger"], width=1.5, dash="dash"))
    fig.update_layout(
        title=dict(text="NetJob ròng theo ngành (nghìn việc làm)",
                   font=dict(color=C["primary"], size=13)),
        xaxis=dict(gridcolor=C["grid"], tickfont=dict(color="#e0e0e0")),
        yaxis=dict(gridcolor=C["grid"], tickfont=dict(color="#e0e0e0")),
        paper_bgcolor=C["bg"], plot_bgcolor=C["card"],
        height=360, margin=dict(l=10,r=10,t=50,b=10),
    )
    return fig


def _chart_threshold():
    """Biểu đồ ngưỡng xH₂ tối thiểu — câu 9.4.2."""
    if not PLOTLY_OK:
        return None
    ratio  = displaced_coeff[1] / d1[1]   # float thuần
    xai_r  = np.linspace(0, 20_000, 300)
    xh_min = ratio * xai_r
    xh_bud = BUDGET - xai_r
    xai_cross = BUDGET / (1 + ratio)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xai_r/1000, y=xh_min/1000, mode="lines",
        name=f"xH₂_min = {ratio:.3f}·xAI₂",
        line=dict(color=C["danger"], width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=xai_r/1000, y=xh_bud/1000, mode="lines",
        name="Giới hạn ngân sách",
        line=dict(color=C["warning"], width=2, dash="dash"),
    ))
    fig.add_vline(x=xai_cross/1000,
                  line=dict(color=C["primary"], width=1.5, dash="dot"),
                  annotation_text=f"xAI₂_max={xai_cross/1000:.1f}k tỷ",
                  annotation_font_color=C["primary"])
    fig.update_layout(
        title=dict(text="Ngưỡng xH₂_min — CN Chế biến chế tạo",
                   font=dict(color=C["primary"], size=13)),
        xaxis=dict(title="xAI₂ (nghìn tỷ)", gridcolor=C["grid"],
                   tickfont=dict(color="#e0e0e0")),
        yaxis=dict(title="xH₂ (nghìn tỷ)", gridcolor=C["grid"],
                   tickfont=dict(color="#e0e0e0")),
        legend=dict(font=dict(color="#e0e0e0"), bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor=C["bg"], plot_bgcolor=C["card"],
        height=390, margin=dict(l=10,r=10,t=50,b=10),
    )
    return fig


def _chart_sankey(res):
    """
    Sankey — câu 9.4.3.
    Luồng: Ngành (1,3,4) → [Bị displaced | Upgrade] → [Retrain | Thất nghiệp] → Kết quả
    Tất cả giá trị đã là float Python thuần.
    """
    if not PLOTLY_OK:
        return None

    # Indices 0-based của ngành 1, 3, 4
    vi = [0, 2, 3]
    sec_labels = [SECTORS[i] for i in vi]

    # Node indices
    # 0,1,2  : 3 ngành nguồn
    # 3       : Bị tự động hóa (Displaced)
    # 4       : Giữ việc & Nâng cấp (Upgrade)
    # 5       : Đào tạo lại thành công
    # 6       : Thất nghiệp tạm thời
    # 7       : Việc làm mới
    # 8       : Kỹ năng nâng cao

    labels = sec_labels + [
        "Bị tự động hóa",
        "Nâng cấp kỹ năng",
        "Đào tạo lại",
        "Thất nghiệp tạm thời",
        "Việc làm mới",
        "Kỹ năng cao hơn",
    ]
    node_colors = (
        ["#3B82F6", "#45B7D1", "#96CEB4",
         "#EF4444", "#10B981",
         "#F59E0B", "#9CA3AF",
         "#8B5CF6", "#06B6D4"]
    )

    src, tgt, val, lcol = [], [], [], []

    for ei, i in enumerate(vi):
        # Ép float() tường minh — res đã là numpy nhưng float() bảo đảm Plotly
        disp  = max(float(res["Disp"][i]),    0.01)
        upg   = max(float(res["Upgrade"][i]), 0.01)
        ok    = disp * 0.85
        fail  = disp * 0.15

        # ngành → displaced
        src.append(ei);  tgt.append(3);  val.append(disp)
        lcol.append("rgba(239,68,68,0.3)")
        # ngành → upgrade
        src.append(ei);  tgt.append(4);  val.append(upg)
        lcol.append("rgba(16,185,129,0.3)")
        # displaced → retrain
        src.append(3);   tgt.append(5);  val.append(max(ok,   0.01))
        lcol.append("rgba(245,158,11,0.3)")
        # displaced → thất nghiệp
        src.append(3);   tgt.append(6);  val.append(max(fail, 0.01))
        lcol.append("rgba(156,163,175,0.3)")
        # retrain → việc làm mới
        src.append(5);   tgt.append(7);  val.append(max(ok * 0.8, 0.01))
        lcol.append("rgba(139,92,246,0.3)")
        # upgrade → kỹ năng cao
        src.append(4);   tgt.append(8);  val.append(max(upg * 0.9, 0.01))
        lcol.append("rgba(6,182,212,0.3)")

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=18, thickness=20,
            line=dict(color="#2a3040", width=0.6),
            label=labels,
            color=node_colors,
            hovertemplate="<b>%{label}</b><br>%{value:,.0f} nghìn người<extra></extra>",
        ),
        link=dict(
            source=src, target=tgt, value=val, color=lcol,
            hovertemplate="%{source.label} → %{target.label}<br>"
                          "%{value:,.0f} nghìn người<extra></extra>",
        ),
    ))
    fig.update_layout(
        title=dict(
            text="Luồng dịch chuyển lao động — Ngành 1, 3, 4 dễ bị tổn thương",
            font=dict(color=C["primary"], size=14)),
        font=dict(color="#e0e0e0", size=11),
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
        height=500, margin=dict(l=20,r=20,t=70,b=20),
    )
    return fig


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def run(df_macro=None, df_sectors=None, df_regions=None):
    """Hàm chính — gọi bởi app.py dispatch."""
    _css()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
        background:linear-gradient(135deg,#0f1117 0%,#1a1f2e 60%,#0f1117 100%);
        border:1px solid #2a3040;border-radius:16px;
        padding:28px 32px;margin-bottom:24px;position:relative;overflow:hidden;">
      <div style="position:absolute;top:-40px;right:-40px;width:180px;height:180px;
        border-radius:50%;background:radial-gradient(circle,rgba(0,212,255,.12),transparent 70%);"></div>
      <h1 style="color:#00D4FF;margin:0;font-size:1.7rem;font-weight:800;">
        🤖 Bài 9 — Tác động AI tới thị trường lao động Việt Nam
      </h1>
      <p style="color:#aaaacc;margin:8px 0 0;font-size:.9rem;line-height:1.5;">
        Tối ưu hóa phân bổ <b style="color:#FFD700;">30.000 tỷ VND</b>
        cho <b style="color:#A855F7;">8 ngành kinh tế</b> · Mô hình LP · ILO Vietnam 2024
      </p>
    </div>""", unsafe_allow_html=True)

    sv_lbl = "✅ CVXPY+HiGHS" if CVXPY_OK else "⚙️ SciPy HiGHS"
    ch_lbl = "✅ Plotly"       if PLOTLY_OK else "⚠️ Chưa cài Plotly"
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Ngân sách", "30.000 tỷ VND")
    m2.metric("Số ngành",  "8 ngành")
    m3.metric("Solver",    sv_lbl)
    m4.metric("Charts",    ch_lbl)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    t1,t2,t3,t4,t5 = st.tabs([
        "📊 9.4.1 Tối ưu LP",
        "📉 9.4.2 Ngưỡng CN CB",
        "🌊 9.4.3 Sankey",
        "🔒 9.4.4 Giới hạn 5%",
        "💬 9.5 Chính sách",
    ])

    # =========================================================================
    # TAB 1 — Câu 9.4.1: Tối ưu hóa LP
    # =========================================================================
    with t1:
        _sh("⚙️", "Câu 9.4.1 — Mô hình tối ưu hóa LP",
            "max Σ NetJobᵢ   s.t.  ngân sách ≤ 30.000 tỷ,  NetJobᵢ ≥ 0,  Displacedᵢ ≤ RetrainCapᵢ")

        st.markdown("""
        <div class="ib">
        <b>Mô hình toán học:</b><br>
        <code>NetJobᵢ = a1ᵢ·xAIᵢ + b1ᵢ·xHᵢ − c1ᵢ·riskᵢ·xAIᵢ = αᵢ·xAIᵢ + b1ᵢ·xHᵢ</code>
        &nbsp; với αᵢ = a1ᵢ − c1ᵢ·riskᵢ<br><br>
        <b>Ràng buộc:</b> ① Tổng ngân sách ≤ 30.000 tỷ &nbsp;|&nbsp;
        ② NetJobᵢ ≥ 0 ∀i &nbsp;|&nbsp; ③ Displacedᵢ ≤ RetrainCapᵢ
        </div>
        """, unsafe_allow_html=True)

        # Bảng tham số — dùng .tolist() tường minh để tránh mọi vấn đề dtype
        with st.expander("📋 Xem bảng tham số 8 ngành", expanded=False):
            df_p = pd.DataFrame({
                "Ngành":         SECTORS,
                "LĐ (triệu)":   L_i.tolist(),
                "Risk (%)":     (risk*100).astype(int).tolist(),
                "a1":           a1.tolist(),
                "b1":           b1.tolist(),
                "c1":           c1.tolist(),
                "d1":           d1.tolist(),
                "α=a1−c1·risk": np.round(alpha,3).tolist(),
            })
            st.dataframe(df_p, use_container_width=True, hide_index=True)

        if PLOTLY_OK:
            st.plotly_chart(_chart_bubble(), use_container_width=True)

        st.divider()
        st.subheader("🔢 Chạy bài toán tối ưu")

        ca, cb = st.columns(2)
        min_ai = ca.number_input("xAI tối thiểu/ngành (tỷ)", 0, 1000, 200, 50)
        min_h  = cb.number_input("xH  tối thiểu/ngành (tỷ)", 0, 1000, 300, 50)

        if st.button("▶ Giải tối ưu LP", type="primary", use_container_width=True):
            with st.spinner("Đang giải…"):
                st.session_state["r1"] = _solve(float(min_ai), float(min_h))

        # Tự chạy lần đầu
        if "r1" not in st.session_state:
            st.session_state["r1"] = _solve(float(min_ai), float(min_h))

        res = st.session_state["r1"]

        # ── Kiểm tra prob.status TRƯỚC KHI hiển thị DataFrame / biểu đồ ──────
        if not res.get("ok"):
            st.markdown(f"""
            <div class="db">❌ <b>Mô hình vô nghiệm</b>
            (status: <code>{res.get('status','unknown')}</code>)<br>
            Thử giảm ngưỡng tối thiểu xAI / xH.
            </div>""", unsafe_allow_html=True)
        else:
            # Metrics
            c1_,c2_,c3_,c4_ = st.columns(4)
            c1_.metric("Tổng NetJob",    f"{res['total']:,.0f}",  "nghìn việc làm ròng")
            c2_.metric("Tổng xAI",       f"{res['xAI'].sum():,.0f} tỷ",
                       f"{100*res['xAI'].sum()/BUDGET:.1f}% ngân sách")
            c3_.metric("Tổng xH",        f"{res['xH'].sum():,.0f} tỷ",
                       f"{100*res['xH'].sum()/BUDGET:.1f}% ngân sách")
            c4_.metric("Tổng Displaced", f"{res['Disp'].sum():,.0f}",
                       "nghìn LĐ dịch chuyển", delta_color="inverse")

            st.markdown("<br>", unsafe_allow_html=True)

            # Bảng kết quả — TẤT CẢ cột đều là Python list thuần
            # (res["xAI"] đã là numpy array; .tolist() ra list[float])
            df_r = pd.DataFrame({
                "Ngành":      SECTORS,
                "xAI (tỷ)":  np.round(res["xAI"],    1).tolist(),
                "xH (tỷ)":   np.round(res["xH"],     1).tolist(),
                "NewJob":     np.round(res["NewJob"],  1).tolist(),
                "Upgrade":    np.round(res["Upgrade"], 1).tolist(),
                "Displaced":  np.round(res["Disp"],    1).tolist(),
                "RetrainCap": np.round(res["Ret"],     1).tolist(),
                "NetJob":     np.round(res["NetJob"],  1).tolist(),
            })
            st.dataframe(df_r, use_container_width=True, hide_index=True)

            if PLOTLY_OK:
                cc1, cc2 = st.columns(2)
                with cc1:
                    st.plotly_chart(_chart_alloc(res),  use_container_width=True)
                with cc2:
                    st.plotly_chart(_chart_netjob(res), use_container_width=True)

            st.markdown("""
            <div class="wb">
            <b>📌 Corner solution LP:</b> Bài toán LP luôn cho nghiệm tại <i>đỉnh</i> tập khả thi.
            Không đặt ngưỡng tối thiểu → solver dồn toàn bộ ngân sách vào ngành có β hoặc α cao nhất.
            Đặt min_ai / min_h để phân bổ thực tế hơn.
            </div>""", unsafe_allow_html=True)

    # =========================================================================
    # TAB 2 — Câu 9.4.2: Ngưỡng xH tối thiểu ngành 2
    # =========================================================================
    with t2:
        _sh("📉", "Câu 9.4.2 — Ngưỡng xH₂ tối thiểu (CN Chế biến chế tạo)",
            "Risk=42%, c1=62.4 (cao nhất), d1=32 — ràng buộc nào binding?")

        i2    = 1   # index ngành 2
        ratio = float(displaced_coeff[i2] / d1[i2])   # float thuần

        st.markdown(f"""
        <div class="ib">
        <b>Phân tích ràng buộc binding — ngành 2:</b><br><br>
        <b>① NetJob₂ ≥ 0:</b><br>
        &nbsp;&nbsp;α₂·xAI₂ + b1₂·xH₂ ≥ 0
        &nbsp;⟹&nbsp; xH₂ ≥ {-alpha[i2]/b1[i2]:.4f}·xAI₂<br>
        &nbsp;&nbsp;→ <b style="color:#00FF88">KHÔNG BINDING</b> vì α₂ = {alpha[i2]:.3f} > 0<br><br>
        <b>② Displaced ≤ RetrainCap (BINDING):</b><br>
        &nbsp;&nbsp;{displaced_coeff[i2]:.4f}·xAI₂ ≤ {d1[i2]:.0f}·xH₂<br>
        &nbsp;&nbsp;⟹&nbsp;
        <code style="color:#FFD700">xH₂_min = <b>{ratio:.4f}</b> × xAI₂</code><br><br>
        ⚠️ Mỗi 1.000 tỷ đầu tư AI vào CN Chế biến cần tối thiểu
        <b style="color:#FF4757">{ratio*1000:.0f} tỷ</b> đào tạo lại.
        </div>""", unsafe_allow_html=True)

        # Bảng kịch bản — tất cả Python scalar → list
        xai_max = BUDGET / (1 + ratio)
        rows = []
        for xai2 in [500, 1_000, 2_000, 5_000, 10_000, xai_max]:
            xh2m  = ratio * xai2
            nj2   = float(alpha[i2]) * xai2 + float(b1[i2]) * xh2m
            rows.append({
                "xAI₂ (tỷ)":    int(round(xai2)),
                "xH₂_min (tỷ)": round(xh2m, 1),
                "NetJob₂":       round(nj2, 0),
                "Tổng chi (tỷ)": round(xai2 + xh2m, 1),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if PLOTLY_OK:
            st.plotly_chart(_chart_threshold(), use_container_width=True)

        # Bảng so sánh các ngành
        st.subheader("📊 So sánh gánh nặng đào tạo lại (xH_min/xAI) giữa 8 ngành")
        ratio_all = displaced_coeff / d1
        df_rat = pd.DataFrame({
            "Ngành":              SECTORS,
            "c1·risk":            np.round(displaced_coeff, 3).tolist(),
            "d1":                 d1.tolist(),
            "xH_min/xAI (%)":    np.round(ratio_all * 100, 2).tolist(),
            "Mức gánh nặng":     [
                "🔴 Rất cao" if r > 0.5 else
                "🟡 Trung bình" if r > 0.2 else "🟢 Thấp"
                for r in ratio_all],
        })
        st.dataframe(df_rat, use_container_width=True, hide_index=True)

        st.markdown(f"""
        <div class="sb">
        <b>💡 Kết luận:</b> Ràng buộc binding là <b>Displaced ≤ RetrainCap</b> (không phải NetJob ≥ 0).
        CN Chế biến có c₁=62.4 cao nhất × risk=42% → tỷ lệ dịch chuyển lao động cực cao.
        Chính sách "đồng hành AI–đào tạo" bắt buộc tỷ lệ tối thiểu {ratio*100:.1f}%.
        </div>""", unsafe_allow_html=True)

    # =========================================================================
    # TAB 3 — Câu 9.4.3: Sankey
    # =========================================================================
    with t3:
        _sh("🌊", "Câu 9.4.3 — Biểu đồ Sankey dịch chuyển lao động",
            "Ngành 1 (Nông-Lâm-TS) · 3 (Xây dựng) · 4 (Bán buôn-bán lẻ)")

        st.markdown("""
        <div class="ib">
        <b>Luồng 4 giai đoạn:</b>
        <code>Ngành nguồn → [Bị displaced | Nâng cấp] → [Đào tạo lại | Thất nghiệp tạm] → Kết quả</code><br>
        Ràng buộc <b>Displaced ≤ RetrainCap</b> đảm bảo toàn bộ lao động bị dịch chuyển
        được hệ thống đào tạo hấp thụ — không để thất nghiệp hàng loạt.
        </div>""", unsafe_allow_html=True)

        # Dùng kết quả từ tab 1 nếu có, không thì giải lại
        res = st.session_state.get("r1")
        if res is None or not res.get("ok"):
            with st.spinner("Giải tối ưu để lấy dữ liệu Sankey…"):
                res = _solve()
            st.session_state["r1"] = res

        if res.get("ok") and PLOTLY_OK:
            st.plotly_chart(_chart_sankey(res), use_container_width=True)
        elif not PLOTLY_OK:
            st.warning("Cần cài Plotly: `pip install plotly`")
        else:
            st.error("Chưa có kết quả tối ưu hóa — vui lòng chạy Tab 9.4.1 trước.")

        st.markdown("""
        <div class="sb">
        <b>💡 Giải thích luồng:</b><br>
        • <b>Nông-Lâm-TS</b>: Displaced thấp (risk=18%, c1=5.2) — luồng Upgrade chiếm ưu thế<br>
        • <b>Xây dựng</b>: Risk=25%, c1=18.5 — lao động phổ thông thi công dễ bị thay thế<br>
        • <b>Bán buôn-bán lẻ</b>: c1=48.2, risk=38% → Displaced cao nhất nhóm, cần Retrain ưu tiên
        </div>""", unsafe_allow_html=True)

    # =========================================================================
    # TAB 4 — Câu 9.4.4: Giới hạn 5%
    # =========================================================================
    with t4:
        _sh("🔒", "Câu 9.4.4 (Mở rộng) — Ràng buộc Displaced ≤ 5%·Lᵢ",
            "Không ngành nào được mất quá 5% lao động do tự động hóa")

        thr5 = 0.05 * L_i * 1000   # nghìn người — numpy array thuần

        st.markdown("""
        <div class="wb">
        <b>Ràng buộc mới:</b>
        <code>c1ᵢ·riskᵢ·xAIᵢ ≤ 0.05·Lᵢ·1000</code>
        &nbsp;⟹&nbsp;
        <code>xAIᵢ ≤ (0.05·Lᵢ·1000) / (c1ᵢ·riskᵢ)</code>
        </div>""", unsafe_allow_html=True)

        # Bảng giới hạn xAI_max
        xai_caps = thr5 / displaced_coeff
        df_cap = pd.DataFrame({
            "Ngành":            SECTORS,
            "Lᵢ (tr)":         L_i.tolist(),
            "5%·Lᵢ (nghìn)":  np.round(thr5, 0).astype(int).tolist(),
            "c1·risk":         np.round(displaced_coeff, 3).tolist(),
            "xAI_max (tỷ)":   np.round(xai_caps, 1).tolist(),
            "Đánh giá":        [
                "⚠️ Rất chặt" if v < 50 else
                "🟡 Chặt"     if v < 200 else "✅ Thoải mái"
                for v in xai_caps],
        })
        st.dataframe(df_cap, use_container_width=True, hide_index=True)

        idx_tight = int(np.argmin(xai_caps))
        st.markdown(f"""
        <div class="db">
        ❗ <b>Ngành bị siết chặt nhất:</b> <b>{SECTORS[idx_tight]}</b>
        — xAI_max chỉ {xai_caps[idx_tight]:.1f} tỷ<br>
        Lý do: c1·risk cao → mỗi tỷ đầu tư AI dịch chuyển nhiều lao động nhất.
        </div>""", unsafe_allow_html=True)

        if st.button("▶ Giải với ràng buộc 5%", type="secondary", use_container_width=True):
            with st.spinner("Đang kiểm tra tính khả thi…"):
                st.session_state["r4"] = _solve(0, 0, cap5pct=True)

        if "r4" not in st.session_state:
            st.session_state["r4"] = _solve(0, 0, cap5pct=True)

        res4 = st.session_state["r4"]

        # ── Kiểm tra prob.status = 'optimal' trước khi vẽ bảng ──────────────
        if not res4.get("ok"):
            st.markdown(f"""
            <div class="db">
            ❌ <b>Mô hình vô nghiệm</b> (status: <code>{res4.get('status')}</code>)<br>
            Ràng buộc 5% cùng các ràng buộc khác tạo thành hệ mâu thuẫn.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="sb">✅ <b>Mô hình vẫn có nghiệm</b> với ràng buộc 5%.</div>
            """, unsafe_allow_html=True)

            base_nj = (st.session_state.get("r1") or {}).get("total", 0) or 0
            delta_nj = float(base_nj) - res4["total"]

            # Bảng kết quả — tất cả cột là .tolist()
            df_r4 = pd.DataFrame({
                "Ngành":          SECTORS,
                "xAI (tỷ)":      np.round(res4["xAI"],  1).tolist(),
                "xH (tỷ)":       np.round(res4["xH"],   1).tolist(),
                "Displaced":     np.round(res4["Disp"],  1).tolist(),
                "Giới hạn 5%":   np.round(thr5, 0).astype(int).tolist(),
                "% dùng":        [
                    f"{100*d/t:.1f}%"
                    for d, t in zip(res4["Disp"].tolist(), thr5.tolist())],
                "NetJob":        np.round(res4["NetJob"], 1).tolist(),
            })
            st.dataframe(df_r4, use_container_width=True, hide_index=True)
            st.metric("Tổng NetJob (5% cap)", f"{res4['total']:,.0f}",
                      delta=f"so với không cap: {-delta_nj:+,.0f}")

        st.markdown("""
        <div class="ib">
        <b>🔍 Phân tích tính khả thi:</b><br><br>
        <b>Tại sao vẫn có nghiệm?</b> Ràng buộc 5% chỉ chặn xAI từ trên, không ảnh hưởng xH.
        Solver có thể xAI→0 và tăng xH để thỏa mãn mọi ràng buộc.<br><br>
        <b>Tại sao NetJob giảm?</b> xAI_max nhỏ với ngành risk cao → mất nguồn tạo NewJob
        → solver bù bằng tăng xH (Upgrade nhiều hơn, nhưng ROI thấp hơn với một số ngành).<br><br>
        <b>Khi nào vô nghiệm?</b> Nếu thêm <code>NetJobᵢ ≥ k·Lᵢ</code> với k đủ lớn, 
        khi xAI bị cap quá thấp sẽ không tạo đủ việc làm mới → mâu thuẫn.<br><br>
        <b>Ý nghĩa chính sách:</b> Ràng buộc 5% = ngưỡng chịu đựng xã hội. 
        Cần tăng d₁ (năng lực đào tạo lại) song song để nới lỏng bottleneck.
        </div>""", unsafe_allow_html=True)

    # =========================================================================
    # TAB 5 — Câu 9.5: Thảo luận chính sách
    # =========================================================================
    with t5:
        _sh("💬", "Câu 9.5 — Thảo luận chính sách",
            "Bốn câu hỏi phân tích — bối cảnh Việt Nam 2024")

        qa, qb, qc, qd = st.tabs([
            "(a) Đào tạo lại", "(b) Tài chính-NH", "(c) Nông nghiệp", "(d) Ràng buộc xã hội"
        ])

        with qa:
            st.markdown("""
            ### (a) Ngành nào cần đầu tư đào tạo lại nhiều nhất?

            <div class="ib">
            Kết quả tối ưu dồn xH lớn nhất cho <b>Giáo dục-Đào tạo</b> (β=55, cao nhất)
            và <b>Nông-Lâm-TS</b> (β=45, lao động 13,2 M). Tuy nhiên về <i>áp lực xã hội</i>,
            <b>CN Chế biến chế tạo</b> cần đào tạo lại khẩn cấp nhất:
            </div>

            - **Displaced/xAI = 26,2** (nghìn người/tỷ) — cao nhất 8 ngành
            - 11,5 triệu lao động phổ thông, kỹ năng thấp, khó chuyển đổi nhanh
            - Risk 42% → hơn 4,8 triệu người bị ảnh hưởng trong 10 năm

            <div class="sb">
            ✅ <b>Khớp thực tế:</b> ILO Vietnam 2024 xác nhận dệt may, điện tử, chế biến thực phẩm
            là nhóm dễ bị tổn thương nhất. Các tỉnh Bình Dương, Đồng Nai, Long An đang đối mặt
            áp lực robot hóa từ 2023–2024.
            </div>
            """, unsafe_allow_html=True)

        with qb:
            st.markdown("""
            ### (b) Chiến lược cho Tài chính-Ngân hàng

            <div class="wb">
            <b>Nghịch lý:</b> Risk thay thế cao nhất (52%) nhưng hệ số tạo việc làm mới cũng cao
            (a1=45,8 — đứng thứ 3). α = 45,8 − 72,5×0,52 = <b>8,1</b> → AI vẫn tạo ròng dương,
            nhưng với c1=72,5 (cao nhất), mỗi tỷ AI dịch chuyển 37,7 nghìn người.
            </div>

            **Mô hình khuyến nghị:**
            - Đầu tư AI **vừa phải** — c1 cực cao, không dồn ngân sách lớn vào xAI
            - Ưu tiên **xH mạnh**: kỹ năng số, phân tích dữ liệu, tư vấn tài chính cá nhân
            - Lộ trình: AI xử lý giao dịch routine → nhân viên chuyển sang advisory / risk / FinTech

            <div class="sb">
            ✅ Vietcombank, BIDV, Techcombank đang triển khai chatbot AI + tự động hóa back-office.
            Cần đẩy nhanh "Chương trình nâng cao năng lực số" cho ~550.000 lao động ngành ngân hàng.
            </div>
            """, unsafe_allow_html=True)

        with qc:
            st.markdown("""
            ### (c) Có nên đầu tư xAI vào Nông-Lâm-Thủy sản?

            **Tham số:** a1=8,5 (thấp nhất), risk=18% (thấp nhất), L=13,2M (lớn nhất)  
            α = 8,5 − 5,2×0,18 = **7,56** (dương nhưng thấp nhất 8 ngành)

            <div class="ib">
            <b>Mô hình nói:</b> Đầu tư AI ở mức tối thiểu — ROI việc làm mới thấp nhất.
            Nhưng vì L=13,2M, ngay cả displaced nhỏ cũng tạo số lượng tuyệt đối lớn.
            Chiến lược tối ưu: dùng <b>xH</b> là chủ lực (β=45 — cao nhất cùng với GD-ĐT).
            </div>

            **Đề xuất:**
            - xAI chọn lọc: drone nông nghiệp, IoT tưới tiêu, phân tích đất → **tăng năng suất, không thay thế lao động**
            - xH mạnh: đào tạo lao động nông nghiệp sang logistics, chế biến thực phẩm, du lịch sinh thái

            <div class="sb">
            ✅ Phù hợp "Nông nghiệp thông minh 4.0" (Bộ NN&PTNT): AI hỗ trợ (augmentation),
            không phải thay thế lao động nông thôn.
            </div>
            """, unsafe_allow_html=True)

        with qd:
            st.markdown("""
            ### (d) Biểu diễn nguyên tắc "tốc độ tự động hóa ≤ năng lực đào tạo lại"

            Ràng buộc trong mô hình:

            ```
            Displacedᵢ ≤ RetrainCapᵢ
            ⟺  c1ᵢ · riskᵢ · xAIᵢ  ≤  d1ᵢ · xHᵢ
            ```

            Đây là ràng buộc **binding** thực sự — không phải NetJobᵢ ≥ 0.  
            Nó cân bằng *tốc độ phá vỡ* (do AI) với *tốc độ tái tạo* (do đào tạo).

            **Các ràng buộc bổ sung đề xuất:**

            | Ràng buộc | Ý nghĩa |
            |---|---|
            | `Displacedᵢ ≤ 0.05·Lᵢ` | Không mất quá 5% lao động/ngành/năm |
            | `xHᵢ ≥ 0.3·(xAIᵢ+xHᵢ)` | Tối thiểu 30% ngân sách mỗi ngành cho đào tạo |
            | `xAIᵢ ≤ 2·xHᵢ` | Tỷ lệ AI/Con người không vượt 2:1 |
            | `NetJobᵢ ≥ 0.02·Lᵢ·1000` | Mỗi ngành tạo thêm ≥ 2% việc làm mới |
            | `Σ RetrainCapᵢ ≥ 1.2·Σ Displacedᵢ` | Đệm an toàn 20% năng lực đào tạo dự phòng |

            <div class="sb">
            <b>💡 Đề xuất cơ chế:</b> Quỹ "AI Transition Fund" — doanh nghiệp triển khai AI
            đóng góp ≥15% tiết kiệm chi phí lao động vào quỹ đào tạo lại, đảm bảo
            xHᵢ được tài trợ tương ứng với mức tăng xAIᵢ.
            </div>
            """, unsafe_allow_html=True)

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("""
    <div style="text-align:center;color:#555;font-size:.78rem;padding:8px 0 16px">
      Bài 9 · Mô hình LP — CVXPY / SciPy HiGHS · ILO Vietnam 2024, OECD AI Employment 2024<br>
      NetJob = NewJob + Upgrade − Displaced &nbsp;|&nbsp; Ràng buộc: Displaced ≤ RetrainCap
    </div>""", unsafe_allow_html=True)