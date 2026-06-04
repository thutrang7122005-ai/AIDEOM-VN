"""
Bài 8: Tối ưu động phân bổ liên thời gian 2026–2035
Module Streamlit – phiên bản ổn định cho môi trường Web
"""

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORT
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import cvxpy as cp
    _CVXPY_OK = True
except ImportError:
    _CVXPY_OK = False


# ─────────────────────────────────────────────────────────────────────────────
#  HẰNG SỐ MÔ HÌNH  (dùng làm giá trị mặc định cho sidebar)
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULT = {
    # Khấu hao
    "delta_K":  0.05,
    "delta_D":  0.12,
    "delta_AI": 0.15,
    # Vốn nhân lực
    "theta_H":  0.8,
    "mu":       0.02,
    # Tác động TFP nội sinh
    "phi1": 0.003,   # đóng góp của D vào TFP
    "phi2": 0.002,   # đóng góp của AI vào TFP
    "phi3": 0.004,   # đóng góp của H vào TFP
    # Chiết khấu và CRRA
    "rho":   0.97,
    "gamma": 1.5,
    # Số mũ Cobb-Douglas
    "alpha_K":  0.33,
    "alpha_L":  0.42,
    "alpha_D":  0.10,
    "alpha_AI": 0.08,
    "alpha_H":  0.07,
    # Điều kiện ban đầu
    "K0":  27500.0,
    "L0":  53.9,
    "D0":  20.3,
    "AI0": 86.0,
    "H0":  30.0,
    "A0":  1.0,
    # Thời gian
    "T": 10,
}

YEARS = list(range(2026, 2036))   # 10 năm điều khiển
YEARS_STATE = list(range(2026, 2037))  # 11 điểm trạng thái


# ─────────────────────────────────────────────────────────────────────────────
#  HÀM TIỆN ÍCH MÀU SẮC
# ─────────────────────────────────────────────────────────────────────────────
def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """
    Chuyển mã màu Hex sang chuỗi rgba() chuẩn của CSS/Plotly.

    Hỗ trợ các định dạng đầu vào:
      - "#RRGGBB"   (6 ký tự, có #)
      - "#RGB"      (3 ký tự rút gọn, có #)
      - "RRGGBB"    (6 ký tự, không có #)
      - "RGB"       (3 ký tự rút gọn, không có #)
      - "#RRGGBBAA" (8 ký tự, có sẵn alpha – alpha sẽ bị ghi đè)
      - Định dạng khác (rgb(...), rgba(...), tên màu CSS): trả về nguyên gốc,
        vì Plotly chấp nhận các định dạng này trực tiếp.

    Parameters
    ----------
    hex_color : str
        Mã màu đầu vào.
    alpha : float
        Giá trị độ trong suốt, từ 0.0 (trong suốt) đến 1.0 (đặc). Mặc định 1.0.

    Returns
    -------
    str
        Chuỗi "rgba(r, g, b, alpha)" nếu đầu vào là Hex hợp lệ,
        hoặc chuỗi gốc nếu không thể phân tích.

    Examples
    --------
    >>> hex_to_rgba("#00C9A7", 0.10)
    'rgba(0, 201, 167, 0.10)'
    >>> hex_to_rgba("FF6B6B", 0.50)
    'rgba(255, 107, 107, 0.50)'
    >>> hex_to_rgba("#ABC", 0.20)
    'rgba(170, 187, 204, 0.20)'
    """
    color = hex_color.strip()

    # Chuẩn hoá: bỏ ký tự # ở đầu nếu có
    if color.startswith("#"):
        color = color[1:]

    # Nếu không phải chuỗi hex thuần (0-9, a-f, A-F), trả về nguyên gốc
    try:
        int(color, 16)
    except ValueError:
        # Ví dụ: "rgb(...)", "rgba(...)", tên màu CSS như "red", "blue"
        return hex_color

    # Mở rộng dạng rút gọn #RGB → #RRGGBB
    if len(color) == 3:
        color = "".join(c * 2 for c in color)

    # Hỗ trợ #RRGGBBAA (8 ký tự): lấy 6 ký tự đầu, bỏ alpha gốc
    if len(color) == 8:
        color = color[:6]

    if len(color) != 6:
        # Định dạng không nhận dạng được → trả về nguyên gốc
        return hex_color

    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)

    # Đảm bảo alpha nằm trong [0.0, 1.0]
    alpha = max(0.0, min(1.0, float(alpha)))

    return f"rgba({r}, {g}, {b}, {alpha:.2f})"


# ─────────────────────────────────────────────────────────────────────────────
#  BƯỚC 0 – KIỂM TRA DỮ LIỆU ĐẦU VÀO
# ─────────────────────────────────────────────────────────────────────────────
def _validate_params(p: dict) -> tuple[bool, str]:
    """Kiểm tra các tham số đầu vào trước khi truyền vào CVXPY."""
    checks = {
        "K0 (vốn vật chất)":    p["K0"],
        "D0 (hạ tầng số)":      p["D0"],
        "AI0 (vốn AI)":         p["AI0"],
        "H0 (vốn nhân lực)":    p["H0"],
        "L0 (lao động)":        p["L0"],
        "A0 (TFP)":             p["A0"],
    }
    errors = []
    for name, val in checks.items():
        if val is None:
            errors.append(f"{name} = None")
        elif val <= 0:
            errors.append(f"{name} = {val} ≤ 0 (cp.log không xác định)")

    bound_checks = [
        ("rho",     p["rho"],     0.0, 1.0),
        ("gamma",   p["gamma"],   0.0, 10.0),
        ("delta_K", p["delta_K"], 0.0, 1.0),
        ("delta_D", p["delta_D"], 0.0, 1.0),
        ("delta_AI",p["delta_AI"],0.0, 1.0),
        ("theta_H", p["theta_H"], 0.0, 2.0),
        ("mu",      p["mu"],      0.0, 1.0),
    ]
    for name, val, lo, hi in bound_checks:
        if not (lo < val < hi):
            errors.append(f"{name} = {val} nằm ngoài khoảng ({lo}, {hi})")

    if errors:
        return False, "Tham số không hợp lệ:\n• " + "\n• ".join(errors)
    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
#  BƯỚC 1 – XÂY DỰNG VÀ GIẢI BÀI TOÁN CVXPY
# ─────────────────────────────────────────────────────────────────────────────
def _cobb_douglas_gm(
    K_t, D_t, AI_t, H_t,
    scale, alpha_K, alpha_D, alpha_AI, alpha_H,
):
    """
    Biểu diễn Y_t = scale · K^α_K · D^α_D · AI^α_AI · H^α_H
    bằng cp.geo_mean, đảm bảo DCP compliance.
    """
    sigma   = alpha_K + alpha_D + alpha_AI + alpha_H
    weights = np.array([alpha_K, alpha_D, alpha_AI, alpha_H])
    gm      = cp.geo_mean(cp.vstack([K_t, D_t, AI_t, H_t]), weights)
    return scale * cp.power(gm, sigma)


def _build_and_solve(p: dict, shock_cap_2028: float | None = None) -> dict | None:
    """Xây dựng và giải bài toán tối ưu động CVXPY."""
    T        = p["T"]
    delta_K  = p["delta_K"]
    delta_D  = p["delta_D"]
    delta_AI = p["delta_AI"]
    theta_H  = p["theta_H"]
    mu       = p["mu"]
    phi1     = p["phi1"]
    phi2     = p["phi2"]
    phi3     = p["phi3"]
    rho      = p["rho"]
    gamma    = p["gamma"]
    alpha_K  = p["alpha_K"]
    alpha_L  = p["alpha_L"]
    alpha_D  = p["alpha_D"]
    alpha_AI = p["alpha_AI"]
    alpha_H  = p["alpha_H"]
    K0       = p["K0"]
    L0       = p["L0"]
    D0       = p["D0"]
    AI0      = p["AI0"]
    H0       = p["H0"]
    A0       = p["A0"]

    # Xấp xỉ TFP (SCA 1 vòng)
    A_vals    = np.zeros(T + 1)
    A_vals[0] = A0
    g_A       = phi1 * (D0 / 100.0) + phi2 * (AI0 / 1000.0) + phi3 * (H0 / 100.0)
    for t in range(T):
        A_vals[t + 1] = A_vals[t] * (1.0 + g_A)
    A_vals = np.maximum(A_vals, 1e-12)

    # Khai báo biến CVXPY
    K_v  = cp.Variable(T + 1, nonneg=True, name="K")
    D_v  = cp.Variable(T + 1, nonneg=True, name="D")
    AI_v = cp.Variable(T + 1, nonneg=True, name="AI")
    H_v  = cp.Variable(T + 1, nonneg=True, name="H")
    I_K  = cp.Variable(T, nonneg=True, name="I_K")
    I_D  = cp.Variable(T, nonneg=True, name="I_D")
    I_AI = cp.Variable(T, nonneg=True, name="I_AI")
    I_H  = cp.Variable(T, nonneg=True, name="I_H")
    C    = cp.Variable(T, nonneg=True, name="C")
    Y_aux = cp.Variable(T, nonneg=True, name="Y_aux")

    csts = []
    csts += [K_v[0] == K0, D_v[0] == D0, AI_v[0] == AI0, H_v[0] == H0]

    for t in range(T):
        scale_t  = float(A_vals[t] * (L0 ** alpha_L))
        Y_expr_t = _cobb_douglas_gm(
            K_v[t], D_v[t], AI_v[t], H_v[t],
            scale=scale_t, alpha_K=alpha_K, alpha_D=alpha_D,
            alpha_AI=alpha_AI, alpha_H=alpha_H,
        )
        csts.append(Y_expr_t >= Y_aux[t])

        if shock_cap_2028 is not None and t == 2:
            csts.append(Y_aux[t] <= float(shock_cap_2028))

        csts.append(C[t] + I_K[t] + I_D[t] + I_AI[t] + I_H[t] <= Y_aux[t])
        csts += [
            K_v[t + 1]  == (1.0 - delta_K)  * K_v[t]  + I_K[t],
            D_v[t + 1]  == (1.0 - delta_D)  * D_v[t]  + I_D[t],
            AI_v[t + 1] == (1.0 - delta_AI) * AI_v[t] + I_AI[t],
            H_v[t + 1]  == (1.0 - mu)       * H_v[t]  + theta_H * I_H[t],
        ]
        csts += [I_K[t] >= 1e-3, I_D[t] >= 1e-3, I_AI[t] >= 1e-3,
                 I_H[t] >= 1e-3, C[t]   >= 1e-3]

    disc = np.array([rho**t for t in range(T)])
    if p.get("utility", "log") == "log":
        obj = cp.Maximize(disc @ cp.log(C))
    else:
        one_minus_g = 1.0 - gamma
        obj = cp.Maximize(disc @ (cp.power(C, one_minus_g) / one_minus_g))

    prob = cp.Problem(obj, csts)

    SOLVER_ORDER = [
        (cp.CLARABEL, {}),
        (cp.ECOS,     {"max_iters": 500}),
        (cp.SCS,      {"max_iters": 10000, "eps": 1e-4}),
    ]
    solver_used = None
    last_error  = None

    for solver, kwargs in SOLVER_ORDER:
        try:
            prob.solve(solver=solver, verbose=False, **kwargs)
            status = prob.status or ""
            if status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                solver_used = str(solver)
                break
            last_error = f"Solver {solver} trả về status: {status}"
        except cp.error.SolverError as e:
            last_error = f"SolverError ({solver}): {e}"
        except Exception as e:
            last_error = f"Lỗi không xác định ({solver}): {e}"

    if C.value is None or K_v.value is None:
        st.error(
            f"❌ Không tìm được nghiệm tối ưu.\n\n"
            f"Lỗi cuối: `{last_error}`\n\n"
            "**Gợi ý:** Kiểm tra lại các tham số đầu vào, "
            "đặc biệt các giá trị K₀, D₀, AI₀, H₀ > 0."
        )
        return None

    final_status = prob.status or "unknown"
    if final_status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        st.warning(
            f"⚠️ Solver kết thúc với status `{final_status}`. "
            "Kết quả có thể không chính xác hoàn toàn."
        )

    K_sol    = K_v.value
    D_sol    = D_v.value
    AI_sol   = AI_v.value
    H_sol    = H_v.value
    C_sol    = C.value
    I_K_sol  = I_K.value
    I_D_sol  = I_D.value
    I_AI_sol = I_AI.value
    I_H_sol  = I_H.value

    A_actual    = np.zeros(T + 1)
    A_actual[0] = A0
    for t in range(T):
        A_actual[t + 1] = A_actual[t] * (
            1.0 + phi1 * float(D_sol[t])
                + phi2 * float(AI_sol[t])
                + phi3 * float(H_sol[t])
        )

    Y_actual = np.array([
        A_actual[t]
        * max(K_sol[t],  1e-12) ** alpha_K
        * L0 ** alpha_L
        * max(D_sol[t],  1e-12) ** alpha_D
        * max(AI_sol[t], 1e-12) ** alpha_AI
        * max(H_sol[t],  1e-12) ** alpha_H
        for t in range(T)
    ])

    if shock_cap_2028 is not None:
        Y_actual[2] = min(float(Y_actual[2]), float(shock_cap_2028))

    if p.get("utility", "log") == "log":
        welfare = float(sum(
            rho**t * np.log(max(float(C_sol[t]), 1e-12)) for t in range(T)
        ))
    else:
        welfare = float(sum(
            rho**t * (max(float(C_sol[t]), 1e-12) ** (1 - gamma) / (1 - gamma))
            for t in range(T)
        ))

    return {
        "status":  final_status,
        "solver":  solver_used,
        "welfare": welfare,
        "K": K_sol, "D": D_sol, "AI": AI_sol, "H": H_sol,
        "C": C_sol, "I_K": I_K_sol, "I_D": I_D_sol,
        "I_AI": I_AI_sol, "I_H": I_H_sol,
        "Y": Y_actual, "A": A_actual,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MÔ PHỎNG CHIẾN LƯỢC CỐ ĐỊNH
# ─────────────────────────────────────────────────────────────────────────────
def _simulate_strategy(strategy: str, p: dict) -> dict:
    T        = p["T"]
    delta_K  = p["delta_K"]
    delta_D  = p["delta_D"]
    delta_AI = p["delta_AI"]
    theta_H  = p["theta_H"]
    mu       = p["mu"]
    phi1     = p["phi1"]
    phi2     = p["phi2"]
    phi3     = p["phi3"]
    alpha_K  = p["alpha_K"]
    alpha_L  = p["alpha_L"]
    alpha_D  = p["alpha_D"]
    alpha_AI = p["alpha_AI"]
    alpha_H  = p["alpha_H"]
    L0, A0   = p["L0"], p["A0"]

    K  = np.zeros(T + 1); K[0]  = p["K0"]
    D  = np.zeros(T + 1); D[0]  = p["D0"]
    AI = np.zeros(T + 1); AI[0] = p["AI0"]
    H  = np.zeros(T + 1); H[0]  = p["H0"]
    A  = np.zeros(T + 1); A[0]  = A0
    Y_arr    = np.zeros(T)
    C_arr    = np.zeros(T)
    I_K_arr  = np.zeros(T)
    I_D_arr  = np.zeros(T)
    I_AI_arr = np.zeros(T)
    I_H_arr  = np.zeros(T)

    w = np.array([0.40, 0.25, 0.20, 0.15])

    for t in range(T):
        Y_t = (
            A[t]
            * max(K[t],  1e-12) ** alpha_K
            * L0 ** alpha_L
            * max(D[t],  1e-12) ** alpha_D
            * max(AI[t], 1e-12) ** alpha_AI
            * max(H[t],  1e-12) ** alpha_H
        )
        Y_arr[t] = Y_t

        inv_share = (0.55 if (strategy == "frontload" and t < 3) else
                     0.30 if (strategy == "frontload" and t >= 3) else 0.40)
        total_inv = inv_share * Y_t

        iK, iD, iAI, iH = w * total_inv
        I_K_arr[t]  = iK
        I_D_arr[t]  = iD
        I_AI_arr[t] = iAI
        I_H_arr[t]  = iH
        C_arr[t]    = max(Y_t - total_inv, 1e-9)

        K[t + 1]  = (1 - delta_K)  * K[t]  + iK
        D[t + 1]  = (1 - delta_D)  * D[t]  + iD
        AI[t + 1] = (1 - delta_AI) * AI[t] + iAI
        H[t + 1]  = (1 - mu)       * H[t]  + theta_H * iH
        A[t + 1]  = A[t] * (1 + phi1 * D[t] + phi2 * AI[t] + phi3 * H[t])

    return {
        "K": K, "D": D, "AI": AI, "H": H, "A": A,
        "Y": Y_arr, "C": C_arr,
        "I_K": I_K_arr, "I_D": I_D_arr,
        "I_AI": I_AI_arr, "I_H": I_H_arr,
    }


def _welfare(C_path: np.ndarray, rho: float, utility: str, gamma: float) -> float:
    T = len(C_path)
    if utility == "log":
        return float(sum(rho**t * np.log(max(C_path[t], 1e-12)) for t in range(T)))
    return float(sum(
        rho**t * (max(C_path[t], 1e-12) ** (1 - gamma) / (1 - gamma))
        for t in range(T)
    ))


# ─────────────────────────────────────────────────────────────────────────────
#  HÀM VẼ BIỂU ĐỒ PLOTLY
# ─────────────────────────────────────────────────────────────────────────────
_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="#0E1117",
    plot_bgcolor="#1A1F2E",
    font=dict(color="#FAFAFA", family="IBM Plex Mono", size=11),
)
_GRID = dict(showgrid=True, gridcolor="#2A2F3E")


def _fig_trajectories(res: dict, title: str = "Quỹ đạo tối ưu 2026–2035") -> go.Figure:
    """6 panel: K, D, AI, H (biến trạng thái T+1 điểm) + Y, C (T điểm)."""
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "Vốn vật chất K (nghìn tỷ VND)", "Hạ tầng số D (% GDP)",
            "Vốn AI (nghìn DN số)",           "Vốn nhân lực H (%)",
            "Sản lượng Y (nghìn tỷ VND)",     "Tiêu dùng C (nghìn tỷ VND)",
        ],
        vertical_spacing=0.13, horizontal_spacing=0.10,
    )

    state_data = [
        (YEARS_STATE, res["K"],  "#00C9A7", "K"),
        (YEARS_STATE, res["D"],  "#4ECDC4", "D"),
        (YEARS_STATE, res["AI"], "#FFE66D", "AI"),
        (YEARS_STATE, res["H"],  "#A8E6CF", "H"),
    ]
    ctrl_data = [
        (YEARS, res["Y"], "#00C9A7", "Y"),
        (YEARS, res["C"], "#FF6B6B", "C"),
    ]
    positions = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]

    for (x, y, color, name), (r, c) in zip(state_data + ctrl_data, positions):
        # ── SỬA LỖI: dùng hex_to_rgba thay vì cộng chuỗi "#RRGGBB" + "1A" ──
        # Trước: fillcolor = color + "1A"  → "#00C9A71A" không hợp lệ với Plotly
        # Sau:   fillcolor = hex_to_rgba(color, 0.10) → "rgba(0, 201, 167, 0.10)" ✓
        fill_color = hex_to_rgba(color, alpha=0.10)

        fig.add_trace(go.Scatter(
            x=x, y=y, name=name, mode="lines+markers",
            line=dict(color=color, width=2.5),
            marker=dict(size=6, color=color),
            fill="tozeroy",
            fillcolor=fill_color,         # ← rgba() chuẩn, Plotly chấp nhận
        ), row=r, col=c)

    fig.update_layout(
        title=dict(text=title, font=dict(size=17, color="#FAFAFA"), x=0.5),
        height=710, showlegend=False, margin=dict(t=80, b=40, l=40, r=40),
        **_DARK,
    )
    fig.update_xaxes(**_GRID, tickfont=dict(size=10))
    fig.update_yaxes(**_GRID, tickfont=dict(size=10))
    return fig


def _fig_shock_comparison(res_base: dict, res_shock: dict) -> go.Figure:
    """So sánh 6 chuỗi giữa kịch bản gốc và cú sốc."""
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=["Sản lượng Y", "Tiêu dùng C", "Vốn K",
                        "Đầu tư I_AI", "Đầu tư I_H", "Đầu tư I_D"],
        vertical_spacing=0.16,
    )
    pairs = [
        (res_base["Y"],      res_shock["Y"],      "Y",    1, 1),
        (res_base["C"],      res_shock["C"],      "C",    1, 2),
        (res_base["K"][:10], res_shock["K"][:10], "K",    1, 3),
        (res_base["I_AI"],   res_shock["I_AI"],   "I_AI", 2, 1),
        (res_base["I_H"],    res_shock["I_H"],    "I_H",  2, 2),
        (res_base["I_D"],    res_shock["I_D"],    "I_D",  2, 3),
    ]
    for base_d, shock_d, name, row, col in pairs:
        fig.add_trace(go.Scatter(
            x=YEARS, y=base_d, name=f"{name} – Baseline",
            line=dict(color="#00C9A7", width=2), mode="lines+markers",
            marker=dict(size=5), showlegend=(row == 1 and col == 1),
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=YEARS, y=shock_d, name=f"{name} – Cú sốc",
            line=dict(color="#FF6B6B", width=2, dash="dash"), mode="lines+markers",
            marker=dict(size=5, symbol="x"), showlegend=(row == 1 and col == 1),
        ), row=row, col=col)
        fig.add_vline(x=2028, line_dash="dot", line_color="#FFE66D",
                      row=row, col=col)

    fig.update_layout(
        title=dict(text="Baseline vs Cú sốc 2028", font=dict(size=15), x=0.5),
        height=560, legend=dict(x=1.01, y=1, bgcolor="rgba(0,0,0,0.4)"),
        margin=dict(t=70, b=40, l=40, r=120),
        **_DARK,
    )
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


def _fig_strategy(res_eq: dict, res_fr: dict,
                  w_eq: float, w_fr: float,
                  w_opt: float | None) -> go.Figure:
    """Bar welfare + đường C cho 2 chiến lược + tối ưu (nếu có)."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Tổng phúc lợi W", "Quỹ đạo tiêu dùng C"])

    labels = ["Trải đều", "Front-load"]
    vals   = [w_eq, w_fr]
    colors = ["#4ECDC4", "#FF6B6B"]
    if w_opt is not None:
        labels.insert(0, "Tối ưu (CVXPY)")
        vals.insert(0, w_opt)
        colors.insert(0, "#00C9A7")

    fig.add_trace(go.Bar(
        x=labels, y=vals, marker_color=colors,
        text=[f"{v:.3f}" for v in vals], textposition="outside",
    ), row=1, col=1)

    for res, label, color, dash in [
        (res_eq, "Trải đều",   "#4ECDC4", "solid"),
        (res_fr, "Front-load", "#FF6B6B", "dot"),
    ]:
        fig.add_trace(go.Scatter(
            x=YEARS, y=res["C"], name=label, mode="lines+markers",
            line=dict(color=color, width=2.5, dash=dash), marker=dict(size=6),
        ), row=1, col=2)

    fig.update_layout(
        height=380, legend=dict(bgcolor="rgba(0,0,0,0.4)"),
        margin=dict(t=60, b=30, l=40, r=40),
        **_DARK,
    )
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


def _fig_inv_ratio(I_AI: np.ndarray, I_H: np.ndarray) -> go.Figure:
    """Tỷ lệ đầu tư AI / Nhân lực theo năm."""
    ratio = I_AI / np.maximum(I_H, 1e-12)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=YEARS, y=ratio, mode="lines+markers",
        line=dict(color="#FFE66D", width=2.5), marker=dict(size=7),
        name="I_AI / I_H",
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="#FF6B6B",
                  annotation_text="Cân bằng AI = H")
    fig.update_layout(
        yaxis_title="Tỷ lệ I_AI / I_H", height=280,
        margin=dict(t=30, b=30, l=40, r=40),
        **_DARK,
    )
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT STREAMLIT
# ─────────────────────────────────────────────────────────────────────────────
def run(df_macro=None, df_sectors=None, df_regions=None):
    """Entry point chính cho module Bài 8 trong ứng dụng Streamlit."""

    if not _CVXPY_OK:
        st.error("❌ Thư viện `cvxpy` chưa được cài đặt. Chạy: `pip install cvxpy`")
        return

    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#0F3460 0%,#16213E 60%,#0E1117 100%);
                    padding:1.8rem 2.5rem;border-radius:12px;margin-bottom:1.5rem;
                    border-left:4px solid #00C9A7;">
            <h1 style="color:#00C9A7;font-family:'IBM Plex Mono',monospace;
                       margin:0;font-size:1.55rem;">
                BÀI 8 – TỐI ƯU ĐỘNG PHÂN BỔ LIÊN THỜI GIAN 2026–2035
            </h1>
            <p style="color:#A0AEC0;margin:0.4rem 0 0;font-size:0.88rem;">
                Mô hình Cobb-Douglas mở rộng · Log-linearize · CVXPY (CLARABEL/ECOS/SCS)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("⚙️ Cấu hình mô hình")

        utility = st.radio(
            "Hàm thỏa dụng U(C)",
            options=["log", "crra"],
            format_func=lambda x: "Logarit  ln(C)" if x == "log"
                                  else "CRRA  C^(1−γ)/(1−γ),  γ=1.5",
        )
        rho = st.slider("Hệ số chiết khấu ρ", 0.85, 0.99, 0.97, 0.01,
                        help="ρ=0.97 → dài hạn; ρ=0.90 → ngắn hạn")

        st.markdown("---")
        shock_2028 = st.checkbox(
            "🌪️ Cú sốc 2028 (−8% sản lượng)",
            help="Mô phỏng khủng hoảng tương tự bão Yagi 2024",
        )

        st.markdown("---")
        st.subheader("Điều kiện ban đầu (2026)")
        K0  = st.number_input("K₀ – Vốn vật chất (nghìn tỷ VND)",
                               min_value=1.0, value=27500.0, step=500.0)
        D0  = st.number_input("D₀ – Hạ tầng số (% GDP)",
                               min_value=0.1, value=20.3, step=0.5)
        AI0 = st.number_input("AI₀ – Vốn AI (nghìn DN số)",
                               min_value=0.1, value=86.0, step=5.0)
        H0  = st.number_input("H₀ – Vốn nhân lực (%)",
                               min_value=0.1, value=30.0, step=1.0)

    params = {
        **_DEFAULT,
        "rho":     rho,
        "utility": utility,
        "K0": K0, "D0": D0, "AI0": AI0, "H0": H0,
    }

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        solve_btn = st.button("▶ Giải bài toán", type="primary",
                              use_container_width=True)

    _ss = st.session_state
    for key in ("res_base", "res_shock", "p_used"):
        if key not in _ss:
            _ss[key] = None

    if solve_btn:
        ok, err_msg = _validate_params(params)
        if not ok:
            st.error(f"❌ {err_msg}")
        else:
            with st.spinner("⏳ Đang giải kịch bản gốc (CLARABEL → ECOS → SCS)…"):
                _ss.res_base = _build_and_solve(params, shock_cap_2028=None)
                _ss.p_used   = params

            if _ss.res_base is not None:
                s = _ss.res_base
                st.success(
                    f"✅ Giải xong — solver: **{s['solver']}** | "
                    f"status: `{s['status']}` | W = `{s['welfare']:.4f}`"
                )

            if shock_2028 and _ss.res_base is not None:
                Y_base_2028 = float(_ss.res_base["Y"][2])
                cap = Y_base_2028 * 0.92
                with st.spinner("⏳ Đang giải kịch bản cú sốc 2028…"):
                    _ss.res_shock = _build_and_solve(params, shock_cap_2028=cap)
                if _ss.res_shock is not None:
                    st.info(
                        f"🌪️ Cú sốc: Y₂₀₂₈ bị giới hạn ≤ {cap:.1f} nghìn tỷ "
                        f"(−8% so với baseline {Y_base_2028:.1f})"
                    )
            else:
                _ss.res_shock = None

    res_base  = _ss.res_base
    res_shock = _ss.res_shock
    p_used    = _ss.p_used or params

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Quỹ đạo tối ưu",
        "🌪️ Phân tích cú sốc 2028",
        "⚖️ So sánh chiến lược",
        "📋 Số liệu & Chính sách",
    ])

    with tab1:
        if res_base is None:
            st.info("Nhấn **▶ Giải bài toán** để xem kết quả.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Tổng phúc lợi W",   f"{res_base['welfare']:.4f}")
            c2.metric("Y₂₀₃₅ (nghìn tỷ)", f"{res_base['Y'][-1]:.1f}")
            c3.metric("C₂₀₃₅ (nghìn tỷ)", f"{res_base['C'][-1]:.1f}")
            c4.metric("Solver",            res_base["solver"] or "—")

            st.plotly_chart(_fig_trajectories(res_base), use_container_width=True)

            st.markdown("#### Tỷ lệ đầu tư AI / Nhân lực theo năm")
            st.plotly_chart(
                _fig_inv_ratio(res_base["I_AI"], res_base["I_H"]),
                use_container_width=True,
            )

    with tab2:
        st.markdown(
            "**Giả định:** Năm 2028 xảy ra khủng hoảng (bão Yagi 2024 mô phỏng), "
            "làm sản lượng giảm **8%** so với kế hoạch. "
            "Bật checkbox ở sidebar rồi nhấn **▶ Giải bài toán**."
        )
        if res_base is None:
            st.info("Cần giải kịch bản gốc trước.")
        elif res_shock is None:
            st.info(
                "Bật **🌪️ Cú sốc 2028** ở sidebar và nhấn ▶ Giải lại "
                "để so sánh hai kịch bản."
            )
        else:
            dw = res_shock["welfare"] - res_base["welfare"]
            ca, cb, cc = st.columns(3)
            ca.metric("W – Kịch bản gốc", f"{res_base['welfare']:.4f}")
            cb.metric("W – Sau cú sốc",   f"{res_shock['welfare']:.4f}")
            cc.metric("∆W (thiệt hại)",   f"{dw:.4f}", delta_color="inverse")

            st.plotly_chart(_fig_shock_comparison(res_base, res_shock),
                            use_container_width=True)

            st.markdown("#### Điều chỉnh đầu tư sau cú sốc (so với baseline)")
            rows = []
            for t in range(p_used["T"]):
                if t >= 2:
                    rows.append({
                        "Năm":   YEARS[t],
                        "∆I_K":  round(float(res_shock["I_K"][t])  - float(res_base["I_K"][t]),  2),
                        "∆I_D":  round(float(res_shock["I_D"][t])  - float(res_base["I_D"][t]),  2),
                        "∆I_AI": round(float(res_shock["I_AI"][t]) - float(res_base["I_AI"][t]), 2),
                        "∆I_H":  round(float(res_shock["I_H"][t])  - float(res_base["I_H"][t]),  2),
                        "∆C":    round(float(res_shock["C"][t])    - float(res_base["C"][t]),    2),
                    })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab3:
        st.markdown(
            "**Hai kịch bản phi tối ưu (luật cố định):**\n"
            "- 📊 **Trải đều**: 40% GDP/năm, cố định mọi năm\n"
            "- 🚀 **Front-load**: 55% GDP (3 năm đầu) → 30% GDP (7 năm sau)"
        )
        with st.spinner("Đang mô phỏng chiến lược cố định…"):
            res_eq = _simulate_strategy("equal",     p_used)
            res_fr = _simulate_strategy("frontload", p_used)

        w_eq  = _welfare(res_eq["C"], p_used["rho"], p_used["utility"], p_used["gamma"])
        w_fr  = _welfare(res_fr["C"], p_used["rho"], p_used["utility"], p_used["gamma"])
        w_opt = res_base["welfare"] if res_base is not None else None

        st.plotly_chart(
            _fig_strategy(res_eq, res_fr, w_eq, w_fr, w_opt),
            use_container_width=True,
        )

        winner   = "Front-load" if w_fr > w_eq else "Trải đều"
        diff_pct = abs(w_fr - w_eq) / max(abs(w_eq), 1e-12) * 100
        st.markdown(
            f"""
            <div style="background:#1A1F2E;border-radius:10px;padding:1.1rem 1.4rem;
                        border-left:4px solid #FFE66D;margin-top:0.5rem;">
                <b style="color:#FFE66D;">Kết luận:</b>
                <span style="color:#FAFAFA;"> Chiến lược <b>{winner}</b> cho phúc lợi
                cao hơn (chênh lệch ~{diff_pct:.1f}%).</span><br><br>
                <span style="color:#A0AEC0;font-size:0.87rem;">
                💡 Front-load xây dựng nền tảng vốn sớm → sản lượng tích lũy cao hơn
                ở các năm sau → tiêu dùng tương lai bù đắp hi sinh ban đầu.
                Chiến lược <em>tối ưu</em> (CVXPY) luôn vượt cả hai vì nó điều chỉnh
                linh hoạt từng năm thay vì dùng tỷ lệ cố định.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab4:
        if res_base is None:
            st.info("Giải bài toán để xem số liệu chi tiết.")
        else:
            T = p_used["T"]

            st.markdown("#### Biến điều khiển (đầu tư & tiêu dùng)")
            df_ctrl = pd.DataFrame({
                "Năm":          YEARS,
                "Y (nghìn tỷ)": np.round(res_base["Y"],    2),
                "C (nghìn tỷ)": np.round(res_base["C"],    2),
                "I_K":          np.round(res_base["I_K"],  2),
                "I_D":          np.round(res_base["I_D"],  2),
                "I_AI":         np.round(res_base["I_AI"], 2),
                "I_H":          np.round(res_base["I_H"],  2),
            })
            st.dataframe(df_ctrl, use_container_width=True, hide_index=True)

            st.markdown("#### Biến trạng thái (cuối mỗi năm)")
            df_state = pd.DataFrame({
                "Năm":     YEARS_STATE,
                "K":       np.round(res_base["K"],  2),
                "D":       np.round(res_base["D"],  2),
                "AI":      np.round(res_base["AI"], 2),
                "H":       np.round(res_base["H"],  2),
                "TFP (A)": np.round(res_base["A"],  4),
            })
            st.dataframe(df_state, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown(
                """
                #### 💬 Thảo luận chính sách (Câu 8.4)

                **a) Front-loaded hay Back-loaded?**
                Mô hình thường đề xuất đầu tư *front-loaded* vào K và D vì
                hiệu ứng tích lũy: vốn hình thành sớm đóng góp vào sản lượng
                nhiều kỳ liên tiếp. Ngược lại, đầu tư I_H có thể được dàn trải
                vì vốn nhân lực tích lũy bền vững hơn và ít khấu hao hơn AI.

                **b) Tỷ lệ I_AI / I_H có ổn định không?**
                Thông thường tỷ lệ này giảm dần theo thời gian, phản ánh nguyên
                tắc *nhân lực đi trước AI*: nền kinh tế cần xây dựng năng lực
                hấp thụ công nghệ (H) trước khi đẩy mạnh đầu tư AI, nếu không
                hiệu quả biên của AI sẽ thấp.

                **c) ρ = 0.97 vs. ρ = 0.90?**
                ρ thấp hơn → chính phủ chiết khấu tương lai mạnh hơn → ưu tiên
                tiêu dùng hiện tại → đầu tư dài hạn (R&D, giáo dục) bị *dưới
                đầu tư*. Đây là cơ sở lý thuyết giải thích bẫy đầu tư ngắn hạn:
                chu kỳ bầu cử 4–5 năm tạo ra ρ hiệu dụng thấp hơn ρ xã hội tối ưu.
                """
            )


# ─────────────────────────────────────────────────────────────────────────────
#  CHẠY ĐỘC LẬP
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    st.set_page_config(
        page_title="Bài 8 – Tối ưu động 2026–2035",
        layout="wide",
        page_icon="📊",
    )
    run()