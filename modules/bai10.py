"""
Bài 10: Quy hoạch ngẫu nhiên hai giai đoạn dưới bất định
=========================================================
Hoạch định ngân sách đầu tư số Việt Nam 2026–2030
dưới các kịch bản kinh tế toàn cầu bất định.

Solver : HiGHS  (pip install highspy)
UI     : Streamlit tabs – đồng bộ với hệ thống bài tập
"""

import streamlit as st
import pandas as pd
import pyomo.environ as pyo

# ══════════════════════════════════════════════════════════
# A. DỮ LIỆU BÀI TOÁN
# ══════════════════════════════════════════════════════════

ITEMS = ['I', 'D', 'AI', 'H']
ITEM_LABELS = {
    'I':  'Hạ tầng vật chất',
    'D':  'Hạ tầng số',
    'AI': 'Trí tuệ nhân tạo',
    'H':  'Vốn nhân lực',
}

SCENARIOS = ['s1', 's2', 's3', 's4']
SCENARIO_LABELS = {
    's1': '🟢 s1 – Lạc quan',
    's2': '🔵 s2 – Cơ sở',
    's3': '🟡 s3 – Bi quan',
    's4': '🔴 s4 – Khủng hoảng',
}
SCENARIO_SHORT = {
    's1': 'Lạc quan',
    's2': 'Cơ sở',
    's3': 'Bi quan',
    's4': 'Khủng hoảng',
}

PROB   = {'s1': 0.30, 's2': 0.45, 's3': 0.20, 's4': 0.05}
BETA   = {'I': 1.00,  'D': 1.10,  'AI': 1.25, 'H': 0.95}
BETA_S = {
    ('s1','I'):1.25, ('s1','D'):1.35, ('s1','AI'):1.55, ('s1','H'):1.05,
    ('s2','I'):1.00, ('s2','D'):1.10, ('s2','AI'):1.25, ('s2','H'):0.95,
    ('s3','I'):0.75, ('s3','D'):0.85, ('s3','AI'):0.90, ('s3','H'):1.00,
    ('s4','I'):0.40, ('s4','D'):0.50, ('s4','AI'):0.55, ('s4','H'):1.10,
}

BUDGET1 = 65_000   # tỷ VND – phân bổ giai đoạn 1
BUDGET2 = 15_000   # tỷ VND – dự phòng giai đoạn 2


# ══════════════════════════════════════════════════════════
# B. SOLVER & MÔ HÌNH PYOMO
# ══════════════════════════════════════════════════════════

def _solver():
    s = pyo.SolverFactory('highs')
    if not s.available():
        st.error(
            "❌ **Solver HiGHS chưa được cài đặt.**  \n"
            "Chạy lệnh: `pip install highspy` rồi khởi động lại ứng dụng."
        )
        st.stop()
    return s


def _solve_sp(fixed_x=None):
    """
    Giải SP hai giai đoạn. Nếu fixed_x != None thì cố định first-stage
    (dùng để tính EEV).
    Returns: (obj, x_dict, y_dict)
    """
    m = pyo.ConcreteModel()
    m.J      = pyo.Set(initialize=ITEMS)
    m.S      = pyo.Set(initialize=SCENARIOS)
    m.p      = pyo.Param(m.S, initialize=PROB)
    m.beta   = pyo.Param(m.J, initialize=BETA)
    m.beta_s = pyo.Param(m.S, m.J, initialize=BETA_S)

    m.x = pyo.Var(m.J, within=pyo.NonNegativeReals)
    m.y = pyo.Var(m.S, m.J, within=pyo.NonNegativeReals)

    if fixed_x is not None:
        for j in ITEMS:
            m.x[j].fix(fixed_x[j])

    m.budget1 = pyo.Constraint(expr=sum(m.x[j] for j in m.J) <= BUDGET1)

    def _b2(m, s):
        return sum(m.y[s, j] for j in m.J) <= BUDGET2
    m.budget2 = pyo.Constraint(m.S, rule=_b2)

    def _ai(m, s):
        return m.y[s, 'AI'] <= 0.5 * m.x['H']
    m.ai_cap = pyo.Constraint(m.S, rule=_ai)

    def _obj(m):
        first  = sum(m.beta[j] * m.x[j] for j in m.J)
        second = sum(m.p[s] * sum(m.beta_s[s,j] * m.y[s,j] for j in m.J)
                     for s in m.S)
        return first + second
    m.obj = pyo.Objective(rule=_obj, sense=pyo.maximize)

    res = _solver().solve(m, tee=False)
    if res.solver.termination_condition != pyo.TerminationCondition.optimal:
        st.warning(f"⚠️ Solver: {res.solver.termination_condition}")

    x_sol = {j: pyo.value(m.x[j]) for j in ITEMS}
    y_sol = {(s,j): pyo.value(m.y[s,j]) for s in SCENARIOS for j in ITEMS}
    return pyo.value(m.obj), x_sol, y_sol


def _solve_det(scenario):
    """Giải xác định cho kịch bản đơn. Returns (obj, x_dict, y_dict)."""
    m = pyo.ConcreteModel()
    m.J = pyo.Set(initialize=ITEMS)
    m.x = pyo.Var(m.J, within=pyo.NonNegativeReals)
    m.y = pyo.Var(m.J, within=pyo.NonNegativeReals)

    m.b1  = pyo.Constraint(expr=sum(m.x[j] for j in m.J) <= BUDGET1)
    m.b2  = pyo.Constraint(expr=sum(m.y[j] for j in m.J) <= BUDGET2)
    m.ai  = pyo.Constraint(expr=m.y['AI'] <= 0.5 * m.x['H'])

    def _obj(m):
        return (sum(BETA[j]  * m.x[j] for j in m.J) +
                sum(BETA_S[(scenario,j)] * m.y[j] for j in m.J))
    m.obj = pyo.Objective(rule=_obj, sense=pyo.maximize)

    _solver().solve(m, tee=False)
    x_sol = {j: pyo.value(m.x[j]) for j in ITEMS}
    y_sol = {j: pyo.value(m.y[j]) for j in ITEMS}
    return pyo.value(m.obj), x_sol, y_sol


def _solve_ev():
    """Giải bài toán kỳ vọng (EV – thay kịch bản bằng trung bình trọng số)."""
    beta_avg = {j: sum(PROB[s]*BETA_S[(s,j)] for s in SCENARIOS) for j in ITEMS}

    m = pyo.ConcreteModel()
    m.J = pyo.Set(initialize=ITEMS)
    m.x = pyo.Var(m.J, within=pyo.NonNegativeReals)
    m.y = pyo.Var(m.J, within=pyo.NonNegativeReals)

    m.b1 = pyo.Constraint(expr=sum(m.x[j] for j in m.J) <= BUDGET1)
    m.b2 = pyo.Constraint(expr=sum(m.y[j] for j in m.J) <= BUDGET2)
    m.ai = pyo.Constraint(expr=m.y['AI'] <= 0.5 * m.x['H'])

    def _obj(m):
        return (sum(BETA[j]     * m.x[j] for j in m.J) +
                sum(beta_avg[j] * m.y[j] for j in m.J))
    m.obj = pyo.Objective(rule=_obj, sense=pyo.maximize)

    _solver().solve(m, tee=False)
    x_sol = {j: pyo.value(m.x[j]) for j in ITEMS}
    y_sol = {j: pyo.value(m.y[j]) for j in ITEMS}
    return pyo.value(m.obj), x_sol, y_sol


def _solve_robust(ws_dict):
    """
    Minimax Regret:
      min  eta
      s.t. eta >= WS(s) - Z(x,y,s)   ∀s
           budget & ai constraints
    """
    m = pyo.ConcreteModel()
    m.J      = pyo.Set(initialize=ITEMS)
    m.S      = pyo.Set(initialize=SCENARIOS)
    m.beta   = pyo.Param(m.J, initialize=BETA)
    m.beta_s = pyo.Param(m.S, m.J, initialize=BETA_S)
    m.ws     = pyo.Param(m.S, initialize=ws_dict)

    m.x   = pyo.Var(m.J, within=pyo.NonNegativeReals)
    m.y   = pyo.Var(m.S, m.J, within=pyo.NonNegativeReals)
    m.eta = pyo.Var(within=pyo.NonNegativeReals)

    m.b1 = pyo.Constraint(expr=sum(m.x[j] for j in m.J) <= BUDGET1)

    def _b2(m, s):
        return sum(m.y[s,j] for j in m.J) <= BUDGET2
    m.b2 = pyo.Constraint(m.S, rule=_b2)

    def _ai(m, s):
        return m.y[s,'AI'] <= 0.5 * m.x['H']
    m.ai = pyo.Constraint(m.S, rule=_ai)

    def _reg(m, s):
        z_s = (sum(m.beta[j]*m.x[j] for j in m.J) +
               sum(m.beta_s[s,j]*m.y[s,j] for j in m.J))
        return m.eta >= m.ws[s] - z_s
    m.regret = pyo.Constraint(m.S, rule=_reg)

    m.obj = pyo.Objective(expr=m.eta, sense=pyo.minimize)
    _solver().solve(m, tee=False)

    x_sol = {j: pyo.value(m.x[j]) for j in ITEMS}
    y_sol = {(s,j): pyo.value(m.y[s,j]) for s in SCENARIOS for j in ITEMS}
    return pyo.value(m.eta), x_sol, y_sol


# ══════════════════════════════════════════════════════════
# C. TIỆN ÍCH HIỂN THỊ
# ══════════════════════════════════════════════════════════

def _vnd(v):
    """Định dạng số tỷ VND có dấu phẩy, 1 chữ số thập phân."""
    if v is None:
        return "—"
    return f"{v:,.1f}"


def _pct(v, total):
    return f"{v/total*100:.1f}%"


def _x_df(x_dict, label="Phân bổ (tỷ VND)"):
    """Tạo DataFrame first-stage từ dict x."""
    rows = []
    for j in ITEMS:
        rows.append({
            "Hạng mục": ITEM_LABELS[j],
            label: _vnd(x_dict[j]),
            "Tỷ trọng": _pct(x_dict[j], BUDGET1),
        })
    rows.append({
        "Hạng mục": "**Tổng**",
        label: _vnd(sum(x_dict.values())),
        "Tỷ trọng": _pct(sum(x_dict.values()), BUDGET1),
    })
    return pd.DataFrame(rows)


def _y_df(y_dict):
    """Tạo DataFrame second-stage (nhiều kịch bản)."""
    rows = []
    for s in SCENARIOS:
        row = {"Kịch bản": SCENARIO_LABELS[s]}
        total = 0
        for j in ITEMS:
            v = y_dict.get((s,j), y_dict.get(j, 0))
            row[ITEM_LABELS[j]] = _vnd(v)
            total += v
        row["Tổng"] = _vnd(total)
        rows.append(row)
    return pd.DataFrame(rows)


def _divider():
    st.markdown("<hr style='border:1px solid #e0e0e0;margin:18px 0'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# D. ENTRY POINT – hàm run() tương thích dispatch Streamlit
# ══════════════════════════════════════════════════════════

def run(df_macro=None, df_sectors=None, df_regions=None):
    # ── Tiêu đề bài ──
    st.markdown("## Bài 10 · Quy hoạch ngẫu nhiên hai giai đoạn")
    st.markdown(
        """
        **Bài toán:** Hoạch định ngân sách đầu tư số Việt Nam 2026–2030 dưới bất định kinh tế toàn cầu.  
        Tổng ngân sách **80.000 tỷ VND** — phân bổ ≤ 65.000 (giai đoạn 1) + dự phòng ≤ 15.000 (giai đoạn 2).
        """
    )

    # ── Bảng kịch bản & hệ số β ──
    with st.expander("📋 Xem dữ liệu đầu vào: kịch bản & hệ số β", expanded=False):
        sc_rows = []
        for s in SCENARIOS:
            r = {
                "Kịch bản":  SCENARIO_LABELS[s],
                "Xác suất":  f"{PROB[s]:.0%}",
            }
            r.update({ITEM_LABELS[j]: BETA_S[(s,j)] for j in ITEMS})
            sc_rows.append(r)
        sc_df = pd.DataFrame(sc_rows)
        st.dataframe(sc_df, use_container_width=True, hide_index=True)
        st.caption(
            "Hệ số β^s_j phản ánh hiệu quả đầu tư bổ sung trong từng kịch bản. "
            "Lưu ý β_H tăng trong khủng hoảng: nhân lực có kỹ năng hấp thụ cú sốc tốt hơn."
        )

    _divider()

    # ══════════════════════════════════════════
    # TABS
    # ══════════════════════════════════════════
    tab1, tab2, tab3, tab4 = st.tabs([
        "📐 Mô hình SP",
        "🔍 So sánh SP vs EV",
        "📊 VSS & EVPI",
        "🛡️ Robust Optimization",
    ])

    # ─────────────────────────────────────────
    # TAB 1 – MÔ HÌNH SP
    # ─────────────────────────────────────────
    with tab1:
        st.markdown("### 10.5.1 · Mô hình Stochastic Programming (SP)")
        st.markdown(
            r"""
            Bài toán tối ưu hai giai đoạn:

            $$\max \sum_j \beta_j x_j + \sum_{s \in S} p_s \left[\sum_j \beta^s_j y^s_j\right]$$

            **Ràng buộc:**  
            - $\sum_j x_j \leq 65.000$ (ngân sách first-stage)  
            - $\sum_j y^s_j \leq 15.000,\ \forall s$ (dự phòng second-stage)  
            - $y^s_{AI} \leq 0{,}5 \cdot x_H,\ \forall s$ (AI phụ thuộc nhân lực)  
            - $x_j,\ y^s_j \geq 0$
            """
        )

        with st.spinner("⏳ Đang giải SP hai giai đoạn..."):
            sp_obj, sp_x, sp_y = _solve_sp()

        st.success(
            f"✅ **Giá trị hàm mục tiêu SP\\* = {sp_obj:,.2f}** "
            f"(đơn vị: tỷ VND × hệ số hiệu quả)"
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Quyết định First-stage $x_j$")
            st.dataframe(
                _x_df(sp_x),
                use_container_width=True,
                hide_index=True,
            )
        with col_b:
            st.markdown("#### Phân bổ ngân sách first-stage")
            chart_data = pd.DataFrame({
                "Hạng mục": [ITEM_LABELS[j] for j in ITEMS],
                "Tỷ trọng (%)": [round(sp_x[j]/BUDGET1*100, 1) for j in ITEMS],
            }).set_index("Hạng mục")
            st.bar_chart(chart_data)

        st.markdown("#### Quyết định Second-stage $y^s_j$ (điều chỉnh theo kịch bản)")
        st.dataframe(_y_df(sp_y), use_container_width=True, hide_index=True)

        st.info(
            "💡 **Nhận xét:** Quyết định first-stage $x_j$ phải được đưa ra **trước** khi biết kịch bản. "
            "Quyết định second-stage $y^s_j$ là phản ứng linh hoạt **sau** khi kịch bản được xác định. "
            "Ràng buộc $y^s_{AI} \\leq 0{,}5 \\cdot x_H$ khiến việc đầu tư vốn nhân lực sớm trở thành "
            "\"quyền chọn thực\" (real option) để khai thác AI trong tương lai."
        )

    # ─────────────────────────────────────────
    # TAB 2 – SO SÁNH SP vs EV vs DET
    # ─────────────────────────────────────────
    with tab2:
        st.markdown("### 10.5.2 · So sánh SP, EV và Deterministic")
        st.markdown(
            """
            | Phương pháp | Mô tả |
            |-------------|-------|
            | **SP** | Tối ưu hai giai đoạn, xét đầy đủ bất định xác suất |
            | **EV** | Thay tất cả kịch bản bằng **một kịch bản trung bình** (kỳ vọng) |
            | **DS-s1…s4** | Giải xác định khi **biết trước** từng kịch bản |
            """
        )

        with st.spinner("⏳ Đang giải EV và 4 bài toán xác định..."):
            ev_obj, ev_x, ev_y   = _solve_ev()
            det                  = {s: _solve_det(s) for s in SCENARIOS}

        # ── Bảng so sánh first-stage ──
        st.markdown("#### So sánh phân bổ first-stage $x_j$ (tỷ VND)")

        comp_rows = []
        # SP
        comp_rows.append({
            "Phương pháp": "🏆 SP (Stochastic)",
            **{ITEM_LABELS[j]: _vnd(sp_x[j]) for j in ITEMS},
            "Tổng": _vnd(sum(sp_x.values())),
        })
        # EV
        comp_rows.append({
            "Phương pháp": "📊 EV (Kỳ vọng)",
            **{ITEM_LABELS[j]: _vnd(ev_x[j]) for j in ITEMS},
            "Tổng": _vnd(sum(ev_x.values())),
        })
        # Deterministic per scenario
        for s in SCENARIOS:
            _, dx, _ = det[s]
            comp_rows.append({
                "Phương pháp": f"DS – {SCENARIO_SHORT[s]}",
                **{ITEM_LABELS[j]: _vnd(dx[j]) for j in ITEMS},
                "Tổng": _vnd(sum(dx.values())),
            })

        st.dataframe(
            pd.DataFrame(comp_rows),
            use_container_width=True,
            hide_index=True,
        )

        # ── So sánh hạng mục H riêng ──
        h_sp  = sp_x['H']
        h_ev  = ev_x['H']
        diff  = h_sp - h_ev
        sign  = "nhiều hơn" if diff >= 0 else "ít hơn"

        st.markdown("#### Phân tích vốn nhân lực H (câu hỏi thảo luận a)")
        col1, col2, col3 = st.columns(3)
        col1.metric("H trong SP",  _vnd(h_sp) + " tỷ",  help="First-stage SP")
        col2.metric("H trong EV",  _vnd(h_ev) + " tỷ",  help="First-stage EV")
        col3.metric("Chênh lệch",  _vnd(abs(diff)) + " tỷ",
                    delta=f"SP {sign} EV",
                    delta_color="normal" if diff >= 0 else "inverse")

        st.info(
            f"💡 **Lý giải:** SP phân bổ H **{sign} {abs(diff):,.0f} tỷ VND** so với EV.  \n"
            "Trong mô hình SP, đầu tư $x_H$ còn đóng vai trò **nới lỏng ràng buộc** "
            "$y^s_{AI} \\leq 0{,}5 \\cdot x_H$ — tức là tăng cơ hội khai thác AI "
            "nếu kịch bản lạc quan xảy ra. EV bỏ qua giá trị \"quyền chọn\" này "
            "vì nó chỉ thấy một kịch bản trung bình."
        )

        # ── Bảng giá trị obj từng phương pháp ──
        st.markdown("#### Giá trị hàm mục tiêu theo phương pháp")
        obj_rows = [
            {"Phương pháp": "SP*",      "Obj": _vnd(sp_obj), "Ghi chú": "Tối ưu dưới bất định"},
            {"Phương pháp": "EV",       "Obj": _vnd(ev_obj), "Ghi chú": "Dùng kịch bản trung bình"},
        ]
        for s in SCENARIOS:
            obj_rows.append({
                "Phương pháp": f"DS – {SCENARIO_SHORT[s]}",
                "Obj": _vnd(det[s][0]),
                "Ghi chú": f"Biết trước kịch bản {s}",
            })
        st.dataframe(pd.DataFrame(obj_rows), use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────
    # TAB 3 – VSS & EVPI
    # ─────────────────────────────────────────
    with tab3:
        st.markdown("### 10.5.3 · VSS và EVPI")
        st.markdown(
            r"""
            | Chỉ số | Công thức | Ý nghĩa |
            |--------|-----------|---------|
            | **SP\*** | Giải SP đầy đủ | Lợi ích tối ưu dưới bất định |
            | **EEV** | $SP(x^{EV}, \cdot)$ — cố định $x^{EV}$, tối ưu $y$ | Lợi ích khi **bỏ qua** bất định |
            | **VSS** | $SP^* - EEV \geq 0$ | **Giá trị** của tư duy xác suất |
            | **WS** | $\sum_s p_s \cdot Z^*(s)$ | Lợi ích khi **biết trước** kịch bản |
            | **EVPI** | $WS - SP^* \geq 0$ | **Giá trị** thông tin hoàn hảo |
            """
        )

        with st.spinner("⏳ Đang tính EEV, WS, VSS, EVPI..."):
            # EEV: cố định x = x^EV, tối ưu y
            eev_obj, _, _ = _solve_sp(fixed_x=ev_x)
            # WS: kỳ vọng trọng số của lời giải xác định
            ws_per_s = {s: det[s][0] for s in SCENARIOS}
            ws_obj   = sum(PROB[s] * ws_per_s[s] for s in SCENARIOS)

        vss  = sp_obj - eev_obj
        evpi = ws_obj - sp_obj

        # ── 4 metric cards ──
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("SP*",    _vnd(sp_obj),  help="Stochastic Programming tối ưu")
        c2.metric("EEV",    _vnd(eev_obj), help="Expected result of EV solution")
        c3.metric("VSS",    _vnd(vss),
                  delta=f"+{vss:,.1f}",
                  delta_color="normal",
                  help="Value of Stochastic Solution = SP* − EEV")
        c4.metric("EVPI",   _vnd(evpi),
                  delta=f"+{evpi:,.1f}",
                  delta_color="normal",
                  help="Expected Value of Perfect Information = WS − SP*")

        _divider()

        # ── Bảng WS chi tiết ──
        st.markdown("#### Chi tiết Wait-and-See (WS)")
        ws_rows = []
        for s in SCENARIOS:
            ws_rows.append({
                "Kịch bản":   SCENARIO_LABELS[s],
                "Xác suất":   f"{PROB[s]:.0%}",
                "Z*(s)":      _vnd(ws_per_s[s]),
                "p_s × Z*(s)": _vnd(PROB[s] * ws_per_s[s]),
            })
        ws_rows.append({
            "Kịch bản":    "**WS = Σ p_s · Z\\*(s)**",
            "Xác suất":    "—",
            "Z*(s)":       "—",
            "p_s × Z*(s)": _vnd(ws_obj),
        })
        st.dataframe(pd.DataFrame(ws_rows), use_container_width=True, hide_index=True)

        # ── Bảng tóm tắt VSS / EVPI ──
        st.markdown("#### Tóm tắt chỉ số giá trị thông tin")
        summary = pd.DataFrame([
            {"Chỉ số": "SP*  (Stochastic opt.)",     "Giá trị (tỷ VND)": _vnd(sp_obj)},
            {"Chỉ số": "EEV (bỏ qua bất định)",      "Giá trị (tỷ VND)": _vnd(eev_obj)},
            {"Chỉ số": "VSS = SP* − EEV",            "Giá trị (tỷ VND)": _vnd(vss)},
            {"Chỉ số": "WS  (thông tin hoàn hảo)",   "Giá trị (tỷ VND)": _vnd(ws_obj)},
            {"Chỉ số": "EVPI = WS − SP*",            "Giá trị (tỷ VND)": _vnd(evpi)},
        ])
        st.dataframe(summary, use_container_width=True, hide_index=True)

        # ── Diễn giải ──
        st.success(
            f"**VSS = {vss:,.1f} tỷ VND** — "
            "Chính phủ thu được thêm giá trị này bằng cách **mô hình hóa bất định** "
            "thay vì chỉ dùng kịch bản trung bình. VSS dương xác nhận rằng tư duy "
            "xác suất có ý nghĩa thực tiễn trong hoạch định ngân sách."
        )
        st.info(
            f"**EVPI = {evpi:,.1f} tỷ VND** — "
            "Đây là **mức tối đa** Chính phủ nên sẵn sàng trả cho các nghiên cứu dự báo "
            "kinh tế, tình báo thương mại, hoặc hệ thống cảnh báo sớm có khả năng "
            "\"tiết lộ\" kịch bản tương lai. Nếu chi phí thu thập thông tin < EVPI → đáng đầu tư."
        )

        with st.expander("💬 Câu hỏi thảo luận chính sách", expanded=False):
            st.markdown(
                f"""
**b) VSS = {vss:,.1f} nói lên điều gì?**  
VSS dương khẳng định: trong bối cảnh Việt Nam có độ mở thương mại ~180% GDP, bất định
kinh tế toàn cầu **không thể bị bình quân hóa** mà không mất giá trị. Mô hình SP giúp
phân bổ ngân sách linh hoạt hơn, duy trì dự phòng đúng chỗ, và phản ứng nhanh khi kịch
bản thực tế xảy ra.

**c) COVID-19 (2020–2022) & bão Yagi (2024) — bài học "dưới đầu tư" nhân lực số?**  
Cả hai cú sốc cho thấy lao động có kỹ năng số **duy trì năng suất tốt hơn** (làm việc từ
xa, chuyển đổi nghề nhanh) và **phục hồi kinh tế nhanh hơn**. Hệ số β_H = 1,10 trong kịch
bản khủng hoảng (cao hơn tất cả hạng mục khác) phản ánh đúng vai trò "bảo hiểm" này.
Nếu Việt Nam tiếp tục ưu tiên hạ tầng cứng trước nhân lực số, EVPI sẽ tăng — tức rủi ro
từ bất định ngày càng lớn hơn.
                """
            )

    # ─────────────────────────────────────────
    # TAB 4 – ROBUST OPTIMIZATION
    # ─────────────────────────────────────────
    with tab4:
        st.markdown("### 10.5.4 · Robust Optimization — Minimax Regret")
        st.markdown(
            r"""
            Thay vì tối đa hóa **kỳ vọng**, Robust Optimization tối thiểu hóa **hối tiếc xấu nhất**:

            $$\min_{x,\,y}\; \max_{s \in S}\;\underbrace{\bigl[Z^*(s) - Z(x,y,s)\bigr]}_{\text{Regret}(s)}$$

            **Linearization** (thêm biến $\eta$):

            $$\min\;\eta \quad \text{s.t.}\quad \eta \geq Z^*(s) - Z(x,y,s),\;\forall s$$
            """
        )

        with st.spinner("⏳ Đang tính WS và giải Robust..."):
            ws_vals             = {s: det[s][0] for s in SCENARIOS}
            rob_eta, rob_x, rob_y = _solve_robust(ws_vals)

        # Tính regret
        rob_regrets = {}
        for s in SCENARIOS:
            z_s = (sum(BETA[j]*rob_x[j] for j in ITEMS) +
                   sum(BETA_S[(s,j)]*rob_y[(s,j)] for j in ITEMS))
            rob_regrets[s] = ws_vals[s] - z_s

        sp_regrets = {}
        for s in SCENARIOS:
            z_s = (sum(BETA[j]*sp_x[j] for j in ITEMS) +
                   sum(BETA_S[(s,j)]*sp_y[(s,j)] for j in ITEMS))
            sp_regrets[s] = ws_vals[s] - z_s

        st.success(
            f"✅ **Minimax Regret = {rob_eta:,.2f} tỷ VND** — "
            "không kịch bản nào bị thiệt hại quá mức này so với lời giải lý tưởng."
        )

        # ── So sánh first-stage SP vs Robust ──
        st.markdown("#### So sánh phân bổ first-stage: SP vs Robust")
        cmp2 = pd.DataFrame([
            {
                "Phương pháp": "SP (Maximize E[Z])",
                **{ITEM_LABELS[j]: _vnd(sp_x[j]) for j in ITEMS},
                "Tổng": _vnd(sum(sp_x.values())),
            },
            {
                "Phương pháp": "Robust (Minimax Regret)",
                **{ITEM_LABELS[j]: _vnd(rob_x[j]) for j in ITEMS},
                "Tổng": _vnd(sum(rob_x.values())),
            },
        ])
        st.dataframe(cmp2, use_container_width=True, hide_index=True)

        # ── Bảng regret theo kịch bản ──
        st.markdown("#### Regret theo kịch bản: SP vs Robust")
        reg_rows = []
        for s in SCENARIOS:
            reg_rows.append({
                "Kịch bản":      SCENARIO_LABELS[s],
                "WS*(s)":        _vnd(ws_vals[s]),
                "Regret – SP":   _vnd(sp_regrets[s]),
                "Regret – Robust": _vnd(rob_regrets[s]),
            })
        reg_df = pd.DataFrame(reg_rows)
        st.dataframe(reg_df, use_container_width=True, hide_index=True)

        # ── Key metrics ──
        max_sp_regret  = max(sp_regrets.values())
        max_rob_regret = max(rob_regrets.values())

        col1, col2 = st.columns(2)
        col1.metric(
            "Max Regret – SP",
            _vnd(max_sp_regret),
            help="Regret lớn nhất khi dùng lời giải SP",
        )
        col2.metric(
            "Max Regret – Robust",
            _vnd(max_rob_regret),
            delta=f"{max_rob_regret - max_sp_regret:,.1f} vs SP",
            delta_color="inverse",
            help="Regret lớn nhất khi dùng lời giải Robust (thấp hơn = tốt hơn)",
        )

        st.info(
            "💡 **Khi nào nên dùng Robust thay vì SP?**  \n"
            "- Khi **không tin tưởng** vào ước lượng xác suất kịch bản  \n"
            "- Khi **hậu quả kịch bản xấu là thảm khốc** và không thể chấp nhận  \n"
            "- Khi **cần giải thích chính trị** rõ ràng: \"chúng ta đã chuẩn bị cho tình huống tệ nhất\"  \n\n"
            "**Nhược điểm:** Robust có thể hy sinh lợi ích trung bình đáng kể. "
            "Trong thực tế, kết hợp SP (để tối ưu kỳ vọng) với phân tích regret (để kiểm tra "
            "khả năng chịu đựng) cho kết quả chính sách cân bằng nhất."
        )

    # ── Footer kỹ thuật ──
    _divider()
    st.caption(
        "🔧 Solver: **HiGHS** · Mô hình hóa: **Pyomo** · "
        "Cài đặt: `pip install highspy pyomo` · "
        "Tham chiếu: Birge & Louveaux, *Introduction to Stochastic Programming*, 2011."
    )