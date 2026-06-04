"""
modules/bai7.py
Bài 7 – Tối ưu đa mục tiêu Pareto với NSGA-II
Hỗ trợ cả pymoo (nếu cài được) lẫn NSGA-II thuần NumPy (fallback).
"""

# ── Standard imports ──────────────────────────────────────────────────────────
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Optional: pymoo ───────────────────────────────────────────────────────────
try:
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize as pymoo_minimize
    from pymoo.termination import get_termination
    _PYMOO_AVAILABLE = True
except ImportError:
    _PYMOO_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
#  PHẦN 1 – NSGA-II THUẦN NUMPY (fallback & cũng dùng khi chạy trong app)
# ══════════════════════════════════════════════════════════════════════════════

def _eval_pop(pop, beta, e, rho, sig, total_budget):
    """Tính F(N,4) và G(N,12) cho toàn quần thể."""
    N = len(pop)
    F = np.zeros((N, 4))
    G = np.zeros((N, 12))
    min_r = total_budget * 0.03          # mỗi vùng tối thiểu 3% ngân sách
    max_r = total_budget * 0.30          # mỗi vùng tối đa 30%

    for i in range(N):
        X = pop[i].reshape(6, 4)         # (vùng, hạng mục)
        sums_r = X.sum(axis=1)           # tổng theo vùng

        f1 = -(beta * X).sum()           # maximize GDP → minimize -GDP
        f2 = np.abs(sums_r - sums_r.mean()).mean()
        f3 = (e * (X[:, 0] + X[:, 2])).sum()
        f4 = (rho * X[:, 2]).sum() - (sig * X[:, 3]).sum()
        F[i] = [f1, f2, f3, f4]

        total = pop[i].sum()
        G[i, 0]  = total - total_budget          # ≤ budget
        G[i, 1]  = 0.10 * total_budget - total   # ≥ 10% budget
        for r in range(6):
            G[i, 2 + r] = min_r - sums_r[r]      # vùng ≥ min
            G[i, 8 + r] = sums_r[r] - max_r      # vùng ≤ max
    return F, G


def _nondom_sort(F):
    """Fast non-dominated sort – trả về list of fronts."""
    N = len(F)
    dom_cnt = np.zeros(N, dtype=int)
    dom_set = [[] for _ in range(N)]
    fronts = [[]]

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                dom_set[i].append(j)
            elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                dom_cnt[i] += 1
        if dom_cnt[i] == 0:
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        nxt = []
        for i in fronts[k]:
            for j in dom_set[i]:
                dom_cnt[j] -= 1
                if dom_cnt[j] == 0:
                    nxt.append(j)
        fronts.append(nxt)
        k += 1
    return [f for f in fronts if f]


def _crowding(F, front):
    n = len(front)
    if n <= 2:
        return np.full(n, np.inf)
    d = np.zeros(n)
    Ff = F[front]
    for m in range(Ff.shape[1]):
        order = np.argsort(Ff[:, m])
        rng = Ff[order[-1], m] - Ff[order[0], m]
        if rng < 1e-12:
            continue
        d[order[0]] = d[order[-1]] = np.inf
        for k in range(1, n - 1):
            d[order[k]] += (Ff[order[k+1], m] - Ff[order[k-1], m]) / rng
    return d


def _select(fronts, cdmap, N):
    rank = {idx: r for r, f in enumerate(fronts) for idx in f}
    pool = [idx for f in fronts for idx in f]
    sel = []
    for _ in range(N):
        a, b = np.random.choice(pool, 2, replace=False)
        if rank[a] < rank[b]:
            sel.append(a)
        elif rank[b] < rank[a]:
            sel.append(b)
        else:
            sel.append(a if cdmap.get(a, 0) >= cdmap.get(b, 0) else b)
    return sel


def _sbx_pm(parents, xl, xu, eta_c=15, eta_m=20):
    """SBX crossover + polynomial mutation."""
    N, nv = parents.shape
    off = parents.copy()
    pc, pm = 0.9, 1.0 / nv
    idx = np.random.permutation(N)
    for k in range(0, N - 1, 2):
        i1, i2 = idx[k], idx[k+1]
        if np.random.rand() < pc:
            c1, c2 = parents[i1].copy(), parents[i2].copy()
            for j in range(nv):
                if np.random.rand() < 0.5:
                    u = np.random.rand()
                    b = (2*u)**(1/(eta_c+1)) if u <= 0.5 else (1/(2*(1-u)))**(1/(eta_c+1))
                    c1[j] = np.clip(.5*((1+b)*parents[i1,j]+(1-b)*parents[i2,j]), xl[j], xu[j])
                    c2[j] = np.clip(.5*((1-b)*parents[i1,j]+(1+b)*parents[i2,j]), xl[j], xu[j])
            off[i1], off[i2] = c1, c2
    for i in range(N):
        for j in range(nv):
            if np.random.rand() < pm:
                u = np.random.rand()
                dq = xu[j] - xl[j]
                d = (2*u)**(1/(eta_m+1))-1 if u < 0.5 else 1-(2*(1-u))**(1/(eta_m+1))
                off[i, j] = np.clip(off[i, j] + d * dq, xl[j], xu[j])
    return off


def _repair(pop, total_budget):
    pop = np.clip(pop, 0, None)
    s = pop.sum(axis=1, keepdims=True)
    s[s == 0] = 1
    return pop / s * total_budget


def _nsga2_numpy(beta, e, rho, sig, total_budget,
                 pop_size, n_gen, seed, progress_bar=None):
    """NSGA-II thuần NumPy. Trả về (F_pareto, X_pareto)."""
    np.random.seed(seed)
    nv = 24
    xl, xu = np.zeros(nv), np.full(nv, total_budget / 6)
    pop = _repair(np.random.uniform(0, xu, (pop_size, nv)), total_budget)

    for gen in range(n_gen):
        F, G = _eval_pop(pop, beta, e, rho, sig, total_budget)
        pen = np.maximum(G, 0).sum(axis=1, keepdims=True)
        Fp = F + pen * 5e3

        fronts = _nondom_sort(Fp)
        cdmap = {}
        for fr in fronts:
            cd = _crowding(Fp, fr)
            for k, idx in enumerate(fr):
                cdmap[idx] = cd[k]

        parents = pop[_select(fronts, cdmap, pop_size)]
        offspring = _repair(_sbx_pm(parents, xl, xu), total_budget)

        comb = np.vstack([pop, offspring])
        Fc, Gc = _eval_pop(comb, beta, e, rho, sig, total_budget)
        Fpc = Fc + np.maximum(Gc, 0).sum(axis=1, keepdims=True) * 5e3

        fc2 = _nondom_sort(Fpc)
        new_idx = []
        for fr in fc2:
            if len(new_idx) + len(fr) <= pop_size:
                new_idx.extend(fr)
            else:
                rem = pop_size - len(new_idx)
                cd = _crowding(Fpc, fr)
                new_idx.extend([fr[k] for k in np.argsort(-cd)[:rem]])
                break
        pop = comb[new_idx]

        if progress_bar is not None and (gen % max(1, n_gen//50) == 0 or gen == n_gen-1):
            pct = int((gen + 1) / n_gen * 100)
            progress_bar.progress(pct, text=f"NSGA-II thế hệ {gen+1}/{n_gen}…")

    F_final, _ = _eval_pop(pop, beta, e, rho, sig, total_budget)
    fronts_final = _nondom_sort(F_final)
    pidx = fronts_final[0]
    return F_final[pidx], pop[pidx]


# ══════════════════════════════════════════════════════════════════════════════
#  PHẦN 2 – pymoo wrapper (nếu có)
# ══════════════════════════════════════════════════════════════════════════════

def _nsga2_pymoo(beta, e, rho, sig, total_budget, pop_size, n_gen, seed):
    """Chạy NSGA-II bằng pymoo. Chỉ gọi khi _PYMOO_AVAILABLE == True."""

    class VNProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(
                n_var=24, n_obj=4, n_ieq_constr=14,
                xl=np.zeros(24), xu=np.full(24, total_budget / 6),
            )
            self.beta = beta
            self.e = e
            self.rho = rho
            self.sig = sig

        def _evaluate(self, x, out, *args, **kwargs):
            X = x.reshape(6, 4)
            sums_r = X.sum(axis=1)
            min_r = total_budget * 0.03
            max_r = total_budget * 0.30
            f1 = -(self.beta * X).sum()
            f2 = np.abs(sums_r - sums_r.mean()).mean()
            f3 = (self.e * (X[:, 0] + X[:, 2])).sum()
            f4 = (self.rho * X[:, 2]).sum() - (self.sig * X[:, 3]).sum()
            out["F"] = [f1, f2, f3, f4]
            total = x.sum()
            g = [
                total - total_budget,
                0.10 * total_budget - total,
            ]
            for r in range(6):
                g.append(min_r - sums_r[r])
            for r in range(6):
                g.append(sums_r[r] - max_r)
            out["G"] = g

    res = pymoo_minimize(
        VNProblem(),
        NSGA2(pop_size=pop_size),
        get_termination("n_gen", n_gen),
        seed=seed,
        verbose=False,
    )
    F_pareto = res.F.copy()
    X_pareto = res.X.copy()
    return F_pareto, X_pareto


# ══════════════════════════════════════════════════════════════════════════════
#  PHẦN 3 – TOPSIS helper
# ══════════════════════════════════════════════════════════════════════════════

def _topsis(F, weights, benefit):
    """TOPSIS trên tập Pareto. Trả về (best_idx, C_star_array)."""
    R = F / np.sqrt((F**2).sum(axis=0) + 1e-12)
    V = R * weights
    A_pos = np.where(benefit, V.max(axis=0), V.min(axis=0))
    A_neg = np.where(benefit, V.min(axis=0), V.max(axis=0))
    S_pos = np.sqrt(((V - A_pos)**2).sum(axis=1))
    S_neg = np.sqrt(((V - A_neg)**2).sum(axis=1))
    C = S_neg / (S_pos + S_neg + 1e-12)
    return int(np.argmax(C)), C


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run(df_macro, df_sectors, df_regions):

    st.header("🧬 Bài 7: Tối ưu Đa Mục tiêu Pareto với NSGA-II")
    st.markdown(
        "Giải bài toán **4 mục tiêu xung đột** cho phân bổ ngân sách đầu tư số "
        "tại 6 vùng kinh tế Việt Nam. Thuật toán **NSGA-II** dựng tập nghiệm "
        "Pareto; phương pháp **TOPSIS** chọn nghiệm thỏa hiệp."
    )

    # ── Cảnh báo pymoo ────────────────────────────────────────────────────────
    if _PYMOO_AVAILABLE:
        st.success("✅ **pymoo** đã được cài đặt – có thể dùng backend pymoo.")
    else:
        st.info(
            "ℹ️ **pymoo** chưa được cài đặt. Ứng dụng sẽ dùng NSGA-II thuần "
            "NumPy (kết quả tương đương). Để dùng pymoo, chạy:\n"
            "```bash\npip install pymoo\n```"
        )

    # ── Tham số dữ liệu ───────────────────────────────────────────────────────
    REGION_NAMES = [
        "Trung du & MN phía Bắc",
        "Đồng bằng sông Hồng",
        "Bắc Trung Bộ & DHMT",
        "Tây Nguyên",
        "Đông Nam Bộ",
        "ĐB sông Cửu Long",
    ]
    INVEST_LABELS = ["Hạ tầng (I)", "Dữ liệu (D)", "AI (AI)", "Nhân lực (H)"]
    OBJ_LABELS    = ["f₁ GDP ↑", "f₂ Bất bình đẳng ↓", "f₃ CO₂ ↓", "f₄ Rủi ro DL ↓"]

    # Đọc df_regions nếu có cột phù hợp, không thì dùng mặc định
    _e_def   = np.array([0.42, 0.55, 0.48, 0.32, 0.62, 0.38])
    _rho_def = np.array([0.18, 0.45, 0.28, 0.12, 0.52, 0.22])
    _sig_def = np.array([0.32, 0.28, 0.30, 0.35, 0.25, 0.30])

    def _try_col(df, col, default):
        if df is not None and col in df.columns and len(df) >= 6:
            return df[col].values[:6].astype(float)
        return default

    e_arr   = _try_col(df_regions, "emission_intensity",  _e_def)
    rho_arr = _try_col(df_regions, "data_risk_coef",      _rho_def)
    sig_arr = _try_col(df_regions, "risk_reduction_coef", _sig_def)

    # Beta: hệ số tác động cận biên (6×4)
    beta = np.array([
        [0.12, 0.15, 0.20, 0.10],
        [0.18, 0.22, 0.35, 0.16],
        [0.14, 0.17, 0.25, 0.13],
        [0.10, 0.12, 0.18, 0.09],
        [0.20, 0.25, 0.38, 0.18],
        [0.11, 0.14, 0.22, 0.10],
    ])

    TOTAL_BUDGET = 72_000  # tỷ VND

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⚙️ Tham số & Mô hình",
        "🚀 Chạy NSGA-II",
        "📊 Phân tích Pareto",
        "🎯 Nghiệm Thỏa hiệp",
        "💬 Thảo luận Chính sách",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 – Tham số & Mô hình
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.subheader("📐 Mô hình toán học")
        st.markdown(r"""
| Hàm mục tiêu | Công thức | Chiều |
|---|---|:---:|
| **f₁** Tăng trưởng GDP | $\max \sum_{r,j} \beta_{j,r}\, x_{j,r}$ | ↑ |
| **f₂** Bất bình đẳng vùng | $\min \; \text{MAD}\!\left(\textstyle\sum_j x_{j,r}\right)$ | ↓ |
| **f₃** Phát thải CO₂ | $\min \sum_r e_r\,(x_{I,r}+x_{AI,r})$ | ↓ |
| **f₄** Rủi ro an ninh DL | $\min \sum_r \rho_r x_{AI,r} - \sum_r \sigma_r x_{H,r}$ | ↓ |

**Biến quyết định:** $x_{j,r} \geq 0$, $j\in\{I,D,AI,H\}$, $r\in\{1,\ldots,6\}$ → **24 biến**

**Ràng buộc:** $\sum x \leq B$; mỗi vùng ≥ 3 % B và ≤ 30 % B
        """)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Ma trận β (tác động cận biên)")
            st.dataframe(
                pd.DataFrame(beta, index=REGION_NAMES, columns=INVEST_LABELS)
                .style.format("{:.2f}").background_gradient(cmap="Blues"),
                use_container_width=True,
            )
        with c2:
            st.markdown("#### Tham số môi trường & an ninh")
            pdf = pd.DataFrame({
                "Vùng": REGION_NAMES,
                "eᵣ CO₂/tỷ": e_arr,
                "ρᵣ rủi ro AI": rho_arr,
                "σᵣ giảm rủi ro/H": sig_arr,
            }).set_index("Vùng")
            st.dataframe(
                pdf.style.format("{:.2f}").background_gradient(cmap="Oranges"),
                use_container_width=True,
            )

        st.subheader("🔧 Cơ chế NSGA-II")
        st.markdown("""
1. **Non-dominated sorting** – phân loại Pareto front F₁ ≻ F₂ ≻ …
2. **Crowding distance** – đo mật độ nghiệm để bảo đảm đa dạng tập Pareto
3. **Tournament selection** – ưu tiên rank thấp; tie-break bằng crowding distance
4. **SBX crossover + Polynomial mutation** – tạo offspring mới trong không gian liên tục
5. **Elitism** – merge bố mẹ + offspring, giữ lại *pop_size* cá thể tốt nhất
        """)

        if _PYMOO_AVAILABLE:
            with st.expander("📦 Cấu trúc lớp pymoo (tham khảo)"):
                st.code(
                    """from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

class VietnamDigitalProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=24, n_obj=4,
                         n_ieq_constr=12,
                         xl=np.zeros(24),
                         xu=np.full(24, 12000))

    def _evaluate(self, x, out, *args, **kwargs):
        X = x.reshape(6, 4)
        sums = X.sum(axis=1)
        out['F'] = [
            -(beta * X).sum(),                          # f1: -GDP
            np.abs(sums - sums.mean()).mean(),           # f2: MAD
            (e * (X[:,0] + X[:,2])).sum(),               # f3: CO2
            (rho * X[:,2]).sum() - (sig * X[:,3]).sum()  # f4: risk
        ]
        out['G'] = [...]  # 12 constraints""",
                    language="python",
                )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 – Chạy NSGA-II
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("🚀 Câu 7.4.1 – Thiết lập & Chạy NSGA-II")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            pop_size = st.slider("Population size", 40, 200, 80, step=10,
                                 help="Số cá thể trong mỗi thế hệ")
        with col_b:
            n_gen = st.slider("Số thế hệ (n_gen)", 50, 300, 120, step=10,
                              help="Bài yêu cầu 200; giảm xuống để chạy nhanh hơn")
        with col_c:
            seed = int(st.number_input("Random seed", 0, 9999, 42, step=1))

        backend = "numpy"
        if _PYMOO_AVAILABLE:
            backend = st.radio("Backend", ["numpy (built-in)", "pymoo"],
                               horizontal=True)
            backend = "pymoo" if "pymoo" in backend else "numpy"

        st.caption(
            f"Ước tính ~{pop_size * n_gen * 2 // 1000}k lần tính hàm mục tiêu. "
            f"Backend: **{backend}**"
        )

        if st.button("▶ Bắt đầu tối ưu hóa", type="primary", key="btn_run"):
            pb = st.progress(0, text="Khởi tạo…")
            try:
                if backend == "pymoo":
                    with st.spinner("pymoo đang tối ưu…"):
                        pF, pX = _nsga2_pymoo(
                            beta, e_arr, rho_arr, sig_arr,
                            TOTAL_BUDGET, pop_size, n_gen, seed,
                        )
                    pb.progress(100, text="Hoàn thành (pymoo)!")
                else:
                    pF, pX = _nsga2_numpy(
                        beta, e_arr, rho_arr, sig_arr,
                        TOTAL_BUDGET, pop_size, n_gen, seed,
                        progress_bar=pb,
                    )
                    pb.empty()

                # Chuyển f1 về GDP dương để hiển thị
                pF_disp = pF.copy()
                pF_disp[:, 0] = -pF_disp[:, 0]

                st.session_state["pF"]      = pF
                st.session_state["pF_disp"] = pF_disp
                st.session_state["pX"]      = pX

                st.success(
                    f"✅ Tìm được **{len(pF)} nghiệm Pareto** "
                    f"sau {n_gen} thế hệ (backend: {backend})."
                )

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("GDP tốt nhất (tỷ VND)", f"{pF_disp[:,0].max():,.0f}")
                m2.metric("Bất bình đẳng thấp nhất", f"{pF_disp[:,1].min():,.1f}")
                m3.metric("CO₂ thấp nhất", f"{pF_disp[:,2].min():,.1f}")
                m4.metric("Rủi ro DL thấp nhất", f"{pF_disp[:,3].min():,.1f}")

            except Exception as ex:
                pb.empty()
                st.error(f"❌ Lỗi khi tối ưu: {ex}")

        if "pF_disp" in st.session_state:
            pF_disp = st.session_state["pF_disp"]
            st.markdown(f"#### Tập nghiệm Pareto ({len(pF_disp)} nghiệm)")
            df_show = pd.DataFrame(pF_disp, columns=OBJ_LABELS)
            df_show.index = [f"P{i+1}" for i in range(len(pF_disp))]
            st.dataframe(
                df_show.style.format("{:.2f}")
                       .background_gradient(cmap="YlOrRd"),
                use_container_width=True,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 – Phân tích Pareto
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("📊 Câu 7.4.2 – Trực quan hóa Pareto Front")

        if "pF_disp" not in st.session_state:
            st.warning("⬅ Hãy chạy NSGA-II ở tab **Chạy NSGA-II** trước.")
            st.stop()

        pF_disp = st.session_state["pF_disp"]
        df_p = pd.DataFrame(pF_disp, columns=["GDP (tỷ)", "Bất bình đẳng",
                                                "CO₂", "Rủi ro DL"])
        df_p["Nghiệm"] = [f"P{i+1}" for i in range(len(pF_disp))]

        # ── Scatter 3D ───────────────────────────────────────────────────────
        st.markdown("#### 🌐 Scatter 3D: f₁ GDP – f₂ Bất bình đẳng – f₃ CO₂")
        fig3d = px.scatter_3d(
            df_p,
            x="GDP (tỷ)", y="Bất bình đẳng", z="CO₂",
            color="Rủi ro DL", color_continuous_scale="RdYlGn_r",
            hover_name="Nghiệm",
            title="Pareto Front 3D (màu = Rủi ro an ninh dữ liệu f₄)",
            labels={"GDP (tỷ)": "f₁: GDP", "Bất bình đẳng": "f₂",
                    "CO₂": "f₃: CO₂"},
        )
        fig3d.update_traces(marker=dict(size=5, opacity=0.85))
        fig3d.update_layout(height=560)
        st.plotly_chart(fig3d, use_container_width=True)

        # ── Parallel Coordinates ─────────────────────────────────────────────
        st.markdown("#### 🎻 Parallel Coordinates – cả 4 mục tiêu")
        fig_pc = go.Figure(go.Parcoords(
            line=dict(
                color=df_p["GDP (tỷ)"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="GDP (tỷ)", len=0.75),
            ),
            dimensions=[
                dict(label="f₁ GDP ↑",       values=df_p["GDP (tỷ)"]),
                dict(label="f₂ Bất bình đẳng ↓", values=df_p["Bất bình đẳng"]),
                dict(label="f₃ CO₂ ↓",       values=df_p["CO₂"]),
                dict(label="f₄ Rủi ro DL ↓", values=df_p["Rủi ro DL"]),
            ],
        ))
        fig_pc.update_layout(
            title="Parallel Coordinates – tập nghiệm Pareto (kéo trục để lọc)",
            height=460,
        )
        st.plotly_chart(fig_pc, use_container_width=True)
        st.caption("💡 Kéo chọn vùng trên mỗi trục để highlight nghiệm quan tâm.")

        # ── Scatter matrix ───────────────────────────────────────────────────
        st.markdown("#### 🔢 Ma trận scatter – phân tích trade-off từng cặp")
        fig_sm = px.scatter_matrix(
            df_p,
            dimensions=["GDP (tỷ)", "Bất bình đẳng", "CO₂", "Rủi ro DL"],
            color="GDP (tỷ)", color_continuous_scale="Blues",
            title="Scatter matrix 4 mục tiêu",
        )
        fig_sm.update_traces(marker=dict(size=3, opacity=0.6))
        fig_sm.update_layout(height=540)
        st.plotly_chart(fig_sm, use_container_width=True)

        # ── Histogram phân phối ───────────────────────────────────────────────
        st.markdown("#### 📈 Phân phối từng mục tiêu trong tập Pareto")
        fig_h = go.Figure()
        for col, color in zip(df_p.columns[:4],
                               ["#2196F3","#FF9800","#4CAF50","#F44336"]):
            fig_h.add_trace(go.Histogram(x=df_p[col], name=col,
                                          opacity=0.7, marker_color=color,
                                          nbinsx=20))
        fig_h.update_layout(barmode="overlay",
                             title="Histogram phân phối mục tiêu", height=360)
        st.plotly_chart(fig_h, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 – Nghiệm Thỏa hiệp
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("🎯 Câu 7.4.3 & 7.4.4 – Nghiệm Thỏa hiệp TOPSIS")

        if "pF_disp" not in st.session_state:
            st.warning("⬅ Hãy chạy NSGA-II trước.")
            st.stop()

        pF_disp = st.session_state["pF_disp"]
        pX      = st.session_state["pX"]

        # ── Trọng số TOPSIS ──────────────────────────────────────────────────
        st.markdown("#### ⚙️ Trọng số ưu tiên chính sách")
        c1, c2, c3, c4 = st.columns(4)
        w1 = c1.slider("w₁ Tăng trưởng", 0.0, 1.0, 0.40, 0.05)
        w2 = c2.slider("w₂ Bao trùm",    0.0, 1.0, 0.25, 0.05)
        w3 = c3.slider("w₃ Môi trường",  0.0, 1.0, 0.20, 0.05)
        w4 = c4.slider("w₄ An ninh DL",  0.0, 1.0, 0.15, 0.05)
        w_raw = np.array([w1, w2, w3, w4])
        if abs(w_raw.sum() - 1.0) > 0.01:
            st.caption(f"⚠️ Tổng = {w_raw.sum():.2f} → tự chuẩn hóa.")
        w_norm = w_raw / (w_raw.sum() + 1e-12)

        benefit = np.array([True, False, False, False])
        best_idx, C_star = _topsis(pF_disp, w_norm, benefit)

        # ── Kết quả nghiệm tốt nhất ──────────────────────────────────────────
        st.markdown("---")
        st.markdown(f"#### 🏆 Nghiệm thỏa hiệp: **P{best_idx+1}**  &nbsp; C\\* = `{C_star[best_idx]:.4f}`")
        bf = pF_disp[best_idx]
        bX = pX[best_idx].reshape(6, 4)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("C* TOPSIS",       f"{C_star[best_idx]:.4f}")
        m2.metric("f₁ GDP (tỷ)",      f"{bf[0]:,.0f}")
        m3.metric("f₂ Bất bình đẳng", f"{bf[1]:,.1f}")
        m4.metric("f₃ CO₂",           f"{bf[2]:,.1f}")
        m5.metric("f₄ Rủi ro DL",     f"{bf[3]:,.1f}")

        # ── Phân bổ ngân sách ────────────────────────────────────────────────
        st.markdown("#### 💰 Phân bổ ngân sách tối ưu (tỷ VND)")
        alloc_df = pd.DataFrame(bX, index=REGION_NAMES, columns=INVEST_LABELS)
        alloc_df["Tổng"] = alloc_df.sum(axis=1)
        alloc_df.loc["TỔNG"] = alloc_df.sum()
        st.dataframe(
            alloc_df.style.format("{:,.0f}")
                    .background_gradient(cmap="Blues", subset=INVEST_LABELS)
                    .background_gradient(cmap="Greens", subset=["Tổng"]),
            use_container_width=True,
        )

        fig_alloc = px.bar(
            alloc_df.iloc[:-1].reset_index().rename(columns={"index": "Vùng"}),
            x="Vùng", y=INVEST_LABELS, barmode="stack",
            color_discrete_sequence=["#2196F3","#FF9800","#9C27B0","#4CAF50"],
            title="Cơ cấu đầu tư theo vùng – Nghiệm thỏa hiệp TOPSIS",
        )
        fig_alloc.update_layout(height=400, xaxis_tickangle=-20)
        st.plotly_chart(fig_alloc, use_container_width=True)

        # ── Câu 7.4.4 – Chi phí cơ hội ───────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 💸 Câu 7.4.4 – Chi phí cơ hội")
        gdp_best  = int(np.argmax(pF_disp[:, 0]))
        inc_best  = int(np.argmin(pF_disp[:, 1]))
        env_best  = int(np.argmin(pF_disp[:, 2]))
        sec_best  = int(np.argmin(pF_disp[:, 3]))

        def _delta(v, ref):
            return (v - ref) / (abs(ref) + 1e-9) * 100

        rows = []
        for label, idx in [
            ("GDP cao nhất", gdp_best),
            ("Bao trùm tốt nhất", inc_best),
            ("Môi trường tốt nhất", env_best),
            ("An ninh tốt nhất", sec_best),
        ]:
            row = {"Kịch bản": label}
            for k, ol in enumerate(OBJ_LABELS):
                row[ol] = pF_disp[idx, k]
                row[f"Δ vs Thỏa hiệp (%)"] = None
            rows.append(row)

        # So sánh GDP cao nhất với thỏa hiệp
        comp = {
            "Chỉ tiêu": OBJ_LABELS,
            "Nghiệm Thỏa hiệp": list(bf),
            "GDP cao nhất": list(pF_disp[gdp_best]),
            "Δ (%)": [_delta(pF_disp[gdp_best, k], bf[k]) for k in range(4)],
        }
        comp_df = pd.DataFrame(comp).set_index("Chỉ tiêu")
        st.markdown(
            "So sánh **GDP cao nhất** vs **Thỏa hiệp** — "
            "Δ dương (f₁) = GDP tốt hơn; Δ âm (f₂–f₄) = chi phí phải trả:"
        )
        st.dataframe(
            comp_df.style.format({
                "Nghiệm Thỏa hiệp": "{:.2f}",
                "GDP cao nhất": "{:.2f}",
                "Δ (%)": "{:+.1f}%",
            }).background_gradient(subset=["Δ (%)"], cmap="RdYlGn"),
            use_container_width=True,
        )

        # Radar 3 nghiệm
        def _norm_radar(fvec, fall):
            n = np.zeros(4)
            for k in range(4):
                lo, hi = fall[:, k].min(), fall[:, k].max()
                n[k] = (fvec[k] - lo) / (hi - lo + 1e-9)
            n[1:] = 1 - n[1:]   # đảo f2-f4: thấp = tốt
            return n

        cats = ["GDP ↑", "Bao trùm ↑", "Môi trường ↑", "An ninh ↑"]
        cats_c = cats + [cats[0]]
        fig_r = go.Figure()
        for idx2, label, color in [
            (best_idx, "Thỏa hiệp (TOPSIS)", "#2196F3"),
            (gdp_best,  "GDP cao nhất",        "#F44336"),
            (inc_best,  "Bao trùm tốt nhất",   "#4CAF50"),
        ]:
            rv = list(_norm_radar(pF_disp[idx2], pF_disp)) + \
                 [_norm_radar(pF_disp[idx2], pF_disp)[0]]
            fig_r.add_trace(go.Scatterpolar(
                r=rv, theta=cats_c, fill="toself",
                name=label, line_color=color, opacity=0.7,
            ))
        fig_r.update_layout(
            polar=dict(radialaxis=dict(range=[0, 1])),
            title="Radar so sánh 3 nghiệm đặc trưng (cao = tốt hơn)",
            height=460,
        )
        st.plotly_chart(fig_r, use_container_width=True)

        # C* toàn tập
        st.markdown("#### 📉 Hệ số C* trên toàn tập Pareto")
        df_c = pd.DataFrame({"Nghiệm": range(len(C_star)), "C*": C_star,
                              "Loại": "Khác"})
        df_c.loc[best_idx, "Loại"] = "🏆 Thỏa hiệp"
        df_c.loc[gdp_best,  "Loại"] = "📈 GDP cao nhất"
        fig_c = px.scatter(df_c, x="Nghiệm", y="C*", color="Loại",
                           color_discrete_map={
                               "Khác": "#BBDEFB",
                               "🏆 Thỏa hiệp": "#F44336",
                               "📈 GDP cao nhất": "#FF9800",
                           },
                           size=[5 if t == "Khác" else 14 for t in df_c["Loại"]],
                           title="Phân phối C* – TOPSIS trên tập Pareto")
        fig_c.update_layout(height=340)
        st.plotly_chart(fig_c, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 – Thảo luận Chính sách
    # ══════════════════════════════════════════════════════════════════════════
    with tab5:
        st.subheader("💬 Câu 7.5 – Thảo luận Chính sách")
        st.markdown("""
---
#### a) Đánh đổi Tăng trưởng – Bao trùm trên đường biên Pareto

Quan sát Scatter 3D và Parallel Coordinates cho thấy trade-off **f₁ ↑ vs f₂ ↑** (GDP tốt hơn
đi kèm bất bình đẳng cao hơn) thường **rõ ràng**: các nghiệm GDP cao nhất tập trung ngân
sách vào Vùng 5 (Đông Nam Bộ) và Vùng 2 (ĐB sông Hồng) – hai cực tăng trưởng có hệ số
β cao – trong khi các vùng khó khăn hơn nhận ít hơn.

Điều này phản ánh **cấu trúc nhị nguyên** của kinh tế Việt Nam: nếu tối đa hóa GDP thuần
túy, nguồn lực sẽ chảy về hai trung tâm công nghiệp-FDI, làm trầm trọng thêm khoảng cách
số giữa các vùng.

---
#### b) Trọng số (0.40; 0.25; 0.20; 0.15) & ưu tiên quốc gia

- Bộ trọng số mặc định phản ánh *ưu tiên tăng trưởng nhanh* theo Đại hội XIII ("phát triển
  nhanh và bền vững").
- **Cam kết COP26 (Net Zero 2050)**: nên tăng **w₃ → 0.30** và giảm w₁ xuống ~0.25.
- **Quyết định 127/QĐ-TTg (AI quốc gia)**: cần tăng **w₄ → 0.20** (an ninh dữ liệu là
  yếu tố sống còn khi triển khai AI quy mô quốc gia).
- Gợi ý bộ trọng số *xanh & số*: **(0.25 – 0.25 – 0.30 – 0.20)**.

---
#### c) NSGA-II vs LP đơn mục tiêu – Có thay thế quyết định chính trị không?

| | LP đơn mục tiêu | NSGA-II đa mục tiêu |
|---|---|---|
| Kết quả | **1 nghiệm tối ưu** duy nhất | **Tập nghiệm Pareto** |
| Xử lý xung đột | Hóa về 1 hàm (trọng số tiên nghiệm) | Giữ nguyên bản chất đánh đổi |
| Vai trò nhà hoạch định | Chọn trọng số *trước* khi tính | Chọn nghiệm *sau* khi thấy frontier |
| Thông tin cung cấp | Hạn chế | Phong phú – toàn bộ trade-off space |

NSGA-II **không thay thế** quyết định chính trị — nó *hỗ trợ* bằng cách cung cấp bản đồ
đầy đủ các đánh đổi khả thi. Lựa chọn cuối cùng (ưu tiên tăng trưởng hay công bằng hay
môi trường) vẫn là **quá trình thảo luận chính trị – xã hội – thể chế**, như luận điểm
Mục 8.2 bài báo nguồn nhấn mạnh.
        """)

        if "pF_disp" in st.session_state:
            st.markdown("---")
            st.markdown("#### 🔍 Bộ lọc nghiệm tương tác")
            pFd = st.session_state["pF_disp"]
            g_min = float(st.slider(
                "GDP tối thiểu (tỷ VND)",
                float(pFd[:, 0].min()), float(pFd[:, 0].max()),
                float(np.percentile(pFd[:, 0], 40)),
                step=float((pFd[:, 0].max() - pFd[:, 0].min()) / 50),
            ))
            mask = pFd[:, 0] >= g_min
            n_f = mask.sum()
            st.info(
                f"**{n_f}** nghiệm thỏa GDP ≥ {g_min:,.0f} tỷ "
                f"({n_f / len(pFd) * 100:.0f}% tập Pareto)."
            )
            if n_f > 0:
                df_filt = pd.DataFrame(pFd[mask], columns=OBJ_LABELS)
                st.dataframe(
                    df_filt.describe().style.format("{:.2f}"),
                    use_container_width=True,
                )