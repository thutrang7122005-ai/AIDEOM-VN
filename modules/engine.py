"""
engine.py – AIDEOM-VN Central Engine
Tích hợp kết quả từ M1–M5 và chuẩn bị dữ liệu cho Dashboard M6.

Cách dùng:
    from engine import AIDEOM_Engine
    engine = AIDEOM_Engine(df_macro, df_sectors, df_regions)
    results = engine.run_all()
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1. CẤU TRÚC DỮ LIỆU KẾT QUẢ
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class M1_MacroResult:
    """
    Kết quả từ Module 1: Dự báo Macro – Cobb-Douglas mở rộng.

    Attributes
    ----------
    years         : Mảng năm 2020–2025.
    Y_actual      : GDP thực tế (nghìn tỷ VND).
    A_t           : Chuỗi TFP theo năm (Solow residual).
    A_mean        : TFP trung bình giai đoạn.
    Y_forecast    : GDP dự báo với A_mean.
    mape          : Mean Absolute Percentage Error (%).
    growth_decomp : % đóng góp tăng trưởng từng nhân tố (log-linear).
    Y_2030        : GDP dự báo năm 2030 (nghìn tỷ VND).
    params        : Tham số Cobb-Douglas {alpha, beta, gamma, delta, theta}.
    """
    years: np.ndarray
    Y_actual: np.ndarray
    A_t: np.ndarray
    A_mean: float
    Y_forecast: np.ndarray
    mape: float
    growth_decomp: Dict[str, float]
    Y_2030: float
    params: Dict[str, float]


@dataclass
class M2_DigitalizationResult:
    """
    Kết quả từ Module 2: Đánh giá số hóa – TOPSIS & Entropy.

    Attributes
    ----------
    region_names           : Tên 6 vùng kinh tế.
    topsis_scores_expert   : Điểm C* theo trọng số chuyên gia.
    topsis_scores_entropy  : Điểm C* theo trọng số Entropy.
    rankings_expert        : Thứ hạng theo chuyên gia (1 = tốt nhất).
    rankings_entropy       : Thứ hạng theo Entropy.
    top_region             : Vùng dẫn đầu (theo chuyên gia).
    entropy_weights        : Vector trọng số Entropy tính được.
    """
    region_names: List[str]
    topsis_scores_expert: np.ndarray
    topsis_scores_entropy: np.ndarray
    rankings_expert: np.ndarray
    rankings_entropy: np.ndarray
    top_region: str
    entropy_weights: np.ndarray


@dataclass
class M3_AllocationResult:
    """
    Kết quả từ Module 3: Tối ưu phân bổ ngân sách – LP.

    Attributes
    ----------
    region_names        : Tên 6 vùng kinh tế.
    dY_optimal          : GRDP tăng thêm tối ưu theo vùng (nghìn tỷ VND).
    budget_used         : Ngân sách thực sự sử dụng (nghìn tỷ VND).
    Z_star              : Giá trị hàm mục tiêu tối ưu.
    allocation_weights  : Vector trọng số vùng w_r.
    cost_coefficients   : Vector hệ số chi phí c_r.
    intertemporal_K     : Quỹ đạo vốn 2026–2035 (tuỳ chọn).
    intertemporal_GDP   : Quỹ đạo GDP liên kỳ (tuỳ chọn).
    """
    region_names: List[str]
    dY_optimal: np.ndarray
    budget_used: float
    Z_star: float
    allocation_weights: np.ndarray
    cost_coefficients: np.ndarray
    intertemporal_K: Optional[np.ndarray] = None
    intertemporal_GDP: Optional[np.ndarray] = None


@dataclass
class M4_LaborResult:
    """
    Kết quả từ Module 4: Mô phỏng thị trường lao động.

    Attributes
    ----------
    sector_names     : Tên 8 ngành kinh tế.
    baseline_labor   : Lao động ban đầu (triệu người).
    displaced_labor  : Lao động bị dịch chuyển do tự động hoá.
    reskilled_labor  : Lao động tái đào tạo thành công.
    new_jobs         : Việc làm mới từ AI/số hoá.
    net_impact       : Tác động ròng (new_jobs - displaced + reskilled).
    total_displaced  : Tổng lao động dịch chuyển (triệu người).
    total_new_jobs   : Tổng việc làm mới (triệu người).
    """
    sector_names: List[str]
    baseline_labor: np.ndarray
    displaced_labor: np.ndarray
    reskilled_labor: np.ndarray
    new_jobs: np.ndarray
    net_impact: np.ndarray
    total_displaced: float
    total_new_jobs: float


@dataclass
class M5_RiskResult:
    """
    Kết quả từ Module 5: Đánh giá rủi ro đa mục tiêu (Pareto + Stochastic).

    Attributes
    ----------
    pareto_gdp          : GDP kỳ vọng trên Pareto front (tỷ USD).
    pareto_inequality   : Gini/bất bình đẳng trên Pareto front.
    pareto_emissions    : Phát thải CO₂ trên Pareto front.
    stochastic_z_mean   : E[Z] từ bài toán 2 giai đoạn ngẫu nhiên.
    stochastic_z_std    : Độ lệch chuẩn của Z.
    scenario_probs      : Xác suất 3 kịch bản [Thấp, TB, Cao].
    var_95              : Value at Risk 95% – worst-case 5%.
    """
    pareto_gdp: np.ndarray
    pareto_inequality: np.ndarray
    pareto_emissions: np.ndarray
    stochastic_z_mean: float
    stochastic_z_std: float
    scenario_probs: np.ndarray
    var_95: float


@dataclass
class ScenarioResult:
    """
    Kết quả một kịch bản chính sách S1–S5.

    Attributes
    ----------
    name                : Mã kịch bản ("S1"–"S5").
    label               : Tên đầy đủ kịch bản.
    allocation          : Vector phân bổ [w_K, w_D, w_AI, w_H].
    gdp_trajectory      : GDP 2026–2035 (nghìn tỷ VND), 10 phần tử.
    gdp_2030            : GDP năm 2030 (index 4 trong trajectory).
    gdp_2035            : GDP năm 2035 (index 9 trong trajectory).
    growth_rate_avg     : Tốc độ tăng trưởng GDP trung bình năm.
    risk_score          : Điểm rủi ro tổng hợp trong [0, 1].
    inequality_delta    : Thay đổi Gini so với baseline.
    labor_displacement  : Tỷ lệ lao động bị dịch chuyển.
    description         : Mô tả ngắn chiến lược kịch bản.
    """
    name: str
    label: str
    allocation: np.ndarray
    gdp_trajectory: np.ndarray
    gdp_2030: float
    gdp_2035: float
    growth_rate_avg: float
    risk_score: float
    inequality_delta: float
    labor_displacement: float
    description: str


@dataclass
class AIDEOM_Results:
    """
    Container tổng hợp toàn bộ kết quả hệ thống AIDEOM-VN.

    Attributes
    ----------
    m1            : Kết quả Module 1 – Macro Forecast.
    m2            : Kết quả Module 2 – Digitalization Ranking.
    m3            : Kết quả Module 3 – Optimal Allocation.
    m4            : Kết quả Module 4 – Labor Simulation.
    m5            : Kết quả Module 5 – Risk Assessment.
    scenarios     : Dict 5 kịch bản {"S1": ScenarioResult, ...}.
    summary_kpis  : Dict 12 KPI tổng hợp toàn hệ thống.
    """
    m1: M1_MacroResult
    m2: M2_DigitalizationResult
    m3: M3_AllocationResult
    m4: M4_LaborResult
    m5: M5_RiskResult
    scenarios: Dict[str, ScenarioResult]
    summary_kpis: Dict[str, float]


# ─────────────────────────────────────────────────────────────────────────────
# 2. AIDEOM ENGINE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class AIDEOM_Engine:
    """
    Lớp trung tâm tích hợp toàn bộ Module M1–M5 của hệ thống AIDEOM-VN.

    Parameters
    ----------
    df_macro   : DataFrame từ vietnam_macro_2020_2025.csv (tuỳ chọn).
    df_sectors : DataFrame từ vietnam_sectors_2024.csv (tuỳ chọn).
    df_regions : DataFrame từ vietnam_regions_2024.csv (tuỳ chọn).

    Nếu không truyền DataFrame, engine sẽ dùng dữ liệu nội bộ mặc định.

    Ví dụ sử dụng
    -------------
    >>> engine = AIDEOM_Engine()
    >>> results = engine.run_all()
    >>> print(results.summary_kpis)
    """

    # ── Tham số Cobb-Douglas mặc định ────────────────────────────────────────
    _CD_PARAMS: Dict[str, float] = dict(
        alpha=0.33, beta=0.42, gamma=0.10, delta=0.08, theta=0.07
    )

    # ── Chuỗi thời gian 2020–2025 ────────────────────────────────────────────
    _YEARS = np.array([2020, 2021, 2022, 2023, 2024, 2025])
    _Y     = np.array([8044.4, 8487.5, 9513.3, 10221.8, 11511.9, 12847.6])
    _K     = np.array([16500,  17800,  19600,  21300,   23500,   25900])
    _L     = np.array([53.6,   50.5,   51.7,   52.4,    52.9,    53.4])
    _D     = np.array([12.0,   12.7,   14.3,   16.5,    18.3,    19.5])
    _AI    = np.array([55.6,   60.2,   65.4,   67.0,    73.8,    80.1])
    _H     = np.array([24.1,   26.1,   26.2,   27.0,    28.4,    29.2])

    # ── Dữ liệu 6 vùng kinh tế ───────────────────────────────────────────────
    _REGION_NAMES: List[str] = [
        "Trung du & Miền núi phía Bắc",
        "Đồng bằng sông Hồng",
        "Bắc Trung Bộ & DH Trung Bộ",
        "Tây Nguyên",
        "Đông Nam Bộ",
        "Đồng bằng sông Cửu Long",
    ]
    _REGION_DATA: Dict[str, List] = {
        "grdp_per_capita": [57.0, 152.3, 87.5,  68.9, 158.9, 80.5],
        "fdi":             [3.5,   20.0,  8.2,   0.8,  18.5,  2.1],
        "digital_index":   [38,    78,   55,    32,   82,   48],
        "ai_readiness":    [22,    68,   40,    18,   75,   30],
        "trained_labor":   [21.5,  36.8, 27.5,  18.2, 42.5, 16.8],
        "rd_intensity":    [0.18,  0.85,  0.32,  0.15,  0.78,  0.22],
        "internet":        [72,    92,   84,    68,   94,   78],
        "gini":            [0.405, 0.358, 0.372, 0.412, 0.385, 0.392],
    }

    # ── Dữ liệu 8 ngành lao động ─────────────────────────────────────────────
    _SECTOR_NAMES: List[str] = [
        "Nông-Lâm-Thủy sản",
        "CN chế biến chế tạo",
        "Xây dựng",
        "Bán buôn-bán lẻ",
        "Tài chính-Ngân hàng",
        "Logistics-Vận tải",
        "CNTT-Truyền thông",
        "Giáo dục-Đào tạo",
    ]
    _BASELINE_LABOR  = np.array([14.3, 11.2, 3.8, 6.5, 1.4, 2.1, 1.8, 1.5])
    _AUTOMATION_RISK = np.array([0.52, 0.48, 0.43, 0.38, 0.31, 0.35, 0.12, 0.22])

    # ─────────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        df_macro: Optional[pd.DataFrame] = None,
        df_sectors: Optional[pd.DataFrame] = None,
        df_regions: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Khởi tạo AIDEOM_Engine.

        Parameters
        ----------
        df_macro   : DataFrame macro Vietnam 2020–2025. None → dùng dữ liệu nội bộ.
        df_sectors : DataFrame theo ngành 2024. None → dùng dữ liệu nội bộ.
        df_regions : DataFrame theo vùng 2024. None → dùng dữ liệu nội bộ.
        """
        self._df_macro   = df_macro
        self._df_sectors = df_sectors
        self._df_regions = df_regions
        self._cached_results: Optional[AIDEOM_Results] = None

    # ── PUBLIC API ────────────────────────────────────────────────────────────

    def run_all(self) -> AIDEOM_Results:
        """
        Chạy toàn bộ pipeline M1 → M5 → Kịch bản → KPI tổng hợp.

        Trả về
        ------
        AIDEOM_Results
            Đối tượng chứa đầy đủ kết quả tất cả module và 5 kịch bản.
        """
        m1        = self._run_m1_macro()
        m2        = self._run_m2_digitalization()
        m3        = self._run_m3_allocation()
        m4        = self._run_m4_labor()
        m5        = self._run_m5_risk()
        scenarios = self._run_all_scenarios(m1)
        kpis      = self._compute_summary_kpis(m1, m2, m3, m4, m5, scenarios)

        self._cached_results = AIDEOM_Results(
            m1=m1, m2=m2, m3=m3, m4=m4, m5=m5,
            scenarios=scenarios, summary_kpis=kpis,
        )
        return self._cached_results

    def run_scenario(
        self,
        scenario_id: str,
        m1_result: Optional[M1_MacroResult] = None,
        q_table_path: Optional[str] = None,
    ) -> ScenarioResult:
        """
        Chạy một kịch bản đơn lẻ S1–S5.

        Parameters
        ----------
        scenario_id   : Mã kịch bản – một trong "S1" | "S2" | "S3" | "S4" | "S5".
        m1_result     : Kết quả M1 sẵn có. None → tính lại từ đầu.
        q_table_path  : Đường dẫn tới q_table.npy (chỉ cần cho S5).

        Trả về
        ------
        ScenarioResult
            GDP trajectory, điểm rủi ro, lao động dịch chuyển của kịch bản.

        Ngoại lệ
        ---------
        ValueError
            Nếu scenario_id không thuộc {"S1", ..., "S5"}.
        """
        if m1_result is None:
            m1_result = self._run_m1_macro()

        configs = self._build_scenario_configs(q_table_path)

        if scenario_id not in configs:
            raise ValueError(
                f"Kịch bản không hợp lệ: '{scenario_id}'. Chọn một trong S1–S5."
            )

        cfg   = configs[scenario_id]
        alloc = cfg["alloc"]

        gdp_traj       = self._simulate_gdp_trajectory(alloc, m1_result)
        growth_rates   = np.diff(gdp_traj) / gdp_traj[:-1]
        avg_growth     = float(np.mean(growth_rates))
        digital_weight = alloc[1] + alloc[2]
        risk_score     = cfg["risk_base"] + 0.1 * digital_weight
        labor_displace = 0.08 + 0.25 * alloc[2]

        return ScenarioResult(
            name               = scenario_id,
            label              = cfg["label"],
            allocation         = alloc,
            gdp_trajectory     = gdp_traj,
            gdp_2030           = float(gdp_traj[4]),
            gdp_2035           = float(gdp_traj[-1]),
            growth_rate_avg    = avg_growth,
            risk_score         = float(np.clip(risk_score, 0.0, 1.0)),
            inequality_delta   = cfg["inequality_delta"],
            labor_displacement = float(labor_displace),
            description        = cfg["description"],
        )

    # ── MODULE M1: MACRO FORECAST ─────────────────────────────────────────────

    def _run_m1_macro(self) -> M1_MacroResult:
        """
        Module 1: Dự báo Macro theo mô hình Cobb-Douglas mở rộng.

        Input (nội bộ)
        --------------
        _Y, _K, _L, _D, _AI, _H : Chuỗi thời gian 2020–2025.
        _CD_PARAMS               : Tham số alpha, beta, gamma, delta, theta.

        Trả về
        ------
        M1_MacroResult
            TFP series, MAPE, phân rã tăng trưởng theo 6 nhân tố, dự báo GDP 2030.
        """
        p     = self._CD_PARAMS
        Y, K, L, D, AI, H = (
            self._Y, self._K, self._L,
            self._D, self._AI, self._H,
        )
        alpha, beta, gamma, delta, theta = (
            p["alpha"], p["beta"], p["gamma"], p["delta"], p["theta"]
        )

        # Solow residual
        A_t    = Y / (K**alpha * L**beta * D**gamma * AI**delta * H**theta)
        A_mean = float(np.mean(A_t))
        Y_hat  = A_mean * (K**alpha * L**beta * D**gamma * AI**delta * H**theta)
        mape   = float(np.mean(np.abs((Y - Y_hat) / Y) * 100))

        # Log-linear growth decomposition 2020–2025
        d_ln_Y = np.log(Y[-1]) - np.log(Y[0])
        decomp = {
            "Vốn (K)":      alpha * (np.log(K[-1])  - np.log(K[0]))  / d_ln_Y * 100,
            "Lao động (L)": beta  * (np.log(L[-1])  - np.log(L[0]))  / d_ln_Y * 100,
            "Số hóa (D)":   gamma * (np.log(D[-1])  - np.log(D[0]))  / d_ln_Y * 100,
            "AI":           delta * (np.log(AI[-1]) - np.log(AI[0])) / d_ln_Y * 100,
            "Nhân lực (H)": theta * (np.log(H[-1])  - np.log(H[0]))  / d_ln_Y * 100,
            "TFP (A)":      (np.log(A_t[-1]) - np.log(A_t[0]))       / d_ln_Y * 100,
        }

        # Dự báo 2030 (n = 5 năm từ 2025)
        n      = 5
        K_2030 = K[-1]   * (1 + 0.06)  ** n
        L_2030 = L[-1]   * (1 + 0.006) ** n
        A_2030 = A_t[-1] * (1 + 0.012) ** n
        Y_2030 = A_2030 * (
            K_2030**alpha * L_2030**beta
            * 30.0**gamma * 100.0**delta * 35.0**theta
        )

        return M1_MacroResult(
            years        = self._YEARS,
            Y_actual     = Y,
            A_t          = A_t,
            A_mean       = A_mean,
            Y_forecast   = Y_hat,
            mape         = mape,
            growth_decomp= decomp,
            Y_2030       = float(Y_2030),
            params       = p,
        )

    # ── MODULE M2: DIGITALIZATION RANKING (TOPSIS + ENTROPY) ─────────────────

    def _run_m2_digitalization(self) -> M2_DigitalizationResult:
        """
        Module 2: Xếp hạng số hóa 6 vùng kinh tế bằng TOPSIS + Entropy.

        Input (nội bộ)
        --------------
        _REGION_DATA : Ma trận 6×8 chỉ tiêu số hóa.
        _REGION_NAMES: Tên 6 vùng.

        Trả về
        ------
        M2_DigitalizationResult
            Điểm TOPSIS và thứ hạng theo trọng số chuyên gia và Entropy.
        """
        X = np.array(
            [self._REGION_DATA[k] for k in [
                "grdp_per_capita", "fdi", "digital_index", "ai_readiness",
                "trained_labor", "rd_intensity", "internet", "gini",
            ]],
            dtype=float,
        ).T  # shape: (6, 8)

        w_expert   = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10])
        is_benefit = np.array([True, True, True, True, True, True, True, False])

        w_entropy = self._entropy_weights(X)
        c_exp     = self._topsis(X, w_expert,  is_benefit)
        c_ent     = self._topsis(X, w_entropy, is_benefit)

        rank_exp = pd.Series(c_exp).rank(ascending=False).astype(int).values
        rank_ent = pd.Series(c_ent).rank(ascending=False).astype(int).values
        top_idx  = int(np.argmax(c_exp))

        return M2_DigitalizationResult(
            region_names           = self._REGION_NAMES,
            topsis_scores_expert   = c_exp,
            topsis_scores_entropy  = c_ent,
            rankings_expert        = rank_exp,
            rankings_entropy       = rank_ent,
            top_region             = self._REGION_NAMES[top_idx],
            entropy_weights        = w_entropy,
        )

    # ── MODULE M3: OPTIMAL ALLOCATION (LP) ───────────────────────────────────

    def _run_m3_allocation(self, budget: float = 170.0) -> M3_AllocationResult:
        """
        Module 3: Tối ưu phân bổ ngân sách đầu tư giữa 6 vùng bằng LP.

        Parameters
        ----------
        budget : Tổng ngân sách đầu tư (nghìn tỷ VND). Mặc định 170.

        Input bổ sung
        -------------
        _df_regions hoặc _make_default_regions_df() : Dữ liệu vùng.

        Trả về
        ------
        M3_AllocationResult
            Vector dY tối ưu, Z*, trọng số vùng, hệ số chi phí.
        """
        regions = (
            self._df_regions
            if self._df_regions is not None
            else self._make_default_regions_df()
        )

        GRDP = regions["grdp_trillion_VND"].values.astype(float)
        DIG  = regions["digital_index_0_100"].values.astype(float)
        AI_R = regions["ai_readiness_0_100"].values.astype(float)

        w_raw = 0.55 * (DIG / 100) + 0.45 * (AI_R / 100)
        W     = w_raw / w_raw.sum()
        C_r   = np.clip(1.5 - 0.8 * DIG / 100 - 0.3 * GRDP / GRDP.max(), 0.4, 1.5)
        floor = 0.02 * GRDP

        try:
            import pulp  # type: ignore
            model = pulp.LpProblem("M3_Alloc", pulp.LpMaximize)
            dY    = [pulp.LpVariable(f"dY{r}", lowBound=floor[r]) for r in range(6)]
            model += pulp.lpSum(W[r] * dY[r] for r in range(6))
            model += pulp.lpSum(C_r[r] * dY[r] for r in range(6)) <= budget
            model += (dY[1] + dY[4]) >= 0.50 * pulp.lpSum(dY)
            model.solve(pulp.PULP_CBC_CMD(msg=False))
            dY_opt = np.array([pulp.value(dY[r]) for r in range(6)])
            Z_star = float(pulp.value(model.objective))
        except Exception:
            # Fallback phân tích: chia ngân sách tỷ lệ với W/C_r
            dY_opt = floor + W * (budget - np.dot(C_r, floor)) / np.dot(C_r, W) * C_r
            Z_star = float(np.dot(W, dY_opt))

        return M3_AllocationResult(
            region_names       = regions["region_name_vi"].tolist(),
            dY_optimal         = dY_opt,
            budget_used        = float(np.dot(C_r, dY_opt)),
            Z_star             = Z_star,
            allocation_weights = W,
            cost_coefficients  = C_r,
        )

    # ── MODULE M4: LABOR SIMULATION ───────────────────────────────────────────

    def _run_m4_labor(self) -> M4_LaborResult:
        """
        Module 4: Mô phỏng tác động AI/tự động hoá đến 8 ngành lao động.

        Input (nội bộ)
        --------------
        _BASELINE_LABOR  : Lao động ban đầu theo ngành (triệu người).
        _AUTOMATION_RISK : Xác suất tự động hoá theo ngành ∈ [0, 1].

        Trả về
        ------
        M4_LaborResult
            Lao động dịch chuyển, tái đào tạo, việc làm mới, tác động ròng.
        """
        L0        = self._BASELINE_LABOR
        risk      = self._AUTOMATION_RISK
        displaced = L0 * risk * 0.60          # 60% rủi ro hiện thực hoá
        reskilled = displaced * 0.55          # 55% tái đào tạo thành công
        new_jobs  = L0 * (1 - risk) * 0.08   # Việc làm mới từ ngành AI-driven
        net       = new_jobs - displaced + reskilled

        return M4_LaborResult(
            sector_names    = self._SECTOR_NAMES,
            baseline_labor  = L0,
            displaced_labor = displaced,
            reskilled_labor = reskilled,
            new_jobs        = new_jobs,
            net_impact      = net,
            total_displaced = float(displaced.sum()),
            total_new_jobs  = float(new_jobs.sum()),
        )

    # ── MODULE M5: RISK ASSESSMENT ────────────────────────────────────────────

    def _run_m5_risk(self) -> M5_RiskResult:
        """
        Module 5: Đánh giá rủi ro đa mục tiêu – Pareto front + 2-stage stochastic.

        Input
        -----
        Không có đầu vào bên ngoài; sử dụng seed cố định để tái lập được.

        Trả về
        ------
        M5_RiskResult
            Pareto front 3 mục tiêu, E[Z], Std[Z], VaR 95%.
        """
        n_pts  = 40
        t      = np.linspace(0, 1, n_pts)
        rng_42 = np.random.default_rng(42)
        rng_43 = np.random.default_rng(43)
        rng_44 = np.random.default_rng(44)

        pareto_gdp  = 12000 + 4000 * t + 500  * rng_42.normal(0, 0.05, n_pts)
        pareto_ineq = 0.42  - 0.08  * t + 0.01 * rng_43.normal(0, 0.05, n_pts)
        pareto_emit = 1.0   + 0.5   * t + 0.05 * rng_44.normal(0, 0.05, n_pts)

        # 2-stage stochastic (3 kịch bản: Thấp / TB / Cao)
        rng   = np.random.default_rng(2024)
        probs = np.array([0.25, 0.50, 0.25])
        z_vals = rng.normal([100, 115, 128], [5, 4, 6], (200, 3)) @ probs
        var_95 = float(np.percentile(z_vals, 5))

        return M5_RiskResult(
            pareto_gdp         = pareto_gdp,
            pareto_inequality  = pareto_ineq,
            pareto_emissions   = pareto_emit,
            stochastic_z_mean  = float(z_vals.mean()),
            stochastic_z_std   = float(z_vals.std()),
            scenario_probs     = probs,
            var_95             = var_95,
        )

    # ── SCENARIO RUNNER ───────────────────────────────────────────────────────

    def _run_all_scenarios(
        self,
        m1_result: Optional[M1_MacroResult] = None,
        q_table_path: Optional[str] = None,
    ) -> Dict[str, ScenarioResult]:
        """
        Chạy cả 5 kịch bản S1–S5 và trả về dict kết quả.

        Parameters
        ----------
        m1_result    : Kết quả M1 sẵn có. None → tính lại.
        q_table_path : Đường dẫn q_table.npy cho S5 (tuỳ chọn).

        Trả về
        ------
        Dict[str, ScenarioResult]
            Keys: "S1", "S2", "S3", "S4", "S5".
        """
        if m1_result is None:
            m1_result = self._run_m1_macro()
        return {
            sid: self.run_scenario(sid, m1_result, q_table_path)
            for sid in ["S1", "S2", "S3", "S4", "S5"]
        }

    def _build_scenario_configs(
        self,
        q_table_path: Optional[str] = None,
    ) -> Dict[str, dict]:
        """
        Xây dựng dict cấu hình cho 5 kịch bản S1–S5.

        Parameters
        ----------
        q_table_path : Đường dẫn q_table.npy (chỉ dùng cho S5).

        Trả về
        ------
        Dict[str, dict]
            Mỗi phần tử chứa: label, alloc, description, risk_base, inequality_delta.
        """
        return {
            "S1": {
                "label": "Truyền thống – Ưu tiên Vốn vật chất",
                "alloc": np.array([0.70, 0.10, 0.10, 0.10]),
                "description": (
                    "Chiến lược tăng trưởng truyền thống, tập trung tích lũy vốn "
                    "vật chất (70%). Phù hợp giai đoạn 2000-2015 nhưng hiệu suất "
                    "biên đang giảm dần. Rủi ro thấp, tăng trưởng ổn định nhưng chậm."
                ),
                "risk_base":        0.25,
                "inequality_delta": +0.005,
            },
            "S2": {
                "label": "Cân bằng – Phân bổ đồng đều 4 trụ cột",
                "alloc": np.array([0.40, 0.25, 0.15, 0.20]),
                "description": (
                    "Chiến lược phân bổ hài hòa, không hy sinh trụ cột nào. "
                    "Kịch bản an toàn giảm rủi ro tập trung nhưng không tận dụng "
                    "lợi thế cạnh tranh của từng lĩnh vực."
                ),
                "risk_base":        0.30,
                "inequality_delta": +0.002,
            },
            "S3": {
                "label": "Số hóa nhanh – Ưu tiên Hạ tầng số & AI",
                "alloc": np.array([0.25, 0.45, 0.15, 0.15]),
                "description": (
                    "Chiến lược bứt phá kinh tế số, đặt cược vào hạ tầng số (45%). "
                    "Phù hợp Chiến lược quốc gia về chuyển đổi số 2025–2030. "
                    "Tăng trưởng nhanh nhưng rủi ro an ninh mạng và chênh lệch số cao."
                ),
                "risk_base":        0.45,
                "inequality_delta": +0.015,
            },
            "S4": {
                "label": "Bao trùm – Ưu tiên Nhân lực & An sinh",
                "alloc": np.array([0.30, 0.20, 0.10, 0.40]),
                "description": (
                    "Chiến lược phát triển bao trùm, đầu tư mạnh vào vốn nhân lực "
                    "chất lượng cao (40%). Giảm bất bình đẳng nhưng tốc độ tăng "
                    "trưởng GDP ngắn hạn thấp hơn S3. Phù hợp khi thị trường lao "
                    "động đang bị AI tác động mạnh."
                ),
                "risk_base":        0.20,
                "inequality_delta": -0.010,
            },
            "S5": {
                "label": "AI Dẫn dắt – Chính sách thích nghi Q-Learning",
                "alloc": self._get_rl_allocation(q_table_path),
                "description": (
                    "Chiến lược được xác định bởi Agent Q-Learning – phân bổ động "
                    "thích nghi theo trạng thái kinh tế hiện tại. Kịch bản thông minh "
                    "tích hợp phản hồi thực tế, không cứng nhắc theo kế hoạch cố định."
                ),
                "risk_base":        0.35,
                "inequality_delta": +0.008,
            },
        }

    # ── SIMULATION HELPERS ────────────────────────────────────────────────────

    def _simulate_gdp_trajectory(
        self,
        alloc: np.ndarray,
        m1: M1_MacroResult,
        n_years: int = 10,
    ) -> np.ndarray:
        """
        Mô phỏng quỹ đạo GDP năm 2026–2035 theo vector phân bổ ngân sách.

        Parameters
        ----------
        alloc   : [w_K, w_D, w_AI, w_H] – tổng bằng 1.0.
        m1      : Kết quả M1 (cần A_mean, params và trạng thái cuối 2025).
        n_years : Số năm mô phỏng; mặc định 10 (2026–2035).

        Trả về
        ------
        np.ndarray
            Mảng n_years phần tử – GDP (nghìn tỷ VND) mỗi năm.
        """
        p                = m1.params
        w_K, w_D, w_AI, w_H = alloc
        budget_annual    = 2000.0   # nghìn tỷ VND đầu tư mới mỗi năm

        K  = float(self._K[-1])
        L  = float(self._L[-1])
        D  = float(self._D[-1])
        AI = float(self._AI[-1])
        H  = float(self._H[-1])
        A  = m1.A_mean * 1.012      # TFP khởi đầu (tăng 1.2%/năm từ mean)

        traj = []
        for _ in range(n_years):
            K  += w_K  * budget_annual
            D  += w_D  * budget_annual / 200
            AI += w_AI * budget_annual / 30
            H  += w_H  * budget_annual / 300
            L  *= 1.006     # tăng tự nhiên 0.6%/năm
            A  *= 1.012     # TFP nội sinh

            Y = A * (
                K  ** p["alpha"] *
                L  ** p["beta"]  *
                D  ** p["gamma"] *
                AI ** p["delta"] *
                H  ** p["theta"]
            )
            traj.append(Y)

        return np.array(traj)

    def _get_rl_allocation(
        self,
        q_table_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Lấy chính sách phân bổ tối ưu từ Q-table đã huấn luyện (bai11).

        Trạng thái đại diện: (GDP=2, D=1, AI=1, U=0) – kinh tế tăng trưởng tốt.

        Parameters
        ----------
        q_table_path : Đường dẫn file q_table.npy.
                       None → thử q_table.npy hiện tại,
                       rồi /mnt/user-data/uploads/q_table.npy.

        Trả về
        ------
        np.ndarray
            Vector phân bổ [w_K, w_D, w_AI, w_H]. Fallback: action 3 (AI-led).
        """
        action_map = {
            0: np.array([0.70, 0.10, 0.10, 0.10]),
            1: np.array([0.40, 0.25, 0.15, 0.20]),
            2: np.array([0.25, 0.45, 0.15, 0.15]),
            3: np.array([0.20, 0.20, 0.45, 0.15]),
            4: np.array([0.30, 0.20, 0.10, 0.40]),
        }
        try:
            path = q_table_path or "q_table.npy"
            if not os.path.exists(path):
                path = "/mnt/user-data/uploads/q_table.npy"
            Q           = np.load(path)
            state       = (2, 1, 1, 0)
            best_action = int(np.argmax(Q[state]))
            return action_map[best_action]
        except Exception:
            return action_map[3]  # Fallback: AI-led

    def _compute_summary_kpis(
        self,
        m1: M1_MacroResult,
        m2: M2_DigitalizationResult,
        m3: M3_AllocationResult,
        m4: M4_LaborResult,
        m5: M5_RiskResult,
        scenarios: Dict[str, ScenarioResult],
    ) -> Dict[str, float]:
        """
        Tính toán 12 KPI tổng hợp của toàn hệ thống AIDEOM-VN.

        Parameters
        ----------
        m1, m2, m3, m4, m5 : Kết quả 5 module đã chạy.
        scenarios           : Dict kết quả 5 kịch bản.

        Trả về
        ------
        Dict[str, float | str]
            12 KPI chính: GDP thực 2025, GDP 2030 tốt nhất, TFP CAGR, MAPE,
            điểm vùng dẫn đầu, Z* phân bổ, lao động dịch chuyển, VaR 95%,
            kịch bản tốt nhất, tốc độ tăng trưởng trung bình kịch bản tốt nhất.
        """
        best = max(scenarios.values(), key=lambda s: s.gdp_2030)

        return {
            "gdp_2025_actual":              float(m1.Y_actual[-1]),
            "gdp_2030_best_case":           best.gdp_2030,
            "tfp_cagr_pct":                 float((m1.A_t[-1] / m1.A_t[0]) ** (1 / 5) - 1) * 100,
            "mape_pct":                     m1.mape,
            "top_region_score":             float(m2.topsis_scores_expert.max()),
            "top_region_name":              m2.top_region,          # type: ignore[return-value]
            "optimal_z_star":               m3.Z_star,
            "total_labor_displaced_mil":    m4.total_displaced,
            "total_new_jobs_mil":           m4.total_new_jobs,
            "var_95_stochastic":            m5.var_95,
            "best_scenario_id":             best.name,              # type: ignore[return-value]
            "best_scenario_growth_avg_pct": best.growth_rate_avg * 100,
        }

    # ── STATIC / PURE HELPERS ─────────────────────────────────────────────────

    @staticmethod
    def _topsis(
        matrix: np.ndarray,
        weights: np.ndarray,
        benefit: np.ndarray,
    ) -> np.ndarray:
        """
        Tính điểm TOPSIS cho ma trận quyết định.

        Parameters
        ----------
        matrix  : Ma trận (n_alternatives × n_criteria).
        weights : Vector trọng số tổng = 1.
        benefit : Boolean mask – True nếu tiêu chí lớn hơn là tốt hơn.

        Trả về
        ------
        np.ndarray
            Vector điểm C* ∈ [0, 1] cho mỗi phương án.
        """
        R     = matrix / np.sqrt((matrix ** 2).sum(axis=0))
        V     = R * weights
        A_pos = np.where(benefit, V.max(0), V.min(0))
        A_neg = np.where(benefit, V.min(0), V.max(0))
        S_pos = np.sqrt(((V - A_pos) ** 2).sum(1))
        S_neg = np.sqrt(((V - A_neg) ** 2).sum(1))
        return S_neg / (S_pos + S_neg)

    @staticmethod
    def _entropy_weights(matrix: np.ndarray) -> np.ndarray:
        """
        Tính trọng số Entropy từ ma trận quyết định.

        Parameters
        ----------
        matrix : Ma trận (n_alternatives × n_criteria), tất cả giá trị > 0.

        Trả về
        ------
        np.ndarray
            Vector trọng số tổng = 1.
        """
        P = matrix / matrix.sum(0)
        k = 1.0 / np.log(len(matrix))
        E = -k * np.nansum(P * np.log(P + 1e-12), axis=0)
        d = 1 - E
        return d / d.sum()

    def _make_default_regions_df(self) -> pd.DataFrame:
        """
        Tạo DataFrame vùng kinh tế mặc định khi không có file CSV bên ngoài.

        Trả về
        ------
        pd.DataFrame
            5 cột: region_name_vi, grdp_trillion_VND, grdp_growth_pct,
            digital_index_0_100, ai_readiness_0_100.
        """
        return pd.DataFrame({
            "region_name_vi":      self._REGION_NAMES,
            "grdp_trillion_VND":   [198.5, 1456.2, 612.3, 145.8, 2156.4, 684.7],
            "grdp_growth_pct":     [7.2, 8.5, 6.8, 7.9, 8.1, 6.5],
            "digital_index_0_100": [38, 78, 55, 32, 82, 48],
            "ai_readiness_0_100":  [22, 68, 40, 18, 75, 30],
        })