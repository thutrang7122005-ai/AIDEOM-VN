"""
test_system.py – Unit Tests cho Hệ thống AIDEOM-VN
====================================================
Chạy bằng lệnh: pytest test_system.py -v

Bao gồm:
    - TestEngineInit      : Kiểm tra khởi tạo AIDEOM_Engine
    - TestM1MacroCalc     : Kiểm tra logic tính toán Module M1
    - TestRunScenario     : Kiểm tra hàm run_scenario() với S1–S5
    - TestRunAll          : Kiểm tra pipeline run_all() đầu đủ
    - TestEdgeCases       : Kiểm tra các trường hợp biên (None input, kịch bản lạ)
"""

from __future__ import annotations

import sys
import os
import pytest
import numpy as np
import pandas as pd

# ── Đảm bảo import được engine.py dù chạy từ thư mục nào ──
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from engine import (
    AIDEOM_Engine,
    M1_MacroResult,
    M2_DigitalizationResult,
    M3_AllocationResult,
    M4_LaborResult,
    M5_RiskResult,
    ScenarioResult,
    AIDEOM_Results,
)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def engine_no_data() -> AIDEOM_Engine:
    """
    Engine khởi tạo không có DataFrame thực (dùng dữ liệu mặc định hardcoded).
    Tái sử dụng toàn module để tiết kiệm thời gian.
    """
    return AIDEOM_Engine(df_macro=None, df_sectors=None, df_regions=None)


@pytest.fixture(scope="module")
def engine_with_regions(engine_no_data) -> AIDEOM_Engine:
    """
    Engine có DataFrame vùng tối thiểu (để test M3 fallback LP).
    """
    df_regions = pd.DataFrame({
        "region_name_vi": [
            "Trung du & Miền núi phía Bắc", "Đồng bằng sông Hồng",
            "Bắc Trung Bộ & DH Trung Bộ", "Tây Nguyên",
            "Đông Nam Bộ", "Đồng bằng sông Cửu Long",
        ],
        "grdp_trillion_VND": [198.5, 1456.2, 612.3, 145.8, 2156.4, 684.7],
        "grdp_growth_pct":   [7.2,   8.5,    6.8,   7.9,   8.1,    6.5],
        "digital_index_0_100": [38,  78,     55,    32,    82,     48],
        "ai_readiness_0_100":  [22,  68,     40,    18,    75,     30],
    })
    return AIDEOM_Engine(df_macro=None, df_sectors=None, df_regions=df_regions)


@pytest.fixture(scope="module")
def m1_result(engine_no_data) -> M1_MacroResult:
    """M1 result được tính một lần, tái dùng cho nhiều test."""
    return engine_no_data._run_m1_macro()


@pytest.fixture(scope="module")
def full_results(engine_no_data) -> AIDEOM_Results:
    """Chạy toàn bộ pipeline một lần dùng cho TestRunAll."""
    return engine_no_data.run_all()


# ─────────────────────────────────────────────────────────────────────────────
# TEST CLASS 1: KHỞI TẠO ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class TestEngineInit:
    """Kiểm tra các biến thể khởi tạo AIDEOM_Engine."""

    def test_init_all_none(self):
        """Engine phải khởi tạo thành công khi tất cả DataFrame là None."""
        engine = AIDEOM_Engine(df_macro=None, df_sectors=None, df_regions=None)
        assert engine is not None

    def test_init_stores_dataframes(self):
        """Engine phải lưu đúng các DataFrame được truyền vào."""
        df_mock = pd.DataFrame({"col": [1, 2, 3]})
        engine = AIDEOM_Engine(df_macro=df_mock, df_sectors=None, df_regions=None)
        assert engine.df_macro is df_mock
        assert engine.df_sectors is None
        assert engine.df_regions is None

    def test_init_results_is_none(self, engine_no_data):
        """_results phải là None trước khi gọi run_all()."""
        engine = AIDEOM_Engine()
        assert engine._results is None

    def test_cd_params_keys(self, engine_no_data):
        """CD_PARAMS phải có đủ 5 hệ số: alpha, beta, gamma, delta, theta."""
        required_keys = {"alpha", "beta", "gamma", "delta", "theta"}
        assert required_keys.issubset(engine_no_data.CD_PARAMS.keys())

    def test_cd_params_sum_approx_one(self, engine_no_data):
        """Tổng các hệ số Cobb-Douglas nên xấp xỉ 1.0 (returns to scale)."""
        params = engine_no_data.CD_PARAMS
        total = sum(params.values())
        assert abs(total - 1.0) < 0.05, f"Tổng hệ số = {total:.4f}, nên gần 1.0"

    def test_static_data_lengths_consistent(self, engine_no_data):
        """Tất cả chuỗi dữ liệu thời gian phải có cùng độ dài."""
        eng = engine_no_data
        lengths = [len(arr) for arr in [eng._Y, eng._K, eng._L, eng._D, eng._AI, eng._H, eng._YEARS]]
        assert len(set(lengths)) == 1, f"Chiều dài không đồng nhất: {lengths}"

    def test_region_data_all_six(self, engine_no_data):
        """Phải có đúng 6 vùng kinh tế."""
        assert len(engine_no_data._REGION_NAMES) == 6

    def test_sector_data_all_eight(self, engine_no_data):
        """Phải có đúng 8 ngành lao động."""
        assert len(engine_no_data._SECTOR_NAMES) == 8
        assert len(engine_no_data._BASELINE_LABOR) == 8
        assert len(engine_no_data._AUTOMATION_RISK) == 8


# ─────────────────────────────────────────────────────────────────────────────
# TEST CLASS 2: MODULE M1 – TÍNH TOÁN GDP / TFP
# ─────────────────────────────────────────────────────────────────────────────

class TestM1MacroCalc:
    """Kiểm tra logic tính toán Module M1: Cobb-Douglas mở rộng."""

    def test_m1_returns_correct_type(self, m1_result):
        """_run_m1_macro() phải trả về M1_MacroResult."""
        assert isinstance(m1_result, M1_MacroResult)

    def test_m1_tfp_series_length(self, m1_result):
        """Chuỗi TFP phải có cùng độ dài với chuỗi năm (6 điểm)."""
        assert len(m1_result.A_t) == len(m1_result.years) == 6

    def test_m1_tfp_all_positive(self, m1_result):
        """Tất cả giá trị TFP phải dương (A_t > 0)."""
        assert np.all(m1_result.A_t > 0), "TFP âm – có lỗi tính toán!"

    def test_m1_tfp_mean_equals_mean_of_series(self, m1_result):
        """A_mean phải bằng np.mean(A_t)."""
        expected = float(np.mean(m1_result.A_t))
        assert abs(m1_result.A_mean - expected) < 1e-9

    def test_m1_mape_low(self, m1_result):
        """
        MAPE phải dưới 10% – mô hình Cobb-Douglas với A_mean là ước lượng
        đơn giản hóa nên cho phép sai số cao hơn mô hình hồi quy đầy đủ.
        Trong thực tế mô hình đạt ~6–7%, vẫn thuộc ngưỡng chấp nhận được.
        """
        assert m1_result.mape < 10.0, f"MAPE = {m1_result.mape:.4f}% ≥ 10% – mô hình kém chính xác!"

    def test_m1_mape_nonnegative(self, m1_result):
        """MAPE không thể âm."""
        assert m1_result.mape >= 0.0

    def test_m1_y_forecast_length(self, m1_result):
        """Y_forecast phải có cùng độ dài với Y_actual."""
        assert len(m1_result.Y_forecast) == len(m1_result.Y_actual)

    def test_m1_y_actual_increasing(self, m1_result):
        """GDP thực tế Việt Nam 2020-2025 phải có xu hướng tăng."""
        # Cho phép 1 năm giảm (COVID 2021 ảnh hưởng)
        diffs = np.diff(m1_result.Y_actual)
        n_positive = np.sum(diffs > 0)
        assert n_positive >= 4, "GDP không có xu hướng tăng – dữ liệu đáng ngờ"

    def test_m1_y2030_greater_than_y2025(self, m1_result):
        """GDP dự báo 2030 phải lớn hơn GDP thực tế 2025."""
        assert m1_result.Y_2030 > m1_result.Y_actual[-1], (
            f"Y_2030={m1_result.Y_2030:.1f} ≤ Y_2025={m1_result.Y_actual[-1]:.1f}"
        )

    def test_m1_growth_decomp_keys(self, m1_result):
        """Phân rã tăng trưởng phải có đủ 6 nhân tố."""
        required = {"Vốn (K)", "Lao động (L)", "Số hóa (D)", "AI", "Nhân lực (H)", "TFP (A)"}
        assert required.issubset(m1_result.growth_decomp.keys())

    def test_m1_growth_decomp_sum_approx_100(self, m1_result):
        """Tổng % đóng góp phải xấp xỉ 100%."""
        total = sum(m1_result.growth_decomp.values())
        assert abs(total - 100.0) < 2.0, f"Tổng đóng góp = {total:.2f}%, nên ≈ 100%"

    def test_m1_params_stored_correctly(self, m1_result):
        """Tham số lưu trong result phải khớp với CD_PARAMS của engine."""
        assert m1_result.params["alpha"] == pytest.approx(0.33)
        assert m1_result.params["beta"] == pytest.approx(0.42)
        assert m1_result.params["gamma"] == pytest.approx(0.10)

    def test_m1_cobb_douglas_formula_consistency(self, engine_no_data, m1_result):
        """
        Kiểm tra công thức Cobb-Douglas: Y = A * K^α * L^β * D^γ * AI^δ * H^θ
        Dùng A_mean và các giá trị năm 2025 (năm cuối), kết quả phải gần Y_actual[-1].
        """
        p = m1_result.params
        eng = engine_no_data
        Y_check = (
            m1_result.A_mean
            * eng._K[-1] ** p["alpha"]
            * eng._L[-1] ** p["beta"]
            * eng._D[-1] ** p["gamma"]
            * eng._AI[-1] ** p["delta"]
            * eng._H[-1] ** p["theta"]
        )
        err_pct = abs(Y_check - m1_result.Y_actual[-1]) / m1_result.Y_actual[-1] * 100
        assert err_pct < 10.0, f"Sai số công thức kiểm chứng = {err_pct:.2f}%"


# ─────────────────────────────────────────────────────────────────────────────
# TEST CLASS 3: HÀM RUN_SCENARIO
# ─────────────────────────────────────────────────────────────────────────────

class TestRunScenario:
    """Kiểm tra hàm run_scenario() với 5 kịch bản S1–S5."""

    @pytest.mark.parametrize("scenario_id", ["S1", "S2", "S3", "S4", "S5"])
    def test_each_scenario_returns_correct_type(self, engine_no_data, scenario_id):
        """Mỗi kịch bản phải trả về ScenarioResult."""
        result = engine_no_data.run_scenario(scenario_id)
        assert isinstance(result, ScenarioResult)

    @pytest.mark.parametrize("scenario_id", ["S1", "S2", "S3", "S4", "S5"])
    def test_each_scenario_name_matches(self, engine_no_data, scenario_id):
        """ScenarioResult.name phải bằng scenario_id truyền vào."""
        result = engine_no_data.run_scenario(scenario_id)
        assert result.name == scenario_id

    @pytest.mark.parametrize("scenario_id", ["S1", "S2", "S3", "S4", "S5"])
    def test_allocation_sums_to_one(self, engine_no_data, scenario_id):
        """Vector phân bổ [w_K, w_D, w_AI, w_H] phải có tổng = 1.0."""
        result = engine_no_data.run_scenario(scenario_id)
        assert abs(result.allocation.sum() - 1.0) < 1e-6, (
            f"{scenario_id}: tổng allocation = {result.allocation.sum():.6f} ≠ 1.0"
        )

    @pytest.mark.parametrize("scenario_id", ["S1", "S2", "S3", "S4", "S5"])
    def test_allocation_all_nonnegative(self, engine_no_data, scenario_id):
        """Mọi tỷ trọng phân bổ phải không âm."""
        result = engine_no_data.run_scenario(scenario_id)
        assert np.all(result.allocation >= 0), f"{scenario_id}: có trọng số âm!"

    @pytest.mark.parametrize("scenario_id", ["S1", "S2", "S3", "S4", "S5"])
    def test_gdp_trajectory_length(self, engine_no_data, scenario_id):
        """Quỹ đạo GDP phải có đúng 10 điểm (2026–2035)."""
        result = engine_no_data.run_scenario(scenario_id)
        assert len(result.gdp_trajectory) == 10

    @pytest.mark.parametrize("scenario_id", ["S1", "S2", "S3", "S4", "S5"])
    def test_gdp_2030_equals_index_4(self, engine_no_data, scenario_id):
        """gdp_2030 phải đúng bằng gdp_trajectory[4] (năm thứ 5 = 2030)."""
        result = engine_no_data.run_scenario(scenario_id)
        assert result.gdp_2030 == pytest.approx(result.gdp_trajectory[4])

    @pytest.mark.parametrize("scenario_id", ["S1", "S2", "S3", "S4", "S5"])
    def test_gdp_trajectory_all_positive(self, engine_no_data, scenario_id):
        """Mọi giá trị GDP trong quỹ đạo phải dương."""
        result = engine_no_data.run_scenario(scenario_id)
        assert np.all(result.gdp_trajectory > 0)

    @pytest.mark.parametrize("scenario_id", ["S1", "S2", "S3", "S4", "S5"])
    def test_risk_score_in_bounds(self, engine_no_data, scenario_id):
        """risk_score phải nằm trong [0, 1]."""
        result = engine_no_data.run_scenario(scenario_id)
        assert 0.0 <= result.risk_score <= 1.0, (
            f"{scenario_id}: risk_score = {result.risk_score:.4f} ngoài [0,1]"
        )

    @pytest.mark.parametrize("scenario_id", ["S1", "S2", "S3", "S4", "S5"])
    def test_description_nonempty(self, engine_no_data, scenario_id):
        """Mỗi kịch bản phải có mô tả không rỗng."""
        result = engine_no_data.run_scenario(scenario_id)
        assert len(result.description.strip()) > 20

    def test_s3_has_higher_digital_than_s1(self, engine_no_data):
        """S3 (Số hóa nhanh) phải có w_D cao hơn S1 (Truyền thống)."""
        s1 = engine_no_data.run_scenario("S1")
        s3 = engine_no_data.run_scenario("S3")
        assert s3.allocation[1] > s1.allocation[1], "S3 phải ưu tiên số hóa hơn S1"

    def test_s4_has_lower_risk_than_s3(self, engine_no_data):
        """S4 (Bao trùm) phải có rủi ro thấp hơn S3 (Số hóa nhanh)."""
        s3 = engine_no_data.run_scenario("S3")
        s4 = engine_no_data.run_scenario("S4")
        assert s4.risk_score < s3.risk_score, (
            f"S4 risk={s4.risk_score:.3f} không nhỏ hơn S3 risk={s3.risk_score:.3f}"
        )

    def test_invalid_scenario_raises_error(self, engine_no_data):
        """Truyền kịch bản không hợp lệ phải raise ValueError."""
        with pytest.raises(ValueError, match="không hợp lệ"):
            engine_no_data.run_scenario("S99")


# ─────────────────────────────────────────────────────────────────────────────
# TEST CLASS 4: PIPELINE RUN_ALL
# ─────────────────────────────────────────────────────────────────────────────

class TestRunAll:
    """Kiểm tra pipeline tổng thể run_all()."""

    def test_run_all_returns_aideom_results(self, full_results):
        """run_all() phải trả về AIDEOM_Results."""
        assert isinstance(full_results, AIDEOM_Results)

    def test_run_all_has_all_modules(self, full_results):
        """Kết quả phải chứa đủ M1–M5."""
        assert isinstance(full_results.m1, M1_MacroResult)
        assert isinstance(full_results.m2, M2_DigitalizationResult)
        assert isinstance(full_results.m3, M3_AllocationResult)
        assert isinstance(full_results.m4, M4_LaborResult)
        assert isinstance(full_results.m5, M5_RiskResult)

    def test_run_all_has_five_scenarios(self, full_results):
        """Phải có đúng 5 kịch bản S1–S5."""
        assert set(full_results.scenarios.keys()) == {"S1", "S2", "S3", "S4", "S5"}

    def test_run_all_summary_kpis_nonempty(self, full_results):
        """summary_kpis phải có ít nhất 10 chỉ số."""
        assert len(full_results.summary_kpis) >= 10

    def test_run_all_kpi_gdp_2025(self, full_results):
        """KPI gdp_2025_actual phải bằng Y_actual[-1] của M1."""
        assert full_results.summary_kpis["gdp_2025_actual"] == pytest.approx(
            full_results.m1.Y_actual[-1]
        )

    def test_run_all_best_scenario_in_scenarios(self, full_results):
        """best_scenario_id phải là một trong S1–S5."""
        assert full_results.summary_kpis["best_scenario_id"] in {"S1", "S2", "S3", "S4", "S5"}

    def test_run_all_m2_top_region_nonempty(self, full_results):
        """top_region_name phải là chuỗi không rỗng."""
        name = full_results.summary_kpis["top_region_name"]
        assert isinstance(name, str) and len(name) > 3

    def test_run_all_m4_labor_balance(self, full_results):
        """Total displaced phải dương và lớn hơn total new jobs."""
        m4 = full_results.m4
        assert m4.total_displaced > 0
        assert m4.total_new_jobs > 0
        # Trong ngắn hạn, dịch chuyển > việc làm mới là điều thực tế
        assert m4.total_displaced > m4.total_new_jobs

    def test_run_all_m5_var_less_than_mean(self, full_results):
        """VaR 95% phải nhỏ hơn E[Z] (worst-case < average case)."""
        m5 = full_results.m5
        assert m5.var_95 < m5.stochastic_z_mean

    def test_run_all_m5_pareto_lengths_equal(self, full_results):
        """Ba mảng Pareto front phải có cùng độ dài."""
        m5 = full_results.m5
        assert len(m5.pareto_gdp) == len(m5.pareto_inequality) == len(m5.pareto_emissions)

    def test_run_all_m2_rankings_valid_range(self, full_results):
        """Rankings TOPSIS phải nằm trong [1, 6] với 6 vùng."""
        m2 = full_results.m2
        assert m2.rankings_expert.min() == 1
        assert m2.rankings_expert.max() == 6
        assert len(set(m2.rankings_expert)) == 6   # không có xếp hạng trùng


# ─────────────────────────────────────────────────────────────────────────────
# TEST CLASS 5: EDGE CASES
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Kiểm tra các trường hợp biên và khả năng chịu lỗi."""

    def test_run_scenario_with_precomputed_m1(self, engine_no_data, m1_result):
        """run_scenario() với m1_result truyền vào phải hoạt động đúng."""
        result = engine_no_data.run_scenario("S2", m1_result=m1_result)
        assert isinstance(result, ScenarioResult)
        assert len(result.gdp_trajectory) == 10

    def test_simulate_gdp_trajectory_output_shape(self, engine_no_data, m1_result):
        """_simulate_gdp_trajectory() phải trả về array 10 phần tử."""
        alloc = np.array([0.25, 0.25, 0.25, 0.25])
        traj = engine_no_data._simulate_gdp_trajectory(alloc, m1_result, n_years=10)
        assert traj.shape == (10,)

    def test_simulate_gdp_trajectory_increasing_under_growth(self, engine_no_data, m1_result):
        """
        Với phân bổ đồng đều và tăng trưởng TFP dương,
        GDP 2035 phải lớn hơn GDP 2026.
        """
        alloc = np.array([0.25, 0.25, 0.25, 0.25])
        traj = engine_no_data._simulate_gdp_trajectory(alloc, m1_result)
        assert traj[-1] > traj[0], "GDP không tăng – kiểm tra lại hàm simulation"

    def test_make_default_regions_df_shape(self, engine_no_data):
        """_make_default_regions_df() phải tạo DataFrame 6 hàng × ≥5 cột."""
        df = engine_no_data._make_default_regions_df()
        assert len(df) == 6
        assert "region_name_vi" in df.columns
        assert "grdp_trillion_VND" in df.columns

    def test_run_all_scenarios_returns_all_five(self, engine_no_data, m1_result):
        """run_all_scenarios() phải trả về dict với đúng 5 kịch bản."""
        all_scen = engine_no_data.run_all_scenarios(m1_result=m1_result)
        assert len(all_scen) == 5
        assert set(all_scen.keys()) == {"S1", "S2", "S3", "S4", "S5"}

    def test_rl_allocation_fallback_when_no_qtable(self, engine_no_data):
        """
        _get_rl_allocation() phải trả về array hợp lệ ngay cả khi
        q_table.npy không tồn tại (fallback về action mặc định).
        """
        alloc = engine_no_data._get_rl_allocation(q_table_path="nonexistent_file.npy")
        assert alloc.shape == (4,)
        assert abs(alloc.sum() - 1.0) < 1e-6

    def test_entropy_weights_sum_to_one(self, engine_no_data):
        """
        Trọng số Entropy (M2) phải có tổng = 1.0
        (được kiểm tra gián tiếp qua kết quả M2).
        """
        m2 = engine_no_data._run_m2_digitalization()
        assert abs(m2.entropy_weights.sum() - 1.0) < 1e-9

    def test_topsis_scores_in_zero_one(self, engine_no_data):
        """Điểm C* TOPSIS phải nằm trong [0, 1]."""
        m2 = engine_no_data._run_m2_digitalization()
        assert np.all(m2.topsis_scores_expert >= 0)
        assert np.all(m2.topsis_scores_expert <= 1)
        assert np.all(m2.topsis_scores_entropy >= 0)
        assert np.all(m2.topsis_scores_entropy <= 1)