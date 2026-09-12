"""writers 单元测试：result1/2/3/4 结构写后读回（V-9），含掩码与行数上限。"""
import numpy as np
import pytest

from drymodel import writers as W

A1 = "时间\\到药材中心的距离"
COLS21 = [round(0.1 * j, 4) for j in range(21)]
COLS20 = [round(0.1 * j, 4) for j in range(20)]


def test_write_result12_structure(tmp_path):
    p = tmp_path / "result1.xlsx"
    t = np.arange(1, 6)
    C = np.full((5, 21), 2.5)
    T = np.full((5, 21), 40.0)
    W.write_result12(p, t, C, T, COLS21, A1)
    r = W.verify_workbook(p, expected_sheets=["温度", "水分浓度"], a1_text=A1,
                          expected_cols=COLS21, n_data_rows=5, t_start=1, t_step=1)
    assert r["ok"], r["issues"]


def test_write_result3_structure(tmp_path):
    p = tmp_path / "result3.xlsx"
    t = np.arange(60, 60 * 6, 60)
    grid = np.full((len(t), 21), 0.3)
    W.write_result34(p, t, grid, COLS21, A1, sheet_name="Sheet1")
    r = W.verify_workbook(p, expected_sheets=["Sheet1"], a1_text=A1,
                          expected_cols=COLS21, n_data_rows=len(t), t_start=60, t_step=60)
    assert r["ok"], r["issues"]


def test_write_result4_masking(tmp_path):
    p = tmp_path / "result4.xlsx"
    t = np.array([60, 120, 180])
    # 构造掩码：第 0 行全域内；第 1/2 行末列域外
    grid = [[0.3] * 20, [0.3] * 19 + [None], [0.3] * 18 + [None, None]]
    surf = [0.2, 0.19, 0.18]
    W.write_result34(p, t, grid, COLS20, A1, sheet_name="Sheet1",
                     surface_header="药材表面", surface_values=surf)
    mask = [[True] * 20, [True] * 19 + [False], [True] * 18 + [False, False]]
    r = W.verify_workbook(p, expected_sheets=["Sheet1"], a1_text=A1,
                          expected_cols=COLS20, n_data_rows=3, t_start=60, t_step=60,
                          surface_header="药材表面", mask=mask)
    assert r["ok"], r["issues"]


def test_row_budget_exceeded(tmp_path):
    p = tmp_path / "toobig.xlsx"
    t = np.arange(1, 3)
    with pytest.raises(ValueError):
        # 伪造超限：直接调用预算检查
        W._check_row_budget(W.EXCEL_MAX_ROWS, p)


def test_rounding_applied(tmp_path):
    import openpyxl
    p = tmp_path / "round.xlsx"
    t = np.array([1])
    C = np.array([[0.149963] + [2.5] * 20])
    T = np.array([[40.12345] + [40.0] * 20])
    W.write_result12(p, t, C, T, COLS21, A1)
    wb = openpyxl.load_workbook(p, read_only=True)
    ws = wb["水分浓度"]
    rows = list(ws.iter_rows(values_only=True))
    assert rows[1][1] == pytest.approx(0.15)          # 0.149963 → 0.1500
    wb.close()


def test_verify_catches_wrong_header(tmp_path):
    """增强检查器：表头列值错误应被捕获。"""
    p = tmp_path / "r.xlsx"
    t = np.arange(60, 60 * 4, 60)
    grid = np.full((len(t), 21), 0.3)
    W.write_result34(p, t, grid, COLS21, A1, sheet_name="Sheet1")
    # 用错误的期望列头（0.15 而非 0.1）应报不一致
    bad_cols = [0.0, 0.15] + [round(0.1 * j, 4) for j in range(2, 21)]
    r = W.verify_workbook(p, expected_sheets=["Sheet1"], a1_text=A1,
                          expected_cols=bad_cols, n_data_rows=len(t), t_start=60, t_step=60)
    assert not r["ok"]


def test_cross_file_check_consistent(tmp_path):
    """V-8：result2(1s) 与 result3(60s) 共同 60s 时刻舍入相等。"""
    # result2：1..180 s 每 1 s；result3：60,120,180 s
    import numpy as _np
    t2 = _np.arange(1, 181)
    C2 = _np.tile(_np.linspace(0.3, 0.2, 21), (len(t2), 1))
    T2 = _np.full((len(t2), 21), 40.0)
    r2 = tmp_path / "result2.xlsx"
    W.write_result12(r2, t2, C2, T2, COLS21, A1)
    t3 = _np.array([60, 120, 180])
    C3 = _np.tile(_np.linspace(0.3, 0.2, 21), (3, 1))
    r3 = tmp_path / "result3.xlsx"
    W.write_result34(r3, t3, C3, COLS21, A1, sheet_name="Sheet1")
    res = W.cross_file_check(r2, r3)
    assert res["ok"] and res["n_common_times"] == 3 and res["n_mismatch"] == 0


def test_cross_file_check_detects_mismatch(tmp_path):
    import numpy as _np
    t2 = _np.arange(1, 181)
    C2 = _np.full((len(t2), 21), 0.3)
    T2 = _np.full((len(t2), 21), 40.0)
    r2 = tmp_path / "result2.xlsx"
    W.write_result12(r2, t2, C2, T2, COLS21, A1)
    t3 = _np.array([60, 120, 180])
    C3 = _np.full((3, 21), 0.25)          # 与 result2 不同源
    r3 = tmp_path / "result3.xlsx"
    W.write_result34(r3, t3, C3, COLS21, A1, sheet_name="Sheet1")
    res = W.cross_file_check(r2, r3)
    assert not res["ok"] and res["n_mismatch"] > 0
