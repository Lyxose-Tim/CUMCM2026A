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


def _verify_result3(path, n_rows):
    return W.verify_workbook(
        path, expected_sheets=["Sheet1"], a1_text=A1,
        expected_cols=COLS21, n_data_rows=n_rows, t_start=60, t_step=60,
    )


@pytest.mark.parametrize(("cell", "value", "issue_text"), [
    ("B1", 999.0, "表头第2列"),
    ("A2", "60", "真实整数"),
    ("B2", "not-a-number", "有限数值"),
    ("B2", None, "有限数值"),
])
def test_verify_workbook_rejects_header_type_and_inside_value_errors(
        tmp_path, cell, value, issue_text):
    import openpyxl
    p = tmp_path / "bad.xlsx"
    W.write_result34(p, [60, 120], np.full((2, 21), 0.3), COLS21, A1)
    wb = openpyxl.load_workbook(p)
    wb["Sheet1"][cell] = value
    wb.save(p)
    wb.close()
    result = _verify_result3(p, 2)
    assert not result["ok"]
    assert any(issue_text in issue for issue in result["issues"])


def test_verify_workbook_rejects_wrong_number_format(tmp_path):
    import openpyxl
    p = tmp_path / "format.xlsx"
    W.write_result34(p, [60], np.full((1, 21), 0.3), COLS21, A1)
    wb = openpyxl.load_workbook(p)
    wb["Sheet1"]["B2"].number_format = "General"
    wb.save(p)
    wb.close()
    result = _verify_result3(p, 1)
    assert not result["ok"]
    assert any("格式" in issue for issue in result["issues"])


def test_verify_result4_checks_surface_value_and_format(tmp_path):
    import openpyxl
    p = tmp_path / "result4_bad_surface.xlsx"
    grid = [[0.3] * 20]
    W.write_result34(
        p, [60], grid, COLS20, A1, surface_header="药材表面", surface_values=[0.2],
    )
    wb = openpyxl.load_workbook(p)
    wb["Sheet1"].cell(2, 22).value = None
    wb.save(p)
    wb.close()
    result = W.verify_workbook(
        p, expected_sheets=["Sheet1"], a1_text=A1, expected_cols=COLS20,
        n_data_rows=1, t_start=60, t_step=60, surface_header="药材表面",
        mask=[[True] * 20],
    )
    assert not result["ok"]
    assert any("表面列" in issue for issue in result["issues"])


def test_result2_result3_streaming_cross_file_check(tmp_path):
    import openpyxl
    p2 = tmp_path / "result2.xlsx"
    p3 = tmp_path / "result3.xlsx"
    t2 = np.arange(1, 121)
    C2 = np.column_stack([np.full(120, j / 100.0) for j in range(21)])
    T2 = np.full((120, 21), 40.0)
    W.write_result12(p2, t2, C2, T2, COLS21, A1)
    W.write_result34(p3, [60, 120], C2[[59, 119]], COLS21, A1)

    good = W.verify_result23_consistency(
        p2, p3, a1_text=A1, expected_cols=COLS21,
    )
    assert good["ok"], good["issues"]
    assert good["common_rows"] == 2
    assert good["common_cells"] == 42
    assert good["last_common_time"] == 120

    wb = openpyxl.load_workbook(p3)
    wb["Sheet1"]["B2"] = 9.9999
    wb.save(p3)
    wb.close()
    bad = W.verify_result23_consistency(
        p2, p3, a1_text=A1, expected_cols=COLS21,
    )
    assert not bad["ok"]
    assert any("result2=" in issue for issue in bad["issues"])


def test_result3_tail_after_result2_end_is_outside_common_range(tmp_path):
    p2 = tmp_path / "result2.xlsx"
    p3 = tmp_path / "result3.xlsx"
    t2 = np.arange(1, 121)
    C2 = np.column_stack([np.full(120, j / 100.0) for j in range(21)])
    T2 = np.full((120, 21), 40.0)
    W.write_result12(p2, t2, C2, T2, COLS21, A1)
    tail = np.r_[C2[[59, 119]], np.full((1, 21), 0.1499)]
    W.write_result34(p3, [60, 120, 180], tail, COLS21, A1)

    result = W.verify_result23_consistency(
        p2, p3, a1_text=A1, expected_cols=COLS21,
    )
    assert result["ok"], result["issues"]
    assert result["common_rows"] == 2
    assert result["last_common_time"] == 120
