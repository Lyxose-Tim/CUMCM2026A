"""writers.py —— result1–4 写盘与写后读回校验（§9.5、V-9）。

- openpyxl write_only 逐行写入，数值型 + `0.0000` 格式，仅写盘时 round(·,4)。
- result1/2：工作表 温度/水分浓度；result3/4：Sheet1。A1=`时间\\到药材中心的距离`。
- result4：列 0..1.9 + 「药材表面」；域外单元格留空（None→空）。
- 数据行 ≤1,048,575（含表头 ≤1,048,576），超限抛错，不截断、不改模板。
- verify_workbook：读回校验表名/A1/表头/A 列连续无重复/行数/掩码。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import openpyxl
from openpyxl.cell import WriteOnlyCell

from .postprocess import round4

EXCEL_MAX_ROWS = 1048576
NUMFMT = "0.0000"


def _num_cell(ws, value):
    """数值型单元格（round4 + 0.0000 格式）；None → 空单元格。"""
    if value is None:
        return None
    if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f"输出数据须为数值或 None，得 {value!r}")
    if not np.isfinite(value):
        raise ValueError(f"输出数据须为有限数值，得 {value!r}")
    c = WriteOnlyCell(ws, value=round4(value))
    c.number_format = NUMFMT
    return c


def _check_row_budget(n_data_rows: int, path):
    if n_data_rows + 1 > EXCEL_MAX_ROWS:
        raise ValueError(f"{path}: 数据行 {n_data_rows} 超出上限 {EXCEL_MAX_ROWS - 1}")


def write_result12(path, t_array, C_grid, T_grid, cols_cm, a1_text):
    """result1/2：两工作表（温度、水分浓度）。C_grid/T_grid 形状 (len(t), len(cols))。"""
    path = Path(path)
    t_array = np.asarray(t_array)
    _check_row_budget(len(t_array), path)
    wb = openpyxl.Workbook(write_only=True)
    for sheet_name, grid, is_temp in (("温度", T_grid, True), ("水分浓度", C_grid, False)):
        ws = wb.create_sheet(sheet_name)
        ws.append([a1_text] + [float(c) for c in cols_cm])
        for i, t in enumerate(t_array):
            row = [int(round(t))] + [_num_cell(ws, grid[i][j]) for j in range(len(cols_cm))]
            ws.append(row)
    wb.save(path)


def write_result34(path, t_array, value_grid, cols_cm, a1_text, *,
                   sheet_name="Sheet1", surface_header=None, surface_values=None):
    """result3/4：单工作表（水分浓度）。value_grid 可含 None（域外留空）。

    result4 传 surface_header='药材表面' 与 surface_values（长度 len(t)）。
    """
    path = Path(path)
    t_array = np.asarray(t_array)
    _check_row_budget(len(t_array), path)
    wb = openpyxl.Workbook(write_only=True)
    ws = wb.create_sheet(sheet_name)
    header = [a1_text] + [float(c) for c in cols_cm]
    if surface_header is not None:
        header.append(surface_header)
    ws.append(header)
    for i, t in enumerate(t_array):
        row = [int(round(t))] + [_num_cell(ws, value_grid[i][j]) for j in range(len(cols_cm))]
        if surface_header is not None:
            row.append(_num_cell(ws, surface_values[i]))
        ws.append(row)
    wb.save(path)


def verify_workbook(path, *, expected_sheets, a1_text, expected_cols, n_data_rows,
                    t_start, t_step, surface_header=None, mask=None):
    """流式写后读回校验（V-9）。

    ``mask[i][j]=True`` 表域内且必须为有限数值；False 表域外且必须为空。
    表面列不属于 mask，若声明则每行始终必须为有限数值。
    """
    path = Path(path)
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    issues = []

    def issue(message):
        if len(issues) < 100:
            issues.append(message)

    def numeric(value):
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and np.isfinite(value))

    def header_equal(got, expected):
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            return numeric(got) and abs(float(got) - float(expected)) <= 1e-12
        return got == expected

    if list(wb.sheetnames) != list(expected_sheets):
        issue(f"工作表名 {wb.sheetnames} != {expected_sheets}")
    for sn in wb.sheetnames:
        ws = wb[sn]
        exp_header = tuple([a1_text] + list(expected_cols) +
                           ([surface_header] if surface_header else []))
        rows = ws.iter_rows(values_only=False)
        try:
            header_cells = next(rows)
        except StopIteration:
            issue(f"{sn}: 工作表为空")
            continue
        got_header = tuple(cell.value for cell in header_cells)
        if len(got_header) != len(exp_header):
            issue(f"{sn}: 表头列数 {len(got_header)} != {len(exp_header)}")
        for j, expected in enumerate(exp_header):
            got = got_header[j] if j < len(got_header) else None
            if not header_equal(got, expected):
                issue(f"{sn}: 表头第{j + 1}列 {got!r} != {expected!r}")

        count = 0
        first_time = None
        last_time = None
        for row in rows:
            count += 1
            row_idx = count - 1
            expected_time = int(t_start) + row_idx * int(t_step)
            t_value = row[0].value if row else None
            if type(t_value) is not int:
                issue(f"{sn}: 数据行{count} A列须为真实整数，得 {t_value!r}")
            elif t_value != expected_time:
                issue(f"{sn}: 数据行{count} A列 {t_value} != {expected_time}")
            if count == 1:
                first_time = t_value
            last_time = t_value

            if mask is not None and row_idx >= len(mask):
                issue(f"{sn}: 掩码缺少数据行 {count}")
                row_mask = [True] * len(expected_cols)
            else:
                row_mask = ([True] * len(expected_cols) if mask is None
                            else list(mask[row_idx]))
            if len(row_mask) != len(expected_cols):
                issue(f"{sn}: 数据行{count} 掩码列数 {len(row_mask)} != {len(expected_cols)}")

            for j in range(len(expected_cols)):
                cell = row[j + 1] if j + 1 < len(row) else None
                value = None if cell is None else cell.value
                should_be_inside = bool(row_mask[j]) if j < len(row_mask) else True
                if should_be_inside:
                    if not numeric(value):
                        issue(f"{sn}: 数据行{count} 列{j + 2} 域内值须为有限数值，得 {value!r}")
                    elif cell.number_format != NUMFMT:
                        issue(f"{sn}: 数据行{count} 列{j + 2} 格式 {cell.number_format!r} != {NUMFMT!r}")
                elif value is not None:
                    issue(f"{sn}: 数据行{count} 列{j + 2} 域外应为空，得 {value!r}")

            if surface_header is not None:
                surface_idx = 1 + len(expected_cols)
                cell = row[surface_idx] if surface_idx < len(row) else None
                value = None if cell is None else cell.value
                if not numeric(value):
                    issue(f"{sn}: 数据行{count} 表面列须为有限数值，得 {value!r}")
                elif cell.number_format != NUMFMT:
                    issue(f"{sn}: 数据行{count} 表面列格式 {cell.number_format!r} != {NUMFMT!r}")

        if count != n_data_rows:
            issue(f"{sn}: 数据行 {count} != {n_data_rows}")
        expected_last = int(t_start) + (n_data_rows - 1) * int(t_step) if n_data_rows else None
        if n_data_rows and (first_time != int(t_start) or last_time != expected_last):
            issue(
                f"{sn}: A列首末值 ({first_time!r}, {last_time!r}) != "
                f"({int(t_start)!r}, {expected_last!r})"
            )
        if mask is not None and len(mask) != n_data_rows:
            issue(f"{sn}: 掩码行数 {len(mask)} != {n_data_rows}")
    wb.close()
    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "path": str(path),
        "expected_rows": int(n_data_rows),
        "expected_last_time": (
            int(t_start) + (n_data_rows - 1) * int(t_step) if n_data_rows else None
        ),
    }


def verify_result23_consistency(result2_path, result3_path, *, a1_text,
                                expected_cols, result3_sheet="Sheet1"):
    """流式比较 result2/3 的共同时间、共同位置四位舍入结果。"""
    wb2 = openpyxl.load_workbook(result2_path, read_only=True, data_only=True)
    wb3 = openpyxl.load_workbook(result3_path, read_only=True, data_only=True)
    issues = []
    common_rows = 0
    common_cells = 0
    last_common_time = None
    last_result2_time = None

    def issue(message):
        if len(issues) < 100:
            issues.append(message)

    try:
        if "水分浓度" not in wb2.sheetnames:
            issue("result2 缺少水分浓度工作表")
            return {"ok": False, "issues": issues, "common_rows": 0,
                    "common_cells": 0, "last_common_time": None}
        if result3_sheet not in wb3.sheetnames:
            issue(f"result3 缺少 {result3_sheet} 工作表")
            return {"ok": False, "issues": issues, "common_rows": 0,
                    "common_cells": 0, "last_common_time": None}

        rows2 = wb2["水分浓度"].iter_rows(values_only=True)
        rows3 = wb3[result3_sheet].iter_rows(values_only=True)
        expected_header = tuple([a1_text] + list(expected_cols))
        header2 = tuple(next(rows2, ()))
        header3 = tuple(next(rows3, ()))

        def header_matches(header):
            if len(header) != len(expected_header):
                return False
            for got, expected in zip(header, expected_header):
                if isinstance(expected, (int, float)) and not isinstance(expected, bool):
                    if not (isinstance(got, (int, float)) and not isinstance(got, bool)
                            and np.isfinite(got)
                            and abs(float(got) - float(expected)) <= 1e-12):
                        return False
                elif got != expected:
                    return False
            return True

        if not header_matches(header2):
            issue(f"result2 水分表头不匹配：{header2!r}")
        if not header_matches(header3):
            issue(f"result3 表头不匹配：{header3!r}")

        row3 = next(rows3, None)
        for row2 in rows2:
            t2 = row2[0] if row2 else None
            if type(t2) is int:
                last_result2_time = t2
            if row3 is None:
                continue
            t3 = row3[0] if row3 else None
            if not (type(t2) is int and type(t3) is int):
                issue(f"共同时间须为真实整数，得 result2={t2!r}, result3={t3!r}")
                break
            while row3 is not None and type(row3[0]) is int and row3[0] < t2:
                t3 = row3[0]
                issue(f"result3 时间 {t3} 在 result2 中缺失")
                row3 = next(rows3, None)
            if row3 is None:
                continue
            t3 = row3[0]
            if type(t3) is not int:
                issue(f"result3 时间须为真实整数，得 {t3!r}")
                break
            if t2 < t3:
                continue

            common_rows += 1
            last_common_time = t2
            for j in range(len(expected_cols)):
                v2 = row2[j + 1] if j + 1 < len(row2) else None
                v3 = row3[j + 1] if j + 1 < len(row3) else None
                if not (isinstance(v2, (int, float)) and not isinstance(v2, bool)
                        and isinstance(v3, (int, float)) and not isinstance(v3, bool)
                        and np.isfinite(v2) and np.isfinite(v3)):
                    issue(f"t={t2}s, r={expected_cols[j]}cm：共同值不是有限数值")
                elif float(v2) != float(v3):
                    issue(
                        f"t={t2}s, r={expected_cols[j]}cm："
                        f"result2={v2!r} != result3={v3!r}"
                    )
                common_cells += 1
            row3 = next(rows3, None)

        if (row3 is not None and type(row3[0]) is int
                and last_result2_time is not None and row3[0] <= last_result2_time):
            issue(f"result3 在共同时间范围内仍有未匹配行，首个时间 {row3[0]!r}")
        if common_rows == 0:
            issue("result2/3 没有共同数据行")
    finally:
        wb2.close()
        wb3.close()

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "common_rows": common_rows,
        "common_cells": common_cells,
        "last_common_time": last_common_time,
    }
