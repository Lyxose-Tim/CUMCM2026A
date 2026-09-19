# 第 1 轮验证

命令：`python -X utf8 -m pytest tests/test_candidate_trajectory.py tests/test_solver_boundaries.py tests/test_production_failures.py tests/test_checker_parity.py tests/test_solvers.py tests/test_balances.py -q -o addopts=''`。

结果：65 passed in 14.99s。

旧实现独立复现：dt=1，t_end=2.4 实际结束于2.0；t_end=2.6 实际结束于3.0；record_times=[0.25,0.5,0.75,1.0] 仅保留1.0。修复后精确落在请求终点/采样/断点，累计通量按实际子步累加。

候选 Q23/Q4 改为事件与续算拼接的唯一轨迹；缺少60s严格合格采样即失败。BDF 接受的物理状态检查有限性和正性，辅助累计通量允许零。生产门检同时检查原始数值与状态标志，不容许NaN绕过比较。

基线遗留13项失败来自赛时 `建模方案v1.1/config_check.py` 与运行检查器不同。该文件为用户未提交修改，保持其内容。将测试的交付副本定位改到真实论文工程的 `paper/code/config_check.py`，原负例（错误物性、输出截断、NaN、布尔网格等）全部保留并通过。

临时完整输出：`_tmp/post_contest/baseline_tests.log`、`_tmp/post_contest/round1_tests.log`。未把历史数值报告记为本轮重算；全问题重算安排在第二轮。
