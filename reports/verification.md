# 验证报告（候选阶段）

> 状态由 `reports/verification.json` 的实际指标生成。局部检查、未运行和失败不会被提升为通过。
> 本报告不构成 D12 授权；正式 result1–4 尚未生成。

## 验证状态总表

| 检查 | 问题 | 状态 | 覆盖 |
|---|---|---|---|
| C-1-config | global | 已实测通过 | 见 JSON |
| V-1-spatial | q1-heat | 已实测通过 | 见 JSON |
| V-2-spatial | q1-mass | 已实测通过 | 见 JSON |
| V-1-temporal | q1-heat | 已实测通过 | 见 JSON |
| V-2-temporal | q1-mass | 已实测通过 | 见 JSON |
| V-3-q1-space | q1 | 已实测通过 | comparison=space, rows=1800, time_start_s=1.0, time_end_s=1800.0 |
| V-3-q1-time | q1 | 已实测通过 | comparison=time, rows=1800, time_start_s=1.0, time_end_s=1800.0 |
| V-3-q23-space | q23 | 已实测通过 | comparison=space, rows=206909, time_start_s=1.0, time_end_s=206940.0 |
| V-3-q23-time | q23 | 已实测通过 | comparison=time, rows=206909, time_start_s=1.0, time_end_s=206940.0 |
| V-3-q4-space | q4 | 已实测通过 | comparison=space, rows=3067, time_start_s=60.0, time_end_s=183960.0 |
| V-3-q4-time | q4 | 已实测通过 | comparison=time, rows=3067, time_start_s=60.0, time_end_s=183960.0 |
| V-4-BDF | q23 | 已实测通过 | t_end_s=207506.49687158727, moving_domain=False |
| V-4-BDF | q4 | 已实测通过 | t_end_s=184531.30085372578, moving_domain=True |
| V-5 | q1 | 已实测通过 | t_end_s=1800.0 |
| V-5 | q23 | 已实测通过 | t_end_s=207506.4981430637 |
| V-5 | q4 | 已实测通过 | t_end_s=184531.30572580043 |
| V-6 | q23 | 已实测通过 | 见 JSON |
| V-6 | q4 | 已实测通过 | 见 JSON |
| V-10 | q4 | 已实测通过 | comparison=R(t)=R0 reference-domain vs fixed-domain |
| V-13 | q4 | 已实测通过 | moving_domain=True |
| V-11-q1-quadrature | q1 | 已实测通过 | comparison=interface_quadrature, rows=1800, time_start_s=1.0, time_end_s=1800.0 |
| V-11-q23-quadrature | q23 | 已实测通过 | comparison=interface_quadrature, rows=206909, time_start_s=1.0, time_end_s=206940.0 |
| V-11-q4-quadrature | q4 | 已实测通过 | comparison=interface_quadrature, rows=3067, time_start_s=60.0, time_end_s=183960.0 |
| V-4a | fixed | 已实测通过 | accepted_substeps=1800, moving_domain=False |
| V-4b | fixed | 仍失败 | accepted_substeps=1800, moving_domain=False |
| V-4a | fixed | 已实测通过 | accepted_substeps=7200, moving_domain=False |
| V-4b | fixed | 已实测通过 | accepted_substeps=7200, moving_domain=False |
| V-4c | fixed | 已实测通过 | accepted_substeps=100, moving_domain=False |
| V-4c | fixed | 已实测通过 | accepted_substeps=200, moving_domain=False |
| V-11-interface-duration | q23 | 局部通过 | 见 JSON |
| V-8 | production | 仍未执行 | 见 JSON |
| V-9 | production | 仍未执行 | 见 JSON |

## 全输出对象空间/时间/求积对照

| 检查 | 状态 | max ΔC | max ΔT / °C | Δt* / h | ΔC 最差位置 |
|---|---|---:|---:|---:|---|
| V-3-q1-space | 已实测通过 | 1.89906494e-04 | 1.92174390e-04 |  | {'time_s': 1.0, 'position': 2.0} |
| V-3-q1-time | 已实测通过 | 2.51087006e-07 | 2.03199379e-04 |  | {'time_s': 2.0, 'position': 2.0} |
| V-3-q23-space | 已实测通过 | 1.54881207e-04 | 2.25045145e-04 | 2.22628448e-05 | {'time_s': 1.0, 'position': 2.0} |
| V-3-q23-time | 已实测通过 | 3.50895836e-07 | 2.27408475e-04 | 3.53187900e-07 | {'time_s': 11161.0, 'position': 2.0} |
| V-3-q4-space | 已实测通过 | 2.19411423e-04 | 2.62441170e-04 | 2.21064022e-05 | {'time_s': 124620.0, 'position': 1.2} |
| V-3-q4-time | 已实测通过 | 6.51698755e-07 | 2.48929782e-04 | 1.35335407e-06 | {'time_s': 7140.0, 'position': 'surface'} |
| V-11-q1-quadrature | 已实测通过 | 2.15587546e-07 | 1.50824998e-04 |  | {'time_s': 1381.0, 'position': 2.0} |
| V-11-q23-quadrature | 已实测通过 | 3.83228602e-07 | 2.13578165e-04 | 1.83000552e-07 | {'time_s': 13201.0, 'position': 2.0} |
| V-11-q4-quadrature | 已实测通过 | 7.38988471e-07 | 2.56729404e-04 | 3.72460764e-06 | {'time_s': 11160.0, 'position': 'surface'} |

## V-4 BE 粗细步与热残差

| 项 | dt / s | 指标 | 限值 | 状态 |
|---|---:|---:|---:|---|
| V-4b 独立梯形通量 | 1 | 1.6426411657e-04 | 1.0e-04 | 仍失败 |
| V-4b 独立梯形通量 | 0.25 | 4.1066774423e-05 | 1.0e-04 | 已实测通过 |
| V-4c 有效热残差 | 1 | 3.2124392035e-10 W/m；rel=1.0225511573e-10 | abs≤1.0e-06 且 rel≤1.0e-08 | 已实测通过 |
| V-4c 有效热残差 | 0.5 | -1.6507195610e-10 W/m；rel=5.2544035560e-11 | abs≤1.0e-06 且 rel≤1.0e-08 | 已实测通过 |

V-4b 实测粗细步比值：3.99992741（dt 缩小 4 倍）；对应收敛阶 0.999987。粗步失败状态保留，细步独立判定通过。

## 候选有效配置

| 问题 | N | 配置摘要 |
|---|---:|---|
| q1 | 800 | `028adb8ba47d8561` |
| q23 | 800 | `54cc2666b51a991f` |
| q4 | 800 | `c11eb5b2c61c4c53` |

## 未通过、局部与未运行记录

- V-4b / fixed：仍失败；详见机器可读 metric/limit/coverage
- V-11-interface-duration / q23：局部通过；仅证明 Q23 时长趋势；不替代全场或 Q4 检验
- V-8 / production：仍未执行；正式 result2/3 未生成；V-8 跨文件一致性按计划留到 D12 后
- V-9 / production：仍未执行；正式工作簿未生成；V-9 检查器已实现但尚无正式文件可验

## D12 前候选门控

- 全局配置证据：通过
- q1: 适用数值检查已通过
- q23: 适用数值检查已通过
- q4: 适用数值检查已通过

总状态：**可提交用户考虑 D12 授权，但本次仍未授权**。

机器可读的每项 metric/limit/config/coverage/worst_location 见 `reports/verification.json`。
