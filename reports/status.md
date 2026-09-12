# 状态汇总（status.md）

生成时间：2026-09-12 10:35:06；软件：Python 3.13.9

> 状态词：已修复并实测通过 / 仍未执行 / 仍失败 / 待用户授权。**由实测阈值判定，不用 pf(True)、不以 t*>0 判收敛、不以 approved 判文件已生成。**

## 阶段状态（实测）

| 项 | 状态 | 实测证据 |
|---|---|---|
| 配置校验（拒八类变异+类型/有限性、-O 不失效、两检查器一致） | 已修复并实测通过 | test_checker_parity 15 例 |
| V-1/V-2 空间二阶、时间一阶 | 已修复并实测通过 | reports/V1_V2.md |
| V-3 分对象（早期表面/通量均量/t*空间与时间/Q4固定厘米含 r=1.2cm） | 已实测分对象登记（最终 N 见 D12 申请） | reports/V3_convergence.md；Q4 r=1.2cm 相邻差 1.06e-03 |
| V-4a 离散代数恒等式（非时间精度） | 已修复并实测通过 | rel 1.5e-14/3.1e-14（<1e-12） |
| V-4b BE 独立梯形（粗步未过/细步通过，1e-4 门槛） | 已修复并实测通过 | 粗步 Δt=1 rel 1.64e-04 > 1e-04（粗步未过）；细步 Δt=0.25 rel 4.11e-05 ≤ 1e-04（细步通过） |
| V-4b BDF 独立连续自适应求积（10×rtol） | 已修复并实测通过 | rel 9.33e-09 ≤ 1e-07 |
| V-4c 有效热残差（W/m，细步安全） | 已修复并实测通过 | rel 1.0e-10（<1e-08） |
| V-5 物理界限包络（H17/H18，接受解） | 已修复并实测通过 | 固定域/动域均无越界 |
| V-10 静态极限 | 已修复并实测通过 | rel 3.6e-08 |
| V-11 integral 界面网格收敛 | 已修复并实测通过 | N400→800 t* 差 8.78e-05 h（<0.02 h） |
| V-6 判据合成回归 | 已修复并实测通过 | test_criterion 0.5 s 真根 |
| S10 附录4 固定半径对照 | 已实测（三情形登记） | reports/verification.md |
| 灵敏度 S1–S6 | 已实测（见报告） | reports/sensitivity.md |
| V-13 Q4 守恒（1/R 收支恒等式） | 已修复并实测通过 | test_balances 增广态 |
| **D12 生产配置授权** | 已授权 | production.approved=True |
| 正式 result1–4 官方导出 | 已生成（磁盘存在） | 由 outputs/ 磁盘实际存在判定（非 approved） |
| 生产编排 run_production | 已实现（--produce 门控，见 test_production 缩比测试） | run_all.run_production |
| V-8/V-9 官方文件终检 | 已实现（导出后执行，非 D12 前置） | writers.verify_workbook/cross_file_check |
| 端面校核 V-15 | 已实测通过 | reports/V15_end_effect.md（max|ΔT|=9.10e-4 °C@1.5h、max|ΔC|=1.13e-5@42h） |
| 论文数值/文本、AI 使用详情 | 待人工核验 | — |

## 正式答案（已授权生产、官方结果）

| 量 | 值 |
|---|---|
| Q2/Q3 达标时长 t* | 57.4740 h（N=800, integral） |
| Q4 达标时长 t* | 51.0920 h（N=800, integral） |
| S10 ②附录4/R0 t* | 129.8487 h |

> 上表为已授权生产配置下的正式结果（config_id D12-20260912-N1600.800.800-integral8pt-bdf1e-8），与 outputs/result1–4、论文表 1–6 同源。
> 配置快照见 exports/config_snapshot.json。