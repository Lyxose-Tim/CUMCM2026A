# CUMCM 2026 A题「药材的烘干问题」求解工程

一维径向（中截面）有效扩散—导热模型的 Q1–Q4 建模求解代码。实现严格对齐冻结方案
《建模方案v1.1/A题_建模方案.md》v1.2，参数唯一来源为 [`config/A题_config.yaml`](config/A题_config.yaml)。

> **状态**：候选计算 + 验证阶段（方案 §9.4 步 0–3）。生产配置 D12 **待用户授权**；
> 生产续算/正式导出编排仍未实现，正式 result1–4 尚未生成。授权标志不等于已经计算或导出。
> **四位小数显示 ≠ 末位精度保证。** 实际检查状态见 `reports/verification.json` 与 `reports/status.md`。

## 环境

```bash
pip install -e .        # 或 pip install -r requirements.txt 后设 PYTHONPATH=src
```

实测环境：Python 3.13.9、numpy 2.3.5、scipy 1.16.3、openpyxl 3.1.5、PyYAML 6.0.3、matplotlib。

## 目录

| 路径 | 内容 |
|---|---|
| `src/drymodel/` | 主 Python 包（模块见下） |
| `config/A题_config.yaml` | 运行时唯一权威配置；方案目录同名 YAML 保持逐字镜像 |
| `tests/` | pytest 单元测试 |
| `cache/candidates/` | 候选计算缓存（不入库） |
| `outputs/` | 正式 result1–4（待 D12 授权后生成） |
| `exports/` | 供 MATLAB 读取的 CSV |
| `reports/` | 验证 / 收敛 / 灵敏度 / 状态报告 |
| `figs/`, `matlab/` | matplotlib 图 与 MATLAB 出图脚本 |
| `附件/`, `A题.pdf` | 原始材料（**只读，不改**） |
| `建模方案v1.1/` | 方案、决策日志及权威检查器的薄入口 |

### 模块（对齐方案 §9.1）

`config` · `config_check` · `data_io` · `props` · `grid` · `operators` · `solver_be` ·
`solver_bdf` · `criterion` · `postprocess` · `writers` · `verify` · `sensitivity` ·
`plots` · `run_all`。物理状态排列为 `y=[C_0..C_N, T_0..T_N]`；V-4/BDF 候选收支检查
启用累计通量时为 `y=[C_0..C_N, T_0..T_N, I]`，`I` 不反馈物理方程。

已实现：候选求解、严格有界同源轨迹、空间/时间/求积对照、BE/BDF 收支、包络与事件检查、
工作簿及 result2/3 跨文件检查器。仍未执行：正式文件的 V-8/V-9 读回验收；仍未实现：生产续算与正式导出编排。

## 运行

```bash
python -m pytest tests/ -q                      # 单元测试
python -m drymodel.config_check config/A题_config.yaml   # 权威配置校验（普通/-O 共用显式异常）
python -m drymodel.run_all                      # 候选计算 + 验证（步 0–3）
python -m drymodel.run_all --produce            # 当前只执行授权门控；生产编排未实现
```

## 建模要点（摘）

- 节点型径向 FVM（散度形式，不改写为 `D∇²C`）；中心节点 0、表面节点 N。
- 时间积分：后向 Euler（验证基线）+ BDF（长时生产候选，稠密输出采样）。
- 界面系数：调和平均（默认）与积分界面（8 点 Gauss，16 点对照，优先待验收候选）。
- 达标判据：未舍入全节点场上求根 `max_i C_i(t) = 0.15`；事件后 600 s 检查是数值异常检查，不冒称全未来数学证明。
- Q4：参考坐标 `x=r/R(t)`，仿射收缩使方程只含 `R(t)`；域外输出留空。
- 判据与验证一律用未舍入值，仅写盘时 `round(·,4)`。

## AI 使用

代码实现曾由 Claude Code 辅助，本次候选审计返修由 OpenAI Codex 辅助；方案设计与决策见
`建模方案v1.1/`。正式数值与论文文本仍须参赛队人工核验。
