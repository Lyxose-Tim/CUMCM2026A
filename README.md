# CUMCM 2026 A题「药材的烘干问题」求解工程

一维径向（中截面）有效扩散—导热模型的 Q1–Q4 建模求解代码。实现严格对齐冻结方案
《建模方案v1.1/A题_建模方案.md》v1.2，参数唯一来源为 [`config/A题_config.yaml`](config/A题_config.yaml)。

> **状态**：候选计算 + 验证阶段（方案 §9.4 步 0–3）。生产配置 D12 **待用户授权**；
> 正式 result1–4 与表 1–6 的官方数值在授权后由生产续算生成。**四位小数显示 ≠ 末位精度保证。**

## 环境

```bash
pip install -e .        # 或 pip install -r requirements.txt 后设 PYTHONPATH=src
```

实测环境：Python 3.13.9、numpy 2.3.5、scipy 1.16.3、openpyxl 3.1.5、PyYAML 6.0.3、matplotlib。

## 目录

| 路径 | 内容 |
|---|---|
| `src/drymodel/` | 主 Python 包（模块见下） |
| `config/A题_config.yaml` | 冻结配置（复制自 `建模方案v1.1/`，唯一真值源） |
| `tests/` | pytest 单元测试 |
| `cache/candidates/` | 候选计算缓存（不入库） |
| `outputs/` | 正式 result1–4（待 D12 授权后生成） |
| `exports/` | 供 MATLAB 读取的 CSV |
| `reports/` | 验证 / 收敛 / 灵敏度 / 状态报告 |
| `figs/`, `matlab/` | matplotlib 图 与 MATLAB 出图脚本 |
| `附件/`, `建模方案v1.1/`, `A题.pdf` | 原始材料（**只读，不改**） |

### 模块（对齐方案 §9.1）

`config` · `config_check` · `data_io` · `props` · `grid` · `operators` · `solver_be` ·
`solver_bdf` · `criterion` · `postprocess` · `writers` · `verify` · `sensitivity` ·
`plots` · `run_all`。状态排列 `y=[C_0..C_N, T_0..T_N]`。

## 运行

```bash
python -m pytest tests/ -q                      # 单元测试
python -m drymodel.config_check config/A题_config.yaml   # 配置校验（八类变异+类型/有限性，-O 不失效）
python -m drymodel.run_all                      # 候选计算 + 验证（配置驱动），出 reports/ 与 exports/config_snapshot.json
python -m drymodel.run_all --produce            # 生产续算与正式导出（正式 result1–4 已由该步生成，见 outputs/ 与 reports/production_receipt.md）
```

> `run_all` 候选模式实测生成 `reports/{V1_V2,V3_convergence,verification,sensitivity,status}.md`
> 与 `exports/config_snapshot.json`（生效配置快照）。N/界面/求积点/容差/情景由
> `config/A题_config.yaml` 的 `numerics.candidate` 与 `per_question` 解析生效。

**D12 已授权并完成正式生产**（final_N 1600/800/800、integral 8 点、BDF rtol 1e-8）：

```bash
python -m drymodel.d12_evidence   # D12 可追溯证据（全部正式采样场点，任一超差→非零退出）
python -m drymodel.d12_prep       # D12 配置申请（已授权，存档）
python -m drymodel.run_all --produce   # 正式 result1–4 + 表 1–6 + V-8/V-9（已执行，见 outputs/）
python -m drymodel.paper          # 论文表 1–6 与文本 → reports/paper_tables.md
python -c "from drymodel import config,verify; verify.run_v15(config.load_config())"  # V-15 端面校核
```

- 正式产出：`outputs/result1–4.xlsx` + `table{1..6}_*.csv`；回执 `reports/production_receipt.md`、
  `outputs/production_receipt_supplementary.json`（失败也记 `production_failures.log`）。
- 未舍入 t*₃=57.474027 h、t*₄=51.092029 h；t* 与严格合格采样 t_sample 的区分见 `reports/paper_tables.md`。

## 建模要点（摘）

- 节点型径向 FVM（散度形式，不改写为 `D∇²C`）；中心节点 0、表面节点 N。
- 时间积分：后向 Euler（验证基线）+ BDF（长时生产候选，稠密输出采样）。
- 界面系数：调和平均（默认）与积分界面（8 点 Gauss，16 点对照，优先待验收候选）。
- 达标判据：未舍入全域重构场上求根 `max_i C_i(t) = 0.15`；不回穿依据比较原理。
- Q4：参考坐标 `x=r/R(t)`，仿射收缩使方程只含 `R(t)`；域外输出留空。
- 判据与验证一律用未舍入值，仅写盘时 `round(·,4)`。

## AI 使用

代码实现由 Claude Code 辅助完成；方案设计与决策见 `建模方案v1.1/`。正式数值与论文文本须参赛队人工核验。
