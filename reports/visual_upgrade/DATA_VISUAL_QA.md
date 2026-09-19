# 数据图视觉与数据保真核验

范围：九张正文数据图；只读取原始输入、历史正式工作簿与既有验证记录，不运行数值求解器。
默认入口和 --data-only 均只重绘这九张图，不覆盖四问关系图或未入正文的潜热诊断图。

## 成品规范

- 所有 PDF/SVG 画布宽 160 mm；保存时不使用 tight 裁切。
- 正文、刻度、图例基础字号最小 8 pt；轴标签 9–9.5 pt，面板标题 10.5 pt。数学上下标按排版比例缩小。
- SVG 保留可编辑文字、线条、标记；大规模场网格作为嵌入栅格保存，避免数十万独立单元阻碍编辑。
- 水分场统一色标 0–2.55 kg/kg；温度场使用暖色。位置、时刻与方法用线型/标记冗余编码。
- 场来自历史四位小数输出；连续根和严格采样标记来自赛后复算事件，不对舍入场作高精度求根。

| 图 | 画布 / mm | 最小基础字号 / pt | 画布内文字 | 数据检查 |
|---|---:|---:|---|---|
| fig_inputs | 160.0 × 94.0 | 8 | 通过 | 通过 |
| fig_q1_profiles | 160.0 × 76.2 | 8 | 通过 | 通过 |
| fig_q23_fields | 160.0 × 147.3 | 8 | 通过 | 通过 |
| fig_q4_fields | 160.0 × 97.8 | 8 | 通过 | 通过 |
| fig_analytic_convergence | 160.0 × 78.7 | 8 | 通过 | 通过 |
| fig_convergence | 160.0 × 80.0 | 8 | 通过 | 通过 |
| fig_be_bdf | 160.0 × 88.9 | 8 | 通过 | 通过 |
| fig_threecase | 160.0 × 66.0 | 8 | 通过 | 通过 |
| fig_sensitivity | 160.0 × 104.1 | 8 | 通过 | 通过 |

## 逐图来源、缺失值与语义

### fig_inputs

0–4 h input samples and interpolation; 4–6.5 h assumed means drawn separately; radius stops at 72 h.

来源：附件/附件1.xlsx、附件/附件2.xlsx、config/A题_config.yaml。

核验：finite_input=True；no_radius_extrapolation=True。

### fig_q1_profiles

Four exact worksheet times; 21 radial output positions, not a dense solver field.

来源：outputs/result1.xlsx。

核验：exact_requested_times=True；finite_fields=True。

### fig_q23_fields

Field uses rounded 0.1 cm samples; moisture every 60 s; temperature every 6 s with first/last retained. Shared moisture scale 0–2.55. Contour is sampled, event markers are independently verified metadata.

来源：outputs/result2.xlsx、outputs/result3.xlsx、reports/post_contest/reproduction.json。

核验：field_finite=True；no_color_scale_clipping=True；temperature_endpoints=True；first_last_moisture=True；strict_sample_below_threshold=True；event_curve_not_interpolated=True。

### fig_q4_fields

Rounded samples linearly interpolated only inside R(t); outside NaN and gray cover; surface is its own moving coordinate. No smoothing/extrapolation beyond valid domain.

来源：outputs/result4.xlsx、附件/附件2.xlsx、config/A题_config.yaml、reports/post_contest/reproduction.json。

核验：outside_remains_nan=True；no_color_scale_clipping=True；first_last_rows=True；shared_moisture_scale=True；exact_strict_event=True。

### fig_analytic_convergence

Full radial maximum errors under constant-property analytic benchmark, not production four-question error.

来源：reports/V1_V2.md。

核验：same_source_points=True；reference_is_slope_only=True。

### fig_convergence

Historical N200/400/800, each method has its own N800 reference; zero self-difference omitted on log axis. Not the new D12 comparison.

来源：exports/convergence.csv。

核验：own_reference=True；zero_not_replaced=True。

### fig_be_bdf

Historical constant-property N400/100s benchmark; differences computed from saved history. Initial exact zero masked on log axis; BDF endpoint is an annotation, not a BE fixed-step datum.

来源：exports/be_bdf_diag.csv、exports/be_bdf_history.csv。

核验：finite_history=True；BE_endpoint_source=True；BDF_not_on_fixed_step_axis=True。

### fig_threecase

All three values selected at historical N400, including both conditional differences. No production N800 values mixed into the comparison.

来源：exports/fig10_diff.csv。

核验：all_three_cases=True；same_N400=True。

### fig_sensitivity

All source scenarios retained in report order (paired perturbations); small panel abs(delta)<0.4 h. Gray band is a screening scale, zero is rounded. Environmental offsets apply only after 4 h.

来源：reports/sensitivity.md。

核验：all_scenarios_retained=True；baseline_parsed=True；no_baseline_fallback=True。

## 事件值（只作独立注释）

| 问题 | 连续根 / s | 严格分钟采样 / s | 采样滞后 / s | 未舍入采样 Cmax |
|---|---:|---:|---:|---:|
| Q2/Q3 | 206906.496833186218 | 206940 | 33.503166813782 | 0.14999024303469236 |
| Q4 | 183931.302882989257 | 183960 | 28.697117010743 | 0.14998526745070481 |

## 保护边界

已复核 65 项受保护文件 SHA-256（必需受保护文件 61 项，历史交付物 4 项），与视觉升级前记录一致。
未随源码分发而不适用的历史交付物：无。
历史 deliverables/post_contest 文件缺失时仅记录不适用；若存在则必须匹配保护哈希。其余受保护文件缺失或变动均阻断核验。
不改变历史数据、物理模型、输入、求解器或新的 D12 证据；历史 N=400 辅助图与正式 N=800 主线明确区分。
解析基准和相对最细网格变化不被表述为真实解误差界；灵敏度灰带不是置信区间。

## 逐图视觉复核（agent）

本次由 agent 逐张打开九张 PNG 成品检查；修正后重新查看 Q1、Q23、Q4、BE/BDF、灵敏度图。此项为 agent 视觉复核，不代表用户或参赛队已审核。结论如下。

| 图 | 视觉复核结果 |
|---|---|
| inputs | 环境上下双面板与右侧半径图阅读顺序清楚；原始圆点可辨，4 h 后均值延续段以阴影、虚线明确标示，未跨断点平滑。 |
| q1_profiles | 四个时刻在两个面板采用相同颜色、线型、标记；水分图例已移至左下空白，避免覆盖平台。 |
| q23_fields | 两场色条与轴标签已分开；阈值窗口只画舍入采样点和独立精确事件标记，不将平台连成假精确穿越；图例加白底避免根参考线穿字。 |
| q4_fields | 浅灰域外与低含水率浅蓝可辨，边界完整；域外保持 NaN，表面单列和固定位置曲线含义明确；色条无截断。 |
| analytic_convergence | 对数刻度只保留所用网格节点；二阶参考斜率线平移后不遮盖测量误差点，未改变数据或拟合斜率。 |
| convergence | 两类界面颜色、标记、线型一致；右图只比较各自 N=800 参考，零差值未伪造为正数。 |
| be_bdf | 历史温度差时程展示各方法实际差异；右图仅 BE 步长数据，BDF 终点另作文字注释；指数改用数学字体，无缺字。 |
| threecase | 三行情景与条件差值全部为历史 N=400；标记、数值与路径含义清楚，非顺序干燥轨迹。 |
| sensitivity | 十种情景按成对顺序保留；负值数码已留足左边界；颜色表示时长变化方向，灰带为筛选尺度，零变化带舍入星号。 |

独立文件核验：用 pdfplumber 读取九份 PDF，均为单页且宽 160.000000 mm；解析九份 SVG，宽均为 160 mm，分别保留 37、36、66、43、21、28、35、14、46 个文字节点（按上表顺序）。

轻量保真核验：事件加载器的根、采样时刻、未舍入最大含水率与 reproduction.json 逐值相同；模拟缺失灵敏度基线时明确抛出 ValueError；65 项受保护文件最终 SHA-256 均未变化。未重跑数值求解或全量测试。

可移植性小修核验：调用生产保护核验函数，在临时目录验证五种情景——旧交付物缺失时列为未分发且允许执行；旧交付物存在且未变时校验通过；现存旧交付物被改动时拒绝；必需数据源缺失时拒绝；必需数据源被改动时拒绝。本机复核仍为 61 项必需文件及 4 项历史交付物全部通过，无不适用项。此次仅修改 QA 逻辑与报告，未重绘 27 个图资产。

## 来源 SHA-256

| 源文件 | SHA-256 |
|---|---|
| config/A题_config.yaml | ee73bc0aff14273f71fc832fd1c361ad33f507add199a9ad57a88646bebfee73 |
| exports/be_bdf_diag.csv | 5901042295e868799c59a9f7919d44cc23fd13f5212b422a0a4d661d9747a0d7 |
| exports/be_bdf_history.csv | 8da33cf8e380a934c02f869528904b7212850b6ba90451b330d5e9141004be1c |
| exports/convergence.csv | 24687e46faabca53bd0c00d4ce4f7b9082fedd84e94308869a627615112be15a |
| exports/fig10_diff.csv | 63bf8aa25bf8e3a30877643a1fed17bb872365ca2fd514fe27547934024c8603 |
| outputs/result1.xlsx | b7270ed1b5eec10ace4a0ef2d70ef7294e4ef92e792353c232f010502f7ccdee |
| outputs/result2.xlsx | 387542f8c9ca09d0dd1af6e3547d47b54d90b0f69b233d4e5943cc13e17de675 |
| outputs/result3.xlsx | 6371fbe98dc5f8e3669c15fe537692e36ea9fe09719af8da81d9fab3c1797ecd |
| outputs/result4.xlsx | fb55c152c5c86d693e0342162f78abb25b0011df98fccec5d0b2093ac119f361 |
| reports/V1_V2.md | 47e14d50a05d2256c6ed4e3e6ff36c2ee21c4a79d527307dd48eca5c79c83825 |
| reports/post_contest/reproduction.json | e95836ea0549bc51d99e2562b508a6681050e03a84c894bc99e0d2c33b5f8372 |
| reports/sensitivity.md | a51b745f7ce6ddd0e9a2906d2cb04e37e1535cfb36950783f6f92699cd10d58a |
| 附件/附件1.xlsx | 7ef32870abeef420b89560b2530ff60dfe4255917805151d89988d0311af9dd7 |
| 附件/附件2.xlsx | 5563acbfa4b4afb10cc6c03e2207e5369bf39da27576672aff14cc5c32e704af |

绘图源 SHA-256：1e3f8ac430e73a81bc4c9eb0f3335a8633b8ed51b9e39e41084f6ecda777354d
运行环境：Matplotlib 3.10.6；NumPy 2.3.5；openpyxl 3.1.5。

命令：D:/Anaconda/python.exe -X utf8 -m drymodel.paper_figs --data-only（PYTHONPATH=src）。
