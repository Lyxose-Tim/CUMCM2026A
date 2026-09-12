# 论文 LaTeX 工程编译说明（Overleaf / 本地）

## 工程结构（相对路径，可直接导入 Overleaf）

```
paper/
├── main.tex              # 主文件（\input 各章节）
├── references.bib        # 参考文献
├── sections/*.tex        # 摘要、问题重述、分析、假设、符号、模型、检验、评价、AI 声明
├── tables/*.tex          # 表 1–6 片段（由 outputs/*.csv 同源生成，见下）
└── figures/*.png         # 图 2/3/4/7/8/10（由 figs/ 复制）
```

## 编译设置

- **主文件**：`paper/main.tex`
- **编译器**：**XeLaTeX**（必须——中文由 `ctex` 宏包，需 XeLaTeX/LuaLaTeX）
- **TeX 发行版**：TeX Live 2023 或更新；Overleaf 选「XeLaTeX」+ TeX Live 2023+
- **字体**：`ctex` 默认调用系统中文字体（Overleaf 自带 Fandol/思源，无需额外安装）
- **宏包**：ctex, geometry, amsmath, amssymb, booktabs, array, graphicx, float, caption, enumitem, siunitx（均为 TeX Live/Overleaf 标准宏包）
- **参考文献**：`thebibliography` 内联（不依赖 bibtex/biber 运行）；`references.bib` 为备查/后续切换 biblatex 用

### Overleaf 导入
1. 新建项目 → Upload Project → 上传 `paper_overleaf.zip`；
2. 菜单 Settings：Compiler 选 **XeLaTeX**，TeX Live 版本 2023+；
3. 主文件设为 `main.tex`，点击 Recompile。

### 本地编译（如有 TeX Live）
```bash
cd paper && xelatex main.tex && xelatex main.tex   # 两遍以稳定交叉引用
```

## 表格/图资源的来源与复现

论文\textbf{不依赖重新运行生产计算}：表 1–6 与图均为已生成资源纳入工程。若需从正式产出重新生成：

```bash
python -m drymodel.paper_latex   # 由 outputs/*.csv 重新生成 paper/tables/*.tex
# 图见 figs/（run_all 生成），复制到 paper/figures/
```

数据源为已授权生产的官方文件 `outputs/result1–4.xlsx` 与 `table{1..6}_*.csv`
（config_id `D12-20260912-N1600.800.800-integral8pt-bdf1e-8`）。

## 格式合规（据 format2026.doc 摘要，具体以规范原文为准）

- 从摘要页开始，不含承诺书/编号页；无目录；A4，四边页边距 2.5 cm；匿名（无校名队号）。
- 摘要通常 ≤1 页；正文 ≤30 页；电子 PDF ≤20 MB。

> 编译验证环境：本仓库计算环境**未安装 TeX**，故 PDF 未在本地编译。请在 Overleaf（或本地 TeX Live）
> 编译并检查全部页面：分页、表格是否溢出页宽、图题/交叉引用、摘要是否 ≤1 页、正文页数、中文字体渲染。
