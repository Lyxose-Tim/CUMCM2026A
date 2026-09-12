# 论文 LaTeX 工程编译说明（Overleaf / 本地）

## 工程结构（相对路径，可直接导入 Overleaf）

```
paper/
├── main.tex              # 主文件（\input 各章节 + 内联参考文献）
├── sections/*.tex        # 摘要、问题重述、问题分析、模型假设、符号说明、
│                         #   模型建立与求解、模型检验与灵敏度、模型评价、
│                         #   AI 使用声明、附录
├── tables/*.tex          # 表 1–6 片段（由 outputs/*.csv 同源生成）
├── figures/*.pdf         # 矢量插图（由 paper_figs.py 读取官方结果重绘）
└── code/*.py             # 附录引用的源程序（\lstinputlisting）
```

## 编译设置

- **主文件**：`paper/main.tex`
- **编译器**：**XeLaTeX**（必需——中文由 `ctex` 宏包，需 XeLaTeX/LuaLaTeX）
- **TeX 发行版**：TeX Live 2023 或更新；Overleaf 选「XeLaTeX」+ TeX Live 2023+
- **中文字体**：`ctex` 在 Overleaf 上默认使用 Fandol 字体，无需额外安装
- **宏包**：ctex, geometry, amsmath, amssymb, booktabs, array, graphicx, float,
  caption, enumitem, siunitx, xcolor, tikz, listings（均为 TeX Live/Overleaf 标准宏包）
- **参考文献**：`thebibliography` 内联，为唯一来源（不再附 `.bib`，不依赖 bibtex/biber）
- **插图**：`figures/*.pdf`（矢量、字体已嵌入，Overleaf 直接显示）；示意图用 TikZ 内联
- **附录源程序**：`\lstinputlisting{code/*.py}`，中文注释由 ctex+xeCJK 渲染

### Overleaf 导入
1. 新建项目 → Upload Project → 上传 `paper_overleaf.zip`；
2. 菜单 Settings：Compiler 选 **XeLaTeX**，TeX Live 版本 2023+；
3. 主文件设为 `main.tex`，点击 Recompile（交叉引用两遍稳定）。

### 本地编译（如有 TeX Live）
```bash
cd paper && xelatex main.tex && xelatex main.tex
```

## 图表/结果的来源与复现

论文**不依赖重新运行生产计算**：表 1–6 与插图均为已生成资源纳入工程。若需从官方产出重新生成：

```bash
python -m drymodel.paper_figs    # 由 outputs/*.xlsx 重绘 figures/*.pdf
python -m drymodel.paper_latex   # 由 outputs/*.csv 重新生成 tables/*.tex
```

数据源为已授权生产的官方文件 `outputs/result1–4.xlsx` 与 `table*.csv`
（config_id `D12-20260912-N1600.800.800-integral8pt-bdf1e-8`）。

## 格式合规（据 format2026.doc）

- 从摘要页开始，不含承诺书/编号页；无目录；A4、四边页边距 2.5 cm；匿名（无校名队号）。
- 摘要控制在一页内；正文另起一页、连续页码；正文不超过 30 页，附录另计；电子 PDF 不超过 20 MB。

> 编译验证环境：本仓库计算环境**未安装 TeX**（经确认无 xelatex/pdflatex/latexmk/tectonic），
> 故 PDF 未在本地编译。请在 Overleaf（Compiler=XeLaTeX）编译并逐页检查：
> 分页、表格是否溢出页宽、图题/交叉引用是否正常、摘要是否 ≤1 页、正文页数、中文字体渲染、匿名。
