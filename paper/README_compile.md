# 论文 LaTeX 工程编译说明（Overleaf / 本地）

## 工程结构（相对路径，可直接导入 Overleaf）

```
paper/
├── main.tex              # 主文件（\input 各章节 + 内联参考文献）
├── sections/*.tex        # 摘要；1 问题重述；2 问题分析；3 模型假设；4 符号说明；
│                         #   5 模型的建立与求解方法；6–9 问题一至问题四；10 工艺参数分析；
│                         #   11 蒸发吸热与能量一致模型；12 模型检验；13 评价与推广；AI 声明；附录
├── tables/*.tex          # 表 1–6 片段（由 outputs/table*.csv 生成，题目要求的表格式）
├── figures/*.pdf         # 10 张矢量插图（由 code/figures.py 读取 exports/redo/*.csv 绘制）
└── code/*.py             # drying_model.py（四问求解）、analysis.py（分析）、figures.py（插图）
```

附录用 `\lstinputlisting` 印出 `code/drying_model.py` 与 `code/analysis.py` 全文；`figures.py` 只随支撑材料提交。

## 编译设置

- **主文件**：`main.tex`
- **编译器**：**XeLaTeX**（中文由 `ctex` 宏包处理）
- **TeX 发行版**：TeX Live 2023 或更新；Overleaf 选「XeLaTeX」+ TeX Live 2023+
- **中文字体**：`ctex` 在 Overleaf 上默认使用 Fandol 字体，无需额外安装
- **宏包**：ctex, geometry, amsmath, amssymb, booktabs, array, graphicx, float, caption,
  enumitem, siunitx, xcolor, longtable, listings, fancyhdr（均为 TeX Live/Overleaf 标准宏包）
- **参考文献**：`thebibliography` 内联，按正文首次引用顺序排列
- **插图**：`figures/*.pdf`（矢量、字体已嵌入）

### Overleaf 导入
1. 新建项目 → Upload Project → 上传仓库根目录的 `paper_overleaf.zip`；
2. 菜单 Settings：Compiler 选 **XeLaTeX**，TeX Live 版本 2023+；
3. 主文件设为 `main.tex`，点击 Recompile（交叉引用需编译两遍）。

### 本地编译（如有 TeX Live）
```bash
cd paper && xelatex main.tex && xelatex main.tex
```

## 图表与数值的来源

```bash
cd paper/code
python drying_model.py --data ../../附件 --out output        # result1–4 与表 1–6 的 CSV
python analysis.py --data ../../附件 --out ../../exports/redo  # 第 2、6–12 节的全部数据（约 9 分钟）
python figures.py                                             # 由 exports/redo/*.csv 绘制 figures/*.pdf
```

`drying_model.py` 与原工程包 `src/drymodel` 的生产结果逐单元格一致（result1–4 未改动）。

## 格式合规（据 format2026.doc）

- 从摘要页开始，不含承诺书/编号页；无目录；A4、四边页边距 2.5 cm；匿名（无校名队号）。
- 摘要控制在一页内；正文另起一页、连续页码；正文不超过 30 页，附录另计；电子 PDF 不超过 20 MB。

> 本仓库计算环境未安装 TeX，PDF 在 Overleaf 编译。编译后请检查：摘要是否 ≤1 页、正文页数、
> 浮动体造成的页面空白、表格是否溢出页宽、交叉引用与中文字体渲染。
