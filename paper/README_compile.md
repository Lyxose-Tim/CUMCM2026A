# 论文编译与核验说明

入口为 `paper/main.tex`。章节、表格、图像与附录代码均使用相对路径；编译时以 `paper/` 为工程目录。现有 LaTeX 工程继续使用 XeTeX 系列中文排版，不更换模板。

## 常规编译

本地完整 TeX Live 或 Overleaf 选择 **XeLaTeX**，推荐 TeX Live 2023 或更新，运行两遍以稳定交叉引用：

```text
cd paper
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

Overleaf 上传本分支的完整 `paper/` 目录并将 `main.tex` 设为主文件。根目录既有 `paper_overleaf.zip` 保留为赛时归档，本轮没有覆写；它不代表当前赛后修订稿。

中文使用 `ctex`，Overleaf 和本次便携编译默认采用 Fandol 字体。常用依赖包括 `geometry`、`amsmath`、`amssymb`、`booktabs`、`array`、`graphicx`、`float`、`caption`、`enumitem`、`siunitx`、`xcolor`、`tabularx`、`longtable`、`tikz`、`listings`、`fancyhdr`。参考文献内联于主文件，不依赖 BibTeX/Biber。

## 本次项目内便携编译

本机缺少完整 TeX 发行版，本轮在 `_tmp/post_contest/` 准备了官方 **Tectonic 0.17.0 Windows MSVC**，已校验官方发布 SHA-256。依赖使用其官方 `tlextras-2022.0r0` bundle；这是本次实测环境，不等同于 TeX Live 2026。可复用现有本地缓存离线编译：

```powershell
& .\_tmp\post_contest\compile_paper.ps1 -Source 'paper' -Output '_tmp/post_contest/paper_round3'
```

脚本、下载来源、缓存说明见 `_tmp/post_contest/LATEX_ENVIRONMENT.md`。便携工具与临时缓存不纳入源码交付；换机器时可使用上面的常规 XeLaTeX 流程。官方安装说明：<https://tectonic-typesetting.github.io/en-US/install.html>。

## 结果与代码来源

- 表 1–6 及半径表沿用原有文件；本次完整复算确认四位小数逐项一致。
- 十张题内相关 PDF 图、两张题面照片以及两幅内联 TikZ 示意图保留；潜热扩展图不再被主稿引用，旧资源留存。
- 当前论文明确区分本次完整加密/通量复算与赛时解析、向后 Euler、参数扰动及端效应辅助证据。
- 附录仅摘录两个时间推进函数；完整可运行 Python 源码与依赖另随支撑材料交付。论文编译只读取 `paper/code/`，不触发模型计算。
- 重绘入口为 `python -m drymodel.paper_figs` 与 `python -m drymodel.paper_latex`。图表已纳入工程，单纯编译不必重绘或重跑生产计算。

## 验收范围

A4、四边页边距 2.5 cm、无目录、无校名队号，从摘要连续编号。摘要限一页；正文按比赛要求不超过 30 页，附录另计，PDF 不超过 20 MB。

实际编译日志、逐页渲染及页数/交叉引用/缺字检查保留于 `_tmp/post_contest/paper_round3/`；第三轮论文验收结论记录于 `reports/post_contest/PAPER_QA.md`。数值收敛仅支持当前数学模型的实现，不替代内部温度或含水率的实测物理验证。
