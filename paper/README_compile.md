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

## 上一轮项目内便携编译（历史记录）

本机缺少完整 TeX 发行版，本轮在 `_tmp/post_contest/` 准备了官方 **Tectonic 0.17.0 Windows MSVC**，已校验官方发布 SHA-256。依赖使用其官方 `tlextras-2022.0r0` bundle；这是本次实测环境，不等同于 TeX Live 2026。可复用现有本地缓存离线编译：

```powershell
& .\_tmp\post_contest\compile_paper.ps1 -Source 'paper' -Output '_tmp/post_contest/paper_round3'
```

脚本、下载来源、缓存说明见 `_tmp/post_contest/LATEX_ENVIRONMENT.md`。便携工具与临时缓存不纳入源码交付；换机器时可使用上面的常规 XeLaTeX 流程。官方安装说明：<https://tectonic-typesetting.github.io/en-US/install.html>。

## 论文插图与复现

当前两图修订版位于 `deliverables/figure_revision/`：`paper_review.pdf`、`paper_overleaf_review.zip`、`reproduction_support.zip`。此前 `deliverables/visual_upgrade/` 与 `deliverables/post_contest/` 中的 PDF 和压缩包均保留为历史版本。

当前成稿共 33 页：摘要第 1 页，正文第 2–28 页，AI 工具使用说明及参考文献第 29 页，附录第 30–33 页；按摘要到参考文献全计共 29 页。图共 14 幅，其中图 1 为原题照片组，图 2–5 为方法示意，图 6–14 为九幅数据图。

- 四幅方法图的排版入口在 `paper/diagrams/`，样式在 `style.tex`。图 3 读取新生成的 `fig_geometry_prototype_ai.png`；图 4(a) 保留原生 TikZ，图 4(b) 读取 `fig_reference_mapping_ai.png`；其余两幅方法图仍为原生 TikZ。两份 PNG 已纳入工程，编译不需要再次生图。
- 九幅数据图提供 PDF、PNG、SVG，沿用上一版且本次没有重绘。PDF 为论文引用版本；SVG 保留可编辑文字，密集场图的网格层使用内嵌栅格、轴和文字仍可编辑。
- 新生成的图 3 与图 4(b) 是 2172×724 像素的栅格插图，160 mm 版心下约 345 dpi，不宣称为矢量图。完整生图及局部修正提示词在 `reports/figure_revision/PROMPTS.md`；药材外观为 AI 示意，不是实验照片。
- 图表已纳入工程，单纯编译不必重绘，也不会运行求解器。附录仅摘录两个时间推进函数；完整源码在支撑材料中。
- 模型、原始输入、正式工作簿和数值证据未在视觉升级中改动。上一轮复算记录仍在 `reports/post_contest/`；全篇升级记录在 `reports/visual_upgrade/`，本次局部修订记录在 `reports/figure_revision/`，不能将其等同于新一轮数值计算。

数据图重绘需要完整支撑包，在项目根目录设置 `PYTHONPATH=src` 后运行：

```text
python -m drymodel.paper_figs --data-only
```

该入口只读取现有输入、结果与核验记录，重绘九幅数据图并更新图形 QA；默认入口效果相同，不覆盖方法关系图。局部重绘可使用 `--figures`，具体名称见 `--help`。图中文字需要 Microsoft YaHei、SimHei 或 Noto Sans CJK SC 中至少一种中文字体；在不同机器编辑 SVG 时应确认字体可用。论文内 PDF 已嵌入字体。

若需要独立方法图 PDF/PNG，可在具备 XeLaTeX 和 Poppler 的环境运行（示例只导出本次两张完整图）：

```text
python scripts/render_paper_diagrams.py --engine <xelatex的完整路径> --diagrams geometry fvm --outdir _tmp/figure_revision/diagram_previews
```

该命令从同一排版入口导出；不传 `--diagrams` 时导出全部四图。仅 `--export-relation` 会同步关系图的独立 PDF/PNG 资产，且要求选择 relation。本次只导出 geometry 和 fvm，未覆盖其余资产。本机使用 Tectonic 0.17.0 及既有 `_tmp/visual_upgrade/tectonic-cache/` 缓存离线编译，最终结果和日志在 `_tmp/figure_revision/paper_build/`；临时工具缓存不随支撑包分发，换机器使用常规 XeLaTeX 流程即可。

编译与审阅完成后，单独生成本版交付包：

```text
python scripts/package_post_contest.py --pdf <已审核PDF路径> --output-dir deliverables/figure_revision
```

每个压缩包均附逐文件 SHA-256 清单，打包时验证 CRC 与内容哈希。完整支撑包含重绘所需的 `outputs/`、`exports/`、`reports/`、`附件/` 及复用技能；旧交付 PDF/ZIP 不递归打入新包。历史交付物若未随源码包分发，重绘保护检查会记录为不适用；必需输入、模型和结果文件缺失或变化仍会阻断检查。

## 验收范围

A4、四边页边距 2.5 cm、无目录、无校名队号，从摘要连续编号。摘要限一页；正文按比赛要求不超过 30 页，附录另计，PDF 不超过 20 MB。

本次完整编译结果保留于 `_tmp/figure_revision/paper_build/`；两目标图位于第 5、8 页，检查范围包括受影响页、连带分页及全篇引用，详见 `reports/figure_revision/PAPER_QA.md`。此前全篇升级说明与逐页验收在 `reports/visual_upgrade/`，三轮复盘验收在 `reports/post_contest/PAPER_QA.md`。数值收敛仅支持当前数学模型的实现，不替代内部温度或含水率的实测物理验证。
