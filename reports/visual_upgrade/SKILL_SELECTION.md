# 论文视觉统筹技能选型与创建

本次需求是从论文整体论证规划流程图、模型示意和数据图，并实际完成图形升级与成稿验证。因此采用项目内新建的 `skills/paper-visual-director/`，而不是仅依靠单张科学图的美化说明。已由主任务安装到个人技能目录 `C:/Users/Lyxose/.codex/skills/paper-visual-director`，安装后逐文件哈希与项目源一致；未新增外部服务或运行依赖。

## 检索参考与取舍

已读取此前检索下载的 [scientific-visualization](https://skills.sh/k-dense-ai/scientific-agent-skills/scientific-visualization)，其原仓库为 [K-Dense-AI/scientific-agent-skills](https://github.com/K-Dense-AI/scientific-agent-skills)。本次参考文件位于 `_tmp/visual_upgrade/skill_research/scientific-visualization.md`。

该技能对科学数据图的真实性、编码、配色、物理尺寸和导出检查有明确指导，适合作为单图制作的参考。但当前任务还需要全篇主张与图谱之间的规划、非数据方法图的语义、正文接入及完整论文验收，因此新技能把这些环节作为主线。新技能独立编写，未整段复制外部说明，不依赖其脚本、锁定版本、API 或另一个必须安装的技能。

选型时公开目录显示 scientific-visualization 约 1.9K 次安装，原仓库 GitHub API 显示 45,532 stars（2026-09-19 UTC 检索）；已直接阅读原始SKILL，不仅依据排名。其独立安装命令为 `npx skills add K-Dense-AI/scientific-agent-skills --skill scientific-visualization`。本次创建的是独立的全篇统筹技能，不能把上游的安装数归给新技能。

## 当前论文提供的现实用例

创建前读取了 `paper/main.tex`、摘要、问题分析、模型与验证部分的图题/引用，以及 `src/drymodel/paper_figs.py` 的生成入口。检查到的具体用例包括：

- 四问关系图需要体现“并列模型”与“复用既有轨迹”的不同关系；概念关联不能被箭头误写成程序分阶段接续。
- 现有 TikZ 几何/离散示意与数值场图承担不同论证角色，不应机械改成同一种图。
- 关系图的原 PDF 是位图封装，扩展名不能证明可编辑或矢量质量。
- 某些 11–13 英寸画布插入约 160 mm 版心后，图源字号会大幅缩小；问题不能靠增加 DPI 解决。
- 阈值根与舍入后的采样、实际输入与假设延续、移动域外与零值、数值核验与真实预测误差需要在图中保持区别。

这些用例转化成了可复用的判断规则。技能没有写死本题名称、物性公式、阈值、四问数量、求解器或结果数值；全篇升级、局部设计、仅规划审阅按用户任务分别执行。

## 交付内容

| 文件 | 用途 |
| --- | --- |
| `skills/paper-visual-director/SKILL.md` | 主张→图谱→证据→视觉语言→图源制作→正文接入→整稿 QA 的执行流程 |
| `skills/paper-visual-director/agents/openai.yaml` | 中文显示信息与 `$paper-visual-director` 调用示例；保持默认可自动匹配策略 |
| `skills/paper-visual-director/references/visual-design.md` | 按需读取的图谱示例、箭头语义、可覆盖风格、成品尺寸与验收取舍 |

“偏活泼科研期刊风格”仅作为用户提出此偏好时的可覆盖参考。技能不要求所有用户采用同一配色，不强制新增固定张数，不要求图像生成服务、批量安装依赖或在已授权任务内反复等待设计审批。

## 验证状态

使用 skill-creator 自带 `quick_validate.py` 检查项目技能目录，结果为 `Skill is valid!`。UI 元数据以 UTF-8 生成，包含显式技能名称的默认提示；引用资源存在且位于技能目录内。

该结构检查不等于论文视觉质量验收。当前论文的实际视觉改造与编译验收由本次主任务另行执行，不能由技能文件自身宣称完成。
