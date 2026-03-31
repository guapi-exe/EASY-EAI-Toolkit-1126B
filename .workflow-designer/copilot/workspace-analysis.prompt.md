# Workspace Analysis Prompt

Workspace: Test-CAM-1126B
Target workflow file: workflows/test-cam-1126b/main.workflow.json

Send the following prompt to GitHub Copilot Chat:

请分析当前 VS Code 工作区项目 "Test-CAM-1126B"。
目标工作流文件是：workflows/test-cam-1126b/main.workflow.json。
当前提示词归档文件是：.workflow-designer/copilot/workspace-analysis.prompt.md。
如果你当前处于可编辑/Agent 模式，请直接把最终 JSON 写入目标工作流文件；如果你当前不能直接写文件，则只返回最终 JSON，不要输出 markdown 代码块或额外解释。
请基于真实项目结构、package 清单、README、源码目录、构建配置、脚本、配置文件和关键模块边界进行分析。
目标不是低层调用图，而是生成一个适合 AI 和人继续推进研发工作的高层流程图。
JSON 顶层结构必须是：
{"version":1,"meta":{"name":"workspace","role":"main"},"children":[],"nodes":[],"edges":[]}
节点和连线要求：
1. 节点数量控制在 6 到 18 个，除非项目非常小。
2. 优先使用节点类型：system、domain、service、api、database、task、artifact、subflow。
3. 如果项目存在明显开发任务，至少包含一个 task 节点。
4. 标题和描述优先使用中文，但真实文件夹名、模块名、技术名保持原样。
5. 每个节点必须包含：id、type、title、x、y、inputs、outputs、data。
6. 每条边必须包含：id、from、to；可选 data.relation。
7. relation 优先使用：depends_on、calls、reads、writes、produces、blocks、feeds_context_to。
8. children 先保持空数组，meta.role 必须为 main。
端口规则：
9. 不要把所有节点都生成为单输入单输出；应根据真实语义设计多输入、多输出端口。
10. 当一个节点依赖多个来源时，使用多个 inputs，例如：需求、配置、接口契约、数据库、上游服务、设计稿、测试约束。
11. 当一个节点存在多个去向或分支时，使用多个 outputs，例如：页面输出、API 输出、构建产物、成功、失败、评审、部署、上下文传递。
12. 端口 label 尽量使用语义化名称，不要机械重复 input/output，除非确实无法命名。
13. 可以存在 0 到 4 个输入或输出端口，按节点真实职责决定。
布局规则：
14. 整体从左到右分层，优先按 4 到 5 列布局，不要把节点堆在一条竖线上。
15. 建议列分布：项目背景/入口在左侧，系统与模块在中左，服务与任务在中部，产物与交付在中右，子流程或外部依赖在右侧。
16. 横向间距至少 280，建议 320 到 420；纵向间距至少 140，建议 160 到 220。
17. 同一列尽量不超过 3 个节点；前后节点避免明显重叠或过密贴边。
18. 前端、后端、数据、测试、部署等可使用不同纵向泳道，减少交叉线。
19. 让重要主链路保持清晰，支线和补充依赖分散到上下两侧。
20. 避免生成过于密集、过于居中重叠、或所有节点 y 值几乎相同的布局。
补充语义：
21. node.data 可按需包含：description、goal、role、layer、inputContext、outputArtifacts、acceptance、tags、status、target。
22. 尽量补充 goal、role、layer、outputArtifacts，让 AI 能继续基于该图推进开发。
23. 避免使用 TBD、unknown、misc、node-1 这类占位文本。