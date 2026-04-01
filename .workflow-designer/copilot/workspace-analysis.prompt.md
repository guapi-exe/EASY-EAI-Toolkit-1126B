# Workspace Analysis Prompt

Workspace: Test-CAM-1126B
Target workflow file: workflows/test-cam-1126b/main.workflow.json
Compact result file: .workflow-designer/copilot/workspace-analysis.result.flow.txt

Send the following prompt to GitHub Copilot Chat:

Analyze the current VS Code workspace project "Test-CAM-1126B".
Target workflow file: workflows/test-cam-1126b/main.workflow.json
Prompt archive file: .workflow-designer/copilot/workspace-analysis.prompt.md
Compact result file: .workflow-designer/copilot/workspace-analysis.result.flow.txt
Output a high-level software delivery workflow, not a low-level call graph.
Use the real project structure, package manifests, README files, source folders, build scripts, and key config files.
Return only compact Service Workflow DSL. Do not return JSON, markdown fences, or extra explanation.
If the current mode can write files, write the final DSL into the compact result file; otherwise return the DSL directly in chat.
Pick node count automatically from project size and complexity. Do not force a fixed range.
The extension will auto-layout locally, so do not emit x/y unless manual placement is truly necessary.
Keep only fields that carry information. Skip empty fields.
DSL:
F|n=<workflow>|r=main
N|id=<id>|t=<type>|n=<title>|i=<portId:label;...>|o=<portId:label;...>|g=<goal>|l=<layer>|r=<role>|d=<desc>|c=<item;...>|a=<item;...>|ok=<item;...>|k=<item;...>
E|f=<node.port>|t=<node.port>|r=<depends_on|calls|reads|writes|produces|blocks|feeds_context_to>
C|id=<id>|n=<name>|file=<relative path>
Rules:
1. Prefer node types: system, domain, service, api, database, task, artifact, subflow.
2. Include at least one task node when the project has clear implementation or delivery work.
3. Use multiple input ports for multi-dependency nodes and multiple output ports for branching nodes.
4. Use short ASCII ids and port ids. Prefer Chinese for titles and descriptions, but keep real folder, module, and technology names unchanged.
5. Do not use | or ; inside free text. Use Chinese punctuation instead when needed.
6. Return DSL lines only.