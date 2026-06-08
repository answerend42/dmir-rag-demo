# DMIR RAG Demo

信息检索课程结课实验项目。当前展示目标是完成一条可解释、可量化、可视化的真实 RAG 链路：文档导入、分块、解析、嵌入、索引、检索、生成和向量投影。07 响应生成支持课程 QA 与 LLM-Wiki 论文两类入口：课程 QA 读取前端导入的题目和候选答案，再从外部知识 Chroma collection 检索证据进行排序；论文 RAG 维持普通文档问答流程。

前端入口：`http://127.0.0.1:5173/`
后端 Swagger：`http://127.0.0.1:8001/docs`

![RAG Frontend](images/RAG-fontend.png)

## 当前状态

- 论文展示目标是 `Retrieval as Reasoning: Self-Evolving Agent-Native Retrieval via LLM-Wiki`。
- 前端 07 响应生成可切换课程 QA 与论文 RAG：课程 QA 执行候选答案选优/排序，论文 RAG 执行普通证据问答。
- 07 已引用证据会显示相似性分数，分数语义为“越大越相关”。
- 07 向量视图会标出用户 Query 向量点，并高亮 TopK 检索命中点。
- 04 向量存储只保留 Qwen（百炼）和 HuggingFace 两类 embedding provider；Qwen 默认 `text-embedding-v4`，HF 可选 `BAAI/bge-small-zh-v1.5` 与 `intfloat/multilingual-e5-small`。
- 04 向量存储内置“向量投影视图”，可查看已生成 embedding 的 3D/2D t-SNE 或 PCA 投影。
- HuggingFace 本地模型目录位于仓库内 `hf_model_path/`，使用 HF embedding 时必须设置 `HF_MODEL_PATH`。

## 一键启动

以下命令都从仓库根目录执行，不写死任何机器上的绝对路径。

启动后端：

```shell
cd backend
HF_MODEL_PATH=../hf_model_path \
HF_ENDPOINT=https://hf-mirror.com \
../.venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8001
```

启动前端：

```shell
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

如果端口被旧进程占用，先查看并停止对应进程：

```shell
lsof -nP -iTCP:8001 -sTCP:LISTEN
lsof -nP -iTCP:5173 -sTCP:LISTEN
```

## 环境变量

本项目不得把 API key 写入源码、fixture、README 示例值、日志或评测结果。需要真实 API 时只从环境变量读取。

百炼相关：

```shell
export DASHSCOPE_API_KEY="..."
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

HuggingFace 本地模型相关：

```shell
export HF_MODEL_PATH=./hf_model_path
export HF_ENDPOINT=https://hf-mirror.com
```

注意：如果后端是从 `backend/` 目录启动，`HF_MODEL_PATH` 应写成 `../hf_model_path`；如果从仓库根目录运行脚本，则写成 `./hf_model_path`。

## 快速验证

```shell
python -m compileall backend/rag_core backend/services/search_service.py backend/main.py
cd frontend && npm run build
```

查看可用 Chroma 索引库：

```shell
curl -sS "http://127.0.0.1:8001/collections?provider=chroma"
```

执行检索并回传 Query 向量：

```shell
curl -sS -X POST http://127.0.0.1:8001/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "LLM-Wiki 在 AuthTrace 的 Single-doc 和 Overall 上与 HippoRAG 2 谁更强？具体数字是多少？",
    "collection_id": "file_2605.25480v2_huggingface_20260608232418",
    "top_k": 10,
    "threshold": 0.3,
    "word_count_threshold": 0,
    "save_results": false,
    "include_query_embedding": true
  }'
```

读取 collection 向量供前端降维展示：

```shell
curl -sS "http://127.0.0.1:8001/collections/chroma/file_2605.25480v2_huggingface_20260608232418/embeddings"
```

## 演示流程

推荐现场演示顺序：

1. **文档导入**：`/load-file`
   - `PDF 文档`：上传 PDF 并持久化到后端文档目录。
   - `课程 QA JSON`：上传课程 QA JSON，前端调用真实 `/load-course-qa-json`，后端将结构化题目转换为可继续分块的已导入文档。
   - `课程知识文档`：上传 Markdown/TXT 外部知识材料，前端调用真实 `/load-course-knowledge-doc`，后端按标题切成课程知识章节。
   - 课程 QA JSON 导入时不会把 `answer_quality` 写入导入结果、前端预览、索引或生成链路。

2. **知识分块**：`/chunk-file`
   - PDF 文档可选择按页、固定大小、段落或句子分块。
   - 课程 QA JSON 会自动使用 `课程 QA 条目分块`，一题一块，保留 topic、qa_id 等结构化 metadata。

3. **文件解析**：`/parse-file`
   - 这是可选的临时解析预览页。
   - 它不会复用 01 已导入文档，也不会持久化解析结果，所以页面会要求重新上传 PDF。
   - 正常 RAG 演示可以跳过这一页。

4. **向量存储**：`/embedding`
   - 从已导入或已分块文档中选择文档生成 embedding。
   - Qwen 使用百炼 `text-embedding-v4`。
   - HuggingFace 保留 `BAAI/bge-small-zh-v1.5` 作为中文轻量模型，并提供 `intfloat/multilingual-e5-small` 作为轻量多语言选择。
   - 页面内“向量投影视图”会用后端降维结果展示 embedding 分布，默认 3D，可切回 2D。

5. **向量库索引**：`/indexing`
   - 将 embedding 文件写入 Chroma。
   - 生成后会得到一个 Chroma collection 名称，后续检索和生成都选择这个索引库。

6. **相似性检索**：`/search`
   - 选择 Chroma collection 并检索证据。
   - `SearchHit.score` 语义必须保持“越大越相关”。

7. **响应生成**：`/generation`
   - 课程 QA：选择 01 导入的课程 QA 文件和题目，再选择外部知识 Chroma 索引库，系统会检索外部证据并调用百炼对候选答案选优/排序。
   - 论文 RAG：选择论文 Chroma 索引库，输入问题并运行普通 RAG 问答。
   - 已引用证据显示相似性分数。
   - 向量视图标出 Query 点和 TopK 命中点。

## 相似性分数

论文 RAG 使用 Chroma HNSW，并在索引时设置 `hnsw:space = cosine`。Chroma query 返回的是 distance，后端展示时转换为：

```text
score = 1 - Chroma distance
```

因此：

- distance 越小，向量越接近。
- score 越大，证据越相关。
- 前端只展示 score，不把它当作生成模型置信度。

## 论文数据

| 路径 | 用途 | 是否允许进入 RAG |
| --- | --- | --- |
| `sample_data/papers/llm_wiki_retrieval_as_reasoning.md` | LLM-Wiki 论文结构化 digest，保留章节、页码和表格信息 | 是 |
| `sample_data/papers/paper_eval_fixture.json` | 论文 metadata、干扰论文和 QA/evidence | 是 |
| `sample_data/papers/demo_research_paper.md` | parser/chunker contract test fixture | 是 |
| `sample_data/course_qa.json` | 课程 QA 任务输入，包含题目和候选答案；导入时隐藏质量标签不会进入 RAG 链路 | 是 |
| `sample_data/daily_life_knowledge_reference.md` | 日常生活主题外部知识库演示文档，可在 01 作为课程知识文档上传 | 是 |

当前仓库没有提交正式论文 PDF。前端旧的 01/03 页面只接受 PDF；仓库锁定的论文语料是 Markdown digest，主要服务于 parser、chunker 和已构建的论文向量集合。

## 项目结构

```text
backend/
  main.py                         # FastAPI 入口
  services/                       # 页面式 RAG 服务层
  rag_core/
    contracts/                    # Pydantic contracts 与 Protocol
    parsers/                      # 论文 Markdown/PDF parser
    chunkers/                     # 论文结构化 chunker
    embeddings/                   # Qwen API/local 与 HF embedding adapter
    retrieval/                    # context packing 与证据组织
    generation/                   # grounded prompt、拒答和引用格式化
    vector_indexes/               # Chroma / Numpy 索引适配

frontend/
  src/                            # React/Vite 前端

sample_data/
  papers/

eval/
  results/

scripts/

tests/

docs/
```

## 常见问题

### 03 文件解析为什么还要重新上传？

因为 `/parse-file` 是临时解析预览页，后端 `/parse` 接口会读取本次上传的 PDF，解析完删除临时文件，不会读取 `/load-file` 已经保存的导入文档。完整 RAG 流程不依赖这一页。

### HF bge-small-zh-v1.5 生成向量为什么会 Broken pipe？

通常是后端启动时没有设置 `HF_MODEL_PATH`。代码会优先查找 `HF_MODEL_PATH` 下的本地模型；没有该环境变量时，会按远程 HuggingFace 模型名加载，现场网络或下载过程可能导致连接中断。按本 README 的后端启动命令重启即可。

### 课程 QA 或论文展示应该用哪个集合？

课程 QA 的 QA JSON 是任务输入，不再作为“知识库答案”来检索；07 页面会从 01 导入结果中读取题目和候选答案。课程 QA 的 collection 应选择外部知识文档构建出来的 Chroma 索引库。论文 RAG 则选择论文文档构建出来的 Chroma 索引库。可用集合以当前后端 `/collections?provider=chroma` 返回为准。示例：

```text
file_2605.25480v2_huggingface_20260608232418
```

### 课程 QA 外部知识库怎么提供？

当前已经提供两种真实来源：

1. 上传已有课程讲义、教材摘录或参考资料 PDF，走 `PDF 文档 -> 02 分块 -> 04 embedding -> 05 Chroma 索引`。
2. 上传仓库内的 `sample_data/daily_life_knowledge_reference.md`，走 `课程知识文档 -> 02 分块 -> 04 embedding -> 05 Chroma 索引`。

第二种用于当前日常生活主题现场演示，后续拿到真实课程资料后可直接替换文档，不需要改 07 逻辑。

## 协作规则

本仓库采用 contract-first 协作方式。任何 Agent 开工前必须先读：

- `AGENTS.md`
- `docs/agent_rules.md`
- `docs/interfaces.md`
- `docs/contribution.md`
- `docs/sprint_board.md`
- `docs/agent_instructions/README.md`

硬性规则摘要：

- 不得随意修改 `backend/rag_core/contracts/`。
- 单元测试不得依赖真实网络、真实 API 或真实模型下载。
- 代码注释必须使用中文 Doxygen 风格。
- 面向 LLM 的 prompt 和展示文案尽量使用中文。
- 新论文 RAG 任务已经并入现有 Issue 的阶段 B，不需要另开一轮任务。

## PR 输出要求

PR 必须写清：

- 修改文件列表。
- 测试命令和结果。
- 是否修改契约。
- 风险点和真实链路不可用时的处理方式。
- 使用 Agent 时是否严格遵守 `ALLOWED_PATHS`。
