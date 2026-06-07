# Retrieval as Reasoning: Self-Evolving Agent-Native Retrieval via LLM-Wiki

[page 1]

## 摘要与元信息

论文标题是 Retrieval as Reasoning: Self-Evolving Agent-Native Retrieval via LLM-Wiki。论文 arXiv 编号是 2605.25480，v2 日期是 2026-05-26，属于 2026-05-25 首次提交后的近一个月新论文。作者是 Haoliang Ming、Feifei Li、Xiaoqing Wu、Wenhui Que，作者单位是 WeChat, Tencent Inc., Beijing, China。这里的 Feifei Li 是腾讯微信作者，不应误写成 Stanford 的 Fei-Fei Li。

这篇论文提出 LLM-Wiki，把外部知识从扁平 chunk 索引改造成可搜索、可阅读、可沿链接遍历、可自我修正的 Wiki 结构。论文核心观点是：检索不应只是一次性的 lookup，而应成为 agent 推理循环中的可组合动作。

## 动机与例子

论文用 2WikiMultiHopQA 的四跳问题说明传统 dense retriever 的弱点：问题比较 The Gamecock 和 Monster A Go-Go 两部电影的导演年龄。扁平向量检索可能拿到电影页，却漏掉导演传记页，因为传记页与原始问题语义距离较远。LLM-Wiki 通过电影页到导演页的显式链接，把复杂多跳问题拆成可遍历步骤。

## Retrieval-as-Lookup 的三个限制

第一，扁平 chunk 把检索降级成匹配，不适合属性比较、关系跟随和跨文档证据聚合。第二，一次性 top-k 检索是黑盒，agent 不能根据中间发现决定下一个实体、链接或修订检索计划。第三，由 LLM 编译出来的知识库会出现悬空链接、索引不一致、无依据事实和跨页矛盾，如果没有持续修正机制会逐渐退化。

[page 2]

## 研究问题与贡献

论文提出三个研究问题。Q1 询问编译式知识组织是否优于 flat RAG 和 graph-enhanced retrieval。Q2 关注 agent 是否能通过 search、read、link-following 和 sufficiency check 利用 Wiki 结构。Q3 关注结构性和语义性编译错误能否通过持久 Error Book 被发现、修复并减少复发。

论文贡献包括三点：提出 agent-native retrieval 系统 LLM-Wiki；把原始文档编译为带双向链接的 Wiki 页面并通过工具接口暴露给 agent；引入持久 Error Book 来记录系统性编译错误、根因、约束和修复状态。

## 与已有方法的差异

RAPTOR 和 MemWalker 主要生成摘要树，GraphRAG 生成社区摘要，HippoRAG 2 暴露 KG triples，LightRAG 使用实体和关系向量索引。LLM-Wiki 的差异不是简单使用图或 LLM，而是把知识组织为人类可审计、机器可遍历的 Wiki 页面和显式链接。

[page 3]

## Retrieval-as-Reasoning 三个原则

Compilability 指原始文档被转换成结构化、显式链接、可长期维护的知识单元。LLM-Wiki 通过 Wiki-structured knowledge 实现它。

Composability 指检索被拆成 search、read、link following 等原子操作，由 agent 在推理循环中组合。LLM-Wiki 通过 compositional retrieval 实现它。

Evolvability 指知识结构会随时间自我修正，而不是在重复错误中静默退化。LLM-Wiki 通过 Error Book 实现它。

## Wiki-Structured Knowledge

LLM-Wiki 的 Wiki 包含 directory indices、structured Markdown pages 和 source archives。每个页面暴露 metadata、aliases、tags、facts、source references 和 bidirectional wikilinks。这样 agent 可以先看目录概览，再定位页面，再沿链接访问实体、事件、概念或 source digest。

## 工具接口

wiki_search(query) 会优先利用页面名、别名、标签和描述等结构化信号，必要时再回退到页面正文。它返回候选页面和 metadata，用于后续阅读与遍历。

wiki_read(paths) 可以批量读取目录索引 _index.md 或完整页面。读取知识页面时，返回内容包含页面间链接，这些链接是后续多跳遍历的可操作入口。

## 遍历策略

Direct access 用于已知实体，可以直接读页面或先搜索再读 top 结果。Bridge queries 用于 A 到 B 再到答案的问题，agent 先读 A 页，通过链接发现 B，再读 B。Exploratory browsing 用于开放式或枚举式查询，agent 先读目录索引，再选择有希望的页面继续阅读。

[page 4]

## Error Book 概念

Error Book 是持久 self-correction 机制，用于检测复发的构建错误、归因根因、转换为可复用约束，并修复受影响页面。LLM 生成的 Wiki 页面可能出现 dangling links、missing sections、malformed references、unsupported facts、factual inconsistencies 和 cross-page contradictions。

论文强调，传统一次性后处理或人工审查难以阻止同类错误在新 ingestion batch 中反复出现。Error Book 会把学到的约束注入后续编译 prompt，并对已有错误页面执行修复。

## Error Book 错误类别

论文把 Error Book 错误分成七类。结构有效性错误包括 Dangling Links、Incomplete Pages、Malformed Refs、Unseen Overwrite、Index Inconsistency。内容一致性错误包括 Unsupported Facts 和 Cross-Page Contradictions。

Dangling Links 在不同 corpus 中占 29.1% 到 63.8%。Malformed Refs 在不同 corpus 中占 18.9% 到 28.5%。这说明链接验证和来源引用验证是维护可遍历 Wiki 的关键。

[page 5]

## Error Book 五阶段生命周期

Error Book 的生命周期是 Discover、Attribute、Constrain、Inject、Verify & Close。

Discover 阶段用确定性 validators 检测结构错误，用 source-grounded LLM verification 和 cross-page consistency checks 检测内容错误。Attribute 阶段把错误追溯到根因，例如没有检查索引就假设链接页面存在。Constrain 阶段把根因写成自然语言约束规则。Inject 阶段把所有 open constraints 加到 Step 2 编译 prompt 中。Verify & Close 阶段周期性复验错误页面，如果错误不再出现，就把条目标记为 closed。

Error Book 持久化为 error_book.yaml。每个条目包含错误现象、根因分析、生成的约束规则、验证方法和生命周期状态 open 或 closed。

## 两层修复机制

Layer 1 是 Code Auto-fix，每次 compilation batch 后运行，处理 dangling links、noisy formatting 和 index inconsistencies 等确定性结构错误。Layer 2 是 LLM Periodic Fix，每隔 N 篇文章触发，用于 missing pages、incomplete digests、unsupported facts 和 cross-page contradictions 等需要推理的语义错误。

[page 6]

## 实验设置

论文在三个公开 multi-hop QA benchmark 和 AuthTrace 上评估。公开 multi-hop QA 包括 HotpotQA、MuSiQue 和 2WikiMultiHopQA。每个 benchmark 使用 500 个问题。AuthTrace 用于测试高主题相似度下的证据构造，包含 Single-doc、Low multi-doc、High multi-doc 和 All 四类 fan-in 设置。

主要 baseline 包括 None closed-book、Vanilla RAG BM25、Vanilla RAG Dense、RAPTOR、GraphRAG、LightRAG 和 HippoRAG 2。论文说明 LLM-Wiki 的优势不是简单拿到更多文本，而是把知识暴露成 agent 能继续搜索、阅读和遍历的结构。

[page 7]

## Table 1 主结果

Table 1 报告 HotpotQA、MuSiQue、2WikiMultiHopQA 的 F1 和 EM。LLM-Wiki 在三项 benchmark 上都是最高。HotpotQA 上 LLM-Wiki 为 F1 0.839、EM 0.710；LightRAG 为 F1 0.819、EM 0.682。MuSiQue 上 LLM-Wiki 为 F1 0.739、EM 0.634；LightRAG 为 F1 0.659、EM 0.550。2WikiMultiHopQA 上 LLM-Wiki 为 F1 0.911、EM 0.854；LightRAG 为 F1 0.847、EM 0.764。

| Method | HotpotQA F1 | HotpotQA EM | MuSiQue F1 | MuSiQue EM | 2WikiMHQA F1 | 2WikiMHQA EM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| None (Closed-book) | 0.551 | 0.442 | 0.456 | 0.372 | 0.638 | 0.546 |
| Vanilla RAG (BM25) | 0.717 | 0.590 | 0.545 | 0.442 | 0.790 | 0.684 |
| Vanilla RAG (Dense) | 0.764 | 0.642 | 0.611 | 0.500 | 0.815 | 0.724 |
| RAPTOR | 0.801 | 0.674 | 0.522 | 0.442 | 0.707 | 0.652 |
| GraphRAG | 0.771 | 0.650 | 0.582 | 0.482 | 0.720 | 0.648 |
| LightRAG | 0.819 | 0.682 | 0.659 | 0.550 | 0.847 | 0.764 |
| HippoRAG 2 | 0.805 | 0.668 | 0.624 | 0.514 | 0.831 | 0.706 |
| LLM-Wiki | 0.839 | 0.710 | 0.739 | 0.634 | 0.911 | 0.854 |

Table 1: 主实验结果摘要，数值来自论文表格。

[page 8]

## AuthTrace 结果

AuthTrace 上 LLM-Wiki 的总体 All accuracy 是 70.4，HippoRAG 2 是 68.3，因此 LLM-Wiki 总体高 2.1 AC points。Low multi-doc 上 LLM-Wiki 是 64.6，HippoRAG 2 是 59.6，高 5.0 AC points。High multi-doc 上 LLM-Wiki 是 55.4，HippoRAG 2 是 46.5，高 8.9 AC points。

一个重要反例是 Single-doc：HippoRAG 2 是 78.3，LLM-Wiki 是 76.0，HippoRAG 2 高 2.3 AC points。原因是很多 Single-doc 问题只需找回原始文章并定位局部细节，而 LLM-Wiki 的结构化页面有时会省略细粒度本地细节。

| Method | Single | Low | High | All |
| --- | ---: | ---: | ---: | ---: |
| None (Closed-book) | 12.3 | 16.8 | 16.2 | 14.0 |
| Vanilla RAG (BM25) | 75.4 | 48.5 | 30.4 | 62.7 |
| Vanilla RAG (Dense) | 72.9 | 49.0 | 35.0 | 61.9 |
| RAPTOR | 69.5 | 44.0 | 33.9 | 58.6 |
| GraphRAG | 55.8 | 35.1 | 26.7 | 46.8 |
| LightRAG | 73.4 | 34.2 | 14.5 | 55.7 |
| HippoRAG 2 | 78.3 | 59.6 | 46.5 | 68.3 |
| LLM-Wiki | 76.0 | 64.6 | 55.4 | 70.4 |

Table 2: AuthTrace fan-in 设置下的 judged accuracy。

## 消融实验

Table 3 显示三种消融。Full LLM-Wiki 在 HotpotQA、MuSiQue、2Wiki 上是 0.839、0.739、0.911。去掉 Wiki Structure 后是 0.778、0.669、0.844。去掉 Progressive Traversal 后是 0.722、0.601、0.789。去掉 Error Book 后是 0.801、0.699、0.877。降幅最大的是 w/o Progressive Traversal。

| Variant | HotpotQA | MuSiQue | 2Wiki |
| --- | ---: | ---: | ---: |
| LLM-Wiki full | 0.839 | 0.739 | 0.911 |
| w/o Wiki Structure | 0.778 | 0.669 | 0.844 |
| w/o Progressive Traversal | 0.722 | 0.601 | 0.789 |
| w/o Error Book | 0.801 | 0.699 | 0.877 |

Table 3: 消融实验 F1 摘要。

[page 9]

## 消融解释

w/o Wiki Structure 使用相同的 source corpus，但移除编译后的 Wiki 表示，改用 flat chunk 检索，因此失去显式页面结构和链接。w/o Progressive Traversal 只执行一次 wiki_search 并读取 top 页面，禁止迭代式重新规划，因此最接近单轮检索。w/o Error Book 保留 Wiki 编译与 traversal，但禁用 Error Book 更新、约束注入和 repair。

总体来看，Wiki 结构、渐进式遍历和 Error Book 都有贡献；其中渐进式遍历贡献最大，因为它让 agent 可以利用中间观察继续访问新页面，而不是停在一次性 top-k 检索。

## 细粒度结果

在 2WikiMHQA 上，LLM-Wiki 与最强 baseline LightRAG 的 F1 差距随 hop 数增加而扩大：2-hop 问题差距是 5.7 F1 points，4-hop 问题差距是 8.3 F1 points。4-hop 上 LLM-Wiki 达到 0.983 F1，Dense RAG 为 0.924 F1。

[page 10]

## AuthTrace Judge 与数据细节

AuthTrace 包含 2099 个过滤后的 QA 实例，来自 5 位现代中文散文作家的 860 篇 public-domain writings。每个实例包含去除标题泄漏的查询、引用的 gold evidence units、atomic gold claim units、精简 reference answer 和 evidence fan-in 标签。

AuthTrace 的 fan-in 分组是 Single-doc 等于 1，Low multi-doc 是 2 到 3，High multi-doc 是大于等于 4。论文使用 GPT-4o-mini 作为 automatic judge，并在人类 audit 中确认自动评判与人工评判保持相同方法排名。

[page 11]

## Table 4 范式对比

Table 4 比较了知识组织范式。Vanilla RAG 的 index product 是 flat chunks，瓶颈是不能遍历 intermediate nodes。RAPTOR 和 MemWalker 产生 summary tree，瓶颈是树缺少 lateral associations。GraphRAG 产生 community summaries，瓶颈是摘要会丢细节且成本高。HippoRAG 2 使用 KG triples，瓶颈是 triples 有损且 PPR 近似。LightRAG 使用 entity/relation vectors，瓶颈是难以发现 intermediate entities。LLM-Wiki 使用 Wiki pages + links，知识形式是 structured knowledge，主要成本是一次性 compilation cost。

| Method | Index Product | Knowledge Form | Core Bottleneck |
| --- | --- | --- | --- |
| Vanilla RAG | flat chunks | embedding-indexed passages | cannot traverse intermediate nodes |
| RAPTOR / MemWalker | summary tree | text compression | tree lacks lateral associations |
| GraphRAG | community summaries | hierarchical compression | summaries lose detail; high cost |
| HippoRAG 2 | KG triples | machine-readable fragments | lossy triples; PPR is approximate |
| LightRAG | entity/relation vectors | vector index | cannot discover intermediate entities |
| LLM-Wiki | Wiki pages + links | structured knowledge | compilation cost one-time |

Table 4: 知识组织范式对比。

[page 12]

## Error Taxonomy Table 6

Table 6 的七类错误中，Dangling Links 是页面间链接指向不存在页面，可通过 filesystem 交叉验证。Incomplete Pages 是缺少 facts 或 sources 等必需章节，可通过模板完整性检查。Malformed Refs 是 source citations 不符合格式，可通过 regex 验证。Unseen Overwrite 是 LLM 修改了 Step 1 未选择的页面，可通过集合比较检测。Index Inconsistency 是 index 与 filesystem 不匹配，可通过双向 diff 检测。Unsupported Facts 是页面包含未被 cited source digest 支撑的声明，可用 source-grounded LLM verification 检测。Cross-Page Contradictions 是相关页面中实体属性、日期或关系冲突，可用抽样一致性检查检测。

[page 13]

## 2WikiMHQA 类型细分

Table 8 报告 2WikiMHQA 的 bridge-comparison、comparison、compositional 和 inference 四类问题。LLM-Wiki 分别为 0.983、0.989、0.833、0.909。Compositional 问题提升最大，LLM-Wiki 比 Dense RAG 高 15.6 F1 points，比 LightRAG 高 7.3 F1 points，但 0.833 的绝对分数也说明复杂组合推理还有改进空间。

| Method | Br.-Comp. | Comp. | Compos. | Infer. |
| --- | ---: | ---: | ---: | ---: |
| Dense RAG | 0.924 | 0.981 | 0.677 | 0.804 |
| LightRAG | 0.900 | 0.960 | 0.760 | 0.857 |
| HippoRAG 2 | 0.958 | 0.981 | 0.680 | 0.856 |
| LLM-Wiki | 0.983 | 0.989 | 0.833 | 0.909 |

Table 8: 2WikiMHQA 类型细分 F1 摘要。

[page 14]

## 跨 Answer LLM 泛化

Table 9 保持 GLM-5.1 编译出的 Wiki 不变，只替换 query-time answer LLM。使用 GPT-4o 时，Dense RAG 在 HotpotQA、MuSiQue、2Wiki 上是 0.741、0.503、0.636；LLM-Wiki 是 0.792、0.608、0.805。LLM-Wiki 对 GPT-4o 的优势分别是 +5.1、+10.5、+16.9 F1 points。

| Method | Answer LLM | HotpotQA | MuSiQue | 2Wiki |
| --- | --- | ---: | ---: | ---: |
| Dense RAG | GLM-5.1 | 0.764 | 0.611 | 0.815 |
| LLM-Wiki | GLM-5.1 | 0.839 | 0.739 | 0.911 |
| Dense RAG | GPT-4o | 0.741 | 0.503 | 0.636 |
| LLM-Wiki | GPT-4o | 0.792 | 0.608 | 0.805 |

Table 9: 只替换 answer LLM 的泛化结果。

## 效率分析

Table 10 报告 query-time latency。LLM-Wiki 在 HotpotQA、MuSiQue、2Wiki 上分别是 14.9、27.1、15.9 秒每题。Dense RAG 是 16.3、26.9、15.6。LightRAG 是 41.4、51.3、39.7。HippoRAG 2 是 33.5、38.2、32.4。GraphRAG 最快，为 14.0、13.9、10.8，但准确率明显低于 LLM-Wiki。

| Method | HotpotQA s/q | MuSiQue s/q | 2Wiki s/q |
| --- | ---: | ---: | ---: |
| Dense RAG | 16.3 | 26.9 | 15.6 |
| GraphRAG | 14.0 | 13.9 | 10.8 |
| LightRAG | 41.4 | 51.3 | 39.7 |
| HippoRAG 2 | 33.5 | 38.2 | 32.4 |
| LLM-Wiki | 14.9 | 27.1 | 15.9 |

Table 10: 三个 benchmark 的 query latency。

[page 15]

## 局限与未来工作

论文承认 LLM-Wiki 的一项限制是 index-time compilation 需要额外成本，虽然这部分成本会在后续查询中摊销。另一项限制是可扩展性：当 Wiki 增长到数万页面时，directory indices 可能变得笨重，页面选择质量可能下降。Web-scale 或频繁变化的 corpus 还需要 hierarchical indexing、sharding、stale-fact handling 和 global directory maintenance。未来方向包括大规模动态维护、多模态 Wiki 和跨 corpus transfer。

## 适合作为本项目 RAG Demo 的原因

这篇论文很新，低成本模型通常不知道它的具体作者、表格数字和消融结论。裸模型容易把 Feifei Li 混淆成其他同名学者，也容易把 LLM-Wiki 误说成 AuthTrace 所有列都赢，或把 Error Book 生命周期编成通用错误处理流程。把本文作为默认论文输入，可以展示 LLM-only 的盲区、Basic RAG 的证据补强，以及 Optimized RAG 在多跳概念和表格数字上的潜在优势。
