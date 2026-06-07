# 三模式离线评测摘要

- 数据集：`paper`
- 问题数：26
- 生成时间：2026-06-07T04:24:01.810022+00:00

| 模式 | 可回答 | 有引用 | 拒答 | citation_hit | groundedness | 平均耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `llm_only` | 26 | 0 | 0 | 0.000 | 0.000 | 2.2 ms |
| `basic_rag` | 26 | 26 | 0 | 1.000 | 1.000 | 2.2 ms |
| `optimized_rag` | 26 | 26 | 0 | 1.000 | 1.000 | 2.3 ms |

## 现场问题

- `paper-q01` LLM-Wiki 论文的完整标题、arXiv 编号和 v2 日期是什么？
- `paper-q02` 这篇论文的作者和单位是什么？为什么 Feifei Li 容易被模型混淆？
- `paper-q03` 论文用 The Gamecock 和 Monster A Go-Go 的例子说明了什么检索问题？
- `paper-q04` Retrieval-as-Lookup 在论文中被指出有哪三个限制？
- `paper-q05` 论文提出的 Q1、Q2、Q3 分别关注什么？
- `paper-q06` Retrieval-as-Reasoning 的三个原则是什么？LLM-Wiki 分别用什么机制实现？
- `paper-q07` LLM-Wiki 的 Wiki-Structured Knowledge 包含哪些组成部分和页面字段？
- `paper-q08` wiki_search(query) 会优先利用哪些信号？它返回什么？
- `paper-q09` wiki_read(paths) 读取什么内容？为什么它有助于后续多跳遍历？
- `paper-q10` LLM-Wiki 的 Direct access、Bridge queries、Exploratory browsing 三种遍历策略分别适合什么问题？
- `paper-q11` Error Book 的作用是什么？它和一次性后处理有什么不同？
- `paper-q12` Error Book 的七类错误是什么？哪些属于结构有效性，哪些属于内容一致性？
- `paper-q13` Dangling Links 和 Malformed Refs 在错误分布中大约占多少？这说明什么？
- `paper-q14` Error Book 的五阶段生命周期按顺序是什么？
- `paper-q15` Error Book 持久化成什么文件？每个条目记录哪些信息？
- `paper-q16` LLM-Wiki 的两层修复机制分别处理什么问题？
- `paper-q17` 论文在哪些 benchmark 上评估？baseline 包括哪些？
- `paper-q18` Table 1 中 LLM-Wiki 在 HotpotQA、MuSiQue、2WikiMultiHopQA 上的 F1/EM 是多少？
- `paper-q19` Table 1 中 LLM-Wiki 相比 LightRAG 在三个 multi-hop benchmark 上 F1 分别高多少？
- `paper-q20` AuthTrace 上 LLM-Wiki 是否四列都超过 HippoRAG 2？请指出反例和总体结果。
- `paper-q21` AuthTrace 的 Low multi-doc 和 High multi-doc 上，LLM-Wiki 比 HippoRAG 2 高多少？
- `paper-q22` Table 3 消融实验中 full、w/o Wiki Structure、w/o Progressive Traversal、w/o Error Book 的三组 F1 分别是多少？
- `paper-q23` 消融实验里去掉哪个组件影响最大？为什么？
- `paper-q24` Table 4 中 GraphRAG、HippoRAG 2、LightRAG 和 LLM-Wiki 的 index product 与瓶颈分别是什么？
- `paper-q25` 2WikiMHQA 类型细分里，LLM-Wiki 在 compositional 问题上比 Dense RAG 和 LightRAG 高多少？
- `paper-q26` Table 10 的效率分析中，LLM-Wiki、Dense RAG、LightRAG、HippoRAG 2 在三个 benchmark 上的延迟大致是多少？
