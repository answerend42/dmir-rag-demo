# DemoRAG: A Tiny Fixture for Retrieval Evaluation

[page 1]

## Abstract

DemoRAG is a synthetic paper fixture used only for offline parser, chunker, and evaluation tests. It describes a three-stage retrieval pipeline that rewrites queries, retrieves evidence, and generates grounded Chinese answers.

## Method

The method first normalizes the user query, then retrieves the top ranked evidence chunks with a score where larger values mean stronger relevance. Finally, the generator refuses to answer when no evidence is available.

Figure 1: DemoRAG pipeline with query rewrite, retrieval, and grounded generation.

[page 2]

## Experiments

The fixture reports that Basic RAG can answer two of three demonstration questions, while Optimized RAG answers all three by using better query wording and stricter evidence packing.

| Mode | Answered | Cited |
| --- | ---: | ---: |
| Basic RAG | 2 | 2 |
| Optimized RAG | 3 | 3 |

Table 1: Demonstration numbers for the offline paper fixture.
