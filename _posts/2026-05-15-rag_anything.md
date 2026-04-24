---
layout: single
title: "RAG-Anything: All-in-One RAG Framework Review"
categories: Study-concept
tag: [RAG, Multimodal-RAG, Document-AI]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2510.12323)

RAG-Anything은 "멀티모달 RAG"를 단순히 image captioning을 붙인 RAG로 보지 않는 논문이다. 이 논문이 흥미로운 이유는 text, image, table, equation을 모두 하나의 retrieval substrate 위에 올리려 한다는 점이다. 기존 RAG가 문서를 chunk로 자르고 vector search로 찾는 구조였다면, RAG-Anything은 문서 안의 모든 요소를 graph node와 dense representation으로 다시 조직한다.

특히 이 논문은 RAG의 병목을 generation model이 아니라 **knowledge representation과 retrieval interface**에서 찾는다. 실제 문서는 텍스트만으로 되어 있지 않다. 논문에는 figure와 caption이 있고, 보고서에는 table과 chart가 있으며, 기술 문서에는 equation과 주변 설명이 함께 존재한다. 이 요소들을 모두 텍스트로 flatten하면 구조 정보가 사라진다. 반대로 modality별 파이프라인을 따로 만들면 시스템이 조각난다. RAG-Anything은 이 둘 사이에서 **dual-graph construction**과 **cross-modal hybrid retrieval**이라는 설계를 제안한다.

> 한 줄 요약: RAG-Anything은 text, image, table, equation을 **atomic knowledge unit**으로 분해하고, cross-modal graph와 text-based graph를 결합해 **multimodal document retrieval**을 하나의 unified RAG pipeline으로 처리하려는 논문이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- RAG가 단순히 better embedding과 reranker의 문제가 아니라, **document structure를 어떻게 indexing할 것인가**의 문제로 이동하고 있다.
- 실무 문서는 PDF, table, chart, equation, slide, report처럼 mixed-content 형태가 많기 때문에 text-only RAG의 한계가 금방 드러난다.
- LightRAG 계열의 **graph-based RAG**가 multimodal document parsing과 결합될 때 어떤 구조가 되는지 보기 좋다.

RAG-Anything의 핵심 메시지는 좋은 multimodal RAG는 모든 modality를 LLM 입력에 억지로 밀어 넣는 시스템이 아니라, **검색 전 단계에서 각 modality가 가진 구조를 retrieval 가능한 지식 단위로 보존하는 시스템**이라는 것이다.

# 1. Problem Setting

## 1-1. Problem definition

- 이 논문이 겨냥하는 핵심 문제는 **real-world knowledge repository가 text-only corpus가 아니라는 점**이다.
- 기존 RAG는 보통 문서를 text chunk로 나누고, query와 chunk embedding similarity를 기준으로 relevant context를 찾는다.
- 하지만 실제 문서에는 figure, chart, table, equation, caption, section hierarchy, cross-reference가 함께 존재한다.
- 이런 요소들은 단순 텍스트 chunk로 변환하는 순간 중요한 구조가 사라진다.
- 예를 들어 figure의 panel, axis, caption 관계나 table의 row, column, unit 관계는 plain text로 바꾸면 ambiguity가 커진다.
- 따라서 문제는 "멀티모달 정보를 LLM에 넣을 수 있는가"가 아니라, "멀티모달 문서를 retrieval 가능한 knowledge representation으로 어떻게 바꿀 것인가"에 가깝다.

## 1-2. Why previous approaches are insufficient

- Text-only RAG는 non-textual evidence를 버리거나 caption으로 대체한다. 이 경우 visual detail과 spatial relationship이 손실된다.
- Naive multimodal RAG는 image를 VLM에 넣을 수는 있지만, document-level retrieval에서 table, equation, figure, text 사이의 relation을 충분히 구조화하지 못한다.
- Modality-specific pipeline은 image용, table용, equation용 처리기를 따로 붙이는 방식이 되기 쉽다. 이런 구조는 새로운 modality가 추가될수록 복잡해지고, cross-modal alignment가 어려워진다.
- GraphRAG나 LightRAG 계열은 textual entity와 relation을 잘 다루지만, non-text modality를 first-class knowledge entity로 다루는 데 한계가 있다.
- MMGraphRAG는 multimodal graph 방향으로 가지만, 논문 설명 기준으로 table과 equation의 구조적 관계까지 충분히 explicit하게 다루지는 못한다.

결국 기존 접근의 한계는 하나의 retriever가 약해서가 아니라, 문서 안의 **heterogeneous evidence를 같은 retrieval plane 위에 올리지 못했다는 데** 있다.

# 2. Core Idea

## 2-1. Main contribution

RAG-Anything의 핵심 기여는 세 가지로 정리할 수 있다.

- 첫째, text, image, table, equation을 **atomic content unit**으로 분해하는 **multimodal knowledge unification**을 제안한다.
- 둘째, cross-modal knowledge graph와 text-based knowledge graph를 따로 만든 뒤 **entity alignment**로 fusion하는 **dual-graph construction**을 사용한다.
- 셋째, **graph navigation**과 **dense vector matching**을 결합하는 **cross-modal hybrid retrieval**을 통해 structural evidence와 semantic evidence를 함께 사용한다.

이 구조에서 중요한 점은 non-text modality를 단순히 caption text로 치환하지 않는다는 것이다. 각 non-text unit은 visual artifact 자체, textual description, entity summary, graph node, dense embedding을 함께 가진다. 즉 검색은 **text proxy** 위에서 효율적으로 하고, 최종 synthesis에서는 필요하면 **원본 visual content**를 다시 dereference해서 VLM에 넣는다.

## 2-2. Design intuition

이 논문의 설계 직관은 명확하다. Multimodal document QA에서는 evidence가 한 곳에 있지 않다. 답이 figure에 있을 수도 있고, figure를 이해하려면 caption과 section text가 필요할 수도 있으며, table value를 읽으려면 row header, column header, unit 정보가 함께 필요할 수 있다.

그래서 RAG-Anything은 문서를 다음처럼 본다.

$$
k_i -> {c_j = (t_j, x_j)}_{j=1}^{n_i}
$$

여기서 $c_j$는 하나의 atomic content unit이고, $t_j$는 modality type, $x_j$는 해당 unit의 raw content다. 중요한 것은 이 decomposition이 단순 parsing이 아니라, structural context와 semantic alignment를 보존해야 한다는 점이다.

내가 보기엔 이 논문의 핵심은 graph를 쓰는 것 자체가 아니다. 핵심은 graph가 document layout과 modality relation을 보존하는 **indexing contract** 역할을 한다는 점이다. Vector DB는 semantic similarity를 잘 찾지만, table cell이 어떤 row와 column에 속하는지, figure panel이 어떤 caption과 연결되는지 같은 structural relation은 명시적으로 들고 있지 않다. RAG-Anything은 이 빈 공간을 graph로 채운다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Heterogeneous multimodal document를 unified RAG pipeline으로 처리 |
| Key modules | Multimodal knowledge unification, dual-graph construction, cross-modal hybrid retrieval, VLM-based synthesis |
| Base idea | 모든 modality를 atomic knowledge unit으로 만들고 graph + vector index에 동시에 올림 |
| Difference from text RAG | Text chunk만 검색하지 않고 image, table, equation의 structural relation까지 검색 단서로 사용 |
| Difference from naive multimodal RAG | Visual artifact를 단순히 caption으로 flatten하지 않고, retrieval 후 원본 content를 다시 dereference |

## 3-2. Module breakdown

### 1) Multimodal Knowledge Unification

- 입력 문서를 text, image, table, equation 등으로 나누고, 각 요소를 atomic content unit으로 만든다.
- Text는 paragraph나 list item 단위로 segment된다.
- Figure는 caption, cross-reference, metadata와 함께 추출된다.
- Table은 cell, header, value structure를 보존하도록 parsing된다.
- Equation은 symbolic representation으로 변환된다.
- 이 단계의 목적은 모든 file format을 같은 abstraction으로 바꾸는 것이다.
- 단, 같은 abstraction으로 바꾼다는 것이 모든 정보를 text로 압축한다는 뜻은 아니다. **원본 modality와 구조 정보**를 보존하는 것이 핵심이다.

### 2) Cross-Modal Knowledge Graph

- Non-text unit은 multimodal LLM을 통해 두 종류의 textual representation을 만든다.
- 하나는 retrieval에 유리한 **detailed description**이고, 다른 하나는 graph construction에 필요한 **entity summary**다.
- 이 생성은 해당 unit만 보지 않고 local neighborhood를 함께 본다.
- 논문 notation으로는 $C_j = {c_k | |k - j| <= delta}$ 형태의 주변 context를 사용한다.
- 이후 non-text unit을 **anchor node**로 두고, description에서 추출한 **fine-grained entity와 relation**을 연결한다.
- 예를 들어 table은 row, column, unit, value 사이의 relation을 가질 수 있고, figure는 panel, caption, axis, legend relation을 가질 수 있다.

이 설계가 중요한 이유는 non-text content를 단순 blob으로 다루지 않기 때문이다. Visual object나 table cell이 **graph 내부에서 독립적으로 참조 가능한 node**가 되면, query가 특정 구조를 요구할 때 retrieval이 훨씬 정밀해질 수 있다.

### 3) Text-Based Knowledge Graph

- Text chunk에 대해서는 LightRAG나 GraphRAG와 유사한 text-centric entity and relation extraction을 수행한다.
- 이 graph는 본문 서술에서 나타나는 explicit entity, relation, semantic connection을 포착한다.
- Cross-modal graph가 visual, table, equation grounding을 맡는다면, text-based graph는 일반적인 textual semantics를 맡는다.
- 두 graph는 역할이 다르기 때문에 처음부터 하나로 만들지 않는다. 이 점이 **dual-graph design의 핵심**이다.

### 4) Entity Alignment and Graph Fusion

- Cross-modal graph와 text-based graph는 entity alignment를 통해 하나의 comprehensive knowledge graph $G = (V, E)$로 합쳐진다.
- 논문은 entity name을 primary matching key로 사용한다고 설명한다.
- 이 fusion은 **visual-textual association**과 **fine-grained textual relation**을 동시에 활용하기 위한 단계다.
- 이후 graph entity, relation, atomic chunk를 모두 embedding table $T$에 올린다.
- 최종 retrieval index는 $I = (G, T)$로 볼 수 있다.

여기서 주의점은 entity alignment다. **Name matching**은 단순하고 실용적이지만, alias, abbreviation, homonym이 많은 domain에서는 충돌이 생길 수 있다. 이 부분은 후속 실험에서 entity linking이나 confidence-aware merge를 붙일 여지가 있다.

### 5) Cross-Modal Hybrid Retrieval

RAG-Anything의 retrieval은 **두 경로를 결합**한다.

| Retrieval path | Role |
| --- | --- |
| Structural knowledge navigation | Query term과 entity를 graph에서 찾고 neighborhood를 확장해 explicit relation을 따라감 |
| Semantic similarity matching | Query embedding과 embedding table을 비교해 graph에 직접 연결되지 않은 relevant evidence를 찾음 |
| Multi-signal fusion | Structural importance, semantic similarity, modality preference를 결합해 candidate를 ranking |

Query에 "figure", "chart", "table", "equation" 같은 lexical cue가 있으면 modality preference로 사용한다. 그 뒤 graph navigation은 explicit relation을 따라가고, dense retrieval은 topology에 직접 연결되어 있지 않지만 의미적으로 가까운 content를 찾는다. 최종 candidate pool은 대략 $C(q) = C_struct(q) + C_sem(q)$ 형태로 합쳐진다.

이 구조는 꽤 실무적이다. Graph만 쓰면 **recall이 떨어질 수 있고**, dense retrieval만 쓰면 **구조를 놓칠 수 있다**. RAG-Anything은 둘을 경쟁시키지 않고 역할을 나눈다.

### 6) Retrieval to Synthesis

- Retrieved component의 entity summary, relation description, chunk content를 structured textual context로 만든다.
- Visual artifact에 해당하는 multimodal chunk는 original visual content를 다시 가져온다.
- 최종 response는 query, textual context, visual content를 함께 condition한 VLM이 생성한다.
- 즉 text proxy는 retrieval efficiency를 위한 것이고, 원본 visual content는 final reasoning fidelity를 위한 것이다.

이 부분이 단순 caption-based RAG와 구별되는 지점이다. Caption만 검색하고 끝내는 것이 아니라, 검색된 근거가 visual artifact라면 **원본 visual evidence를 synthesis 단계에서 다시 사용**한다.

# 4. Training / Data / Recipe

## 4-1. Data

이 논문은 새로운 foundation model을 학습하는 논문이라기보다, **RAG framework와 evaluation setup**을 제안하는 논문이다. 따라서 여기서의 data는 training corpus보다 benchmark와 document processing recipe에 가깝다.

논문은 두 개의 multimodal document QA benchmark를 사용한다.

| Dataset | Documents | Avg. pages | Avg. tokens | Doc types | Questions |
| --- | ---: | ---: | ---: | ---: | ---: |
| DocBench | 229 | 66 | 46377 | 5 | 1102 |
| MMLongBench | 135 | 47.5 | 21214 | 7 | 1082 |

DocBench는 Academia, Finance, Government, Law, News domain을 포함하고, MMLongBench는 long-context multimodal document comprehension을 겨냥한다. 즉 이 논문의 evaluation은 짧은 image QA보다는 긴 문서에서 evidence를 찾아야 하는 setting에 가깝다.

## 4-2. Training strategy

- 별도의 end-to-end model training이 핵심은 아니다.
- 모든 baseline은 GPT-4o-mini를 backbone LLM으로 사용한다.
- Document parsing에는 MinerU를 사용해 text, image, table, equation을 추출한다.
- Embedding model은 text-embedding-3-large이며, dimension은 3072로 설정된다.
- Reranker는 bge-reranker-v2-m3를 사용한다.
- Graph-based RAG methods에는 entity and relation token limit 20000, chunk token limit 12000을 적용한다.
- Output은 one-sentence format으로 제한한다.
- 최종 accuracy evaluation도 GPT-4o-mini를 사용한다.

이 setup은 장점과 한계를 모두 가진다. 장점은 baseline 비교가 **같은 LLM과 parser stack** 위에서 이루어진다는 점이다. 한계는 evaluator 역시 GPT-4o-mini라서 **automated judge의 bias**를 완전히 배제하기 어렵다는 점이다.

## 4-3. Engineering notes

실무 관점에서 재사용할 만한 recipe는 다음과 같다.

- Document parsing 단계에서 modality별 unit을 명확히 분리한다.
- 각 non-text unit에 대해 detailed description과 entity summary를 분리해 만든다.
- Textual proxy는 retrieval을 위해 사용하되, synthesis 단계에서는 original visual artifact를 다시 사용할 수 있게 pointer를 유지한다.
- Graph index와 vector index를 동시에 만든다.
- Query analysis에서 modality cue를 추출한다.
- Graph navigation과 semantic matching을 따로 수행한 뒤, multi-signal fusion으로 ranking한다.
- Reranking은 성능을 조금 더 다듬지만, 논문 ablation 기준으로 가장 큰 gain은 graph construction에서 나온다.

실무에 옮길 때 핵심은 model choice보다 **indexing contract**다. 각 chunk가 어떤 source page, layout region, modality, parent node, textual description, visual pointer를 가지는지 schema를 먼저 잘 잡아야 한다.

# 5. Evaluation

## 5-1. Main results

주요 결과는 다음과 같다.

| Benchmark | GPT-4o-mini | LightRAG | MMGraphRAG | RAG-Anything |
| --- | ---: | ---: | ---: | ---: |
| DocBench overall accuracy | 51.2 | 58.4 | 61.0 | 63.4 |
| MMLongBench overall accuracy | 33.5 | 38.9 | 37.7 | 42.8 |

DocBench에서는 RAG-Anything이 overall 63.4로 가장 높다. 특히 **multimodal type에서 76.3**을 기록하며, MMGraphRAG 66.0, LightRAG 59.7, GPT-4o-mini 43.8보다 높다. 이 부분은 이 논문의 주장과 잘 맞는다. Non-text modality를 first-class entity로 다룬 효과가 multimodal question에서 크게 나타난다.

MMLongBench에서도 RAG-Anything은 overall 42.8로 가장 높다. Research reports, guidebooks, brochures, financial reports에서 강점을 보인다. 다만 모든 domain에서 최고는 아니다. Tutorial domain에서는 GPT-4o-mini가 44.0으로 RAG-Anything의 43.5보다 높고, administration domain에서는 MMGraphRAG가 46.9로 RAG-Anything의 45.7보다 높다.

따라서 이 결과는 "모든 경우에 압도"라기보다, long multimodal document QA에서 graph-based multimodal indexing이 평균적으로 유리하다는 근거로 읽는 편이 더 정확하다.

## 5-2. What really matters in the experiments

가장 중요한 실험은 final score보다 **길이별 결과와 ablation**이다.

- DocBench에서 101-200 pages 구간은 RAG-Anything 68.2, MMGraphRAG 54.6으로 13.6 point 차이가 난다.
- DocBench에서 200+ pages 구간은 RAG-Anything 68.8, MMGraphRAG 55.0으로 13.8 point 차이가 난다.
- MMLongBench에서는 11-50 pages, 51-100 pages, 101-200 pages 구간에서 각각 3.4, 9.3, 7.9 point gain을 보고한다.

이 결과가 말하는 것은 단순하다. 짧은 문서에서는 VLM이나 기존 multimodal RAG도 어느 정도 버틸 수 있다. 하지만 문서가 길어질수록 evidence가 여러 페이지와 modality에 흩어지고, 이때 graph-based structure가 retrieval robustness에 더 중요해진다.

Ablation도 방향이 명확하다.

| Method | DocBench overall |
| --- | ---: |
| Chunk-only | 60.0 |
| w/o Reranker | 62.4 |
| RAG-Anything | 63.4 |

Chunk-only에서 full model로 가면 +3.4 point다. 반면 reranker를 제거해도 62.4로, full 대비 -1.0 point다. 즉 이 논문에서 가장 중요한 gain은 reranker가 아니라 **graph construction과 cross-modal integration**에서 나온다.

또 하나 주의 깊게 봐야 할 부분은 unanswerable query다. DocBench에서 RAG-Anything의 unanswerable score는 46.0이고, MMGraphRAG의 60.5보다 낮다. 이는 multimodal evidence를 더 많이 찾아오는 구조가 항상 abstention이나 unanswerability 판단에 유리한 것은 아니라는 신호일 수 있다. 실무 RAG에서는 answer quality만큼이나 "모르면 모른다고 말하기"가 중요하므로, 이 부분은 반드시 따로 봐야 한다.

# 6. Limitations

1. **Parser dependency**가 크다. RAG-Anything은 text, image, table, equation을 잘 분해해야 작동한다. MinerU나 VLM-based extraction이 실패하면 graph도 잘못 만들어진다.
2. **Entity alignment**가 단순할 수 있다. 논문은 entity name을 primary matching key로 사용한다고 설명한다. 실제 domain에서는 alias, acronym, duplicate entity, ambiguous label이 많기 때문에 merge error가 생길 수 있다.
3. **Automated evaluation bias**가 남는다. Query result accuracy를 GPT-4o-mini로 평가하므로, human evaluation과 완전히 같은 신뢰도를 기대하기는 어렵다.
4. **Cost, latency, storage overhead** 분석이 상대적으로 약하다. Multimodal parsing, VLM description generation, graph construction, dense embedding, reranking을 모두 포함하면 production cost가 커질 수 있다.
5. **Unanswerable query 성능**이 강하지 않다. DocBench에서 RAG-Anything은 overall은 가장 높지만, unanswerable category에서는 MMGraphRAG보다 낮다.
6. Appendix의 failure analysis에 따르면 current multimodal RAG는 **text-centric retrieval bias**와 **rigid spatial processing** 문제를 여전히 가진다. RAG-Anything이 방향을 제시하지만, 복잡한 layout과 noisy multimodal evidence를 완전히 해결한 것은 아니다.

# 7. My Take

## 7-1. Why this matters for my work

- 이 논문은 RAG를 chunking and embedding pipeline이 아니라 **document understanding pipeline**으로 보게 만든다.
- 실제 서비스에서 RAG가 어려운 이유는 retriever가 약해서만이 아니다. PDF parsing, table structure, figure grounding, citation, section hierarchy가 함께 무너지기 때문이다.
- RAG-Anything의 dual-graph 설계는 이 문제를 잘 드러낸다. 좋은 RAG 시스템은 **질문에 답하기 전에 문서를 어떤 구조로 기억할지**부터 정해야 한다.
- 특히 enterprise document QA, research assistant, financial report QA, technical document search 같은 영역에서는 이 방향이 실용적이다.

## 7-2. Reuse potential

재사용하고 싶은 포인트는 네 가지다.

1. **Atomic content unit schema**
   - 모든 chunk에 modality, page, section, parent node, description, source pointer를 붙인다.
2. **Dual index**
   - Graph index는 relation과 layout context를 맡고, vector index는 semantic recall을 맡긴다.
3. **Modality-aware query routing**
   - Query에서 table, figure, equation cue를 감지하고 retrieval weight를 조정한다.
4. **Visual dereferencing**
   - 검색은 textual proxy로 하되, 답변 생성 단계에서는 원본 image/table region을 다시 가져온다.

이 네 가지는 RAG-Anything 전체를 구현하지 않아도 기존 RAG stack에 부분적으로 붙일 수 있다.

## 7-3. Follow-up papers

- LightRAG: Simple and Fast Retrieval-Augmented Generation
- From Local to Global: A Graph RAG Approach to Query-Focused Summarization
- MMGraphRAG: Bridging Vision and Language with Interpretable Multimodal Knowledge Graphs
- MinerU: An Open-Source Solution for Precise Document Content Extraction
- MMLongBench-Doc: Benchmarking Long-Context Document Understanding with Visualizations
- VisRAG 계열 document-as-image RAG 논문

# 8. Summary

- RAG-Anything은 multimodal document를 text-only chunk로 flatten하지 않고, **modality-aware atomic knowledge unit**으로 분해한다.
- Cross-modal graph와 text-based graph를 따로 만든 뒤 fusion해, **visual-textual relation과 textual semantics**를 함께 보존한다.
- Retrieval은 **graph navigation과 dense semantic matching**을 결합하고, synthesis 단계에서는 retrieved visual artifact를 다시 dereference한다.
- DocBench와 MMLongBench overall 기준으로 기존 baselines보다 높은 결과를 보이며, 특히 긴 문서에서 gap이 커진다.
- 다만 parser dependency, entity alignment error, automated evaluation bias, cost/latency overhead, unanswerable query 성능은 실무 적용 전에 반드시 확인해야 한다.
