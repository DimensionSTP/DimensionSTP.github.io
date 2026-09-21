---
layout: single
title: "Self-Evolving Search Index Review"
categories: Study-concept
tag: [SelfIndex, InformationRetrieval, RAG]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/pdf/2609.19656)

[Project page](https://augustinlib.github.io/Self-Index/)

[Code repository](https://github.com/augustinLib/Self-Index)

RAG or search agent가 실패할 때 보통 query rewriting, reranking, retriever fine-tuning을 먼저 생각한다. 하지만 failure가 document representation 자체에서 시작될 수도 있다. Original document에 answer가 있어도 index key가 user query와 맞지 않으면 retriever는 해당 document를 노출하지 못한다.

Index key를 manually enrich하는 방법은 오래전부터 있었다.

- Document title and heading을 추가한다.
- Synthetic query를 생성한다.
- Summary or keyword를 붙인다.
- Domain-specific synonym을 넣는다.
- Retriever training pair를 만든다.

문제는 어떤 enrichment strategy가 항상 좋은 것이 아니라는 점이다. BM25에 유리한 lexical expansion이 dense retriever에서는 noise가 될 수 있고, one domain에서 좋은 synthetic query style이 다른 domain에서는 mismatch를 만든다. Retrieval environment가 변할 때마다 human이 failure를 분석하고 전체 index를 다시 처리해야 한다.

SELF-INDEX는 index key를 static artifact가 아니라 versioned, revisable representation으로 본다. Optimizer가 retrieval failure를 진단하고, responsible document의 key set만 수정하며, faithfulness, specificity, separation을 통과한 revision만 index에 반영한다. Query Simulator는 observed query 밖의 demand를 생성해 proactive evolution을 만든다.

> 한 줄 요약: SELF-INDEX는 retrieval outcome에서 index shortfall을 진단하고, affected document key set을 선택적으로 수정한 뒤 three-part validation을 통과한 key만 반영하며, Query Simulator로 unseen demand까지 탐색하는 self-evolving index framework다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Retrieval optimization target을 retriever weight가 아니라 document-side key set으로 옮긴다.
- Human relevance labels 없이 co-retrieval pattern and outcome으로 failure를 진단한다.
- Incremental selective revision으로 entire corpus reprocessing cost를 줄인다.
- Generated key를 faithfulness, specificity, separation의 세 gate로 검증한다.
- Natural language, code, math, table retrieval and downstream agent use case까지 평가한다.
- 매우 큰 reported gain과 동시에 code release and offline cost accounting의 한계를 함께 볼 수 있다.

# 1. Problem Setting

## 1-1. Retrieval quality는 document content만으로 결정되지 않는다

Document $d$를 key set $K(d)$로 표현한다고 하자. Query $q$와 document score는 key 중 가장 잘 맞는 representation으로 계산할 수 있다.

$$
s(q,d)
=
\max_{k \in K(d)}
\operatorname{sim}(q,k)
$$

$K(d)$에는 original text, title, summary, synthetic query, keyword, code description 등이 들어갈 수 있다.

같은 document라도 key set이 달라지면 retrieval surface가 달라진다. Original text에 query vocabulary가 없더라도 synthetic query가 user wording을 포함하면 hit할 수 있다. 반대로 generic key를 많이 추가하면 unrelated query에서도 false positive가 늘어난다.

따라서 index optimization은 key recall and specificity의 trade-off다.

## 1-2. Fixed enrichment strategy의 한계

### 1) Environment마다 effective key가 다르다

- BM25는 lexical overlap에 민감하다.
- Dense retriever는 semantic representation에 민감하다.
- Code retrieval task은 API, behavior, error signature가 중요하다.
- Table retrieval task은 schema, value range, column semantics가 중요하다.
- Agent memory는 event, goal, outcome, temporal relation이 중요하다.

One prompt template로 모든 document에 synthetic query를 붙이는 방식은 optimal하지 않다.

### 2) Entire index reprocessing이 비싸다

Optimization policy가 바뀔 때 every document를 다시 summarize or generate하면 corpus size에 비례해 cost가 발생한다. 실제 failure는 일부 key and document에 집중될 수 있다.

### 3) Human diagnosis가 bottleneck이다

Retrieval miss를 보고 어떤 document key가 부족한지, 경쟁 document와 왜 구분되지 않는지, 어떤 query class가 uncovered인지 manually 분석해야 한다.

### 4) Generated key가 hallucination을 넣을 수 있다

LLM이 document에 없는 concept를 key에 추가하면 recall은 올라가는 것처럼 보여도 index faithfulness가 깨진다. Search result가 source content를 잘못 대표할 수 있다.

## 1-3. Query-side optimization만으로 부족한 경우

Query rewriting은 current query를 바꾸지만 document representation은 그대로다. User language가 반복적으로 document language와 어긋난다면 매 query마다 rewrite cost를 낸다.

SELF-INDEX는 recurring mismatch를 offline index improvement로 흡수하려 한다. Query-time agent가 매번 same bridge를 만들기보다 document-side key에 학습된 retrieval entry point를 남긴다.

# 2. Core Idea

## 2-1. Optimizer loop

Optimizer는 세 단계로 index를 evolve한다.

1. Self-Diagnosis
2. Self-Revision
3. Self-Validation

Validated update만 next index generation에 들어간다.

$$
K_{t+1}(d)
=
\begin{cases}
\operatorname{Revise}(K_t(d),E_t), & \text{if validation passes} \\
K_t(d), & \text{otherwise}
\end{cases}
$$

$E_t$는 observed retrieval evidence and diagnosis다.

## 2-2. Self-Diagnosis

Self-Diagnosis는 query-document relevance label을 직접 받지 않는다. 대신 retrieval outcome and co-retrieval pattern을 본다.

직관적으로 다음 질문을 한다.

- Correct evidence document가 consistently below cutoff에 있는가.
- 특정 competitor document가 같은 query에서 반복적으로 앞서는가.
- Missed document의 existing key가 query intent를 어떤 부분에서 놓치는가.
- Failure가 recall 부족인지, false-positive competition인지 구분할 수 있는가.

Co-retrieval profile은 같은 query set에서 어떤 documents가 함께 올라오는지 보여준다. Responsible document and key를 좁히는 pseudo-relevance-feedback-like signal로 사용된다.

## 2-3. Self-Revision

Revision은 single keyword append가 아니다. Targeted document의 full key set을 다시 본다.

- Redundant keys를 제거할 수 있다.
- Missing intent를 설명하는 key를 추가할 수 있다.
- Too generic key를 more specific하게 바꿀 수 있다.
- Competitor document와 구분되는 discriminative detail을 넣을 수 있다.

전체 corpus가 아니라 affected documents만 수정한다. 이 selective update가 main cost advantage다.

## 2-4. Self-Validation

Generated key는 세 기준을 통과해야 한다.

### 1) Faithfulness

Key가 source document에서 support되는가. Hallucinated attribute or claim을 추가하지 않는가.

### 2) Specificity

Key가 document content를 충분히 구체적으로 나타내는가. 여러 unrelated documents에 똑같이 적용되는 generic phrase가 아닌가.

### 3) Separation

Confusing competitor document와 구분되는가. Target document를 expose하면서 false positive를 줄이는가.

이 validation은 key generation보다 더 중요한 safety layer다. More keys always improve retrieval이라는 lottery effect를 막으려 한다.

## 2-5. Query Simulator and Self-Exploration

Observed query만 최적화하면 index가 known demand에 overfit될 수 있다. Query Simulator는 corpus에서 documents를 sampling하고 plausible information need를 생성한다.

Self-Exploration은 generated query가 existing optimization query와 너무 비슷하지 않도록 dissimilarity filter를 사용한다. New query는 retrieval loop에 들어가 uncovered demand를 찾는다.

즉 evolution signal은 두 종류다.

- Reactive signal: 실제 or 관측된 query failure
- Proactive signal: simulated query exploration

# 3. Architecture / Method

## 3-1. Overview

| Component | Role |
| --- | --- |
| Base index | Original document plus current key set |
| Retriever | BM25, dense embedding retriever, table retriever 등 |
| Optimizer | Diagnosis, revision, validation loop |
| Query Simulator | New information need generation |
| Evolution pool | Observed and simulated query outcome |
| Selective updater | Affected document key only revision |
| Downstream consumer | Search agent, RAG, agent memory system |

## 3-2. Index generation

Initial key set은 단순할 수 있다.

$$
K_0(d)=\{\operatorname{text}(d)\}
$$

Evolution round가 진행되면 document마다 multiple keys를 가질 수 있다.

$$
K_t(d)
=
\{k_{d,1}^{(t)},\ldots,k_{d,n_d}^{(t)}\}
$$

Retriever score가 max-over-keys라면 key count and quality가 both important하다. Key budget을 통제하지 않으면 more-key lottery and index size growth가 생긴다.

## 3-3. Environment-specific optimization

SELF-INDEX는 retriever and corpus pair를 environment로 본다. Same document collection도 BM25 and BGE에서 다른 failure pattern을 만들 수 있다.

Optimizer는 fixed universal key template를 적용하지 않고, current environment의 retrieval output을 바탕으로 revision strategy를 결정한다.

이 design이 paper의 central thesis다.

> Good index representation은 document 하나의 property가 아니라 document, query demand, retriever가 함께 만드는 interaction property다.

## 3-4. Selective update

Round $t$의 failure set을 $D_t^{fail}$이라고 하자.

$$
D_t^{fail}
\subseteq
\mathcal{D}
$$

SELF-INDEX는 $D_t^{fail}$ or related competitor set만 LLM revision 대상으로 보낸다. Entire corpus를 rerun하지 않는다.

이 구조는 continuous operation에 유리하다. New query class가 들어오면 affected slice만 version up할 수 있다.

## 3-5. Version and rollback

Paper의 core method는 validation-gated update다. Production implementation에서는 이를 explicit versioning으로 확장하는 것이 좋다.

- Index generation ID 기록
- Document key diff 기록
- Trigger query set 기록
- Validator result 기록
- Retrieval delta 기록
- Regression set result 기록
- Rollback pointer 기록

Index가 self-evolve한다면 model checkpoint처럼 lineage를 관리해야 한다.

# 4. Training / Data / Recipe

## 4-1. This is not retriever training

SELF-INDEX main loop는 retriever weight를 update하지 않는다. Document source text도 바꾸지 않는다. 바뀌는 것은 index key set이다.

따라서 training이라기보다 iterative offline optimization에 가깝다.

- LLM이 diagnosis and revision을 생성한다.
- Validator가 candidate keys를 accept or reject한다.
- Changed documents만 retriever에 re-indexing한다.
- Evaluation queries로 next round를 측정한다.

Main configuration은 large instruction model을 Optimizer and Query Simulator에 사용한다. Exact decoding, prompt, round budget은 reproduction에서 중요하다.

## 4-2. Evaluation domains

Paper는 retrieval type을 넓게 잡는다.

- Natural-language retrieval task
- Code retrieval task
- Math retrieval task
- Table retrieval task
- Search-agent retrieval task
- Agent-memory retrieval task

이 breadth는 fixed enrichment method가 environment마다 다르게 작동한다는 motivation과 맞는다.

## 4-3. Query Simulator data

Simulator는 document from corpus를 sampling하고 answerable query를 만든다. Generated query는 source document와 relation이 있어야 하며 existing demand와 충분히 달라야 한다.

Practical implementation에서는 아래 filter가 필요하다.

- Source에서 answerability 확인
- Query에 answer leakage가 없는지 확인
- Existing queries와 dissimilarity 확인
- Domain과 difficulty balance 확인
- Duplicate 제거
- Holdout separation 확인

Simulator query를 optimizer and evaluation에 동시에 쓰면 leakage가 생긴다. Independent test query를 유지해야 한다.

## 4-4. Engineering notes

### 1) Key budget을 고정해야 한다

Document마다 key 수가 계속 늘어나면 max score opportunity and index cost가 함께 증가한다. Key count, total token, embedding count를 budget으로 통제해야 한다.

### 2) Validation model diversity가 필요하다

Same LLM이 query, diagnosis, revision, faithfulness judgment를 모두 하면 shared blind spot이 생길 수 있다. Independent entailment model or rule-based source check를 추가하는 편이 안전하다.

### 3) Regression set을 separate해야 한다

Current failure query에서 좋아져도 previously solved query가 나빠질 수 있다. Per-document and global regression test가 필요하다.

### 4) Online and offline cost를 분리해야 한다

Search agent call cost가 줄어도 index evolution에 large LLM calls가 많이 들 수 있다. Deployment TCO에는 amortization horizon을 포함해야 한다.

### 5) Query distribution drift를 감시해야 한다

Simulator가 current traffic보다 특정 domain or writing style을 과대표현하면 index가 synthetic distribution에 맞춰질 수 있다.

### 6) Source document는 immutable해야 한다

Key는 retrieval entry point이고 source of truth가 아니다. Final answer and citation은 original document evidence에 grounded되어야 한다.

# 5. Evaluation

## 5-1. BRIGHT retrieval

Project page가 보고하는 BRIGHT average nDCG@10은 다음과 같다.

| Retriever | Base index | SELF-INDEX | Relative gain |
| --- | ---: | ---: | ---: |
| BM25 | 14.5 | 20.4 | 40.4% |
| BGE | 13.9 | 21.8 | 57.0% |
| Qwen3-Emb-8B | 18.8 | 26.1 | 38.8% |

Relative gain은 percentage point가 아니다. 예를 들어 BGE는 13.9에서 21.8로 7.9 points 올라가고, base 대비 relative 57.0%다.

Paper는 Doc2Query, SPIKE, RL-Index 계열과 비교하며 sparse and dense retriever 모두에서 improvement를 보고한다.

## 5-2. Table retrieval

Spider2, FIBEN, BEAVER table retrieval에서도 BM25, BGE, Qwen3 embedding route에서 average nDCG@10 improvement를 보고한다.

Table retrieval은 natural-language document와 key design이 다르기 때문에, cross-domain consistency를 확인하는 데 의미가 있다.

## 5-3. Search agent

BrowseComp-Plus에서 Kimi-K2.5 plus BM25 setting은 base index 대비 reported relative change가 다음과 같다.

- Accuracy는 +41.1%
- Project summary 기준 online cost는 -24.0%

Detailed table에서는 search-call and cost definition에 따라 value가 다르게 보일 수 있으므로 exact denominator를 확인해야 한다.

Index improvement가 agent에게 주는 효과는 두 가지다.

- Correct evidence가 earlier rank에 올라와 answer accuracy가 좋아진다.
- Search iteration and retrieval call이 줄어 online cost가 내려갈 수 있다.

## 5-4. Agent memory

LongMemEval-V2의 Query-to-Slice baseline은 overall score 0.415에서 0.472로 올라가며 relative 13.9% improvement를 보고한다.

이는 self-evolving key idea가 external knowledge search뿐 아니라 past interaction memory retrieval에도 적용될 수 있음을 보여준다.

## 5-5. What really matters in the experiments

### 1) Average gain and per-dataset regression을 분리해야 한다

Overall average가 올라가도 일부 subset은 내려갈 수 있다. Index evolution은 regression-free guarantee가 아니다.

### 2) Key count and index size를 같이 봐야 한다

More keys가 retrieval quality를 올렸다면 storage and scoring cost도 변한다. Same key budget or same index token budget comparison이 중요하다.

### 3) Offline evolution cost는 online cost table에 들어가지 않는다

Project page도 online cost excludes offline index construction이라고 명시한다. Search call saving과 optimizer LLM cost를 분리해 해석해야 한다.

### 4) Test query independence가 핵심이다

Query Simulator and Optimizer가 test demand에 가까운 query를 보았다면 gain이 overestimated될 수 있다. Main result는 unseen test query에서 평가했다고 하지만 split construction을 확인해야 한다.

# 6. Limitations

1. **Official code가 아직 공개되지 않았다.**
   - 2026-09-21 repository에는 "The code will be released soon"만 있다.
   - Prompt, decoding, update budget, key schema를 independent하게 재현하기 어렵다.

2. **Work-in-progress preprint다.**
   - Method and result table이 revision될 수 있다.
   - Peer-reviewed acceptance로 표현하면 안 된다.

3. **LLM generation and validation이 shared bias를 가질 수 있다.**
   - Same model family가 revision and faithfulness를 모두 담당하면 plausible hallucination을 함께 통과시킬 수 있다.

4. **Max-over-keys lottery effect가 있다.**
   - Key를 늘릴수록 chance match가 늘어난다.
   - Quality gain과 key budget expansion을 분리해야 한다.

5. **Synthetic query distribution에 overfit될 수 있다.**
   - Query Simulator가 실제 user demand를 충분히 반영하지 못할 수 있다.

6. **Offline cost and maintenance complexity가 크다.**
   - Selective update라도 continuous diagnosis, validation, embedding refresh, regression testing이 필요하다.

7. **Index key는 source evidence가 아니다.**
   - Generated key가 retrieval을 돕더라도 final claim은 original document span에서 검증해야 한다.

# 7. My Take

## 7-1. Why this matters for my work

SELF-INDEX의 가장 좋은 framing은 "RAG가 틀릴 때 query만 고치지 말고 index도 학습 가능한 artifact로 보자"는 것이다.

Production RAG에서 recurring miss는 자주 나타난다.

- User term과 internal document term의 mismatch
- Product code와 natural language의 mismatch
- Korean-English terminology variation
- 노출되지 않은 table column semantics
- Long document의 buried fact
- Memory event의 implicit goal과 outcome

이 failure를 query-time rewrite로 계속 지불하는 대신 validated index key로 amortize할 수 있다.

## 7-2. Reuse potential

### 1) Document AI evidence retrieval

Page, table, field, clause마다 multi-key representation을 만들고, failed query에서 responsible key만 revise할 수 있다.

### 2) Korean enterprise search

English acronym, Korean formal term, user colloquial term을 document key set에 함께 관리할 수 있다. 단 source-faithful synonym만 허용해야 한다.

### 3) Agent memory

Interaction transcript를 raw text만 index하지 않고 goal, decision, failure, outcome, unresolved issue key로 versioning할 수 있다.

### 4) Evaluation-first implementation

Full paper implementation보다 아래 minimal loop가 먼저 필요하다.

1. Failure query 수집
2. Responsible document 진단
3. Candidate key 생성
4. Source entailment 검사
5. Held-out regression test 수행
6. Versioned index update or rollback

## 7-3. Follow-up papers

- Doc2Query: document indexing을 위한 synthetic query expansion
- SPIKE and EnrichIndex: LLM 기반 index enrichment
- RL-Index: reinforcement learning 기반 index optimization
- BRIGHT: reasoning-intensive retrieval benchmark
- LongMemEval: long-term conversational memory 평가

# 8. Summary

- SELF-INDEX는 document source를 바꾸지 않고 index key set을 evolve한다.
- Optimizer는 Self-Diagnosis, Self-Revision, Self-Validation의 three-stage loop를 사용한다.
- Query Simulator는 observed query 밖의 information need를 proactively 탐색한다.
- BRIGHT에서 BM25 +40.4%, BGE +57.0%, Qwen3 embedding +38.8% relative nDCG@10 gain을 보고한다.
- Search agent and memory system에서도 accuracy and efficiency improvement를 보고한다.
- Code not yet released, offline evolution cost, key-budget lottery, synthetic-query overfitting은 중요한 한계다.
