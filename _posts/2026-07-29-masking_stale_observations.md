---
layout: single
title: "Masking Stale Observations Helps Search Agents -- Until It Doesn't: A Regime Map and Its Mechanism Review"
categories: Study-concept
tag: [AI-Agent, Search-Agent, Context-Management]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2606.00408)

[Code link](https://github.com/i-DeepSearch/observation-masking)

[Eval logs](https://huggingface.co/datasets/i-DeepSearch/observation-masking-eval-logs)

> 한 줄 요약: 이 논문은 long-horizon search agent에서 stale observation을 active context에서 masking하는 간단한 context management가 언제 도움이 되고 언제 해로운지, model capacity와 retriever quality의 상호작용으로 설명하는 regime map을 제안한다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Agentic search, deep research, RAG agent에서 context budget은 실제 병목이다.
- Observation masking은 구현이 단순하지만, 논문은 이 기법이 항상 이득이라는 가정을 깨고 regime-dependent intervention으로 다시 정의한다.
- 단순한 평균 accuracy 비교보다, retriever recall, model의 implicit filtering capacity, search turn 수, evidence eviction을 함께 봐야 한다는 점을 보여준다.
- 평가 로그와 scaffold가 공개되어 있어 production RAG agent나 browsing agent의 context policy를 진단하는 출발점으로 쓰기 좋다.

이 논문의 핵심 메시지는 "masking을 켜면 context가 줄어드니 좋다"가 아니다. 더 정확히는 다음 질문이다.

> 현재 agent가 실패하는 이유가 context 안의 noise를 못 거르는 것인가, 아니면 아직 evidence 자체를 충분히 못 찾은 것인가?

이 차이를 구분하지 않으면 observation masking은 비용 최적화가 아니라 evidence deletion이 될 수 있다.

# 1. Problem Setting

## 1-1. Problem definition

Long-horizon search agent는 여러 번의 search, open, browse, reasoning step을 반복한다. 이 과정에서 매 turn마다 tool observation이 누적되고, active context는 다음 요소들로 빠르게 커진다.

- 이전 search query
- 검색 결과 snippet
- 열어본 page content
- intermediate reasoning
- 이미 확인했지만 최종 답에는 덜 중요한 evidence 후보
- 실제 정답에 필요한 crucial evidence

문제는 이들이 모두 같은 중요도를 갖지 않는다는 점이다. 어떤 observation은 초반에는 필요하지만 이후에는 거의 쓰이지 않는다. 반대로 어떤 observation은 오래전에 나온 page라도 최종 reasoning에서 결정적인 evidence일 수 있다.

Observation masking은 이 문제에 대한 가장 단순한 개입이다. 일정 window 밖의 오래된 observation을 active context에서 가리거나 archive해서, 모델이 보는 input을 줄인다. 이렇게 하면 context budget을 아껴 더 많은 turn을 수행할 수 있다. 하지만 동시에 정답에 필요한 evidence까지 지울 위험이 생긴다.

따라서 이 논문의 problem setting은 다음처럼 정리할 수 있다.

| Question | Meaning |
| --- | --- |
| When does masking help? | 오래된 observation을 지우는 것이 noise removal이 되는 구간은 어디인가 |
| When does masking fail? | 오래된 observation이 실제 evidence라서 지우면 성능이 떨어지는 구간은 어디인가 |
| What controls the boundary? | model capacity, retriever recall, active context complexity 중 무엇이 중요한가 |
| How should we evaluate it? | 단일 model 평균이 아니라 model x retriever x benchmark sweep으로 봐야 하는가 |

## 1-2. Why previous approaches are insufficient

기존 context management 접근은 보통 세 가지 방향으로 간다.

1. Full context retention
   - 모든 observation을 계속 들고 간다.
   - evidence loss 위험은 낮지만, context cost와 attention noise가 커진다.

2. Summarization
   - 오래된 observation을 요약해 context를 줄인다.
   - 정보 압축이 가능하지만, summarizer가 evidence를 잘못 버리거나 왜곡할 수 있다.

3. Observation masking or truncation
   - 오래된 observation을 active context에서 제거한다.
   - 구현은 쉽지만, 어떤 observation이 stale인지 판단이 단순하다.

이 논문이 지적하는 핵심은 세 방법의 우열이 고정되어 있지 않다는 점이다. 특히 masking은 다음 두 조건이 동시에 맞을 때 가장 도움이 된다.

- retriever가 answer-supporting evidence를 어느 정도 찾아온다.
- model은 그 evidence와 noise를 active context에서 완전히 분리할 만큼 충분히 강하지 않다.

반대로 retriever가 약해서 context 안에 evidence 자체가 부족하면 masking은 구할 것이 없다. 또한 model이 이미 강해서 긴 context 속 evidence를 잘 걸러낼 수 있으면, aggressive masking은 오히려 중요한 evidence를 지워 성능을 떨어뜨릴 수 있다.

즉 context management는 단순히 token 수를 줄이는 문제가 아니라, evidence availability와 evidence filtering 사이의 tradeoff다.

# 2. Core Idea

## 2-1. Main contribution

이 논문의 핵심 기여는 observation masking의 효과를 single average gain으로 보지 않고, regime map으로 해석했다는 점이다.

논문 abstract 기준으로 저자들은 4B부터 284B까지의 다양한 agent backbone과 3가지 retriever를 사용해 offline 및 live-web agentic search benchmark에서 observation masking을 비교한다. 그리고 masking gain을 No-CM baseline accuracy에 대해 그리면 asymmetric inverted-U 형태가 나온다고 보고한다.

이 패턴은 대략 세 구간으로 읽을 수 있다.

| Regime | Condition | Masking effect | Interpretation |
| --- | --- | --- | --- |
| Retriever bottleneck | Weak retriever, low evidence recall | small or limited gain | context를 줄여도 answer evidence가 부족하다 |
| Middle regime | Strong retriever, mid-capacity model | largest gain | useful evidence는 있지만 model이 noise를 충분히 못 거른다 |
| Saturated model | Strong model, high baseline accuracy | small gain or collapse | masking이 이미 활용 가능했던 evidence를 지울 수 있다 |

이 관점이 중요한 이유는 observation masking을 default engineering trick이 아니라 diagnostic intervention으로 바꾸기 때문이다. masking을 켰을 때 rescued case가 많다면, 모델은 context 안의 useful signal을 찾는 데 어려움을 겪고 있을 수 있다. 반대로 harmed case가 많다면, masking window가 evidence를 너무 빨리 제거하거나 retriever와 agent가 이미 충분히 강한 구간일 수 있다.

## 2-2. Design intuition

설계 직관은 token-for-turn tradeoff다.

Observation masking은 active context에서 오래된 observation을 제거한다. 그 결과 같은 context budget 안에서 agent는 더 많은 search turn을 수행할 수 있다. 이때 tradeoff는 다음과 같다.

- Benefit: active context noise가 줄고, 추가 search turn을 확보한다.
- Risk: 예전에 찾은 crucial evidence가 final answer context에서 사라진다.

따라서 masking의 효과는 "지운 token" 자체보다 "확보한 turn이 실패를 성공으로 바꾸었는지"에 달려 있다. 추가 turn이 더 좋은 query, 더 좋은 page open, 더 정확한 evidence collection으로 이어지면 masking은 도움이 된다. 하지만 추가 turn이 별 이득을 만들지 못하거나, masking으로 evidence가 사라지면 성능은 떨어진다.

이 논문은 context pruning paper라기보다 agent failure analysis paper에 가깝다. Context를 얼마나 줄일 수 있는지보다, agent가 어떤 상황에서 evidence를 기억하고 어떤 상황에서 noise에 묻히는지를 보여주는 데 가치가 있다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Long-horizon search agent에서 observation masking이 유효한 regime을 찾는다 |
| Intervention | Stale observation을 active context에서 masking하고, 최근 window와 필요한 state만 유지한다 |
| Comparison | No CM vs CM paired run |
| Variables | Model backbone, retriever, benchmark, masking window |
| Main mechanism | token-for-turn tradeoff, implicit filtering capacity, evidence eviction |
| Released artifact | Search scaffold, evaluation logs, trajectory records |

여기서 CM은 context management를 뜻한다. 논문과 공개 로그에서는 No CM과 CM을 paired setting으로 비교한다. CM은 observation mask context management가 적용된 run을 의미한다.

## 3-2. Module breakdown

### 1) Search agent scaffold

공개 GitHub 기준 scaffold는 browser execution과 model serving을 분리한다. Shared browser pool이 search, page opening, observation collection을 담당하고, model worker가 masking window가 적용된 trajectory를 소비한다.

이 설계는 연구적으로 중요하다. 동일한 browsing environment와 CM policy 아래에서 model-retriever pair를 바꾸어 비교할 수 있기 때문이다. Context management 효과를 보려면 agent scaffold 차이가 결과에 섞이면 안 된다.

### 2) Observation masking policy

Observation masking은 오래된 tool observation을 active model context에서 제거하거나 압축된 상태로 처리한다. 다만 공개 repo는 trajectory preservation을 지원한다. 즉 active context에서 지워진 observation이 분석 자체에서 사라지는 것은 아니다.

이 차이가 중요하다.

- Active context: 모델이 다음 turn에서 실제로 보는 context
- Full trajectory: 사후 분석과 evaluation을 위해 저장되는 전체 record

Production agent에서도 이 분리는 유용하다. 모델에게 모든 것을 보여주지 않더라도, debug, audit, retrieval recall analysis를 위해 full trace는 보존해야 한다.

### 3) Retriever and benchmark sweep

논문은 masking 효과가 retriever와 model capacity의 상호작용이라고 본다. 따라서 단일 retriever, 단일 model, 단일 benchmark로는 결론을 내기 어렵다.

공개 repo와 eval logs에서 확인되는 benchmark는 다음과 같다.

| Benchmark | Size | Language | Search backend |
| --- | ---: | --- | --- |
| BrowseComp-Plus | 830 | EN | local |
| xBench | 100 | ZH | Serper |
| GAIA-text | 103 | EN | Serper |
| BrowseComp-ZH | 289 | ZH | Serper |

BrowseComp-Plus는 local BM25, Qwen3-Embedding-8B, AgentIR 같은 retriever를 사용하고, 다른 benchmark들은 online search log 기반으로 비교된다.

### 4) Mechanistic analysis

논문과 공개 README는 masking 효과를 설명하기 위해 몇 가지 분석 축을 둔다.

1. Sparse signal and complex inputs
   - useful signal이 sparse하고 input trace가 복잡할수록 masking이 도움이 되는지 본다.

2. Reasoning attention over tool results
   - reasoning token이 tool-result token에 얼마나 attention을 두는지 본다.
   - CM gain이 큰 setting에서는 useful tool result에 대한 attention이 유지되는 경향을 본다.

3. Page reopening behavior
   - agent가 예전에 열었던 page를 다시 여는 위치 분포를 본다.
   - middle page를 덜 다시 여는 lost-in-the-middle pattern이 관찰된다.

이 분석은 masking을 단순히 token saving으로 보지 않고, agent behavior가 어떤 evidence를 다시 참조하는지와 연결한다.

# 4. Training / Data / Recipe

## 4-1. Data

이 논문은 새로운 model training objective를 제안하는 논문이라기보다, search agent scaffold와 context management policy를 평가하는 논문이다. 따라서 Training 섹션은 model fine-tuning보다 evaluation recipe로 보는 편이 맞다.

공개 eval log dataset에는 다음 정보가 포함된다.

| Field | Meaning |
| --- | --- |
| qid | question id |
| question | benchmark question |
| answer or correct_answer | gold answer |
| full_messages | 전체 input message record |
| final_messages | final answer에 사용된 message record |
| turn_stats | turn별 input token, output token, latency |
| retrieved_urls | run에서 retrieved된 URL set |
| correct | judge result |

특히 `final_messages`와 `full_messages`의 차이가 중요하다. CM run에서는 earlier context가 compacted 또는 archived될 수 있기 때문에, 모델이 실제로 본 context와 전체 trajectory가 달라진다.

## 4-2. Training strategy

논문에서 중요한 recipe는 model training이 아니라 paired evaluation이다.

1. 같은 benchmark question을 준비한다.
2. 같은 model-retriever pair에서 No CM run을 수행한다.
3. observation masking을 켠 CM run을 수행한다.
4. final answer correctness, trajectory statistics, retrieved URLs, turn-level token statistics를 비교한다.
5. 결과를 model capacity와 retriever strength에 따라 regime으로 묶는다.

이 구조는 실무에서도 그대로 쓸 수 있다. Production RAG agent에서 context policy를 바꿀 때도 단순히 평균 latency와 token만 보면 안 된다. 최소한 다음 세 그룹으로 나누어야 한다.

- Rescued cases: No CM은 실패, CM은 성공
- Harmed cases: No CM은 성공, CM은 실패
- Unchanged cases: 둘 다 성공 또는 둘 다 실패

이 세 그룹을 분리해야 masking이 실제로 noise를 제거했는지, 아니면 evidence를 지웠는지 볼 수 있다.

## 4-3. Engineering notes

공개 repo 기준으로 실무적으로 유용한 점은 다음과 같다.

1. Search backend를 분리한다.
   - BrowseComp-Plus에서는 BM25, dense retriever, AgentIR를 로컬 search service로 구동할 수 있다.
   - 이 구조는 retriever bottleneck과 model bottleneck을 분리해 보기 좋다.

2. Model serving과 browsing execution을 분리한다.
   - browser pool과 model worker를 분리하면 같은 browsing stack에서 여러 model을 비교하기 쉽다.

3. Masking window를 configurable하게 둔다.
   - context policy는 one-size-fits-all 값이 아니다.
   - model과 retriever가 바뀌면 optimal window도 바뀔 수 있다.

4. Full trajectory를 보존한다.
   - active context를 줄여도 debug record는 남아야 한다.
   - 특히 failed case에서는 어떤 evidence가 언제 사라졌는지 재구성할 수 있어야 한다.

5. Evaluation script와 run directory layout을 명시한다.
   - paired run을 반복하려면 결과 파일 구조가 중요하다.
   - 단일 leaderboard보다 trajectory-level audit이 더 유용하다.

# 5. Evaluation

## 5-1. Main results

공개 Hugging Face eval log card 기준으로, 대표 결과를 뽑아보면 다음과 같다.

| Dataset | Model | Retriever | No CM | CM | Delta |
| --- | --- | --- | ---: | ---: | ---: |
| BrowseComp-Plus | GPT-OSS-20B | AgentIR | 63.3% | 73.3% | +10.0 pp |
| BrowseComp-Plus | Qwen3.5-4B | AgentIR | 48.1% | 58.9% | +10.8 pp |
| BrowseComp-Plus | Qwen3.5-35B-A3B | AgentIR | 62.9% | 74.6% | +11.7 pp |
| BrowseComp-Plus | Qwen3.6-35B-A3B | Qwen3-Emb-8B | 55.9% | 66.3% | +10.4 pp |
| BrowseComp-Plus | GPT-OSS-120B | AgentIR | 79.4% | 79.5% | +0.1 pp |
| BrowseComp-Plus | Tongyi-DeepResearch-30B-A3B | AgentIR | 80.7% | 79.6% | -1.1 pp |
| GAIA-text | GPT-OSS-120B | Serper | 72.8% | 68.0% | -4.8 pp |
| BrowseComp-ZH | DeepSeek-V4-Flash-Max | Serper | 73.4% | 73.4% | +0.0 pp |

이 표에서 가장 중요한 것은 최고 점수가 아니다. 중요한 것은 masking gain이 단조롭지 않다는 점이다.

- 중간 capacity model과 강한 retriever 조합에서는 gain이 크다.
- 이미 강한 baseline에서는 gain이 거의 없거나 음수가 될 수 있다.
- live-web benchmark나 다른 language setting에서는 gain이 더 작게 나타날 수 있다.
- 같은 model이라도 retriever가 바뀌면 gain이 바뀐다.

즉 이 논문은 "CM이 평균적으로 좋다"가 아니라 "CM이 언제 좋은지 map을 그려야 한다"는 쪽에 가깝다.

## 5-2. What really matters in the experiments

### 1) No-CM baseline accuracy가 regime axis다

논문 abstract의 핵심 표현은 masking gain을 No-CM accuracy에 대해 볼 때 asymmetric inverted-U가 나온다는 것이다. 이 해석은 매우 유용하다.

보통 context policy를 비교할 때는 평균 score만 본다. 하지만 이 논문은 baseline agent가 이미 어느 정도 성공하는지에 따라 masking의 의미가 달라진다고 본다.

- 너무 낮은 baseline: evidence가 부족하거나 retrieval이 약하다.
- 중간 baseline: evidence는 있지만 filtering이 어렵다.
- 높은 baseline: 이미 evidence를 잘 활용한다.

따라서 masking은 중간 baseline에서 가장 잘 작동할 수 있다.

### 2) Retriever recall과 model filtering capacity는 분리해서 볼 수 없다

Weak retriever에서는 context 안에 answer-supporting evidence가 부족하다. 이 경우 masking은 noise를 줄일 수는 있어도, 없는 evidence를 만들지는 못한다.

Strong retriever에서는 evidence가 들어올 가능성이 높다. 하지만 model이 그 evidence를 긴 context 안에서 잘 찾지 못하면 masking이 도움이 된다. 반대로 model이 이미 잘 찾는다면 masking은 evidence deletion risk를 만든다.

즉 retriever와 model은 독립된 축이 아니라 상호작용한다.

### 3) Token saving 자체가 목표가 아니다

Observation masking은 context token을 줄인다. 하지만 이 논문에서 중요한 것은 token saving 자체가 아니라, 줄인 token으로 확보한 extra turn이 성공을 만들었는지다.

실무적으로는 다음 지표를 같이 봐야 한다.

- total input tokens
- final_messages_tokens
- number of turns
- retrieved_urls recall
- rescued vs harmed case 비율
- final answer correctness

비용만 낮아지고 harmed case가 늘어나면 좋은 context management가 아니다.

### 4) Lost-in-the-middle은 agent trajectory에서도 나타난다

README의 page reopening 분석은 agent가 middle page를 덜 다시 여는 경향을 보여준다. 이는 long-context LLM의 lost-in-the-middle 문제와 유사하게, agent trajectory에서도 중간에 위치한 evidence가 덜 활용될 수 있음을 시사한다.

Masking은 이 문제를 완전히 해결하지 않는다. 오히려 CM이 U-shaped reopening pattern을 더 날카롭게 만들 수 있다. 따라서 오래된 observation을 지울 때는 recency만 기준으로 삼으면 위험하다.

### 5) Strong model에서는 pruning보다 retrieval fidelity가 더 중요해질 수 있다

Hugging Face paper page의 author comment는 future effort를 aggressive pruning보다 high-fidelity retrieval 쪽으로 옮길 필요를 언급한다. 이 해석은 강한 model 구간에서 특히 중요하다.

강한 model은 긴 context에서 noise를 어느 정도 걸러낼 수 있다. 이때 bottleneck은 context length보다 evidence quality일 수 있다. 이런 구간에서 무리한 masking은 이미 확보한 evidence를 지우는 역효과를 낼 수 있다.

# 6. Limitations

1. Benchmark와 scaffold dependence가 크다.
   - BrowseComp-Plus, xBench, GAIA-text, BrowseComp-ZH는 유용한 benchmark지만 모든 search agent workload를 대표하지는 않는다.
   - 실제 product RAG에서는 user query distribution, document freshness, access control, latency budget이 다르다.

2. Masking window가 중요한 hyperparameter다.
   - 같은 observation masking이라도 window size, archived representation, page reopening policy에 따라 결과가 달라질 수 있다.
   - 논문의 regime map을 그대로 다른 system에 이식하기는 어렵다.

3. Stale observation의 정의가 단순할 수 있다.
   - 오래된 observation이 항상 덜 중요하지는 않다.
   - early page가 final answer의 핵심 evidence일 수 있으므로 recency-only policy는 위험하다.

4. Retriever recall 추정이 쉽지 않다.
   - BrowseComp-Plus처럼 gold document matching이 가능한 경우는 좋지만, live web setting에서는 retrieved_urls가 충분한지 판정하기 어렵다.
   - 실제 서비스에서는 gold evidence가 없는 경우가 많다.

5. Attention analysis는 해석에 주의가 필요하다.
   - reasoning-token attention over tool results는 유용한 diagnostic이지만, attention이 곧 causal evidence use를 뜻한다고 단정하면 안 된다.
   - Mechanism claim은 case study와 ablation을 함께 봐야 한다.

6. 비용과 latency 관점의 정량 분석은 추가 확인이 필요하다.
   - masking은 context token을 줄이지만, 더 많은 turn을 허용하면 total latency가 늘 수도 있다.
   - production setting에서는 accuracy, cost, latency, stability를 같이 봐야 한다.

# 7. My Take

## 7-1. Why this matters for my work

Agentic RAG나 deep-search pipeline을 만들 때 context policy는 자주 heuristic으로 처리된다. 예를 들어 "최근 N개 observation만 유지한다" 또는 "오래된 page는 summary로 바꾼다" 같은 방식이다.

이 논문이 유용한 이유는 그런 heuristic을 평가 가능한 intervention으로 바꾸기 때문이다. 단순히 CM on/off 평균을 보는 것이 아니라, rescued case와 harmed case를 나누고, 그 차이를 retriever와 model capacity 관점에서 해석하게 만든다.

특히 다음 질문을 던지는 데 좋다.

- 우리 agent는 evidence를 못 찾는가, 찾고도 못 쓰는가?
- context가 길어서 실패하는가, retrieved evidence quality가 낮아서 실패하는가?
- masking이 성공을 늘리는가, 아니면 비용만 줄이고 evidence loss를 만드는가?
- model이 커지면 같은 masking policy를 계속 써도 되는가?

실무 관점에서는 이 논문이 context engineering을 좀 더 measurement-driven하게 만드는 데 의미가 있다.

## 7-2. Reuse potential

바로 재사용해볼 만한 포인트는 다음과 같다.

1. Paired evaluation
   - 같은 query set에 대해 No CM과 CM을 모두 돌린다.
   - 평균 score 외에 rescued, harmed, unchanged를 분리한다.

2. Retriever-strength sweep
   - BM25, dense retriever, reranker, agentic retriever를 바꾸어 같은 masking policy를 평가한다.
   - retriever가 약한데 masking만 튜닝하는 실수를 피한다.

3. Evidence-retention audit
   - final answer에 필요한 evidence가 언제 retrieved되었고, CM 후 final context에 남아 있었는지 추적한다.
   - 이것은 hallucination debugging에도 연결된다.

4. Context policy by regime
   - 모든 model에 같은 window를 쓰지 않는다.
   - 작은 model, mid model, strong model 별로 masking aggressiveness를 다르게 둔다.

5. Production monitoring
   - token reduction만 보지 말고 harmed case를 모니터링한다.
   - 특히 high-value query나 compliance-sensitive task에서는 conservative masking을 우선 고려한다.

## 7-3. Follow-up papers

- Simple Observation Masking Is as Efficient as LLM Summarization for Agent Context Management
- AgentIR: Reasoning-Aware Retrieval for Deep Research Agents
- BrowseComp-Plus: A More Fair and Transparent Evaluation Benchmark of Deep-Research Agent
- GrepSeek: Training Search Agents for Direct Corpus Interaction
- LongTraceRL: Learning Long-Context Reasoning from Search Agent Trajectories with Rubric Rewards

# 8. Summary

- 이 논문은 observation masking을 universal context trick이 아니라 regime-dependent intervention으로 재해석한다.
- Masking gain은 weak retriever, mid-capacity model, saturated model 구간에서 다르게 나타난다.
- 핵심 mechanism은 token-for-turn tradeoff다. context를 줄여 extra search turn을 얻지만, crucial evidence를 지울 위험도 생긴다.
- Evaluation은 No CM vs CM paired run, model x retriever x benchmark sweep, trajectory-level analysis를 결합한다.
- 실무적으로는 context policy를 default로 고정하기보다 rescued/harmed case를 나눠 측정하는 방식이 더 안전하다.
