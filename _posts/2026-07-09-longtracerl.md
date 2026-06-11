---
layout: single
title: "LongTraceRL: Learning Long-Context Reasoning from Search Agent Trajectories with Rubric Rewards Review"
categories: Study-concept
tag: [LongContext, RLVR, SearchAgent]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2605.31584)

이 글의 중심 질문은 "긴 context에서 맞는 답을 낸 모델이 정말 맞는 경로로 읽었는가" 이다.

LongTraceRL은 long-context reasoning RL을 단순히 더 긴 prompt와 더 큰 모델 문제로 보지 않는다. 저자들이 보는 병목은 두 가지다. 첫째, 기존 long-context RL 데이터의 distractor가 너무 쉽다. 무작위로 끼워 넣은 문서는 실제 검색 과정에서 모델이 헷갈리는 문서와 다르다. 둘째, final answer만 보는 outcome reward는 너무 sparse하다. 특히 context가 100K token 이상으로 길어지면, 모델이 우연히 정답을 맞히거나 중간 evidence를 잘못 따라가도 reward는 똑같이 들어간다.

그래서 이 논문은 데이터와 reward를 같이 바꾼다. 데이터 쪽에서는 knowledge graph random walk로 multi-hop question을 만들고, search agent가 실제로 검색하면서 읽은 trajectory에서 distractor를 뽑는다. Reward 쪽에서는 reasoning chain에 등장해야 하는 gold entity를 rubric으로 두고, 정답을 맞힌 response 안에서만 entity-level process reward를 준다.

> 한 줄 요약: LongTraceRL은 search agent trajectory에서 hard distractor를 만들고, gold entity 기반 rubric reward를 positive-only 방식으로 결합해 long-context reasoning RL의 data quality와 reward sparsity 문제를 동시에 다루는 논문이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- long-context RL에서 데이터 난도가 왜 중요한지 꽤 선명하게 보여준다.
- final answer reward만으로는 reasoning path를 통제하기 어렵다는 문제를 entity-level reward로 풀어낸다.
- search agent trajectory를 학습 데이터 생성에 재활용하는 방식이 deep search agent와 RAG 학습에도 바로 연결된다.

**핵심 메시지는 final answer accuracy가 아니라 evidence-grounded reasoning trace를 어떻게 학습시킬 것인가에 있다.**

# 1. Problem Setting

## 1-1. Problem definition

- 이 논문이 다루는 문제는 long-context multi-hop QA에서 모델이 긴 문맥 안의 핵심 evidence를 찾아 연결하고, 그 reasoning chain에 근거해 최종 답을 내도록 학습하는 것이다.
- 입력 context는 단순히 긴 문서 하나가 아니라, gold evidence와 많은 distractor가 섞인 long context다.
- 실험 설정에서는 128K token prompt와 32K token response budget을 합쳐 최대 160K token context로 RL을 수행한다.
- 핵심 목표는 "정답만 맞히는 모델"이 아니라, 관련 evidence를 포괄적으로 찾아서 reasoning에 반영하는 모델이다.

## 1-2. Why previous approaches are insufficient

- 기존 long-context synthetic data는 short-context QA에 random distractor를 붙이거나, 문서 chunk를 단순 확장하는 경우가 많다.
- 이런 distractor는 실제 search agent가 헷갈리는 문서와 다르기 때문에, 모델이 진짜로 evidence selection을 배웠는지 확인하기 어렵다.
- RLVR 계열은 final answer correctness를 reward로 쓰기 쉽지만, long-context에서는 reward가 매우 sparse해진다.
- 특히 긴 context에서 모델이 틀린 중간 hop을 따라가도 우연히 final answer를 맞힐 수 있다.
- chunk-level reward나 document-level reward는 process signal을 보강하지만, 어떤 entity chain을 따라갔는지까지 촘촘히 보는 것은 아니다.

**즉 이 논문은 long-context RL의 병목을 model scale보다 training signal design 문제로 본다.**

# 2. Core Idea

## 2-1. Main contribution

LongTraceRL의 기여는 크게 두 개다.

1. Search agent trajectory 기반 data construction
   - Knowledge graph random walk로 multi-hop question을 만들고, search agent가 실제로 검색한 trajectory에서 distractor를 뽑는다.
   - Agent가 열어봤지만 최종 answer citation에는 쓰지 않은 문서를 Tier-1 distractor로 둔다.
   - Search result에는 나왔지만 열어보지 않은 문서를 Tier-2 distractor로 둔다.

2. Entity-level rubric reward
   - 각 question의 reasoning chain에 필요한 gold entity들을 rubric으로 사용한다.
   - Model response가 gold entity를 얼마나 회수했는지를 process signal로 본다.
   - 다만 이 reward는 final answer가 맞은 response에만 적용한다.

논문 표현을 빌리면 핵심은 "trajectory-based tiered distractors"와 "entity-level rubric reward"의 결합이다.

## 2-2. Design intuition

이 설계의 직관은 꽤 좋다.

- 좋은 distractor는 임의로 뽑은 문서가 아니라, 실제 agent가 읽을 만큼 그럴듯한 문서여야 한다.
- 좋은 process reward는 모델의 chain-of-thought 문장을 직접 평가하기보다, 반드시 거쳐야 할 entity chain을 봐야 한다.
- 좋은 reward shaping은 아무 response에나 process 점수를 주면 안 된다. 그렇지 않으면 모델이 정답 없이 entity만 나열하는 방향으로 reward hacking할 수 있다.

**LongTraceRL은 data hardness와 reward hacking을 같은 프레임 안에서 다룬다.**

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | 긴 context에서 evidence-grounded multi-hop reasoning을 강화 |
| Data source | KILT Wikipedia snapshot 기반 knowledge graph와 search agent trajectory |
| Key data idea | Tier-1 / Tier-2 distractor를 trajectory에서 추출 |
| Key reward idea | Gold entity recall 기반 rubric reward |
| RL algorithm | GRPO |
| Main guardrail | Positive-only reward combination |

## 3-2. Module breakdown

### 1) Multi-hop question generation

- 먼저 Wikipedia hyperlink graph 위에서 controlled random walk를 수행해 entity path를 만든다.
- 논문은 각 step에서 LLM이 최대 5개의 unvisited candidate 중 다음 entity를 고르는 방식으로 path를 구성한다고 설명한다.
- 이후 powerful LLM을 사용해 path 전체를 따라가야만 풀 수 있는 multi-hop question을 만든다.
- Question synthesis prompt는 simple keyword matching으로 답을 찾기 어렵게 identifying information을 paraphrase하도록 요구한다.
- 이 과정에서 final answer뿐 아니라 중간 gold entity list도 같이 얻는다.

이 gold entity list가 뒤에서 reward rubric으로 쓰인다.

### 2) Agent search trajectory collection

Search agent는 question을 풀기 위해 다음 행동들을 수행한다.

- search
- open
- cite

저자들은 agent trajectory를 기록하고, final answer가 correct인 trajectory만 남긴다. 이렇게 하면 distractor가 단순 잡음이 아니라, 실제 search process에서 등장한 문서가 된다.

중요한 점은 "agent가 틀릴 뻔한 경로"가 학습 데이터의 hard negative가 된다는 것이다.

### 3) Tiered distractor extraction

논문은 retrieved document를 두 tier로 나눈다.

- Tier-1 distractor: agent가 열어봤지만 final response에서 cite하지 않은 문서
- Tier-2 distractor: search result에는 있었지만 agent가 열어보지 않은 문서

Tier-1은 topic relevance가 높고, 실제로 agent가 읽을 가치가 있다고 판단한 문서다. 그래서 random distractor보다 훨씬 헷갈린다.

Long context assembly에서는 gold passage를 넣은 뒤 Tier-1을 먼저 채우고, 남은 budget을 Tier-2로 채운다. 그리고 document order는 positional shortcut을 막기 위해 shuffle한다.

### 4) Rubric reward

Outcome reward는 final short answer가 맞는지 본다. Rubric reward는 response가 gold entity를 얼마나 포함하는지 본다.

개념적으로는 아래처럼 이해할 수 있다.

$$
r_{rubric} = \frac{|E_{response} \cap E_{gold}|}{|E_{gold}|}
$$

여기서 $E_{gold}$는 question의 reasoning chain에 필요한 gold entity 집합이고, $E_{response}$는 response에서 회수된 entity 집합이다.

GRPO에서는 같은 question에 대해 여러 rollout을 샘플링하므로, 논문은 group-level normalization으로 question별 난도 차이를 줄인다.

### 5) Positive-only reward combination

Rubric reward를 모든 response에 주면 문제가 생긴다. 모델이 정답을 맞히지 않고도 context 안의 entity를 많이 나열해 점수를 얻을 수 있기 때문이다.

그래서 LongTraceRL은 final answer가 correct인 response에만 rubric reward를 준다.

개념적으로는 아래처럼 볼 수 있다.

$$
r_{total} = r_{outcome} + eta \cdot I(r_{outcome} > 0) \cdot \hat{r}_{rubric}
$$

이 식은 논문 수식을 그대로 재현한다기보다, positive-only 전략의 의미를 읽기 쉽게 표현한 것이다. 정확한 notation은 원문 수식 확인이 필요하다.

**이 설계의 핵심은 process reward를 correctness gate 뒤에 둔다는 점이다.**

# 4. Training / Data / Recipe

## 4-1. Data

Training set은 2,815개의 long-context QA example로 구성된다.

- 각 example은 eight-hop question을 포함한다.
- Gold evidence passage는 Wikipedia 기반이다.
- Context는 tiered distractor와 함께 128K token target length로 조립된다.
- Rubric annotation은 gold entity chain에서 나온다.

비교 데이터셋은 다음 세 가지다.

| Dataset | Size | Context |
| --- | ---: | --- |
| DocQA | 1,591 | 2K to 20K |
| LoongRL | 15,000 | 16K |
| LongRLVR | 18,870 | 8K to 64K |
| LongTraceRL | 2,815 | 128K target |

숫자만 보면 LongTraceRL 데이터는 다른 baseline보다 작다. 하지만 이 논문이 강조하는 것은 quantity가 아니라 confusability다.

## 4-2. Training strategy

실험 대상 모델은 세 가지다.

| Model | Type |
| --- | --- |
| Qwen3-4B-Thinking-2507 | 4B dense reasoning model |
| DeepSeek-R1-0528-Qwen3-8B | 8B dense distilled reasoning model |
| Qwen3-30B-A3B-Thinking-2507 | 30B total, 3B active MoE model |

Training setup은 다음과 같다.

| Item | Value |
| --- | --- |
| Framework | Slime |
| RL algorithm | GRPO |
| Context length | 128K prompt + 32K response |
| Global batch size | 128 |
| Training iterations | 200 |
| Learning rate | 2e-6 constant |
| Rubric reward weight | eta = 0.3 |
| Rollout temperature | 1.0 |
| Eval temperature | 0.6 |
| Checkpoint interval | every 20 steps |
| Hardware | 32 H800 GPUs |

## 4-3. Engineering notes

- 공식 GitHub는 dataset, model checkpoints, training scripts, evaluation scripts를 공개한다.
- 공개 model은 LongTraceRL-4B, LongTraceRL-8B, LongTraceRL-30B로 정리되어 있다.
- Full 128K context training에는 README 기준 4 nodes x 8 GPUs 수준의 H800 80GB 환경이 필요하다.
- Reward server는 outcome reward와 rubric reward를 제공하며, LLM judge API endpoint를 설정하는 방식으로 동작한다.
- Eval script는 SGLang 기반 only-eval mode를 제공한다.

**재현 관점에서 이 논문은 아이디어만 던지는 것이 아니라, 실제 training/eval entry point까지 공개한 편이다.**

# 5. Evaluation

## 5-1. Main results

평가는 다섯 개 long-context benchmark에서 수행된다.

- AA-LCR
- MRCR
- FRAMES
- LongBench V2
- LongReason

대표 결과를 평균 점수 중심으로 보면 다음과 같다.

| Backbone | Base | Strongest baseline | LongTraceRL-GRPO | LongTraceRL |
| --- | ---: | ---: | ---: | ---: |
| DeepSeek-R1-0528-Qwen3-8B | 42.7 | 40.9 | 42.9 | 43.8 |
| Qwen3-4B-Thinking-2507 | 53.3 | 56.5 | 53.7 | 59.0 |
| Qwen3-30B-A3B-Thinking-2507 | 60.5 | 63.3 | 62.3 | 63.7 |

Qwen3-4B에서 가장 뚜렷하다.

- Base 평균은 53.3이다.
- LongRLVR는 56.5이다.
- LongTraceRL은 59.0이다.
- AA-LCR은 33.2에서 41.8로 오른다.

저자들은 Qwen3-4B 기준 base 대비 +5.7 point, strongest baseline 대비 +2.5 point를 보고한다.

## 5-2. Ablation 1: rubric reward weight

Rubric reward weight는 eta = 0.3이 가장 좋게 나온다.

| Setting | Avg |
| --- | ---: |
| Base | 53.3 |
| LongTraceRL eta = 0.1 | 58.3 |
| LongTraceRL eta = 0.3 | 59.0 |
| LongTraceRL eta = 0.5 | 57.1 |

이 결과는 process signal이 필요하지만, 너무 강하면 outcome objective를 흐릴 수 있음을 보여준다.

## 5-3. Ablation 2: source of distractors

Distractor construction도 결과 차이를 만든다.

| Strategy | Avg |
| --- | ---: |
| random | 55.7 |
| search | 56.7 |
| traj-random | 57.4 |
| traj-tiered | 59.0 |

Random distractor보다 trajectory 기반 distractor가 좋고, trajectory 내부에서도 confusability를 반영해 Tier-1을 우선하는 방식이 가장 좋다.

더 흥미로운 것은 distractor가 rubric entity와 얼마나 겹치는지다.

| Strategy | Macro Avg overlap |
| --- | ---: |
| random | 1.35 |
| search | 15.00 |
| traj-random | 42.16 |
| traj-tiered | 50.03 |
| Tier-1 only inside traj-tiered | 63.23 |

이 표는 hard distractor가 그냥 비슷한 문서가 아니라, reasoning chain 근처의 entity를 포함해야 진짜로 어렵다는 것을 보여준다.

## 5-4. Ablation 3: positive-only strategy

Positive-only를 제거하고 incorrect response에도 rubric reward를 주면 성능이 떨어진다.

| Strategy | AA-LCR | MRCR | FRAMES | LongBench V2 | LongReason | Avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| positive and negative | 37.0 | 40.5 | 79.5 | 45.5 | 83.1 | 57.1 |
| positive-only | 41.8 | 45.8 | 79.5 | 44.1 | 83.8 | 59.0 |

즉 rubric reward는 dense signal이지만, gate 없이 쓰면 entity enumeration shortcut을 만들 수 있다.

## 5-5. What really matters in the experiments

이 논문의 가장 중요한 결과는 최종 score보다 세 가지 ablation이다.

1. Rubric reward 제거 시 Qwen3-4B 평균이 59.0에서 53.7로 떨어진다.
2. Trajectory-tiered distractor가 random/search/traj-random보다 낫다.
3. Positive-only gate가 없으면 rubric reward가 reward hacking 방향으로 흐를 수 있다.

**따라서 LongTraceRL의 성능은 GRPO 자체보다 data hardness + entity-level process reward + reward gating의 조합에서 나온다.**

# 6. Limitations

1. Knowledge source가 KILT Wikipedia snapshot에 묶여 있다. 논문은 downstream benchmark transfer를 보여주지만, training data의 reasoning pattern diversity는 제한될 수 있다.
2. Distractor quality는 search agent capability에 의존한다. 더 강한 agent 또는 약한 agent를 쓰면 trajectory distribution이 달라질 수 있다.
3. Gold entity recall은 유용하지만 완전한 reasoning correctness는 아니다. 모델이 entity를 언급했더라도 실제 추론이 맞는지는 별도 문제다.
4. Outcome reward에는 LLM judge가 들어간다. Judge error나 prompt sensitivity가 전체 RL signal에 영향을 줄 수 있다.
5. Full training은 32 H800 GPU와 160K token setup을 요구하므로, 작은 팀이 그대로 재현하기에는 비용이 크다.
6. 2,815개 training example은 정교하지만, 더 넓은 domain에서 같은 pipeline이 어느 정도 유지되는지는 추가 검증이 필요하다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문은 long-context RL에서 가장 실용적인 질문을 찌른다.

"정답을 맞혔는가"가 아니라 "그 정답을 맞히기 위해 필요한 evidence path를 실제로 따라갔는가"를 묻는다.

RAG, deep search, long-context agent를 만들다 보면 final answer accuracy만으로는 모델의 retrieval failure를 잡기 어렵다. LongTraceRL은 이 문제를 search trajectory와 rubric entity로 분해한다.

**실무적으로는 evaluator보다 data generator를 먼저 의심해야 한다는 메시지가 강하다.**

## 7-2. Reuse potential

바로 가져갈 수 있는 아이디어는 네 가지다.

1. Search trajectory를 hard negative source로 쓰기
   - Agent가 열어봤지만 cite하지 않은 문서는 매우 좋은 distractor 후보가 된다.

2. Gold entity chain을 rubric으로 쓰기
   - Long-context QA뿐 아니라 document QA, legal QA, scientific QA에서도 entity/event/table-cell chain을 rubric으로 만들 수 있다.

3. Positive-only process reward
   - Process reward를 그냥 더하면 reward hacking이 생긴다. Correctness gate 뒤에 넣는 방식이 중요하다.

4. Distractor difficulty를 overlap으로 진단하기
   - Random negative가 충분히 어렵다는 착각을 줄일 수 있다.

## 7-3. Follow-up papers

- LongRLVR: long-context RL에서 verifiable context reward를 다룬 baseline이다.
- LongR: long-context reasoning에서 dense utility reward를 다룬다.
- QwenLong-L1 / QwenLong-L1.5: long-context reasoning post-training recipe 비교에 좋다.
- DeepDive: knowledge graph와 deep search agent trajectory를 함께 보는 후속 흐름으로 연결된다.
- NExtLong: long document 없이 long-context training signal을 만드는 관점에서 비교할 만하다.

# 8. Summary

- LongTraceRL은 long-context RL을 data construction과 reward design의 결합 문제로 본다.
- Data 쪽에서는 search agent trajectory에서 Tier-1 / Tier-2 distractor를 뽑아 random distractor보다 어려운 context를 만든다.
- Reward 쪽에서는 gold entity recall 기반 rubric reward를 쓰되, final answer가 맞은 response에만 적용한다.
- Qwen3-4B에서는 평균 53.3에서 59.0으로 오르고, strongest baseline 56.5도 넘는다.
- 가장 재사용 가치가 높은 부분은 trajectory-based hard distractor와 positive-only process reward 설계다.
