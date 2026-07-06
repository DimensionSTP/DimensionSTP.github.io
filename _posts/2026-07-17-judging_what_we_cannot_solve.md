---
layout: single
title: "Judging What We Cannot Solve: A Consequence-Based Approach for Oracle-Free Evaluation of Research-Level Math Review"
categories: Study-concept
tag: [LLM, Evaluation, MathReasoning, LLMJudge, ResearchAutomation]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2602.06291)

> 한 줄 요약: 이 논문은 research-level math solution을 직접 판정하기 어려울 때, 후보 solution이 주변의 검증 가능한 related question을 더 잘 풀게 만드는지를 utility로 삼아 oracle-free validation을 수행하는 Consequence-Based Utility를 제안한다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- LLM이 어려운 수학 문제에 plausible attempt를 만들 수 있게 되면서, 이제 병목은 생성보다 검증으로 이동하고 있다.
- LLM-as-a-Judge나 reward model은 그럴듯한 문체, authority signal, 장황한 reasoning에 흔들릴 수 있다.
- 논문은 solution 자체를 읽고 점수를 주는 대신, 그 solution이 어떤 downstream consequence를 만드는지 본다.
- AI scientist나 자동 연구 에이전트에서 가장 위험한 부분인 validation bottleneck을 평가 프로토콜로 다룬다.
- 정답 oracle이 없는 문제를 평가할 때, neighborhood task와 verifier를 어떻게 설계할지에 대한 실무적 힌트를 준다.

이 논문을 단순히 새로운 math judge 논문으로 읽으면 핵심을 놓치기 쉽다. 여기서 중요한 변화는 judge의 모델 크기를 키우는 것이 아니다. 후보 답안을 직접 심사하는 대신, 후보 답안이 다른 검증 가능한 문제를 풀 때 얼마나 useful한 context가 되는지를 측정한다.

즉 이 논문은 검증을 "답안 품질 판정"에서 "답안이 만들어내는 transfer effect 측정"으로 바꾼다. 이 전환이 research automation에서 꽤 중요하다. 실제 연구 문제는 정답지가 없는 경우가 많고, 사람이 모든 candidate proof를 정독하기에는 비용이 크다. 이때 후보 solution이 주변 문제 해결에 일관되게 도움을 준다면, 그 후보에는 최소한 재사용 가능한 method-level information이 들어 있을 가능성이 있다.

# 1. Problem Setting

## 1-1. Problem definition

이 논문이 겨냥하는 문제는 research-level math에서 solution validation이 너무 비싸다는 점이다.

일반적인 benchmark에서는 정답이 있다. 답이 숫자면 exact match를 보면 되고, 선택형이면 label을 보면 된다. 하지만 research-level math에서는 상황이 다르다.

- solution이 길고 subtle하다.
- 최종 answer만 맞아도 derivation이 틀릴 수 있다.
- proof sketch가 완전한 증명인지 판단하려면 domain expert가 필요하다.
- candidate가 겉보기에는 매우 설득력 있어도 핵심 lemma나 reduction이 틀릴 수 있다.
- 아직 frontier model도 풀지 못하는 문제에서는 model judge가 solver보다 충분히 강하다는 보장이 약하다.

따라서 문제는 "어떤 LLM judge가 더 잘 평가하는가"가 아니라, 더 근본적으로는 "정답 oracle 없이 candidate solution을 어떻게 ranking할 것인가"에 가깝다.

논문은 이를 oracle-free validation 문제로 본다. Target question $Q$와 candidate solution $C$가 있을 때, 우리는 $C$가 맞는지 직접 알려주는 oracle을 갖고 있지 않다. 그럼에도 여러 candidate 중 어떤 것을 사람이 먼저 읽어야 하는지, 어떤 것을 더 신뢰할 수 있는지 순위를 정해야 한다.

## 1-2. Why previous approaches are insufficient

기존 방식은 크게 세 계열로 볼 수 있다.

1. Majority voting
   - 여러 sample이 같은 final answer를 내면 그 답을 믿는 방식이다.
   - 짧은 수학 문제나 competition-style problem에는 유용할 수 있다.
   - 하지만 research-level proof에서는 correctness가 discrete answer 하나로 줄어들지 않는다.

2. Reward model 또는 generative reward model
   - 답안 전체를 보고 scalar score를 뽑는다.
   - preference data나 critique generation을 활용할 수 있다.
   - 하지만 long proof의 subtle error를 잡으려면 reward model 자체가 해당 domain을 깊게 이해해야 한다.

3. LLM-as-a-Judge
   - 가장 직접적인 방식이다.
   - 후보 solution을 주고 1에서 10 같은 score를 내게 한다.
   - 문제는 judge가 문체, 포맷, confidence, 장황함에 흔들릴 수 있다는 점이다.

이 방식들의 공통 한계는 solution을 isolation 상태에서 평가한다는 점이다. Candidate가 실제로 어떤 method를 담고 있는지, 그 method가 주변 문제로 transfer되는지는 보지 않는다.

논문은 이 지점을 바꾼다. 수학에서는 직접 증명하지 못한 claim도 그 claim이 많은 correct consequence를 만들어내면 더 강하게 지지받는다. Consequence-Based Utility, 이하 CBU는 이 관점을 LLM evaluation protocol로 구현한다.

# 2. Core Idea

## 2-1. Main contribution

CBU의 핵심은 간단하다.

Target question $Q$에 대한 candidate solution $C_i$가 있을 때, 이 candidate를 solver model의 in-context exemplar로 넣는다. 그런 다음 $Q$와 가까운 related question $Q_star$를 풀게 한다. 만약 $C_i$가 올바른 방법이나 구조를 담고 있다면, solver는 주변 문제를 더 잘 풀 가능성이 높다.

이를 아주 단순하게 쓰면 다음과 같다.

$$
U(C_i) = mean score(M(Q, C_i, Q_star))
$$

여기서 $M$은 solver model이고, $Q_star$는 target question의 neighborhood question이다. $score$는 $Q_star$에 대해 검증 가능한 verifier가 주는 점수다.

이 아이디어의 중요한 점은 direct correctness를 묻지 않는다는 것이다. CBU는 "이 solution이 맞는가"라고 judge에게 묻지 않는다. 대신 "이 solution을 context로 주었을 때 related task를 더 잘 푸는가"를 본다.

논문의 기여는 세 가지로 정리할 수 있다.

1. Consequence-Based Utility 제안
   - Candidate solution을 downstream performance로 평가한다.
   - Research-level target question 자체의 oracle 없이도 candidate ranking signal을 만든다.

2. ExpertMath 구성
   - Expert-written research-level math problem과 neighborhood variant를 포함한다.
   - Candidate solution은 여러 frontier model이 생성하고 human validation을 거친다.

3. Baseline 대비 검증
   - Reward model, generative reward model, LLM judge와 비교한다.
   - CBU가 Acc@1, AUC, HumanWin 등 ranking quality에서 더 강한 결과를 보인다.

## 2-2. Design intuition

CBU의 설계 직관은 "proof는 그 자체보다 쓰임을 통해 드러난다"에 가깝다.

좋은 solution은 단순히 정답 문장을 포함하는 것이 아니다. 좋은 solution은 key lemma, reduction, invariant, construction, proof strategy 같은 재사용 가능한 정보를 담고 있다. 이런 정보가 있으면 비슷한 문제를 풀 때 solver가 더 나은 path를 선택할 수 있다.

반대로 틀린 solution은 그럴듯한 문장을 많이 포함할 수 있다. 하지만 그 안의 method가 실제로 틀렸다면, 주변 문제를 푸는 데 일관되게 도움이 되기 어렵다. 특히 잘못된 reasoning, 과도하게 압축된 argument, 정당화되지 않은 interpretation은 neighborhood question에서 실패로 드러날 가능성이 커진다.

이 차이를 code review와 unit test의 차이로 볼 수 있다.

- LLM judge는 code review처럼 solution을 읽고 plausibility를 평가한다.
- CBU는 unit test처럼 solution이 만들어내는 downstream behavior를 본다.

물론 수학 proof가 code처럼 자동 test 가능하다는 뜻은 아니다. 핵심은 evaluation signal을 surface-level judgment에서 consequence-level behavior로 옮긴다는 점이다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | 정답 oracle 없이 research-level math candidate solution을 ranking |
| Core metric | Consequence-Based Utility |
| Main signal | Candidate solution이 related verifiable question을 푸는 데 주는 도움 |
| Evaluation target | Candidate solution 자체의 direct score가 아니라 downstream transfer utility |
| Baselines | LLM judges, reward models, generative reward models |
| Dataset | ExpertMath |

## 3-2. Module breakdown

### 1) Target question and candidate solutions

먼저 target research-level question $Q$가 있다. 이 문제는 frontier model이 쉽게 풀 수 없고, 사람이 직접 검증하려면 expert time이 많이 드는 문제다.

각 $Q$에 대해 여러 candidate solution이 준비된다. Candidate에는 expert-written solution과 LLM-generated solution이 섞인다. 논문은 candidate solution을 사람이 다시 검증해 correct와 wrong label을 구성하고, evaluator가 이 correct solution을 얼마나 잘 상위에 올리는지 본다.

중요한 점은 CBU가 evaluation 단계에서는 target question의 oracle을 쓰지 않는다는 것이다. Label은 연구 실험을 위한 ground truth로 쓰이고, CBU score 자체는 neighborhood question에서 나온다.

### 2) Neighborhood questions

Neighborhood question은 target question과 같은 핵심 아이디어를 공유하지만, 좀 더 검증 가능한 related problem이다.

좋은 neighborhood question은 다음 조건을 만족해야 한다.

- 원문 target과 method-level relation이 있어야 한다.
- Candidate solution이 담은 핵심 아이디어가 transfer될 여지가 있어야 한다.
- Solver output을 verifier로 평가할 수 있어야 한다.
- 너무 쉽거나 너무 멀리 떨어진 문제가 아니어야 한다.

논문에서는 expert가 target problem마다 variant를 만든다. 원문에서는 각 problem package에 main problem, neighborhood question, reference solution이 포함된다고 설명한다.

### 3) Solver-conditioned evaluation

각 candidate $C_i$는 solver prompt 안에서 context로 들어간다. Solver는 $Q$와 $C_i$를 본 상태에서 neighborhood question $Q_star$를 푼다.

이 구조가 중요한 이유는 candidate를 직접 채점하지 않는다는 점이다. Candidate는 solver의 reasoning에 영향을 주는 exemplar가 된다. Candidate가 좋은 method를 담고 있으면 solver가 related problem을 더 잘 풀고, candidate가 잘못된 method를 담고 있으면 solver가 덜 도움을 받거나 오히려 잘못된 방향으로 갈 수 있다.

### 4) Utility aggregation

CBU score는 neighborhood question과 rollout에 대해 평균화된다. 논문은 안정적인 utility estimate를 위해 여러 rollout을 사용하고, practitioner guide에서 rollout 수와 neighborhood construction에 대한 논의를 제공한다.

이 aggregation은 중요하다. 한 번의 solver output은 noise가 크다. 특히 research-level math에서는 model이 같은 context에서도 다른 path를 시도할 수 있다. 따라서 CBU는 single judgment가 아니라 repeated downstream performance를 본다.

# 4. Training / Data / Recipe

## 4-1. Data

이 논문은 새로운 model을 학습하는 논문이 아니라 evaluation protocol을 제안하는 논문이다. 따라서 핵심 data는 ExpertMath다.

원문 기준으로 ExpertMath는 faculty-authored research-level question에서 출발하고, expert-written variant와 solution을 포함한다. Dataset 설명에서 중요한 숫자는 다음과 같다.

| Item | Value |
| --- | --- |
| Starting expert questions | 70 |
| Total research-level problems with variants | 192 |
| LLM-generated candidate solutions | 630 |
| Candidate pool per original problem | 9 LLM-generated solutions plus expert solution |
| Domains | representation theory, algebraic combinatorics, geometry, homotopy theory 등 |

이 dataset 설계가 중요한 이유는 difficulty다. 논문은 ExpertMath가 AIME 같은 competition benchmark보다 훨씬 어렵고, FrontierMath에 가까운 난이도 영역을 겨냥한다고 설명한다. 즉 CBU가 필요한 영역은 이미 model이 쉽게 푸는 문제가 아니라, solver와 judge 모두 불안정해지는 high-difficulty tail이다.

## 4-2. Training strategy

CBU 자체는 training method가 아니다. 하지만 inference-time evaluation recipe로는 꽤 명확한 절차를 갖는다.

1. Target question을 준비한다.
2. Candidate solutions를 생성한다.
3. Candidate마다 neighborhood question에 대한 solver rollout을 생성한다.
4. 각 rollout을 verifier로 채점한다.
5. 평균 score를 candidate utility로 삼는다.
6. Utility가 높은 candidate를 상위에 ranking한다.

여기서 설계상 가장 중요한 선택은 neighborhood question이다. 너무 쉬운 neighborhood question이면 solver가 candidate 없이도 잘 풀 수 있다. 그러면 utility 차이가 사라진다. 너무 어려우면 verifier signal이 noise가 된다. 너무 멀리 떨어져 있으면 target solution의 method가 transfer되지 않는다.

## 4-3. Engineering notes

실무 관점에서 CBU를 쓰려면 몇 가지 비용을 받아들여야 한다.

첫째, CBU는 rollout-heavy하다. Candidate 수, neighborhood question 수, rollout 수가 곱으로 늘어난다. LLM judge 한 번보다 비쌀 수 있다.

둘째, verifier가 필요하다. Neighborhood question이 verifiable해야 하기 때문에, 자동 채점 가능한 final answer나 checkable property가 있어야 한다. 완전한 proof verification까지는 아니더라도, 적어도 neighborhood output의 correctness를 판별할 장치가 필요하다.

셋째, neighborhood design이 성능을 좌우한다. CBU는 candidate의 transfer utility를 보는 방법이므로, neighborhood가 target과 잘 연결되어 있어야 한다. 이 부분은 자동화가 가장 어려운 지점이다.

넷째, utility score는 correctness proof가 아니다. CBU가 높은 solution은 사람이 먼저 읽을 가치가 높은 candidate일 수 있지만, 그 자체가 최종 수학적 증명은 아니다.

# 5. Evaluation

## 5-1. Main results

논문은 CBU를 reward model, generative reward model, LLM judge와 비교한다. 평가 지표는 Acc@1, Recall@5, AUC, HumanWin, MeanWin이다.

대표적으로 GPT-OSS-120B setting에서 LLM judge는 Acc@1 67.21, AUC 71.42를 보이고, CBU는 Acc@1 76.27, AUC 79.63을 보인다. GPT-OSS-20B에서도 AUC가 69.03에서 79.18로 올라간다.

또한 CBU는 HumanWin에서 큰 차이를 보인다. 이는 human-written solution이 때로 짧고 intuition-driven한데, LLM judge가 장황한 LLM answer를 더 높게 볼 수 있다는 문제와 연결된다. CBU는 style이 아니라 downstream consequence를 보기 때문에 이런 surface bias를 덜 받을 수 있다.

## 5-2. What really matters in the experiments

이 논문의 실험에서 중요한 점은 absolute score보다 difficulty-dependent behavior다.

쉬운 문제에서는 candidate context가 없어도 solver가 잘 풀 수 있다. 이 경우 CBU의 차별성이 줄어든다. 반대로 매우 어려운 문제에서는 direct judge가 solution을 읽고도 틀릴 가능성이 커진다. 이때 CBU는 neighborhood question에서 나오는 behavior signal을 활용하므로 correct와 wrong candidate separation을 더 잘 유지한다.

논문은 이를 solver-evaluator gap 관점에서 해석한다. Solver가 target question을 못 푸는 상황에서도, candidate solution이 주변 문제 해결을 개선하는지 보는 방식은 여전히 유효할 수 있다.

이 메시지는 AI research automation에서 중요하다. 앞으로 model이 더 많은 plausible theorem, proof sketch, algorithm idea를 생성하면, 사람이 볼 candidate를 줄이는 triage가 필요하다. CBU는 이 triage를 score-based judge가 아니라 consequence-based test로 설계하려는 시도다.

# 6. Limitations

1. Neighborhood question 생성 비용이 크다.
   - CBU는 neighborhood가 있어야 동작한다.
   - Expert가 좋은 variant를 만들어야 한다면 scalability에 제약이 생긴다.

2. Verifier 설계가 필요하다.
   - Related question이 검증 가능해야 utility를 계산할 수 있다.
   - 자동 verifier가 약하면 CBU score도 오염된다.

3. Inference cost가 높다.
   - Candidate 수, neighborhood 수, rollout 수가 커질수록 비용이 빠르게 증가한다.
   - LLM judge보다 비싼 evaluator가 될 수 있다.

4. Transfer assumption이 항상 맞지는 않는다.
   - 올바른 solution이어도 neighborhood question으로 잘 transfer되지 않을 수 있다.
   - 반대로 틀린 solution이 우연히 주변 문제에 도움이 될 수도 있다.

5. 최종 검증을 대체하지는 않는다.
   - CBU는 candidate ranking과 triage에 가깝다.
   - 실제 연구 결과로 받아들이려면 expert validation이 여전히 필요하다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문은 LLM evaluation보다 AI research workflow 설계에 더 큰 의미가 있다.

지금까지 많은 evaluation은 answer를 직접 judge하는 방향으로 갔다. 하지만 research-level task에서는 direct judgment가 가장 어려운 부분이다. CBU는 이 문제를 우회한다. 답안을 직접 믿지 말고, 그 답안이 만들어내는 operational consequence를 보자는 것이다.

이 관점은 math에만 제한되지 않는다. Scientific hypothesis, code patch, data analysis plan, agent prompt rule에도 비슷하게 적용할 수 있다. 어떤 artifact가 맞는지 직접 판정하기 어렵다면, 그 artifact를 context로 넣었을 때 더 작은 verifiable task에서 성능이 좋아지는지 볼 수 있다.

## 7-2. Reuse potential

실무적으로 재사용 가능한 부분은 세 가지다.

1. Candidate triage
   - 여러 자동 생성 solution 중 사람이 먼저 읽을 후보를 고른다.

2. Evaluation harness design
   - Target task 자체보다 작고 검증 가능한 neighborhood task를 만든다.

3. Agent memory or prompt rule selection
   - 어떤 memory, rule, strategy가 downstream task에서 실제로 도움을 주는지 평가한다.

특히 agent optimization에서는 CBU와 비슷한 생각이 유용하다. Prompt rule이나 skill document가 그럴듯한지 읽는 대신, 그 rule을 넣었을 때 related tasks에서 success rate가 올라가는지 보면 된다.

## 7-3. Follow-up papers

- LLM-as-a-Judge and reward model reliability 관련 논문
- FrontierMath, HLE, IMProofBench 같은 high-difficulty evaluation benchmark
- In-context example valuation, DemoShapley, data valuation 관련 논문
- AI scientist validation, automated theorem proving, proof verification 관련 논문

# 8. Summary

- CBU는 research-level math candidate solution을 직접 judge하지 않고, related verifiable question에서의 downstream utility로 평가한다.
- 핵심 가정은 올바른 solution에는 주변 문제로 transfer되는 method-level information이 들어 있다는 것이다.
- ExpertMath는 expert-written problem, neighborhood variant, LLM-generated candidate solution을 통해 이 가정을 검증한다.
- 실험에서는 CBU가 LLM judge, reward model, generative reward model보다 ranking quality에서 강한 결과를 보인다.
- 다만 neighborhood question과 verifier 설계 비용이 크므로, 최종 검증 대체보다 candidate triage 도구로 읽는 편이 안전하다.
