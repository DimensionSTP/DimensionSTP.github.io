---
layout: single
title: "Predictable Compression Failures Review"
categories: Study-concept
tag: [LLM, RAG, Reliability, Compression, Calibration]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2509.11208v3)

[arXiv HTML](https://arxiv.org/html/2509.11208)


Predictable Compression Failures는 evidence-grounded binary adjudication에서 hallucination을 어떻게 줄일지 묻는 논문이다. 여기서 binary adjudication은 support/refute, yes/no, multiple-choice correctness, unit-test pass/fail처럼 최종 판단을 $g(Y) \in \{0, 1\}$ predicate로 볼 수 있는 setting을 뜻한다.

이 논문의 핵심은 단순히 모델이 틀릴 수 있다는 이야기가 아니다. 같은 evidence set이라도 chunk 순서를 바꾸면 모델의 answer probability가 달라질 수 있고, 이 order sensitivity를 information budget으로 계량해서 answer/abstain decision으로 연결하겠다는 주장이다.

> 한 줄 요약: 이 논문은 evidence order를 nuisance variable로 보고, permutation mixture와 Bernoulli information budget을 이용해 모델이 답해야 할지 abstain해야 할지 결정하는 ISR gate를 제안한다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- RAG와 document QA에서 retrieval order가 사실상 hidden confounder가 될 수 있음을 정면으로 다룬다.
- Hallucination을 unrestricted generation 문제가 아니라 verifier-relative binary decision 문제로 좁혀서 수학적으로 다룬다.
- QMV, EDFL, B2T, RoH, ISR이라는 개념을 통해 order sensitivity와 abstention을 하나의 interface로 연결한다.
- 3,059개 grounded item과 528개 held-out audit으로 실제 operating point를 검증하려 한다.
- 실무적으로는 answer/abstain gate, risk-coverage curve, logprob 기반 grounding verifier 설계에 바로 연결된다.

이 논문은 "hallucination을 완전히 없애는 방법"이라기보다, evidence가 충분하지 않은데 모델이 자신 있게 답하는 상황을 어떻게 deployment interface에서 막을지에 대한 논문에 가깝다.

# 1. Problem Setting

## 1-1. Problem definition

이 논문이 겨냥하는 문제는 evidence-grounded binary adjudication에서 evidence order가 모델 판단을 흔드는 현상이다.

RAG나 document QA에서는 보통 retrieval system이 여러 evidence chunk를 가져오고, LLM은 그 chunk들을 보고 답한다. 이상적으로는 같은 evidence multiset이라면 chunk 순서가 바뀌어도 support/refute, yes/no, pass/fail 판단은 바뀌지 않아야 한다. 즉 evidence order는 nuisance variable이어야 한다.

하지만 transformer는 positional encoding을 사용하고, long-context 입력에서는 앞, 중간, 뒤 위치에 따라 evidence 활용도가 달라질 수 있다. 따라서 같은 chunk라도 어느 위치에 놓였는지에 따라 모델이 correct answer에 부여하는 probability가 달라진다.

논문은 각 evidence ordering $\pi$에 대해 모델이 Bernoulli predicate $g = 1$에 부여하는 probability를 다음처럼 둔다.

$$
q_{\pi}(x) = S_{\pi}(g = 1)
$$

여러 permutation을 sample하면 평균 probability와 conservative lower value를 얻을 수 있다.

$$
\bar{q}(x) = \frac{1}{m} \sum_{k = 1}^{m} q_{\pi_k}(x)
$$

$$
q_{lo}(x) = \min_k q_{\pi_k}(x)
$$

여기서 중요한 질문은 단순히 $\bar{q}$가 높은가가 아니다. 특정 ordering에서는 답할 수 있어 보이지만 다른 ordering에서는 불안정하다면, 그 system은 실제 deployment에서 신뢰하기 어렵다.

## 1-2. Why previous approaches are insufficient

기존 evidence-grounded QA evaluation은 보통 canonical evidence order 하나를 고정하고 accuracy를 잰다. 이 방식은 간단하지만 세 가지 문제가 있다.

첫째, order sensitivity를 놓친다. Retrieval score 순서, timestamp 순서, document order, reranker order는 모두 그럴듯한 순서지만, evidence semantics 자체와는 별개일 수 있다. 만약 모델이 이 순서에 과하게 민감하면, 같은 evidence로도 다른 결론이 나온다.

둘째, accuracy와 calibration만으로는 answer/abstain interface를 만들기 어렵다. 모델이 틀릴 가능성이 높을 때 그냥 낮은 confidence라고 표시하는 것과, 실제로 답하지 않고 evidence를 더 요구하는 것은 다르다.

셋째, self-consistency나 prompt ensemble은 같은 문제를 풀지만 estimand가 다르다. Self-consistency는 여러 reasoning path를 평균내는 방식이고, prompt ensemble은 prompt variation을 평균내는 방식이다. 이 논문은 같은 evidence multiset의 order nuisance family를 marginalize한다는 점이 다르다.

넷째, unrestricted generation에서는 correctness predicate 자체가 불명확하다. 논문은 이 문제를 피하기 위해 Bernoulli adjudication으로 scope를 좁힌다. 즉 guarantee는 항상 verifier, gold label, deterministic tool, calibrated predicate에 상대적이다.

정리하면, 이 논문이 보는 병목은 "모델이 knowledge를 모른다"가 아니라 "evidence가 주어져도 그 evidence를 어떤 순서로 보느냐에 따라 decision reliability가 달라진다"는 점이다.

# 2. Core Idea

## 2-1. Main contribution

이 논문의 핵심 기여는 세 가지다.

1. QMV, Quantified Martingale Violation
   - Evidence permutation이 만드는 dispersion을 positional sensitivity 관점에서 bound한다.
   - Adjacent-rank sensitivity가 harmonic하게 줄어드는 regime에서는 dispersion이 $O(\log n)$처럼 커질 수 있다고 본다.

2. EDFL, Expectation-level Decompression Law
   - KL convexity와 data processing을 Bernoulli predicate로 specialization한다.
   - 특정 target reliability에 도달하려면 얼마만큼의 information budget이 필요한지 계산한다.

3. ISR gate
   - B2T, RoH, ISR을 계산해 answer/abstain decision을 만든다.
   - Threshold는 post-hoc tuning이 아니라 analytic boundary인 $ISR = 1$로 고정한다.

가장 중요한 framing은 expectation-realization gap이다. Next-token training은 여러 ordering에 대한 expected conditional description length를 줄일 수 있다. 하지만 실제 deployment에서는 하나의 fixed ordering이 들어온다. 평균적으로는 좋아 보여도 특정 ordering에서는 position-sensitive failure가 생길 수 있다.

이 차이를 논문은 compression failure로 해석한다. 모델이 evidence distribution을 평균적으로 잘 압축하더라도, 특정 realization의 evidence order에서는 충분한 information budget을 확보하지 못해 잘못 답할 수 있다는 관점이다.

## 2-2. Design intuition

설계 직관은 다음처럼 볼 수 있다.

첫째, evidence order는 nuisance variable이다. 같은 evidence set에서 order만 바꿨는데 answer probability가 크게 흔들리면, 그 decision은 evidence semantics가 아니라 position artifact에 의존하고 있을 수 있다.

둘째, permutation mixture는 order nuisance를 marginalize한다. 여러 ordering에서 probability를 계산하고 평균을 내면, single ordering이 가진 positional artifact를 줄일 수 있다.

셋째, 평균 confidence만으로는 부족하다. 평균 probability가 높아도 일부 ordering에서 낮다면 deployment 관점에서는 보수적으로 봐야 한다. 그래서 논문은 $q_{lo}$를 사용해 conservative lower value를 넣는다.

넷째, abstention은 failure가 아니라 interface다. 답을 내지 않는 것은 모델 능력 부족의 표시가 아니라, information budget이 target reliability를 만족하지 못했다는 auditable decision이 될 수 있다.

이 논문의 가장 좋은 점은 hallucination을 "모델 내부의 신비한 오류"로 두지 않고, evidence order, label probability, information budget, reject rule로 분해했다는 데 있다. 이 분해 덕분에 RAG system에서 어떤 부분을 instrument해야 하는지 비교적 명확해진다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Target setting | Evidence-grounded binary adjudication |
| Main risk | Same evidence, different order, different answer probability |
| Nuisance variable | Evidence ordering $\pi$ |
| Main theory | QMV for order dispersion, EDFL for information budget |
| Main planner | B2T, RoH, ISR |
| Decision rule | Answer if $ISR >= 1$, otherwise abstain or acquire more evidence |
| Main data | Factuality Slice, 3,059 grounded items |
| Audit setup | 528 held-out items, Gemma-2-9B fine-tuned audit model |

## 3-2. Module breakdown

### 1) Permutation mixture over evidence chunks

입력 evidence chunk set이 있을 때 논문은 여러 ordering $\pi_1, ..., \pi_m$을 만든다. 각 ordering에 대해 모델이 correct predicate에 부여하는 probability $q_{\pi_k}$를 계산한다.

실험에서는 label token probability를 사용한다. 예를 들어 binary label이 "1"과 "0"이라면 두 label token probability를 normalize해 $q_{\pi}(x)$를 얻는다. 이렇게 하면 long-form answer를 여러 번 생성하지 않아도, answer predicate에 대한 probability-space signal을 얻을 수 있다.

Permutation mixture의 핵심은 아래 평균이다.

$$
\bar{q}(x) = \frac{1}{m} \sum_{k = 1}^{m} q_{\pi_k}(x)
$$

이 값은 단일 order가 아니라 permutation family에 대한 expected probability다. 하지만 deployment에서는 보수성이 중요하므로 논문은 $q_{lo}$도 함께 본다.

### 2) QMV: order-induced dispersion을 설명한다

QMV는 evidence rank가 바뀔 때 probability residual이 얼마나 흔들릴 수 있는지 bound하는 부분이다. 논문의 메시지를 직관적으로 쓰면 다음과 같다.

- Transformer는 position에 민감하다.
- Evidence chunk 순서가 바뀌면 각 chunk의 positional contribution도 바뀐다.
- Adjacent-rank sensitivity가 느리게 감소하면 context depth가 늘수록 dispersion이 log-scale로 누적될 수 있다.

실험에서는 mean absolute residual을 사용한다.

$$
|R_{\pi}| = E_{\pi} | q_{\pi}(x) - \bar{q}(x) |
$$

그리고 다음 empirical law를 fitting한다.

$$
|R_{\pi}| \approx a + b \log n
$$

여기서 $n$은 evidence chunk 수다. 이 결과가 중요한 이유는 order sensitivity가 단순한 noise가 아니라 context depth와 함께 예측 가능한 형태로 증가할 수 있다는 점을 보여주기 때문이다.

### 3) EDFL: Bernoulli information budget으로 바꾼다

EDFL은 target reliability를 만족하기 위해 필요한 information budget을 계산하는 부분이다. 논문은 일반적인 KL convexity와 data-processing 관점을 Bernoulli predicate로 내린다.

Binary adjudication에서는 downstream decision이 $g(Y) \in \{0, 1\}$이므로 Bernoulli distribution으로 coarse-graining할 수 있다. 이때 target success probability를 $p^*$라고 두면, 필요한 bits-to-trust는 대략 다음 형태로 이해할 수 있다.

$$
B2T(p^*, q_{lo}) = KL(Ber(p^*) || Ber(q_{lo}))
$$

Measured available budget을 $\bar{\Delta}$라고 두면 ISR은 다음 비율이다.

$$
ISR = \frac{\bar{\Delta}}{B2T}
$$

Decision rule은 간단하다.

$$
\text{Answer if } ISR >= 1
$$

$$
\text{Abstain otherwise}
$$

이 식에서 중요한 점은 threshold가 0.7, 0.8 같은 tuned confidence가 아니라는 것이다. 논문은 $ISR = 1$ boundary를 analytic boundary로 고정하고, held-out audit에서 이 boundary가 어떻게 작동하는지 확인한다.

### 4) ISR gating procedure

Algorithm 1은 deployment interface에 가깝다.

1. Evidence chunk의 permutation을 $m$개 만든다.
2. 각 permutation에 대해 model을 query한다.
3. Correct predicate에 대한 probability $q_k$를 계산한다.
4. Clipped negative log probability로 budget term을 계산한다.
5. $\bar{q}$, $q_{lo}$, $\bar{\Delta}$, B2T, ISR을 계산한다.
6. $ISR >= 1$이면 answer한다.
7. 아니면 abstain하거나 evidence를 더 가져온 뒤 다시 평가한다.

이 절차는 RAG system에서 꽤 직관적으로 들어갈 수 있다. Retrieval 결과가 있을 때, answer generation 전에 label probability 기반 gate를 먼저 돌리고, gate가 부족하면 더 많은 retrieval, reranking, user clarification, human escalation으로 넘길 수 있다.

### 5) Predicate and verifier interface

논문이 다루는 guarantee는 항상 predicate-relative다. 즉 $g(Y)$를 누가 정의하고 검증하는지가 중요하다.

| Deployment case | Predicate example | Practical requirement |
| --- | --- | --- |
| Binary adjudication | support/refute, approve/deny | deterministic rule or gold label |
| Multiple choice | correct option id | normalized option probability |
| Tool-use | unit test pass, query success | executable verifier |
| Rubric decision | pass/fail rubric | calibrated verifier or audited judge |
| Long-form generation | factuality of a draft | post-hoc verifier and release gate |

이 관점은 강력하지만 동시에 한계를 만든다. Verifier가 부정확하면 ISR gate의 guarantee도 verifier에 상대적으로만 의미가 있다. LLM-as-judge를 쓰려면 judge calibration을 별도로 audit해야 한다.

# 4. Training / Data / Recipe

## 4-1. Data

논문은 Factuality Slice라는 evidence-grounded QA meta-benchmark를 사용한다.

| Source | Cap | Task type | Why included |
| --- | ---: | --- | --- |
| FEVER | 2,000 | Fact verification | Clean support/refute evidence |
| HotpotQA | 2,000 | Multi-hop QA | Multi-hop evidence order sensitivity |
| NQ-Open | 1,000 | Open-domain QA | Retrieved Wikipedia grounding |
| PopQA | 500 | Long-tail QA | Rare entity stress test |
| Controls | 300+ | NEI + recency trap | Insufficient or outdated evidence |

최종 suite는 3,059개 binary adjudication item으로 구성된다. Evidence sentence는 48 token cap으로 chunking되고, source document, sentence id, retrieval score 같은 metadata를 갖는다. Hard negative는 BM25 retrieval로 mining하며, context는 최대 60 evidence chunk까지 cap한다.

Dataset construction에서 중요한 point는 support span과 hard negative를 함께 넣는다는 점이다. 이 구조 덕분에 모델이 answer-bearing evidence를 실제로 활용하는지, 아니면 distractor나 order artifact에 끌리는지 측정할 수 있다.

## 4-2. Training strategy

이 논문은 새로운 LLM을 크게 pretrain하는 논문이 아니다. 핵심은 evaluation protocol과 answer/abstain gate다.

실험은 크게 세 개다.

| Experiment | Model / setup | Purpose |
| --- | --- | --- |
| Exp. 1 | Qwen2-7B-Instruct, Llama-3.1-8B-Instruct, 4-bit NF4 | Permutation dispersion and mixture gains |
| Exp. 2 | Support dose randomized with fixed prompt length | Causal dose-response of support evidence |
| Exp. 3 | Gemma-2-9B fine-tuned audit model, 528 held-out items | Pre-specified ISR boundary audit |

Exp. 1에서는 3,059개 item 전체를 사용하고, evidence chunk 수 $n$은 3부터 60까지 변한다. 각 item마다 $m = 16$개의 banded permutation을 sample한다. Banded permutation은 6개 band 안에서 shuffle하는 방식이다.

Exp. 2에서는 prompt length를 4 chunks로 고정한 상태에서 support dose $d \in \{0, 1, 2, 3\}$를 바꾼다. 이렇게 하면 evidence가 많아져서 좋아진 것인지, prompt가 길어져서 달라진 것인지의 confound를 줄일 수 있다.

Exp. 3에서는 four non-control benchmarks, 즉 FEVER, HotpotQA, NQ-Open, PopQA로 fine-tuned된 Gemma-2-9B model을 528개 held-out audit item에서 평가한다. Controls까지 포함한 full five-benchmark suite에서 held-out item을 뽑고, permutation seed 0부터 5까지 총 $m = 6$개 permutation을 고정한다.

## 4-3. Engineering notes

실무 관점에서는 다음 포인트가 중요하다.

1. Token logprob access가 필요하다
   - ISR gate는 label probability를 계산해야 한다.
   - API나 serving stack이 logprobs를 제공하지 않으면 구현이 어렵다.

2. Multiple forward pass cost가 있다
   - $m$개 permutation을 돌리면 최소 $m$번의 model forward가 필요하다.
   - 논문은 latency-sensitive setting에서는 작은 $m$ cascade를 제안한다.

3. Evidence order가 정말 nuisance인지 확인해야 한다
   - Legal timeline, causal chain, code execution trace처럼 order 자체가 의미를 갖는 task에는 그냥 permute하면 안 된다.
   - 이 경우에는 permutation family를 task semantics에 맞게 다시 정의해야 한다.

4. Verifier calibration이 system boundary다
   - Unit test나 deterministic database lookup이면 비교적 명확하다.
   - LLM judge라면 judge 자체의 calibration, bias, failure mode를 따로 audit해야 한다.

5. Retrieval pipeline과 잘 맞는다
   - RAG에서는 retrieval result를 rerank한 뒤 바로 generation하는 경우가 많다.
   - 이 논문 관점에서는 generation 전에 permutation sensitivity와 ISR을 계산해 release gate를 둘 수 있다.


# 5. Evaluation

## 5-1. Main results

### Experiment 1: permutation dispersion and mixture gains

Exp. 1은 3,059개 item, $m = 16$ banded permutations, $n \in [3, 60]$ chunks 조건에서 수행된다.

| Metric | Qwen2-7B | Llama-3.1-8B |
| --- | ---: | ---: |
| Dispersion slope $b$ vs. $\log n$ | 0.377 | 0.147 |
| 95% CI for $b$ | [0.319, 0.435] | [0.109, 0.184] |
| $R^2$ | 0.742 | 0.515 |
| Jensen gap, nats/token | 0.1041 | 0.00982 |
| Mixture optimality gap | less than 1e-4 | at most 5.3e-5 |

해석은 명확하다. Evidence order를 바꾸면 label probability residual이 chunk depth와 함께 증가하고, uniform permutation mixture는 single permutation보다 cross entropy를 줄인다. Qwen2-7B 쪽이 Llama-3.1-8B보다 더 큰 positional sensitivity를 보인다.

흥미로운 점은 uniform mixture가 globally optimized mixture와 거의 같은 수준이라는 것이다. 즉 복잡한 weight learning 없이 uniform permutation average만으로도 order nuisance를 꽤 잘 marginalize할 수 있다는 결과다.

### Experiment 2: support dose response

Exp. 2는 support evidence가 실제로 hallucination risk를 줄이는지를 보기 위한 실험이다. Prompt length는 $L = 4$ chunks로 고정하고, support chunk 개수 $d$만 0부터 3까지 바꾼다.

논문이 보고한 변화는 다음과 같다.

| Change from dose 0 to 3 | Reported value |
| --- | ---: |
| Answer rate | +37.5 pp |
| Accuracy on attempts | +45.6 pp |
| Hallucination rate | -17.6 pp |
| Information-budget estimate change | -0.375 nats per additional support chunk |
| OLS slope | 0.127 fewer hallucinations per additional nat of decrease |

이 실험은 매우 중요하다. 단순히 evidence가 많으면 좋아진다는 말을 하는 것이 아니라, prompt length를 고정하고 support dose만 randomize해서 information budget estimate와 hallucination의 관계를 본다.

다만 변수 방향성은 원문 figure를 보면서 다시 확인하는 것이 좋다. 논문 표현은 information-budget estimate의 decrease와 hallucination decrease를 연결한다. 블로그 관점에서는 "support evidence가 measured budget을 바꾸고, 그 변화가 hallucination 감소와 연결된다"는 수준으로 읽는 것이 안전하다.

### Experiment 3: pre-specified held-out audit

Exp. 3은 가장 deployment에 가까운 실험이다. Fine-tuned Gemma-2-9B model을 528개 held-out audit item에서 평가하고, $ISR = 1$ gate를 threshold tuning 없이 적용한다.

| Metric | Result |
| --- | ---: |
| Boundary alignment | 96.2% [94.3, 97.5] |
| Hallucination rate | 0.0-0.7% |
| Abstention | 24.1% [20.6, 27.9] |
| Accuracy on attempts | 80.5% [76.8, 83.8] |
| Default permutations | $m = 6$ |

또한 permutation count sensitivity도 제시된다.

| Number of permutations | Boundary alignment |
| ---: | ---: |
| 3 | 94.7% |
| 6 | 96.2% |
| 12 | 97.1% |

이 결과의 핵심은 low hallucination이 low answering으로 얻어진 trivial result인지 보는 것이다. 논문은 abstention 24.1% 수준에서 hallucination 0.0-0.7%와 accuracy on attempts 80.5%를 보고한다. 즉 answer coverage를 일부 포기하고, high-risk case를 gate로 걸러낸다.

## 5-2. What really matters in the experiments

### 1) This is a risk-coverage paper, not a pure accuracy paper

이 논문은 모델이 모든 item에서 더 높은 accuracy를 낸다는 주장보다, 어떤 item에 답하지 말아야 하는지를 계산한다는 데 초점이 있다. 따라서 main metric은 accuracy 하나가 아니라 coverage, hallucination rate, accuracy on attempts, boundary alignment를 함께 봐야 한다.

### 2) Order sensitivity is measurable and exploitable

Evidence order가 바뀔 때 probability가 흔들리는 현상은 단순 nuisance지만, 그 자체가 diagnostic signal이 된다. Dispersion이 크면 model이 evidence를 안정적으로 통합하지 못하고 있을 가능성이 있다.

### 3) Uniform permutation mixture is surprisingly strong

Uniform mixture가 optimized mixture와 거의 같은 cross entropy를 보이는 결과는 practical하게 중요하다. 복잡한 mixture learning 없이도 여러 ordering을 평균내는 것만으로 order artifact를 줄일 수 있기 때문이다.

### 4) ISR gate is conservative by design

$q_{lo}$를 쓰는 것은 성능을 공격적으로 끌어올리려는 선택이 아니라, deployment risk를 줄이려는 선택이다. 평균이 높아도 worst ordering에서 낮으면 abstain할 수 있다.

### 5) The verifier is part of the theorem boundary

이 논문을 RAG에 바로 붙일 때 가장 조심해야 할 부분이다. Predicate가 정확하고 verifier가 calibrated되어야 한다. 그렇지 않으면 ISR이 수학적으로 깔끔해도 system-level guarantee는 약해진다.

# 6. Limitations

1. Scope is narrow by design
   - 논문은 evidence-grounded Bernoulli adjudication을 다룬다.
   - Open-ended long-form generation 자체에 대한 universal hallucination guarantee는 아니다.

2. Verifier dependence가 크다
   - $g(Y)$를 정의하고 검증할 수 있어야 한다.
   - LLM judge를 verifier로 쓰면 judge calibration을 별도로 audit해야 한다.

3. Multiple permutation cost가 있다
   - $m = 6$이면 label-probability call이 6번 필요하다.
   - Latency-sensitive product에서는 cascade나 selective invocation이 필요하다.

4. Held-out audit은 하나의 operating point다
   - 논문은 fine-tuned Gemma-2-9B에서 528-item audit을 제시한다.
   - 더 큰 model, 다른 domain, untuned model, production RAG distribution에서 그대로 calibration된다고 단정하면 안 된다.

5. Permutation assumption을 잘못 쓰면 위험하다
   - Evidence order가 semantics를 갖는 task에서는 arbitrary permutation이 오히려 문제를 왜곡한다.
   - 예를 들어 timeline reasoning, code trace, procedural instruction에서는 order를 nuisance로 볼 수 없다.

6. Dataset and code release 상태 확인이 필요하다
   - Paper appendix는 public release should include dataset builder, seeds, splits, scripts라고 말한다.
   - 실제 공개 repo와 논문 실험 package가 어느 정도 대응되는지는 publish 전에 다시 확인해야 한다.

7. Author metadata inconsistency가 있다
   - arXiv abstract metadata에는 Maggie Chlon이 포함된 4-author list가 보인다.
   - arXiv HTML/PDF header에는 3-author 형태로 보이는 부분이 있어, 최종 게시 전 author list는 arXiv metadata와 PDF를 다시 대조하는 것이 좋다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문은 RAG, Document AI, verifier-backed agent system을 만드는 입장에서 꽤 실용적인 질문을 던진다.

Retrieval result는 보통 score순으로 붙는다. 하지만 score순으로 붙인다는 것은 모델에게 특정 order prior를 주는 것과 같다. 만약 모델이 첫 chunk나 마지막 chunk에 과하게 의존하면, retrieval quality가 같아도 ordering policy가 answer reliability를 바꿀 수 있다.

이 논문을 읽고 가장 먼저 해볼 수 있는 것은 simple diagnostic이다.

1. 같은 retrieved evidence set을 여러 order로 shuffle한다.
2. 모델의 answer probability 또는 verifier probability를 기록한다.
3. Prediction dispersion이 큰 query를 high-risk query로 표시한다.
4. High-risk query에는 answer 대신 additional retrieval, reranking, abstention, human review를 붙인다.

이 정도만 해도 RAG system evaluation이 꽤 달라진다. 단일 ordering accuracy만 보는 것보다, order robustness를 함께 보는 것이 product reliability에 더 가깝다.

## 7-2. Reuse potential

재사용 가능성이 높은 부분은 다음과 같다.

### 1) RAG evaluation에 permutation robustness 추가

기존 eval은 query, retrieved context, answer를 고정해서 본다. 여기에 context permutation axis를 추가하면 model이 retrieval order에 얼마나 취약한지 볼 수 있다.

### 2) Answer/abstain gate로 사용

High-stakes QA에서는 wrong answer보다 abstention이 나을 때가 많다. ISR gate는 confidence threshold보다 해석 가능성이 좋다. Target reliability를 정하고, measured budget이 부족하면 answer를 막는다.

### 3) Verifier-backed generation release gate

Code generation, SQL generation, document extraction, compliance check 같은 task는 pass/fail verifier가 존재하는 경우가 많다. 이 setting에서는 Bernoulli predicate를 만들 수 있으므로 EDFL/ISR interface가 잘 맞는다.

### 4) Latency-aware cascade

항상 $m = 12$ permutation을 돌리는 것은 비싸다. 논문 결과처럼 $m = 3$에서도 상당한 boundary alignment가 나오면, low-risk query는 작은 $m$으로 끝내고 boundary 근처에서만 더 많은 permutation을 돌리는 cascade를 만들 수 있다.

### 5) Retrieval ordering policy evaluation

Reranker를 바꿨을 때 final accuracy만 볼 것이 아니라, order dispersion도 같이 봐야 한다. 좋은 reranker는 높은 relevance뿐 아니라 low dispersion ordering을 만들 수 있어야 한다.

## 7-3. Follow-up papers

- Language Modeling Is Compression
  - Next-token prediction과 compression cost의 연결을 더 넓게 이해하기 좋은 논문이다.

- Lost in the Middle: How Language Models Use Long Contexts
  - Position sensitivity와 evidence placement 문제를 이해하는 데 직접 연결된다.

- Detecting Hallucinations in Large Language Models Using Semantic Entropy
  - Uncertainty와 hallucination detection을 generation diversity 관점에서 보는 논문이다.

- Calibrated Language Models Must Hallucinate
  - Calibration과 hallucination의 이론적 한계를 같이 보면 이 논문의 scope를 더 잘 이해할 수 있다.

- LLMs are Bayesian, in Expectation, not in Realization
  - Expectation-realization gap이라는 관점을 더 넓게 보는 follow-up 축으로 적합하다.

# 8. Summary

- Evidence-grounded QA에서 같은 evidence라도 chunk order가 바뀌면 answer probability가 흔들릴 수 있다.
- 이 논문은 order를 nuisance variable로 보고 permutation mixture로 order sensitivity를 측정한다.
- QMV는 dispersion을, EDFL은 Bernoulli information budget을, ISR은 answer/abstain decision을 담당한다.
- 3,059개 grounded item에서는 log-scale dispersion과 positive Jensen gap이 관찰되고, 528-item held-out audit에서는 $ISR = 1$ gate가 low hallucination/high abstention operating point를 보인다.
- 실무적으로는 RAG, Document AI, verifier-backed agent에서 confidence threshold보다 auditable abstention gate를 설계하는 데 참고할 만하다.
