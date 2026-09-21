---
layout: single
title: "Random Attention: Rethinking KV Cache Eviction for Efficient Reasoning Review"
categories: Study-concept
tag: [LLM, KVCache, Reasoning]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/pdf/2609.03430)

[Code link](https://github.com/SalesforceAIResearch/Random-Attention)

[Project page](https://arthur-heng.github.io/Random-Attention-page/)

Long reasoning model에서 KV cache를 줄이려면 어떤 token이 중요한지 잘 골라야 한다는 생각이 자연스럽다. 그래서 최근 KV eviction method는 attention score, value norm, recency, retrieval relevance, learned selector를 사용해 cache entry에 우선순위를 준다.

Random Attention은 이 전제를 의심한다.

> Generated reasoning trace에서 정교한 importance score가 정말 필요한가?

논문의 답은 생각보다 공격적이다. Prompt를 보호하고, generated token은 KV head별로 무작위 선택해도 많은 reasoning benchmark에서 강한 selector와 비슷하거나 더 좋다. 더 중요한 것은 random method가 특별히 똑똑해서가 아니라, long reasoning trace가 redundancy와 cross-head replication을 많이 가진다는 점이다.

> 한 줄 요약: Random Attention은 prompt KV를 영구 보존하고 generated KV를 head별 uniform random top-K로 유지하는 selector-free eviction baseline이며, long reasoning에서는 정교한 importance scoring의 이득이 예상보다 작고 selector overhead가 오히려 throughput을 제한할 수 있음을 보여준다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- KV eviction 연구에서 random baseline을 단순 sanity check가 아니라 serious system baseline으로 끌어올린다.
- Prompt protection과 generated-trace eviction을 분리해 기존 method의 failure 원인을 재해석한다.
- Math, science, competition math, code benchmark와 여러 reasoning model을 함께 평가한다.
- Accuracy뿐 아니라 selector overhead, fixed-batch throughput, equal-memory capacity를 분리해서 측정한다.
- Once-only fact probe를 통해 random eviction이 실패하는 경계도 명확히 보여준다.

# 1. Problem Setting

## 1-1. Long reasoning의 KV cache 문제

Autoregressive decoder는 각 layer에서 이전 token의 key and value를 저장한다. Sequence length를 $L$, layer 수를 $N_L$, KV head 수를 $N_H$, head dimension을 $D_H$, element bit width를 $b$라고 하면 cache memory는 대략 다음에 비례한다.

$$
M_{\mathrm{KV}}
\propto
L N_L N_H D_H b
$$

Reasoning model은 answer보다 훨씬 긴 intermediate trace를 생성할 수 있다. Output이 16K or 32K tokens까지 늘어나면 KV cache가 batch capacity와 throughput을 직접 제한한다.

## 1-2. 기존 eviction의 비용

기존 method는 보통 아래 중 하나를 계산한다.

- Recent token priority
- Attention accumulation
- Key or value norm
- Query-key relevance
- Learned importance score
- Layer or head specific heuristic

문제는 importance score도 공짜가 아니라는 점이다.

1. Score tensor를 계산한다.
2. Cache entry를 정렬하거나 top-K를 찾는다.
3. Head별 metadata를 유지한다.
4. Eviction round마다 score update를 반복한다.
5. Selector kernel이 attention kernel 사이의 synchronization point가 된다.

Selection quality가 조금 좋아도 score pass가 비싸면 end-to-end throughput은 나빠질 수 있다.

## 1-3. Prompt와 generated trace는 같은 memory가 아니다

논문은 cache를 두 영역으로 나눈다.

- Prefill region: system prompt, chat history, question, retrieved context
- Decode region: model이 생성한 reasoning trace

Prefill token은 task definition and evidence를 담는다. Generated trace는 모델이 스스로 만든 intermediate computation이다.

많은 eviction method가 두 영역을 동일한 score rule로 처리한다. 그러나 prompt token이 한번 사라지면 모델은 원래 문제를 다시 복원하기 어렵다. 반면 generated trace는 같은 claim을 여러 번 반복하거나 여러 head에 비슷한 정보가 복제될 수 있다.

Random Attention의 핵심 전제는 아래와 같다.

> Prompt는 fragile memory이고, generated reasoning trace는 redundant working memory일 수 있다.

# 2. Core Idea

## 2-1. Prompt는 무조건 보호한다

Prompt length를 $\ell_p$라고 하자. Prompt token에는 infinite priority를 준다.

Generated token에는 uniform random score를 부여한다.

$$
s_i
=
\begin{cases}
+\infty, & i \le \ell_p \\
u_i,\quad u_i \sim \operatorname{Uniform}(0,1), & i > \ell_p
\end{cases}
$$

KV head $h$는 generated region에서 top-K random score를 가진 entry를 유지한다.

$$
\mathcal K_h
=
\operatorname{TopK}
\left(
\{s_{h,i}\}_{i>\ell_p},
K
\right)
$$

Random score는 KV head별로 독립적이다. 따라서 모든 head가 같은 token을 버리지 않는다.

## 2-2. Selector가 없는 null hypothesis

Random Attention은 token importance를 추정하지 않는다.

- Attention map을 누적하지 않는다.
- Query-aware relevance를 계산하지 않는다.
- Learned policy를 실행하지 않는다.
- Value statistic을 저장하지 않는다.

이 method의 연구적 역할은 명확하다.

> 정교한 selector가 random보다 얼마나 더 좋은가?

Selector method가 random을 이기지 못한다면 두 가능성이 있다.

1. 중요 token을 찾는 scoring rule이 충분히 좋지 않다.
2. 해당 task에서 중요 token selection 자체가 병목이 아니다.

Random Attention은 이 둘을 구분하기 위한 null baseline이다.

## 2-3. Head-wise randomness가 만드는 분산 저장

같은 random mask를 모든 head에 적용하면 특정 token은 model 전체에서 완전히 사라진다. 반면 head별 독립 random mask는 token representation을 여러 head에 분산해서 남긴다.

Generated trace의 정보가 여러 head에 복제되어 있다면, 한 head에서 entry가 사라져도 다른 head가 같은 단서를 유지할 수 있다.

이것은 learned redundancy가 random eviction을 견디는 메커니즘 중 하나다.

## 2-4. Reasoning trace가 스스로 복제하는 정보

논문은 long reasoning에서 다음 현상을 강조한다.

- Intermediate result를 다시 서술한다.
- Question condition을 중간에 재확인한다.
- 이전 식을 변형해서 반복한다.
- Candidate answer를 여러 번 검증한다.
- Summary sentence가 앞 reasoning을 압축한다.

따라서 특정 generated token 하나가 사라져도 semantic content가 이후 token에 다시 등장할 수 있다.

Random Attention은 이 redundancy를 활용한다. 반대로 한번만 등장한 external fact는 재생성되지 않기 때문에 random eviction에 취약하다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Random Attention |
| --- | --- |
| Protected region | Entire prefill prompt |
| Eviction target | Generated KV only |
| Score | Uniform random |
| Sampling granularity | Per KV head |
| Cache budget | Head별 fixed K |
| Learned parameter | None |
| Selector pass | None |
| Intended scope | Long generated reasoning |

## 3-2. Eviction lifecycle

Runtime에서는 대략 다음과 같이 동작한다.

1. Prefill KV를 생성하고 protected region으로 표시한다.
2. Decode token이 recent buffer에 들어간다.
3. Cache가 budget을 넘으면 generated region에 random priority를 부여한다.
4. KV head별 top-K entry를 선택한다.
5. Prompt entry와 selected generated entry를 유지한다.
6. 다음 decode step을 계속한다.

실제 implementation은 vLLM memory page and recent-window behavior와 결합된다. 따라서 algorithm의 단순성만 보고 full implementation cost가 0이라고 해석해서는 안 된다. 핵심은 별도 semantic score pass가 없다는 점이다.

## 3-3. 비교 방법

논문은 다음 baseline과 비교한다.

- SnapKV
- R-KV
- VaSE
- TriAttention
- Full attention

모델은 다음 네 개다.

- Qwen3-4B
- Qwen3-14B
- Qwen3-32B
- Phi-4-reasoning

Benchmark는 여섯 underlying task를 포함한다.

- MATH500
- GPQA-Diamond
- AIME25
- AIME26
- HMMT
- LiveCodeBench-v6 medium

Main result table은 AIME25 and AIME26을 하나의 AIME column으로 묶어 다섯 reported column을 사용한다. Maximum context는 32K다. 일반 reasoning benchmark는 약 4x cache compression, LiveCodeBench는 약 3x compression setting을 사용한다.

# 4. Training / Data / Recipe

## 4-1. Training-free deployment

Random Attention은 model training이나 calibration data가 필요하지 않다. Existing checkpoint and inference engine에 eviction policy만 추가한다.

이 특성은 baseline으로 특히 중요하다.

- Dataset leakage가 없다.
- Model-specific training cost가 없다.
- Selector checkpoint가 없다.
- New model family에 바로 적용할 수 있다.
- Random seed만으로 behavior를 재현할 수 있다.

## 4-2. Evaluation protocol

Accuracy comparison은 benchmark별 released sampling setting을 따르고, paired comparison을 사용한다.

논문은 다음 statistical procedure를 보고한다.

- Paired clustered bootstrap
- Exact sign test

이 방식은 단순 average 차이보다 instance-level win and loss를 본다. Random Attention이 특정 easy subset에서만 이기는지, 전반적으로 안정적인지를 확인하기 위한 것이다.

## 4-3. Efficiency protocol을 두 개로 분리한다

논문은 throughput을 한 가지 숫자로만 보고하지 않는다.

### Fixed-load protocol

같은 request load and batch condition에서 selector overhead를 비교한다.

이 setting은 Random Attention과 TriAttention의 순수 runtime 차이를 보기 좋다.

### Equal-memory protocol

같은 GPU memory에서 각 method가 허용하는 largest batch를 사용한다.

이 setting은 KV compression이 batch capacity를 얼마나 늘리는지 보여준다. 다만 batching, preemption, scheduling 효과가 함께 들어가기 때문에 algorithm-only speedup으로 읽으면 안 된다.

# 5. Evaluation

## 5-1. Main accuracy result

Random Attention은 네 model and 여섯 underlying task에서 강한 결과를 보인다.

예를 들어 일부 score는 다음과 같다.

| Model | MATH500 | GPQA-D | AIME | HMMT | LiveCodeBench |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B Random | 0.874 | 0.530 | 0.610 | 0.438 | 0.744 |
| Phi-4-reasoning Random | 0.910 | 0.678 | 0.662 | 0.430 | 0.667 |
| Qwen3-32B Random | 0.891 | 0.683 | 0.664 | 0.509 | 0.806 |

논문의 paired significance comparison에서 Random Attention은 baseline comparison 60 cells 중 31개에서 유의하게 앞서고, 유의하게 뒤지는 경우는 1개다.

더 중요한 결과는 math, science, competition-math task에서 어떤 selector도 Random Attention을 유의하게 이기지 못했다는 점이다.

이는 random이 optimal이라는 증명이 아니다. 현재 benchmark and budget에서 selector quality가 selector overhead and prompt fragility를 상쇄할 만큼 크지 않았다는 뜻이다.

## 5-2. Prompt protection ablation

Prompt protection을 추가하면 SnapKV, recency, random method가 크게 좋아진다. R-KV의 개선 폭은 상대적으로 작다. R-KV score가 원래 prompt token을 어느 정도 보존하기 때문이다.

이 결과는 기존 eviction 비교의 일부가 selector quality보다 prompt survival 차이를 측정했을 가능성을 보여준다.

따라서 future KV eviction paper는 최소한 아래를 분리해 보고해야 한다.

- Prompt protected vs unprotected
- Generated-only eviction
- Same effective prompt budget
- Selector overhead included end-to-end latency

## 5-3. Once-only fact probe

Random Attention의 경계는 planted fact experiment에서 선명하다.

Task는 다음처럼 구성된다.

1. Generated trace 안에 중요한 fact가 한번만 등장한다.
2. 이후 긴 distractor reasoning이 이어진다.
3. 마지막 answer에서 그 fact를 다시 사용해야 한다.
4. 중간에 fact를 restate하지 않는다.

이 setting에서 Random Attention retrieval score는 0이다. 반면 R-KV는 0.836, VaSE는 0.344를 기록한다.

이 결과는 semantic importance가 무의미하다는 주장을 반박한다. 정확한 결론은 더 좁다.

> Redundant reasoning trace에서는 random eviction이 강하지만, persistent once-only memory에는 signal-aware retention이 필요하다.

## 5-4. Throughput

H200, $K=2048$, 약 1K prompt, 32K generation setting에서 Random Attention은 TriAttention보다 다음 fixed-load throughput gain을 보인다.

- Qwen3-4B: +37%
- Phi-4-reasoning: +43%
- Qwen3-14B: +40%
- Qwen3-32B: +32%

Full attention 대비 throughput은 model에 따라 약 1.6x to 2.7x다.

Eviction round cost는 Random Attention이 약 0.30 ms, TriAttention이 약 1.47 to 1.64 ms다. Selector score pass가 decode loop에 누적되는 비용을 직접 보여준다.

Equal-memory largest-batch protocol에서는 일부 setting에서 full attention 대비 8.8x to 10.0x throughput도 보고된다. 이 큰 배수에는 cache saving으로 늘어난 batch capacity and reduced preemption이 포함되므로, fixed-load 결과와 구분해서 읽어야 한다.

# 6. Limitations

1. **Long reasoning에 특화된 결론임**

   RAG, long-document QA, agent memory처럼 external fact를 오래 유지해야 하는 task에는 그대로 일반화할 수 없다.

2. **Prompt protection도 memory를 사용함**

   Generated cache budget만 강조하면 total KV footprint를 과소평가할 수 있다. Long retrieved prompt에서는 protected region 자체가 매우 클 수 있다.

3. **Once-only fact에 취약함**

   Trace가 정보를 restate하지 않으면 random survival probability에 전적으로 의존한다. Persistent state가 필요한 task에는 위험하다.

4. **Random variance**

   Head-wise independent sampling은 평균적으로 강하지만 request-level tail failure가 생길 수 있다. Production에서는 seed and repeated-run stability를 따로 봐야 한다.

5. **Full attention과의 accuracy gap**

   Random Attention이 selector보다 강한 경우가 많아도 full attention을 항상 복원하지는 않는다. Compression budget이 더 작아지면 degradation이 커질 수 있다.

6. **Engine-specific result**

   Throughput은 vLLM page management, batching, preemption, custom eviction implementation에 의존한다. 다른 serving engine에서 같은 배수를 기대할 수 없다.

7. **Semantic selector의 upper bound가 아님**

   비교한 selector가 future optimal method를 대표하지 않는다. Random baseline이 강하다는 사실은 better signal-aware method가 불가능하다는 뜻이 아니다.

# 7. My Take

## 7-1. 이 논문의 가장 큰 기여는 baseline discipline이다

Efficiency paper에서는 복잡한 method가 단순 baseline을 이기는 것이 당연하다고 가정하기 쉽다. Random Attention은 그 가정을 깨고 다음 질문을 강제한다.

- Prompt protection을 동일하게 적용했는가.
- Selector overhead를 포함했는가.
- Generated trace에 실제 unique information이 얼마나 있는가.
- Head-wise redundancy가 결과를 대신 설명하지 않는가.
- Fixed-load와 equal-memory throughput을 분리했는가.

이 체크리스트 자체가 Random Attention의 가장 재사용 가치가 큰 결과다.

## 7-2. Reasoning trace를 working memory로 본다

Generated CoT는 external database라기보다 scratchpad에 가깝다. Scratchpad에는 반복, 재계산, 중간 summary가 많다. 따라서 모든 token을 보존할 필요가 없을 수 있다.

하지만 scratchpad 안에도 두 종류의 정보가 섞인다.

| Information type | Suitable policy |
| --- | --- |
| 반복되는 local reasoning | Random or recency-biased eviction |
| 한번만 생성된 key fact | Semantic pinning |
| Original prompt and evidence | Full protection or retrieval-aware retention |
| Final intermediate answer | Checkpoint or summary retention |

따라서 practical design은 pure random보다 hybrid에 가까울 가능성이 높다.

## 7-3. Hybrid eviction으로 확장한다면

다음과 같은 policy를 생각할 수 있다.

1. Prompt region은 항상 보호한다.
2. Generated trace 대부분은 head-wise random sampling으로 유지한다.
3. Number, code symbol, tool result, citation, state variable은 semantic pinning한다.
4. Periodic summary token은 protected checkpoint로 승격한다.
5. Unique-information detector가 높은 token만 score-based selector에 보낸다.

이 구조는 expensive selector를 모든 token에 적용하지 않고, random baseline이 약한 once-only fact에만 compute를 쓴다.

## 7-4. Follow-up papers

- TriAttention
- R-KV
- SnapKV
- VaSE
- H2O
- StreamingLLM

함께 읽으면 attention importance, recency, heavy hitter, retrieval, random retention이 각각 어떤 memory assumption을 갖는지 비교하기 좋다.

# 8. Summary

- Prompt와 generated reasoning trace는 같은 KV memory로 다루기 어렵다.
- Random Attention은 prompt를 보호하고 generated KV만 head-wise random top-K로 유지한다.
- 여러 reasoning benchmark에서 강한 selector와 비슷하거나 더 좋은 accuracy를 보인다.
- Long CoT의 textual redundancy and cross-head replication이 random eviction을 견디게 한다.
- Once-only fact에는 실패하므로 reasoning cache와 persistent memory의 경계를 구분해야 한다.
