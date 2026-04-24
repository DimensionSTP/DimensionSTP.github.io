---
layout: single
title: "TriAttention: Efficient Long Reasoning with Trigonometric KV Compression Review"
categories: Study-concept
tag: [LLM, KV-Cache, Efficient-Attention, Long-Context, RoPE]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2604.04921)

[Code link](https://github.com/WeianMao/triattention)

TriAttention은 long reasoning 모델에서 점점 더 커지는 KV cache 병목을 다루는 논문이다. 문제의식은 단순히 context length가 길어서 memory가 많이 든다는 정도가 아니다. reasoning model은 답을 생성하는 과정 자체가 길어지고, chain-of-thought 또는 scratchpad가 수천에서 수만 token까지 늘어날 수 있다. 이때 decode step마다 KV cache가 계속 쌓이면서 memory capacity와 memory bandwidth가 동시에 병목이 된다.

기존 KV cache compression은 대체로 최근 query의 attention score를 보고 어떤 key가 중요한지 추정한다. 그런데 RoPE를 쓰는 모델에서는 query 방향이 position에 따라 계속 회전한다. 따라서 post-RoPE query를 기준으로 중요도를 추정하면, 현재 position 근처의 아주 짧은 "observation window"만 representative하게 쓸 수 있다. TriAttention은 이 "observation window" 자체가 잘못된 관측 대상이라고 본다.

> 한 줄 요약: **TriAttention은 post-RoPE attention score를 관측하는 대신, pre-RoPE Q/K vector가 고정된 center 주변에 모인다는 Q/K concentration을 이용해 distance preference를 trigonometric series로 예측하고, 이를 KV cache pruning score로 쓰는 방법이다.**

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- reasoning model serving에서 KV cache는 단순 memory 문제가 아니라 긴 decode를 가능하게 하는 핵심 병목이다.
- 기존 eviction 방법의 약점이 token importance 추정 자체보다 post-RoPE 관측 방식에 있을 수 있다는 관점을 제시한다.
- RoPE geometry, Q/K norm, calibration statistic, pruning schedule을 하나의 inference-time compression recipe로 묶는다.
- AIME, MATH 500, LongBench, RULER, single GPU agent demo까지 연결해 방법의 실용성을 보여주려 한다.

이 논문의 핵심 메시지는 꽤 선명하다. long reasoning KV compression에서 중요한 것은 과거 attention score를 더 잘 평균내는 것이 아니라, 미래 query가 어떤 distance의 key를 선호할지 안정적인 "model-intrinsic statistic"으로 예측하는 것이다. TriAttention은 이 statistic을 pre-RoPE Q/K center에서 찾는다.

# 1. Problem Setting

## 1-1. Problem definition

이 논문이 겨냥하는 문제는 long reasoning generation에서 KV cache를 줄이면서 reasoning accuracy를 최대한 보존하는 것이다.

일반적인 autoregressive decoding에서 각 layer는 이전 token들의 K/V를 cache에 저장한다. sequence 길이가 $T$라면 KV cache memory도 대체로 $T$에 비례해서 증가한다. input context만 긴 long-context task도 문제지만, reasoning model에서는 output 자체가 길다. 즉 prompt가 길지 않아도 생성 중에 KV cache가 계속 커진다.

이 상황에서 필요한 것은 두 가지다.

1. 어떤 cached token을 남길지 판단하는 importance estimation
2. 그 판단을 inference 중에 충분히 싸게 수행하는 pruning mechanism

Full Attention은 모든 token을 남기므로 accuracy upper bound 역할을 하지만, 긴 reasoning에서는 memory와 throughput이 빠르게 악화된다. 반대로 aggressive pruning은 memory를 줄이지만, 중간 추론 상태나 나중에 다시 필요한 retrieval token을 지우면 reasoning chain이 깨진다.

## 1-2. Why previous approaches are insufficient

기존 KV cache compression은 크게 세 방향으로 볼 수 있다.

| Type | Main idea | Limitation |
| --- | --- | --- |
| Heuristic | sink token, recent window 등 고정 규칙 유지 | content-dependent importance를 잘 반영하기 어렵다 |
| Attention-based | 최근 query의 attention score로 important key 추정 | post-RoPE query가 position에 따라 회전해서 관측 window가 짧다 |
| Norm-based | key/value norm으로 salience 추정 | direction 정보와 distance preference를 놓친다 |

TriAttention이 특히 비판하는 것은 attention-based compression이다. post-RoPE query는 position encoding이 적용된 후의 representation이다. RoPE는 position에 따라 2D subspace를 회전시키므로, 같은 semantic query direction도 position이 달라지면 post-RoPE 방향이 바뀐다. 그래서 과거 query들을 많이 모아도 현재 또는 미래 query의 orientation을 대표하지 못할 수 있다.

이 문제가 reasoning에서 더 치명적인 이유는 important token이 항상 최근에 높은 attention을 받는 것이 아니기 때문이다. 어떤 중간 계산 state나 retrieval evidence는 한동안 dormant하게 있다가 훨씬 뒤에서 필요해질 수 있다. 짧은 observation window에서 낮게 보였다고 지우면, 나중에 필요한 순간에는 이미 복구할 수 없다.

# 2. Core Idea

## 2-1. Main contribution

TriAttention의 핵심 기여는 pre-RoPE 공간에서 Q/K vector가 고정된 non-zero center 주변에 강하게 모인다는 관찰을 KV importance estimation으로 연결한 것이다.

논문은 이 현상을 "Q/K concentration"이라고 부른다. pre-RoPE Q/K가 center 주변에 안정적으로 모여 있다면, RoPE 적용 후의 attention logit은 query-key distance에 대한 trigonometric series로 근사될 수 있다. 즉 특정 attention head가 어떤 거리의 key를 선호하는지가 Q/K center와 RoPE frequency에 의해 어느 정도 예측 가능해진다.

이 아이디어는 다음처럼 정리할 수 있다.

1. Offline calibration에서 pre-RoPE Q/K statistic을 수집한다.
2. Q/K center를 이용해 "distance preference curve"를 만든다.
3. Cached key의 position과 pre-RoPE representation을 이용해 trigonometric score를 계산한다.
4. Q/K norm signal을 보조로 결합한다.
5. Q/K concentration 정도에 따라 trigonometric score와 norm score의 비중을 조절한다.
6. 일정 decode window마다 score가 낮은 KV를 evict하고 top scoring KV만 유지한다.

## 2-2. Design intuition

이 논문의 설계 직관은 post-RoPE space와 pre-RoPE space를 분리해서 보는 데 있다.

post-RoPE space는 실제 attention이 계산되는 공간이다. 하지만 position rotation이 이미 섞여 있기 때문에, 여기서 방향 정보를 직접 관측하면 position-dependent noise가 크다. 반대로 pre-RoPE space는 position rotation 전이므로 Q/K direction의 model-intrinsic structure를 보기 좋다.

RoPE attention을 단순화하면 아래와 같이 볼 수 있다.

$$
a(i,j) = \sum_f r_{q,f} r_{k,f} \cos((i - j)\theta_f + \phi_f)
$$

여기서 $i - j$는 query-key distance, $f$는 frequency band, $r_{q,f}$와 $r_{k,f}$는 각 band의 norm 계열 값, $\theta_f$는 RoPE frequency, $\phi_f$는 Q/K의 phase 차이에 해당한다.

만약 pre-RoPE Q/K가 token마다 크게 흔들리지 않고 center로 잘 근사된다면, 각 band의 coefficient와 phase가 거의 고정된다. 그러면 attention logit은 token content 전체를 다시 보지 않아도 distance에 대한 predictable curve로 근사된다.

이게 TriAttention의 중요한 지점이다. key importance를 현재 query attention score로 직접 측정하는 대신, future query가 가질 distance preference를 Q/K center로 예측한다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | long reasoning decoding에서 KV cache memory를 줄이면서 Full Attention에 가까운 reasoning quality 유지 |
| Key observation | pre-RoPE Q/K vector가 fixed non-zero center 주변에 강하게 concentrate됨 |
| Key module | trigonometric series score, norm-based score, concentration-based adaptive weighting, window-based pruning |
| Prior difference | recent post-RoPE attention observation에 의존하지 않고 model-intrinsic pre-RoPE statistic을 사용 |
| Deployment target | decoding 중 KV cache pruning을 통해 memory footprint와 throughput 개선 |

TriAttention pipeline은 크게 calibration phase와 inference phase로 나뉜다.

Calibration phase에서는 모델의 pre-RoPE Q/K distribution을 보고 head별, frequency band별 statistic을 계산한다. 여기서 중요한 것은 center direction, expected query norm, concentration 정도다. 논문은 concentration을 Mean Resultant Length, MRL로 측정한다.

Inference phase에서는 cache에 들어 있는 key들을 scoring한다. scoring은 매 token마다 하지 않고, 일정 window마다 수행한다. 논문 설정에서는 128 generated tokens마다 compression round를 실행하며, cache가 budget을 넘으면 score를 계산하고 top budget만 유지한다.

## 3-2. Module breakdown

### 1) Q/K concentration

TriAttention의 출발점은 pre-RoPE Q/K distribution이 생각보다 안정적이라는 관찰이다.

- Q와 K는 random하게 흩어져 있지 않고 non-zero center 주변에 모인다.
- 이 concentration은 token position과 input context가 달라도 유지되는 경향이 있다.
- 논문은 dominant frequency band를 중심으로 이 현상을 시각화하고, MRL로 정량화한다.
- Qwen3-8B에서 Math, Coding, Chat domain의 MRL이 0.977에서 0.980 범위로 거의 같게 나온다고 보고한다.

이 관찰이 맞다면, future query를 정확히 모르더라도 query center를 proxy로 쓸 수 있다. 즉 다음 query가 어떤 token content를 가질지 몰라도, 적어도 head-level geometric preference는 안정적으로 활용할 수 있다.

### 2) Trigonometric series score

RoPE는 position을 rotation으로 넣는다. 따라서 pre-RoPE Q/K center를 RoPE 공식에 대입하면 attention logit이 relative distance의 함수가 된다. 논문은 이 함수를 trigonometric series로 해석한다.

직관적으로는 다음과 같다.

- 어떤 head는 가까운 token을 선호하는 local pattern을 가질 수 있다.
- 어떤 head는 특정 distant position을 선호할 수 있다.
- 어떤 head는 sink-like 또는 retrieval-like distance preference를 가질 수 있다.
- 이런 preference는 post-RoPE attention map에서 관측되지만, 그 원인은 pre-RoPE center와 RoPE frequency 조합에서 온다.

TriAttention은 이 distance preference를 key score로 쓴다. cached key는 이미 position과 pre-RoPE representation을 가지고 있으므로, future offset set에 대해 expected attention score를 계산하고 평균화한다.

### 3) Norm-based score

Trigonometric series score는 Q/K가 center에 정확히 붙어 있다는 근사를 사용한다. 하지만 실제 vector는 center 주변에서 변동한다. 특히 concentration이 약한 head에서는 center만으로는 부족할 수 있다.

그래서 TriAttention은 norm-based score를 추가한다. norm은 vector magnitude를 통해 token salience를 잡는 보조 signal이다. 다만 단순히 key norm만 쓰지 않고, query side의 expected contribution까지 frequency band별로 반영한다.

내가 보기엔 이 부분이 중요하다. TriAttention은 direction-based geometry만 밀어붙이지 않는다. center approximation이 강한 head에서는 trigonometric score를 더 믿고, 약한 head에서는 norm signal을 더 살린다.

### 4) Concentration-based adaptive weighting

두 score를 어떻게 섞을지는 MRL 기반 concentration이 결정한다.

- concentration이 높으면 Q/K center approximation이 정확하므로 trigonometric score의 비중이 커진다.
- concentration이 낮으면 center 주변 변동이 크므로 norm-based score가 더 중요해진다.

이를 단순화하면 아래처럼 볼 수 있다.

$$
score(k) = S_{tri}(k) + S_{norm}(k)
$$

여기서 $S_{norm}$ 쪽은 concentration에 따라 조절된다. 정확한 수식은 원문 Eq. 8에서 Eq. 11을 다시 확인하는 편이 좋다. HTML 변환에서는 일부 symbol이 빠져 있어, 최종 업로드 전 PDF 기준 재검증이 필요하다.

### 5) Window-based pruning

매 decode step에서 모든 key를 다시 scoring하면 overhead가 커진다. 그래서 TriAttention은 R-KV 설정을 따라 window-based pruning을 쓴다.

- 128 generated tokens마다 compression round를 실행한다.
- cache size가 budget을 넘으면 모든 key를 score한다.
- score가 높은 top budget key를 유지하고 나머지를 evict한다.

이 방식은 매 step eviction보다 덜 세밀하지만, inference overhead를 줄이기 쉽다. 실제 serving에서는 이 부분이 kernel과 memory manager에 얼마나 잘 붙는지가 중요할 것이다.

### 6) GQA handling

Grouped-Query Attention에서는 여러 query head가 하나의 KV head를 공유한다. query head마다 statistic과 score scale이 다를 수 있으므로, key score를 그대로 비교하면 불안정하다.

TriAttention은 normalize-then-aggregate 전략을 쓴다.

1. query head별 score를 z-score normalize한다.
2. 같은 KV head를 공유하는 query head들에서 max aggregation을 적용한다.
3. 어떤 query head라도 중요하다고 보는 key는 살아남을 수 있게 한다.

이 설계는 보수적이다. 한 head에서라도 필요하다고 판단되는 key를 쉽게 버리지 않기 때문이다.

# 4. Training / Data / Recipe

## 4-1. Data

TriAttention은 새로운 model training 논문이라기보다 inference-time compression 논문이다. 따라서 대규모 학습 데이터 recipe보다는 calibration data와 evaluation setup이 중요하다.

Calibration은 pre-RoPE Q/K statistic을 얻기 위해 필요하다. 논문은 이 statistic이 task-specific data에 과하게 overfit되는 것이 아니라 model-intrinsic property에 가깝다고 주장한다.

근거는 크게 두 가지다.

- Math, Coding, Chat domain에서 Qwen3-8B의 MRL 값이 거의 유사하다.
- Calibration data size와 quality에 대한 sensitivity가 작다고 보고한다.

Appendix H에서는 Qwen3-8B AIME24 기준 calibration size를 50k, 200k, 960k tokens로 바꿔도 accuracy가 45.4에서 45.8 범위에 머문다. 또한 low quality HTML, code, chat data를 사용해도 성능이 크게 무너지지 않는다고 보고한다.

## 4-2. Training strategy

모델 자체를 새로 학습하지 않는다. Recipe의 중심은 offline calibration과 inference-time pruning이다.

| Stage | What happens | Why it matters |
| --- | --- | --- |
| Offline calibration | pre-RoPE Q/K center, expected norm, concentration statistic 계산 | future query를 직접 보지 않고도 score를 만들기 위한 prior 확보 |
| Decode-time scoring | cached key에 대해 trigonometric score와 norm score 계산 | key importance를 post-RoPE recent attention에 의존하지 않음 |
| Periodic pruning | 128 token window마다 budget 초과 시 top scoring key 유지 | overhead와 memory saving 사이의 operating point 제공 |
| GQA aggregation | query head별 normalize 후 max aggregation | shared KV head에서 head별 score scale 차이 완화 |

## 4-3. Engineering notes

실무 관점에서는 아래 포인트가 중요해 보인다.

1. Pre-RoPE representation 접근

TriAttention은 pre-RoPE Q/K statistic이 필요하다. 따라서 inference engine이 이 statistic을 수집하거나, calibration phase에서 별도로 hook을 걸 수 있어야 한다.

2. Scoring overhead

Trigonometric score는 attention matrix를 직접 만들지 않는 대신, distance offset과 frequency band 기반 계산이 들어간다. 논문은 window-based pruning으로 overhead를 낮추지만, 전용 kernel 없이도 production serving에서 충분히 이득인지 확인해야 한다.

3. Budget selection

논문에서는 default KV budget을 2048 tokens로 두고, task와 response length에 따라 512 같은 작은 budget도 사용한다. 실제 시스템에서는 target latency, model size, context length, batch size에 따라 budget을 다시 잡아야 한다.

4. Future offset design

TriAttention은 key가 미래의 여러 query position에서 얼마나 중요할지 보려고 future offset set을 사용한다. Appendix G에서는 max distance와 spacing strategy가 성능에 영향을 준다고 보고한다. 특히 geometric spacing이 linear spacing보다 좋게 나온다.

5. Kernel integration

저자들도 limitation에서 dedicated hardware-aware inference kernel의 필요성을 언급한다. 즉 논문 결과를 실제 serving gain으로 재현하려면 algorithm만이 아니라 kernel path, cache layout, eviction scheduler까지 같이 봐야 한다.

# 5. Evaluation

## 5-1. Main results

실험은 mathematical reasoning을 중심으로 구성되어 있다.

- Models: Qwen3-8B, DeepSeek-R1-Distill-Llama-8B, DeepSeek-R1-Distill-Qwen-7B, GPT-OSS-20B
- Baselines: Full Attention, SnapKV, R-KV
- Benchmarks: AIME 2024, AIME 2025, MATH 500
- Extra benchmarks: LongBench, RULER in appendix
- Generation: max length 32,768 tokens, temperature 0.6, top-p 0.95
- AIME: each problem sampled 8 times, average pass rate reported
- Compression: 128 token마다 pruning, default KV budget 2048

대표적으로 Qwen3-8B에서 AIME 결과를 보면 다음과 같다.

| Method | AIME24 | AIME25 |
| --- | ---: | ---: |
| Full Attention | 57.1 | 40.8 |
| SnapKV | 34.6 | 20.0 |
| R-KV | 25.4 | 17.5 |
| TriAttention | 42.1 | 32.9 |

이 표만 보면 TriAttention이 Full Attention을 완전히 복구한 것은 아니다. 하지만 같은 KV budget에서 SnapKV와 R-KV보다 훨씬 덜 무너진다. 특히 AIME25에서 R-KV 17.5 대비 TriAttention 32.9로 차이가 크다.

MATH 500에서는 KV budget 512 기준으로 아래 결과가 보고된다.

| Method | Qwen3-8B | DS-Llama | DS-Qwen | GPT-OSS |
| --- | ---: | ---: | ---: | ---: |
| Full Attention | 69.6 | 82.4 | 87.0 | 91.4 |
| SnapKV | 49.2 | 65.5 | 66.4 | 68.2 |
| R-KV | 46.4 | 76.9 | 71.6 | 77.4 |
| TriAttention | 56.0 | 80.6 | 79.6 | 81.2 |

여기서도 TriAttention은 모든 모델에서 compression baseline 중 가장 높다. 다만 Full Attention과의 gap은 여전히 남는다. 따라서 이 결과는 압도적 대체라기보다, 같은 memory budget에서 훨씬 나은 accuracy-efficiency trade-off를 제공한다고 읽는 편이 맞다.

## 5-2. Efficiency result

논문이 가장 강하게 제시하는 efficiency claim은 comparable accuracy setting에서 나온다.

| Benchmark | Full Attention accuracy | TriAttention accuracy | TriAttention KV budget | Throughput speedup |
| --- | ---: | ---: | ---: | ---: |
| MATH 500 | 69.6 | 68.4 | 1024 | 6.3x |
| AIME24 | 57.1 | 54.6 | 4096 | 1.9x |
| AIME25 | 40.8 | 40.8 | 3072 | 2.5x |

AIME25에서는 Full Attention과 같은 40.8 accuracy를 유지하면서 2.5x throughput 또는 10.7x KV memory reduction을 달성한다고 보고한다. 이 수치가 이 논문의 headline result다.

또 MATH 500에서는 Full Attention 222.8 tokens/s 대비 TriAttention 1405.2 tokens/s가 보고된다. 다만 이 수치는 benchmark, budget, sequence length, batch size, GPU 조건에 민감하므로, production에 그대로 대입하기보다는 operating point 비교로 보는 것이 안전하다.

## 5-3. What really matters in the experiments

이 논문에서 가장 중요한 실험은 최종 score보다 ablation이다. 왜냐하면 TriAttention의 주장 자체가 Q/K concentration과 trigonometric series가 실제로 importance estimation에 도움이 된다는 것이기 때문이다.

### 1) Trigonometric score ablation

Qwen3-8B, KV budget 2048에서 trigonometric series score를 제거하면 성능이 크게 떨어진다.

| Setting | AIME24 | AIME25 |
| --- | ---: | ---: |
| w/o trigonometric score | 18.8 | 21.2 |
| TriAttention | 42.1 | 32.9 |

이 결과는 norm signal만으로는 부족하고, distance preference를 직접 모델링하는 trigonometric component가 핵심이라는 점을 보여준다.

### 2) Concentration-based weighting ablation

Concentration-based weighting을 제거하면 AIME25가 28.7로 떨어지고, full TriAttention은 32.9를 유지한다. AIME24에서는 차이가 작지만, AIME25에서는 weighting이 꽤 의미 있게 작동한다.

### 3) Cross-domain calibration

Coding data로 calibration하고 reasoning benchmark에서 평가해도 Reasoning calibration과 비슷한 수준이 나온다. AIME24에서는 Coding calibration이 44.2, Reasoning calibration이 42.1이고, AIME25에서는 각각 29.2와 32.9다.

이건 Q/K statistic이 task-specific feature라기보다 model-intrinsic geometry에 가깝다는 논문 주장과 연결된다. 다만 AIME25에서는 reasoning calibration이 더 좋으므로, 완전히 domain-free라고 단정하기보다는 calibration domain sensitivity가 제한적이라고 보는 편이 맞다.

### 4) Memory retention benchmark

논문은 recursive simulation 기반 memory retention benchmark도 제안한다. DFS recursion처럼 중간 state를 저장했다가 나중에 backtracking해야 하는 상황을 만들어, KV pruning이 reasoning memory를 얼마나 깨뜨리는지 본다.

Qwen3-8B, KV budget 2048에서 TriAttention은 depth 16까지 Full Attention과 비슷하게 유지되고, depth 18 이후부터 lag가 나타난다고 보고한다. 반면 R-KV는 depth 16 부근에서 급격히 무너진다.

이 실험은 꽤 흥미롭다. long reasoning KV compression은 단순히 정답률만 볼 것이 아니라, 중간 state retention을 별도로 stress test해야 한다는 메시지를 준다.

# 6. Limitations

1. 전용 kernel 필요성이 남아 있다.

논문도 dedicated hardware-aware inference kernel이 추가 latency reduction의 주요 future work라고 말한다. Algorithm이 좋아도 scoring, pruning, cache compaction이 serving engine에 잘 붙지 않으면 실제 throughput gain은 줄어들 수 있다.

2. Calibration step이 필요하다.

Calibration이 robust하다고 보고되지만, 그래도 model별 pre-RoPE statistic을 수집해야 한다. 다양한 architecture, quantization, fine-tuned checkpoint, system prompt distribution에서 statistic이 얼마나 안정적인지는 실제 배포 전 확인이 필요하다.

3. Evaluation은 reasoning math 중심이다.

Appendix에서 LongBench와 RULER를 보지만, main claim은 AIME와 MATH 500 중심이다. coding agent, tool-use agent, multi-document workflow 같은 실서비스 long reasoning에서는 token importance의 성격이 다를 수 있다.

4. Full Attention과의 gap이 완전히 사라지는 것은 아니다.

일부 comparable accuracy setting에서는 Full Attention과 맞추지만, fixed budget table에서는 여전히 gap이 남는다. 따라서 TriAttention은 Full Attention의 완전 대체라기보다, memory budget이 제한된 상황에서 더 나은 trade-off를 제공하는 방법으로 보는 편이 맞다.

5. Semantic core preservation과는 다른 접근이다.

TriAttention은 geometric distance preference와 norm을 통해 key importance를 예측한다. 하지만 어떤 token이 reasoning의 semantic core인지, scratchpad 중 어디가 redundancy인지 직접 이해하는 방식은 아니다. 특정 agent trace에서는 structural memory policy와 결합해야 할 수 있다.

6. Quantization, batch serving, paged KV cache와의 상호작용은 더 봐야 한다.

Appendix J는 RTX 4090 single GPU에서 Qwen3-32B INT4 OpenClaw demo를 보여주지만, 일반적인 production serving은 paged attention, batching, prefix cache, speculative decoding 등과 얽힌다. TriAttention을 실제 stack에 넣을 때는 이 상호작용을 별도로 검증해야 한다.

# 7. My Take

## 7-1. Why this matters for my work

내 관점에서 TriAttention의 가장 큰 가치는 KV compression을 attention score 관측 문제가 아니라 representation geometry 문제로 바꿔 본다는 점이다.

기존 방식은 대부분 중요한 token은 최근 query에서도 어느 정도 attention을 받을 것이라는 가정에 기대었다. 하지만 reasoning에서는 이 가정이 약하다. 중간 계산 state는 나중에야 다시 필요해질 수 있고, retrieval token도 오랫동안 dormant하다가 특정 step에서만 중요해질 수 있다.

TriAttention은 그래서 관측된 attention history보다 pre-RoPE Q/K center를 더 신뢰한다. 이 관점은 꽤 유용하다. 특히 RoPE 기반 LLM에서는 position rotation 때문에 post-RoPE pattern을 그대로 평균내는 것이 생각보다 불안정할 수 있다.

이 논문을 읽으며 가장 인상적으로 본 부분은 다음 문장으로 요약된다.

- post-RoPE attention은 결과이고, pre-RoPE Q/K concentration은 그 결과를 만드는 더 안정적인 원인일 수 있다.

이 프레이밍은 efficient attention, KV eviction, long-context serving을 설계할 때 꽤 좋은 출발점이 된다.

## 7-2. Reuse potential

실무적으로 재사용 가치가 높은 포인트는 아래 4가지다.

1. Calibration-based KV policy

모델별 pre-RoPE statistic을 한 번 수집하고, inference 중에는 이 statistic을 policy prior로 쓰는 방식은 다른 eviction method에도 붙일 수 있다.

2. Distance preference curve

head별로 어떤 distance를 선호하는지 curve로 보는 분석은 debugging tool로도 유용하다. 특정 head가 local, sink-like, retrieval-like pattern을 갖는지 직접 볼 수 있기 때문이다.

3. Norm과 geometry의 결합

norm-only, attention-only, distance-only로 가기보다 head concentration에 따라 signal을 섞는 방식은 안정적인 설계다. 특히 production에서는 하나의 heuristic에 과하게 의존하는 policy가 위험할 수 있다.

4. Recursive memory benchmark

KV pruning을 평가할 때 단순 QA score만 보지 않고, backtracking이 필요한 recursive task를 보는 것은 좋은 평가 아이디어다. agent memory나 long reasoning trace 평가에도 응용할 수 있다.

반대로 바로 가져오기 어려운 부분도 있다.

- pre-RoPE hook과 calibration pipeline 구성
- decoding engine 내부의 pruning scheduler 구현
- cache compaction과 FlashAttention 계열 kernel과의 integration
- model별, task별 budget tuning

그래서 현실적인 도입 순서는 다음이 좋아 보인다.

1. Offline으로 head별 Q/K concentration과 distance preference curve를 먼저 시각화한다.
2. 기존 KV eviction baseline에 TriAttention-style score를 붙여 ablation한다.
3. 작은 budget부터 quality drop curve를 측정한다.
4. kernel integration 없이도 이득이 있는지 확인한 뒤, serving path 최적화를 검토한다.

## 7-3. Follow-up papers

- StreamingLLM: attention sink와 sliding window 기반 streaming inference를 이해하기 좋다.
- H2O: heavy hitter 기반 KV cache eviction의 초기 대표 방법이다.
- SnapKV: observation window 기반 token selection의 대표 baseline으로 TriAttention과 비교하기 좋다.
- R-KV: reasoning model용 KV compression baseline으로, TriAttention의 주요 비교 대상이다.
- LazyEviction: long reasoning에서 observation-based eviction의 최근 흐름을 같이 보기 좋다.
- DuoAttention: retrieval head와 streaming head 관점에서 long-context attention head를 해석하는 데 도움이 된다.

# 8. Summary

- TriAttention은 long reasoning에서 KV cache memory와 throughput 병목을 줄이기 위한 inference-time compression 방법이다.
- 핵심 관찰은 pre-RoPE Q/K vector가 fixed non-zero center 주변에 concentrate된다는 Q/K concentration이다.
- 이 concentration을 RoPE 공식에 넣으면 attention preference가 query-key distance에 대한 trigonometric series로 근사된다.
- TriAttention은 trigonometric score, norm score, concentration-based weighting을 결합해 key importance를 계산하고, 128 token window마다 top scoring KV를 유지한다.
- 내가 보기엔 이 논문은 KV compression을 recent attention observation 문제가 아니라 pre-RoPE representation geometry 문제로 재정의했다는 점에서 의미가 크다.
