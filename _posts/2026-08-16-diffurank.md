---
layout: single
title: "DiffuRank: Effective Document Reranking with Diffusion Language Models Review"
categories: Study-concept
tag: [InformationRetrieval, DiffusionLLM, Reranking]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2602.12528)

[Code link](https://github.com/liuqi6777/DiffusionRank)

DiffuRank은 "document reranking에 diffusion language model을 사용했다"는 한 문장만으로 읽으면 핵심을 놓치기 쉬운 논문이다. 이 논문이 실제로 묻는 질문은 diffusion model이 autoregressive LLM보다 더 좋은가가 아니다. 더 정확히는, **ranking이라는 구조화된 출력 문제를 left-to-right text generation으로 풀어야만 하는가**다.

최근 LLM reranker는 query와 여러 candidate document를 prompt에 넣고, relevant order를 document identifier sequence로 생성하는 listwise 방식까지 확장되었다. 이런 방식은 document 사이의 관계를 한 번에 읽을 수 있다는 장점이 있지만, output은 여전히 autoregressive하게 한 token씩 만들어진다. 첫 identifier가 잘못되면 뒤 순서가 영향을 받고, duplicate나 missing identifier를 막기 위한 format control도 필요하다. Candidate 수가 늘어날수록 generation latency도 커진다.

DiffuRank은 이 문제를 masked diffusion language model의 bidirectional context와 parallel prediction으로 다시 설계한다. 단순히 하나의 diffusion reranker를 제안하는 대신, pointwise score, parallel listwise logits, permutation generation이라는 세 단계의 design space를 비교한다. 특히 permutation output을 iterative constrained sampling 또는 one-shot assignment로 바꾼 부분이 이 논문의 가장 중요한 설계다.

> 한 줄 요약: DiffuRank은 document reranking을 autoregressive identifier generation이 아니라 parallel denoising과 constrained permutation prediction으로 재구성하고, pointwise, logits-listwise, permutation-listwise의 세 가지 방법을 통해 diffusion LLM의 bidirectional modeling과 structured decoding을 평가한 framework다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- LLM reranking의 병목을 model size보다 **sequential decoding과 output structure mismatch**에서 찾는다.
- dLLM을 일반 text generator로 쓰지 않고 ranking score, parallel logits, permutation이라는 서로 다른 interface로 분해해 비교한다.
- One-shot probability matrix와 Hungarian algorithm을 결합해 valid ranking을 보장하는 structured decoder를 보여준다.
- Iterative denoising step 수로 effectiveness와 latency를 조절하는 새로운 operating point를 제시한다.
- Retrieval system에서 generative model의 자유도를 그대로 믿기보다 task constraint를 inference algorithm에 명시해야 한다는 점을 ablation으로 보여준다.

이 논문은 diffusion LLM application paper이면서 동시에, **neural ranking output을 어떻게 구조화할 것인가**에 대한 systems paper로 읽는 편이 더 유용하다.

# 1. Problem Setting

## 1-1. Problem definition

일반적인 multi-stage retrieval system은 먼저 BM25나 dense retriever로 많은 document 중 candidate를 가져오고, 더 비싼 reranker로 top result의 순서를 다시 정한다. DiffuRank이 다루는 setting은 query $q$와 candidate set $D = \{d_1, ..., d_N\}$가 주어졌을 때, document를 relevance 순서로 재배열하는 문제다.

전통적인 reranker는 대체로 다음 두 방식에 가깝다.

- Cross-encoder가 각 query-document pair에 scalar relevance score를 준다.
- Listwise LLM이 여러 document를 함께 보고 identifier ranking sequence를 생성한다.

Pointwise cross-encoder는 단순하고 안정적이지만 candidate 사이의 상대 관계를 직접 모델링하기 어렵다. Listwise LLM은 candidate interaction을 볼 수 있지만, ranking을 자연어 generation처럼 처리하면서 새로운 문제가 생긴다.

이 논문이 겨냥하는 핵심 문제는 다음과 같다.

> 여러 candidate의 상대 relevance를 전역적으로 보면서도, left-to-right decoding의 latency, error propagation, format instability를 피할 수 있는가?

## 1-2. Why previous approaches are insufficient

기존 autoregressive LLM reranker의 한계는 크게 네 가지로 정리할 수 있다.

첫째, **decoding이 순차적이다**. Ranking sequence가 $N$개의 identifier로 구성되면, 뒤 token은 앞 token을 기다려야 한다. 여러 position을 동시에 확정하기 어렵기 때문에 list가 길어질수록 latency가 누적된다.

둘째, **초기 오류가 뒤 순서에 영향을 준다**. 첫 번째 identifier를 잘못 생성하면 이후 token distribution도 잘못된 prefix를 조건으로 계산한다. Ranking은 앞뒤 position 전체가 서로 제약되는 permutation 문제인데, autoregressive factorization은 이 global structure를 left-to-right conditional chain으로 바꾼다.

셋째, **output validity가 model generation에 의존한다**. 동일 identifier가 두 번 나오거나, 어떤 candidate가 빠지는 문제가 생길 수 있다. Post-processing으로 고칠 수 있지만, model score와 최종 ranking 사이에 heuristic이 들어간다.

넷째, **효율을 위해 generation을 제거한 방법은 model을 encoder처럼만 쓸 수 있다**. FIRST처럼 첫 token logits를 사용하거나 attention score를 활용하는 방법은 빠르지만, iterative refinement라는 generative capability는 거의 쓰지 않는다.

DiffuRank이 선택한 대안은 masked diffusion language model이다. dLLM은 전체 masked response position을 동시에 예측하고, 필요한 position만 다시 mask해 반복적으로 수정할 수 있다. 이 특성은 ranking에 다음 두 가지 가능성을 준다.

- 모든 candidate relevance를 parallel하게 계산한다.
- Ranking sequence 전체를 한 번에 보면서 uncertain position만 수정한다.

다만 이 가능성이 자동으로 좋은 reranker를 만드는 것은 아니다. Ranking은 identifier가 중복되지 않아야 하는 permutation이므로, diffusion decoding에도 별도의 structure-aware training과 constrained inference가 필요하다.

# 2. Core Idea

## 2-1. Main contribution

DiffuRank의 핵심 기여는 하나의 method보다 **세 가지 formulation을 같은 dLLM backbone 위에서 비교한 것**이다.

1. **Pointwise DiffuRank**
   - Query-document pair마다 binary relevance mask를 예측한다.
   - dLLM을 bidirectional cross-encoder처럼 사용한다.

2. **Logits-based Listwise DiffuRank**
   - 여러 document를 한 prompt에 넣고, document별 masked relevance position을 동시에 예측한다.
   - 한 forward에서 여러 relevance score를 얻는다.

3. **Permutation-based Listwise DiffuRank**
   - Ranking output 자체를 masked identifier sequence로 둔다.
   - Iterative constrained sampling 또는 one-shot assignment로 valid permutation을 만든다.

세 방법은 같은 diffusion backbone을 사용하지만, model output을 해석하는 방법이 다르다. Pointwise와 logits-listwise는 diffusion model의 representation과 parallel logits를 사용하고, permutation-listwise는 denoising generation을 structured prediction으로 사용한다.

논문의 실험 결과에서 가장 강한 variant는 permutation-based method다. 이 결과는 dLLM의 장점이 단순히 bidirectional encoder라는 데 있지 않고, **여러 output position을 동시에 보고 순서를 직접 구성하는 능력**에 있다는 해석을 가능하게 한다.

## 2-2. Design intuition

DiffuRank의 설계 직관은 ranking과 diffusion의 구조를 맞추는 데 있다.

Autoregressive ranking은 아래처럼 factorize된다.

$$
p(r_1, ..., r_N \mid q, D)
= \prod_{i=1}^{N} p(r_i \mid r_{<i}, q, D)
$$

이 표현은 자연어 sequence에는 익숙하지만, permutation에서는 모든 position이 서로 one-to-one constraint를 가진다는 점을 직접 반영하지 않는다.

반면 masked dLLM은 여러 masked position의 token distribution을 한 번에 낸다. 따라서 각 rank position $i$와 document identifier $j$ 사이의 score matrix를 만들 수 있다. 이 matrix를 assignment problem으로 보면, model은 text를 생성하는 대신 **candidate를 rank slot에 배치하는 cost function**을 제공한다.

이 관점이 중요한 이유는 두 가지다.

- Model의 역할과 algorithm의 역할을 분리할 수 있다.
  - dLLM은 각 position-document pair의 compatibility를 예측한다.
  - Hungarian algorithm은 duplicate와 missing이 없는 global permutation을 선택한다.
- Iterative sampling과 one-shot assignment를 같은 score space에서 비교할 수 있다.
  - Sampling은 uncertain position을 여러 step에 걸쳐 고친다.
  - Assignment는 한 번의 forward 뒤 global constraint를 즉시 만족시킨다.

결국 DiffuRank의 중요한 메시지는 "diffusion이면 parallel하다"보다 더 구체적이다.

> Parallel token prediction을 ranking에 쓰려면, token distribution을 permutation constraint와 연결하는 decoder가 필요하다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Top candidate document를 query relevance 순서로 rerank |
| Backbone | LLaDA-1.5-8B masked diffusion language model |
| Input | Query와 BM25 top-100 candidate document |
| Main variants | Pointwise, Logits-Listwise, Permutation-Listwise |
| Parallelism | Masked position logits를 한 forward에서 동시에 계산 |
| Structured inference | Constrained sampling 또는 Hungarian assignment |
| Training supervision | RankZephyr의 GPT-4 annotated ranking data |
| Main metric | NDCG@10 |
| Main benchmarks | TREC DL19, TREC DL20, BEIR 8 datasets |
| Difference from AR reranking | Left-to-right identifier decoding 대신 parallel scoring과 denoising 사용 |

## 3-2. Method comparison

| Method | Model output | Inference | Training signal | Main trade-off |
| --- | --- | --- | --- | --- |
| DiffuRank Pointwise | Pair별 binary relevance probability | Document마다 one-step denoising | Listwise distillation with RankNet or Cross-Entropy | Stable하지만 candidate interaction이 약함 |
| DiffuRank Logits-List | Document별 masked relevance probability | 한 window의 score를 single forward로 parallel 계산 | Listwise distillation with RankNet or Cross-Entropy | 빠르지만 long list는 sliding window 필요 |
| DiffuRank Perm-Samp | Masked ranking identifier sequence | Constrained multi-step denoising | MDLM SFT with DocID masking | Refinement 가능하지만 step 수만큼 latency 증가 |
| DiffuRank Perm-Assign | Position-document probability matrix | Single forward plus Hungarian algorithm | MDLM SFT with DocID masking | 빠르고 valid permutation을 보장하지만 external solver 사용 |

## 3-3. Pointwise reranking

Pointwise variant는 query-document pair마다 prompt를 만들고, response 끝에 하나의 mask token을 둔다. Model은 non-relevant token `0`과 relevant token `1`의 probability를 예측한다.

Document $d_i$의 relevance score는 다음처럼 정규화된다.

$$
s(q, d_i)
= \frac{p_1^{(i)}}{p_0^{(i)} + p_1^{(i)}}
$$

이 방식은 generation을 거의 하지 않는다. 한 번의 denoising step에서 mask position의 logits만 읽기 때문에, dLLM을 bidirectional cross-encoder처럼 사용한다.

장점은 output format이 단순하고 기존 reranking pipeline에 붙이기 쉽다는 점이다. 단점은 candidate document 사이의 상대 관계를 직접 보지 못한다는 점이다. 같은 query에 대한 score라도 다른 candidate set이 주어졌을 때 ranking context가 반영되지 않는다.

## 3-4. Logits-based listwise reranking

Logits-listwise variant는 여러 document를 하나의 prompt에 넣고, document마다 binary relevance mask를 하나씩 배치한다. Model은 모든 masked position의 logits를 한 번에 계산한다.

각 document의 score는 pointwise와 같은 식을 사용하지만, 차이는 **모든 document가 같은 context 안에 들어 있다는 점**이다. 따라서 $d_i$의 relevance prediction은 query뿐 아니라 다른 candidate document의 내용에도 영향을 받을 수 있다.

이 방식의 핵심은 explicit ranking sequence를 만들지 않는 것이다. Score를 얻은 뒤 descending order로 sort하면 된다. Autoregressive token generation이 없고, window 안에서는 single forward로 처리된다.

다만 paper는 현재 dLLM의 long-context capability가 충분하지 않다고 보고, candidate list가 길 때 RankGPT의 sliding window를 사용한다.

- Window size: 20
- Step size: 10
- Top-100 candidate를 overlapping window로 처리
- Local ranking을 반복적으로 merge

따라서 "top-100을 완전한 single forward로 jointly rerank한다"고 해석하면 안 된다. Parallelism은 각 window 안에서 성립하고, global ranking은 sliding-window composition에 의존한다.

## 3-5. Permutation-based listwise reranking

Permutation variant는 document에 `A`, `B`, `C` 같은 unique identifier를 붙이고, response에 ranking length만큼 mask slot을 만든다. 목표는 mask sequence를 가장 relevant한 identifier부터 least relevant identifier까지 채우는 것이다.

이 formulation은 ranking을 score sorting이 아니라 **identifier permutation generation**으로 다룬다. Paper는 두 inference strategy를 제안한다.

### 1) Constrained sampling

Vanilla dLLM sampling은 각 masked position이 독립적으로 identifier를 선택할 수 있다. 그러면 여러 position이 같은 identifier를 고르거나, 어떤 identifier가 끝까지 나오지 않는 문제가 생긴다.

DiffuRank은 매 denoising step에서 feasible identifier set을 갱신한다.

1. 현재 unmasked response에 이미 등장한 identifier를 feasible set에서 제거한다.
2. 모든 masked position과 남은 identifier 조합의 probability를 계산한다.
3. Position-identifier pair를 confidence 순서로 정렬한다.
4. 한 position과 한 identifier가 각 step에서 최대 한 번만 사용되도록 greedy assignment한다.
5. Low-confidence position은 다시 mask해 다음 step에서 수정한다.

Sampling step $K$는 effectiveness와 latency를 조절하는 knob다.

- 작은 $K$: 적은 forward call, 낮은 latency
- 큰 $K$: 더 많은 refinement 기회, 높은 latency
- 너무 큰 $K$: dataset에 따라 diminishing return 또는 성능 하락 가능

이 방법은 diffusion의 iterative correction을 유지하면서 permutation validity를 강제한다.

### 2) Permutation as assignment

Assignment variant는 iterative denoising을 사용하지 않는다. Fully masked ranking response에 대해 model을 한 번 실행하고, rank position $i$에서 document identifier $j$가 나올 probability를 계산한다.

$$
P_{i,j}
= p_{\theta}(r_i = ID_j \mid q, D, r_{mask})
$$

이를 cost matrix로 바꾼다.

$$
C_{i,j} = -\log P_{i,j}
$$

최종 ranking은 다음 minimum-cost assignment로 구한다.

$$
\pi^*
= \arg\min_{\pi \in S_N}
\sum_{i=1}^{N} C_{i, \pi(i)}
$$

여기서 $S_N$은 $N$개 document의 모든 permutation 집합이다. Paper는 Hungarian algorithm으로 이 문제를 푼다.

이 설계는 매우 실용적이다.

- Model forward는 한 번이다.
- 모든 identifier가 정확히 한 번 배치된다.
- Duplicate와 missing이 구조적으로 불가능하다.
- Probability matrix에 다른 business constraint를 추가할 여지가 있다.

Hungarian algorithm의 계산량은 일반적으로 $O(N^3)$이지만, paper setting의 listwise window는 20개 document다. 이 규모에서는 8B model forward에 비해 assignment overhead가 상대적으로 작을 가능성이 높다. 다만 더 큰 candidate window에서의 end-to-end latency는 별도 검증이 필요하다.

## 3-6. Structure-aware training

Pointwise와 logits-listwise는 teacher ranking을 score supervision으로 바꾼다. Paper는 permutation distillation을 사용하고, 두 loss를 비교한다.

- Listwise Cross-Entropy
- Pairwise RankNet loss

Permutation variant는 masked diffusion SFT를 사용한다. Prompt와 target ranking sequence를 concatenate하고, response token 일부를 mask한 뒤 원래 ranking을 복원하도록 학습한다.

Paper가 추가한 중요한 장치는 **DocID Mask**다. Vanilla random mask는 response의 모든 token을 독립적으로 가릴 수 있지만, ranking에서 중요한 것은 identifier position이다. DocID Mask는 document identifier token만 선택적으로 corrupt해, partial ranking을 보고 missing identifier를 복원하도록 학습한다.

이 장치는 training corruption을 target structure와 맞추는 역할을 한다. Ablation에서도 Perm-Assign은 Random Mask보다 DocID Mask에서 더 좋은 결과를 보인다.

# 4. Training / Data / Recipe

## 4-1. Data

Training에는 RankZephyr가 사용한 MS MARCO 기반 dataset을 사용한다.

| Item | Setting |
| --- | --- |
| Training samples | 약 40K |
| Source | MS MARCO |
| Annotation | GPT-4가 만든 ranked candidate list |
| Candidate count | Sample당 최대 20 documents |
| Teacher style | RankGPT 계열 permutation supervision |
| Evaluation candidates | BM25 top-100 |

이 data choice는 결과 해석에서 중요하다. DiffuRank은 human relevance label만으로 학습된 것이 아니라, strong autoregressive teacher가 만든 listwise ordering을 distill한다. 따라서 diffusion architecture의 효과와 teacher ranking prior의 효과가 함께 들어간다.

## 4-2. Training strategy

Paper의 주요 training recipe는 다음과 같다.

| Item | Value |
| --- | --- |
| Backbone | LLaDA-1.5-8B |
| Adaptation | LoRA |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0 |
| Epochs | 3 |
| Optimizer | AdamW |
| Learning rate | $1 \times 10^{-4}$ constant |
| Weight decay | 0.01 |
| Total batch size | 64 |
| Hardware | 8 x NVIDIA A100 80GB |
| Memory and parallelism | Mixed precision, DeepSpeed ZeRO-3 |
| Long context | RoPE scaling factor 14.0, context length 16K |

Pointwise와 logits-listwise는 같은 backbone과 data를 사용하면서 relevance score objective를 학습한다. Permutation variant는 ranking response를 직접 denoise하는 SFT objective를 사용한다.

다만 paper recipe와 현재 공개 config 사이에는 차이가 있다. Paper는 RoPE scaling factor 14.0과 DeepSpeed ZeRO-3를 보고하지만, 공개된 logits RankNet config와 random-mask config에는 factor 4.0과 `scripts/zero2.json`이 적혀 있다. Method 해석에는 영향을 주지 않지만, exact reproduction을 시도할 때는 어떤 config가 paper table에 대응하는지 먼저 확인해야 한다.

## 4-3. Engineering notes

실무적으로 가져갈 만한 engineering point는 다음과 같다.

1. **Ranking output에는 grammar가 아니라 constraint가 필요하다**
   - Identifier sequence가 syntactically valid해 보여도 duplicate나 missing이 있으면 ranking으로는 invalid하다.
   - Constrained sampling이나 assignment처럼 validity를 algorithm level에서 보장하는 편이 안전하다.

2. **One-shot assignment가 기본 baseline이어야 한다**
   - Iterative sampling은 diffusion의 장점을 잘 보여주지만, serving에서는 forward call 수가 직접 latency로 이어진다.
   - Single forward와 Hungarian algorithm 조합은 더 단순한 operating point다.

3. **Adaptive step policy가 가능하다**
   - 모든 query에 같은 $K$를 쓸 필요는 없다.
   - Probability matrix entropy나 top assignment margin이 낮을 때만 추가 denoising을 수행하는 방식이 자연스럽다.

4. **Sliding window는 hidden system component다**
   - Listwise model 성능은 model만이 아니라 window size, step size, merge order에 영향을 받는다.
   - Production comparison에서는 prompt length와 window composition을 동일하게 맞춰야 한다.

5. **Score calibration을 따로 봐야 한다**
   - Pointwise와 logits-listwise는 binary token probability를 relevance score로 사용한다.
   - Ranking에는 relative order만 중요해 보여도, adaptive filtering이나 fallback에 쓰려면 score calibration이 필요하다.

6. **Assignment matrix는 product constraint를 넣기 좋은 interface다**
   - Diversity, source cap, freshness, policy constraint 같은 추가 rule을 optimization layer에 넣을 수 있다.
   - 다만 이 경우 standard Hungarian assignment보다 더 일반적인 constrained optimization이 필요할 수 있다.

# 5. Evaluation

## 5-1. Experimental setup

Paper는 다음 benchmark에서 BM25 top-100 candidate를 rerank한다.

- TREC DL19
- TREC DL20
- TREC Covid
- NFCorpus
- Touche2020
- DBPedia
- SciFact
- Signal1M
- TREC News
- Robust04

Metric은 NDCG@10이다. Zero-shot에서는 LLaMA-3.1-8B, Qwen3-8B, vanilla LLaDA-1.5-8B와 비교한다. Fine-tuned setting에서는 monoBERT, monoT5, RankZephyr와 함께, 동일한 training data를 사용해 fine-tune한 LLaMA-3.1-8B와 Qwen3-8B reranker를 비교한다.

## 5-2. Main results

핵심 수치를 과장 없이 정리하면 다음과 같다. 표의 값은 paper가 보고한 NDCG@10 scale을 그대로 적었다.

| Setting | Model | DL19 | DL20 | BEIR 8-dataset Avg. |
| --- | --- | ---: | ---: | ---: |
| Zero-shot AR best avg. | Qwen3-8B Listwise | - | - | 50.91 |
| Zero-shot dLLM best avg. | DiffuRank Perm-Assign | - | - | 49.30 |
| Fine-tuned AR best avg. | LLaMA-3.1-8B Listwise | 74.79 | 71.56 | 54.28 |
| Fine-tuned dLLM | DiffuRank Perm-Samp | 72.92 | 71.72 | 55.10 |
| Fine-tuned dLLM | DiffuRank Perm-Assign | 73.44 | 71.81 | 55.21 |

이 표에서 중요한 점은 세 가지다.

첫째, **zero-shot에서 DiffuRank이 best AR listwise model을 넘지는 않는다**. Qwen3-8B Listwise의 BEIR average가 50.91이고, 가장 좋은 dLLM variant인 Perm-Assign은 49.30이다. 따라서 diffusion architecture 자체가 prompting만으로 즉시 우위를 만든다고 해석하면 안 된다.

둘째, **fine-tuning 이후 permutation variant가 강해진다**. BEIR 8-dataset average에서 Perm-Samp는 55.10, Perm-Assign은 55.21로 paper가 직접 fine-tune한 AR LLM baseline보다 높은 average를 보고한다.

셋째, **모든 dataset에서 일관되게 이기는 것은 아니다**. 예를 들어 DL19에서는 LLaMA-3.1-8B Listwise가 74.79로 Perm-Assign의 73.44보다 높다. 반면 DL20에서는 Perm-Assign이 71.81로 71.56보다 조금 높다. DiffuRank의 claim은 universal dominance보다 cross-domain average와 efficiency-effectiveness trade-off에 가깝다.

## 5-3. What really matters in the experiments

### 1) Permutation structure가 score formulation보다 강하다

Fine-tuned DiffuRank 중 Pointwise average는 52.68, Logits-List는 54.09, Perm-Samp는 55.10, Perm-Assign은 55.21이다. Candidate relevance를 independent scalar로 보는 것보다, ranking output을 permutation으로 직접 모델링하는 편이 더 강하다.

이 결과는 diffusion model이 reranking에 맞는 이유를 조금 더 구체화한다. Bidirectional encoder이기 때문만이 아니라, 여러 rank position을 동시에 예측하고 global order를 구성할 수 있기 때문이다.

### 2) Ranking-aware objective가 중요하지만 결과는 variant별로 다르다

Table 3에서 Pointwise는 RankNet이 Cross-Entropy보다 명확히 좋다.

| Model | Training | DL19 | DL20 |
| --- | --- | ---: | ---: |
| Pointwise | RankNet | 73.06 | 69.69 |
| Pointwise | Cross-Entropy | 70.53 | 66.94 |
| Logits-List | RankNet | 72.95 | 69.78 |
| Logits-List | Cross-Entropy | 70.69 | 70.63 |

Pointwise에서는 relative preference를 학습하는 RankNet이 두 dataset 모두 좋다. 하지만 Logits-List는 DL19에서 RankNet이 높고, DL20에서는 Cross-Entropy가 70.63으로 RankNet 69.78보다 높다. Paper 본문은 RankNet이 두 method와 두 dataset에서 일관되게 우수하다고 서술하지만, table 수치와 완전히 일치하지 않는다. 이 부분은 원문 revision에서 확인할 필요가 있다.

### 3) Structure-aware masking은 Perm-Assign에서 특히 중요하다

| Model | Mask | DL19 | DL20 |
| --- | --- | ---: | ---: |
| Perm-Samp | DocID Mask | 72.70 | 71.14 |
| Perm-Samp | Random Mask | 72.55 | 71.20 |
| Perm-Assign | DocID Mask | 73.44 | 71.81 |
| Perm-Assign | Random Mask | 72.06 | 71.02 |

Perm-Samp는 mask strategy에 크게 민감하지 않고 DL20에서는 Random Mask가 0.06 높다. 반면 Perm-Assign은 DocID Mask가 두 dataset 모두 높고, DL19에서는 1.38 차이가 난다.

이 결과는 one-shot assignment가 training score matrix의 구조에 더 민감할 수 있음을 시사한다. Iterative sampling은 remasking으로 일부 오류를 고칠 수 있지만, assignment는 한 번의 matrix 품질에 더 직접적으로 의존한다.

### 4) Constrained sampling은 optional trick이 아니다

Vanilla sampling은 valid permutation을 거의 만들지 못한다.

| Sampling | DL19 NDCG@10 | DL19 Correct% | DL20 NDCG@10 | DL20 Correct% |
| --- | ---: | ---: | ---: | ---: |
| Vanilla | 69.22 | 16.54% | 65.07 | 17.08% |
| Constrained | 72.92 | 100% | 71.72 | 100% |

Duplicate와 missing을 막는 constraint를 넣자 Correct%가 100%가 되고 ranking quality도 크게 오른다. 이 ablation의 메시지는 명확하다.

> Structured task에서 model의 flexible generation을 쓰려면, output structure를 inference algorithm에 명시해야 한다.

### 5) Denoising step은 많을수록 좋은 hyperparameter가 아니다

DL19에서는 $K$가 1에서 약 10까지 증가할 때 성능이 좋아지고, 20에서는 조금 떨어진다. DL20에서는 약 3에서 4 step 부근이 가장 좋고, 이후 step이 늘수록 성능이 낮아진다.

즉 diffusion reranking은 "더 오래 denoise하면 더 정확하다"는 단순 관계가 아니다. Dataset별 ambiguity와 model calibration에 따라 over-refinement가 생길 수 있다. Serving에서는 fixed large $K$보다 small $K$ 또는 adaptive stopping이 더 현실적이다.

### 6) Model은 extreme rank를 먼저 채우고 middle을 나중에 정리한다

Iterative filling analysis에서 초기 step은 top position과 bottom position을 먼저 resolve하는 경향을 보인다. Middle position은 뒤 step에서 더 많이 채워진다.

이는 model이 다음과 같은 coarse-to-fine 전략을 학습했을 가능성을 보여준다.

1. 매우 relevant한 document와 매우 irrelevant한 document를 먼저 분리한다.
2. 남은 ambiguous candidate의 middle order를 뒤에서 조정한다.

이 behavior는 ranking task의 직관과 잘 맞는다. 실제 reranking에서도 top과 bottom은 상대적으로 확실하고, 중간 candidate 사이의 차이가 더 어렵다.

### 7) Efficiency claim은 small-step operating point로 읽어야 한다

Paper의 Figure 4는 vanilla Transformers implementation에서 top-100 reranking latency를 비교한다. Perm-Samp는 step 수가 늘어날수록 latency가 거의 선형으로 증가한다. Paper는 $K=4$에서 AR listwise baseline보다 높은 effectiveness를 보이면서 latency는 3분의 1 미만이라고 보고한다. Perm-Assign은 single forward라 더 낮은 latency operating point를 보인다.

다만 이 결과는 acceleration technique을 사용하지 않은 비교다. Production에서는 continuous batching, KV cache behavior, custom kernel, prompt length, GPU type, batch size가 latency를 크게 바꾼다. 따라서 상대적인 design insight는 유효하지만 절대 latency를 serving SLA로 바로 옮기면 안 된다.

# 6. Limitations

논문에는 별도의 Limitations section이 없으므로, 아래 항목은 paper setting과 공개 artifact를 바탕으로 구분해 정리한 주의점이다.

1. **Backbone이 LLaDA-1.5-8B 하나에 집중되어 있다**
   - Diffusion model family, scale, post-training recipe가 바뀌어도 같은 ranking advantage가 유지되는지는 알기 어렵다.
   - Dream이나 이후 dLLM에 대한 architecture-level replication이 필요하다.

2. **Top-100 global listwise inference가 실제로는 sliding window다**
   - Window size 20, step 10으로 local ranking을 merge한다.
   - 따라서 candidate 100개 전체를 동시에 보는 global permutation과는 다르다.
   - Window boundary와 merge order가 최종 ranking에 영향을 줄 수 있다.

3. **Training supervision이 GPT-4 annotated teacher ranking에 의존한다**
   - DiffuRank은 architecture만 학습하는 것이 아니라 RankZephyr-style teacher preference를 distill한다.
   - Teacher bias와 annotation error가 student ranking에 들어갈 수 있다.

4. **Evaluation scope가 BM25 top-100 reranking과 NDCG@10에 집중되어 있다**
   - Dense retriever candidate, multilingual search, domain-specific enterprise search, freshness-sensitive search는 별도 검증이 필요하다.
   - End-to-end recall, throughput, cost per query, memory usage는 주요 table에 없다.

5. **Efficiency result의 production portability가 제한적이다**
   - Figure 4는 acceleration 없이 vanilla Transformers로 비교한다.
   - Hardware, batch policy, kernel maturity, serving framework가 바뀌면 상대 latency가 달라질 수 있다.

6. **Permutation validity는 dLLM 단독 능력이 아니다**
   - Vanilla sampling의 Correct%는 약 16%에서 17%다.
   - 좋은 결과는 constrained sampling 또는 Hungarian solver를 포함한 system 전체의 결과다.

7. **Assignment solver가 커질 때의 비용과 constraint interaction이 남는다**
   - Window 20에서는 Hungarian overhead가 작을 수 있지만, 더 큰 global candidate set에서는 $O(N^3)$ 비용이 무시되지 않을 수 있다.
   - Diversity나 policy constraint를 추가하면 standard assignment보다 복잡한 solver가 필요하다.

8. **Uncertainty와 calibration이 충분히 평가되지 않는다**
   - Probability matrix와 denoising confidence를 제공하지만, confidence calibration이나 failure detection 실험은 제한적이다.
   - 실제 search에서는 low-confidence query를 stronger reranker로 넘기는 fallback policy가 중요하다.

9. **공개 repository의 reproduction completeness는 확인이 필요하다**
   - README는 세 종류의 released model과 evaluation command를 제공한다.
   - 현재 공개 tree에서 README가 호출하는 training entrypoint의 완전한 제공 여부는 다시 확인할 필요가 있다.

# 7. My Take

## 7-1. Why this matters for my work

DiffuRank의 가장 큰 가치는 diffusion LLM 자체보다 **model logits와 structured solver를 결합하는 interface**에 있다.

LLM application은 model이 output text 전체를 자유롭게 생성하도록 두는 경우가 많다. 하지만 ranking, matching, routing, allocation처럼 hard constraint가 있는 문제는 자유 generation과 잘 맞지 않는다. DiffuRank은 model이 각 local choice의 score를 만들고, algorithm이 global validity를 보장하는 분업을 보여준다.

특히 Perm-Assign은 production 관점에서 매력적이다.

- Model forward는 한 번이다.
- Output format failure가 없다.
- Probability matrix를 inspection할 수 있다.
- Business constraint를 optimization layer로 확장할 수 있다.
- Sampling step hyperparameter가 없다.

반대로 Perm-Samp는 research 관점에서 더 흥미롭다. Model이 top과 bottom을 먼저 결정하고 middle을 refine하는 behavior는 iterative ranking policy로 해석할 수 있다. 다만 serving에 넣을 때는 every-query fixed $K$보다 uncertainty-aware adaptive step이 필요해 보인다.

이 논문은 "dLLM이 AR LLM을 대체한다"는 결론보다 다음 원칙을 준다.

> Structured output task에서는 generation paradigm보다 score representation, constraint enforcement, stopping policy의 결합이 더 중요할 수 있다.

## 7-2. Reuse potential

재사용하고 싶은 포인트는 다음과 같다.

1. **One-forward structured decoding**
   - LLM이 position-item score matrix를 만들고, external optimizer가 valid output을 선택한다.
   - Ranking뿐 아니라 job assignment, candidate routing, resource allocation에 적용할 수 있다.

2. **Constrained denoising**
   - Diffusion step마다 feasible token set을 업데이트한다.
   - Tool plan에서 이미 사용한 action을 제외하거나, entity matching에서 one-to-one constraint를 유지하는 방식으로 확장할 수 있다.

3. **Adaptive refinement budget**
   - Easy query는 one-shot assignment로 끝낸다.
   - Ambiguous query만 추가 sampling step을 사용한다.
   - Confidence margin이나 assignment entropy를 stopping signal로 쓸 수 있다.

4. **Coarse-to-fine candidate handling**
   - Top과 bottom을 먼저 고정하고 middle candidate에 compute를 집중한다.
   - Large candidate set에서 dynamic pruning과 결합할 수 있다.

5. **Score model과 constraint layer의 분리**
   - Relevance model은 neural component로 유지한다.
   - Diversity, freshness, source quota, policy rule은 deterministic optimization layer에서 처리한다.
   - Product requirement가 바뀌어도 model을 매번 retrain하지 않는 구조를 만들 수 있다.

6. **A fair production baseline**
   - DiffuRank을 도입하기 전에 strong cross-encoder, FIRST-style logit reranker, small generative listwise model과 latency, throughput, NDCG를 같은 serving stack에서 비교해야 한다.
   - Diffusion의 novelty보다 operational gain이 실제로 남는지 먼저 확인하는 편이 좋다.

## 7-3. Follow-up papers

- Large Language Diffusion Models
- LLaDA 1.5: Variance-Reduced Preference Optimization for Large Language Diffusion Models
- Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents
- FIRST: Faster Improved Listwise Reranking with Single Token Decoding
- RankZephyr: Effective and Robust Zero-Shot Listwise Reranking Is a Breeze
- DiffuGR: Generative Document Retrieval with Diffusion Language Models
- E2Rank: Your Text Embedding Can Also Be an Effective and Efficient Listwise Reranker

# 8. Summary

- DiffuRank은 document reranking을 pointwise score, parallel listwise logits, permutation generation의 세 formulation으로 dLLM에 맞게 재설계한다.
- 가장 강한 variant는 permutation-based method이며, constrained sampling과 one-shot Hungarian assignment를 통해 valid ranking을 만든다.
- Zero-shot에서는 best AR listwise baseline보다 낮지만, fine-tuning 후 BEIR average에서 Perm-Samp와 Perm-Assign이 strong 8B AR baseline을 넘는다.
- 핵심 ablation은 diffusion model만으로는 부족하고 ranking-aware loss, DocID masking, permutation constraint가 필요하다는 점을 보여준다.
- 실무적으로 가장 재사용 가치가 큰 아이디어는 free-form generation보다 probability matrix와 structured solver를 결합하는 one-forward decoding pattern이다.
