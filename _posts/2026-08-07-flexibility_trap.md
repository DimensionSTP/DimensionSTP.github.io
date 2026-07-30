---
layout: single
title: "The Flexibility Trap: Rethinking the Value of Arbitrary Order in Diffusion Language Models Review"
categories: Study-concept
tag: [Diffusion-LLM, Reinforcement-Learning, Reasoning]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2601.15165)

[Project page](https://nzl-thu.github.io/the-flexibility-trap)

> 한 줄 요약: 이 논문은 diffusion language model의 arbitrary-order generation이 일반 reasoning에서 더 넓은 탐색 공간을 제공한다는 통념을 뒤집고, high-entropy logical fork를 뒤로 미루는 과정에서 solution diversity가 줄어드는 entropy degradation을 밝힌다. 이를 바탕으로 RL 단계에서는 left-to-right policy로 standard GRPO를 적용하고, inference에서는 parallel decoding을 유지하는 JustGRPO를 제안한다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Diffusion LLM의 장점으로 자주 함께 묶이는 parallel decoding과 arbitrary-order generation을 분리해서 평가한다.
- Pass@1이 높다고 reasoning boundary도 넓다고 볼 수 없으며, Pass@k와 solution coverage가 다른 결론을 줄 수 있음을 보여준다.
- Diffusion-specific RL의 복잡성을 정당화하던 arbitrary-order policy 자체가 오히려 general reasoning exploration을 제한할 수 있다는 반례를 제시한다.
- RL rollout order와 inference execution order를 분리하면, sequential exploration과 parallel decoding을 동시에 취할 수 있다는 실용적 설계를 보여준다.
- ICML 2026 Outstanding Paper로 선정된 만큼, diffusion language model의 post-training 방향을 다시 생각하게 만드는 문제 제기가 선명하다.

Diffusion language model, dLLM은 전체 sequence를 masked state에서 점진적으로 복원한다. 이 구조는 여러 token을 동시에 복원할 수 있고, 반드시 왼쪽에서 오른쪽으로 생성할 필요도 없다. 그래서 arbitrary-order generation은 autoregressive model보다 더 넓은 reasoning path를 탐색할 수 있을 것처럼 보인다.

하지만 이 논문은 "순서를 자유롭게 고를 수 있다"와 "실제로 더 많은 reasoning path에 도달한다"가 같은 말이 아니라고 지적한다. Confidence-based sampler는 어려운 token보다 쉬운 token을 먼저 채운다. 문제는 reasoning에서 어려운 token이 단순한 noise가 아니라, 이후 경로를 갈라놓는 logical fork일 수 있다는 점이다.

논문의 핵심은 flexibility 자체를 부정하는 데 있지 않다. 어떤 flexibility가 inference efficiency에 도움이 되고, 어떤 flexibility가 exploration을 약화하는지를 분해한다. Parallel decoding은 유지할 수 있지만, RL rollout에서 arbitrary order를 반드시 보존할 필요는 없다는 것이 이 논문의 결론이다.

# 1. Problem Setting

## 1-1. Problem definition

Diffusion language model은 fully masked sequence에서 시작해 여러 denoising step을 거치며 token을 복원한다. 일반적인 confidence-based decoding에서는 현재 예측 confidence가 높은 위치를 먼저 확정하고, confidence가 낮은 위치는 이후 step으로 미룬다.

이 과정은 두 가지 서로 다른 자유도를 제공한다.

1. Parallel decoding
   - 한 step에서 여러 token을 동시에 복원할 수 있다.
   - 동일한 sequence length에서도 autoregressive decoding보다 적은 model call로 생성을 끝낼 가능성이 있다.

2. Arbitrary-order generation
   - 다음에 복원할 token 위치를 left-to-right로 고정하지 않는다.
   - 현재 confidence에 따라 sequence 뒤쪽이나 중간 위치를 먼저 채울 수 있다.

기존 직관은 arbitrary order가 autoregressive order를 포함하는 더 큰 action space를 가지므로, 더 넓은 solution space를 탐색할 수 있다는 것이다. Sudoku처럼 constraint가 전체에 퍼져 있는 문제에서는 실제로 non-sequential decoding이 유리할 수 있다.

그러나 mathematics와 coding 같은 general reasoning은 다른 구조를 가진다. Reasoning trace에는 다음 단계의 방향을 결정하는 소수의 high-entropy token이 존재한다. 예를 들어 "Therefore", "Thus", "Since" 같은 transition token은 문장 장식이 아니라, 어떤 논리적 분기를 선택할지 결정하는 역할을 할 수 있다.

논문은 이때 다음 질문을 던진다.

| Question | Meaning |
| --- | --- |
| Arbitrary order는 solution space를 실제로 넓히는가 | 이론적 permutation 수가 아니라 practical sampling에서 더 많은 정답 경로를 찾는가 |
| Pass@1과 reasoning boundary는 같은가 | 단일 sample의 성공률과 많은 sample에서의 coverage를 구분해야 하는가 |
| Diffusion-specific RL이 필요한가 | Arbitrary-order trajectory를 보존하기 위한 복잡성이 실제 성능에 기여하는가 |
| Training order와 inference order를 분리할 수 있는가 | RL은 sequential하게 하고 inference는 parallel하게 유지할 수 있는가 |

논문은 reasoning potential을 보기 위해 Pass@k를 사용한다. $n$개의 sample 중 $c$개가 정답일 때 unbiased estimator는 다음과 같다.

$$
\mathrm{Pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}
$$

Pass@1은 현재 sampling policy가 한 번에 정답을 낼 확률을 본다. 반면 큰 $k$의 Pass@k는 sampling budget을 늘렸을 때 새로운 정답 path가 계속 발견되는지를 본다. RLVR 관점에서는 base policy가 positive reward를 받을 수 있는 정답 trajectory를 sampling해야만 그 경로를 강화할 수 있으므로, Pass@k는 exploration boundary의 proxy로 쓰인다.

## 1-2. Why previous approaches are insufficient

기존 diffusion RL 접근은 arbitrary-order generation을 dLLM의 본질적 장점으로 보고, RL에서도 이를 유지하려 했다. 하지만 이 선택은 세 가지 복잡성을 만든다.

### 1) Token-level credit assignment가 모호하다

Autoregressive policy에서는 $o_k$의 probability가 prefix $o_{<k}$에 대해 정의된다. 따라서 token-level importance ratio와 policy gradient를 계산하기 쉽다.

반면 dLLM의 state는 어떤 token이 언제 unmask되었는지에 따라 달라진다. 같은 final sequence라도 서로 다른 denoising trajectory를 거칠 수 있고, 특정 token의 probability를 하나의 고정된 causal prefix에 대응시키기 어렵다.

### 2) Sequence likelihood가 trajectory marginalization을 요구한다

Diffusion policy의 final output likelihood는 가능한 denoising trajectory를 모두 합쳐야 한다.

$$
\pi_\theta(o \mid q) = \sum_{\tau \in \mathcal{T}} \pi_\theta(o, \tau \mid q)
$$

Sequence length가 $N$일 때 order trajectory의 수는 대략 $O(N!)$로 증가한다. Exact likelihood는 사실상 계산하기 어렵고, 기존 방법은 ELBO나 trajectory-level surrogate에 의존한다.

### 3) Rollout policy와 learner objective가 어긋날 수 있다

실제 rollout은 confidence-based sampler가 만든 distribution에서 나온다. 하지만 optimization target은 base diffusion likelihood나 그 approximation일 수 있다. 그러면 sample을 만든 policy와 update가 최적화하는 policy가 달라지는 sampler-learner mismatch가 생긴다.

이 복잡성은 arbitrary order가 reasoning exploration에 실질적 이득을 줄 때 정당화된다. 반대로 arbitrary order가 high-entropy fork를 회피해 solution coverage를 줄인다면, 복잡한 diffusion-specific RL은 불필요한 flexibility tax가 될 수 있다.

# 2. Core Idea

## 2-1. Main contribution

이 논문의 핵심 기여는 세 단계로 정리할 수 있다.

### 1) Arbitrary order가 general reasoning boundary를 줄일 수 있음을 보인다

저자들은 LLaDA-Instruct, Dream-Instruct, LLaDA 1.5를 GSM8K, MATH-500, HumanEval, MBPP에서 비교한다. Arbitrary order는 일부 setting에서 Pass@1이 경쟁력 있거나 더 높지만, $k$가 커질수록 Pass@k curve가 AR order보다 평평해진다.

즉 한 번에 그럴듯한 답을 만드는 능력과, 여러 sample을 통해 다른 정답 경로를 찾는 능력이 분리된다.

### 2) Entropy degradation이라는 mechanism을 제시한다

Confidence-based decoding은 high-confidence token을 먼저 확정한다. Reasoning trace에서 high-entropy logical fork는 뒤로 밀리고, 그 사이 future context가 먼저 채워진다.

나중에 fork token을 복원할 때는 이미 오른쪽 context가 결정되어 있다. 원래 여러 방향으로 갈라질 수 있었던 token은 이제 완성된 문장을 이어주는 connector로 바뀐다. 논문은 이처럼 logical fork의 entropy가 사후 context에 의해 낮아지는 현상을 entropy degradation이라고 부른다.

### 3) JustGRPO로 training order와 inference order를 분리한다

JustGRPO는 RL rollout을 left-to-right order로 제한한다. 그러면 dLLM 위에 exact autoregressive policy를 정의할 수 있고, standard GRPO를 그대로 적용할 수 있다.

중요한 점은 model architecture를 causal transformer로 바꾸지 않는다는 것이다. Bidirectional attention과 masked diffusion objective는 유지된다. AR order는 RL exploration을 위한 scaffold일 뿐이며, 학습 후 inference에서는 다시 parallel decoding을 사용할 수 있다.

## 2-2. Design intuition

이 논문의 설계 직관은 "학습에서 어떤 경로를 탐색할 것인가"와 "배포에서 어떻게 빠르게 실행할 것인가"를 분리하는 데 있다.

Autoregressive rollout은 매 step에서 현재의 uncertainty를 피할 수 없다. High-entropy token이 나오면 그 자리에서 선택해야 한다. 이 선택이 이후 reasoning branch를 만든다.

Arbitrary-order rollout은 uncertainty를 건너뛸 수 있다. 쉬운 future token을 먼저 생성하고, 어려운 fork를 마지막에 맞춘다. 이 방식은 single-sample coherence에는 도움이 될 수 있지만, 다양한 branch를 sampling할 기회를 줄인다.

JustGRPO는 이 차이를 다음처럼 이용한다.

- RL exploration: AR order로 uncertainty를 직접 통과한다.
- Policy optimization: exact factorization이 가능한 standard GRPO를 사용한다.
- Model representation: bidirectional dLLM backbone을 그대로 유지한다.
- Inference execution: multiple token을 한 번에 복원하는 parallel sampler를 사용한다.

즉 reasoning capability를 얻는 order와 serving efficiency를 얻는 order를 같게 둘 필요가 없다는 주장이다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | dLLM의 general reasoning boundary를 넓히면서 diffusion inference의 parallelism을 유지 |
| Diagnosis | Arbitrary-order decoding이 logical fork를 우회해 entropy degradation을 유발 |
| Training policy | dLLM 위에 정의한 exact left-to-right autoregressive policy |
| Optimization | Standard GRPO without diffusion-specific trajectory approximation |
| Architecture change | 없음. Bidirectional attention과 masked diffusion backbone 유지 |
| Inference | Confidence-based parallel decoding 또는 training-free parallel sampler 사용 가능 |
| Main distinction | Reasoning exploration order와 inference execution order를 분리 |

## 3-2. Module breakdown

### 1) AR order와 arbitrary order 비교

논문의 reasoning-boundary 분석은 두 decoding mode를 비교한다.

- AR Order
  - 항상 가장 왼쪽의 unresolved position을 다음 token으로 복원한다.
  - $B=1$인 block decoding과 같다.
  - High-entropy decision을 순서대로 통과한다.

- Arbitrary Order
  - 한 block 안에서 confidence가 높은 위치를 우선 복원한다.
  - Low-confidence position은 remask하거나 뒤로 미룬다.
  - 논문의 기본 분석에서는 maximum 256 tokens, 256 steps, block size $B=32$, temperature 0.6을 사용한다.

Block size $B$는 order flexibility의 정도를 조절한다. $B=1$이면 pure AR이고, $B$가 클수록 한 block 안에서 sampler가 위치를 자유롭게 고른다. HumanEval에서 $B$를 늘리면 $k=8$, $32$, $128$ 모두 Pass@k가 일관되게 감소한다.

### 2) Logical fork와 entropy degradation

논문은 arbitrary-order sampler가 자주 미루는 token을 조사한다. 그 결과 logical connector와 transition word가 많이 포함된다.

이 token들은 global average에서는 소수이지만, reasoning path를 가르는 역할을 한다. AR order에서는 해당 위치에 도달했을 때 여러 후보가 열려 있어 entropy가 높다. Arbitrary order에서는 future context가 먼저 정해지므로, 나중에 같은 위치를 채울 때 entropy가 낮아진다.

여기서 중요한 점은 전체 token 평균 entropy가 비슷하더라도, reasoning에 결정적인 minority token의 entropy는 크게 다를 수 있다는 것이다. 평균 uncertainty만 보면 exploration collapse를 놓칠 수 있다.

### 3) dLLM 위의 autoregressive policy 정의

JustGRPO는 position $k$의 token을 예측할 때, 이전 token은 observed state로 두고 이후 token은 모두 mask한다.

$$
\widetilde{x}_k = [o_1, \ldots, o_{k-1}, \mathrm{[MASK]}, \ldots, \mathrm{[MASK]}]
$$

Model은 모든 masked position에 logits를 출력하지만, AR policy는 position $k$의 logits만 사용한다.

$$
\pi_\theta^{\mathrm{AR}}(\cdot \mid o_{<k}, q)
= \mathrm{Softmax}(f_{\theta,k}(\widetilde{x}_k, q))
$$

이렇게 정의하면 sequence likelihood가 exact product로 factorize된다.

$$
\pi_\theta^{\mathrm{AR}}(o \mid q)
= \prod_{k=1}^{|o|} \pi_\theta^{\mathrm{AR}}(o_k \mid o_{<k}, q)
$$

이 formulation은 causal mask를 추가하는 것이 아니다. Model은 여전히 full bidirectional attention을 사용하지만, input에서 future token을 mask해 policy interface를 causal하게 만든다.

### 4) Standard GRPO optimization

각 query마다 group of rollouts를 생성하고, verifiable reward를 group statistics로 normalize한다. 이후 token-level importance ratio와 clipping을 사용하는 standard GRPO objective로 update한다.

Diffusion-specific trajectory를 따로 정의하지 않기 때문에 다음 요소가 사라진다.

- Denoising permutation marginalization
- ELBO-based policy-ratio approximation
- Arbitrary-order token credit assignment
- Confidence sampler와 diffusion likelihood 사이의 mismatch

Method의 새로움은 새로운 policy gradient 수식을 추가하는 데 있지 않다. 오히려 unnecessary flexibility를 제거해 standard objective가 다시 정확하게 적용되도록 만든다.

### 5) Parallel decoding preservation

RL은 AR rollout으로 수행하지만, 학습된 model은 diffusion architecture를 유지한다. 따라서 inference에서는 multiple positions를 동시에 unmask할 수 있다.

논문은 training-free EB Sampler를 적용해 한 step에서 여러 token을 복원한다. Parallelism을 높여도 base model보다 JustGRPO의 gain이 유지되며, MBPP에서는 token per step을 늘렸을 때도 개선 폭이 남는다.

이 결과는 AR training이 반드시 AR serving을 의미하지 않는다는 점을 보여준다. Model capability acquisition과 execution schedule을 분리한 것이 JustGRPO의 가장 실용적인 부분이다.

# 4. Training / Data / Recipe

## 4-1. Data

JustGRPO의 base model은 LLaDA 8B Instruct다. 별도의 task-specific SFT를 먼저 수행하지 않고, 바로 full-parameter RL을 적용한다.

Training domain은 mathematics와 coding으로 나뉜다.

- Mathematics
  - GSM8K와 MATH 계열의 official training split을 사용한다.
  - Reward는 generated answer의 exact match 또는 수학적 equivalence를 검증하는 binary reward다.

- Coding
  - AceCoder-87K에서 unit test를 실행할 수 있고 난도가 높은 21K sample을 선택한다.
  - Reward는 unit-test pass rate와 output format reward를 결합한다.
  - Valid code block과 syntax를 만족하면 format reward 1, code block은 있으나 syntax error가 있으면 0.5, code block이 없으면 0을 부여한다.

논문은 reasoning task별로 별도 SFT recipe를 만들기보다, dLLM base policy가 이미 가진 solution support를 RL로 elicitation하는 setting에 초점을 둔다.

## 4-2. Training strategy

주요 training configuration은 다음과 같다.

| Hyperparameter | Value |
| --- | ---: |
| Base model | LLaDA 8B Instruct |
| Update | Full-parameter RL |
| GPUs | 16 x H100 |
| Optimizer | AdamW |
| Learning rate | 5e-6 |
| Schedule | Constant |
| Weight decay | 0 |
| Betas | 0.9, 0.999 |
| Global batch size | 64 |
| Group size | 16 |
| Policy update steps | 1 |
| Maximum completion length | 256 |
| Rollout temperature | 1.0 |
| KL coefficient | 0 |
| Maximum training steps | 125 |

Mathematics는 비교적 빠르게 개선되어 GSM8K가 약 50 step 부근에서 높은 수준에 도달한다. Coding은 unit-test feedback이 더 sparse하고 task가 다양해 125 step까지 추가 이득이 이어진다.

KL coefficient를 0으로 둔 설정은 결과 해석에서 중요하다. 성능 향상이 strong reference regularization에서 나온 것이 아니라, AR rollout과 verifiable reward optimization만으로 나타났다는 의미다. 다만 longer training이나 distribution shift가 큰 setting에서는 KL-free recipe의 stability를 별도로 검증해야 한다.

## 4-3. Engineering notes

### 1) Exact likelihood의 계산 비용

JustGRPO는 sequence likelihood를 exact하게 계산하지만, 각 position의 AR conditional을 얻으려면 future mask가 다른 input을 평가해야 한다. Autoregressive transformer처럼 한 번의 causal forward에서 모든 token likelihood를 얻는 구조가 아니므로, dLLM에서는 per-position evaluation overhead가 생긴다.

### 2) JustGRPO-Fast

저자들은 이 비용을 줄이기 위해 highest-entropy position의 일부만 policy-ratio 계산에 사용한다. Top 25% position을 선택하면 약 75%의 forward evaluation을 제거할 수 있다.

이 선택은 모든 token이 RL update에 동일하게 중요하지 않다는 entropy 관찰과 연결된다. Logical fork처럼 high-entropy token을 우선 업데이트하면 wall-clock efficiency를 개선하면서 주요 learning signal을 보존할 수 있다.

### 3) Random order는 해결책이 아니다

Confidence-based order가 문제라면 random order를 쓰면 될 것처럼 보일 수 있다. 하지만 논문에서 random order는 Pass@128이 AR보다 낮고, Pass@1도 크게 하락한다. JustGRPO-Random의 GSM8K 결과도 standard JustGRPO보다 낮다.

핵심은 confidence heuristic만 제거하는 것이 아니라, exploration을 안정적인 causal scaffold에 연결하는 것이다.

### 4) Training과 serving benchmark를 분리해야 한다

JustGRPO는 training simplicity와 inference compatibility를 보여주지만, 실제 serving throughput은 batch size, cache strategy, sampler, hardware utilization에 따라 달라진다. Paper의 parallel decoding 결과를 곧바로 production latency 수치로 해석해서는 안 된다.

# 5. Evaluation

## 5-1. Main results

### 1) Reasoning boundary 분석

세 dLLM과 네 benchmark에서 arbitrary order는 작은 $k$에서 경쟁력 있는 경우가 있지만, 큰 $k$로 갈수록 AR order가 더 가파르게 개선된다.

LLaDA-Instruct의 Pass@1024 coverage를 보면 arbitrary order가 해결하는 problem은 대부분 AR order가 해결하는 set 안에 포함된다.

| Benchmark | AR-only solved | Arbitrary-order-only solved |
| --- | ---: | ---: |
| HumanEval | 21.3% | 0.6% |
| MBPP | 14.0% | 0.8% |
| MATH-500 | 4.8% | 0.6% |
| GSM8K | 1.2% | 0.0% |

이 결과는 arbitrary order가 완전히 다른 solution family를 탐색한다기보다, practical sampler에서 AR이 찾는 solution의 일부만 더 일관되게 선택하는 경향을 시사한다.

### 2) JustGRPO main performance

논문이 통일한 evaluation protocol에서 주요 결과는 다음과 같다.

| Method | GSM8K | MATH-500 | HumanEval | MBPP |
| --- | ---: | ---: | ---: | ---: |
| d1 | 83.8 | 39.2 | - | - |
| ESPO | 84.7 | 40.3 | 42.1 | 44.6 |
| SPG | 86.9 | 41.8 | - | - |
| JustGRPO | 89.1 | 45.1 | 49.4 | 52.4 |

JustGRPO는 diffusion-specific RL objective를 추가하지 않고도 네 task에서 강한 결과를 보인다. 특히 code benchmark에서 improvement가 크다는 점은 exact AR policy와 unit-test reward의 조합이 효과적임을 보여준다.

다만 논문의 전체 비교표에는 generation length, rollout protocol, base checkpoint가 다른 선행 연구 결과도 포함된다. Method 간 공정한 비교는 저자들이 matched configuration으로 재현한 unified table을 우선해서 봐야 한다.

### 3) Parallel decoding 결과

JustGRPO model에 EB Sampler를 적용하면 한 step에서 여러 token을 복원할 수 있다. MBPP에서 base model 대비 gain은 1 token per step일 때 10.6 percentage points이며, 약 5 tokens per step 부근에서는 25.5 percentage points까지 커진다고 보고한다.

이 결과는 RL에서 AR order를 사용한 model이 parallel inference에서 capability를 잃지 않았음을 보여준다. 다만 이는 sampler compatibility와 task accuracy에 대한 결과이며, end-to-end throughput이나 memory efficiency 전체를 측정한 production benchmark는 아니다.

### 4) General capability preservation

RL 이후 일반 benchmark 성능은 대체로 유지된다.

| Benchmark | Base | JustGRPO |
| --- | ---: | ---: |
| MMLU | 65.5 | 65.8 |
| MMLU-Pro | 37.0 | 36.7 |
| HellaSwag | 74.6 | 74.8 |
| ARC-C | 88.5 | 87.5 |

Reasoning RL이 broad capability를 크게 훼손하지 않았다는 신호지만, 네 benchmark만으로 instruction following, safety, multilingual ability까지 보존되었다고 결론 내리기는 어렵다.

## 5-2. What really matters in the experiments

이 논문의 실험에서 가장 중요한 것은 최고 accuracy 하나가 아니다.

### 1) Pass@1과 Pass@k의 방향이 다를 수 있다

Arbitrary order는 easy token을 먼저 확정해 single sample의 coherence를 높일 수 있다. 그러나 같은 mechanism이 high-entropy branch를 제거해 large-k coverage를 줄일 수 있다. RL potential을 보려면 Pass@1만으로는 부족하다.

### 2) Solution set overlap이 mechanism claim을 강화한다

Arbitrary order가 AR과 다른 정답 family를 찾았다면 낮은 Pass@k도 complementary exploration으로 해석할 수 있다. 하지만 AO-only solved case가 매우 적고 AR-only case가 많다. 이 coverage asymmetry가 flexibility trap 주장에 더 직접적인 근거를 준다.

### 3) Block size sweep이 binary comparison을 넘는다

AR 대 arbitrary order 두 setting만 비교하면 implementation artifact일 가능성이 남는다. $B$가 커질수록 Pass@k가 monotonic하게 떨어지는 결과는 order freedom과 exploration boundary 사이의 연속적인 관계를 보여준다.

### 4) Entropy는 평균이 아니라 위치별 역할로 봐야 한다

Global mean entropy가 비슷해도 logical fork entropy가 줄어들 수 있다. RL에서 token importance를 분석할 때 frequency나 average confidence보다, decision-critical position을 찾아야 한다는 메시지가 있다.

### 5) Simplicity는 ablation 대상이어야 한다

JustGRPO의 장점은 method component가 적다는 데 있다. 따라서 성능뿐 아니라 exact policy definition, random-order baseline, fast approximation, parallel inference, general capability preservation이 함께 제시되어야 단순화 주장이 설득력을 얻는다.

# 6. Limitations

1. General reasoning과 constraint satisfaction을 구분해야 한다
   - 논문은 mathematics와 coding을 general reasoning의 대표로 사용한다.
   - Sudoku나 zebra puzzle처럼 global constraint를 먼저 채우는 것이 유리한 task에서는 arbitrary order가 여전히 강할 수 있다.
   - 결론은 "arbitrary order는 항상 나쁘다"가 아니라, 현재 confidence-based decoding이 general reasoning에서 exploration을 줄일 수 있다는 것이다.

2. Pass@k는 reasoning potential의 proxy다
   - 큰 $k$에서 정답을 sampling할 수 있다는 것은 RL이 활용할 positive trajectory가 존재한다는 뜻이다.
   - 하지만 실제 RL이 그 trajectory를 안정적으로 강화할 수 있는지, training이 새로운 support를 만들 수 없는지는 algorithm과 optimization regime에 따라 달라질 수 있다.

3. Mechanism이 sampler에 의존할 수 있다
   - Entropy degradation은 confidence-based 또는 semi-autoregressive decoding과 밀접하다.
   - 다른 order policy, learned scheduler, uncertainty-preserving sampler에서는 결과가 달라질 수 있다.
   - Appendix의 random-order와 additional sampler 분석이 범위를 넓히지만, arbitrary-order sampler 전체를 포괄하지는 않는다.

4. RL generalization은 주로 하나의 base family에서 검증된다
   - Reasoning-boundary 분석은 세 dLLM을 포함하지만, JustGRPO training은 LLaDA 8B Instruct를 중심으로 한다.
   - 다른 model scale, tokenizer, diffusion objective, multimodal dLLM에서도 같은 recipe가 유지되는지 확인이 필요하다.

5. Exact AR policy가 training compute를 공짜로 만들지는 않는다
   - Per-position masked forward는 standard causal LM보다 비쌀 수 있다.
   - JustGRPO-Fast가 비용을 줄이지만, high-entropy token selection이 task와 training stage에 따라 안정적인지 추가 검증이 필요하다.

6. Baseline table의 조건이 모두 같지 않다
   - 선행 연구의 quoted result는 generation length, checkpoint, prompt, sampling configuration이 다를 수 있다.
   - 결론은 matched protocol table과 ablation을 중심으로 읽어야 한다.

7. Parallel decoding 결과는 full serving study가 아니다
   - EB Sampler와 maximum length 256 setting에서 compatibility를 보여준다.
   - 실제 latency, throughput, batching, KV cache, long-context behavior는 별도 시스템 평가가 필요하다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문의 가장 중요한 기여는 "diffusion model의 장점"을 하나의 묶음으로 보지 않고, capability를 만드는 flexibility와 execution을 빠르게 만드는 flexibility를 분리한 데 있다.

최근 model architecture 연구에서는 training objective, rollout policy, inference sampler가 같은 구조를 공유해야 자연스럽다고 가정하기 쉽다. 하지만 deployment 목적이 다르면 각 단계의 최적 schedule도 다를 수 있다.

- Training은 diverse positive trajectory를 찾아야 한다.
- Optimization은 stable하고 exact한 likelihood를 선호한다.
- Inference는 latency와 parallelism을 우선할 수 있다.

JustGRPO는 이 세 요구를 한 order policy로 해결하려 하지 않는다. 이런 decoupling은 diffusion LLM뿐 아니라 speculative decoding, latent reasoning, non-causal encoder-decoder, agent rollout design에도 적용 가능한 관점이다.

또 하나 중요한 점은 high-entropy token을 단순한 low-confidence error로 취급하지 않는다는 것이다. 일부 uncertainty는 제거해야 할 noise가 아니라 exploration을 유지하는 자원이다. RLVR에서 entropy regularization이나 token weighting을 설계할 때, 어떤 token의 entropy인지가 전체 entropy 크기보다 중요할 수 있다.

## 7-2. Reuse potential

### 1) Base model evaluation에서 Pass@k curve를 먼저 본다

RL 실험 전에 base checkpoint의 Pass@1, Pass@8, Pass@32, Pass@128 curve를 비교한다. 평균 accuracy만 보지 말고, sampling budget이 늘어날 때 solution coverage가 계속 확장되는지 확인한다.

### 2) Rollout order와 deployment order를 별도 hyperparameter로 둔다

Training rollout은 exploration-friendly order를 사용하고, inference는 latency-friendly sampler를 사용한다. 두 단계가 반드시 같은 generation schedule을 쓸 필요는 없다.

### 3) Critical-token entropy를 추적한다

전체 token entropy 대신 reasoning connector, operator, API choice, branch decision 같은 task-critical token set을 정의한다. RL 전후와 sampler별로 해당 위치의 entropy가 어떻게 바뀌는지 본다.

### 4) Resolved-set overlap을 분석한다

두 sampler의 Pass@k가 다를 때, 어느 problem을 각 sampler만 해결하는지 Venn-style coverage를 본다. 한 sampler가 complementary solution을 제공하는지, 다른 sampler의 strict subset인지 구분할 수 있다.

### 5) Exact objective와 fast approximation을 함께 설계한다

먼저 exact but expensive version으로 learning signal을 검증한 뒤, high-entropy position subsampling처럼 구조적 근거가 있는 approximation을 추가한다. 처음부터 approximate objective만 쓰면 failure 원인을 찾기 어렵다.

## 7-3. Follow-up papers

- Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions
- d1: Scaling Reasoning in Diffusion Large Language Models via Reinforcement Learning
- SPG: Sandwiched Policy Gradient for Masked Diffusion Language Models
- Principled RL for Diffusion LLMs Emerges from a Sequence-Level Perspective
- Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning
- Fast-dLLM: Training-Free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding

# 8. Summary

- Diffusion LLM의 parallel decoding과 arbitrary-order generation은 서로 다른 장점이며, 반드시 함께 보존해야 하는 것은 아니다.
- Confidence-based arbitrary order는 high-entropy logical fork를 미루고 future context를 먼저 고정해 entropy degradation을 만들 수 있다.
- Arbitrary order는 Pass@1에서는 경쟁력 있을 수 있지만, 큰 $k$의 Pass@k와 solution coverage에서는 AR order보다 좁은 reasoning boundary를 보인다.
- JustGRPO는 RL 동안 dLLM을 exact autoregressive policy로 정의해 standard GRPO를 적용하고, architecture와 bidirectional attention은 그대로 둔다.
- 학습 후에도 parallel decoding을 사용할 수 있으므로, sequential exploration과 efficient inference를 분리해 설계할 수 있다.
