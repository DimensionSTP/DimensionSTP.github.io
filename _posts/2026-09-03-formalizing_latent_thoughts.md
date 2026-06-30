---
layout: single
title: "Formalizing Latent Thoughts: Four Axioms of Thought Representation in LLMs Review"
categories: Study-concept
tag: [LatentThoughts, Reasoning, RepresentationLearning, LLM, Interpretability]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2606.27378)

[Project page](https://fard-lab.github.io/formalize-thoughts)

Formalizing Latent Thoughts는 latent reasoning 논문을 읽을 때 자주 느끼는 불편함을 정면으로 다룬다. 어떤 방법이 explicit CoT token을 continuous latent vector로 대체했다고 주장할 때, 우리는 보통 downstream accuracy만 본다. 하지만 accuracy가 올랐다고 해서 그 latent vector가 실제로 "thought representation"으로 작동한다는 뜻은 아니다.

이 논문은 질문을 바꾼다.

> Thought representation이라면 downstream answer accuracy와 독립적으로 어떤 functional property를 만족해야 하는가?

저자들은 네 가지 axiom을 제안한다.

1. Causality
2. Minimality
3. Separability
4. Stability

그리고 각 axiom을 source LLM 위에서 직접 측정하는 metric을 만든다. 중요한 점은 downstream benchmark score를 보지 않는다는 것이다. Representation 자체가 output distribution에 causal하게 기여하는지, input에서 불필요한 정보를 줄이는지, semantic difference를 분리하는지, output distribution uncertainty를 안정적으로 담는지를 본다.

> 한 줄 요약: Formalizing Latent Thoughts는 latent thought representation을 downstream accuracy가 아니라 Causality, Minimality, Separability, Stability 네 functional axiom으로 평가하는 intrinsic protocol을 제안하고, 5개 open-weight LLM과 BBEH 23 tasks에서 LIT, Soft Thinking, Latent Thinking 후보가 네 axiom을 동시에 만족하지 못함을 보인 audit paper다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Latent reasoning representation을 benchmark score가 아니라 representation object로 평가한다.
- "성능이 올랐으니 thought다"라는 inference를 피한다.
- Candidate representation이 prompt embedding보다 추가 정보를 담는지 직접 비교한다.
- Same-task separability collapse처럼 downstream accuracy가 숨기는 failure를 보여준다.
- Soft Thinking, Latent Thinking, hidden-state extraction을 같은 protocol에서 비교한다.
- Future latent thought methods가 최적화할 수 있는 decomposable targets를 제시한다.

이 글에서는 이 논문을 "latent CoT가 실패했다"가 아니라, **latent thought representation을 무엇으로 정의하고 어떻게 audit할지 정리한 evaluation framework**로 읽는다.

# 1. Problem Setting

## 1-1. Problem definition

Recent reasoning research는 continuous latent thoughts를 점점 더 많이 탐색한다. 긴 explicit chain-of-thought token을 쓰는 대신, model이 hidden state, soft token, pause token, recursive latent vector를 intermediate reasoning state로 사용할 수 있다.

하지만 무엇이 vector를 "thought representation"으로 만드는가?

Downstream task accuracy만으로는 충분하지 않다. Latent token을 쓴 model이 잘한다면, improvement는 다음 중 무엇 때문일 수 있다.

- Extra compute
- Better prompt format
- More trainable parameters
- Decoder robustness
- Representation quality
- Benchmark artifact

논문은 intrinsic evaluation을 원한다.

Input을 $x$, model output distribution을 $P(Y \mid x)$, candidate thought representation을 $T=g(x)$라고 하자. Valid thought representation은 input과 semantic output을 매개하는 sufficient functional state를 포착해야 한다.

논문은 $T$가 human interpretable할 것을 요구하지 않는다. $T$가 model output distribution을 functional하게 support하면 된다.

## 1-2. Why previous approaches are insufficient

### 1) Accuracy conflates representation and decoder

Downstream score는 latent thought가 좋은지, 아니면 뒤따르는 decoder가 약한 representation을 보상하는지 구분하지 못한다.

### 2) CoT imitation is not definition

Latent vector가 explicit CoT behavior를 모방한다고 해서 올바른 semantic state를 담는다는 뜻은 아니다. Textual trace artifact를 encode하거나 서로 다른 reasoning path를 collapse할 수 있다.

### 3) Probing alone is underspecified

Probe는 hidden state 안에 어떤 정보가 존재한다는 점을 보여줄 수 있다. 하지만 그 representation이 thought representation으로서 causal, minimal, separable, stable한지는 말해주지 않는다.

### 4) Existing methods lack common criteria

Soft Thinking, Latent Thinking, Last Input Token hidden state, output embedding은 보통 서로 다른 setup에서 평가된다. Axiomatic criteria는 직접 비교를 가능하게 한다.

# 2. Core Idea

## 2-1. Main contribution

논문의 contribution은 다음과 같다.

1. **Functional definition of thought representation**
   - Thought는 input을 semantic output distribution으로 매개하는 latent state다.
   - Human interpretability가 아니라 function으로 정의한다.

2. **Four axioms**
   - Causality
   - Minimality
   - Separability
   - Stability

3. **Intrinsic measurement protocol**
   - Causality를 위한 KL substitution error
   - IB-inspired surrogate를 사용한 Minimality Gap
   - Separability를 위한 same-task와 cross-task discriminator accuracy
   - Stability를 위한 Distributional Consistency Score

4. **Empirical audit**
   - Five open-weight LLMs
   - BBEH 23 reasoning tasks
   - Candidate families: Last Input Token, Soft Thinking, Soft Thinking with Gumbel noise, Latent Thinking, input embedding, output embedding

## 2-2. Design intuition

네 axiom은 서로 다른 질문에 답한다.

| Axiom | Question |
| --- | --- |
| Causality | Representation이 model computation에서 reasoning prefix를 대체할 수 있는가? |
| Minimality | Output-relevant information은 유지하고 irrelevant input detail은 버리는가? |
| Separability | 특히 같은 task 안에서 semantically different output을 구분하는가? |
| Stability | Semantically equivalent output 전반의 stable distributional uncertainty를 반영하는가? |

한두 축만 만족하는 candidate로는 충분하지 않다. 예를 들면 다음이 가능하다.

- Representation이 causal하지만 minimal하지 않을 수 있다.
- Task type은 encode하지만 question identity는 구분하지 못할 수 있다.
- Stable하지만 prompt embedding 이상의 추가 정보를 담지 않을 수 있다.
- Benchmark accuracy는 올리지만 intrinsic separability는 실패할 수 있다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Intrinsic evaluation of latent thought representations |
| Axioms | Causality, Minimality, Separability, Stability |
| Tasks | 23 BBEH reasoning tasks |
| Source LLMs | 5 open-weight models |
| Candidates | LIT, ST, STN, LT, IE, OE, RV |
| Main finding | No candidate satisfies all four axioms |
| Key failure | Same-task per-question separability collapse |
| Reference | Input Embedding is competitive on all axes |

## 3-2. Axiom 1: Causality

Representation은 model computation에서 explicit reasoning prefix를 대체할 수 있어야 한다. $y_{\mathrm{pre}}$가 reasoning prefix이고 $y_{\mathrm{suf}}$가 answer suffix라면, $y_{\mathrm{pre}}$를 projected $T$로 대체해도 continuation distribution이 보존되어야 한다.

논문은 KL divergence를 측정한다.

$$
\mathrm{CausalityError}
=
D_{\mathrm{KL}}
\left(
P(y_{\mathrm{suf}} \mid y_{\mathrm{pre}})
\|
P(y_{\mathrm{suf}} \mid T)
\right)
$$

낮을수록 좋다.

Audit 결과 candidates는 random vector보다 좋지만 input embedding reference를 일관되게 넘지는 못한다. 즉 continuation-relevant information은 담지만, prompt가 이미 제공하는 정보보다 명확히 더 많다고 보기 어렵다.

## 3-3. Axiom 2: Minimality

Thought representation은 output과 관련된 것을 유지하면서 input을 압축해야 한다. 논문은 information bottleneck principle로 이를 동기화한다.

$$
\min
I(X;T)
-
\beta I(T;Y)
$$

Mutual information term은 직접 계산하기 어렵기 때문에 논문은 Minimality Gap 또는 IB residual gap이라는 cross-entropy surrogate를 사용한다.

결과는 mixed다. 일부 Soft Thinking candidate는 input embedding 근처 또는 그 이상이지만, LIT는 input embedding보다 낮은 경우가 많다. 어떤 candidate도 source LLM 전반에서 prompt embedding reference보다 더 좋은 output-relevant compression을 일관되게 제공하지 못한다.

## 3-4. Axiom 3: Separability

Thought representation은 semantically different output distribution을 구분해야 한다. 논문은 frozen Llama-3.2-1B backbone과 trainable projection/classification head를 사용해 bounded discriminator를 학습한다.

두 mode가 중요하다.

| Mode | Meaning |
| --- | --- |
| Cross-task | 서로 다른 task의 output을 구분 |
| Same-task | 같은 task 안에서 서로 다른 question을 구분 |

Cross-task accuracy는 input embedding을 포함해 거의 모든 candidate에서 높다. Same-task accuracy는 output embedding을 제외하면 모든 candidate가 random baseline에 가깝다.

이것이 논문의 가장 중요한 empirical result다. Candidate latent thoughts는 coarse task type은 알지만, 같은 task 안의 individual question을 구분하지 못한다.

## 3-5. Axiom 4: Stability

Stability는 representation이 distributional consistency를 추적하는지 묻는다. Beam output이 하나의 semantic equivalence class에 속하는지, 아니면 여러 class로 갈라지는지를 representation이 반영할 수 있는가를 본다.

Metric은 Distributional Consistency Score, DCS다. Beam output이 하나 이상의 semantic equivalence class로 퍼지는지를 예측하는 AUROC로 구현된다.

논문은 beam cluster가 informative한 5개 LLM 중 4개에서 candidate들이 random vector baseline보다 좋다고 보고한다. 하지만 input embedding이 iterative thinking candidate와 비슷하거나 더 좋은 경우가 많다.

## 3-6. Candidate representations

Audit 대상은 다음과 같다.

| Candidate | Description |
| --- | --- |
| LIT | Last Input Token hidden states |
| ST | Soft Thinking without noise |
| STN | Soft Thinking with Gumbel noise |
| LT | Latent Thinking |
| IE | Input prompt embedding |
| OE | Output embedding upper anchor |
| RV | Random vector sanity check |

Soft thinking과 latent thinking candidate는 thinking steps 1, 16, 32, 64, 128에서 평가된다.

## 3-7. Source models

논문은 다섯 open-weight LLM을 다룬다.

| Source LLM | Family |
| --- | --- |
| Llama-3.1-8B-Instruct | Dense |
| Llama-3.3-70B-Instruct | Dense |
| DeepSeek-R1-Distill-Qwen-32B | Reasoning-distilled |
| Skywork-OR1-32B | Native RL |
| GPT-OSS-20B | Sparse MoE |

이 range가 중요한 이유는 failure가 model size나 training method에만 국한되지 않는 structural issue라고 논문이 주장하기 때문이다.

# 4. Training / Data / Recipe

## 4-1. Data

Evaluation은 Big Bench Extra Hard, BBEH의 23 tasks를 사용한다. 논문은 original benchmark prompts를 사용한다.

각 prompt에 대해 beam search는 최대 8192 token의 8개 sequence를 반환한다. 이 candidate output들이 semantic equivalence와 representation behavior를 평가하는 empirical high-probability output region을 구성한다.

## 4-2. Probe setup

논문은 frozen Llama-3.2-1B를 shared decoding surface로 사용한다. Trainable projection은 candidate thought representation을 이 model의 token embedding space로 mapping하고, discriminator가 필요한 경우 classification head를 추가한다.

이렇게 하면 metric cost가 source LLM size와 독립적이다. Source LLM을 retrain할 필요가 없다.

Output 간 semantic similarity는 EmbedNemotron-8B를 사용해 semantic equivalence class를 근사한다.

## 4-3. Engineering notes

1. **Do not use benchmark accuracy as representation metric**
   - Model이 task를 풀 수 있어도 candidate thought representation은 intrinsic criteria에서 실패할 수 있다.

2. **Always compare to input embedding**
   - Thought candidate가 prompt embedding을 넘지 못하면 useful state를 추가하지 않을 수 있다.

3. **Cross-task와 same-task separability를 분리한다**
   - Cross-task success는 within-task collapse를 숨길 수 있다.

4. **Treat output embedding as upper anchor, not usable thought**
   - Output embedding은 direct output information을 포함한다.

5. **Measurement cost가 있다**
   - Generation과 small probe training이 필요하다.
   - Cheap benchmark replacement가 아니라 audit protocol이다.

# 5. Evaluation

## 5-1. Causality

논문은 모든 candidate representation이 random vector baseline보다 낮은 KL을 보인다고 보고한다. 따라서 continuation-relevant information을 어느 정도 encode한다.

하지만 어떤 candidate도 input embedding reference를 일관되게 넘지 못한다. 이는 current latent thought candidate가 prompt beyond causal information을 추가한다는 주장을 약하게 만든다.

## 5-2. Minimality

Minimality result는 mixed다. Output embedding은 output을 직접 encode하므로 valid minimality reference가 아니다. 다른 candidate 중에서는 Soft Thinking이 input embedding보다 높게 나오는 경우가 있지만, LIT는 종종 낮다. Source LLM 전반에서 지배적인 candidate는 없다.

## 5-3. Separability collapse

Same-task separability가 가장 명확한 failure다.

논문은 cross-task accuracy가 input embedding을 포함해 거의 모든 candidate에서 saturation에 가깝다고 보고한다. 하지만 output embedding을 제외한 모든 candidate의 same-task accuracy는 random baseline에 가깝다.

보고된 same-task value는 많은 candidate에서 약 50%-55% 수준이며, output embedding은 model에 따라 62%-73%까지 간다.

이는 representation이 task identity는 구분하지만 같은 task 안의 question identity는 구분하지 못한다는 뜻이다.

## 5-4. Stability

DCS result는 candidate가 distributional uncertainty를 일부 encode할 수 있음을 보여준다. 하지만 input embedding이 iterative candidate와 비슷하거나 더 나은 경우가 많다. Thinking step을 늘리면 특히 Latent Thinking 계열에서 DCS가 나빠질 수 있다.

## 5-5. Joint result

논문의 synthesis는 강하다.

- 어떤 candidate도 모든 axis에서 input embedding을 넘지 못한다.
- 어떤 candidate도 네 axiom을 모두 만족하지 못한다.
- Iterative thinking variant는 step count가 늘수록 degrade된다.
- Downstream accuracy는 representational quality를 예측하지 못한다.
- Failure pattern은 dense, MoE, reasoning-distilled, RL-trained model 전반에서 나타난다.

## 5-6. What really matters in the experiments

### 1) Input embedding is a hard baseline

Latent thought representation이 prompt embedding을 넘지 못한다면, computation은 추가했지만 representation은 추가하지 않았을 수 있다.

### 2) Same-task separability is the key stress test

Task identity는 쉽다. 같은 task 안의 두 question을 구분하는 곳에서 current candidate들이 실패한다.

### 3) Output embedding shows what is possible

Output embedding은 usable thought representation은 아니지만, representation이 실제로 per-question semantic difference를 담고 있으면 measurement가 이를 감지할 수 있음을 보여준다.

### 4) More thinking step은 해로울 수 있다

Soft thinking이나 latent thinking step을 늘린다고 representation이 자동으로 좋아지지 않는다. 일부 metric은 step이 늘수록 나빠진다.

# 6. Limitations

1. **English reasoning task만 다룬다**
   - Audit은 BBEH 23 tasks와 English-language open-weight LLM을 사용한다.
   - Multilingual reasoning과 domain-specific reasoning은 아직 테스트되지 않았다.

2. **Candidate set이 제한적이다**
   - LIT, Soft Thinking, Soft Thinking with noise, Latent Thinking, IE, OE, RV를 다룬다.
   - Future trained representation은 더 나을 수 있다.

3. **Metric implementation choice가 중요하다**
   - 각 axiom은 alternative quantification이 가능하다.
   - Current metrics는 하나의 realization이다.

4. **Probe training cost가 추가된다**
   - 이 protocol은 benchmark accuracy보다 더 비싸다.

5. **Semantic equivalence는 approximation이다**
   - Framework는 embedding과 beam output으로 semantic space를 근사한다.
   - 이 과정에서 error가 들어갈 수 있다.

6. **Output embedding은 deployable하지 않다**
   - Upper anchor 역할을 하지만 output information을 사용한다.

7. **Causality substitution은 approximate하다**
   - Representation을 token embedding position에 project하는 방식이 true computational substitution을 완벽히 반영하지는 않을 수 있다.

8. **Downstream accuracy를 대체하지 않는다**
   - Intrinsic representation audit은 benchmark evaluation을 보완하지 대체하지 않는다.

9. **Optimization method를 제안하지 않는다**
   - 논문은 evaluation target을 제시하지, 새로운 latent thought training algorithm을 제안하지 않는다.

10. **Thought가 distributed되어 있을 수 있다**
    - 단일 vector나 vector set extraction이 distributed model computation을 포착하지 못할 수 있다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문의 가장 중요한 메시지는 "latent thinking does not work"가 아니다. 더 중요한 점은 **latent thought를 성능 향상 장치가 아니라 representation object로 평가해야 한다는 것**이다.

Reasoning research는 benchmark improvement를 쉽게 신뢰하는 경향이 있다. 하지만 latent vector가 같은 task 안의 두 question을 구분하지 못한다면, 그것을 thought라고 부르기는 어렵다. 그것은 functional thought state가 아니라 compute, regularization, prompt perturbation일 수 있다.

## 7-2. Reuse potential

### Latent reasoning research

새 latent thought method는 최소한 다음을 보고해야 한다.

- Causality substitution error
- Minimality 또는 compression score
- Same-task separability
- Stability under output distribution variation
- Comparison to input embedding

### Process supervision

Model이 internally reason하도록 학습한다면, final answer 전의 internal state를 평가하는 방법이 필요하다. 이 axiom들은 testable target을 제공한다.

### Interpretability

이 framework는 functional validity와 human interpretability를 명확히 분리한다. Internal thought는 opaque하지만 functionally valid할 수 있기 때문에 유용한 구분이다.

### Agent memory와 planning

Agent의 latent plan representation도 비슷하게 평가할 수 있다. Causal influence, minimality, task/instance separability, equivalent plan under stability를 볼 수 있다.

## 7-3. Production considerations

- Benchmark score가 오른다고 hidden vector를 meaningful thought로 가정하지 않는다.
- Input embedding을 minimum baseline으로 사용한다.
- Task classification만 보지 말고 same-task separability를 본다.
- Latent vector에서 interpretability를 과하게 주장하지 않는다.
- Internal planning을 사용하는 model validation에는 latent representation audit을 포함한다.

## 7-4. Follow-up papers

- COCONUT
- CODI
- Soft Thinking
- Latent Thinking
- PonderLM
- Pause token reasoning
- Chain-of-thought faithfulness papers
- Probing internal representations before generation
- Information bottleneck과 representation learning papers

# 8. Summary

- 이 논문은 functional thought representation을 위한 네 axiom을 정의한다.
- 네 축은 Causality, Minimality, Separability, Stability다.
- Downstream benchmark accuracy와 독립적으로 candidate representation을 평가한다.
- Current candidate들은 네 axiom을 모두 만족하지 못하고, input embedding을 넘지 못하는 경우도 많다.
- Same-task separability collapse가 가장 중요한 empirical warning이다.
