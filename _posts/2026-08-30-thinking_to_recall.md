---
layout: single
title: "Thinking to Recall: How Reasoning Unlocks Parametric Knowledge in LLMs Review"
categories: Study-concept
tag: [ThinkingToRecall, Reasoning, ParametricKnowledge, Factuality, LLM]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2603.09906)

Thinking to Recall은 reasoning을 "복잡한 문제를 단계별로 푸는 능력"으로만 보는 관점을 흔드는 논문이다. 이 논문이 다루는 질문은 단순하다. **single-hop factual question처럼 reasoning이 필요 없어 보이는 문제에서도, 왜 thinking mode가 parametric knowledge recall을 개선하는가**다.

일반적으로 reasoning은 수학, 코드, multi-hop QA처럼 subproblem decomposition이 필요한 task에서 유용하다고 생각한다. 하지만 "어떤 인물이 언제 태어났는가"나 "어떤 작품의 저자는 누구인가" 같은 single-hop factual question은 logical derivation이 필요 없어 보인다. 답을 알고 있으면 바로 말하면 되고, 모르면 모르는 것이다.

그런데 이 논문은 reasoning ON mode가 pass@k를 크게 끌어올린다는 것을 보인다. 특히 pass@1보다 larger k에서 gap이 더 벌어지는 경우가 많다. 이는 reasoning이 단순히 top-1 answer를 조금 더 잘 calibrate하는 것이 아니라, model output distribution 안에서 원래 거의 unreachable하던 correct answer path를 더 많이 열어준다는 해석으로 이어진다.

논문은 두 가지 mechanism을 제시한다.

1. Computational buffer effect
   - reasoning token을 생성하는 과정 자체가 추가 latent computation을 제공한다.
   - semantic content가 없는 dummy trace도 일정 길이까지 성능을 올릴 수 있다.

2. Factual priming
   - reasoning trace가 related facts를 생성하면, 그 fact가 target answer recall을 돕는 semantic bridge가 된다.
   - 다만 intermediate fact가 hallucination이면 final answer hallucination 위험도 커진다.

> 한 줄 요약: Thinking to Recall은 reasoning이 single-hop factual QA에서도 parametric recall boundary를 확장하며, 그 이유가 단순 decomposition이 아니라 reasoning token이 제공하는 computational buffer와 related fact generation이 만드는 factual priming에 있음을 controlled experiments로 분석한 논문이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Reasoning의 이득을 task decomposition이 아니라 parametric knowledge access 문제로 재해석한다.
- pass@k를 통해 "현재 top-1 accuracy"보다 "model 안에 reachable한 correct path가 있는가"를 본다.
- Dummy reasoning trace로 compute-only effect를 분리한다.
- Extracted fact list and dummy fact control로 factual priming을 검증한다.
- Reasoning trace hallucination이 final answer hallucination과 연결된다는 risk를 정량화한다.
- Reasoning trajectory selection and process reward 설계에 직접적인 implication을 준다.

이 글에서는 Thinking to Recall을 "thinking을 켜면 성능이 오른다"가 아니라, **LLM reasoning token이 latent computation buffer와 self-retrieval prompt로 어떻게 작동하는지 분석한 paper**로 읽는다.

# 1. Problem Setting

## 1-1. Problem definition

Closed-book factual QA에서 model은 external search 없이 parametric knowledge만으로 답해야 한다. Query $q$가 주어졌을 때 model은 answer $a$를 생성한다.

Reasoning OFF mode는 바로 답을 생성한다.

$$
a \sim p_{\theta}(a \mid q, \mathrm{OFF})
$$

Reasoning ON mode는 먼저 reasoning trace $z$를 생성하고, 그 뒤 final answer를 생성한다.

$$
z \sim p_{\theta}(z \mid q, \mathrm{ON})
$$

$$
a \sim p_{\theta}(a \mid q,z,\mathrm{ON})
$$

핵심 질문은 다음이다.

> $q$ 자체가 single-hop factual question이라면, $z$는 왜 도움이 되는가?

논문은 이 질문을 top-1 accuracy보다 pass@k로 본다. pass@k는 $k$개의 sample 중 적어도 하나가 정답일 확률을 추정한다.

$$
\mathrm{pass@}k
=
\Pr
\left[
\exists i \leq k : a_i \in A^*
\right]
$$

Pass@k는 model distribution 안에 correct answer path가 존재하는지 보는 데 유용하다. Reasoning ON이 pass@k를 크게 높인다면, reasoning은 correct answer를 더 reachable하게 만들고 있다고 볼 수 있다.

## 1-2. Why previous explanations are insufficient

### 1) Task decomposition explanation

Reasoning은 복잡한 task를 subproblem으로 나누기 때문에 유용하다는 설명이 있다. 하지만 이 논문의 target은 mostly single-hop factual QA다. SimpleQA-Verified에는 metadata가 있고, 논문은 1000개 중 903개가 single-hop이라고 보고한다. EntityQuestions는 template-based single-hop question이다.

따라서 decomposition만으로 reasoning benefit을 설명하기 어렵다.

### 2) Sampling efficiency explanation

Reasoning이 단지 correct answer의 probability를 조금 올려 top-1이 개선되는 것일 수도 있다. 하지만 논문은 pass@k gap이 larger k에서 더 벌어지는 경우를 관찰한다. 이는 reasoning이 model capability boundary, 즉 output distribution에서 correct path가 등장할 가능성을 넓히는 것으로 해석된다.

### 3) Trace semantics only explanation

Reasoning trace의 semantic content가 전부라면 dummy trace는 도움이 되지 않아야 한다. 하지만 논문은 dummy trace length를 늘리면 성능이 일정 구간까지 오르는 non-monotonic pattern을 보인다. 이는 reasoning token generation 자체의 compute-buffer effect를 시사한다.

# 2. Core Idea

## 2-1. Main contribution

논문의 핵심 기여는 다섯 가지다.

1. **Parametric recall boundary expansion**
   - Reasoning ON/OFF를 같은 hybrid model에서 비교한다.
   - Gemini-2.5-Flash, Gemini-2.5-Pro, Qwen3-32B를 사용한다.
   - SimpleQA-Verified and EntityQuestions에서 pass@k curve를 비교한다.

2. **Complexity vs recall difficulty separation**
   - Reasoning benefit이 complex-labeled question에서만 큰지 본다.
   - 결과적으로 question complexity보다 parametric recall difficulty가 더 중요한 요인으로 보인다.

3. **Computational buffer experiment**
   - Reasoning trace를 dummy sequence로 대체한다.
   - Dummy length를 늘리며 semantic-free computation budget effect를 본다.

4. **Factual priming experiment**
   - Reasoning trace에서 recalled fact list를 추출한다.
   - Answer-disclosing statements를 제거하고 fact-only context를 넣어 effect를 본다.

5. **Hallucination audit and trajectory selection**
   - Intermediate facts를 search-enabled verifier로 검증한다.
   - Hallucinated intermediate fact가 final hallucination과 연결되는지 본다.
   - Fact-containing and hallucination-free trace를 선택하면 expected accuracy가 올라가는지 simulation한다.

## 2-2. Design intuition

이 논문의 central intuition은 reasoning trace를 final answer의 explanation이 아니라 **answer retrieval process의 workspace**로 보는 것이다.

Reasoning trace는 두 방식으로 작동할 수 있다.

### 1) Compute workspace

Trace token을 생성하는 동안 model은 여러 forward pass를 수행한다. 그 자체가 추가 computation depth처럼 작동한다. Trace text가 semantically meaningless dummy여도 길이가 충분하면 일부 이득이 생길 수 있다.

### 2) Semantic workspace

Trace 안에 related facts가 등장하면, 그 fact들이 target answer recall의 bridge가 된다. Model이 직접 answer를 찾지 못해도 주변 facts를 말하면서 answer basin에 가까워질 수 있다.

이 두 mechanism은 서로 다르다.

| Mechanism | What matters | Main risk |
| --- | --- | --- |
| Computational buffer | Extra generated tokens and inference depth | Optimal length unknown |
| Factual priming | Correct related facts in trace | Hallucinated facts can mislead |
| Direct decomposition | Logical substeps | Mostly not required for single-hop QA |

논문의 메시지는 reasoning benefit이 "step-by-step logic"만이 아니라는 점이다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Reasoning이 single-hop factual recall을 왜 돕는지 분석 |
| Models | Gemini-2.5-Flash, Gemini-2.5-Pro, Qwen3-32B |
| Mode | Same hybrid model with reasoning ON/OFF |
| Datasets | SimpleQA-Verified, EntityQuestions |
| Samples | Up to 100 per question |
| Main metric | pass@k와 reasoning effectiveness $\Omega$ |
| Mechanism 1 | Computational buffer |
| Mechanism 2 | Factual priming |
| Risk analysis | Intermediate fact hallucination과 final answer error |

## 3-2. Experimental setup

### 1) Models

논문은 control token이나 system instruction으로 reasoning을 ON/OFF할 수 있는 hybrid model을 사용한다.

| Model | Role |
| --- | --- |
| Gemini-2.5-Flash | Main controlled experiments, latency-quality tradeoff |
| Gemini-2.5-Pro | ON/OFF comparison |
| Qwen3-32B | ON/OFF comparison |

같은 model에서 reasoning만 toggle하는 것이 중요하다. 이렇게 해야 model 간 parametric knowledge 차이를 통제할 수 있다.

### 2) Datasets

| Dataset | Size used | Reason for use |
| --- | ---: | --- |
| SimpleQA-Verified | 1000 examples | Realistic closed-book factual QA, corrected subset |
| EntityQuestions | 1000 examples | Template-based single-hop QA, controlled phrasing |

EntityQuestions에서는 large answer space와 unambiguous answer를 가진 네 relation에서 각각 250개 example을 sampling한다.

### 3) Grading

Predicted answer는 Gemini-2.5-Flash autorater로 grade한다. 논문은 prior work에서 가져온 SimpleQA와 EntityQuestions grading prompt를 조정해 사용한다. pass@k는 unbiased estimator를 사용한다.

## 3-3. Reasoning effectiveness metric

논문은 $k=1,\ldots,N$ 전반의 relative pass@k gain을 aggregate하기 위해 summary metric $\Omega$를 정의한다. 더 큰 $k$에 더 높은 weight를 둔다.

개념적으로는 다음과 같다.

$$
\Omega(N)
=
\frac{
\sum_{k=1}^{N}
k
\cdot
\frac{
\mathrm{pass@}k_{\mathrm{ON}}
-
\mathrm{pass@}k_{\mathrm{OFF}}
}{
\mathrm{pass@}k_{\mathrm{OFF}}
}
}{
\sum_{k=1}^{N} k
}
$$

Default는 $N=100$이다. Larger $k$에 weight를 주는 것은 pass@1만이 아니라 capability boundary를 강조하기 위해서다.

## 3-4. Computational buffer experiment

Compute-only effect를 검증하기 위해 논문은 reasoning trace를 dummy content로 대체한다.

- ON Single Dummy: reasoning을 짧은 dummy sequence로 대체한다.
- ON Dummy: original trace length에 맞게 dummy sequence를 반복한다.
- ON Dummy X: fixed length $X$의 dummy trace를 사용한다.

Dummy trace가 pass@k를 개선한다면 reasoning benefit 일부는 semantic content와 독립적이라는 뜻이다.

논문은 non-monotonic pattern을 보고한다. SimpleQA-Verified에서는 dummy length가 약 2048 token까지 도움이 되지만, 4096, 8192, 16384처럼 더 긴 dummy trace에서는 성능이 떨어진다. 즉 thinking token은 많을수록 항상 좋은 것이 아니다.

## 3-5. Factual priming experiment

Reasoning trace는 논리적 derivation을 포함하지 않는 경우가 많다. 대신 candidate answer를 나열하거나, related fact를 recall하거나, search plan을 설명한다. 논문은 이 recalled fact가 answer retrieval을 돕는지 검증한다.

Pipeline은 다음과 같다.

1. LLM으로 reasoning trace에서 fact를 추출한다.
2. Question을 restate하는 fact를 제거한다.
3. Answer-disclosing statement를 제거한다.
4. Extracted fact를 context로 넣는다.
5. 비슷한 길이의 dummy fact string과 비교한다.

두 variant가 사용된다.

| Variant | Description |
| --- | --- |
| ON Facts | Reasoning ON에서 trace를 extracted fact list로 대체 |
| OFF Facts | Reasoning OFF지만 fact list를 extra context로 추가 |
| ON/OFF Dummy Facts | 같은 길이의 dummy string을 control로 사용 |

결과는 real fact가 dummy string보다 낫다는 것이다. Fact는 reasoning이 disabled된 경우에도 도움이 된다. 이는 factual content 자체가 recall에 유용하다는 해석을 지지한다.

## 3-6. Hallucination audit

논문은 search-enabled Gemini-2.5-Flash로 intermediate fact를 검증한다. Fact는 correct, incorrect, illegal, unknown으로 labeling된다. Fact가 없거나 unverified fact가 있는 trace는 clean vs hallucinated comparison에서 제외된다.

Main result는 상당히 뚜렷하다.

| Dataset | Clean trace final correct | Hallucinated trace final correct |
| --- | ---: | ---: |
| SimpleQA-Verified | 41.4% | 26.4% |
| EntityQuestions | 71.1% | 32.2% |

Within-question control 이후에도 hallucinated intermediate fact는 낮은 final correctness와 연결된다. Fitted regression slope는 SimpleQA-Verified에서 0.84, EntityQuestions에서 0.86으로 1보다 낮다.

이것이 주요 cautionary result다. Reasoning은 knowledge를 unlock할 수 있지만, hallucinated reasoning content는 final answer를 오염시킬 수 있다.

# 4. Training / Data / Recipe

## 4-1. Data

새로운 model training은 수행하지 않는다. 이 논문은 inference-time analysis study다.

Evaluation data는 다음과 같다.

| Dataset | Used subset |
| --- | --- |
| SimpleQA-Verified | 1000 verified examples |
| EntityQuestions | 1000 examples, 250 from each of 4 relations |

각 question과 model setting에 대해 논문은 최대 100개 response를 sample하여 pass@k를 추정하고 trace를 분석한다.

## 4-2. Inference settings

핵심 설계는 같은 hybrid model 안에서 reasoning ON/OFF를 toggle하는 것이다. 이는 서로 다른 model weight를 비교하는 문제를 피한다.

Reasoning ON은 final answer 전에 trace를 생성한다. Reasoning OFF는 trace generation을 억제한다. 이후 final answer를 grade한다.

Controlled mechanism 분석에서는 Gemini-2.5-Flash를 사용한다. 비싼 trace intervention에서 latency-quality tradeoff가 가장 좋기 때문이다.

## 4-3. Fact extraction and filtering

Factual priming과 hallucination analysis에는 여러 LLM-based filter가 필요하다.

- Thinking content에서 fact를 추출한다.
- Question content를 restate하는 fact를 제거한다.
- Answer를 직접 드러내는 fact를 제거한다.
- Fact list를 structured list로 parse한다.
- Search-enabled Gemini-2.5-Flash로 각 fact를 verify한다.

Appendix에는 이 단계들의 prompt가 포함된다. 논문은 fact verification procedure에 대한 small human validation도 보고하며, abstention을 제외한 classified correct/incorrect fact에서 거의 100% accuracy를 추정한다.

## 4-4. Engineering notes

1. **Use pass@k, not only pass@1**
   - Reasoning은 top-1 gain이 작더라도 reachable path를 확장할 수 있다.

2. **Longer reasoning이 항상 좋다고 가정하면 안 된다**
   - Dummy compute effect는 bounded하고 non-monotonic하다.

3. **Trace content matters**
   - Correct related fact는 도움이 된다.
   - Hallucinated fact는 해롭다.

4. **Trace selection can be useful**
   - Fact가 있고 hallucinated intermediate fact가 없는 trace를 선택하면 simulation에서 expected accuracy가 올라간다.

5. **Process reward design은 factuality를 확인해야 한다**
   - 긴 reasoning이나 confident reasoning만 reward하면 충분하지 않다.
   - Intermediate fact correctness가 더 좋은 signal이다.

# 5. Evaluation

## 5-1. Reasoning expands pass@k

평가된 model과 dataset 전반에서 reasoning ON은 pass@k를 일관되게 높인다. 논문은 benefit이 higher $k$에서 더 두드러지는 경우가 많고, Qwen3-32B on SimpleQA-Verified에서는 일부 경우 pass@k가 거의 두 배가 된다고 설명한다.

이는 reasoning이 model의 parametric recall boundary를 확장한다는 점을 보여준다. OFF mode에서는 사실상 unreachable하던 correct answer가 ON mode에서는 reachable해진다.

## 5-2. Less capable models benefit more

Reasoning effectiveness metric $\Omega$는 base model capability가 높아질수록 줄어든다. 논문은 덜 강한 model이 parameter 안에는 있지만 direct answering으로 쉽게 꺼내지 못하는 hidden knowledge를 더 많이 갖고 있기 때문이라고 해석한다.

SimpleQA도 EntityQuestions보다 높은 $\Omega$를 보인다. SimpleQA의 OFF baseline performance가 낮아 recall improvement의 headroom이 더 크기 때문일 가능성이 있다.

## 5-3. Complexity is not the main driver

SimpleQA-Verified metadata를 사용해 논문은 complex로 label된 question과 simple question을 비교한다. Complex question이 더 큰 reasoning gain을 받는다는 evidence는 찾지 못한다. Confidence interval이 겹치고, Gemini-2.5-Pro에서는 complex subset confidence interval이 zero를 지난다.

이는 main thesis를 지지한다. 이 setup에서 reasoning benefit은 주로 complex question decomposition에서 나오는 것이 아니라, parametric knowledge access 개선에서 나온다.

## 5-4. Computational buffer effect

Dummy reasoning trace는 어느 정도까지 performance를 개선한다. 이는 model이 trace semantics와 독립적으로 generated token을 extra computation으로 사용할 수 있음을 시사한다.

하지만 더 긴 dummy trace는 결국 성능을 해치고, dummy trace는 reasoning ON performance를 완전히 회복하지 못한다. 따라서 compute buffer는 gain의 일부만 설명한다.

## 5-5. Factual priming effect

Reasoning trace에서 추출한 fact는 reasoning OFF 상태에서 context로 넣어도 도움이 된다. 이는 factual priming을 지지한다. Related factual content가 answer로 가는 semantic bridge 역할을 한다.

이는 reasoning trace를 self-retrieval로 재해석하게 만든다는 점에서 중요하다. Model이 반드시 answer를 논리적으로 derive하는 것은 아니다. 주변 fact를 생성하면서 answer를 recall하기 쉬운 basin에 가까워지는 것일 수 있다.

## 5-6. Hallucination risk and trajectory selection

논문의 가장 실용적인 table은 test-time trace selection이다.

| Strategy | SimpleQA-Verified | EntityQuestions |
| --- | ---: | ---: |
| Regular | 27.9 | 56.9 |
| Only Facts | 30.2 | 58.4 |
| Only Correct Facts | 31.3 | 59.8 |

Fact를 recall한 trace를 선택하면 relative improvement는 8.2%와 2.6%다. Correct fact만 포함한 trace로 제한하면 relative improvement는 12.2%와 5.1%다.

이는 deployed algorithm은 아니다. Model을 generation oracle처럼 다루고 selection을 simulate하기 때문이다. 하지만 방향은 명확하다. Factual process supervision은 factual QA를 개선할 수 있다.

## 5-7. What really matters in the experiments

### 1) Reasoning can help non-reasoning questions

이것이 surprising result다. Trace는 logical proof가 아닐 수 있다. Computation buffer이거나 self-generated retrieval scaffold일 수 있다.

### 2) Reasoning traces are not automatically trustworthy

도움을 주는 factual priming mechanism은 동시에 model을 오도할 수도 있다. Intermediate fact가 hallucinated되면 final answer accuracy가 떨어진다.

### 3) Process supervision은 fact-aware해야 한다

"has reasoning"을 reward하는 것만으로는 충분하지 않다. Process reward는 intermediate statement가 factual하고 relevant한지 확인해야 한다.

### 4) Pass@k reveals hidden capability

Pass@1만 측정하면 reachable correct path의 확장을 과소평가할 수 있다.

# 6. Limitations

1. **Single-hop factual QA에 초점을 둔다**
   - 논문은 주로 closed-book factual recall을 연구한다.
   - Math, code, planning, multi-hop reasoning에서는 결과가 달라질 수 있다.

2. **Hybrid model dependency가 있다**
   - 이 setup은 reasoning을 ON/OFF로 toggle할 수 있는 model을 필요로 한다.
   - 모든 model이 clean control interface를 노출하지는 않는다.

3. **Autorater dependency가 있다**
   - Final answer grading은 Gemini-2.5-Flash를 사용한다.
   - Autorater error는 pass@k estimate에 영향을 줄 수 있다.

4. **Fact extraction과 verification은 model-dependent하다**
   - Extracted fact는 LLM prompt로 생성되고 filtering된다.
   - Search-enabled verifier는 자체 bias를 가질 수 있다.

5. **Test-time selection은 simulation이다**
   - "Only Correct Facts"는 correct intermediate fact를 식별할 방법이 있다고 가정한다.
   - 실제 deployment에는 efficient verifier나 trained process reward가 필요하다.

6. **Dummy trace experiment는 인위적이다**
   - Dummy token은 compute effect를 분리하지만 distribution shift를 만들 수 있다.

7. **Optimal reasoning length는 아직 모른다**
   - 더 긴 dummy trace는 결국 성능을 해친다.
   - Generation budget을 "more is better"로 정할 수 없다.

8. **Hallucination causality를 분리하기 어렵다**
   - Hallucinated trace와 wrong final answer는 강하게 연관되어 있다.
   - 논문이 within-question control을 추가하더라도, 일부는 underlying question difficulty에서 올 수 있다.

9. **Training intervention은 없다**
   - 논문은 process reward 방향을 제안하지만, 이를 사용해 model을 학습하지는 않는다.

10. **External knowledge는 여전히 closed-book이다**
    - 이 연구는 parametric recall에 관한 것이지, retrieval을 사용하는 live factuality에 관한 것이 아니다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문의 핵심은 "thinking을 켜면 factual QA가 좋아진다"가 아니다. 더 중요한 점은 **reasoning trace가 answer generation 이전의 self-retrieval interface로 작동한다**는 해석이다.

RAG에서는 model이 external store에서 evidence를 가져온다. Thinking to Recall에서는 model이 자기 parameter 안에서 related fact를 먼저 표면화한다. 그 fact가 answer를 꺼내는 bridge가 된다. 이 관점은 closed-book QA뿐 아니라 model introspection, process reward, factuality training과 연결된다.

## 7-2. Reuse potential

### Factual QA inference

Closed-book QA에서 바로 답하게 하는 대신, related fact recall phase를 두는 것이 도움이 될 수 있다. 다만 related fact를 검증하지 않으면 hallucination risk가 커진다.

### Process reward

Reasoning length나 formatting보다 intermediate factual statement의 correctness를 reward해야 한다. "좋아 보이는 reasoning"이 아니라 "factually clean reasoning"이 중요하다.

### Self-retrieval evaluation

Reasoning trace에서 answer-disclosing statement를 제거한 fact list를 만들고, 그 fact만으로 answer recall이 개선되는지 보는 evaluation은 다른 model에도 적용할 수 있다.

### RAG plus thinking

External RAG와 thinking trace를 결합할 때, trace가 hallucinated fact를 생성해 retrieved evidence를 오염시키지 않게 해야 한다. Retrieval과 reasoning 모두 fact-check 대상이어야 한다.

## 7-3. Production considerations

- Thinking mode를 항상 켜는 것은 비용과 latency가 크다.
- 더 긴 reasoning이 항상 더 좋은 것은 아니다.
- Factuality가 중요한 경우 intermediate fact는 검증 가능해야 한다.
- Final answer confidence는 trace cleanliness를 반영해야 한다.
- High-stakes factual QA에서는 여전히 external evidence retrieval이 필요하다.
- Final answer가 맞더라도 process reward는 hallucinated intermediate fact를 penalize해야 한다.

## 7-4. Follow-up papers

- SimpleQA와 SimpleQA-Verified
- EntityQuestions
- Gekhman et al. on hidden knowledge in LLMs
- Process reward models
- Self-consistency and test-time compute
- Search-o1 and agentic retrieval
- Fact verification and hallucination detection
- Reasoning LLM post-training with verifiable rewards

# 8. Summary

- Reasoning은 decomposition이 불필요한 single-hop factual recall에서도 성능을 개선한다.
- Gain은 pass@k에서 나타나며, parametric recall boundary가 확장됨을 시사한다.
- 두 mechanism이 중요하다: computational buffer와 factual priming.
- Factual priming은 intermediate fact가 correct할 때만 도움이 되며, hallucinated fact는 final error risk를 높인다.
- Practical next step은 단순히 더 긴 thinking이 아니라 fact-aware trace selection 또는 process reward다.
