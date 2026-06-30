---
layout: single
title: "Reliability without Validity: A Systematic, Large-Scale Evaluation of LLM-as-a-Judge Models Across Agreement, Consistency, and Bias Review"
categories: Study-concept
tag: [LLMJudge, Evaluation, Reliability, Bias, Benchmarking]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2606.19544)

Reliability without Validity는 LLM-as-a-Judge를 쓰는 팀이라면 꽤 불편하게 읽어야 하는 논문이다. 논문의 핵심 메시지는 "LLM judge가 불안정하다" 정도가 아니다. 더 정확히는, **일관적인 judge가 반드시 valid한 judge는 아니라는 것**이다.

요즘 model evaluation pipeline에서 LLM-as-a-Judge는 거의 default가 되었다. Open-ended answer, summarization, coding explanation, agent trajectory, RAG response처럼 정답을 exact match로 비교하기 어려운 task에서는 judge model이 human annotation을 대체하거나 보조한다. 문제는 이 judge를 검증할 때 여전히 raw exact-match agreement를 headline metric으로 쓰는 경우가 많다는 점이다.

이 논문은 그 관행을 정면으로 비판한다. Exact match는 chance agreement를 보정하지 않기 때문에, benchmark label distribution에 따라 judge의 discriminative ability를 크게 과대평가할 수 있다. 그래서 같은 judge도 MT-Bench, JudgeBench, RewardBench에서 서로 다른 순위를 보이고, 심지어 test-retest reliability가 매우 높은 judge가 position bias도 크게 보이는 경우가 나온다.

> 한 줄 요약: 이 논문은 21개 LLM judge를 9개 provider, 3개 benchmark, 3개 protocol, 118 runs, 약 541k judgments로 평가해, exact-match agreement가 judge validity를 과대평가하고, high consistency가 severe bias와 공존할 수 있음을 보인 뒤, Cohen's kappa, position swap, repeated runs를 포함한 Minimum Viable Validation Protocol을 제안한다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- LLM judge validation에서 exact match 대신 chance-corrected metric을 headline으로 써야 하는 이유를 수치로 보여준다.
- Test-retest reliability와 position bias가 서로 다른 failure axis임을 명확히 분리한다.
- Judge ranking이 benchmark에 따라 최대 14 positions까지 움직일 수 있음을 보여준다.
- Production evaluation에서 "consistent but biased" judge가 더 위험할 수 있다는 paradox를 제시한다.
- LLM judge를 쓰는 실무 pipeline에 바로 넣을 수 있는 minimum validation checklist를 제공한다.
- Verbosity bias가 과거 literature보다 작게 나타난 점도 함께 보고해, bias를 고정된 folklore로 다루지 않는다.

이 글에서는 이 논문을 "LLM judge는 못 믿는다"가 아니라, **LLM judge를 deployment artifact로 쓰려면 무엇을 최소한 검증해야 하는가**를 정리한 evaluation-methodology paper로 읽는다.

# 1. Problem Setting

## 1-1. Problem definition

LLM-as-a-Judge validation은 보통 다음 구조를 가진다.

1. Human label $y_i$가 있는 benchmark item을 준비한다.
2. Judge model $J$가 verdict $\hat{y}_i$를 낸다.
3. Human label과 judge verdict가 얼마나 일치하는지 본다.

가장 흔한 metric은 exact match다.

$$
\mathrm{EM}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}
\left[
\hat{y}_i=y_i
\right]
$$

문제는 EM이 chance agreement를 보정하지 않는다는 점이다. Label distribution이 한쪽으로 치우쳐 있으면, 별로 discriminative하지 않은 judge도 높은 EM을 얻을 수 있다. Pairwise preference benchmark에서는 tie/non-tie 처리나 class prior에 따라 이 문제가 더 커진다.

Cohen's kappa는 observed agreement $p_o$에서 expected-by-chance agreement $p_e$를 제거한다.

$$
\kappa
=
\frac{
p_o-p_e
}{
1-p_e
}
$$

논문은 exact match와 Cohen's kappa의 차이를 kappa deflation으로 부른다.

$$
\Delta_{\kappa}(j,b)
=
\mathrm{EM}(j,b)
-
\kappa(j,b)
$$

여기서 $j$는 judge, $b$는 benchmark다.

이 논문의 문제 설정은 다음 질문으로 요약된다.

> LLM judge가 human label과 얼마나 raw agreement를 보이는가가 아니라, chance-corrected agreement, repeated-run consistency, position bias, verbosity bias를 함께 봤을 때 실제로 deployment-worthy한가?

## 1-2. Why previous approaches are insufficient

### 1) Exact match is not enough

Exact match는 직관적이고 설명하기 쉽다. 하지만 chance correction이 없기 때문에 label distribution에 민감하다. 논문은 MT-Bench에서 exact match와 Cohen's kappa 사이 gap이 모든 21개 judge에서 33-41 percentage points 수준으로 나타난다고 보고한다.

즉 "85% agreement"라는 숫자는 judge가 human과 85%의 meaningful agreement를 가진다는 뜻이 아닐 수 있다. 논문은 실제로 MT-Bench에서 85% agreement가 $\kappa \approx 0.48$ 정도에 해당할 수 있다고 지적한다.

### 2) Test-retest reliability is not validity

Test-retest reliability는 같은 judge가 같은 item을 반복 평가했을 때 얼마나 안정적으로 같은 verdict를 내는지 본다.

$$
\mathrm{TRR}
=
\mathrm{Agreement}
\left(
J^{(1)}(x),
J^{(2)}(x)
\right)
$$

하지만 stable output이 correct process를 의미하지는 않는다. 어떤 judge가 항상 position A를 선호한다면, repeated run에서는 매우 안정적일 수 있지만 valid한 judge는 아니다.

이 논문이 말하는 consistency-bias paradox가 바로 이것이다.

### 3) Bias types should be measured separately

Position bias, verbosity bias, prompt sensitivity, test-retest reliability는 서로 다른 failure mode다. 한 metric이 좋다고 다른 metric이 자동으로 좋다는 보장은 없다.

이 논문은 세 protocol을 분리한다.

| Protocol | What it tests |
| --- | --- |
| Agreement | Human label과 chance-corrected agreement |
| Consistency | Repeated identical evaluation stability |
| Bias audit | Position bias and verbosity bias |

### 4) Benchmark transfer is not guaranteed

Judge가 MT-Bench에서 높다고 JudgeBench나 RewardBench에서도 높을 것이라는 보장은 없다. 논문은 judge ranking이 benchmark에 따라 크게 움직인다고 보고한다. Evaluation deployment에서는 target domain and target rubric에 맞춘 validation이 필요하다.

# 2. Core Idea

## 2-1. Main contribution

논문의 주요 contribution은 네 가지다.

1. **Large-scale judge evaluation**
   - 21 judges
   - 9 providers
   - MT-Bench, JudgeBench, RewardBench
   - agreement, consistency, bias audit protocol
   - 118 runs
   - 약 541k individual judgments

2. **Kappa deflation**
   - Exact match와 Cohen's kappa 사이 systematic gap을 측정한다.
   - MT-Bench에서 33.8-41.2 pp deflation을 보고한다.

3. **Consistency-bias paradox**
   - Test-retest reliability가 매우 높은 judge가 severe position bias도 가질 수 있음을 보인다.
   - Qwen 3 8B and Gemini 2.5 Flash case가 대표적이다.

4. **Minimum Viable Validation Protocol**
   - Chance-corrected agreement를 headline metric으로 쓴다.
   - Position swap bias audit을 수행한다.
   - Repeated runs로 test-retest reliability를 측정한다.
   - Verbosity bias는 scope caveat와 함께 보고한다.

## 2-2. Design intuition

이 논문의 design intuition은 "judge validation도 benchmark score가 아니라 measurement system validation이어야 한다"는 것이다.

LLM judge는 model output을 평가하는 measurement instrument다. Instrument를 검증할 때는 다음 세 가지가 필요하다.

1. Agreement with target
   - Human label or target rubric과 얼마나 맞는가.

2. Reliability
   - 같은 input에 대해 얼마나 재현 가능한가.

3. Bias invariance
   - Response order, length, prompt form 같은 irrelevant factor에 흔들리지 않는가.

이 세 가지는 서로 다른 축이다.

| Axis | Good number means | Still possible failure |
| --- | --- | --- |
| Exact match | Human label과 많이 맞음 | Chance-corrected discrimination이 약함 |
| Cohen's kappa | Chance-corrected agreement가 높음 | Bias or instability가 남음 |
| Test-retest | Repeated run이 안정적 | Stable position bias |
| Position bias | Order invariant | Human alignment가 낮을 수 있음 |
| Verbosity bias | Length-independent under current rubric | Other bias가 남음 |

논문 제목의 "Reliability without Validity"는 이 distinction을 잘 요약한다. Reliability는 필요조건이지만 충분조건이 아니다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | LLM-as-a-Judge validation protocol의 failure mode 측정 |
| Judges | 21 models from 9 providers |
| Benchmarks | MT-Bench, JudgeBench, RewardBench |
| Protocols | Agreement, consistency, bias audit |
| Judgments | Around 541k across 118 runs |
| Main metric | Cohen's kappa rather than exact match |
| Bias metrics | Position bias, position flip rate, verbosity bias |
| Main output | Minimum Viable Validation Protocol |

## 3-2. Evaluated judges

논문은 21개 judge를 세 tier로 나눈다.

| Tier | Description |
| --- | --- |
| Tier 1 | Widely deployed production judges |
| Tier 2 | Cost-conscious models |
| Tier 3 | April 2026 frontier models and open-source systems |

Model들은 8B부터 100B 이상까지 다양한 parameter scale을 포함하고, 9개 provider에 걸쳐 있다. 논문은 evaluation 시점의 release date와 per-token cost도 함께 제시한다.

이 구성이 중요한 이유는 judge validation이 frontier model 비교만으로 끝나면 안 되기 때문이다. 실제 production에서는 비용 때문에 smaller or cheaper judge를 쓰는 경우가 많다. 따라서 frontier judge와 cost-conscious judge를 함께 비교해야 deployment decision에 더 가깝다.

## 3-3. Benchmarks

세 benchmark는 label distribution과 difficulty가 서로 다르다.

| Benchmark | Role in paper |
| --- | --- |
| MT-Bench | Widely used pairwise benchmark, but compressed kappa range |
| JudgeBench | Harder benchmark with sharper discrimination |
| RewardBench | Reward/preference benchmark with different label structure |

논문은 kappa deflation이 benchmark마다 다르게 나타난다고 보고한다. Cohort-mean deflation은 MT-Bench에서 38.6 pp, JudgeBench에서 23.7 pp, RewardBench에서 10.2 pp다. 즉 raw agreement gap은 judge 자체의 속성만이 아니라 benchmark label distribution에도 의존한다.

## 3-4. Agreement protocol

Agreement protocol에서는 각 judge verdict를 human label과 비교한다. 논문은 exact match, Cohen's kappa, kappa deflation을 함께 보고한다.

핵심 교훈은 exact match를 headline number로 두면 안 된다는 것이다.

$$
\Delta_{\kappa}
=
\mathrm{EM}
-
\kappa
$$

$\Delta_{\kappa}$가 크다는 것은 raw agreement가 chance-corrected discriminative ability를 과대평가하고 있다는 뜻이다.

## 3-5. Consistency protocol

Consistency protocol은 같은 item을 repeated run으로 평가한다. 중요한 metric은 다음과 같다.

| Metric | Meaning |
| --- | --- |
| Test-retest reliability | Corpus-level stability across runs |
| Self-consistency | Item-level agreement with majority verdict |
| Flip rate | How often verdict changes under position swap |

논문은 harder benchmark에서 consistency가 떨어진다고 보고한다. 예를 들어 16개 judge의 mean test-retest는 MT-Bench에서 0.943이지만 JudgeBench에서는 0.911로 내려간다.

## 3-6. Bias audit

Position bias는 response order를 swap해 측정한다.

$$
\mathrm{PB}
=
\left|
P(A\ \mathrm{wins}) - 0.5
\right|
$$

$\mathrm{PB}=0$이면 aggregate level에서 A/B position preference가 없다는 뜻이다. 값이 크면 judge가 answer quality가 아니라 response order에 영향을 받고 있다는 뜻이다.

Verbosity bias는 response length difference와 judge verdict 사이의 correlation으로 측정한다. 논문은 single pairwise rubric 조건에서 이 cohort의 verbosity bias가 작다고 보고한다.

더 우려되는 것은 position bias다. MT-Bench bias audit에서 Gemini 2.5 Pro는 position bias 0.002로 가장 낮지만, Qwen 3 8B는 0.192까지 올라간다.

## 3-7. Minimum Viable Validation Protocol

논문이 제안하는 MVVP는 단순하지만 중요하다.

1. **Chance-correct**
   - Exact match와 함께 Cohen's kappa나 Krippendorff's alpha를 보고한다.
   - Chance-corrected metric을 headline reliability number로 둔다.

2. **Swap positions**
   - AB와 BA paired evaluation을 수행한다.
   - $\left|P(A\ \mathrm{wins})-0.5\right|$를 보고한다.

3. **Replicate**
   - Temperature 0에서도 response caching을 끄고 최소 3회 독립 평가를 수행한다.
   - Test-retest reliability를 보고한다.

이는 full audit framework가 아니다. Judge를 deploy하기 전 수행해야 하는 minimum protocol이다.

# 4. Training / Data / Recipe

## 4-1. Data

이 논문은 새 judge를 학습하는 논문이 아니라, existing judge benchmark에서 judge model을 체계적으로 평가하는 논문이다.

| Dataset | Usage |
| --- | --- |
| MT-Bench | Agreement and bias analysis |
| JudgeBench | Harder judge validation and rank transfer |
| RewardBench | Additional agreement and rank-transfer check |

논문은 약 541k individual judgments를 생성한다. 여기에는 protocol별 repeated run과 position-swap condition이 포함된다.

## 4-2. Evaluation recipe

Evaluation recipe는 다음 순서로 요약할 수 있다.

1. Provider와 cost tier가 다른 judge model들을 선택한다.
2. 각 benchmark에서 human label 대비 agreement를 측정한다.
3. EM, Cohen's kappa, kappa deflation을 계산한다.
4. Repeated evaluation으로 test-retest reliability를 측정한다.
5. Answer position을 swap해 position bias와 flip rate를 측정한다.
6. Pairwise rubric 아래 verbosity bias를 측정한다.
7. Benchmark 간 rank trajectory를 비교한다.
8. 결과를 MVVP로 정리한다.

## 4-3. Engineering notes

1. **Always log raw verdicts**
   - Aggregate score만 저장하면 kappa, flip rate, bias를 나중에 계산할 수 없다.

2. **Use paired AB/BA item IDs**
   - Position bias는 같은 item과 같은 answer를 순서만 바꿔 비교해야 한다.

3. **Disable caching for retest**
   - Test-retest는 cached replay가 아니라 independent run이어야 한다.

4. **Use chance-corrected metric as headline**
   - Exact match는 readability를 위해 남길 수 있지만 primary metric이면 안 된다.

5. **Report uncertainty**
   - Judge variance, repeated-run variance, bootstrap interval을 함께 보고해야 한다.

6. **Validate on target rubric**
   - Cross-benchmark ranking shift가 크므로 general judge leaderboard만 믿으면 안 된다.

# 5. Evaluation

## 5-1. Kappa deflation

가장 중요한 결과는 universal kappa deflation이다.

| Benchmark | Cohort-mean deflation |
| --- | ---: |
| MT-Bench | 38.6 pp |
| JudgeBench | 23.7 pp |
| RewardBench | 10.2 pp |

MT-Bench에서는 21개 model 모두에서 33.8-41.2 pp deflation이 나타난다. 이는 raw exact-match agreement가 actual chance-corrected judge quality를 크게 과대평가할 수 있음을 뜻한다.

## 5-2. Cross-benchmark rank instability

Judge ranking은 benchmark에 따라 최대 14 positions까지 이동한다. 이는 common benchmark에서 높은 점수를 받은 judge가 다른 evaluation domain에서도 reliable할 것이라는 deployment assumption을 약하게 만든다.

실무적 결론은 단순하다. Product evaluation이 custom rubric을 쓴다면, 그 rubric에서 직접 validation해야 한다.

## 5-3. Consistency-bias paradox

논문의 가장 기억에 남는 결과는 consistency-bias paradox다.

| Judge | Test-retest | Position bias |
| --- | ---: | ---: |
| Qwen 3 8B | 0.992 | 0.192 |
| Gemini 2.5 Flash | 0.988 | 0.125 |

이 judge들은 highly reproducible하지만, 동시에 position에 의해 systematic하게 bias된다. Test-retest reliability만 보고하면 안전해 보이지만, 실제 decision process는 position invariance를 위반한다.

## 5-4. Position bias heterogeneity

Position bias는 model마다 크게 다르다.

| Model | Position bias |
| --- | ---: |
| Gemini 2.5 Pro | 0.002 |
| Qwen 3 8B | 0.192 |
| Gemini 2.5 Flash | 0.125 |
| GPT-4o-mini | 0.047 |

이 차이는 position bias가 피할 수 없는 상수가 아니라는 뜻이다. Judge와 evaluation protocol마다 별도로 측정해야 한다.

## 5-5. Verbosity bias

논문은 single pairwise rubric에서 cohort 전체 verbosity bias가 0.011 미만이라고 보고한다. 이는 과거 literature에서 length가 20-40% variance를 설명한다고 보고한 것보다 훨씬 작다.

다만 이를 "verbosity bias가 해결됐다"로 읽으면 안 된다. 이 논문의 tested rubric과 cohort에서 length bias가 position bias와 kappa deflation보다 작게 관측됐다는 의미다.

## 5-6. Provider and family patterns

논문은 Anthropic judge들이 JudgeBench에서 평균 $\kappa=0.770$, average position bias 0.020을 보였고, 세 OpenAI flagship model은 JudgeBench 평균 $\kappa=0.467$을 보였다고 보고한다. OpenAI model family 안에서는 JudgeBench 기준 within-family progression도 관찰된다.

이 숫자는 흥미롭지만 snapshot으로 봐야 한다. Provider model version, API behavior, evaluation cost는 빠르게 바뀔 수 있다.

## 5-7. What really matters in the experiments

### 1) Exact match is not validity

Raw agreement는 이해하기 쉬운 statistic으로는 유용하지만, validation headline이 되어서는 안 된다.

### 2) Consistency can be dangerous if biased

Noisy stochastic judge도 문제지만, stable position bias를 가진 deterministic judge는 production에서 더 교묘하게 위험하다.

### 3) Benchmark choice changes the story

MT-Bench는 JudgeBench보다 kappa difference를 더 압축한다. Harder or differently distributed benchmark는 common benchmark가 숨긴 차이를 드러낼 수 있다.

### 4) Minimum validation can be simple

MVVP는 무겁지 않다. Kappa, position swap, repeated runs면 된다. 진지한 evaluation pipeline이라면 충분히 수행 가능한 수준이다.

# 6. Limitations

1. **Benchmark scope가 제한적이다**
   - MT-Bench, JudgeBench, RewardBench는 중요한 evaluation regime을 다루지만 모든 domain을 대표하지는 않는다.
   - Medical, legal, code security, agent tool-use 같은 domain-specific task는 별도 validation이 필요하다.

2. **Model snapshot이다**
   - Evaluation은 2026년 4월 전후 model version과 provider behavior를 반영한다.
   - API update가 judge behavior를 바꿀 수 있다.

3. **Pairwise rubric scope가 제한적이다**
   - Verbosity bias는 tested pairwise rubric 아래에서 작게 나타났다.
   - 다른 prompt, scoring format, domain에서는 length bias가 다르게 나타날 수 있다.

4. **Human label quality에 의존한다**
   - Cohen's kappa는 reference label set의 품질에 의존한다.
   - Human label도 noisy하거나 biased할 수 있다.

5. **Full causal diagnosis는 아니다**
   - 논문은 failure mode를 측정하지만, 각 model이 왜 그런 behavior를 보이는지 완전한 causal explanation을 제공하지는 않는다.

6. **MVVP는 minimum protocol일 뿐이다**
   - MVVP는 adversarial prompt injection, rubric drift, multi-turn judging, cross-lingual evaluation을 모두 다루지는 않는다.

7. **Cost-quality trade-off는 시간에 따라 바뀐다**
   - Per-token cost와 model version은 불안정하다.
   - Production judge choice는 주기적으로 다시 평가해야 한다.

8. **Temperature와 decoding setting에 의존한다**
   - 논문은 특정 evaluation setting을 사용한다.
   - Decoding이나 system prompt가 바뀌면 consistency와 bias도 달라질 수 있다.

9. **Rank transfer는 all-or-nothing이 아니다**
   - 어떤 judge가 한 benchmark에서 약해도 더 좁은 internal rubric에서는 유용할 수 있다.
   - Validation은 반드시 task-specific이어야 한다.

10. **Bias audit은 아직 제한적이다**
    - Position과 verbosity는 중요하지만 exhaustive하지 않다.
    - Self-preference, style bias, safety bias, language bias, provider-family bias도 별도 확인이 필요하다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문의 가장 중요한 메시지는 LLM-as-a-Judge를 없애자는 것이 아니다. 오히려 반대다. LLM judge를 계속 쓰려면, **judge를 model이 아니라 measurement instrument로 검증해야 한다**는 것이다.

AI product와 research pipeline에서 judge model은 점점 더 central해지고 있다.

- SFT data filtering
- RL reward modeling
- benchmark ranking
- regression testing
- RAG answer grading
- agent trajectory scoring
- production eval dashboard

이 judge가 biased하면 downstream model training과 product decision도 biased해진다. 특히 "consistent but biased" judge는 더 위험하다. 안정적인 숫자가 나오기 때문에 더 빨리 trust를 얻는다.

## 7-2. Practical reuse

내부 evaluation pipeline에는 최소한 다음을 넣는 편이 좋다.

| Check | Implementation |
| --- | --- |
| Chance correction | Cohen's kappa나 Krippendorff's alpha |
| Position invariance | AB/BA swap |
| Retest | 3+ independent runs |
| Tie handling | Explicit tie policy |
| Uncertainty | Bootstrap confidence interval |
| Target-domain validation | Product-specific rubric and examples |
| Bias dashboard | Position, length, style, language |

이 중 position swap과 kappa는 비용 대비 효과가 매우 크다. 평가 pipeline을 이미 돌리고 있다면, AB/BA swap과 kappa 계산을 추가하는 것은 어렵지 않다.

## 7-3. When LLM judge is still useful

LLM judge가 완벽하지 않아도 쓸 수 있는 상황은 많다.

- Low-stakes triage
- Pairwise regression signal
- Human review prioritization
- Data cleaning pre-filter
- Qualitative feedback generation
- Draft evaluator before formal benchmark

다만 high-stakes decision에서는 single judge single run을 쓰면 안 된다. Multi-judge, repeated-run, position-swap, human spot-check를 함께 둬야 한다.

## 7-4. Follow-up papers

- MT-Bench와 Chatbot Arena
- RewardBench
- JudgeBench
- G-Eval
- AlpacaEval
- Arena-Hard
- BabelJudge
- The Coin Flip Judge?
- LLM-as-a-Judge survey and reliability studies

# 8. Summary

- Exact-match agreement는 LLM judge quality를 크게 과대평가할 수 있다.
- Cohen's kappa는 exact match와 함께, 가능하면 그보다 앞에 보고해야 한다.
- High test-retest reliability는 severe position bias와 공존할 수 있다.
- Judge ranking은 benchmark 사이에서 안정적으로 transfer되지 않는다.
- Minimum judge validation protocol에는 chance correction, position swap, repeated runs가 포함되어야 한다.
