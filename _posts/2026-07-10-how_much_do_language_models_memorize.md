---
layout: single
title: "How much do language models memorize? Review"
categories: Study-concept
tag: [LLM, Memorization, Privacy, ScalingLaw, Evaluation]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2505.24832)

How much do language models memorize?는 LLM memorization 논의에서 꽤 중요한 구분을 제안하는 논문이다. 지금까지 memorization은 주로 두 방식으로 이야기되었다. 하나는 model이 training data를 그대로 뱉어내는 extraction이고, 다른 하나는 어떤 sample이 training set에 있었는지 맞히는 membership inference다.

하지만 이 논문은 두 지표만으로는 memorization을 제대로 정의하기 어렵다고 본다. 모델이 어떤 문장을 생성했다고 해서 반드시 그 문장을 외웠다는 뜻은 아니다. 반대로 문장을 verbatim으로 생성하지 못해도 sample-level pattern을 저장했을 수 있다. 더 큰 문제는 memorization과 generalization이 섞인다는 점이다. 모델이 수학 문제를 맞혔다고 해서 그 exact equation을 외웠다고 볼 수 없는 것처럼, 모델이 training sample을 낮은 loss로 예측한다고 해서 그 sample만을 외운 것인지, 데이터 생성 분포를 일반화한 것인지 구분해야 한다.

> 한 줄 요약: 이 논문은 memorization을 compression 관점에서 정의하고, unintended memorization과 generalization을 분리해 GPT-style transformer의 empirical capacity와 membership inference scaling law를 측정한다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- LLM privacy 논의를 extraction 성공 여부가 아니라, model이 sample에 대해 몇 bit의 정보를 갖고 있는가로 다시 정의한다.
- Memorization과 generalization을 분리하려는 수학적 framework를 제공한다.
- Synthetic random data에서는 generalization을 제거할 수 있으므로 model capacity를 직접 측정할 수 있다.
- 논문은 GPT-style model capacity가 대략 3.6 bits per parameter 근처라고 보고한다.
- Dataset size, model capacity, membership inference success가 어떻게 연결되는지 scaling law 관점으로 설명한다.

# 1. Problem Setting

## 1-1. Problem definition

이 논문이 겨냥하는 문제는 "언어 모델이 training data를 얼마나 외웠는가"를 instance level에서 측정하는 것이다. 여기서 instance level이라는 점이 중요하다. Dataset 전체에 대한 privacy risk도 중요하지만, 실제 질문은 대개 특정 sample에 대해 생긴다.

예를 들어 다음 질문들이 있다.

- 이 문서가 모델에 얼마나 저장되어 있는가.
- 이 특정 sample은 membership inference attack에 취약한가.
- 모델이 어떤 문장을 생성했을 때, 그 문장을 외운 것인가 아니면 일반화한 것인가.
- 모델 크기와 dataset size가 memorization risk를 어떻게 바꾸는가.

기존 연구는 extraction과 membership inference를 많이 사용했다. 하지만 extraction은 memorization의 충분조건도 필요조건도 아닐 수 있다. 모델은 prompt coercion으로 임의 문자열을 출력할 수 있고, 반대로 저장된 정보를 verbatim으로 복원하지 못할 수도 있다.

Membership inference도 마찬가지다. Loss가 낮아서 train sample로 판정되었다고 해도, 그 이유가 sample-specific memorization인지, 일반적인 distribution knowledge인지 구분하기 어렵다.

## 1-2. Why previous approaches are insufficient

기존 접근의 핵심 한계는 generalization을 memorization으로 착각할 수 있다는 점이다.

좋은 언어 모델은 training set에 없던 문장도 그럴듯하게 예측한다. 이것은 좋은 일이다. 하지만 loss-based membership inference에서는 이런 generalization이 train sample과 test sample의 차이를 흐릴 수 있다. 또한 extraction-based evaluation에서는 모델이 생성한 문자열이 실제 training data와 같아도, 그것이 data copying인지 distributional regularity인지 판단하기 어렵다.

그래서 이 논문은 memorization을 두 부분으로 나눈다.

- Intended memorization: 데이터 생성 분포에 대한 일반화. 논문에서는 generalization으로 본다.
- Unintended memorization: 특정 dataset이나 sample에 대한 추가 정보 저장. Privacy risk와 더 직접적으로 연결된다.

이 구분이 없으면 "모델이 알고 있다"와 "모델이 외웠다"가 섞인다. 이 논문의 핵심은 바로 이 둘을 분리하는 것이다.

# 2. Core Idea

## 2-1. Main contribution

논문은 memorization을 compression 관점에서 정의한다. 직관은 단순하다. 어떤 reference model만 있을 때보다 target model이 있을 때 sample을 더 짧게 encode할 수 있다면, target model은 그 sample에 대한 정보를 갖고 있는 것이다.

간단히 쓰면 다음과 같은 차이를 본다.

$$
M(x, f) = L(x | ref) - L(x | f)
$$

여기서 $x$는 sample, $f$는 target model, $ref$는 data-generating process를 잘 approximates하는 reference model이다. $L(x | ref)$는 reference만 있을 때의 code length이고, $L(x | f)$는 target model을 사용할 때의 code length다. 차이가 클수록 target model이 그 sample을 더 잘 compress한다는 뜻이다.

이 논문은 이 관점을 Kolmogorov complexity와 Shannon information 관점으로 연결하고, 실제 측정에서는 likelihood를 사용한다.

## 2-2. Design intuition

Memorization을 compression으로 보면 여러 문제가 깔끔해진다.

첫째, verbatim extraction에 의존하지 않는다. 모델이 sample을 직접 출력하지 않아도, 그 sample을 더 짧게 encode하는 데 도움이 되면 정보가 저장된 것이다.

둘째, generalization과 unintended memorization을 분리할 수 있다. Reference model이 data distribution을 잘 알고 있다면, reference가 이미 설명할 수 있는 부분은 generalization으로 처리할 수 있다. Target model이 reference보다 추가로 설명하는 부분이 sample-specific memorization에 가깝다.

셋째, model capacity를 bit 단위로 측정할 수 있다. Random bitstring처럼 generalization이 불가능한 data를 쓰면, 모델이 줄이는 code length는 사실상 raw memorization이다. 이 경우 총 memorization이 모델의 empirical capacity를 나타낸다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | language model memorization을 bit 단위로 측정 |
| Core definition | compression gain by target model |
| Key split | unintended memorization vs generalization |
| Synthetic setup | random data로 generalization 제거 |
| Text setup | reference or oracle model로 generalization 보정 |
| Main outputs | model capacity estimate, double descent explanation, membership inference scaling law |

## 3-2. Module breakdown

### 1) Statistical memorization definition

논문은 먼저 statistical view에서 시작한다. Model이 dataset으로부터 얻은 정보 전체를 보되, data-generating process에 대한 정보와 특정 sample에 대한 정보를 분리한다.

이 관점에서 unintended memorization은 model이 dataset instance에 대해 갖고 있는 정보 중, underlying distribution을 알고 있어도 남는 부분이다. 반대로 generalization은 underlying distribution에 대한 정보다.

### 2) Kolmogorov-style measurement

Entropy 기반 정의는 random variable에는 자연스럽지만, 우리가 실제로 다루는 것은 single trained model과 single sample이다. 그래서 논문은 Kolmogorov complexity 기반의 compression view로 넘어간다.

정확한 Kolmogorov complexity는 계산할 수 없지만, language model likelihood는 code length의 practical approximation이 된다. Model이 sample에 높은 probability를 주면 더 짧게 encode할 수 있다.

### 3) Synthetic random data experiment

Generalization을 제거하기 위해 uniform random sequence를 사용한다. Random data는 reusable pattern이 없기 때문에, 모델이 잘 예측한다면 그것은 거의 memorization이다.

이 설정에서는 dataset entropy를 정확히 계산할 수 있고, target model의 likelihood를 통해 compressed length를 계산할 수 있다. Dataset size를 키우면 모델은 처음에는 sample을 외우다가 capacity에 도달한 뒤 plateau에 도달한다.

### 4) Real text experiment

Real text에서는 generalization이 가능하다. 따라서 논문은 FineWeb 기반 text data를 사용하고, reference model 또는 oracle reference model을 통해 distribution-level knowledge를 보정한다.

흥미로운 점은 text에서도 dataset size와 model capacity의 관계가 double descent와 연결된다는 해석이다. Dataset이 capacity보다 작을 때는 sample-level memorization으로 loss를 낮출 수 있지만, dataset이 capacity를 넘어서면 모델은 개별 sample을 외우는 대신 공유되는 structure를 학습해야 한다.

### 5) Membership inference scaling law

마지막으로 논문은 model parameter count, dataset size, token count를 바탕으로 loss-based membership inference F1을 예측하는 scaling law를 제안한다. 직관은 명확하다. 같은 model capacity에서 dataset이 커질수록 average sample에 대한 unintended memorization은 작아지고, membership inference는 어려워진다.

# 4. Training / Data / Recipe

## 4-1. Data

논문은 두 가지 data regime을 사용한다.

1. Synthetic random sequences.
   - Generalization을 제거하기 위한 setting이다.
   - Random sequence는 true distribution에서 reusable structure가 없으므로 memorization capacity 측정에 적합하다.

2. Real text from FineWeb.
   - 실제 language modeling setting에서 memorization과 generalization을 함께 본다.
   - Deduplication을 중요하게 다룬다. 중복이 남아 있으면 extraction이나 membership inference를 잘못 해석할 수 있기 때문이다.

## 4-2. Training strategy

논문은 수백 개의 transformer language model을 scratch에서 학습한다. 모델 크기는 작은 GPT-style transformer부터 1.5B parameter scale까지 포함된다. Synthetic setting에서는 architecture, data size, seed를 바꿔가며 capacity plateau를 측정한다.

핵심은 single large production model 하나를 분석하는 것이 아니라, controlled training grid를 만들어 capacity와 dataset size의 관계를 보는 것이다. 이런 방식 덕분에 membership inference scaling law도 empirical validation이 가능해진다.

## 4-3. Engineering notes

이 논문에서 실무적으로 가져갈 점은 다음과 같다.

1. Memorization 측정에는 deduplication이 중요하다.
   - 중복 sample이 있으면 extraction을 memorization으로 과대평가할 수 있다.
   - FineWeb experiment에서도 additional deduplication을 수행한다.

2. Extraction rate와 membership inference는 다른 지표다.
   - Membership inference가 쉬워도 exact extraction은 어려울 수 있다.
   - 반대로 extraction이 되었다고 해서 모든 memorization을 설명하는 것도 아니다.

3. Model capacity는 privacy risk의 upper bound처럼 읽을 수 있다.
   - 하지만 실제 risk는 data distribution, training recipe, deduplication, prompt attack, access level에 따라 달라진다.

4. Average sample과 rare sample을 구분해야 한다.
   - 논문 scaling law는 average-case membership inference를 설명하는 데 초점이 있다.
   - Unique, duplicated, high-loss, sensitive sample은 다른 risk profile을 가질 수 있다.

# 5. Evaluation

## 5-1. Main results

가장 headline이 되는 결과는 GPT-style transformer가 대략 3.6 bits per parameter 수준의 memorization capacity를 보인다는 추정이다. 논문은 synthetic random data에서 generalization을 제거하고 capacity plateau를 관찰한다.

또한 dataset size가 model capacity를 넘어서면 unintended memorization이 더 이상 늘지 않고, 모델이 generalization으로 이동하는 현상을 관찰한다. 논문은 이 지점이 double descent와 연결된다고 해석한다.

Real text setting에서는 model size가 클수록 sample-level unintended memorization이 커지고, dataset size가 커질수록 average sample에 대한 memorization과 membership inference가 어려워진다. 이 결과는 현대 LLM처럼 token-per-parameter ratio가 큰 pretraining regime에서는 average training sample에 대한 loss-based membership inference가 통계적으로 어려울 수 있다는 해석으로 이어진다.

## 5-2. What really matters in the experiments

### 1) "얼마나 외웠나"를 bit로 묻는다

이 논문은 memorization을 binary event로 보지 않는다. 외웠다 또는 안 외웠다가 아니라, model이 sample에 대해 몇 bit의 정보를 갖는지를 묻는다. 이 framing은 privacy evaluation을 더 정량적으로 만든다.

### 2) Generalization 제거 setting이 중요하다

Random data experiment는 단순한 toy experiment가 아니다. Generalization이 완전히 제거된 setting이기 때문에 capacity measurement가 가능하다. Real text만 보면 memorization과 generalization이 섞여 해석이 훨씬 어려워진다.

### 3) Double descent를 capacity 관점으로 읽는다

논문은 dataset size가 capacity를 넘는 시점에서 모델이 개별 sample memorization에 의존하기 어려워지고, reusable structure를 학습하기 시작한다고 본다. 이 해석은 double descent를 loss curve 현상이 아니라 information allocation 문제로 읽게 만든다.

### 4) Membership inference는 extraction보다 민감하다

논문은 membership inference와 extraction을 모두 본다. Membership inference는 exact text generation보다 더 쉬울 수 있다. 이는 privacy risk를 extraction demo만으로 판단하면 안 된다는 뜻이다.

# 6. Limitations

1. 실험 환경 일반화에 주의해야 한다.
   - 논문도 결과가 특정 dataset, architecture, training setup에 의존할 수 있다고 명시한다.
   - 대규모 production LLM의 tokenizer, optimizer, curriculum, data filtering은 다를 수 있다.

2. Capacity estimate는 empirical lower bound에 가깝다.
   - Gradient descent가 global optimum을 찾는다는 보장은 없다.
   - 따라서 측정된 bits per parameter가 이론적 최대 저장량은 아니다.

3. Average-case privacy와 tail risk는 다르다.
   - 현대 LLM에서 average sample membership inference가 어렵더라도, rare or duplicated sensitive samples는 여전히 취약할 수 있다.
   - 이 논문 결과를 privacy risk가 사라졌다는 주장으로 읽으면 안 된다.

4. Reference model choice가 중요하다.
   - Real text에서 generalization을 보정하려면 reference model이 필요하다.
   - Reference가 약하면 unintended memorization을 과대평가할 수 있고, reference가 너무 강하면 반대로 해석이 달라질 수 있다.

5. Loss-based membership inference 중심이다.
   - Prompt-based extraction, adaptive attack, black-box API constraint, fine-tuning leakage 등은 별도 문제다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문은 LLM privacy와 memorization 논의를 훨씬 차분하게 만든다. "모델이 training data를 외웠다"라는 말은 너무 넓다. 실제로는 model capacity, dataset size, distributional regularity, duplication, reference model, attack type이 모두 함께 작동한다.

이 논문의 가장 유용한 점은 memorization을 compression budget으로 보는 관점이다. 모델은 제한된 capacity 안에서 어떤 정보를 저장할지 선택한다. Dataset이 작고 random하면 sample을 외우는 것이 loss를 줄이는 가장 쉬운 방법이다. Dataset이 크고 structure가 많으면 개별 sample보다 reusable pattern을 저장하는 편이 낫다. 이 관점은 pretraining data curation, deduplication, privacy review를 모두 하나의 프레임으로 묶어준다.

## 7-2. Reuse potential

재사용해볼 만한 포인트는 다음과 같다.

1. Dataset audit에서 duplication과 rarity를 별도로 측정하기.
   - Average sample보다 duplicated or rare sample의 risk가 클 수 있다.
   - Loss-based risk score를 dataset metadata와 함께 봐야 한다.

2. Model release risk를 capacity and data ratio로 설명하기.
   - 단순히 parameter count만 볼 것이 아니라, training data size와 token-per-parameter ratio를 같이 봐야 한다.

3. Extraction benchmark와 membership benchmark를 분리하기.
   - 둘은 다른 공격 surface다.
   - 둘 중 하나만으로 memorization을 결론내리면 위험하다.

4. Reference model을 사용한 sample-level memorization estimate.
   - Target model loss와 reference model loss의 차이를 통해 sample-specific information을 추정할 수 있다.

## 7-3. Follow-up papers

- Quantifying Memorization Across Neural Language Models.
- Extracting Training Data from Large Language Models.
- Membership Inference Attacks Against Machine Learning Models.
- Differential privacy and data reconstruction 관련 논문들.
- Dataset deduplication and contamination analysis 논문들.

# 8. Summary

- 이 논문은 memorization을 extraction 여부가 아니라 compression gain으로 정의한다.
- Intended memorization은 generalization이고, unintended memorization은 sample-specific information에 가깝다.
- Random data setting에서는 generalization을 제거해 GPT-style transformer capacity를 추정한다.
- 논문은 대략 3.6 bits per parameter 수준의 empirical capacity를 보고한다.
- Dataset size가 커질수록 average sample membership inference는 어려워지지만, rare or duplicated sample risk는 별도로 봐야 한다.
