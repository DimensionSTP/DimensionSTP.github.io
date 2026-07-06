---
layout: single
title: "VALUEFLOW: Toward Pluralistic and Steerable Value-based Alignment in Large Language Models Review"
categories: Study-concept
tag: [LLM, Alignment, Values, Steering]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2602.03160)

[Code link](https://github.com/AIDASLab/VALUEFLOW)

VALUEFLOW는 alignment를 preference ranking만으로 보기 어렵다는 문제에서 출발한다. 많은 RLHF와 preference optimization은 "어떤 답변이 더 좋은가"를 학습한다. 하지만 실제 인간 가치에는 여러 가치 체계가 있고, 같은 가치도 얼마나 강하게 표현할지에 따라 모델 출력이 달라진다.

이 논문은 value-based alignment를 extraction, evaluation, steering이 연결된 infrastructure 문제로 본다. 가치 category를 뽑는 것만으로는 부족하고, 가치의 hierarchy, intensity, cross-theory relation, controllable steering까지 함께 다루어야 한다는 입장이다.

> 한 줄 요약: VALUEFLOW는 hierarchical value embedding space인 HIVES, value intensity anchor database인 VIDB, anchor-based evaluator를 결합해 LLM output의 value intensity를 평가하고, pluralistic value를 조절 가능한 steering target으로 다루려는 framework다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같다.

- Alignment를 average preference가 아니라 pluralistic value profile 문제로 본다.
- Value가 존재하는지만 보는 것이 아니라, 어느 정도 intensity로 표현되는지를 평가하려 한다.
- 여러 value theory를 하나의 embedding space와 anchor evaluation pipeline으로 연결한다.
- Steering을 단순 prompt instruction이 아니라 calibrated intensity control 문제로 다룬다.
- OpinionQA 같은 behavior prediction task와 연결해 demographic value profile 기반 generation을 실험한다.

이 논문은 "모델을 착하게 만들자"보다 훨씬 구체적이다. 어떤 가치 체계의 어떤 value를 어느 강도로 표현하게 할 것인지, 그리고 그것을 어떻게 측정할 것인지가 핵심이다.

# 1. Problem Setting

## 1-1. Problem definition

LLM alignment에서 human values를 다루는 일은 어렵다. 이유는 세 가지다.

첫째, 인간 가치는 하나가 아니다. 문화, 정치 성향, 종교, 공동체, profession, risk preference에 따라 중요하게 여기는 value가 다르다. 하나의 average preference로 모든 사용자를 대표하기 어렵다.

둘째, value는 flat label이 아니다. Value theory는 대개 hierarchy를 갖는다. 예를 들어 broad value category 아래에 더 세부적인 sub-value가 있고, 서로 다른 theory가 같은 행동을 다른 vocabulary로 설명할 수 있다.

셋째, value는 binary presence가 아니라 intensity를 갖는다. 어떤 답변이 fairness를 언급했는지보다, 얼마나 강하게 fairness 관점을 밀고 있는지가 중요할 수 있다. Alignment system이 실제로 steerable하려면 intensity control이 필요하다.

VALUEFLOW는 이 세 문제를 하나의 pipeline으로 묶는다.

## 1-2. Why previous approaches are insufficient

기존 preference-based alignment는 유용하지만 한계가 있다.

첫째, preference pair는 deeper motivational principle을 직접 표현하지 않는다. A가 B보다 낫다는 label은 알 수 있지만, 그 차이가 care, loyalty, autonomy, authority, equality 중 무엇 때문인지 분리하기 어렵다.

둘째, value extraction이 hierarchy를 무시하면 value space가 납작해진다. 그러면 비슷한 value와 다른 value를 구분하기 어렵고, cross-theory mapping도 약해진다.

셋째, value evaluator가 presence만 보면 steering quality를 검증하기 어렵다. "benevolence가 있음"과 "benevolence가 매우 강하게 표현됨"은 다른 상태다.

넷째, multi-value steering은 composition 문제를 만든다. 두 value를 동시에 올리면 서로 강화될 수도 있고, 충돌할 수도 있다. VALUEFLOW는 이런 asymmetry와 composition law를 study 대상으로 삼는다.

# 2. Core Idea

## 2-1. Main contribution

VALUEFLOW의 핵심 구성은 세 가지다.

1. HIVES
   - Hierarchical value embedding space다.
   - Intra-theory value hierarchy와 cross-theory alignment를 함께 반영한다.
   - Qwen3-Embedding-0.6B 위에서 학습된 embedding model로 공개되어 있다.

2. VIDB
   - Value Intensity DataBase다.
   - Value-labeled texts와 intensity estimate를 포함하는 anchor resource다.
   - Full set과 filtered set이 구분되어 있으며, filtered set은 LLM rating과 human rating이 더 일치하는 subset이다.

3. Anchor-based evaluator
   - Open-ended response를 VIDB anchor panel과 함께 ranking한다.
   - Ranking 결과를 Plackett-Luce utility optimization으로 intensity score로 바꾼다.
   - Output intensity range는 repository README 기준으로 [-10, 10] 형태다.

이 세 구성요소가 합쳐져 extraction, evaluation, steering을 연결한다.

## 2-2. Design intuition

VALUEFLOW의 설계 직관은 anchor-based measurement다. Open-ended response 하나만 보고 "이 답변의 value intensity는 7점"이라고 직접 판단하기 어렵다. 대신 이미 intensity가 추정된 anchor text들과 함께 상대 ranking을 만들면 더 안정적인 score를 얻을 수 있다.

이 구조는 다음과 같이 이해할 수 있다.

$$
response + anchor panel -> ranking -> utility score -> value intensity
$$

또 하나 중요한 직관은 cross-theory embedding이다. Value theory는 여러 개다. PVQ, Moral Foundations Theory, duty, rights 같은 이론은 서로 다른 vocabulary와 hierarchy를 가진다. HIVES는 이 heterogeneity를 shared semantic space로 연결하려 한다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | pluralistic value alignment의 extraction, evaluation, steering 통합 |
| Embedding | HIVES hierarchical value embedding |
| Database | VIDB value-labeled texts with intensity estimates |
| Evaluator | VIDB anchor panel ranking 기반 intensity scoring |
| Steering | value profile과 intensity level 기반 generation |
| Value theories | PVQ, MFT, duty, rights 계열 |
| Example task | OpinionQA generation and evaluation |
| Key issue | value presence가 아니라 calibrated intensity |

## 3-2. Module breakdown

### 1) Value categorization

Repository README 기준으로 VALUEFLOW의 data creation은 value categorization 단계에서 human-LLM collaboration을 사용한다. 각 text는 value theory hierarchy에 mapping된다. 각 level에서 7개 LLM panel이 best category에 vote하고, 충분한 agreement가 있으면 label을 채택한다. Agreement가 애매하면 neutral option을 다시 prompt한다.

이 과정은 value label을 단일 flat label로 끝내지 않고 root-to-leaf path로 만든다. 즉 value가 어느 hierarchy를 타고 내려갔는지가 보존된다.

### 2) HIVES embedding model

HIVES는 hierarchical value embedding model이다. Repository는 Qwen3-Embedding-0.6B 위에서 학습되었다고 설명한다.

Training은 두 stage로 구성된다.

1. Intra-theory alignment
   - 같은 theory 안에서 hierarchy prefix와 direction label을 공유하는 text를 가깝게 만든다.

2. Inter-theory and anchor alignment
   - Cross-theory anchors와 value instances를 InfoNCE objective로 연결한다.

이 설계는 하나의 value theory 내부 구조와 여러 value theory 간 mapping을 동시에 다루기 위한 것이다.

### 3) VIDB

VIDB는 evaluator와 steering의 기준점 역할을 한다. Value-labeled corpus와 intensity estimate를 담고, evaluation 시 anchor set으로 쓰인다.

중요한 점은 VIDB가 단순 label dataset이 아니라 intensity database라는 것이다. 모델 output을 평가할 때 해당 output을 VIDB anchor와 함께 ranking하고, anchor relative position을 통해 calibrated intensity를 계산한다.

### 4) Anchor-based evaluator

Evaluator는 open-ended response를 target value에 대해 평가한다. README 기준 workflow는 다음과 같다.

1. Target value와 theory를 정한다.
2. Response와 VIDB anchor를 함께 ranking window에 넣는다.
3. Judge model이 ranking을 만든다.
4. 여러 window의 ranking을 모은다.
5. Plackett-Luce utility optimization으로 intensity score를 계산한다.

이 방식은 direct score prompting보다 안정적인 상대 비교를 기대할 수 있다. 다만 judge model bias와 anchor sampling strategy에 의존한다는 점은 주의해야 한다.

### 5) Steering and profiling prompts

VALUEFLOW repository는 demographic value profile을 사용해 OpinionQA generation과 evaluation을 수행하는 script를 제공한다. Qwen3, Phi-4, GLM-4 계열 preset이 있고, value relevance score와 threshold를 사용해 profile prompt를 구성한다.

이 부분은 VALUEFLOW가 단순 evaluator가 아니라 steering pipeline까지 연결되어 있음을 보여준다. 즉 value profile을 입력으로 하고, model output이 해당 profile을 얼마나 잘 반영하는지 평가한다.

# 4. Training / Data / Recipe

## 4-1. Data

VIDB는 여러 source corpus 위에 구성된다. Repository license section은 MFRC, Social Chemistry, ValueNet, ValueEval, ValuePrism 등 source corpora와 license를 언급한다. 이 점은 실제 사용에서 중요하다. ValueNet은 non-commercial restriction이 있고, ValuePrism은 별도 agreement가 필요하다.

Filtered VIDB는 LLM과 human rating이 agree하는 curated subset으로 공개되어 있다. Full set은 README 기준으로 link placeholder가 남아 있어, 사용 전 release 상태를 확인해야 한다.

## 4-2. Training strategy

HIVES training은 크게 두 단계다.

- Stage 1: intra-theory hierarchical contrastive learning
- Stage 2: inter-theory and anchor alignment with InfoNCE

Evaluation은 model training이 아니라 ranking-based scoring이다. Response 하나와 여러 anchor text를 judge model에게 비교하게 하고, ranking 결과를 utility optimization으로 intensity score에 mapping한다.

Steering은 demographic profile prompt와 value relevance를 사용한다. 예를 들어 OpinionQA setting에서는 profile prompt directory, relevance CSV, theory, prompt count, threshold가 key argument가 된다.

## 4-3. Engineering notes

실무적으로 VALUEFLOW에서 가져갈 만한 점은 다음과 같다.

1. Value evaluator는 direct scoring보다 anchor comparison이 더 안정적일 수 있다.
   - 특히 open-ended response에서는 absolute score보다 상대 ranking이 더 쉬울 수 있다.

2. Value theory를 명시해야 한다.
   - 같은 텍스트도 PVQ, MFT, duty, rights theory에서 다르게 해석될 수 있다.

3. Intensity control은 prompt만으로 끝나지 않는다.
   - Steering 후 평가까지 같은 value space에서 닫혀 있어야 한다.

4. Dataset license를 반드시 확인해야 한다.
   - Alignment resource는 source corpus license가 deployment 가능성을 좌우할 수 있다.

5. Judge model과 anchor sampling을 logging해야 한다.
   - Evaluator가 ranking 기반이면 sampling method, ranking window size, judge model choice가 score에 영향을 준다.

# 5. Evaluation

## 5-1. Main results

arXiv abstract 기준으로 논문은 10개 model과 4개 value theory에 걸쳐 large-scale study를 수행한다. 주요 분석 대상은 value steerability의 asymmetry와 multi-value control의 composition law다.

이 결과를 해석할 때 중요한 점은 VALUEFLOW가 single safety score를 내는 benchmark가 아니라는 것이다. 모델이 어떤 value를 얼마나 잘 올리고 내릴 수 있는지, 여러 value를 동시에 조절할 때 어떤 interaction이 생기는지를 본다.

Repository는 text value evaluation script를 제공하며, target value, intensity level, judge model, value theory, anchor sampling strategy 등을 설정할 수 있다. 즉 논문 결과는 framework claim뿐 아니라 재사용 가능한 evaluation pipeline으로 이어진다.

## 5-2. What really matters in the experiments

VALUEFLOW에서 진짜 봐야 할 것은 세 가지다.

1. Value intensity calibration
   - 단순히 value가 언급되었는지보다, response가 anchor set 대비 어느 정도 강한 intensity를 갖는지가 중요하다.

2. Cross-theory consistency
   - 여러 value theory를 하나의 system에서 다루려면 theory 간 anchor alignment가 필요하다.
   - HIVES의 cross-theory embedding quality가 framework 전체의 기반이 된다.

3. Multi-value interaction
   - 하나의 value를 올리는 것은 비교적 쉽다.
   - 여러 value를 동시에 제어하면 충돌, saturation, asymmetry가 생길 수 있다.

이 논문은 alignment evaluation을 "좋은 답변 점수"에서 "value vector measurement"로 바꿔 읽게 만든다. 이 차이는 pluralistic alignment에서 매우 크다.

# 6. Limitations

1. Value theory 선택이 결과를 좌우한다.
   - PVQ, MFT, duty, rights가 모든 문화적 가치 체계를 포괄하지는 않는다.

2. Anchor database의 품질이 evaluator 품질이다.
   - VIDB label, intensity estimate, source corpus bias가 그대로 평가에 반영될 수 있다.

3. Judge model bias가 남는다.
   - Ranking-based evaluator도 결국 judge model의 판단에 의존한다.

4. Intensity score가 실제 사용자 만족과 같지는 않다.
   - 특정 value intensity를 높였다고 해서 답변이 전체적으로 더 바람직해지는 것은 아니다.

5. Dataset license와 commercial use 제약이 중요하다.
   - VIDB source corpus 중 일부는 non-commercial 또는 별도 agreement가 필요하다.

6. Full paper 수치 확인이 필요하다.
   - 현재 확인한 공개 abstract와 repository README만으로는 model별 steering success 수치와 composition law의 세부 결과를 모두 검증하기 어렵다.

# 7. My Take

## 7-1. Why this matters for my work

VALUEFLOW는 LLM alignment를 product personalization과 연결해서 생각하게 만든다. 실제 서비스에서는 모든 사용자가 같은 answer style과 value priority를 원하는 것이 아니다. 어떤 사용자는 autonomy를 강하게 원하고, 어떤 사용자는 safety나 conformity를 더 중요하게 볼 수 있다.

이때 필요한 것은 단순 preference optimization이 아니라 value profile을 읽고, 특정 value intensity를 조절하고, 조절 결과를 다시 평가하는 closed loop다. VALUEFLOW는 이 loop의 infrastructure를 제안한다.

## 7-2. Reuse potential

재사용 가능성이 큰 영역은 다음과 같다.

- Persona-aware assistant evaluation
- Domain-specific policy alignment
- Cultural adaptation benchmark
- Debate or deliberation system의 value diversity 측정
- Multi-agent system에서 value drift monitoring
- User profile 기반 answer style control

실무 적용에서는 VALUEFLOW를 그대로 steering system으로 쓰기보다, 먼저 evaluator로 쓰는 것이 안전해 보인다. 특정 model output이 어떤 value direction으로 치우치는지 측정하고, 이후 prompt 또는 activation steering을 설계하는 순서가 낫다.

## 7-3. Follow-up papers

- VISPA: Pluralistic Alignment via Automatic Value Selection and Activation
- COUPLE: Counterfactual Reasoning for Steerable Pluralistic Value Alignment
- ValueFlow: Measuring the Propagation of Value Perturbations in Multi-Agent LLM Systems
- Operationalizing Pluralistic Values in LLM Alignment
- Constitutional AI and value-based alignment 계열 논문

# 8. Summary

- VALUEFLOW는 value-based alignment를 extraction, evaluation, steering pipeline으로 통합한다.
- HIVES는 hierarchy와 cross-theory relation을 담는 value embedding space다.
- VIDB는 value-labeled text와 intensity estimate를 담는 anchor database다.
- Anchor-based evaluator는 response를 VIDB panel과 비교해 calibrated intensity score를 만든다.
- Pluralistic alignment에서는 value presence보다 value intensity와 multi-value interaction이 더 중요하다.
