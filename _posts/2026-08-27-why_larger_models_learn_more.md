---
layout: single
title: "Why Larger Models Learn More: Effects of Capacity, Interference, and Rare-Task Retention Review"
categories: Study-concept
tag: [LLM, ScalingLaws, RepresentationLearning, TrainingDynamics, DataMixture]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2605.29548)

Why Larger Models Learn More는 "큰 모델이 작은 모델보다 성능이 좋다"를 다시 말하는 논문이 아니다. 이 논문이 묻는 질문은 더 구체적이다. **왜 어떤 task는 작은 모델이 무한히 오래 학습해도 잘 배우지 못하는데, 큰 모델은 같은 data mixture 안에서 배울 수 있는가**다.

논문의 답은 model expressivity 하나가 아니다. 작은 모델도 특정 task를 표현할 수 있는 solution을 가질 수 있다. 문제는 data mixture 안에서 여러 task가 같은 neuron과 feature resource를 두고 경쟁한다는 데 있다. High-frequency 또는 low-complexity task가 먼저 representation capacity를 차지하고, rare 또는 complex task signal은 천천히 들어오다가 frequent task gradient에 의해 지워진다.

이 논문은 이를 rare-task retention과 gradient interference 문제로 설명한다. Larger model은 common task를 충분히 표현하고 나면 그 task의 residual gradient가 약해진다. 그러면 rare task가 드물게 등장할 때 생기는 update가 다음 rare observation까지 더 오래 남고, 여러 batch에 걸쳐 누적될 수 있다. Smaller model에서는 같은 update가 frequent task update에 의해 빠르게 overwrite된다.

> 한 줄 요약: 이 논문은 larger model의 장점을 단순 sample efficiency가 아니라 data mixture 내부의 capacity competition과 gradient interference 완화로 설명하고, synthetic multi-task regression과 OLMo 4M-4B pretraining injection 실험을 통해 rare task와 complex task가 왜 큰 모델에서 더 잘 보존되고 학습되는지 분석한다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같다.

- Scaling law를 benchmark curve가 아니라 data distribution의 tail learning 문제로 해석한다.
- "큰 모델은 더 잘 외운다"와 "큰 모델은 더 잘 일반화한다" 사이를 rare-task retention mechanism으로 연결한다.
- Capacity, frequency, complexity, interference를 하나의 toy model과 OLMo pretraining pipeline에서 함께 검증한다.
- Data mixture design과 model sizing을 연결하는 practical insight를 준다.
- Rare task upsampling, curriculum, replay, model size selection을 생각할 때 중요한 framing을 제공한다.

이 글에서는 이 논문을 scaling law paper라기보다, **model capacity가 data mixture 속 task competition을 어떻게 바꾸는지 설명하는 training dynamics paper**로 읽는다.

# 1. Problem Setting

## 1-1. Problem definition

LLM scaling에서 흔한 관찰은 큰 모델이 작은 모델보다 더 많은 task를 푼다는 것이다. 하지만 여기에는 여러 설명이 섞인다.

- 큰 모델이 더 많은 parameter로 더 복잡한 function을 표현한다.
- 큰 모델이 같은 data에서 더 sample efficient하다.
- 큰 모델이 rare pattern을 더 잘 memorize한다.
- 큰 모델이 feature를 더 다양하게 보존한다.
- 작은 모델은 frequent task에 capacity를 빼앗긴다.

이 논문은 마지막 두 가지를 특히 강조한다. 문제는 단순히 data가 부족한 것이 아니라, 같은 data mixture 안에서 task frequency와 complexity가 model capacity와 상호작용한다는 점이다.

Data distribution을 여러 task의 mixture로 생각해 보자.

$$
\mathcal{D}
=
\sum_{r=1}^{R}
\pi_r
\mathcal{D}_r
$$

여기서 $\pi_r$는 task $r$의 frequency다. Small model은 limited representation width를 갖고 있으므로 모든 task feature를 동시에 보존할 수 없다. 그러면 high-frequency 또는 high-utility feature부터 representation에 들어간다. Rare 또는 complex task feature는 lower utility로 밀릴 수 있다.

논문은 이 현상을 "larger models learn more"라고 표현한다. 여기서 more는 단순히 loss가 낮다는 뜻이 아니라, smaller model이 asymptotic training에서도 잘 배우지 못하는 distribution의 일부를 larger model이 배운다는 뜻이다.

## 1-2. Why previous explanations are insufficient

### 1) Expressivity alone is not enough

어떤 작은 모델도 target task를 표현할 수 있는 parameter setting을 가질 수 있다. 하지만 실제 training은 data mixture, gradient descent, task frequency에 의해 움직인다. 따라서 "표현 가능한가"와 "학습 과정에서 그 solution으로 가는가"는 다르다.

논문은 rare task failure가 solution absence 때문이 아니라, mixture training dynamics에서 나타날 수 있음을 보인다.

### 2) Sample efficiency alone is not enough

Larger model이 적은 data로 더 빨리 배우는 것은 흔한 설명이다. 하지만 논문은 더 강한 claim을 다룬다. Smaller model이 infinite training data regime에서도 도달하지 못하는 distribution의 일부가 있을 수 있다는 것이다.

즉 문제는 "더 많은 data를 주면 작은 모델도 결국 다 배울까"가 아니다. 특정 mixture에서는 data scaling만으로는 작은 model의 bottleneck이 사라지지 않을 수 있다.

### 3) Rare task frequency만으로는 부족하다

Rare task를 많이 보면 더 잘 배울 수 있다. 그러나 같은 global frequency를 유지하더라도 rare task instance 사이 gap이 길면 learning이 약해질 수 있다. Task signal이 한 번 들어왔다가 다음 observation 전에 frequent-task update로 사라지기 때문이다.

따라서 frequency뿐 아니라 retention gap과 interference가 중요하다.

# 2. Core Idea

## 2-1. Main contribution

논문의 핵심 contribution은 세 단계로 정리할 수 있다.

1. **Phenomenological scaling argument**
   - Power-law scaling 관점에서 data scaling으로 learnable한 부분과 model scaling으로만 learnable한 부분을 구분한다.
   - Larger model이 smaller model의 asymptotic limit 밖에 있는 distribution part를 배울 수 있음을 문제로 제시한다.

2. **Synthetic multi-task regression analysis**
   - Task frequency와 complexity가 다른 mixture를 만든다.
   - Width-limited encoder가 어떤 task feature를 보존하는지 분석한다.
   - Feature가 utility 순서대로 학습된다는 rule을 제시한다.
   - Larger width가 frequent-task interference를 줄이고 rare-task signal을 보존하게 함을 보인다.

3. **OLMo pretraining validation**
   - Dolma v1.7 pretraining corpus에 controlled special tasks를 주입한다.
   - OLMo 4M, 20M, 300M, 1B, 4B models를 학습한다.
   - Comparison task와 modular addition task에서 behavioral, representational, gradient evidence를 확인한다.

## 2-2. Design intuition

이 논문의 직관은 "capacity is competition relief"에 가깝다.

Small model에서는 여러 task가 같은 neuron과 feature를 두고 경쟁한다.

$$
\text{common task update}
+
\text{rare task update}
\rightarrow
\text{interference}
$$

Frequent task는 자주 gradient를 발생시키므로 representation을 계속 자기 방향으로 당긴다. Rare task는 드물게 나타나므로 update가 약하고 intermittent하다. Small model에서는 rare update가 다음 rare batch가 오기 전에 사라진다.

Large model에서는 common task를 설명하는 데 필요한 capacity가 충분하다. Common task의 residual error가 줄어들면 common-task gradient norm도 약해진다. 그러면 rare-task update가 완전히 덮어쓰이지 않고 남아, 다음 rare observation과 누적된다.

이 흐름을 간단히 쓰면 다음과 같다.

$$
\text{more capacity}
\rightarrow
\text{lower common-task residual}
\rightarrow
\text{weaker common-task gradient}
\rightarrow
\text{less rare-task overwriting}
\rightarrow
\text{rare-task retention}
$$

즉 larger model이 rare task를 배우는 이유는 rare task에 더 큰 gradient를 주기 때문만이 아니라, frequent task가 rare task를 덜 방해하게 만들기 때문이다.

# 3. Architecture / Method

## 3-1. Overview

| 항목 | 내용 |
| --- | --- |
| 핵심 질문 | 왜 큰 모델은 작은 모델이 못 배우는 task를 배우는가 |
| 핵심 메커니즘 | Capacity competition과 gradient interference 완화 |
| Synthetic setup | 여러 linear regression task의 mixture |
| 주요 변수 | Task frequency, task complexity, model width |
| 이론적 관점 | Feature utility와 residual gradient |
| 현실적 검증 | Injected task를 넣은 OLMo pretraining |
| 모델 크기 | 4M, 20M, 300M, 1B, 4B |
| 주입 task | Comparison과 modular addition |
| 증거 유형 | 행동 지표, representation, gradient 분석 |

## 3-2. Phenomenological model

논문은 먼저 scaling curve 관점에서 문제를 정의한다. Smaller model이 data를 더 많이 봐서 얻을 수 있는 loss reduction이 있고, model size를 키워야만 얻는 loss reduction이 있다.

이를 개념적으로 나누면 다음과 같다.

| 구간 | 의미 |
| --- | --- |
| Data scaling으로 학습 가능한 구간 | 작은 모델도 충분한 data와 compute가 있으면 접근 가능한 영역 |
| Model scaling으로만 학습되는 구간 | 작은 모델의 asymptotic regime 밖에 있어 큰 모델에서야 설명되는 영역 |
| Random baseline gap | Model이 task distribution을 거의 배우지 못한 영역 |

이 distinction은 중요하다. Larger model의 advantage를 단순 finite-data speedup으로 보지 않고, data mixture 일부가 smaller model의 learnable set 밖에 있을 수 있다는 질문으로 바꾼다.

## 3-3. Synthetic multi-task regression

Synthetic setup은 여러 linear regression task의 mixture다. 각 task는 frequency $\pi_r$와 covariance spectrum을 가진다. Spectrum이 천천히 decay하면 더 많은 direction이 필요하므로 task complexity가 높다.

Student model은 shared width-$k$ encoder와 task-specific linear decoder를 가진다. 이 구조에서 encoder가 어떤 feature direction을 보존하는지가 핵심이다.

논문의 Theorem 3은 features are learned in order of utility라고 요약할 수 있다. 각 task feature의 utility는 task frequency와 spectrum eigenvalue의 곱으로 이해할 수 있다.

$$
u_{r,i}
=
\pi_r
\lambda_{r,i}
$$

Width-$k$ encoder는 가장 큰 utility를 가진 $k$개 feature를 보존한다. 따라서 rare task나 complex task의 feature는 낮은 utility를 가지기 쉽고, small width에서는 representation에 들어가지 못한다.

이 관점에서 larger width가 사는 것은 단순 parameter count가 아니라 **lower-utility feature를 보존할 수 있는 slot**이다.

## 3-4. Frequency and complexity

Rare task는 $\pi_r$가 작아서 utility가 낮다. Complex task는 signal이 여러 direction으로 퍼져 있어서 later feature의 eigenvalue가 작다. 둘 다 lower utility feature를 만든다.

따라서 small model은 다음 순서로 배울 가능성이 크다.

1. 자주 나오고 단순한 task feature
2. 자주 나오지만 어느 정도 복잡한 task feature
3. 드물게 나오지만 단순한 task feature
4. 드물고 복잡한 task feature

물론 실제 order는 frequency만으로 결정되지 않는다. Appendix complexity sweep에서는 complexity gap이 커지면 frequency ordering이 깨질 수 있고, utility가 더 좋은 predictor가 된다.

## 3-5. Gradient interference and retention

Theorem 4는 common task gradient가 현재 representation이 아직 설명하지 못한 residual covariance를 통해 작용한다고 본다. Common task를 이미 충분히 설명하면 residual이 줄고 gradient도 약해진다.

Small model에서는 common-task residual이 계속 크므로 frequent task gradient가 rare task direction을 밀어낸다. Larger model에서는 common task feature를 충분히 담아 residual을 줄일 수 있고, rare task direction이 더 안정적으로 남는다.

Matched-frequency injection experiment가 이 intuition을 분리한다.

- Rare task를 일정 gap 동안 withheld한다.
- 이후 rare samples를 큰 batch로 inject하여 long-run frequency를 맞춘다.
- Global frequency는 같지만 observation gap이 다르다.
- Small model에서는 rare signal이 gap 사이에 거의 0에 가깝게 decay된다.
- Large model은 rare signal을 더 오래 retain하고 accumulate한다.

이 결과는 rare task learning이 단순 frequency뿐 아니라 update retention에 의존한다는 것을 보여준다.

## 3-6. OLMo pretraining pipeline

Realistic validation에서는 OLMo pipeline을 사용한다.

| 항목 | 내용 |
| --- | --- |
| Pretraining corpus | Dolma v1.7 |
| Training scale | 최대 210B tokens, 50K steps |
| Model sizes | 4M, 20M, 300M, 1B, 4B |
| Special tasks | Comparison과 modular addition |
| Task encoding | Three-token sequence, `TOK1`, `TOK2`, `LABEL` |
| Frequency control | Special task instance를 정해진 비율로 주입 |
| Main evidence | Loss, eval accuracy, representation feature, gradient interference |

두 injected task는 단순 memorization이 아니라 generalizable structure learning을 요구하도록 설계된다.

- Comparison task: token order와 number comparison feature 필요
- Modular addition task: Fourier mode structure 필요

논문은 이 feature를 representation에서 localize하고, model size와 task frequency에 따라 해당 feature가 얼마나 나타나는지 측정한다.

## 3-7. Representational evidence

Comparison task에서는 global order feature가 1-D subspace로 localize된다. Modular addition task에서는 Fourier modes가 residual stream에 나타난다.

논문은 task를 잘 배운 model에서 target feature를 먼저 찾고, 모든 model representation에서 이 feature가 얼마나 존재하는지 측정한다. Larger model과 higher task frequency에서 feature presence가 커지고, test accuracy와 강하게 관련된다.

이 분석의 의미는 behavioral accuracy가 단순 train instance memorization 때문인지, 실제 task-relevant representation이 생긴 것인지 분리하려는 데 있다.

## 3-8. Gradient evidence

논문은 task neurons를 first layer MLP에서 찾고, task reference direction과 batch gradient의 cosine similarity를 본다.

Gradient는 task token contribution과 non-task token contribution으로 나뉜다.

- Task token gradient가 task reference direction과 align하면 learning signal이다.
- Non-task token gradient가 task reference direction과 align하거나 anti-align하면 interference signal이다.

결과적으로 larger model에서는 task gradient direction이 더 안정적이고, non-task gradient interference가 작다. Smaller model에서는 batch gradient similarity가 더 noisy하고 unstable하게 나타난다.

# 4. Training / Data / Recipe

## 4-1. Synthetic data

Synthetic data는 multi-task linear regression mixture다.

- Task마다 frequency $\pi_r$가 다르다.
- Task마다 covariance spectrum이 다르다.
- Feature matrix는 orthonormal columns를 가진다.
- 서로 다른 task는 orthogonal block을 차지한다.
- Model width는 보존할 수 있는 feature direction 수를 제한한다.

이 setting의 장점은 무엇을 배웠는지 analytically 확인할 수 있다는 점이다. Feature utility가 explicit하게 계산되기 때문에 width, frequency, complexity의 효과를 분리하기 쉽다.

## 4-2. OLMo injected task data

OLMo experiment에서는 natural pretraining corpus에 special task를 삽입한다.

- Base corpus: Dolma v1.7
- Injected sequence: `TOK1`, `TOK2`, `LABEL`, end-of-document token
- Insert location: training sequence의 first four tokens를 대체
- Task instances: train/test split을 가진 controlled special task
- Frequency: high-frequency에서 rare injection까지 sweep

논문은 comparison과 modular addition task가 normal pretraining data에 우연히 포함될 가능성이 낮은 task라고 설명한다. 이를 통해 task frequency를 control한다.

## 4-3. Model sizes

OLMo model size는 다음 범위를 포함한다.

| Size | Note |
| --- | --- |
| 4M | Depth 8 |
| 20M | Depth 16 |
| 300M | Depth 16 |
| 1B | Depth 16 |
| 4B | Depth 16 |

Scaling은 hidden dimension, MLP dimension, attention head 수를 조정하는 방식이다. 4M model만 depth가 8이고, 나머지는 depth 16이다.

이 점은 해석에 중요하다. 모든 size difference가 pure width scaling은 아니다. Synthetic theory는 width 중심으로 설명되지만, OLMo experiment는 practical architecture scaling을 사용한다.

## 4-4. Training strategy

논문은 OLMo pipeline으로 최대 210B tokens와 50K steps까지 학습한다. Injected task frequency와 model size를 바꾸면서 다음을 본다.

1. Task training loss
2. Task eval accuracy
3. Representation feature presence
4. Gradient alignment와 interference

Behavioral result만 보면 larger model이 rare task를 더 잘 푸는지 알 수 있다. Representation과 gradient analysis는 왜 그런지 보기 위한 추가 evidence다.

## 4-5. Engineering notes

실무적으로 가져갈 수 있는 point는 다음과 같다.

1. **Rare task frequency를 평균만 보지 말 것**
   - 같은 frequency라도 observation gap이 길면 signal retention이 어려울 수 있다.

2. **Data mixture는 capacity allocation 문제다**
   - Frequent task가 representation slot을 차지할 수 있다.
   - Rare target capability가 있으면 frequency와 interference를 같이 봐야 한다.

3. **Upsampling은 scaling의 실제 대안이 될 수 있다**
   - 논문 discussion도 target task frequency를 올리는 것이 model size scaling보다 efficient할 수 있다고 말한다.

4. **Memorization이 abstraction을 도울 수 있다**
   - Rare task instance를 오래 보존해야 abstract task feature를 누적해서 배울 수 있다.

5. **Task feature 분석이 중요하다**
   - Accuracy만으로 rare task가 memorized됐는지 feature를 학습했는지 구분하기 어렵다.

6. **Small model failure가 항상 data shortage는 아니다**
   - More data만으로 해결되는 failure와 model capacity가 필요한 failure를 나눠야 한다.

# 5. Evaluation

## 5-1. Main synthetic results

Synthetic setup에서 결과는 utility ordering과 잘 맞는다.

- Width가 커질수록 lower-frequency task loss가 줄어든다.
- Feature utility가 learning order를 예측한다.
- Rare task와 complex task는 small width에서 representation에 들어가기 어렵다.
- Common task residual이 줄면 rare task signal retention이 좋아진다.

Matched-frequency injection experiment에서는 같은 long-run frequency라도 gap이 길수록 learning이 어려워진다. Smaller model에서 rare signal decay가 더 빠르고, larger model은 injection 사이에서 signal을 더 오래 유지한다.

## 5-2. OLMo behavioral evidence

OLMo experiment에서도 synthetic result와 같은 pattern이 나타난다.

- Larger model은 lower-frequency injected task를 더 잘 배운다.
- Task는 frequency order에 따라 학습되는 경향이 있다.
- Comparison task에서는 larger model과 higher frequency에서 test accuracy가 더 잘 올라간다.
- Modular addition에서도 task-relevant structure가 larger model에서 더 잘 나타난다.
- Larger model의 improvement는 train loss memorization뿐 아니라 eval accuracy로도 나타난다.

이 마지막 점이 중요하다. 논문은 rare task instance memorization이 generalizable structure learning으로 이어질 수 있음을 보여주려 한다.

## 5-3. Representational evidence

논문은 task feature가 representation에 나타나는 정도를 측정한다.

| Task | Feature |
| --- | --- |
| Comparison | 전체 순서를 나타내는 direction |
| Modular addition | Fourier modes |

Larger model과 higher task frequency에서 task feature가 더 빨리, 더 많이 나타난다. Feature presence는 test accuracy와 상관이 크다.

이는 rare task learning이 output behavior만의 문제가 아니라 internal representation feature acquisition과 연결되어 있음을 보여준다.

## 5-4. Gradient evidence

Gradient analysis는 task signal과 interference를 분리한다.

- Task reference direction은 task loss gradient를 aggregate해서 만든다.
- Batch gradient를 task token과 non-task token contribution으로 나눈다.
- Larger model은 task injection step에서 task direction과 더 안정적으로 align한다.
- Larger model에서는 non-task gradient가 task direction과 거의 orthogonal해져 interference가 줄어든다.
- Smaller model에서는 gradient direction이 unstable하고 rare task direction이 자주 덮인다.

이 결과는 논문의 central mechanism인 reduced gradient interference를 뒷받침한다.

## 5-5. What really matters in the experiments

### 1) Larger model은 tail task를 더 잘 배운다

이 논문에서 larger model의 advantage는 average loss보다 tail task learning으로 해석된다. Frequent task가 아니라 rare task와 complex task를 보존하는 능력이 중요하다.

### 2) Task frequency와 complexity는 상호작용한다

Rare task만 어려운 것이 아니다. Complex task는 여러 feature direction을 필요로 하므로 small width에서 lower-utility feature가 빠지기 쉽다.

### 3) Retention은 memorization과 generalization을 잇는다

Rare instance를 기억하는 것이 단순 overfitting이 아닐 수 있다. Signal을 다음 observation까지 유지해야 abstract structure를 누적 학습할 수 있다.

### 4) Data mixture design이 중요하다

Model size를 키우는 것만이 답은 아니다. Target capability frequency를 올리거나, observation gap을 줄이거나, interference를 낮추는 training schedule이 더 efficient할 수 있다.

### 5) Toy result는 유용하지만 충분하지 않다

Synthetic regression은 mechanism을 명확히 보이지만, real LLM pretraining의 task boundary, feature complexity, optimizer dynamics는 훨씬 복잡하다. OLMo validation이 이 gap을 줄이지만 완전히 없애지는 않는다.

# 6. Limitations

1. **Injected task setting은 인위적이다**
   - Comparison과 modular addition task는 controlled analysis에는 좋지만 natural task mixture를 완전히 대표하지 않는다.

2. **Task complexity를 정의하기 어렵다**
   - Synthetic task에서는 spectrum decay로 complexity를 조절할 수 있다.
   - Natural language task에서는 complexity를 이렇게 깨끗하게 정의하기 어렵다.

3. **OLMo scaling은 pure width scaling이 아니다**
   - Synthetic theory는 width 중심이지만 OLMo setting에서는 dimension, head, model scale이 함께 변한다.

4. **Rare task가 항상 좋은 signal은 아니다**
   - 모든 rare pattern을 보존하는 것이 좋은 것은 아니다.
   - Toxic, spurious, low-quality rare data도 있을 수 있다.

5. **Upsampling trade-off는 아직 완전히 해결되지 않았다**
   - Target task frequency를 올리면 rare task learning에는 좋을 수 있다.
   - 하지만 general distribution coverage나 overfitting에 영향을 줄 수 있다.

6. **Representation localization은 analysis tool에 의존한다**
   - DAS와 Fourier feature analysis는 task-specific하다.
   - General capability에 그대로 적용하기 어렵다.

7. **Gradient interference 분석 범위가 제한적이다**
   - First layer MLP task neurons 중심 analysis가 model 전체 dynamics를 완전히 설명하지는 않는다.

8. **Production LLM training을 위한 직접 recipe는 아니다**
   - Paper는 mechanism account에 가깝다.
   - 실제 data mixture design rule, sampling schedule, model size decision은 추가 work가 필요하다.

9. **Infinite data argument는 개념적 주장에 가깝다**
   - Smaller model의 asymptotic limitation을 scaling curve와 toy theory로 설명하지만, real training에서는 compute, optimizer, schedule, data order가 모두 섞인다.

10. **Benchmark task learning과 deployment capability는 다르다**
    - Injected comparison/modular addition을 배우는 것과 broad real-world skill acquisition은 다를 수 있다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문의 가장 중요한 메시지는 "큰 모델이 더 좋다"가 아니다. 더 중요한 것은 **data mixture에서 어떤 task가 representation slot을 차지하는지 봐야 한다**는 점이다.

LLM training에서 흔히 data quantity와 quality를 이야기하지만, 이 논문은 frequency와 interference를 더 세밀하게 본다.

- High-quality rare data가 있어도 model이 충분히 크지 않으면 보존되지 않을 수 있다.
- Common task가 이미 잘 학습되면 gradient가 약해져 rare task learning이 가능해질 수 있다.
- Rare task를 너무 sparse하게 넣으면 signal이 다음 exposure 전에 사라질 수 있다.
- Memorization은 무조건 나쁜 것이 아니라, abstraction으로 가는 bridge가 될 수 있다.

이 관점은 model sizing과 data mixture planning에 직접 연결된다.

## 7-2. Reuse potential

### Data mixture design

Target capability가 rare하다면 단순히 dataset에 포함시키는 것만으로 부족하다.

- Sampling frequency
- Injection gap
- Curriculum order
- Task complexity
- 관련 common task와의 interference
- Model capacity

를 함께 봐야 한다.

### Model sizing

Small model을 만들 때 "이 task가 표현 가능한가"보다 "이 data mixture에서 이 task signal이 살아남는가"를 물어야 한다. On-device 또는 edge model에서 rare but critical task를 요구하면, parameter budget뿐 아니라 data schedule을 설계해야 한다.

### Post-training

SFT나 RL data mixture에서도 비슷한 현상이 생길 수 있다. Frequent easy task가 gradient를 지배하면 rare high-value behavior가 사라질 수 있다.

가능한 strategy는 다음과 같다.

- Rare task replay
- Difficulty-aware sampling
- Gradient conflict detection
- Task-specific adapter나 routing
- Injection gap을 줄이는 data curriculum
- 평균 score가 아니라 tail capability 기준의 evaluation

### Continual learning

Rare-task retention은 continual learning의 interference problem과 닿아 있다. New task update와 old task retention만 문제가 아니라, frequent task update가 rare task signal을 계속 지우는 문제도 봐야 한다.

## 7-3. Follow-up papers

- Neural scaling laws와 Chinchilla scaling
- Transfer를 위한 scaling law
- Emergent abilities와 measurement artifact
- Grokking과 modular arithmetic
- Neural language model의 memorization
- LLM pretraining의 data mixture와 curriculum learning
- Multi-task learning의 gradient interference
- DAS 기반 representation localization
- Lottery ticket과 neural network의 capacity allocation

# 8. Summary

- Larger model이 더 많이 배우는 이유 중 하나는 lower-utility rare task와 complex task feature를 보존할 수 있기 때문이다.
- Small model은 어떤 task를 원리적으로 표현할 수 있더라도, mixture dynamics 아래에서는 그 task를 학습하지 못할 수 있다.
- Frequent task는 gradient interference를 통해 rare-task signal을 덮어쓸 수 있다.
- Larger model은 common-task residual을 줄이고, rare-task update가 누적될 만큼 오래 보존한다.
- 실무적으로는 model size, data frequency, task complexity, sampling gap을 함께 설계해야 한다.
