---
layout: single
title: "Proving and Improving the Platonic Representation Hypothesis with JEPAs Review"
categories: Study-concept
tag: [JEPA, Representation-Learning, Multimodal-Alignment]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2610.39302)

- 제목: Proving and Improving the Platonic Representation Hypothesis with JEPAs
- 저자: Randall Balestriero, Yann LeCun
- 공개일: 2026-10-29
- 형식: arXiv preprint. 정식 학회 채택과 동일하게 취급하지 않음.
- [Project / code](https://github.com/galilai-group/prh-proof)

> 한 줄 요약: 서로 다른 modality의 모델이 같은 세계를 관찰할 때, 어떤 positive pair로 학습하는지가 표현 공간의 정렬을 결정한다. 논문은 이상화된 JEPA 설정에서 조건부 정리를 제시하고, good-view 학습이 next-token prediction보다 훨씬 적은 샘플로 시각 표현과 정렬될 수 있음을 실험한다.

이 연구가 중요한 이유는 '모델을 키우면 서로 다른 modality의 표현이 자연스럽게 비슷해진다'는 관찰을 학습 목표의 설계 문제로 바꾸기 때문이다.

Platonic Representation Hypothesis(PRH)는 서로 다른 관찰자가 동일한 세계를 학습하면 어떤 공통의 표현에 접근할 수 있다는 가설이다. 이 연구는 모델 크기만 늘리는 접근 대신 **어떤 두 관찰을 같은 대상으로 묶어 주는가**에 주목한다.

주의할 점은 제목의 'Proving'을 과도하게 읽지 않는 것이다. 저자가 제시한 정리는 특정한 세계 모델, 관찰 함수, 목적함수, 최적화 가정을 전제로 한다. 오늘날의 모든 LLM과 VLM이 동일한 세계 모델을 학습한다는 보편적인 증명이 아니다.

# 1. Problem Setting

## 1-1. 기존 PRH의 질문

이미지 인코더와 텍스트 인코더가 각각 따로 학습되었더라도 두 표현 공간에는 유사한 구조가 관찰될 수 있다. 예를 들어 같은 장면의 이미지와 설명문을 서로 대응시킬 때, 표현을 적절하게 회전시켜 일치시키는 것이 가능할 수 있다.

하지만 단순히 두 공간이 비슷하다는 관찰로는 다음 질문에 답하기 어렵다.

- 어떤 학습 목표에서 정렬이 일어나는가?
- 데이터가 많아질수록 왜 정렬이 좋아지는가?
- 서로 다른 modality의 표현이 완전히 같지 않은 이유는 무엇인가?
- 학습 단계에서 무엇을 바꾸면 정렬을 더 효율적으로 만들 수 있는가?

이 연구는 핵심 요인을 positive view의 구성으로 좁힌다.

## 1-2. 세계 상태와 표면적 표현의 차이

하나의 장면을 두 사람이 다른 문장으로 설명했다고 하자. 장면에 포함된 대상과 행동은 공유되지만, 단어 선택과 문장 길이는 다를 수 있다. 이미지에서도 촬영 시점, 빛, 배경 등이 달라진다.

여기서 저자는 두 종류의 정보를 구분한다.

- World state: 장면이나 사건 자체를 설명하는 공통 요인
- Modality-specific nuisance: 같은 장면을 서로 다르게 표현하는 방식

모달리티 간 정렬이 목표라면, 모델이 nuisance에 많은 차원을 사용하지 않는 것이 유리하다. 그러나 항상 nuisance가 쓸모없는 정보인 것은 아니다. 실제 영상 제어나 세밀한 인식 과제에서는 시점과 조명이 중요할 수 있다. 이 논문의 nuisance는 **cross-modal alignment 목표에 한정된 개념**이다.

# 2. Core Idea

## 2-1. Platonic Representation Theorem의 설정

논문은 유한한 세계 상태 집합 $\mathcal{Z}$와 상태 분포 $p(z)$를 가정한다. Modality $m$은 세계 상태를 관찰 $x=g_m(z)$로 변환한다. 먼저 $g_m$이 injective라고 가정하므로, 서로 다른 세계 상태가 하나의 관찰로 합쳐지지 않는다.

Modality마다 encoder $f_m$이 있고, 다음과 같은 형태의 목적함수를 최소화한다.

$$
\mathcal{L}_m(f) = \mathbb{E}_{(z,z') \sim \pi_m}
\left[\ell(f(g_m(z)), f(g_m(z')))\right] + \Omega(f)
$$

- $\pi_m$: 양성 쌍을 뽑는 세계 상태의 joint distribution
- $\ell$: 두 표현을 일치시키는 prediction 또는 alignment term
- $\Omega$: 모든 입력을 같은 벡터로 보내는 collapse를 방지하는 항

핵심은 모든 modality가 동일한 $\pi$를 사용하며, 목적함수의 최적해가 회전 변환 이외에는 유일하다는 조건이다. 이때 각 modality의 최적 표현은 하나의 공통 표현과 직교 변환 관계가 된다.

$$
f_m^*(g_m(z)) = Q_m h^*(z), \qquad Q_m^\top Q_m = I
$$

이 식은 정리의 핵심적인 결론을 요약한 표기다. 여기서 $h^*(z)$는 공통 세계 표현이고 $Q_m$은 modality별 회전 또는 반사에 해당하는 orthogonal map이다.

**중요한 조건:** 공유 positive, injective rendering, population optimum, rotation을 제외한 최적해의 유일성이 필요하다. 정리는 특정 조건에서 같은 문제를 서로 다른 좌표계로 푼 결과라는 해석에 가깝다.

## 2-2. Positive pair와 spectral graph

논문은 positive pair가 세계 상태 위에 하나의 그래프를 만든다고 해석한다. 다음과 같이 정규화한 positive-pair kernel을 정의한다.

$$
K_{\pi} = D^{-1/2} \Pi D^{-1/2}
$$

여기서 $\Pi$는 $\pi(z,z')$의 행렬 표현이고 $D$는 $p(z)$를 대각 원소로 갖는 행렬이다. Whitened second-order anti-collapse 조건에서는 JEPA의 표현이 이 그래프의 주요 eigenvector와 연결된다. 고유값 사이에 충분한 spectral gap이 있으면 중요한 표현 부분공간이 안정적으로 결정된다.

서로 다른 modality가 다른 positive-pair kernel을 사용하면 정렬 오차가 발생한다. 저자는 이 오차가 kernel의 불일치와 spectral gap에 의해 제한됨을 보인다. 직관적으로는 두 modality가 무엇을 서로 가깝다고 간주하는지가 비슷할수록 표현도 정렬되기 쉽다.

## 2-3. Good view와 bad view

이 논문의 가장 실용적인 설계 선택이다.

| 구분 | Positive pair의 예시 | 공유되는 정보 |
| --- | --- | --- |
| Good view | 하나의 이미지에 대해 서로 다른 두 caption | 공통 장면의 의미 |
| Bad view | 하나의 caption에서 잘라낸 두 span | 장면 의미와 특정 문장의 표현 방식 |
| Next-token prediction | 문서의 prefix와 continuation | 문맥, 문체, 단어 연결 등 |

Good view는 동일한 세계 상태를 바라보는 별도의 관찰을 결합한다. Bad view는 같은 관찰의 일부를 변형하기 때문에 단어 선택과 같은 nuisance를 함께 유지할 수 있다.

따라서 같은 양의 데이터라도 good-view objective는 장면 수준의 의미를 더 직접적으로 학습할 수 있다는 것이 연구진의 주장이다.

## 2-4. Next-token prediction을 JEPA로 읽는 방식

저자는 일반적인 next-token loss를 다음처럼 분해한다.

$$
-\log p_{\theta}(y\mid c)
= - h_{\theta}(c)^\top w_y
+ \log \sum_{v \in V} \exp(h_{\theta}(c)^\top w_v)
$$

첫 항은 문맥 $c$와 다음 토큰 $y$의 표현을 맞추고, 두 번째 항은 vocabulary 전체에 대한 softmax log-partition이다. 이 구조를 통해 next-token prediction을 prefix와 continuation에 기반한 넓은 의미의 JEPA로 해석한다.

단, 엄밀한 구분이 필요하다. 이 목적함수의 대수적 분해가 위의 spectral JEPA 정리의 모든 가정을 충족시킨다는 의미는 아니다. 실제 원문도 next-token prediction에 대한 nuisance 설명을 정리의 직접적인 따름정리가 아니라, 실험으로 확인할 가설적 해석으로 제시한다.

# 3. Architecture / Method

## 3-1. 실험 구조의 전체 그림

핵심 실험에서는 서로 독립적으로 학습한 image encoder와 text encoder를 비교한다. 학습 과정에서 image-text matching loss를 직접 사용하지 않는다는 점이 중요하다.

- Image side: 주로 ViT 계열 encoder와 DINOv2 등 기존 vision encoder
- Text side: BERT 기반 JEPA와 같은 구조의 next-token baseline
- Alignment evaluation: 별도 모델 학습 없이 nearest-neighbor 구조 비교, 또는 한 개의 orthogonal map을 이용한 retrieval

평가할 때만 paired validation 자료로 회전 행렬을 맞춘다. 따라서 '학습 중 cross-modal pair를 쓰지 않는다'와 '모든 평가와 응용에서 paired data를 전혀 쓰지 않는다'는 별개 주장이다.

## 3-2. 세 가지 텍스트 학습 방식

Flickr30k 설정에서 텍스트 encoder의 주요 구조는 12-layer BERT-S, width 384, 약 33M parameters다.

1. Good-view JEPA: 같은 이미지의 서로 다른 caption을 positive로 선택한다.
2. Bad-view JEPA: 하나의 caption에서 서로 다른 span을 만들고 masking한다.
3. Next-token baseline: 같은 BERT 구조를 causal attention으로 바꾸고 다음 토큰을 예측한다.

Good-view와 bad-view JEPA에서는 global view 2개와 local view 6개를 구성한다. 텍스트 view는 일정 길이의 연속 span을 선택하고 일부 토큰을 masking한다. Good-view는 서로 다른 caption에서 해당 view를 만들고, bad-view는 같은 caption에서 만든다.

이처럼 architecture와 원본 데이터를 유사하게 통제하여 positive 구성의 차이를 보려는 설계다.

## 3-3. Anti-collapse와 학습의 안정성

JEPA는 서로 다른 입력을 모두 같은 벡터로 보내도 alignment loss가 낮아질 수 있다. 이 collapse를 방지하기 위해 논문은 LeJEPA의 SIGReg를 활용한다. Flickr 기본 설정에서 invariance term과 SIGReg를 사용하며, SIGReg 계수는 0.02다.

하지만 실제 실험은 이론에서 정확한 최적해 유일성을 증명한 whitened second-order objective와 동일하지 않다. SIGReg의 전체 분포 제약에 대한 동일한 유일성까지 증명한 것은 아니라는 점을 기억해야 한다.

또한 일부 설정에서는 학습이 collapse되는 현상이 나타난다. 원문 ablation은 gradient clipping과 더 낮은 learning rate가 안정성에 중요하다고 보고한다.

## 3-4. 표현 정렬을 어떻게 평가하는가

원문에서 사용한 대표 지표는 다음과 같다.

- Mutual 10-NN: 서로 다른 표현 공간에서 10개 이웃의 겹침 정도를 측정한다. 별도의 회전 행렬 fitting이 필요하지 않다.
- Retrieval R@1: validation image-caption pair를 통해 회전 행렬을 추정한 뒤, test에서 올바른 caption을 첫 번째로 찾는 비율이다.
- Linear CKA: 두 표현의 유사성을 평가하는 다른 지표로, 후속 대규모 비교에 사용한다.

첫 번째는 학습된 표현의 구조적 유사도를, 두 번째는 적은 수의 paired data로 실제 결합이 가능한지를 본다는 점에서 성격이 다르다.

# 4. Training / Data / Recipe

## 4-1. Flickr30k 실험

Flickr30k의 약 29k개 학습 이미지와 이미지당 5개 caption을 사용한다. 기본 실험은 image ViT-S/16과 text BERT-S를 각각 학습하며 다음 설정을 따른다.

- Training: 200 epochs, batch size 256
- Optimizer: AdamW
- Learning rate: 0.0005, warmup 후 cosine decay
- Weight decay: 0.05
- Random seed: 주요 비교는 3개 seed 평균
- Text views: 연속 span 선택 및 일부 masking

일부 대규모 또는 장기 학습에서는 learning rate를 0.0001로 낮추고 gradient clipping을 적용한다. 이 세부사항을 생략하면 학습의 안정성이나 샘플 효율의 차이를 정확하게 비교하기 어렵다.

## 4-2. 웹 규모의 caption 데이터

대규모 실험에서는 Recap-DataComp 이미지에 연결된 텍스트를 사용한다. 이미지별 alt-text와, 이미지 captioner가 생성한 2개의 recaption을 서로 다른 view로 활용한다.

- 소규모부터 50M, 100M 이미지 등으로 확장하는 실험을 포함한다.
- 가장 큰 비교에서는 약 940M 이미지에 연결된 텍스트를 사용한다.
- 주요 모델은 109M-parameter BERT-B 기반 text JEPA다.
- 같은 구조와 텍스트를 이용한 bad-view JEPA, next-token baseline도 비교한다.

여기에는 중요한 사실이 있다. 이미지 인코더와 텍스트 인코더를 직접 짝지어 공동 학습하지는 않았지만, good view로 사용한 텍스트 자체는 이미지 정보를 활용하는 captioner가 생성한 부분이 있다. 따라서 광범위한 의미에서 어떤 시각 정보도 들어가지 않았다고 해석하면 안 된다.

## 4-3. 실험 구현에서 특히 중요한 요소

연구진의 ablation에서는 caption 전체를 쓰는 것보다 적절히 trimming하는 방식이 정렬에 유리했다. 동일한 장면을 설명하는 문장들이 실제로 공유하는 정보에 집중하도록 만드는 효과로 해석한다.

하지만 '좋은 positive pair'를 얻으려면 같은 장면에 속하는 서로 다른 관찰을 알아야 한다. 실무에서는 view grouping의 품질, caption 생성기의 편향, 의미상 비슷하지만 다른 장면을 혼동하는 문제를 함께 제어해야 한다.

# 5. Evaluation

## 5-1. 같은 데이터에서 학습 목표를 비교

Frozen DINOv2 ViT-B/14와의 Flickr caption retrieval에서, validation 1k pair로 회전을 fit한 결과는 다음과 같다. 같은 논문의 Table 2에 보고된 설정이다.

| Text encoder | R@1, rotation fit on 1k pairs |
| --- | ---: |
| Good-view JEPA, BERT-S | 0.246 |
| Bad-view JEPA | 0.133 |
| Next-token prediction | 0.118 |

성능 차이는 단지 모델이 더 크기 때문이 아니다. 비교군은 동일 계열의 encoder와 caption 데이터를 사용하고 positive 구성 및 objective를 바꾼다. 수치의 단위는 0-1 범위의 R@1이며, 백분율로 바꾸면 각각 24.6%, 13.3%, 11.8%다.

이 결과는 특정한 Flickr retrieval 조건을 평가한 것이므로, 모든 언어 이해 과제에서 JEPA가 next-token model보다 우수하다는 뜻은 아니다.

## 5-2. 42배 샘플 효율 주장의 정확한 의미

같은 Recap 계열의 텍스트를 사용한 비교에서, good-view JEPA는 약 21.6M training samples로 WIT alignment 0.109에 도달한다. Next-token prediction은 약 900M samples에서 0.111에 도달한다.

두 수치가 비슷한 수준의 alignment이므로 저자는 약 42배의 sample efficiency 차이를 보고한다. 하지만 다음 점도 함께 봐야 한다.

- 한 번의 good-view sample은 여러 텍스트 view를 사용하므로 token 수가 같지 않다.
- 저자가 계산한 text-token 기준 차이는 약 14배다.
- 두 학습 run은 각 규모에 맞는 recipe와 learning rate, batch size를 사용한다.
- 비교 지표는 WIT alignment이며 언어 생성 정확도가 아니다.

따라서 '동일 연산량에서 범용 LLM보다 42배 학습 효율이 좋다'로 바꾸어 쓰면 부정확하다.

## 5-3. 109M text JEPA와 72B LLM의 비교

PRH benchmark의 WIT-1024 mutual 10-NN으로 보면 다음과 같다.

| Text representation | WIT mutual 10-NN |
| --- | ---: |
| Good-view JEPA, 109M | 0.184 |
| Next-token baseline, 109M | 0.111 |
| Bad-view JEPA, 109M | 0.088 |
| Qwen2.5-72B | 0.154 |
| Llama-3-70B | 0.151 |
| CLIP B/32 text | 0.124 |

Good-view JEPA의 수치가 가장 높지만, 이것은 언어 생성 또는 instruction following 벤치마크가 아니다. 서로 독립적으로 학습된 vision encoder와 representation geometry가 얼마나 잘 맞는지를 비교한 결과다.

또한 원문은 recaptioner의 image encoder에 CLIP이 사용됐다는 점, 일부 WIT 텍스트가 Recap alt-text에 존재한다는 점, seed 및 SIGReg weight에 따른 결과 차이도 별도로 밝힌다.

## 5-4. 실제 응용은 가능한가

저자는 image encoder, text JEPA, text-only 학습 decoder 사이에 회전 행렬 하나를 맞추어 image captioning을 시도한다. 30k Flickr pair로 fit한 map을 사용한 최종 구성에서 COCO CIDEr 0.838을 보고한다.

그러나 이를 'image-text pair 없이 captioner를 만들었다'고 간단히 표현하면 틀릴 수 있다. Encoder와 decoder는 image-text pair에 대한 공동 학습을 하지 않았지만, **회전 행렬은 Flickr image-caption pair로 추정했다.** Pair-free alignment 실험은 별도 설정이다.

Pair-free 실험에서는 unpaired geometry로 map을 찾고 ImageNet-100 top-5 64.4%를 보고한다. 이 수치는 논문이 비교한 언어 모델 기준보다 높지만, 같은 표의 CLIP B/32 text encoder 68.7%보다는 낮다. 비교 대상과 지표를 정확히 구분해야 한다.

# 6. Limitations

## 6-1. 이론과 실제 모델 사이의 거리

1. 정리는 유한 세계, injective rendering, matched marginal, population optimum 등 강한 가정을 사용한다.
2. 실제 이미지나 텍스트는 세계 정보를 일부 잃으므로, injectivity 가정이 직접 성립한다고 보기 어렵다.
3. 최적해의 rotation-uniqueness는 특정 second-order 설정에서 증명된다. 실제 실험에서 사용한 LeJEPA의 전체 분포 목적함수에 대한 동일한 보장은 없다.
4. 실제 이미지 쪽은 DINOv2처럼 한 이미지의 crop을 positive로 사용하는 경우가 많아, 양쪽 모두 이론상의 good-view 조건을 충족하지는 않는다.

## 6-2. 데이터와 평가의 한계

- Good views를 구성하기 위해 같은 장면을 설명하는 텍스트들을 묶는 정보가 필요하다.
- 웹 규모 데이터에는 image-conditioned recaptioning의 지식이 간접적으로 반영된다.
- 큰 모델과 baseline 비교에서 token budget, pretraining 목적, 모델 사용 목적이 완전히 같은 것은 아니다.
- 실험 중 일부 설정에서는 collapse가 발생했고, scaling law는 extrapolation 범위에서 과대 예측하기도 한다.
- 높은 cross-modal alignment는 compositional reasoning, robustness, 일반적 world modeling 능력의 보증이 아니다.

# 7. My Take

## 7-1. 모델 크기보다 학습 쌍을 먼저 설계하기

가장 재사용하기 좋은 기여는 '더 큰 모델을 학습할까?'보다 '현재 positive pair가 실제로 공유해야 하는 의미를 반영하는가?'를 먼저 묻게 한다는 점이다.

특히 문서와 이미지처럼 하나의 대상에 여러 설명이나 시점이 존재하는 데이터에서는 view의 정의가 표현 품질을 크게 바꿀 수 있다. 이것은 단순한 data augmentation 선택이 아니라 학습하고자 하는 불변성의 정의다.

## 7-2. 실무 적용을 위한 검증 설계

VLM 또는 document representation을 학습한다면 다음 비교가 유의미하다.

1. 같은 문서의 서로 다른 설명, 같은 페이지의 서로 다른 crop, 같은 장면의 서로 다른 caption을 구분한다.
2. Positive의 의미가 실제로 같은지 사람이 검수한 작은 집합을 확보한다.
3. 동일한 tokenizer, text-token budget, encoder 크기로 good-view와 bad-view를 비교한다.
4. Cross-modal retrieval뿐 아니라 OCR fidelity, fine-grained grounding, hard negative 구분 능력을 별도로 측정한다.
5. Good-view training이 세밀한 문서 정보나 특정 어휘를 잃게 만들지 않았는지 평가한다.

이것은 원문에 이미 증명된 응용이 아니라, 연구 결과를 멀티모달 데이터 파이프라인으로 확장하는 제안이다.

## 7-3. 후속으로 읽을 연구

- [Position: The Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987): 기존 가설의 관찰적 출발점
- [LeJEPA: provable and scalable self-supervised learning without the heuristics](https://arxiv.org/abs/2511.08544): 실제 실험에 사용된 학습 목표의 배경
- [Shared geometry as a rosetta stone: cross-modal alignment without paired data](https://arxiv.org/abs/2610.09411): pair-free alignment 비교의 배경

# 8. Summary

- 이 연구는 Platonic Representation Hypothesis를 이상화된 JEPA 설정에서 조건부 정리로 다룬다.
- 다른 modality가 같은 세계 상태의 관계를 positive로 학습하면 표현이 회전 변환을 제외하고 정렬될 수 있다.
- 같은 문장의 일부보다 같은 장면에 대한 서로 다른 설명을 묶는 good-view 설계가 중요하다.
- 109M text JEPA가 일부 대형 LLM보다 높은 cross-modal alignment를 보였으나, 이는 범용 언어 능력 비교가 아니다.
- 다음 연구 과제는 좋은 view를 현실 데이터에서 구성하고, 이론상의 조건과 실무 성능 사이의 간극을 검증하는 것이다.
