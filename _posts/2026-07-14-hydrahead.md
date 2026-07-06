---
layout: single
title: "HydraHead: From Head-Level Functional Heterogeneity to Specialized Attention Hybridization Review"
categories: Study-concept
tag: [LLM, Attention, LongContext, Architecture]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2606.20097)

HydraHead는 long-context Transformer를 다룰 때 꽤 중요한 질문을 던지는 논문이다. Full Attention은 precise retrieval에 강하지만 quadratic cost가 크다. Linear Attention은 long context scaling에는 좋지만, 고정 크기 state로 과거를 압축하기 때문에 세밀한 recall에서 약해질 수 있다. 그래서 최근에는 두 attention을 섞는 hybrid architecture가 많이 등장한다.

대부분의 hybrid model은 layer 단위로 full attention layer와 linear attention layer를 섞는다. HydraHead는 이 granularity가 너무 거칠다고 본다. 논문은 interpretability analysis를 통해 같은 layer 안의 head들이 서로 다른 기능을 수행하고, retrieval-critical head는 sparse하게 존재한다고 주장한다. 따라서 attention hybridization은 layer보다 head axis에서 하는 편이 더 자연스럽다는 것이 핵심이다.

> 한 줄 요약: HydraHead는 retrieval-critical head만 Full Attention으로 남기고 나머지 head를 Linear Attention 계열로 바꾸는 head-wise hybrid attention architecture로, long-context 효율과 retrieval fidelity 사이의 trade-off를 더 세밀하게 조정하려는 논문이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같다.

- Hybrid attention을 layer-wise design이 아니라 head-wise design space로 확장한다.
- Mechanistic interpretability를 architecture selection에 직접 연결한다.
- Full Attention과 Linear Attention output의 distribution mismatch를 scale-normalized fusion으로 다룬다.
- Pretrained Transformer를 hybrid model로 바꾸기 위한 three-stage transfer pipeline을 제안한다.
- Long-context retrieval과 general reasoning을 함께 유지해야 한다는 실무적 trade-off를 잘 보여준다.

이 논문은 "linear attention이 full attention을 대체할 수 있는가"를 묻지 않는다. 오히려 어떤 head는 full attention으로 남겨야 하고, 어떤 head는 linear attention으로 바꿔도 되는지를 묻는다.

# 1. Problem Setting

## 1-1. Problem definition

Long-context LLM의 attention 병목은 분명하다. Full Attention은 모든 token pair interaction을 계산하기 때문에 context length가 길어질수록 memory와 compute가 빠르게 증가한다. 특히 KV cache는 long-context inference에서 큰 부담이 된다.

Linear Attention이나 recurrent attention 계열은 이 문제를 줄인다. 그러나 full attention이 제공하는 precise token-level interaction을 완전히 대체하기는 어렵다. Long-context retrieval, multi-key recall, needle-in-a-haystack task에서는 특정 token을 정확히 찾아야 하므로 compression error가 성능으로 바로 드러난다.

그래서 hybrid attention이 등장한다. 일부 layer는 Full Attention으로 유지하고, 일부 layer는 Linear Attention으로 바꾸는 방식이다. 하지만 HydraHead는 layer-wise hybrid가 구조적으로 비효율적일 수 있다고 본다. 한 layer 안에도 retrieval에 중요한 head와 그렇지 않은 head가 섞여 있기 때문이다.

## 1-2. Why previous approaches are insufficient

기존 layer-wise hybrid의 한계는 세 가지다.

첫째, layer는 기능 단위가 너무 크다. 같은 layer 안에서도 head마다 역할이 다르다. 어떤 head는 retrieval-critical하고, 어떤 head는 거의 영향을 주지 않을 수 있다. Layer 전체를 Full Attention으로 남기면 불필요한 head까지 비싼 attention을 유지하게 된다.

둘째, Full Attention과 Linear Attention은 feature distribution이 다르다. Softmax attention은 query norm과 token-level interaction에 민감하고, sharper distribution을 만든다. Linear Attention은 normalization과 recurrence 구조 때문에 smoother output을 만들 수 있다. 이 둘을 단순히 concatenate하면 optimization이 불안정해질 수 있다.

셋째, pretrained Transformer를 hybrid model로 바꾸는 것은 단순 replacement 문제가 아니다. Linear branch는 full attention head와 다른 dynamics를 갖는다. Direct fine-tuning은 distribution mismatch와 optimization instability를 만들 수 있다.

# 2. Core Idea

## 2-1. Main contribution

HydraHead의 핵심 기여는 다음 세 가지로 정리할 수 있다.

1. Head-level functional heterogeneity 분석
   - Layer output은 depth 방향으로 상대적으로 smooth하게 변한다.
   - 반면 같은 layer 안의 individual head는 final output에 대한 contribution이 크게 다르다.
   - Retrieval-critical head는 sparse subset으로 나타난다.

2. Head-wise hybrid attention
   - 중요한 head는 Full Attention branch에 남긴다.
   - 나머지 head는 Linear Attention 계열인 Gated DeltaNet branch로 보낸다.
   - Attention mechanism을 layer axis가 아니라 head axis에서 섞는다.

3. Scale-normalized fusion과 transfer pipeline
   - FA head output과 LA head output을 독립적으로 normalize하고 head-wise scale을 학습한다.
   - Pretrained checkpoint를 활용해 parameter reuse, layer-wise alignment, global distillation, long-context fine-tuning을 거친다.

## 2-2. Design intuition

HydraHead의 직관은 간단하다.

> 모든 attention head가 같은 값어치를 갖지 않는다면, 모든 head에 같은 attention mechanism을 줄 필요도 없다.

Full Attention은 비싸지만 특정 head에서는 꼭 필요하다. 특히 long-context retrieval에서 needle을 정확히 찾는 head는 full token interaction을 유지해야 한다. 반면 retrieval-critical하지 않은 head는 long context scaling을 위해 Linear Attention으로 바꿀 수 있다.

이 구조는 mixture of attention mechanisms에 가깝다. 다만 token별 router를 쓰는 것이 아니라, pretrained model의 internal functional anatomy를 보고 head별 mechanism을 정한다. 그래서 HydraHead는 architecture search와 interpretability의 접점에 있는 논문이다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Full Attention의 retrieval fidelity와 Linear Attention의 long-context efficiency 결합 |
| Granularity | layer-wise가 아니라 head-wise hybridization |
| FA branch | retrieval-critical head 유지 |
| LA branch | 나머지 head를 Gated DeltaNet으로 변환 |
| Selection | activation patching 기반 interpretability score |
| Fusion | head-wise scale-normalized fusion |
| Transfer | parameter reuse, layer-wise alignment, global distillation, long-context fine-tuning |
| Evaluation | RULER NIAH, general reasoning benchmark |

## 3-2. Module breakdown

### 1) Head importance estimation

HydraHead는 먼저 어떤 head를 Full Attention으로 남길지 정해야 한다. 이를 위해 long-context retrieval task에서 activation patching 기반 causal importance를 계산한다.

논문은 head를 receiver role과 sender role로 나누어 contribution을 본다. 특정 head activation을 교체했을 때 correct answer token logit이 얼마나 떨어지는지를 측정하고, capability별 score를 normalize해 통합한다. 최종적으로 head importance score가 높은 head는 FA branch에 남고, 나머지는 LA branch로 간다.

이 방식은 단순 random allocation과 다르다. 논문은 global interpretability screening이 fixed allocation, layer-wise random, global random보다 RULER retrieval에서 훨씬 좋은 결과를 낸다고 보고한다.

### 2) Head-wise hybrid attention

HydraHead는 전체 head set을 두 subset으로 나눈다.

- $H_F$: Full Attention branch에 남는 head
- $H_L$: Linear Attention branch로 가는 head

$H_F$의 head는 standard softmax attention을 사용한다. $H_L$의 head는 Gated DeltaNet recurrence를 사용한다. 이렇게 하면 retrieval-critical head는 token-level interaction을 유지하고, 나머지 head는 long context 효율을 얻는다.

### 3) Scale-normalized fusion

FA output과 LA output은 통계가 다르다. FA는 sharp하고 query norm에 민감한 output을 만들 수 있다. LA는 smoother하고 recurrent state compression의 영향을 받는다. 이를 그대로 concatenate하면 특정 branch output이 다른 branch를 압도하거나, optimization이 불안정해질 수 있다.

HydraHead는 각 head output에 독립적인 normalization을 적용하고, head index를 유지한 채 concat한 뒤 learnable scale로 조정한다. 이 설계는 head의 functional identity를 보존하면서 heterogeneous attention output을 섞기 위한 장치다.

### 4) Efficient hybrid transfer learning

Pretrained Transformer를 hybrid architecture로 바꾸는 과정은 세 단계로 진행된다.

1. Parameter migration and layer-wise output alignment
   - FA branch는 pretrained functionality를 최대한 유지한다.
   - GDN branch는 original Q, K, V projection weight를 재사용해 초기 mismatch를 줄인다.
   - 각 layer output이 teacher layer output과 맞도록 align한다.

2. Global distillation
   - 전체 model output distribution이 original teacher model과 맞도록 distill한다.

3. Long-context fine-tuning
   - target long-context behavior에 맞게 fine-tuning한다.

이 pipeline은 LA branch가 처음부터 full attention과 다른 dynamics를 갖는 문제를 완화하기 위한 것이다.

# 4. Training / Data / Recipe

## 4-1. Data

주요 실험은 Qwen3-1.7B를 backbone으로 한다. Training data는 FineWeb-Edu를 사용한다. 기본 setting에서는 head의 25%를 Full Attention으로 유지하고, 나머지 75%를 GDN 구조로 사용한다.

Head selection calibration은 RULER benchmark의 NIAH sub-probe에서 만든다. Single-key와 multi-key retrieval task를 사용하고, 4K context에서 counterfactual pair를 만들어 head importance를 계산한다.

## 4-2. Training strategy

Training은 세 단계 transfer pipeline을 따른다.

- Stage 1: layer-wise output alignment
- Stage 2: global distillation
- Stage 3: long-context fine-tuning

이 구조는 pretraining을 처음부터 다시 하지 않고도 hybrid architecture로 이식하는 데 초점을 둔다. 즉 HydraHead의 실용성은 "새 model을 처음부터 학습"이 아니라 "기존 model을 hybrid로 변환"하는 데 있다.

## 4-3. Engineering notes

실무적으로 주의할 점은 다음과 같다.

1. Head selection은 one-time cost로 볼 수 있다.
   - Activation patching은 비싸지만, 한 checkpoint에서 좋은 head allocation을 찾으면 이후 여러 hybrid model에 재사용할 수 있다.

2. FA budget은 너무 작게 줄이면 안 된다.
   - 논문은 retrieval-critical head가 sparse하다고 보지만, 매우 aggressive한 FA reduction에서는 성능이 떨어질 수 있다.

3. Fusion normalization은 필수에 가깝다.
   - FA와 LA output distribution이 다르기 때문에 direct concatenation은 성능 손실을 만들 수 있다.

4. General reasoning과 long-context retrieval을 같이 봐야 한다.
   - Long-context score만 좋아지고 general reasoning이 무너지면 실용성이 낮다.

# 5. Evaluation

## 5-1. Main results

HydraHead는 RULER NIAH류 long-context retrieval과 general reasoning benchmark를 함께 평가한다. 논문은 16K, 32K, 64K, 128K, 256K context length를 보고, native context와 extended context를 나누어 해석한다.

주요 결과는 다음과 같다.

- Head-wise hybrid가 layer-wise hybrid보다 long-context retrieval에서 강하다.
- Global interpretability screening은 random allocation보다 훨씬 안정적인 head selection을 제공한다.
- 3:1 GDN-to-FA ratio뿐 아니라 7:1 같은 더 aggressive한 setting에서도 interpretability-guided selection이 중요하다.
- 15B tokens scale training 후, Qwen3-1.7B baseline 대비 512K context length에서 큰 improvement를 보고한다.
- General reasoning benchmark에서는 일부 hybrid model보다 long-context와 reasoning의 균형이 좋다고 주장한다.

## 5-2. What really matters in the experiments

HydraHead의 실험에서 핵심은 세 가지다.

1. Head granularity가 실제로 의미 있는가
   - Random head allocation이 아니라 interpretability-guided allocation이 성능을 크게 좌우한다.
   - 이는 head-level functional heterogeneity가 architecture design에 쓸 수 있는 signal임을 보여준다.

2. Fusion module이 필요한가
   - FA와 LA output을 naive하게 섞으면 retrieval fidelity가 떨어진다.
   - Scale-normalized fusion은 heterogeneous branch를 안정적으로 합치는 역할을 한다.

3. Long-context gain이 general reasoning loss로 바뀌지 않는가
   - Efficient attention model은 long context만 잘하고 general capability가 무너질 수 있다.
   - HydraHead는 RULER와 general benchmark를 같이 제시해 이 trade-off를 보려 한다.

이 논문의 좋은 점은 interpretability를 post-hoc analysis로 끝내지 않는다는 것이다. Head importance를 실제 architecture allocation으로 바꾼다. 이 방향은 efficient LLM design에서 점점 중요해질 가능성이 있다.

# 6. Limitations

1. 실험 scale이 제한적이다.
   - 주요 backbone은 Qwen3-1.7B이고, scaling experiment도 15B tokens 수준으로 제한된다.
   - Larger LLM에서 같은 head specialization pattern이 유지되는지는 추가 확인이 필요하다.

2. Head selection procedure가 비싸다.
   - Full activation patching은 head별 forward pass가 필요하다.
   - 1.7B scale에서는 가능하지만 frontier-scale model에는 부담이 커질 수 있다.

3. Calibration task 의존성이 있다.
   - NIAH sub-probe로 찾은 retrieval-critical head가 다른 capability에도 최적인지는 보장되지 않는다.
   - Domain-specific capability마다 head selection을 다시 해야 할 수 있다.

4. Minimal FA budget과 interpretability score 사이에 gap이 있다.
   - 논문은 매우 적은 head가 중요하다고 분석하지만, 실제 FA budget을 너무 줄이면 성능이 저하된다.
   - Importance estimation만으로 deployable budget을 완전히 결정하기 어렵다.

5. LA variant가 GDN 중심이다.
   - 다른 linear attention이나 SSM 계열에도 generalize되는지는 추가 실험이 필요하다.

# 7. My Take

## 7-1. Why this matters for my work

HydraHead는 long-context architecture를 설계할 때 매우 실용적인 질문을 던진다. Attention layer를 통째로 바꾸는 것이 아니라, head별로 어떤 attention mechanism이 필요한지 볼 수 있다면 훨씬 세밀한 cost-performance trade-off를 만들 수 있다.

특히 retrieval-heavy document AI나 codebase assistant에서는 모든 head가 full attention일 필요는 없을 수 있다. 하지만 특정 retrieval head는 반드시 full attention으로 남겨야 할 수 있다. HydraHead는 이 판단을 data-driven하게 만들려는 시도다.

## 7-2. Reuse potential

재사용 가능성이 큰 부분은 다음과 같다.

- Long-context model conversion pipeline
- Retrieval-critical head selection
- Full attention budget allocation
- Attention mechanism mixture design
- Interpretability-driven pruning 또는 compression

실무 적용에서는 전체 HydraHead architecture를 바로 쓰기보다, 먼저 기존 model에서 retrieval-critical head를 찾아보는 diagnostic tool로 활용할 수 있다. 어떤 layer와 head가 실제 retrieval에 기여하는지 알면 KV cache compression이나 sparse attention 설계에도 도움이 된다.

## 7-3. Follow-up papers

- Gated DeltaNet 계열 linear attention 논문
- HypeNet: Hybrid Attention Transformer
- Liger 또는 layer-wise hybrid attention 계열 논문
- RULER: Long-context benchmark
- Mechanistic interpretability 기반 attention head analysis 논문

# 8. Summary

- HydraHead는 hybrid attention의 granularity를 layer에서 head로 바꾼다.
- Retrieval-critical head는 Full Attention으로 유지하고, 나머지는 Linear Attention branch로 보낸다.
- Interpretability-guided head selection이 random allocation보다 훨씬 강한 결과를 낸다.
- Scale-normalized fusion은 FA와 LA output distribution mismatch를 줄이는 핵심 모듈이다.
- Long-context 효율과 retrieval fidelity를 동시에 다루려는 architecture design으로 읽을 만하다.
