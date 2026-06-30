---
layout: single
title: "PerceptionDLM: Parallel Region Perception with Multimodal Diffusion Language Models Review"
categories: Study-concept
tag: [PerceptionDLM, DiffusionLM, VLM, RegionCaptioning, MultimodalAI]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2606.19534)

[Code link](https://github.com/MSALab-PKU/PerceptionDLM)

[Model collection](https://huggingface.co/collections/MSALab/perceptiondlm-model-zoo)

PerceptionDLM은 diffusion language model이 multimodal perception에서 왜 쓸모 있을 수 있는지 꽤 직접적으로 보여주는 논문이다. 핵심 질문은 다음이다. **한 이미지 안의 여러 region을 caption해야 할 때, 꼭 region마다 autoregressive decoding을 따로 돌려야 하는가?**

기존 region captioning이나 localized perception model은 대개 autoregressive decoding에 의존한다. Region mask가 1개면 괜찮다. 하지만 한 이미지에서 5개, 10개 region을 동시에 설명해야 한다면 비용이 region 수에 거의 선형으로 늘어난다. 각 region description을 token-by-token으로 생성하고, region별로 independent pass를 돌리기 때문이다.

PerceptionDLM은 diffusion language model의 병렬 denoising 구조를 사용한다. 여러 region의 answer span을 모두 mask로 두고, structured attention mask와 region prompting을 통해 서로 다른 region description을 동시에 생성한다. 즉 sequence level과 token level 양쪽에서 parallelism을 얻으려 한다.

> 한 줄 요약: PerceptionDLM은 LLaDA-style discrete diffusion language backbone을 multimodal visual instruction tuning으로 강화한 뒤, region prompting, RoI-aligned feature replay, structured attention masking을 결합해 여러 image region의 captions를 한 denoising process에서 병렬 생성하는 diffusion VLM framework다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같다.

- Diffusion LM의 병렬 생성 장점을 concrete VLM perception task에 연결한다.
- Multi-region captioning에서 AR decoding의 region-wise latency growth를 정면으로 겨냥한다.
- PerceptionDLM-Base를 먼저 만들어 open-source diffusion VLM baseline을 강화한다.
- ParaDLC-Bench를 만들어 caption quality와 inference efficiency를 함께 평가한다.
- Structured attention masking으로 multiple region outputs의 interference를 줄이는 설계를 제시한다.
- "Diffusion LM은 느리다"는 일반 인식과 다르게, dense perception에서는 parallelism이 practical advantage가 될 수 있음을 보여준다.

이 글에서는 PerceptionDLM을 "diffusion VLM 하나 더"보다, **multi-target perception task에서 non-autoregressive decoding이 어디서 실제 이득을 줄 수 있는지 보여주는 논문**으로 읽는다.

# 1. Problem Setting

## 1-1. Problem definition

Region captioning task는 image $I$와 여러 region masks $\{R_i\}_{i=1}^{N}$가 주어졌을 때, 각 region에 대한 description $\{y_i\}_{i=1}^{N}$를 생성한다.

Autoregressive pipeline은 보통 region마다 별도 generation을 수행한다.

$$
y_i
=
f_{\mathrm{AR}}(I,R_i)
$$

전체 비용은 region 수 $N$에 따라 증가한다.

$$
\mathrm{Cost}_{\mathrm{AR}}
\approx
\sum_{i=1}^{N}
\mathrm{Cost}(y_i)
$$

PerceptionDLM은 여러 region output을 하나의 denoising problem으로 본다.

$$
\{y_i\}_{i=1}^{N}
=
f_{\mathrm{DLM}}(I,\{R_i\}_{i=1}^{N})
$$

즉 multiple masked response spans를 동시에 denoise한다. Region 수가 늘어나도 latency가 linear하게 증가하지 않도록 하는 것이 목표다.

## 1-2. Why previous approaches are insufficient

### 1) AR region captioning

Autoregressive model은 각 region caption을 sequentially generate한다. Token-level sequentiality와 region-level sequentiality가 함께 붙는다. Dense scene understanding, robotics perception, visual editing interface, image annotation system에서는 multiple region을 한 번에 처리해야 하므로 bottleneck이 된다.

### 2) General VLMs

General MLLM은 image-level QA에는 강하지만, fine-grained region-level perception에는 부족할 수 있다. Region identity, mask grounding, per-region disentanglement가 필요하다.

### 3) Existing diffusion VLMs

Diffusion VLM은 token-level parallelism을 제공할 가능성이 있지만, 기존 모델은 perception quality가 약하거나 multi-region parallel captioning을 직접 설계하지 않았다. PerceptionDLM은 먼저 PerceptionDLM-Base를 통해 strong diffusion VLM baseline을 만들고, 그 위에 region parallelism을 붙인다.

# 2. Core Idea

## 2-1. Main contribution

PerceptionDLM의 contribution은 세 가지다.

1. **PerceptionDLM-Base**
   - SigLIP-2 vision encoder, connector, LLaDA-8B diffusion decoder를 사용한다.
   - Four-stage visual instruction tuning으로 open diffusion VLM baseline을 강화한다.

2. **Parallel region perception architecture**
   - Region prompting으로 target region identity를 넣는다.
   - RoI-aligned feature replay로 local visual features를 answer span에 연결한다.
   - Structured attention masking으로 region outputs끼리 interference를 줄인다.

3. **ParaDLC-Bench**
   - DLC-Bench를 multiple region mask per image setting으로 확장한다.
   - Quality와 efficiency를 같이 본다.
   - Sequential AR baseline과 diffusion parallelism을 비교한다.

## 2-2. Design intuition

DLM은 mask token들을 동시에 예측할 수 있다. 이 특성이 모든 task에서 자동으로 유용한 것은 아니지만, multi-region perception은 자연스러운 parallel structure를 가진다.

각 region caption은 같은 image global context를 공유하지만, target region은 다르다. 따라서 다음 조건이 필요하다.

- Region identity가 명시적으로 주어져야 한다.
- Region-local visual evidence가 model에 다시 제공되어야 한다.
- 서로 다른 region의 caption이 서로 섞이면 안 된다.
- Shared global image context는 계속 사용할 수 있어야 한다.

PerceptionDLM의 architecture가 바로 이 네 조건을 맞춘다.

# 3. Architecture / Method

## 3-1. Overview

| 항목 | 내용 |
| --- | --- |
| Goal | 여러 region caption을 병렬 생성 |
| Base LM | LLaDA-8B diffusion language model |
| Vision encoder | SigLIP-2 |
| Connector | Two-layer MLP with GELU |
| Base training | Four-stage visual instruction tuning |
| Region mechanism | Region prompting과 RoI-aligned feature replay |
| Interference control | Structured attention masking |
| Benchmark | ParaDLC-Bench |
| Main benefit | Region quality를 유지하면서 dense-region throughput 개선 |

## 3-2. PerceptionDLM-Base

PerceptionDLM-Base는 discrete diffusion LM을 multimodal visual instruction tuning으로 확장한다.

Image $X_v$, instruction $X_q$, answer $X_a$가 주어졌다고 하자.

1. Vision encoder가 visual feature를 추출한다.

$$
Z_v
=
\Phi_v(X_v)
$$

2. Connector가 이를 language embedding space로 mapping한다.

$$
H_v
=
\Phi_c(Z_v)
$$

3. DLM decoder는 $H_v$와 $X_q$에 condition하면서 response token을 denoise한다.

Training은 image feature나 instruction token이 아니라 target response token에만 diffusion corruption을 적용한다.

$$
\mathcal{L}_{\mathrm{base}}
=
\mathbb{E}
\left[
\sum_{i \in M_a}
-\log
p_{\theta}
(x_i^0
\mid
x_t,H_v,X_q)
\right]
$$

Model은 dynamic resolution을 사용한다. Raw aspect ratio와 resolution에 따라 image를 512 x 512 tile로 나누고, optional thumbnail도 사용한다. Pixel unshuffle로 visual token 수를 줄인다.

## 3-3. Four-stage training

PerceptionDLM-Base는 four-stage training pipeline을 사용한다.

| Stage | 목적 |
| --- | --- |
| Stage 1 | Bee-Training-Data-Stage1을 사용한 vision-language alignment |
| Stage 2 | Bee-Training-Data-Stage2를 사용한 large-scale multimodal middle-stage training |
| Stage 3 | LLaVA-OneVision-1.5-Instruct-Data의 22M sample instruction tuning |
| Stage 4 | Honey-Data-15M을 사용한 high-quality SFT refinement |

핵심은 이 논문이 parallel region architecture만 추가한 것이 아니라, 먼저 더 강한 diffusion VLM baseline을 만든다는 점이다.

## 3-4. Region prompting

각 region $R_i$에 대해 PerceptionDLM은 learnable region embedding $e_i$를 추가한다. 이 embedding은 spatially broadcast되어 masked region에 해당하는 visual token에 fuse된다.

이렇게 해야 model이 여러 simultaneous target을 구분할 수 있다.

Region prompting이 없으면 multiple masked output이 같은 visual evidence를 두고 경쟁하면서 region identity를 혼동할 수 있다.

## 3-5. RoI-aligned feature replay

각 region mask에 대해 localized visual feature를 vision encoder에서 추출하고, language embedding space로 project해 placeholder token으로 넣는다. 이 RoI feature들은 해당 output region 근처에 replay된다.

이렇게 하면 각 caption span이 global image token만 보고 모든 것을 복구할 필요 없이 direct local evidence를 얻는다.

## 3-6. Structured attention masking

핵심 challenge는 cross-region interference다. 모든 region의 denoising token이 서로 자유롭게 attend하면, caption content가 region 사이에 leak될 수 있다.

Structured attention masking은 region-specific output token의 attention을 제한한다. 개념적으로 region $i$에 속한 token은 다음에 attend할 수 있다.

- Global image와 instruction context
- 해당 region의 prompt와 RoI feature
- 해당 region의 answer token
- 다른 region의 answer token에는 attend하지 않음. 단, 허용된 global context를 통한 indirect sharing은 가능

이 구조는 하나의 parallel process 안에 independent per-region denoising lane을 만든다.

# 4. Training / Data / Recipe

## 4-1. Data

Training data는 open-source multimodal instruction dataset을 포함한다.

| Data | 용도 |
| --- | --- |
| Bee-Training-Data-Stage1 | 초기 vision-language alignment |
| Bee-Training-Data-Stage2 | large-scale middle-stage training |
| LLaVA-OneVision-1.5-Instruct-Data | 22M sample instruction tuning |
| Honey-Data-15M | high-quality SFT refinement |

Evaluation에서는 ParaDLC-Bench가 DLC-Bench를 multiple region masks per image setting으로 확장한다.

## 4-2. Training strategy

Training은 diffusion LM에 맞춰 조정된다.

- Image feature와 prompt token은 condition으로 유지된다.
- Response token은 mask되고 denoise된다.
- Region-level training은 multiple mask와 structured attention을 사용한다.
- Model은 모든 region caption을 하나의 process에서 output하도록 학습된다.

## 4-3. Engineering notes

1. **Parallelism에는 structure가 필요하다**
   - DLM parallel decoding만으로는 충분하지 않다.
   - Region identity와 attention isolation이 필요하다.

2. **Base diffusion VLM quality가 중요하다**
   - Base perception이 강해야 parallel region captioning이 작동한다.

3. **Cross-region leakage를 막아야 한다**
   - Structured attention masking이 핵심이다.

4. **Efficiency와 quality를 함께 평가해야 한다**
   - Region caption quality가 competitive하게 유지될 때만 faster multi-region captioning이 유용하다.

5. **Benchmark는 region count를 늘려 평가해야 한다**
   - Single-region benchmark로는 latency benefit을 보여줄 수 없다.

# 5. Evaluation

## 5-1. Base diffusion VLM result

논문은 PerceptionDLM-Base가 16개 multimodal benchmark 중 15개에서 LLaDA-V를 outperform한다고 보고한다. 이는 더 강한 open diffusion VLM baseline이라는 주장을 뒷받침한다.

Fine-grained number를 인용하기 전에는 original table에서 exact benchmark list와 per-benchmark value를 확인해야 한다.

## 5-2. ParaDLC-Bench

논문은 PerceptionDLM이 ParaDLC-Bench에서 average accuracy 62.4%를 달성해 LLaDA-V의 35.2%를 거의 두 배로 넘긴다고 보고한다. 이는 diffusion backbone만이 아니라 region-aware architecture가 중요함을 보여준다.

논문은 또한 strong AR region-specific model과 비교해 competitive accuracy를 보이며 inference efficiency를 개선한다고 주장한다.

## 5-3. Throughput and parallelism

Introduction은 5 masks per image 같은 dense perception scenario에서 up to 3.5x throughput speedup을 보고한다. Figure 1도 4 masks per image의 constant workload 아래 up to 3.44 throughput speedup을 보고한다.

핵심 결과는 PerceptionDLM이 region count 증가에 따른 linear latency growth를 피한다는 점이다. Multi-region task에서는 이것이 region-by-region AR processing 대비 직접적인 advantage가 된다.

## 5-4. What really matters in the experiments

### 1) Region count가 stress test다

Region이 하나뿐이면 diffusion parallelism의 advantage는 작다. 여러 simultaneous region target이 있을 때 이 방법이 중요해진다.

### 2) Quality는 competitive하게 유지되어야 한다

Parallelism만으로는 부족하다. Caption quality가 AR region-specific model에 가깝게 유지되어야 한다.

### 3) Output이 자연스럽게 병렬일 때 DLM이 유용하다

이 논문은 DLM parallel decoding이 workload structure와 직접 맞는 task를 찾았다. Non-autoregressive generation에 대한 추상적 주장보다 더 설득력 있다.

### 4) 차이를 만드는 것은 structured attention이다

Naive parallel denoising은 region content leakage를 만들 수 있다. Structured mask가 multiple caption을 disentangle하는 핵심이다.

# 6. Limitations

1. **Task-specific advantage다**
   - PerceptionDLM은 multi-region perception에서 가장 설득력 있다.
   - 모든 VLM task에서 DLM이 우월하다는 증거는 아니다.

2. **Region mask가 주어진다고 가정한다**
   - Task는 given mask에서 시작한다.
   - Open-world region proposal과 grounding은 별도 문제다.

3. **Benchmark novelty가 있다**
   - ParaDLC-Bench는 저자들이 새로 도입한 benchmark다.
   - Community adoption과 independent validation이 필요하다.

4. **Decoder iteration cost가 있다**
   - DLM generation은 여전히 denoising step이 필요하다.
   - Latency advantage는 region count, block length, implementation에 의존한다.

5. **Structured mask complexity**
   - Attention mask design은 기존 serving stack에 구현하기 어려울 수 있다.

6. **AR baseline도 task-specific하다**
   - Matched scale과 prompt 아래 strong AR model과 주의 깊게 비교해야 한다.

7. **Caption independence issue가 생길 수 있다**
   - 일부 region description은 cross-region relational reasoning이 필요할 수 있다.
   - Strict independence mask는 이런 interaction을 제한할 수 있다.

8. **Model size와 data scale이 크다**
   - PerceptionDLM은 LLaDA-8B backbone과 large multimodal training data를 사용한다.
   - Smaller deployment는 별도 validation이 필요하다.

9. **Evaluation detail을 다시 확인해야 한다**
   - ParaDLC-Bench scoring, mask count, throughput hardware가 중요하다.

10. **Human quality evaluation이 필요하다**
    - Caption quality는 automated accuracy metric만으로 완전히 포착되지 않는다.

# 7. My Take

## 7-1. Why this matters for my work

PerceptionDLM의 핵심은 "diffusion VLM도 가능하다"보다, **non-autoregressive generation이 실질적으로 유리한 visual workload를 찾았다는 점**이다.

많은 DLM paper는 general parallel token generation을 주장한다. PerceptionDLM은 구체적인 case를 보여준다. Multiple region captions는 jointly generate할 만큼 독립적이지만, 같은 image context를 공유한다. 이는 diffusion decoding에 자연스럽게 맞는다.

## 7-2. Reuse potential

### Dense annotation

Image annotation tool은 많은 box나 mask에 대한 caption이 필요하다. Sequential region captioning은 비싸다. PerceptionDLM-style parallel denoising은 latency를 줄일 수 있다.

### Robotics perception

Robot scene understanding은 object, affordance, region에 대한 parallel description이 필요할 수 있다. Mask나 proposal이 있다면 parallel captioning이 유용하다.

### Image editing interface

Editing system은 여러 masked region을 이해해야 하는 경우가 많다. Joint region perception은 더 빠른 interactive UI를 지원할 수 있다.

### Document와 chart understanding

Region이 chart element, table cell, document block에 해당한다면 parallel localized captioning이 structure extraction에 도움이 될 수 있다.

## 7-3. Production considerations

- Upstream mask quality가 중요하다.
- Region 간 relational question에는 fallback이 필요하다.
- Target mask count에서 throughput을 측정해야 한다.
- Region prompt index collision을 test해야 한다.
- Attention mask implementation을 최적화해야 한다.
- Caption verification에는 per-region grounding check를 포함해야 한다.

## 7-4. Follow-up papers

- LLaDA
- LLaDA-V
- Dream
- iLLaDA
- DLC-Bench
- GAR와 DAM region captioning baselines
- LLaVA-OneVision
- Honeybee / Bee multimodal training datasets
- Region-level VLM과 grounding papers

# 8. Summary

- PerceptionDLM은 AR decoding이 region count에 따라 비싸지는 multi-region captioning을 겨냥한다.
- 먼저 더 강한 diffusion VLM baseline인 PerceptionDLM-Base를 만든다.
- Region prompting, RoI-aligned feature replay, structured attention masking이 parallel region denoising을 가능하게 한다.
- ParaDLC-Bench는 caption quality와 efficiency를 함께 평가한다.
- 가장 강한 insight는 task가 자연스럽게 many simultaneous output region을 가질 때 DLM parallelism이 practical해진다는 점이다.
