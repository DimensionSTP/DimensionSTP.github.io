---
layout: single
title: "MegaTrain: Full Precision Training of 100B+ Parameter Large Language Models on a Single GPU Review"
categories: Study-concept
tag: [LLM, Systems, Offloading, MemoryHierarchy, PostTraining]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2604.05091)

[Code link](https://github.com/DLYuanGod/MegaTrain)

MegaTrain은 "single GPU에서도 100B+ parameter LLM을 실제로 학습시킬 수 있나"라는 질문을 정면으로 다루는 systems paper다. 겉으로 보면 offloading 논문처럼 보이지만, 이 논문의 진짜 포인트는 조금 다르다. 기존 시스템이 CPU memory를 spill buffer처럼 다뤘다면, MegaTrain은 아예 관점을 뒤집어 host memory를 "authoritative store"로 두고 GPU를 "transient compute engine"으로 재정의한다.

이 차이는 생각보다 크다. 많은 offloading 시스템은 결국 GPU가 모델의 주 저장소라는 가정을 유지한 채 일부 상태만 밖으로 빼낸다. 그래서 모델이 커질수록 parameter staging duplication, fragmented transfer, persistent autograd graph, optimizer state round-trip이 한꺼번에 병목이 된다. MegaTrain은 이 병목을 "GPU memory를 조금 더 아껴 보자"가 아니라 "어떤 상태를 어느 memory tier에 영구적으로 둘 것인가"의 문제로 다시 푼다.

> 한 줄 요약: MegaTrain은 parameter, gradient, optimizer state를 host memory에 두고 GPU는 레이어 단위로 가중치를 스트리밍해 계산만 수행하도록 바꾼 뒤, double-buffered pipeline과 stateless layer template을 결합해 single GPU에서도 100B+ 모델의 full-parameter training feasibility를 열어 보인 memory-centric training system이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- post-training, instruction tuning, domain adaptation처럼 "연산량보다 메모리"가 더 먼저 막히는 구간을 제대로 겨냥한다.
- 기존 offloading을 조금 개선한 정도가 아니라, **GPU를 모델 저장소로 보는 기본 가정 자체를 뒤집는다.**
- 120B scale, 512K context, A100/A6000/RTX3090까지 이어지는 결과 덕분에 "가능성 데모"를 넘어서 시스템 경계가 어디인지 비교적 명확하게 보여준다.

이 논문의 핵심 메시지는 단순하다. 큰 모델 학습 = 많은 GPU 라는 공식이 항상 맞는 것은 아니다. 특히 post-training regime에서는 **메모리 계층을 어떻게 쓰는가**가 GPU 개수만큼 중요할 수 있다. MegaTrain은 바로 그 지점을 아주 선명하게 보여준다.

# 1. Problem Setting

## 1-1. Problem definition

LLM training memory는 대략 아래 3가지로 나뉜다.

$$
M_{total} = M_{persist} + M_{act} + M_{workspace}
$$

여기서 persistent state는 parameter, gradient, optimizer state다. 논문은 mixed-precision Adam 기준으로 parameter 하나당 BF16 weight 2B, BF16 gradient 2B, FP32 moment 8B가 필요하다고 정리한다. 즉 persistent state만 대략 $12N$ byte 수준이 된다. 70B 모델이면 이것만으로도 최소 840GB 수준이라, 모델 전체를 GPU에 상주시킨다는 가정이 바로 깨진다.

문제는 여기서 끝나지 않는다.

- fine-tuning이나 RL post-training은 trillion-scale pretraining보다 계산량은 가벼운 편이지만, 여전히 full model parameter와 optimizer state를 들고 있어야 한다.
- 연구실이나 작은 팀 입장에서는 GPU 수보다 먼저 HBM 용량이 막힌다.
- offloading을 써도 device memory residency라는 기본 가정을 유지하면 host memory duplication과 transfer overhead가 커진다.
- 결국 실제 병목은 "GPU가 느리다"가 아니라 "persistent state를 어느 계층에 고정할 것인가"가 된다.

즉 이 논문의 문제 설정은 offloading을 더 잘할 수 있는가가 아니라, **single GPU training의 scaling boundary를 device memory에서 host memory로 재정의할 수 있는가**에 가깝다.

## 1-2. Why previous approaches are insufficient

기존 접근의 한계는 아래처럼 정리할 수 있다.

| Type | Main idea | Limitation |
| --- | --- | --- |
| PyTorch native | parameter와 optimizer state를 GPU에 상주시킴 | small scale에서는 빠르지만 GPU memory를 넘는 순간 바로 실패 |
| ZeRO-3 CPU offload | 일부 상태를 CPU로 내림 | GPU가 여전히 모델의 중심 저장소이고, staging duplication과 PCIe synchronization 비용이 큼 |
| ZeRO-Infinity | CPU와 NVMe까지 확장 | capacity는 늘어나지만 storage tier까지 내려가면 bandwidth penalty가 더 커짐 |
| FSDP offloading | shard와 recompute로 memory를 줄임 | 깊이나 폭이 커질수록 fragmentation과 activation pressure가 빠르게 커짐 |

MegaTrain이 정확히 찌르는 지점은 이것이다. 기존 시스템은 host memory를 "임시 대피소"로 쓴다. 하지만 training에서 parameter와 optimizer state는 activation보다 접근 빈도가 낮다. 그럼 이런 cold state를 굳이 가장 비싼 HBM에 오래 붙잡아 둘 이유가 약하다. MegaTrain은 바로 이 점을 이용해 host memory를 1차 저장소로 승격한다.

# 2. Core Idea

## 2-1. Main contribution

MegaTrain의 핵심 기여는 크게 4가지다.

1. **Memory hierarchy inversion**
   - parameter, gradient, optimizer state를 host memory에 둔다.
   - GPU는 레이어 단위 계산을 위한 일시적 cache처럼만 쓴다.

2. **Streaming forward/backward execution**
   - forward에서는 다음 layer weight를 prefetch하면서 현재 layer를 계산한다.
   - backward에서는 block-wise recomputation으로 activation residency를 한정하고, gradient는 계산 직후 host memory로 내린다.

3. **Double-buffered pipelined engine**
   - compute stream, weight transfer stream, gradient transfer stream을 분리한다.
   - weight prefetch, compute, gradient offload를 겹쳐서 CPU-GPU bandwidth bottleneck을 critical path 밖으로 밀어낸다.

4. **Stateless layer templates**
   - persistent autograd graph를 유지하지 않는다.
   - 빈 template에 stream-in된 weight view를 동적으로 bind해서 실행한다.

이 조합 덕분에 MegaTrain은 "모델 전체가 GPU에 올라가 있어야 한다"는 전제를 버리고도, throughput을 너무 크게 잃지 않으면서 scale을 밀어 올린다.

## 2-2. Design intuition

이 논문의 설계 직관은 surprisingly simple하다.

- activation은 지금 당장 필요하니 빠른 memory tier에 둬야 한다.
- parameter와 optimizer state는 레이어 단위 접근이므로, 굳이 항상 GPU에 붙어 있을 필요가 없다.
- 그렇다면 GPU는 resident model store가 아니라, stream-in된 layer를 계산하고 바로 비워주는 engine이 되는 편이 더 낫다.

이 직관은 그냥 capacity만 늘리는 것이 아니다. 오히려 중요한 것은 transfer를 어떻게 숨기느냐다. 논문이 double buffering을 핵심으로 두는 이유도 여기 있다. CPU에 두는 것 자체는 새로운 발상이 아니다. 하지만 계산과 weight prefetch, gradient offload가 겹치지 않으면 offloading은 바로 느려진다. MegaTrain의 진짜 기술적 무게중심은 offloading 그 자체보다 **streaming schedule을 steady-state pipeline으로 만든 것**에 있다.

이 논문은 "memory-centric training"이라는 말을 꽤 정직하게 쓴다. 많은 시스템이 결국 GPU-centric 최적화의 연장선에 머무르는데, MegaTrain은 처음부터 host memory를 주 저장소로 두고 나머지 설계를 그 가정에 맞춰 다시 짠다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | single GPU에서 100B+ LLM을 full-parameter training 가능한 형태로 만드는 것 |
| Persistent state location | host memory |
| GPU role | streamed layer만 계산하는 transient compute engine |
| Key modules | block-wise recomputation, double-buffered pipeline, layer-contiguous tiling, stateless template binding |
| Difference from prior work | CPU memory를 spill buffer가 아니라 authoritative store로 둠 |

## 3-2. Module breakdown

### 1) Host memory as the authoritative store

MegaTrain에서는 persistent state가 모두 host memory에 있다.

- model parameter
- accumulated gradient
- optimizer state

이게 가장 중요한 출발점이다. GPU는 비어 있는 상태에서 시작하고, lightweight한 layer template pool만 가진다. 각 layer weight는 필요할 때만 device buffer로 들어오고, 계산이 끝나면 바로 해제된다.

이 구조의 장점은 명확하다. device memory 사용량이 전체 model size가 아니라 "현재 계산 중인 layer와 그 주변 buffer"에 묶인다. 즉 모델 깊이가 커져도 GPU memory bound가 훨씬 천천히 온다.

### 2) Streaming forward and block-wise backward

Forward phase에서는 host memory에서 weight를 layer-by-layer로 stream-in한다. 현재 layer를 계산하는 동안 다음 layer weight를 미리 옮겨 둔다.

Backward phase에서는 block-wise recomputation을 사용한다.

- activation을 전부 저장하지 않는다.
- checkpoint interval 단위로만 activation을 남긴다.
- backward 때는 해당 block의 forward를 다시 계산해 activation을 복원한다.
- gradient가 계산되면 즉시 host memory로 offload한다.

이 방식은 compute를 조금 더 쓰는 대신 activation residency를 강하게 제한한다. 논문이 long context에서 비교적 안정적인 memory를 유지하는 이유도 결국 여기 있다.

### 3) Pipelined double-buffered execution engine

논문의 핵심 시스템 모듈은 이 부분이다.

MegaTrain은 3개의 CUDA stream을 분리한다.

- compute stream
- weight transfer stream
- gradient transfer stream

그리고 CPU/GPU 양쪽에 double buffer를 둔다. compute stream이 Buffer 0의 layer를 계산하는 동안 weight transfer stream은 Buffer 1로 다음 layer를 채우고, backward에서는 gradient offload도 background에서 진행한다.

이 구조를 단순화하면 다음과 같다.

$$
\text{next layer transfer} \parallel \text{current layer compute} \parallel \text{previous grad offload}
$$

즉 transfer를 없애는 것이 아니라 겹쳐서 숨긴다. 논문 ablation에서 double buffering을 제거하면 throughput이 266.3 TFLOPS에서 182.9 TFLOPS로 크게 떨어진다. 이 결과가 말해주는 것은 분명하다. **MegaTrain의 성능 핵심은 memory placement만이 아니라 overlapping schedule이다.**

### 4) Layer-contiguous tiling and pinned slabs

논문은 fragmented tensor layout도 병목으로 본다. 그래서 layer별로 BF16 weight, BF16 gradient, FP32 Adam moment를 하나의 contiguous block으로 묶는다. 이렇게 하면 작은 DMA를 수없이 날리는 대신 burst transfer를 만들 수 있다.

또한 host memory 전체를 pinning하지 않고, 작은 수의 pinned staging buffer만 두고 JIT packing으로 pageable layer store에서 slab로 복사한 뒤 DMA를 건다. 이 설계는 pinning footprint를 일정하게 유지하면서 transfer bandwidth를 확보하는 절충안이다.

### 5) Stateless execution model

기존 autograd graph는 "parameter와 activation이 backward가 끝날 때까지 GPU에 살아 있다"는 가정 위에서 잘 작동한다. 하지만 MegaTrain은 weight를 layer마다 stream-in하고 바로 버린다. 이 상황에서는 global autograd graph가 자연스럽게 맞지 않는다.

그래서 MegaTrain은 stateless template pool을 쓴다.

- template은 Attention, MLP 같은 CUDA kernel structure만 갖는다.
- persistent weight pointer는 없다.
- 실행 직전에 streaming buffer의 weight view를 bind한다.

즉 수학 구조와 실제 data pointer를 분리한다. 이 덕분에 massive graph metadata를 유지할 필요가 없고, buffer ownership과 synchronization을 더 명시적으로 다룰 수 있다.

### 6) CPU-side optimizer update

논문은 optimizer step도 CPU에서 수행한다. 이유는 단순하다. Adam update는 arithmetic intensity가 낮고 I/O가 큰 연산이다. GPU에서 하려면 moment state를 다시 올렸다가 다시 내려야 한다. 오히려 CPU에서 vector instruction으로 처리하는 편이 bandwidth 관점에서 유리하다.

이 부분은 실무적으로도 중요한 포인트다. MegaTrain은 GPU가 더 빠르니 다 GPU에서 하자라는 발상이 아니라, **어떤 연산이 bandwidth-bound인지 compute-bound인지에 따라 실행 위치를 바꾸는 시스템**이다.

# 4. Training / Data / Recipe

## 4-1. Data

이 논문은 새로운 모델 recipe보다는 system paper다. 그래서 데이터도 "무슨 데이터가 더 좋은가"보다는 correctness preservation을 검증하는 benchmark 역할이 더 크다.

논문이 accuracy evaluation에 쓰는 데이터는 MetaMathQA다.

- 약 395K 영어 수학 문제-답 쌍
- GSM8K, MATH 기반 augmentation 데이터
- train 70%, test 30% 분할
- metric은 exact match accuracy

중요한 점은 MegaTrain이 새 model capability를 주장하는 논문이 아니라는 것이다. 정확도 평가는 "이 시스템이 수치적 drift 없이 정상 학습되는가"를 보여주기 위한 sanity check에 가깝다.

## 4-2. Training strategy

논문은 GH200와 H200 두 단일 GPU 플랫폼을 중심으로 본다.

| System | Main use in paper |
| --- | --- |
| GH200 | 7B, 14B, 32B와 long context evaluation |
| H200 + 1.5TB host memory | 72B, 120B scale feasibility |
| A100 PCIe | commodity datacenter verification |
| RTX A6000 / RTX 3090 | workstation and consumer verification |

실험에 쓰인 대표 모델은 아래와 같다.

- Qwen2.5-7B
- Qwen2.5-14B
- Qwen2.5-32B
- Qwen2.5-72B
- GPT-OSS-120B

학습 전략에서 핵심은 model recipe가 아니라 runtime recipe다.

- layer-wise parameter streaming
- block-wise recomputation
- CPU-side Adam update
- checkpoint interval tuning
- pinned slab staging
- multi-stream overlap

즉 이 논문에서 "recipe"는 data mixture가 아니라 scheduling policy에 가깝다.

## 4-3. Engineering notes

실무 관점에서 특히 눈에 띄는 포인트는 아래 5가지다.

1. **Host memory capacity가 진짜 scaling boundary다.**
   - 논문은 large model single-device training에서 device memory보다 host memory가 더 먼저 중요해질 수 있다고 본다.
   - 120B demo도 결국 H200 + 1.5TB host memory 위에서 성립한다.

2. **Checkpoint interval은 throughput과 memory의 trade-off다.**
   - 너무 자주 checkpoint하면 activation pressure가 커지고 batch size가 줄어든다.
   - 너무 드물면 recomputation cost가 커진다.

3. **Width scaling과 depth scaling의 bottleneck이 다르다.**
   - depth가 커지면 orchestration과 scheduling의 질이 더 중요해진다.
   - width가 커지면 per-layer tensor size가 직접 커져 bandwidth와 activation pressure가 더 세게 온다.

4. **Long context에서 arithmetic intensity가 오히려 utilization을 올릴 수 있다.**
   - 1K에서 512K로 갈수록 TFLOPS가 올라가는 결과는 흥미롭다.
   - 물론 latency는 길어지지만, utilization 관점에서는 나쁘지 않다.

5. **이 시스템은 framework-level 구현 난도가 꽤 높다.**
   - explicit stream scheduling
   - buffer lifecycle management
   - stateless template binding
   - pinned slab management
   - CPU-side async optimizer
   - 이런 부분이 모두 맞물려야 한다.

MegaTrain은 "아이디어는 쉬운데 구현은 어려운" 시스템이다. 논문만 보고 개념을 이해하는 것과, 실제 training backend로 옮기는 것은 난이도가 꽤 다르다.

# 5. Evaluation

## 5-1. Main results

가장 중요한 결과는 아래 4개로 요약할 수 있다.

| Result | What it means |
| --- | --- |
| single H200 + 1.5TB host memory에서 120B까지 학습 가능 | single GPU feasibility boundary를 크게 밀어 올림 |
| single GH200에서 14B 기준 ZeRO-3 Offload 대비 1.84x throughput | offloading baseline 대비 실질 성능 이득을 보임 |
| single GH200에서 7B 512K context 학습 가능 | long context training에도 구조적으로 확장 가능 |
| RTX 3090에서 14B를 30.19 TFLOPS로 학습 | consumer급 환경에서도 의미 있는 scale을 보여줌 |

정확도 쪽도 시스템 논문치고는 깔끔하다.

| Metric | Baseline | ZeRO-3 Offload | ZeRO-Infinity | PyTorch Native | Ours |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7B Acc. (%) | 33.47 | 88.93 | 88.97 | 88.91 | 88.99 |
| 14B Acc. (%) | 37.58 | 92.41 | 92.36 | - | 92.52 |

즉 MegaTrain은 속도와 scale만 챙기고 correctness를 잃는 방식이 아니다. 적어도 이 설정에서는 full-GPU training이나 ZeRO계열과 거의 같은 정확도를 유지한다.

Long context 결과도 꽤 인상적이다.

| Context | BS | Tokens | Step (s) | TFLOPS | Mem |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1K | 158 | 162.7K | 27.05 | 284.7 | 74.2 GB |
| 8K | 25 | 204.8K | 32.36 | 294.5 | 86.5 GB |
| 32K | 6 | 196.6K | 32.18 | 316.7 | 84.0 GB |
| 128K | 1 | 131.1K | 26.13 | 305.3 | 62.1 GB |
| 256K | 1 | 262.1K | 236.1 | 401.2 | 88.2 GB |
| 512K | 1 | 524.3K | 871.4 | 407.4 | 81.9 GB |

이 표가 보여주는 것은 간단하다. context length가 커져도 activation residency가 layer 단위로 제한되기 때문에 memory가 폭발하지 않는다. 물론 step time은 매우 길어지지만, 이건 "안 된다"와는 다른 이야기다.

## 5-2. What really matters in the experiments

### 1) 이 논문의 핵심은 120B demo 하나가 아니다

가장 headline-friendly한 결과는 "single H200에서 120B"다. 하지만 더 중요한 것은 host memory footprint curve다. 기존 offloading 시스템은 30B를 넘어서면서 host memory demand가 빠르게 커지고, 결국 duplication과 staging overhead 때문에 practical boundary를 먼저 맞는다. MegaTrain은 flat-tensor layout과 CPU master store 덕분에 이 증가를 더 linear하게 가져간다.

즉 이 논문의 진짜 메시지는 120B도 된다보다 **왜 기존 시스템은 30B 이후 급격히 힘들어지는가**를 구조적으로 설명한다는 데 있다.

### 2) double buffering은 optional trick이 아니다

Ablation에서 double buffering을 빼면 throughput이 266.3 TFLOPS에서 182.9 TFLOPS로 크게 떨어진다. 반면 gradient slab pool 제거는 영향이 훨씬 작다. 이 차이는 명확하다. MegaTrain의 중심은 memory pooling이 아니라 transfer-compute overlap이다.

이건 실무에도 바로 연결된다. CPU memory offload 자체는 누구나 떠올릴 수 있다. 하지만 compute stream과 transfer stream을 어떻게 steady-state로 맞물리게 할지 설계하지 않으면, offload는 그냥 느린 시스템이 된다.

### 3) depth scaling에서 특히 강하다

Depth scaling 실험이 인상적인 이유는 폭을 고정하고 layer 수만 늘렸기 때문이다. 이 설정에서는 device allocation을 고정한 채 parameter count를 키우므로, 순수하게 scheduling과 recomputation orchestration이 얼마나 견디는지 볼 수 있다.

논문 결과를 보면 MegaTrain은 28 layer에서 180 layer로 가도 throughput이 284에서 227 TFLOPS로만 줄어든다. 반면 FSDP와 ZeRO-3는 depth가 커질수록 급격히 무너지고 OOM으로 이어진다.

이 실험이 중요한 이유는, MegaTrain이 단순히 HBM이 커서 유리했다가 아니라 **깊은 모델에서 scheduling 자체가 더 잘 버틴다**는 점을 보여주기 때문이다.

### 4) width scaling은 여전히 어렵다

반면 width scaling에서는 모두 throughput이 줄어든다. MegaTrain도 1.0x에서 3.0x width로 가며 406에서 264 TFLOPS로 떨어진다. 즉 이 시스템이 bandwidth 법칙을 무시하는 것은 아니다. 다만 degradation curve가 flatter하고, ZeRO-3와 FSDP가 더 빨리 OOM으로 넘어간다.

이 결과는 균형 있게 읽어야 한다. MegaTrain은 "모든 scaling 문제를 해결했다"가 아니다. 오히려 depth scaling 쪽에서 더 구조적 이점을 보이고, width scaling에서는 결국 per-layer tensor size 증가의 물리적 비용을 같이 진다.

### 5) consumer GPU 결과는 꽤 실용적이다

A6000 48GB와 RTX 3090 24GB 결과는 이 논문을 단순 H200 paper로 끝내지 않게 만든다.

- A6000에서 14B를 56.82 TFLOPS
- RTX 3090에서 14B를 30.19 TFLOPS
- ZeRO-3는 같은 조건에서 14B에서 OOM

물론 host memory를 251GB 공유하는 workstation 설정이 필요하다. 그래도 "consumer-grade single GPU에서 14B full-parameter training"은 실무적으로 상징성이 크다. 작은 팀이 7B, 14B급 adaptation을 실험할 여지를 꽤 넓혀 준다.

# 6. Limitations

1. 논문의 가장 큰 장점이자 한계는 "single GPU" framing이다.
   - 멋지지만, 실제로는 H200 + 1.5TB host memory 같은 꽤 큰 시스템을 쓰기도 한다.
   - 즉 GPU 수는 1개여도 host side 자원 요구가 결코 가볍지 않다.

2. 평가가 MetaMathQA 중심이다.
   - correctness preservation을 보기에는 적절하지만, 다양한 domain SFT나 RL post-training에서 optimizer dynamics가 똑같이 유지되는지는 추가 확인이 필요하다.

3. width scaling에서는 여전히 물리적 한계가 남아 있다.
   - per-layer tensor size가 커지면 bandwidth와 activation pressure가 직접 커진다.
   - MegaTrain이 이 문제를 마법처럼 없애는 것은 아니다.

4. 시스템 구현 난도가 높다.
   - explicit multi-stream scheduling
   - stateless template binding
   - pinned staging slab
   - async CPU optimizer
   - 이런 요소를 training framework에 안정적으로 붙이는 것은 쉽지 않다.

5. host memory bandwidth와 interconnect quality에 민감하다.
   - GH200의 NVLink-C2C와 H200 PCIe Gen4, A100 PCIe, A6000, RTX3090 결과가 모두 다르다.
   - 따라서 논문 수치를 일반화할 때는 hardware stack을 같이 봐야 한다.

6. 비교군 해석은 조금 조심해야 한다.
   - 논문은 ZeRO-3, ZeRO-Infinity, FSDP, Gemini를 강하게 이기지만, 각 시스템은 설계 목적과 maturity가 다르다.
   - 특히 SSD-centric 계열이나 distributed setting까지 포함한 완전한 landscape comparison은 아니다.

추가로 주의할 점은 하나 더 있다. MegaTrain은 post-training regime에 특히 어울리는 시스템처럼 보이지만, 그것이 바로 "작은 팀이 120B를 쉽게 다룬다"는 뜻은 아니다. 실제 도입 난도는 host memory, CPU bandwidth, framework integration, debugging 비용까지 같이 계산해야 한다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문이 중요한 이유는 모델 학습을 "compute scaling" 문제만이 아니라 "state placement" 문제로 보게 만든다는 점이다. 실제 서비스나 연구에서도 큰 모델 adaptation을 할 때 가장 먼저 막히는 것은 FLOPS가 아니라 HBM residency인 경우가 많다. 그럴 때 MegaTrain 같은 관점은 꽤 유용하다.

특히 아래 같은 상황에 잘 맞는다.

- long context SFT
- domain adaptation
- instruction tuning
- memory-bound RL post-training
- single node 환경에서의 14B, 32B, 그 이상 scale 실험

이 논문의 가장 실무적인 교훈은 "GPU를 더 구하기 전에, 지금 가진 memory hierarchy를 어떻게 쓰고 있는지 보라"는 것이다.

## 7-2. Reuse potential

재사용할 수 있는 아이디어는 아래 4가지다.

1. **authoritative store 분리**
   - parameter, optimizer state, gradient를 같은 tier에 둘 필요가 있는지 다시 보게 된다.

2. **stream overlap first**
   - offloading을 넣을 때 가장 먼저 설계해야 하는 것은 data movement를 숨기는 스케줄이다.

3. **depth-heavy model adaptation**
   - deep model에서 device memory가 먼저 막히는 경우 MegaTrain식 orchestration이 특히 유리할 수 있다.

4. **long-context post-training**
   - 512K context 결과는 "될 수 있다"는 것 자체가 의미가 있다.
   - 물론 step time은 매우 크지만, 연구용 feasibility boundary를 확인하는 데는 충분히 중요하다.

실무적으로는 바로 120B를 노리기보다, 14B와 32B adaptation에서 먼저 가치를 볼 가능성이 더 크다고 생각한다. 특히 A6000과 RTX3090 결과는 이 시스템의 메시지가 꼭 H200 전용은 아니라는 점을 보여준다.

## 7-3. Follow-up papers

- ZeRO-Offload
- ZeRO-Infinity
- ColossalAI Gemini
- Ratel
- activation checkpointing and recomputation 계열 시스템 논문들

# 8. Summary

- MegaTrain은 CPU memory를 persistent state의 주 저장소로 두고, GPU는 layer-by-layer 계산만 수행하는 memory-centric training system이다.
- 핵심 기술은 double-buffered multi-stream pipeline과 stateless layer template binding이다.
- single H200 + 1.5TB host memory에서 120B까지, single GH200에서 512K context까지 feasibility를 보인다.
- 가장 중요한 포인트는 "offloading을 더 했다"가 아니라 GPU-centric 가정을 버리고 memory hierarchy를 다시 설계했다는 점이다.
- 다만 host memory 요구량, hardware sensitivity, 구현 난도는 실제 도입 전에 반드시 같이 봐야 한다.
