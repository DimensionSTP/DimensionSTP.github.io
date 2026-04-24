---
layout: single
title: "Efficient Memory Management for Large Language Model Serving with PagedAttention Review"
categories: Study-concept
tag: [LLM, Serving, PagedAttention]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2309.06180)

PagedAttention은 이름만 보면 attention algorithm 논문처럼 보이지만, 실제로는 **LLM serving에서 KV cache를 어떻게 관리할 것인가**에 대한 시스템 논문에 가깝다. 이 논문이 중요한 이유는 단순하다. LLM inference의 병목을 "모델이 크다", "attention이 비싸다" 정도로만 보면 production serving에서 자주 마주치는 문제를 놓치게 된다. 실제 online serving에서는 batch를 크게 잡아야 throughput이 올라가는데, batch를 키우려면 GPU memory 안에 여러 request의 KV cache를 효율적으로 담아야 한다.

논문이 겨냥하는 병목은 바로 이 지점이다. autoregressive decoding에서는 request마다 prompt 길이와 output 길이가 다르고, output 길이는 미리 정확히 알기 어렵다. 기존 방식처럼 request별 KV cache를 contiguous chunk로 미리 잡으면, 아직 쓰지 않는 reservation, internal fragmentation, external fragmentation이 생긴다. 논문은 기존 시스템에서 실제 token state로 쓰이는 KV cache 비율이 20.4%~38.2%에 그칠 수 있다고 보고한다. 즉 GPU memory가 부족한 것이 아니라, GPU memory를 LLM serving workload에 맞게 관리하지 못하는 것이 병목이다.

PagedAttention의 핵심은 operating system의 virtual memory / paging 아이디어를 KV cache에 가져오는 것이다. request의 논리적 sequence는 연속적이지만, 실제 KV cache는 fixed-size block 단위로 GPU memory의 비연속 위치에 저장될 수 있다. attention kernel은 block table을 따라 필요한 KV block을 읽는다. 이 위에 vLLM은 scheduler, KV cache manager, block allocator, copy-on-write, preemption을 결합해 high-throughput serving engine을 만든다.

> 한 줄 요약: PagedAttention은 KV cache를 OS-style paged memory로 관리해 fragmentation과 중복 저장을 줄이고, vLLM은 이를 serving scheduler와 결합해 같은 latency 수준에서 더 큰 batch와 높은 throughput을 가능하게 만든 논문이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- LLM serving 비용은 model FLOPs뿐 아니라 KV cache memory utilization에 크게 좌우된다.
- vLLM의 기반 논문으로, 현재 LLM inference stack을 이해하려면 거의 필수 배경에 가깝다.
- long-context, beam search, parallel sampling, shared prefix, agentic workload처럼 KV cache가 커지거나 공유될 수 있는 상황에서 이 설계가 왜 중요한지 설명해 준다.
- "attention을 더 빠르게 계산한다"보다 "attention이 읽는 state를 어떻게 배치한다"가 serving throughput을 바꿀 수 있음을 보여준다.

이 논문의 핵심 메시지는 꽤 명확하다. **LLM serving의 병목은 모델 architecture 안에만 있는 것이 아니라, request state를 담는 memory abstraction에도 있다.** PagedAttention은 attention approximation이 아니라, 정확한 attention을 유지하면서 KV cache layout과 scheduler를 바꾸는 시스템 설계다.

# 1. Problem Setting

## 1-1. Problem definition

- 이 논문이 겨냥하는 문제는 high-throughput LLM serving에서 KV cache memory를 효율적으로 관리하는 것이다.
- LLM serving은 보통 throughput을 높이기 위해 여러 request를 batch로 묶는다. 하지만 batch size를 키우려면 각 request의 KV cache를 GPU memory에 충분히 담을 수 있어야 한다.
- autoregressive generation에서는 prompt phase와 decoding phase가 분리된다.
  - prompt phase는 prompt token 전체를 병렬 처리할 수 있다.
  - decoding phase는 token을 하나씩 생성해야 하며, 매 step마다 이전 token들의 key/value state를 참조한다.
- 따라서 decoding 중에는 각 sequence의 KV cache가 계속 증가한다.
- 문제는 request마다 input length, output length, lifetime이 다르고, output length는 보통 사전에 정확히 알 수 없다는 점이다.

논문은 13B model을 NVIDIA A100 40GB에서 serving할 때 memory layout을 예로 든다. model parameter가 약 65%를 차지하고, dynamic request state인 KV cache가 약 30%를 차지한다. activations 등 나머지 memory는 상대적으로 작다. 즉 serving throughput을 제한하는 핵심 동적 자원은 KV cache다.

## 1-2. Why previous approaches are insufficient

기존 LLM serving system의 주요 한계는 KV cache를 contiguous memory로 잡는다는 점이다.

- request마다 최대 sequence length만큼 memory를 미리 reserve하면, 실제 output이 짧을 때 대부분이 unused reservation이 된다.
- 실제 output 길이를 안다고 해도, request lifetime 동안 전체 chunk가 점유되므로 다른 request가 빈 공간을 활용하기 어렵다.
- request별 chunk size가 다르면 external fragmentation도 생긴다.
- parallel sampling이나 beam search처럼 하나의 prompt에서 여러 output sequence가 갈라지는 경우, prompt KV cache를 공유할 수 있음에도 기존 contiguous allocation에서는 중복 저장되기 쉽다.

논문 기준 Figure 2는 이 문제를 꽤 직관적으로 보여준다. Orca 계열 baseline에서는 KV cache usage 중 실제 token states가 20.4%~38.2% 수준으로 표시된다. 반면 vLLM은 token states 비율이 96.3%로 표시된다. 물론 이 수치는 특정 experiment setup에서의 profiling 결과이지만, 논문이 주장하는 방향은 명확하다. **성능 병목은 계산량만이 아니라 memory waste에서 온다.**

이 점에서 PagedAttention은 FlashAttention류와 문제 설정이 다르다. FlashAttention은 attention computation의 IO와 kernel efficiency를 줄이는 쪽에 가깝다면, PagedAttention은 long-lived KV cache를 serving workload 안에서 어떻게 allocate/share/free할지에 집중한다.

# 2. Core Idea

## 2-1. Main contribution

PagedAttention의 핵심 기여는 KV cache를 fixed-size block으로 나누고, logical block과 physical block을 분리한 것이다.

- sequence 입장에서는 KV cache가 token order대로 연속된 logical KV blocks처럼 보인다.
- GPU memory 입장에서는 각 logical block이 임의의 physical KV block에 mapping될 수 있다.
- PagedAttention kernel은 block table을 참조해 비연속 physical memory에 저장된 KV block을 읽고 attention을 계산한다.
- 새로운 token이 생성될 때마다 필요한 만큼만 block을 추가 할당한다.
- 모든 block 크기가 같기 때문에 external fragmentation을 줄일 수 있다.
- request별로 마지막 block 정도만 partially filled 상태가 되므로 internal waste가 block 하나 이내로 제한된다.
- block 단위 reference counting과 copy-on-write를 통해 prompt sharing, beam sharing, prefix sharing을 구현할 수 있다.

이 위에 vLLM은 다음 system component를 얹는다.

1. centralized scheduler
2. GPU / CPU block allocator
3. KV cache manager와 block table
4. custom PagedAttention CUDA kernel
5. fork / append / free abstraction for decoding algorithms
6. preemptive request scheduling with swapping or recomputation
7. tensor parallel distributed execution support

즉 이 논문은 "PagedAttention이라는 kernel 하나"보다 **PagedAttention을 중심으로 serving engine 전체를 재구성한 논문**으로 보는 편이 맞다.

## 2-2. Design intuition

이 논문의 설계 직관은 OS의 virtual memory analogy로 거의 설명된다.

| OS virtual memory | vLLM / PagedAttention |
| --- | --- |
| process | request / sequence |
| byte | token state |
| virtual page | logical KV block |
| physical page | physical KV block on GPU/CPU memory |
| page table | block table |
| copy-on-write page | shared KV block with ref count |

중요한 점은 "logical continuity"와 "physical contiguity"를 분리했다는 것이다. LLM 입장에서는 이전 token들의 KV cache가 순서대로 존재해야 한다. 하지만 그것이 GPU physical memory에서도 꼭 연속적으로 저장될 필요는 없다. kernel이 block table을 따라 정확히 읽을 수 있다면, attention 결과는 유지하면서 allocation policy만 바꿀 수 있다.

내가 보기엔 이 설계의 미묘한 장점은 LLM-specific semantic을 OS idea에 맞게 다시 해석했다는 데 있다. 예를 들어 일반 OS에서는 page eviction이 page 단위로 일어나지만, vLLM에서는 sequence를 처리하려면 해당 sequence의 모든 KV block이 필요하다. 그래서 preemption도 all-or-nothing sequence 단위가 된다. 즉 단순히 paging을 GPU에 복사한 것이 아니라, autoregressive decoding의 access pattern에 맞게 paging을 재설계한 것이다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | LLM serving에서 KV cache memory waste를 줄이고 batch size / throughput을 높이는 것 |
| Target bottleneck | dynamic KV cache growth, unknown output length, fragmentation, duplicated KV cache |
| Key abstraction | logical KV block <-> physical KV block mapping via block table |
| Core kernel | non-contiguous KV blocks를 읽는 PagedAttention kernel |
| System | PagedAttention 기반 LLM serving engine vLLM |
| Main benefit | near-zero KV cache waste, block-level sharing, higher serving throughput |
| Important caveat | attention kernel 자체는 block table indirection 때문에 latency overhead가 있다. end-to-end gain은 더 큰 batch와 memory utilization에서 나온다. |

## 3-2. Module breakdown

### 1) PagedAttention kernel: 비연속 KV cache를 읽는 attention

PagedAttention은 attention score를 계산할 때 key/value tensor가 하나의 contiguous tensor에 있다고 가정하지 않는다. 대신 KV cache를 fixed-size KV block으로 나누고, query token이 참조해야 하는 block들을 block table을 통해 찾아간다.

- 각 KV block은 일정 개수의 token에 대한 key/value vector를 담는다.
- logical block들은 sequence order를 표현한다.
- physical block들은 GPU memory 안의 실제 저장 위치를 표현한다.
- PagedAttention kernel은 block 단위로 key/value를 fetch하고 attention computation을 수행한다.

이 방식은 attention 값을 근사하거나 일부 token을 버리는 sparse attention이 아니다. 필요한 KV cache가 모두 존재한다면, logical sequence에 대한 정확한 attention을 계산한다. 따라서 논문이 말하는 성능 향상은 model accuracy를 바꾸는 성격이 아니라, serving memory layout을 바꾸는 성격이다.

### 2) KV cache manager: logical block과 physical block의 분리

vLLM의 memory manager는 request마다 logical KV block list를 유지한다. 새로운 token이 생성되면 현재 마지막 block에 공간이 있는지 확인하고, 공간이 있으면 거기에 KV cache를 추가한다. 마지막 block이 꽉 차면 새로운 physical block을 할당하고 block table을 업데이트한다.

이 구조의 장점은 다음과 같다.

- request 시작 시 최대 output length만큼 memory를 reserve하지 않아도 된다.
- KV cache는 실제 token이 생성될 때마다 on-demand로 증가한다.
- 같은 크기의 physical block을 재사용하므로 external fragmentation이 줄어든다.
- request가 끝나면 block 단위로 즉시 free되어 다른 request가 사용할 수 있다.
- 마지막 block을 제외하면 block이 꽉 찬 상태로 유지되므로 internal waste가 작다.

논문은 이를 통해 request 하나의 memory waste를 한 block 이내로 제한할 수 있다고 설명한다.

### 3) Sharing과 copy-on-write: parallel sampling / beam search를 위한 핵심

LLM serving에서는 하나의 prompt에서 여러 output을 만드는 경우가 많다.

- parallel sampling: 같은 prompt에서 여러 candidate output 생성
- beam search: 같은 prefix를 공유하다가 beam별로 sequence가 갈라짐
- shared prefix: 여러 request가 같은 instruction / few-shot examples를 공유

이때 prompt나 prefix의 KV cache는 여러 sequence가 공유할 수 있다. vLLM은 fork / append / free abstraction과 block-level reference count를 사용한다.

- `fork`: 기존 sequence에서 새 sequence를 만든다. 이때 기존 block은 복사하지 않고 reference count만 증가시킨다.
- `append`: 새 token을 붙인다. 공유 중인 block에 write가 필요하면 copy-on-write로 새 block을 만든다.
- `free`: sequence가 끝나면 해당 block reference를 줄이고, 더 이상 쓰이지 않는 block을 반환한다.

이 설계는 단순하지만 중요하다. memory sharing을 sequence-level string prefix가 아니라 **KV block-level physical memory sharing**으로 구현하기 때문이다. 특히 beam search처럼 공유 prefix가 긴 decoding algorithm에서 효과가 커진다.

### 4) Preemption: swapping vs recomputation

GPU memory의 physical block이 부족해지면 vLLM은 일부 sequence를 preempt할 수 있다. 여기서도 일반 OS paging과 달리 LLM-specific policy가 들어간다.

- vLLM은 sequence의 일부 block만 eviction하지 않고 all-or-nothing으로 처리한다.
- beam candidate처럼 같은 request 안에서 공유 관계가 있는 sequence들은 sequence group으로 묶여 함께 preempt / reschedule된다.
- evicted KV cache를 복구하는 방법은 두 가지다.
  - swapping: GPU에서 CPU memory로 block을 옮겼다가 다시 가져온다.
  - recomputation: prompt와 generated tokens를 다시 prompt phase처럼 넣어 KV cache를 재계산한다.

논문은 작은 block size에서는 swapping이 많은 작은 CPU-GPU transfer를 만들 수 있어 overhead가 커질 수 있고, recomputation과 swapping의 trade-off가 block size와 hardware bandwidth에 따라 달라진다고 분석한다.

### 5) Distributed execution과 kernel optimization

vLLM은 단일 GPU만을 가정하지 않는다. OPT-66B, OPT-175B처럼 model이 단일 GPU에 들어가지 않는 경우를 위해 tensor parallelism도 지원한다.

- centralized scheduler가 request state와 block table을 관리한다.
- GPU worker들은 동일한 logical-to-physical mapping 정보를 받아 각자 담당하는 model shard / attention heads에 대해 execution한다.
- memory management synchronization은 매 decoding iteration 시작 시 control message로 전달된다.

또한 PagedAttention은 비연속 memory access를 도입하기 때문에 kernel-level optimization이 필요하다.

- fused reshape and block write
- fused block read and attention
- fused block copy for copy-on-write
- variable sequence length support
- warp-level block read로 coalesced memory access 유지

이 부분이 중요하다. 단순히 "block table을 둔다"는 아이디어만으로는 빠르지 않다. block table lookup, branch, variable length 처리, non-contiguous access가 모두 overhead가 되기 때문이다. 그래서 이 논문의 실질적 contribution은 memory abstraction과 CUDA kernel co-design에 있다.

# 4. Training / Data / Recipe

## 4-1. Data

이 논문은 training paper가 아니라 serving system paper이므로, 학습 데이터보다는 evaluation workload가 중요하다.

논문에서 사용한 주요 workload는 다음과 같다.

| Workload | Role |
| --- | --- |
| ShareGPT | 실제 ChatGPT conversation에서 온 길이 분포를 기반으로 serving request trace 합성 |
| Alpaca | instruction dataset 기반으로 상대적으로 짧은 sequence workload 구성 |
| WMT16 English-to-German | shared prefix translation workload 구성 |
| Synthetic arrival process | dataset에 timestamp가 없기 때문에 Poisson distribution으로 request arrival time 생성 |

ShareGPT는 Alpaca보다 평균 input prompt가 8.4배 길고, 평균 output이 5.8배 길다고 논문은 설명한다. 따라서 ShareGPT 쪽이 KV cache pressure와 length variance를 더 강하게 보여주는 workload로 볼 수 있다.

## 4-2. System / serving recipe

논문이 사용한 model과 server setup은 다음과 같다.

| Model | GPU setup | Main purpose |
| --- | --- | --- |
| OPT-13B | 1x A100 40GB | single-GPU serving 분석 |
| OPT-66B | 4x A100 | tensor parallel distributed serving |
| OPT-175B | 8x A100-80GB | very large model serving |
| LLaMA-13B | A100 setup | shared prefix / translation workload |

baseline은 크게 두 축이다.

1. **FasterTransformer**
   - latency-optimized inference engine
   - 자체 scheduler가 없으므로 논문에서는 dynamic batching scheduler를 붙여 비교한다.

2. **Orca variants**
   - throughput-oriented LLM serving system
   - Orca가 공개되어 있지 않아 저자들이 구현한 version으로 비교한다.
   - output reservation 가정에 따라 Orca (Oracle), Orca (Pow2), Orca (Max)를 둔다.

여기서 Orca (Oracle)은 실제 output length를 미리 안다고 가정하는 upper-bound에 가깝다. 현실적으로는 불가능한 setting이지만, memory management 설계의 상한 비교군으로 유용하다.

metric은 serving throughput을 직접적으로 보기 위해 normalized latency를 쓴다. 즉 request end-to-end latency를 output length로 나눈 값을 보고, request rate가 증가할 때 normalized latency가 어느 지점에서 급격히 폭발하는지 비교한다. serving system 관점에서는 단일 kernel latency보다 "같은 latency budget에서 얼마나 높은 request rate를 버티는가"가 더 중요하기 때문이다.

## 4-3. Engineering notes

논문에서 실무적으로 중요한 engineering note는 다음과 같다.

- **Block size trade-off**
  - block size가 너무 작으면 GPU parallelism을 충분히 활용하기 어렵다.
  - block size가 너무 크면 internal fragmentation이 늘고 sharing 가능성이 줄어든다.
  - 논문은 default block size를 16으로 둔다.

- **PagedAttention kernel overhead**
  - block table access, extra branch, variable length handling 때문에 attention kernel latency는 FasterTransformer 대비 20%~26% 높게 나올 수 있다.
  - 하지만 이 overhead는 attention operator에 국한되고, end-to-end에서는 더 큰 batch를 담을 수 있는 memory efficiency가 이를 상쇄한다.

- **Recomputation vs swapping**
  - 작은 block size에서는 swapping이 많은 작은 transfer를 만들 수 있다.
  - recomputation은 decoding token들을 다시 prompt처럼 한 번에 처리할 수 있어 예상보다 싸게 복구될 수 있다.
  - 실제 선택은 CPU-GPU bandwidth, GPU compute, block size, request pattern에 의존한다.

- **Copy-on-write 구현 비용**
  - block-level sharing 자체는 memory를 아끼지만, 공유 block에 write가 발생하면 copy-on-write가 필요하다.
  - 따라서 block copy kernel을 따로 최적화하지 않으면 작은 copy operation이 많아져 overhead가 커질 수 있다.

# 5. Evaluation

## 5-1. Main results

논문의 main claim은 vLLM이 FasterTransformer / Orca 대비 같은 latency 수준에서 2x~4x throughput improvement를 보인다는 것이다. 이 숫자는 abstract와 conclusion 모두에서 강조된다.

좀 더 세부적으로 보면 다음 결과들이 중요하다.

### Basic sampling

- ShareGPT dataset에서 vLLM은 Orca (Oracle) 대비 1.7x~2.7x 더 높은 request rate를 유지한다.
- Orca (Max) 대비로는 2.7x~8x 더 높은 request rate를 유지한다.
- FasterTransformer 대비로는 최대 22x 높은 request rate를 유지한다고 보고한다.
- OPT-13B ShareGPT trace에서 평균 batched requests는 Orca (Max) 7.00, Orca (Oracle) 13.62, vLLM 30.42로 표시된다.
- Alpaca처럼 sequence가 짧고 GPU memory 여유가 큰 setting에서는 advantage가 줄어든다. 논문은 이 경우 system이 memory-bound보다 compute-bound에 가까워지기 때문이라고 해석한다.

### Parallel sampling / beam search

PagedAttention의 block sharing은 multi-output decoding에서 더 두드러진다.

- Alpaca trace 기준 parallel sampling memory saving은 6.1%~9.8%로 보고된다.
- beam search memory saving은 37.6%~55.2%로 더 크다.
- ShareGPT dataset에서는 parallel sampling 16.2%~30.5%, beam search 44.3%~66.3% memory saving을 보고한다.
- OPT-13B / Alpaca에서 vLLM의 Orca (Oracle) 대비 improvement는 basic sampling 1.3x에서 beam width 6의 beam search 2.3x로 커진다.

이 결과는 이 논문을 이해할 때 중요하다. PagedAttention의 장점은 단순히 allocation waste를 줄이는 데서 끝나지 않는다. **여러 sequence가 같은 prefix / prompt / beam history를 공유할 때 KV cache를 실제 physical memory 수준에서 공유할 수 있다.**

### Shared prefix / chatbot workload

- shared prefix translation workload에서는 1-shot prefix 공유 시 Orca (Oracle) 대비 1.67x throughput, 5-shot prefix 공유 시 3.58x throughput을 보고한다.
- chatbot workload에서는 vLLM이 Orca baselines 대비 2x higher request rate를 sustain한다고 보고한다.

chatbot workload에서는 long prompt가 많고, Orca baseline이 output side reservation을 크게 잡기 때문에 fragmentation / reservation 문제가 더 두드러진다. 이 지점은 실제 서비스 관점에서 특히 중요하다. 사용자 대화 history가 길어질수록 KV cache pressure가 커지고, output length는 여전히 예측하기 어렵다.

## 5-2. What really matters in the experiments

이 논문의 evaluation에서 가장 중요한 포인트는 "kernel latency만 보면 PagedAttention이 손해일 수 있다"는 점이다.

논문은 PagedAttention kernel이 FasterTransformer attention kernel보다 20%~26% 높은 latency를 가질 수 있다고 보고한다. 그럼에도 end-to-end serving throughput이 좋아지는 이유는 다음과 같다.

- KV cache waste가 줄어 더 많은 request를 동시에 batch에 넣을 수 있다.
- request가 끝날 때 block 단위로 빠르게 memory를 회수할 수 있다.
- parallel sampling / beam search / shared prefix에서 KV cache duplication을 줄일 수 있다.
- memory-bound workload에서는 batch size 증가가 kernel overhead보다 더 큰 이득을 준다.

따라서 이 논문을 "더 빠른 attention kernel"로 읽으면 오해가 생긴다. PagedAttention은 단일 attention call을 빠르게 만드는 논문이 아니라, **serving system 전체의 effective batch capacity를 키우는 논문**이다.

또 하나 중요한 점은 workload dependence다. sequence가 짧고 GPU memory가 충분한 경우, PagedAttention의 이점은 줄어든다. 논문도 OPT-175B / Alpaca 일부 setup에서 vLLM advantage가 덜 두드러진다고 설명한다. 이 경우는 memory management가 병목이라기보다 compute-bound에 가까워지기 때문이다.

# 6. Limitations

1. **custom kernel과 serving engine integration이 필요하다.**  
   PagedAttention은 단순 Python-level memory policy가 아니다. non-contiguous KV block을 읽는 attention kernel, block write, block copy, scheduler integration이 필요하다. 따라서 아이디어는 단순하지만 구현은 heavy system work다.

2. **workload가 compute-bound이면 gain이 줄어든다.**  
   짧은 sequence, 작은 KV pressure, 충분한 GPU memory, 단순 decoding에서는 memory management 이득이 제한적일 수 있다. 논문도 Alpaca / large-memory setup에서 advantage가 덜 두드러지는 사례를 언급한다.

3. **baseline 해석에 주의가 필요하다.**  
   Orca는 공개 implementation이 없어 저자들이 구현한 version을 사용한다. Orca (Oracle)은 현실적으로 output length를 미리 아는 upper-bound setting이다. 비교 자체는 유용하지만, 실제 production engine끼리의 절대 성능 비교로 바로 읽으면 과하다.

4. **modern model architecture와 serving stack에서는 trade-off가 달라질 수 있다.**  
   논문은 OPT / LLaMA 계열과 A100 기반 setup에서 평가한다. 이후 GQA/MQA, quantized KV cache, speculative decoding, prefix caching, disaggregated serving, heterogeneous KV cache policy 등이 많이 등장했기 때문에 현재 stack에서는 병목이 달라질 수 있다.

5. **memory indirection overhead는 실제로 존재한다.**  
   block table lookup과 non-contiguous access는 공짜가 아니다. 논문이 kernel fusion으로 완화했지만, 이 overhead는 block size, hardware, cache locality, sequence length distribution에 따라 달라질 수 있다.

6. **preemption policy는 SLA와 충돌할 수 있다.**  
   swapping이나 recomputation은 throughput 관점에서는 합리적일 수 있지만, tail latency와 user-facing SLA에는 민감하게 작용할 수 있다. 특히 interactive serving에서는 average throughput뿐 아니라 p95/p99 latency를 같이 봐야 한다.

# 7. My Take

## 7-1. Why this matters for my work

내가 보기엔 PagedAttention의 가장 큰 의미는 LLM serving을 "model inference"가 아니라 **stateful online system**으로 보게 만든다는 점이다. LLM은 stateless matrix multiplication만 반복하는 것이 아니다. request마다 prompt, generated tokens, KV cache, sampling state, stop condition, shared prefix, beam group이 계속 변한다. 이 state를 어떻게 저장하고 이동시키는지가 throughput을 결정한다.

이 관점은 agent serving이나 long-context RAG에도 그대로 연결된다.

- agent는 tool call과 multi-turn history 때문에 prompt가 길어지고 request lifetime이 불규칙해진다.
- RAG는 system prompt / instruction / retrieved chunks처럼 shared prefix 또는 반복 prefix가 생기기 쉽다.
- batch traffic은 항상 균일하지 않고, short request와 long request가 섞인다.
- speculative decoding이나 parallel sampling은 하나의 request에서 여러 branch를 만들 수 있다.

결국 production LLM serving에서 중요한 것은 "한 request를 얼마나 빨리 처리하는가"뿐 아니라, **서로 다른 request state를 한 GPU memory pool 안에서 얼마나 잘 multiplexing하는가**다. PagedAttention은 이 문제를 매우 깔끔한 abstraction으로 풀었다.

## 7-2. Reuse potential

실무적으로 재사용 가치가 높은 포인트는 다음과 같다.

1. **KV cache를 token tensor가 아니라 paged state로 본다.**  
   long-context serving을 다룰 때 KV cache를 단순 tensor allocation 문제가 아니라 memory management policy 문제로 보게 만든다.

2. **block table abstraction**  
   logical sequence와 physical memory layout을 분리하면 allocation, sharing, eviction policy를 훨씬 유연하게 만들 수 있다.

3. **copy-on-write prefix sharing**  
   beam search, parallel sampling, shared system prompt, few-shot prefix가 많은 workload에서 매우 직접적으로 유용하다.

4. **metric selection**  
   serving evaluation은 single-request latency만 보면 부족하다. request rate를 올리면서 normalized latency가 폭발하는 지점을 보는 방식은 production capacity planning에 더 가깝다.

5. **block size tuning 관점**  
   block size는 단순 hyperparameter가 아니라 hardware utilization과 fragmentation 사이의 trade-off다. sequence length distribution이 다른 서비스에서는 optimal block size도 달라질 수 있다.

이 논문에서 바로 가져갈 수 있는 생각법은 "vLLM을 쓰자"보다 한 단계 위에 있다. serving stack을 설계할 때 request state를 어떻게 chunking하고, 어떤 state를 공유하고, 어떤 state를 recompute할지 먼저 정해야 한다. PagedAttention은 그 대표적인 성공 사례다.

## 7-3. Follow-up papers

- **Orca: A Distributed Serving System for Transformer-Based Generative Models**
  - iteration-level scheduling과 LLM serving throughput 관점에서 PagedAttention의 비교축으로 읽기 좋다.

- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**
  - attention computation의 IO 효율화와 PagedAttention의 KV cache memory management를 구분해서 이해하기 좋다.

- **Fast Transformer Decoding: One Write-Head is All You Need**
  - MQA/GQA 계열이 KV cache footprint를 어떻게 줄이는지 비교축으로 볼 수 있다.

- **Jenga: Effective Memory Management for Serving LLM with Heterogeneity**
  - PagedAttention 이후 modern heterogeneous architecture에서 memory allocation 문제가 어떻게 확장되는지 보기 좋다.

- **PagedEviction: Structured Block-wise KV Cache Pruning for Efficient Large Language Model Inference**
  - PagedAttention의 block abstraction 위에서 KV eviction / pruning을 어떻게 설계할 수 있는지 후속 관점으로 읽기 좋다.

# 8. Summary

- PagedAttention은 attention approximation이 아니라 KV cache memory layout과 allocation을 바꾸는 serving system 논문이다.
- 핵심은 logical KV block과 physical KV block을 분리하고, block table을 통해 non-contiguous KV cache를 정확히 읽는 것이다.
- vLLM은 이 abstraction 위에 scheduler, block allocator, copy-on-write, preemption, distributed execution을 결합한다.
- 논문은 vLLM이 같은 latency 수준에서 FasterTransformer / Orca 대비 2x~4x serving throughput improvement를 보인다고 보고한다.
- 실무적으로는 "LLM serving = stateless inference"가 아니라 "dynamic request state memory management"라는 관점을 준다는 점이 가장 크다.
