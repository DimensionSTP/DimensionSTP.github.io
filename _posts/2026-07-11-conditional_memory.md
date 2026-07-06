---
layout: single
title: "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models Review"
categories: Study-concept
tag: [LLM, SparseModel, Memory, MoE, Architecture]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2601.07372)

[Code link](https://github.com/deepseek-ai/Engram)

Conditional Memory via Scalable Lookup은 sparse LLM 설계를 볼 때 꽤 중요한 축 하나를 추가하는 논문이다. 지금까지 sparse scaling이라고 하면 대부분 MoE를 먼저 떠올린다. MoE는 token마다 일부 expert만 활성화해서, total parameter는 크게 늘리되 activated parameter와 FLOPs는 제한하는 conditional computation이다.

이 논문은 여기서 한 걸음 옆으로 간다. 언어 모델에는 동적 추론뿐 아니라 정적 lookup에 가까운 부분도 많다는 것이다. 이름, 관용구, formulaic pattern, local dependency처럼 매번 깊은 attention과 FFN computation으로 재구성할 필요가 없는 정보가 있다. Transformer는 이런 static pattern lookup primitive가 없기 때문에, early layer compute를 써서 사실상 lookup table을 runtime에 재구성한다. Engram은 이 비효율을 conditional memory라는 새 sparsity axis로 분리하려는 시도다.

> 한 줄 요약: 이 논문은 MoE의 conditional computation과 별개로, Engram이라는 hashed n-gram lookup memory를 추가해 static local pattern을 sparse memory에서 가져오고, backbone은 더 복잡한 reasoning과 global context에 compute를 쓰게 만드는 conditional memory architecture를 제안한다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Sparse scaling을 expert routing 하나로 보지 않고, computation sparsity와 memory sparsity를 분리한다.
- Classic n-gram embedding을 modern Transformer, MoE, multi-branch architecture, system prefetching과 결합한다.
- Iso-parameter and iso-FLOPs MoE baseline과 비교해 memory allocation이 실제로 도움이 되는지 본다.
- Engram이 factual knowledge뿐 아니라 reasoning, code, math, long-context retrieval에도 영향을 준다는 분석을 제시한다.
- CPU offload와 deterministic prefetching까지 포함해 architecture and system co-design 관점으로 읽을 수 있다.

# 1. Problem Setting

## 1-1. Problem definition

이 논문이 겨냥하는 문제는 현재 Transformer가 knowledge lookup과 dynamic computation을 같은 neural pathway로 처리한다는 점이다.

언어 모델의 작업을 아주 거칠게 나누면 두 종류가 있다.

- Static local pattern retrieval.
  - named entity, idiom, frequent phrase, local collocation, morphology-like pattern.
- Dynamic compositional reasoning.
  - long-range dependency, multi-hop reasoning, code, math, instruction following, global context integration.

MoE는 두 번째 축, 즉 computation capacity를 sparse하게 늘리는 데 강하다. 하지만 static pattern retrieval도 expert computation으로 처리한다면, 모델은 매 token마다 불필요하게 깊은 compute를 사용해 local pattern을 다시 구성해야 한다.

Engram의 질문은 다음과 같다.

- Static pattern은 computation이 아니라 lookup으로 처리할 수 없는가.
- Lookup memory를 parameter budget 안에서 MoE와 어떻게 나눌 것인가.
- Massive lookup table을 GPU HBM에 모두 올리지 않고도 inference overhead를 작게 만들 수 있는가.
- 이런 memory가 단순 knowledge benchmark뿐 아니라 reasoning과 long context에도 도움을 주는가.

## 1-2. Why previous approaches are insufficient

첫째, pure MoE는 static memory와 dynamic compute를 분리하지 않는다. Expert는 capacity를 늘리지만, token마다 expert를 계산해야 한다. 정적 local pattern을 위한 dedicated lookup primitive는 아니다.

둘째, input embedding scaling만으로는 layer 내부의 computation 부담을 줄이기 어렵다. 단순히 embedding table을 키우면 input representation은 풍부해질 수 있지만, early layer에서 local pattern을 조합하는 부담은 여전히 남을 수 있다.

셋째, external retrieval은 editable knowledge에는 좋지만, 모든 local dependency를 retrieval system으로 보내는 것은 너무 무겁다. Named entity나 frequent n-gram 수준의 pattern은 model-internal lookup이 더 자연스러울 수 있다.

넷째, memory-augmented architecture는 system cost를 같이 봐야 한다. 거대한 table을 붙여도 실제 serving에서 latency와 bandwidth가 무너지면 의미가 없다. Engram은 deterministic addressing을 사용해 prefetching and overlap을 가능하게 만든다.

# 2. Core Idea

## 2-1. Main contribution

논문의 핵심 기여는 conditional memory라는 sparsity axis를 제안하고, 이를 Engram module로 구현한 것이다.

Engram은 suffix n-gram을 key로 사용해 sparse embedding table에서 memory vector를 lookup한다. 이후 현재 hidden state를 query로 삼아 context-aware gating을 수행하고, lookup memory를 residual stream에 더한다. 즉 memory는 context-independent prior로 시작하지만, current hidden state가 그 memory를 얼마나 받아들일지 결정한다.

전체 흐름은 다음처럼 볼 수 있다.

$$
tokens -> ngram\_id -> memory\_lookup -> gated\_fusion -> transformer\_block
$$

여기서 중요한 점은 Engram이 모든 layer에 들어가지 않는다는 것이다. 논문은 early layer에 static local pattern을 offload하는 것이 중요하다고 보고, system latency와 modeling gain 사이의 균형을 맞춰 placement를 정한다.

## 2-2. Design intuition

Engram의 직관은 "모든 knowledge를 compute로 복원할 필요는 없다"다.

Transformer가 "Diana, Princess of Wales" 같은 entity를 처리한다고 생각해보자. 표준 모델은 여러 layer를 거치며 token sequence를 조합하고, entity representation을 만들어야 한다. 하지만 이런 frequent multi-token pattern은 lookup table에 가까운 방식으로 처리할 수 있다. 이때 backbone은 entity reconstruction에 쓰던 early compute를 아끼고, 더 높은 수준의 reasoning에 depth를 쓸 수 있다.

이 관점에서 Engram은 모델을 더 깊게 만드는 효과를 낸다기보다, 낮은 layer가 하던 static reconstruction workload를 memory로 우회시킨다. 논문은 LogitLens와 CKA 분석을 통해 Engram representation이 baseline의 더 깊은 layer와 유사해지는 현상을 제시한다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | static local pattern retrieval을 neural compute에서 분리 |
| Module | Engram |
| Key idea | hashed n-gram lookup plus context-aware gating |
| Sparsity axis | conditional memory |
| Complementary axis | MoE conditional computation |
| System trick | deterministic addressing and prefetching |
| Main claim | MoE-only sparse budget보다 memory plus compute hybrid가 효율적 |

## 3-2. Module breakdown

### 1) Tokenizer compression

Subword tokenizer는 reconstruction을 위해 세밀한 token ID를 사용한다. 하지만 lookup memory에서는 같은 의미의 변형이 너무 많은 separate ID로 흩어지면 table density가 떨어진다.

Engram은 raw token ID를 normalized canonical ID로 collapse하는 vocabulary projection을 사용한다. 논문은 128k tokenizer에서 effective vocabulary size를 약 23% 줄였다고 보고한다. 이 단계는 lookup key space를 더 조밀하게 만들기 위한 것이다.

### 2) Multi-head hashing

n-gram space는 조합적으로 너무 크다. 모든 n-gram을 직접 parameterize할 수 없으므로 Engram은 hashing을 사용한다. 서로 다른 n-gram order에 대해 distinct hash head를 두고, 각 head가 embedding table index를 만든다.

Hash collision은 피할 수 없지만, multi-head structure와 context-aware gating이 collision noise를 줄이는 역할을 한다.

### 3) Context-aware gating

Lookup memory는 static prior이기 때문에 polysemy와 collision에 취약하다. 같은 local pattern이라도 문맥에 따라 쓸모가 다를 수 있다. 그래서 Engram은 현재 hidden state를 query로 사용해 retrieved memory와 gating을 수행한다.

간단히 쓰면 다음과 같은 선택이다.

$$
h' = h + gate(h, m) * m
$$

여기서 $h$는 current hidden state이고, $m$은 retrieved memory다. Gate가 낮으면 memory를 거의 무시하고, gate가 높으면 static lookup을 residual stream에 반영한다.

### 4) Causal convolution

Engram은 gated memory sequence에 lightweight depthwise causal convolution을 적용한다. 이는 lookup된 local memory를 약간 더 넓은 receptive field로 섞고, non-linearity를 추가하는 역할을 한다. 다만 ablation에서는 convolution 제거의 영향이 상대적으로 작다고 보고된다.

### 5) Multi-branch integration

논문은 standard residual stream이 아니라 multi-branch backbone과 결합한다. Engram memory table과 value projection은 공유하고, branch별 key projection을 다르게 둔다. 이 설계는 branch-specific gating을 가능하게 하면서도 GPU-friendly dense operation으로 묶을 수 있게 한다.

### 6) System efficiency

Engram의 lookup ID는 hidden state routing 결과가 아니라 input token sequence에서 결정된다. 이 deterministic addressing 덕분에 inference에서 다음 Engram lookup을 미리 계산하고 host memory에서 prefetch할 수 있다.

MoE expert routing은 runtime hidden state에 의존하지만, Engram은 token ID 기반이다. 이 차이가 system design에서 중요하다.

# 4. Training / Data / Recipe

## 4-1. Data

논문은 Engram을 large-scale pretraining setting에서 평가한다. 주요 model들은 262B tokens로 train되며, Dense-4B, MoE-27B, Engram-27B, Engram-40B를 비교한다.

Engram-27B는 MoE-27B에서 routed experts를 줄이고, 그 parameter budget을 5.7B-parameter Engram memory로 옮긴 구조다. Engram-40B는 activated parameter budget은 유지하면서 Engram memory를 더 키운다.

## 4-2. Training strategy

Engram은 backbone과 함께 pretraining된다. 주요 recipe는 다음과 같다.

- Dense baseline과 MoE baseline을 같은 activated parameter budget에서 비교한다.
- Engram-27B는 total parameter와 FLOPs를 MoE-27B와 맞춘다.
- Engram module은 특정 layer에만 삽입한다.
- Embedding parameters는 별도 Adam optimizer로 업데이트한다.
- Convolution은 zero initialization으로 시작해 identity behavior를 보존한다.
- Long-context extension에서는 YaRN 기반 32768-token context training을 사용한다.

## 4-3. Engineering notes

실무 관점에서 핵심은 Engram이 단순히 "큰 table을 붙이면 된다"가 아니라는 점이다.

1. Memory allocation ratio가 중요하다.
   - Pure MoE와 pure memory 사이에 optimum이 있다.
   - 논문은 sparse parameter budget을 MoE와 Engram 사이에 나누는 Sparsity Allocation problem을 제시한다.

2. Insertion layer가 중요하다.
   - 너무 이른 layer는 global context가 부족해 gating이 부정확할 수 있다.
   - 너무 늦은 layer는 static reconstruction offload 효과가 줄어든다.
   - 논문 ablation에서는 early plus mid layer 배치가 중요하게 나온다.

3. System prefetching이 architecture design에 들어간다.
   - Engram은 host memory offload를 전제로 설계할 수 있다.
   - Memory index가 deterministic이기 때문에 communication-computation overlap이 가능하다.

4. Cache hierarchy를 활용할 여지가 있다.
   - n-gram frequency는 Zipfian distribution을 따른다.
   - frequent memory slot은 HBM이나 DRAM에 두고, long tail은 더 느린 storage에 둘 수 있다.

# 5. Evaluation

## 5-1. Main results

Engram-27B는 iso-parameter and iso-FLOPs MoE-27B baseline보다 여러 benchmark에서 개선을 보인다. 논문은 MMLU, CMMLU 같은 knowledge benchmark뿐 아니라 BBH, ARC-Challenge, DROP, HumanEval, GSM8K, MATH에서도 개선을 보고한다.

주요 메시지는 memory가 단순 factual lookup만 돕는 것이 아니라는 점이다. Static local pattern을 memory가 담당하면, backbone의 early layer가 더 빨리 prediction-ready representation에 도달하고, 남은 effective depth를 reasoning에 사용할 수 있다는 해석이다.

Long-context evaluation도 중요하다. Engram은 local dependency를 lookup으로 넘겨 attention capacity를 global context에 더 쓰게 만든다고 주장한다. RULER의 Multi-Query NIAH 같은 retrieval task에서 MoE baseline 대비 큰 개선을 보인다.

## 5-2. What really matters in the experiments

### 1) Iso-parameter and iso-FLOPs 비교

Engram-27B는 MoE-27B와 total parameter 및 activated parameter를 맞춘 비교다. 따라서 단순히 parameter가 많아서 좋아졌다는 설명을 어느 정도 줄인다. 핵심은 sparse budget을 routed experts에만 쓰는 것보다 일부를 lookup memory에 배정하는 것이 낫다는 것이다.

### 2) U-shaped sparsity allocation

논문은 MoE expert capacity와 Engram memory 사이의 allocation ratio를 바꾸며 validation loss를 본다. Pure MoE도 suboptimal이고, memory가 너무 많아 computation capacity가 부족해도 나쁘다. 중간의 hybrid allocation에서 optimum이 나온다.

이 결과가 논문의 가장 중요한 scaling 메시지다. Sparse scaling은 하나의 축이 아니라, computation과 memory 사이의 budget allocation 문제다.

### 3) Mechanistic analysis

LogitLens와 CKA 분석은 Engram이 early layer의 static reconstruction 부담을 줄인다는 가설을 뒷받침한다. Engram representation은 baseline의 더 깊은 layer와 유사하게 나타나며, 이는 effective depth 증가로 해석된다.

### 4) Functional dichotomy

Engram output을 inference에서 suppression하면 factual knowledge benchmark는 크게 무너지지만, reading comprehension은 상대적으로 덜 무너진다. 이는 Engram이 parametric knowledge repository에 가까운 역할을 하고, context-grounded comprehension은 여전히 attention backbone에 크게 의존한다는 해석을 가능하게 한다.

### 5) System throughput

논문은 100B-parameter Engram table을 host DRAM에 offload한 inference harness도 제시한다. Dense-4B와 Dense-8B backbone에서 throughput penalty가 작게 나타난다. 완성된 serving stack은 아니지만, deterministic prefetching이 실용적인 방향일 수 있음을 보여준다.

# 6. Limitations

1. 완전한 product-scale serving 검증은 아니다.
   - nano-vLLM 기반 prototype과 controlled benchmark는 의미가 있지만, real production scheduler, batching, multi-tenant serving, cache eviction까지 검증한 것은 아니다.

2. Memory가 stale knowledge를 어떻게 다루는지는 열려 있다.
   - Engram은 model-internal parametric memory다.
   - RETRO나 RAG처럼 외부 memory를 쉽게 update하는 구조와는 다르다.

3. Hash collision과 polysemy는 근본적으로 남는다.
   - Context-aware gating으로 완화하지만, collision이 많은 rare pattern에서는 noise가 생길 수 있다.
   - Higher-order n-gram scaling이 항상 좋은지도 더 큰 scale에서 봐야 한다.

4. MoE baseline과의 공정성은 넓은 환경에서 재검증이 필요하다.
   - 논문은 iso-parameter and iso-FLOPs를 맞추지만, architecture choice, data curriculum, optimizer, layer placement에 따라 결과가 달라질 수 있다.

5. Security and privacy implication이 있다.
   - Static memory table이 factual knowledge repository 역할을 한다면, memorization, editing, redaction, attribution 문제가 중요해진다.
   - 어떤 memory slot이 어떤 training pattern을 저장했는지 추적하는 도구가 필요하다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문은 sparse model 설계를 MoE routing만으로 보지 않게 만든다. 최근 LLM architecture 논의에서 "더 많은 parameter를 어떻게 inactive로 둘 것인가"는 중요했다. 하지만 대부분 그 inactive parameter는 expert capacity였다. Engram은 그 일부를 memory capacity로 쓰자는 제안이다.

가장 중요한 메시지는 static knowledge와 dynamic reasoning을 같은 FFN compute로 처리할 필요가 없다는 점이다. 실제 모델이 자주 쓰는 local phrase, entity, idiom, formulaic pattern을 매번 계산으로 복원한다면, early layer compute가 낭비될 수 있다. 이를 lookup으로 넘기면 backbone은 더 복잡한 context integration에 집중할 수 있다.

## 7-2. Reuse potential

재사용해볼 만한 포인트는 다음과 같다.

1. MoE plus memory hybrid allocation.
   - Sparse budget을 expert와 memory로 나누는 실험 설계는 다른 architecture에도 적용 가능하다.

2. Deterministic memory prefetch.
   - Token-derived memory index는 serving stack에서 미리 계산할 수 있다.
   - CPU offload, DRAM cache, NVMe long-tail storage 설계와 연결된다.

3. Context-aware gating.
   - Static lookup을 그대로 더하지 않고, hidden state가 받아들일지 결정하게 하는 설계는 RAG fusion이나 adapter fusion에도 유용하다.

4. Mechanistic analysis protocol.
   - LogitLens, CKA, memory suppression을 조합해 memory module이 실제로 어떤 기능을 맡는지 볼 수 있다.

5. Long-context attention offload.
   - Local dependency를 memory로 넘기면 attention은 global context에 집중할 수 있다는 관점은 KV cache, retrieval, attention compression 연구와도 연결된다.

## 7-3. Follow-up papers

- DeepSeekMoE.
- DeepSeek-V3.
- RETRO and REALM.
- Product Key Memory.
- Byte Latent Transformer.
- SuperBPE and SCONE.
- Mechanistic interpretability work on FFN as key-value memory.

# 8. Summary

- Engram은 MoE의 conditional computation과 별개로 conditional memory라는 sparsity axis를 제안한다.
- Hashed n-gram lookup, tokenizer compression, context-aware gating, multi-branch integration으로 static local pattern을 memory에서 가져온다.
- Sparse budget을 MoE expert와 Engram memory 사이에 나누면 pure MoE보다 나은 U-shaped allocation optimum이 나타난다.
- Engram은 factual knowledge뿐 아니라 reasoning, code, math, long-context retrieval에도 개선을 보인다고 보고된다.
- 다만 serving stack, stale memory, hash collision, privacy, broader baseline 검증은 후속 확인이 필요하다.
