---
layout: single
title: "Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning Review"
categories: Study-concept
tag: [Nemotron, Mamba-2, TrainRecipe]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2604.12374)

Nemotron 3 Super는 "좋은 open reasoning model이 하나 더 나왔다" 정도로 읽기에는 아까운 기술 리포트다. 이 보고서의 진짜 가치는 **LatentMoE + hybrid Mamba-Attention + MTP + NVFP4 pretraining + staged post-training + deployment quantization**을 하나의 end-to-end recipe로 묶었다는 데 있다. 즉, 이것은 architecture paper이면서 동시에 **train recipe report**, **agentic RL systems report**, **serving note**다.

요즘 open model을 읽다 보면 성능 차이가 단순히 backbone에서만 나오지 않는다는 사실이 점점 분명해진다. 어떤 토큰 믹서를 썼는가도 중요하지만, 실제로는 **어떤 precision으로 학습했는가**, **긴 문맥을 어떤 방식으로 붙였는가**, **agentic RL을 어떻게 운영했는가**, **최종 checkpoint를 어떤 hardware path에 맞춰 내보냈는가**가 더 큰 차이를 만들 때가 많다. Nemotron 3 Super는 그 흐름을 아주 정직하게 드러낸다.

> 한 줄 요약: Nemotron 3 Super는 **120B total / 약 12B active 규모의 LatentMoE 기반 hybrid Mamba-Transformer**에 **NVFP4 25T pretraining**, **2-stage SFT + multi-stage RL + MTP healing**, **FP8/NVFP4 deployment checkpoint**까지 연결한, 말 그대로 **agentic open model full-stack report**다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- architecture 하나보다 **시스템 전체를 어떻게 맞물리게 설계하는가**를 보여준다.
- long-context, tool use, SWE, quantization, throughput 같은 **실서비스 관점의 문제**가 한 문서 안에서 이어진다.
- weights만 푸는 것이 아니라 **dataset, checkpoint, recipe, serving artifact**까지 같이 열어 둬서 재사용 포인트를 찾기 쉽다.

이 보고서의 핵심은 "NVIDIA가 큰 모델 하나 잘 만들었다"가 아니다. 오히려 **현대 open reasoning model의 경쟁력은 architecture, data curriculum, RL infra, deployment format을 같이 맞춰야 나온다**는 점을 가장 노골적으로 보여주는 사례에 가깝다.

# 1. Problem Setting

## 1-1. Problem definition

- 이 리포트가 겨냥하는 핵심 문제는 단순히 "더 정확한 LLM 만들기"가 아니다.
- 더 직접적으로는 아래 네 가지를 동시에 만족시키려는 문제다.
  1. **긴 문맥**에서도 버틸 것
  2. **reasoning / coding / tool use / SWE** 같은 agentic workload를 다룰 것
  3. **inference throughput**이 높을 것
  4. 실제 배포 가능한 **low-precision checkpoint**까지 제공할 것
- 특히 multi-agent나 long-horizon agent loop에서는 생각보다 토큰이 아주 빨리 불어난다. 긴 history, intermediate tool outputs, reasoning trace, retrieval context가 계속 누적되기 때문이다.
- 따라서 이 문제는 단순 benchmark problem이 아니라, **context explosion + thinking tax + deployment cost**를 함께 다루는 문제다.

## 1-2. Why previous approaches are insufficient

- **dense Transformer**는 정확도는 강하지만, long-context로 갈수록 KV cache 비용이 빠르게 커진다.
- **MoE alone**는 active parameter를 줄여 FLOP efficiency를 높여주지만, attention 자체의 문맥 비용이나 serving latency 병목을 자동으로 해결해주지는 않는다.
- **hybrid SSM / Mamba 계열**은 throughput에는 유리하지만, pure linear-time mixer만으로는 정밀한 associative recall이나 global routing이 약해질 수 있다.
- **post-training**도 쉽지 않다. tool use, terminal use, SWE, long-context QA 같은 long-horizon RL task는 rollout 비용이 높고 verifier도 느리다.
- **after-the-fact quantization**만으로는 배포 효율을 얻을 수 있어도, 학습 자체가 deployment path를 고려하지 않았다면 품질 손실이나 예기치 않은 failure mode가 생기기 쉽다.
- 논문이 직접 짚는 또 하나의 중요한 문제는, **single-stage SFT가 long-input-short-output 시나리오를 망가뜨릴 수 있다**는 점이다. 이건 agent/RAG 시스템에서 꽤 치명적이다.

즉 기존 접근들의 한계는 하나의 성능 숫자가 낮다는 것이 아니라, **capacity, context, decode speed, agentic behavior, quantization**이 서로 다른 층위의 문제인데 이를 따로따로 다루는 경우가 많았다는 데 있다.

# 2. Core Idea

## 2-1. Main contribution

- **LatentMoE**: expert routing과 computation을 latent space에서 수행해 accuracy per FLOP뿐 아니라 **accuracy per parameter / byte**까지 같이 최적화하려 한다.
- **Hybrid Mamba-Attention backbone**: stack의 대부분은 Mamba-2로 처리하고, 소수의 attention layer를 **global anchor**처럼 배치한다.
- **Multi-Token Prediction (MTP)**: shared-weight MTP head를 넣어 모델 품질을 올리면서, 별도 draft model 없이 **native speculative decoding**을 가능하게 한다.
- **NVFP4 pretraining**: Nemotron 3 family에서 처음으로 NVFP4로 pretraining을 수행해 low-precision을 학습 단계부터 끌어들인다.
- **Two-phase pretraining + LC-Phase**: 25T token pretraining을 broad phase와 HQ phase로 나누고, 마지막에 long-context continuous pretraining을 붙인다.
- **2-stage SFT + 3-stage RL + MTP healing**: post-training도 단일 stage가 아니라, long-output bias를 다루는 SFT, multi-environment RLVR, SWE-RL, RLHF, 그리고 마지막 MTP healing으로 나눈다.
- **FP8 / NVFP4 PTQ**: 모델 소개에서 끝나지 않고, 실제 Hopper/Blackwell 배포용 checkpoint까지 같이 낸다.

## 2-2. Design intuition

이 리포트를 읽으면서 가장 좋았던 점은, "throughput 문제를 한 군데서만 풀지 않는다"는 것이다. 설계 직관을 내 방식으로 정리하면 아래 4단계다.

### 1) expert 단계에서 비용을 줄인다

- MoE를 쓰더라도 expert matrix read와 all-to-all routing은 여전히 비싸다.
- 그래서 hidden dimension 전체에서 expert를 돌리지 않고, **latent space로 먼저 압축한 뒤 routing / expert computation**을 수행한다.

### 2) sequence mixer 단계에서 KV cache 압박을 줄인다

- stack의 대부분을 Mamba-2로 두면 generation 중 state size가 constant에 가깝게 유지된다.
- 하지만 pure Mamba로 가면 recall과 global interaction이 약해질 수 있으므로, attention을 완전히 없애지 않고 **global anchor**처럼 배치한다.

### 3) decoding 단계에서 next-token bottleneck을 줄인다

- 긴 reasoning trace나 code generation에서는 decode latency가 커진다.
- MTP는 여기서 단순 보조 loss가 아니라, **실제 speculative decoding path**를 여는 모듈로 쓰인다.

### 4) behavior 단계에서 long-horizon task를 따로 다룬다

- long-horizon agentic task는 일반 chat RL과 runtime 성격이 다르다.
- 그래서 RLVR, SWE-RL, RLHF를 한 번에 섞지 않고, **rollout cost와 task structure에 맞춰 나눠서** 학습한다.

결국 이 논문은 "하나의 멋진 layer를 제안했다"보다, **capacity / context / decoding / behavior / deployment를 각기 다른 레이어에서 동시에 조정하는 시스템 설계**로 읽는 편이 더 정확하다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | long-context reasoning과 agentic deployment를 높은 throughput으로 지원하는 open model 구축 |
| Backbone | 120.6B total / 12.7B active per forward pass의 hybrid Mamba-2 + LatentMoE + Attention |
| Key modules | LatentMoE, periodic global attention anchors, shared-weight 2-layer MTP |
| Pretraining | 25T-token two-phase pretraining + long-context CPT |
| Post-training | 2-stage SFT -> RLVR -> SWE-RL -> RLHF -> MTP healing |
| Deployment story | BF16 / FP8 / NVFP4 checkpoint와 vLLM / SGLang / TRT-LLM artifact 제공 |
| Difference from prior work | sparse scaling, long-context mixer, speculative decoding, agentic RL, quantization을 한 보고서 안에서 같이 다룸 |

## 3-2. Module breakdown

### 1) LatentMoE

LatentMoE는 이 보고서에서 가장 "NVIDIA다운" 부분이다. 논문은 기존 MoE가 FLOP 기준으로는 좋아 보이지만, 실제 온라인 serving에서는 **memory bandwidth, expert weight read, routing communication**이 더 지배적인 경우가 많다고 본다.

- 입력 token을 full hidden dimension에서 바로 expert로 보내지 않는다.
- 먼저 **learnable down-projection**으로 latent dimension으로 압축한다.
- routing과 expert computation을 **latent space 내부**에서 처리한다.
- 이후 **up-projection**으로 다시 model dimension으로 올린다.

Table 1 기준으로 Super는 다음과 같은 구성을 가진다.

- total layers: **88**
- model dim: **4096**
- total experts per layer: **512**
- activated experts: **top-22**
- latent size: **1024**

중요한 건 이 구조가 단순 compression이 아니라는 점이다. 논문은 latent routing으로 아낀 cost를 다시 **더 많은 expert 수와 더 큰 active expert budget**에 재투자한다. 즉 목표는 "expert를 싸게 쓰는 것"이 아니라, **같은 비용으로 더 세분화된 specialization을 얻는 것**이다.

내가 보기엔 이 모듈의 핵심은 accuracy per FLOP보다 **accuracy per byte**를 전면에 꺼냈다는 데 있다. MoE를 실제 서비스 관점에서 읽으면 훨씬 자연스러운 기준이다.

### 2) Hybrid interleaving and global anchors

Nemotron 3 Super는 Mamba를 attention의 완전한 대체재로 취급하지 않는다.

- stack의 대부분은 **Mamba-2**가 담당한다.
- attention layer는 소수만 남겨 **global anchor**처럼 삽입한다.
- 이 위에 LatentMoE가 결합된다.

Figure 2를 보면 layer pattern은 **Mamba-2와 LatentMoE의 반복이 중심**이고, attention은 중간중간 들어가 전체 token interaction을 복구하는 형태다. 이 구조의 의도는 꽤 명확하다.

- Mamba-2: long-context throughput, constant-sized state, lower memory overhead
- Attention: full-token interaction, precise routing, recall safety net
- LatentMoE: sparse capacity scaling

즉 이 모델은 "Mamba냐 Transformer냐"가 아니라, **어디까지를 linear-time path로 보내고, 어디에서 full interaction을 허용할 것인가**를 정한 hybrid 설계다.

### 3) Multi-Token Prediction (MTP)

MTP는 이 보고서에서 꽤 실무적인 역할을 한다.

- shared-weight MTP head를 **2개** 둔다.
- pretraining에서부터 사용하고, SFT에서도 유지한다.
- post-training 마지막에는 **MTP healing** stage를 따로 둬서 RL 이후 draft quality를 다시 복구한다.

이게 중요한 이유는 두 가지다.

1. **quality side**
   - 여러 미래 token을 동시에 예측하도록 만들면, 모델이 조금 더 긴 의존성을 학습하게 된다.

2. **serving side**
   - 별도 draft model 없이도 speculative decoding을 구성할 수 있다.

Table 2 기준으로 SPEED-Bench에서 draft length 7일 때 Nemotron 3 Super의 **average acceptance length는 3.45**다. 이는 논문이 MTP를 단순 marketing point가 아니라, 실제 decode path에서 작동하는 모듈로 보고 있다는 증거다.

### 4) 왜 이 조합이 중요한가

이 모델의 설계는 각 모듈이 따로 멋진 것이 아니라, 서로 **다른 병목을 담당한다**는 점에서 의미가 있다.

- LatentMoE는 **capacity / bandwidth 병목**을 푼다.
- Mamba-2는 **context-length / KV cache 병목**을 푼다.
- attention anchors는 **global recall 병목**을 막는다.
- MTP는 **decode latency 병목**을 줄인다.

내가 보기엔 Nemotron 3 Super는 "hybrid model"이라는 한 단어로 요약하면 오히려 놓치는 게 많다. 더 정확히는, **서로 다른 병목을 다른 메커니즘으로 나눠서 해결하는 layered systems design**이다.

# 4. Training / Data / Recipe

## 4-1. Data

### 1) Two-phase pretraining

Nemotron 3 Super의 pretraining은 **총 25T tokens**로 구성된다.

- **Phase 1 (20T)**: diversity와 broad coverage 중심
- **Phase 2 (5T)**: high-quality source와 benchmark accuracy 중심

논문은 Phase 2에서 Wikipedia 같은 high-quality source 비중을 올린다고 설명한다. 즉 이 모델은 처음부터 끝까지 같은 분포를 보는 것이 아니라, 후반부에 좀 더 **quality-focused curriculum**으로 이동한다.

또 specialized synthetic data도 적극적으로 사용한다.

- synthetic code concepts / algorithmic data
- economics
- formal logic
- multiple choice

이 부분은 단순 data scaling보다 **capability shaping**에 가깝다. 즉 pretraining 단계부터 reasoning 성격을 미리 밀어두는 셈이다.

### 2) Long-context phase

pretraining 마지막에는 **LC-Phase**가 따로 붙는다.

- 방식은 continuous pretraining (CPT)
- Nemotron 2 / Nemotron 3 Nano에서 사용한 long-context document QA dataset을 재사용
- 1M context 지원을 위한 별도 long-context adaptation

중요한 건, 이 보고서가 long context를 단순 config 항목으로 다루지 않는다는 점이다. **base training 끝나고 LC-Phase를 따로 둔다**는 것은 long context가 아키텍처 옵션이 아니라, 별도의 학습 단계라는 뜻이다.

### 3) SFT data

SFT도 꽤 공격적으로 확장됐다.

- **7M+ total samples**
- Figure 12 기준으로 **80B tokens**
- agentic task 비중을 Nano보다 훨씬 늘림

특히 눈에 띄는 건 tool-use data scale이다.

- specialized customer-service style pipeline에서 **279,116 conversations / 838 domains**
- general-purpose synthetic tool-calling pipeline에서 **1.5M trajectories**

즉 Nemotron 3 Super의 SFT는 일반적인 chat+instruction blend가 아니라, **tool calling / multi-turn interaction / agentic workflow를 전제로 한 corpus**에 가깝다.

## 4-2. Training strategy

### 1) Pretraining recipe

논문이 공개하는 pretraining recipe에서 중요한 점은 다음과 같다.

- schedule: **Warmup-Stable-Decay (WSD)**
- initial warmup: **200B tokens**
- final 5T tokens에서 **minus-sqrt decay**
- optimizer: **AdamW**
- weight decay: **0.1**
- sequence length: **8192**
- batch size: **3072 sequences**
- batch token count: 약 **25.17M tokens**

숫자 하나하나보다 더 중요한 건, 이 리포트가 "큰 모델이라 잘 됐다"로 끝나지 않고 **어떤 schedule과 curriculum을 썼는지**를 꽤 구체적으로 보여준다는 점이다.

### 2) Two-stage SFT loss

SFT는 이 논문에서 생각보다 중요한 포인트다. 저자들은 single-stage SFT가 **long-input-short-output** 상황을 망친다고 관찰하고, 그래서 2-stage SFT를 사용한다.

- **Stage 1**: token-level global average
  - 긴 reasoning output을 강하게 학습시키는 역할
- **Stage 2**: sample-level average
  - 긴 출력 sample이 loss를 과도하게 지배하지 않도록 조정

설정도 다르다.

- Stage 1: **256K packed sequence**, global batch size 64
- Stage 2: **512K packed sequence**, global batch size 32, 512K long-context data 포함

이건 꽤 재사용 가치가 큰 아이디어다. 실제 서비스에서는 "긴 입력 + 짧은 정답"이 정말 많기 때문이다. 문서 QA, RAG, tool routing, bug triage 모두 그렇다.

### 3) Reasoning control

Nemotron 3 Super는 세 가지 reasoning mode를 학습한다.

- **reasoning-off**
- **regular**
- **low-effort**

이 중 low-effort reasoning은 Super에서 새로 강조되는 부분이다.

- low-effort sample은 SFT 전체의 **2%**
- GPT-OSS-120B의 low-effort mode generated data를 사용
- math reasoning, STEM QA, instruction following을 포함

또 reasoning-off mode는 reasoning trace를 일부 제거해 만들고, budget control을 위해 **350-step semi-on-policy SFT**를 추가해 **12%의 reasoning traces를 random budget으로 truncate**한다.

즉 reasoning control은 단순 inference trick이 아니라, **학습 단계부터 mode separation과 budget control을 심어두는 구조**다.

### 4) Multi-stage RL

Figure 12를 기준으로 post-training pipeline을 보면 아래 흐름이 핵심이다.

| Stage | Primary purpose | Why it matters |
| --- | --- | --- |
| SFT | behavior foundation, tool-use format, reasoning modes | RL 이전의 broad behavior prior 형성 |
| RLVR | 21 environments / 37 RL datasets 기반 multi-environment training | breadth를 유지한 채 verifiable reward 최적화 |
| SWE-RL | end-to-end software engineering | 느리고 긴 rollout을 별도로 최적화 |
| RLHF | instruction following / robustness / interaction quality | 마지막 behavior alignment |
| MTP healing | MTP head accuracy recovery | RL 이후 speculative decoding 품질 보정 |

RL 쪽에서 내가 중요하게 본 포인트는 아래와 같다.

- **unified RLVR**: 21 environments, 37 RL datasets
- domains: math, code, STEM, safety, chat, instruction following, long context, puzzles, agentic tool use
- 이유: single-environment RL은 다른 benchmark를 심하게 깎아먹고, unified mixture가 breadth 유지에 더 낫다고 본다.

또 SWE-RL은 왜 분리하는가도 분명하다.

- SWE rollout은 **느리다**
- context가 **길다**
- co-train하면 throughput bottleneck이 생긴다

그래서 end-to-end SWE RL은 별도 stage로 뽑는다. 이 판단은 꽤 설득력 있다. agentic RL을 실제로 굴려본 사람일수록 왜 이렇게 했는지 바로 이해된다.

### 5) Algorithm and infrastructure

알고리즘은 크게 보면 **asynchronous GRPO**다.

- training과 inference를 분리
- inference worker가 trajectory를 계속 생성
- rollout buffer가 batch를 만들면 training engine이 update
- 새 weight가 나오면 inference worker에 바로 push
- policy lag를 줄이기 위해 inference worker는 최신 model보다 최대 한 step만 뒤처지게 제한
- training / inference mismatch 안정화를 위해 importance ratio를 mask

다시 말해, 이 보고서의 RL 핵심은 새로운 objective 발명보다 **대규모 agentic RL을 어떻게 운영 가능한 형태로 만들 것인가**에 더 가깝다.

여기에 PivotRL도 붙는다.

- long-horizon agentic RL은 online interactive rollout 비용이 너무 비싸다.
- PivotRL은 offline expert trajectory에서 정보량이 큰 pivot turn을 골라 RL에 재사용한다.
- agentic programming, search, terminal use, conversational tool use에 적용한다.

개인적으로 이 부분이 꽤 좋았다. NVIDIA가 agentic RL을 "좋아 보이는 demo" 수준이 아니라, **실제로 감당 가능한 training system**으로 바꾸려 했다는 흔적이 선명하다.

## 4-3. Engineering notes

### 1) Serving artifact가 꽤 실전적이다

HF model card와 repo를 보면 단순히 weights만 던져놓은 수준이 아니다.

- `enable_thinking=True/False`로 reasoning on/off 제어 가능
- vLLM / SGLang / TRT-LLM용 quick start 제공
- 1M context는 기본값이 아니라, serving flags를 명시적으로 켜야 사용 가능
- BF16 checkpoint 기준 **기본 최소 요구사항은 8× H100-80GB**, B200/B300에서는 2 GPU 구성이 가능하다고 안내한다

즉 논문과 artifact 사이 간극이 비교적 작다.

### 2) Open-source recipe와 tech report 결과는 다르다

공식 Nemotron repo는 아주 중요한 단서를 남긴다.

- 공개 recipe는 **open-sourced subset of training data only**를 사용한다.
- 따라서 **tech report benchmark와 동일 결과를 기대하면 안 된다**.

이 문장은 실무적으로 매우 중요하다. "recipe를 공개했다"와 "논문 결과를 그대로 재현할 수 있다"는 전혀 다른 말이기 때문이다.

### 3) Quantization은 paper의 부록이 아니라 본편이다

이 논문은 quantization을 나중에 붙인 marketing appendix처럼 다루지 않는다.

- FP8: **W8A8** for Hopper
- NVFP4: **W4A4** for Blackwell
- AutoQuantize로 mixed-precision assignment 탐색

특히 Table 8 기준으로 FP8 / NVFP4 optimized model은 BF16 대비 품질을 꽤 잘 유지한다. 또 PTQ 전체 mixed-precision search는 **single B200 node with 8 GPUs에서 2시간 미만**으로 끝났다고 보고한다.

### 4) Mamba state quantization은 생각보다 까다롭다

가장 재미있었던 엔지니어링 디테일 중 하나는 **Mamba SSM cache**다.

- 단순히 FP16 cache로 내리면 verbosity가 크게 늘 수 있다.
- Table 9 설명 기준으로 W8A8와 결합 시 **최대 40% verbosity increase**가 발생할 수 있다.
- 저자들은 stochastic rounding을 적용해 이 문제를 해결한다.

이건 아주 좋은 교훈이다. Mamba 계열 모델은 attention KV cache만 보면 안 되고, **state cache precision 자체가 generation behavior를 건드릴 수 있다**.

# 5. Evaluation

## 5-1. Main results

### 1) Base model 결과가 먼저 중요하다

개인적으로 Nemotron 3 Super를 평가할 때 가장 먼저 보는 표는 post-trained table이 아니라 **Table 4 (base model)** 이다. 이유는 간단하다. architecture와 data curriculum이 정말 먹혔는지는 post-training보다 **base checkpoint**에서 더 잘 드러나기 때문이다.

Table 4에서 눈에 띄는 결과는 아래와 같다.

- **MMLU 86.01**
- **MMLU-Pro 75.65**
- **GPQA-Diamond 60.00**
- **MATH 84.84**
- **AIME 2024 pass@32 = 53.33**
- **HumanEval 79.40**
- **MMLU Global Lite 85.72**
- **RULER 1M 71.00**

Ling-flash-Base-2.0, GLM-4.5-Air-Base와 비교하면 base model 단계에서 이미 꽤 강하다. 이건 중요한 포인트다. post-training만 세게 한 모델이 아니라, **base recipe 자체가 경쟁력 있다**는 뜻이기 때문이다.

### 2) Post-trained model은 "전면적 승리"보다 "경쟁적이고 빠른 generalist"에 가깝다

Table 5를 보면 headline은 분명하다. Nemotron 3 Super는 **Qwen3.5-122B-A10B, GPT-OSS-120B와 경쟁 가능한 open model**이다. 다만 읽는 방식은 조금 조심해야 한다.

가장 깔끔한 강점은 long-context와 일부 tool-augmented reasoning이다.

- **RULER 1M: 91.64**
  - Qwen3.5: 91.33
  - GPT-OSS: 22.30
- **HMMT Feb25 (with tools): 94.73**
  - Qwen3.5: 89.55
- **HLE (with tools): 22.82**
  - GPT-OSS: 19.0
- **HMMT Feb25 (no tools): 93.67**
  - Qwen3.5: 91.40
  - GPT-OSS: 90.00

반대로 "universal best model"로 읽으면 과하다.

- **MMLU-Pro: 83.73 < 86.70 (Qwen3.5)**
- **GPQA (no tools): 79.23 < 86.60 (Qwen3.5)**
- **Terminal Bench Core 2.0: 31.00 < 37.50 (Qwen3.5)**
- **SWE-Bench (OpenHands): 60.47 < 66.40 (Qwen3.5)**
- **MMLU-ProX: 79.36 < 85.06 (Qwen3.5)**

즉 내 해석으로는 Nemotron 3 Super의 포지션은 "benchmark 하나를 끝장내는 단일 챔피언"보다는, **throughput, long context, tool use, reasoning을 높은 수준으로 묶은 deployable generalist**에 더 가깝다.

### 3) Throughput / PTQ story는 꽤 설득력 있다

Figure 1은 이 보고서의 핵심 그림 중 하나다.

- 8K input / 64K output 조건에서
  - **GPT-OSS-120B 대비 최대 2.2× throughput**
  - **Qwen3.5-122B 대비 최대 7.5× throughput**

또 quantized checkpoint 쪽도 인상적이다.

- FP8 / NVFP4 optimized checkpoint는 BF16 대비 major benchmark를 꽤 잘 유지한다.
- Table 8 설명 기준으로 mixed-precision PTQ 이후 **99.8% median accuracy relative to BF16**를 달성했다고 한다.

이건 단순히 "작동한다" 수준이 아니라, **논문 안에서 train-time design과 deployment-time path가 이어진다**는 증거다.

## 5-2. What really matters in the experiments

이 논문에서 진짜 의미 있는 실험 포인트는 아래 네 가지라고 본다.

### 1) base vs post-trained를 분리해서 봐야 한다

많은 모델 리포트는 post-training 표만 보면 architecture 기여와 alignment 기여가 섞인다. Nemotron 3 Super는 Table 4와 Table 5가 둘 다 있어서, 최소한 **base backbone + data curriculum의 힘**과 **post-training의 힘**을 어느 정도 나눠서 볼 수 있다.

### 2) long-context win은 비교적 신뢰도가 높다

throughput 그림, Mamba-heavy hybrid design, RULER 1M 결과가 서로 같은 방향을 가리킨다. 이 조합은 꽤 설득력 있다. 즉 long-context 쪽에서는 이 모델의 강점을 비교적 일관되게 읽을 수 있다.

### 3) agentic benchmark는 harness 의존성이 크다

SWE-Bench도 OpenHands / OpenCode / Codex로 나뉘고, TauBench나 TerminalBench도 environment 차이가 크다. 그래서 이 영역은 "모델이 더 똑똑하다"보다 **어떤 harness와 tool protocol을 같이 썼는가**의 영향을 많이 받는다.

Nemotron 3 Super가 agentic에서 강한 건 맞지만, 그 강점을 architecture 한 줄의 승리로 해석하면 너무 단순해진다.

### 4) cross-model comparison에는 caveat가 있다

논문은 GPT-OSS / Qwen3.5 비교에서, 공식 수치가 없을 때는 **reputable public aggregators**를 쓰거나 자체 평가를 수행했다고 밝힌다. 또 일부 benchmark는 아직 open-source evaluation stack에 완전히 onboard되지 않았다고 적는다.

즉 Table 5는 충분히 유용하지만, **완전히 동일한 실험실 조건의 apple-to-apple 표**라고 보기는 어렵다. 공개 글에서도 이 점은 같이 적는 편이 공정하다.

# 6. Limitations

이 보고서는 별도의 limitations section을 길게 두지는 않는다. 그래서 아래는 논문 내용과 artifact를 바탕으로 정리한 **내 주의점**이다.

1. **attribution이 흐리다.**  
   LatentMoE, hybrid Mamba-Attention, MTP, NVFP4 pretraining, two-phase data curriculum, multi-stage RL, PTQ가 모두 같이 들어간다. 따라서 "무엇이 최종 성능의 주원인인가"를 깔끔하게 분리하기 어렵다.

2. **post-trained model이 전 benchmark에서 최강인 것은 아니다.**  
   long-context와 일부 reasoning/tool setting은 강하지만, agentic 일부 harness, general knowledge, multilingual에서는 Qwen3.5가 앞선다.

3. **evaluation stack이 완전히 균일하지 않다.**  
   경쟁 모델 수치 일부는 official report, 일부는 public aggregator, 일부는 자체 평가를 섞어 쓴다. 또 BrowseComp with Search, Terminal Bench Core 2.0 등은 내부 scaffolding 또는 아직 완전히 open되지 않은 path가 섞인다.

4. **repo recipe와 tech report result는 일치하지 않는다.**  
   공식 repo가 스스로 밝혔듯, 공개 recipe는 open-source subset data만 사용하므로 tech report benchmark와 차이가 날 수밖에 없다.

5. **throughput / quantization 이야기는 NVIDIA stack 친화적이다.**  
   B200, NVFP4, TRT-LLM, vLLM, Blackwell 최적화가 강하게 들어가 있다. 따라서 다른 hardware stack에서 같은 효율 곡선이 그대로 나올지는 추가 검증이 필요하다.

6. **multilingual 해석은 표를 직접 보는 편이 안전하다.**  
   본문 서술과 Table 5 수치 사이에 약간의 긴장감이 있다. 공개 글에서는 서술보다 **표 숫자 자체**를 기준으로 쓰는 편이 더 보수적이다.

# 7. My Take

## 7-1. Why this matters for my work

- 이 리포트가 좋은 이유는, 현대 open reasoning model의 경쟁력을 **pipeline design** 관점으로 보게 만들기 때문이다.
- 특히 서비스 관점에서는 아래 질문들이 중요하다.
  - backbone을 어떻게 hybrid로 짤 것인가?
  - long-context를 training stage로 따로 둘 것인가?
  - long-horizon RL은 general RL과 분리할 것인가?
  - quantization을 논문 밖의 후처리로 둘 것인가, 아니면 처음부터 design loop에 넣을 것인가?
- Nemotron 3 Super는 이 질문들에 대해 꽤 일관된 대답을 준다.

내가 보기엔 이 논문의 핵심 메시지는 "Mamba가 좋다"나 "MoE가 좋다"가 아니다. 더 정확히는, **agentic open model은 architecture보다 stage separation과 deployment-awareness가 더 중요해지고 있다**는 것이다.

## 7-2. Reuse potential

내가 실무나 연구에서 바로 재사용해보고 싶은 포인트는 아래와 같다.

- **LatentMoE식 사고방식**: FLOP보다 byte / bandwidth 기준으로 expert design을 다시 볼 것
- **2-stage SFT loss**: long-output bias를 제어하는 간단하지만 실용적인 방법
- **SWE-RL 분리 전략**: 느리고 긴 rollout task를 broad RLVR와 분리하는 stage design
- **async RL infra**: training / inference decoupling, one-step lag 관리, rollout buffer 운영
- **reasoning mode control**: reasoning-off / regular / low-effort를 학습에서부터 심는 방식
- **quantization을 본편에 포함**: PTQ와 serving artifact를 연구 결과에서 분리하지 않는 태도

이 중에서 가장 실용적인 건 2-stage SFT와 stage-separated RL이라고 본다. 120B 모델이 아니어도 충분히 응용 가능하다.

## 7-3. Follow-up papers

- **Nemotron 3 Nano**  
  Super가 무엇을 계승했고 무엇을 확장했는지 보기 위한 기준점.

- **Nemotron-Cascade 2**  
  NVIDIA가 이후 smaller open model에서 post-training을 어떻게 더 공격적으로 밀었는지 연결해서 보기 좋다.

- **PivotRL**  
  long-horizon agentic RL 효율 개선 아이디어를 더 자세히 보고 싶다면 직접 이어서 읽을 가치가 있다.

- **LatentMoE technical report**  
  Super의 가장 중요한 backbone scaling 아이디어를 따로 해부하고 싶다면 필수다.

# 8. Summary

- Nemotron 3 Super는 단순한 "큰 open model"이 아니라 **architecture + data + RL infra + quantization**을 묶은 full-stack report다.
- backbone의 핵심은 **LatentMoE + Mamba-heavy hybrid + global attention anchors + MTP**다.
- training의 핵심은 **25T two-phase pretraining, LC-Phase, 2-stage SFT, multi-stage RL, MTP healing**이다.
- 결과는 universal dominance보다는 **long-context / throughput / deployability가 강한 competitive generalist**로 읽는 편이 정확하다.
- 실무적으로 가장 재사용 가치가 큰 포인트는 **2-stage SFT, SWE-RL 분리, async RL infra, deployment-aware PTQ**다.
