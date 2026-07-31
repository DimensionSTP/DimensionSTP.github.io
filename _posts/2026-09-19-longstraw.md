---
layout: single
title: "LongStraw: Long-Context RL Beyond 2M Tokens under a Fixed GPU Budget Review"
categories: Study-concept
tag: [LLM, Long-Context, Reinforcement-Learning]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2607.14952)

[Code link](https://github.com/MindLab-Research/longstraw)

> 한 줄 요약: LongStraw는 multi-million-token prompt를 사용하는 GRPO에서 prompt graph와 모든 response graph를 동시에 유지하지 않고, prompt state를 architecture별로 보존한 뒤 response를 하나씩 replay해 gradient를 누적하는 long-context RL execution system이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Long-context RL의 병목을 attention FLOPs만이 아니라 state와 gradient의 lifetime 문제로 다시 정의한다.
- 동일한 prompt를 old policy, reference policy, 여러 policy response가 공유하는 GRPO 구조를 활용한다.
- Qwen3.6-27B의 GDN plus full attention과 GLM-5.2의 MLA, DSA, MoE처럼 서로 다른 architecture에 같은 transaction contract를 적용한다.
- 2M token을 넘는 실제 execution receipt를 memory, wall-clock, communication까지 공개한다.
- 단순한 context extension이 아니라 rollout, replay, distributed gradient finalization이 하나의 시스템으로 맞물려야 한다는 점을 보여준다.

Long-context inference와 long-context RL은 비슷해 보이지만 요구사항이 다르다. Inference는 한 번의 forward state만 관리하면 되지만, GRPO training은 하나의 prompt에서 여러 response를 생성하고, old policy와 reference policy의 log probability를 계산한 뒤, current policy graph로 각 response를 다시 평가해야 한다.

Conventional autograd schedule에서는 prompt graph, 여러 response graph, model weight, optimizer state, KV cache, recurrent state, collective communication buffer가 동시에 살아 있을 수 있다. Context가 수백만 token으로 늘어나면 attention kernel 하나를 최적화하는 것만으로는 이 lifetime overlap을 해결할 수 없다.

LongStraw의 핵심은 계산량을 없애는 것이 아니라, 무엇을 언제 resident memory에 두어야 하는지를 바꾸는 데 있다.

# 1. Problem Setting

## 1-1. Problem definition

GRPO에서 prompt를 $P$, group size를 $G$, 각 response를 $Y_i$라고 하자. 하나의 optimization step에는 대략 다음 작업이 필요하다.

1. Old policy로 response를 rollout한다.
2. Old policy의 token log probability를 계산한다.
3. Reference policy의 token log probability를 계산한다.
4. Group reward로 advantage를 만든다.
5. Current policy에서 response별 loss와 gradient를 계산한다.
6. 모든 response gradient를 모아 한 번의 optimizer step을 수행한다.

Policy objective의 대표적인 형태는 다음과 같다.

$$
L_{policy} = -\frac{1}{G}\sum_{i=1}^{G}\frac{1}{R_i}\sum_{t=1}^{R_i}
\min\left(\rho_{i,t}A_i,\mathrm{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)A_i\right)
$$

여기서 $R_i$는 response length이고, $\rho_{i,t}$는 current policy와 old policy의 token probability ratio다.

문제는 group member가 같은 prompt를 공유한다는 사실과 conventional execution이 이 공유 구조를 충분히 이용하지 못한다는 점이다. Prompt forward를 response마다 다시 계산하면 비싸고, 한 번 계산한 prompt graph를 모든 response가 끝날 때까지 유지하면 memory가 폭발한다.

LongStraw는 이를 다음 세 가지 lifetime으로 나눈다.

| Lifetime | 내용 | 기존 schedule의 문제 |
| --- | --- | --- |
| Prompt lifetime | 수백만 token prompt의 architecture state | 모든 response graph와 함께 오래 살아 있음 |
| Response lifetime | 각 response suffix의 activation과 gradient graph | Group size에 따라 동시에 늘어날 수 있음 |
| Step lifetime | Gradient bucket, optimizer state, distributed buffer | Prompt와 response state에 겹쳐 peak memory를 높임 |

## 1-2. Why previous approaches are insufficient

### 1) Attention optimization만으로는 부족하다

FlashAttention, context parallelism, paged KV cache는 attention execution을 줄이거나 분산할 수 있다. 하지만 RL에서는 reference scoring, old-policy scoring, policy replay, gradient accumulation이 추가된다. Forward attention이 가능하다는 사실만으로 backward transaction 전체가 가능해지는 것은 아니다.

### 2) Gradient checkpointing은 lifetime 구조를 바꾸지 않는다

Checkpointing은 activation을 재계산해 memory를 줄이지만, multi-million-token prompt를 response마다 다시 계산하면 wall-clock이 크게 늘어난다. 또한 recurrent state, sparse attention index, MoE routing, communication buffer 같은 architecture-specific state는 일반적인 activation checkpointing만으로 다루기 어렵다.

### 3) Group 전체를 한 번에 backward하면 response graph가 누적된다

GRPO group의 response를 batch로 묶으면 throughput은 좋아 보일 수 있지만, long context에서는 group member별 suffix graph가 동시에 resident하게 된다. Group size가 커질수록 peak memory가 response count에 비례해 늘 수 있다.

### 4) CPU offload만으로는 execution contract가 불명확하다

State를 CPU로 옮기는 것 자체는 간단해 보이지만, 어떤 page를 누가 소유하고, 어느 layer에서 다시 stage하며, global attention이나 sparse selection을 어떻게 복원할지가 정해져야 한다. Offload는 storage choice이고, LongStraw가 다루는 것은 state ownership과 replay semantics를 포함한 transaction design이다.

# 2. Core Idea

## 2-1. Main contribution

LongStraw의 핵심 transaction은 다섯 단계로 정리할 수 있다.

1. Prompt capture without autograd
   - 긴 prompt를 graph 없이 forward한다.
   - 이후 response replay에 필요한 architecture-specific state만 남긴다.

2. Old and reference scoring
   - 보존된 prompt state에서 old policy와 reference policy의 response log probability를 graph 없이 계산한다.

3. Group advantage construction
   - Response reward를 모아 group-relative advantage를 만든다.

4. Serial policy replay
   - Response를 한 번에 하나씩 current policy graph로 replay한다.
   - 해당 response의 backward가 끝나면 suffix graph를 해제하고 gradient만 누적한다.

5. One distributed finalization
   - Group 전체 response의 gradient가 누적된 뒤 distributed gradient finalization과 optimizer step을 한 번만 수행한다.

이 schedule의 핵심은 live autograd graph의 크기가 prompt plus all responses가 아니라, 현재 replay 중인 response suffix로 제한된다는 점이다.

## 2-2. Design intuition

LongStraw는 prompt를 immutable shared state처럼 취급한다. Prompt를 graph 없이 한 번 capture하고, group member는 그 경계 상태에서 각자 response suffix를 시작한다.

이를 수식으로 보면 중요한 근사 경계가 드러난다. Prompt boundary state를 $z_P(\theta)$, response loss를 $l(\theta,z_P(\theta))$라고 하면 full gradient는 다음과 같다.

$$
\nabla_\theta l(\theta,z_P(\theta)) =
\frac{\partial l}{\partial \theta}\bigg|_{z_P}
+
\frac{\partial l}{\partial z_P}\frac{\partial z_P}{\partial \theta}
$$

Qwen path의 response-only closure는 $z_P$를 `stopgrad` boundary로 두어 첫 번째 항을 유지하고, prompt를 통해 다시 흘러가는 두 번째 항을 생략한다. 따라서 이는 exact full-sequence gradient가 아니라 response-side objective다.

반면 GLM path는 논문이 full exact-2M online transaction으로 보고하는 별도의 execution path를 제공한다. 이 차이는 매우 중요하다. LongStraw라는 하나의 이름 아래에서도 architecture와 mode에 따라 gradient semantics가 다르므로, memory 숫자만 비교하면 안 된다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Qwen3.6-27B path | GLM-5.2 path |
| --- | --- | --- |
| Main structure | 48 recurrent GDN plus 16 full-attention layers | 78-layer MLA/DSA stack plus 256-expert top-8 MoE |
| Prompt state | Compact recurrent state plus CP8-sharded KV pages | CPU-resident CP-sharded MLA latent and DSA index-key pages |
| Replay | Blockwise response replay | One-layer staging and distributed sparse selection |
| Parallelism | Context parallelism with global LSE/output merge | CP32 plus EP32 |
| Gradient semantics | Response-only closure in main long run | Full exact-2M online transaction reported |
| Main goal | Bound graph by suffix and reuse prompt capture | Make full long-context MoE RL transaction executable |

## 3-2. Module breakdown

### 1) No-grad prompt capture

Prompt capture는 단순한 KV cache 생성이 아니다. 각 architecture가 response continuation에 요구하는 최소 state를 보존한다.

Qwen3.6-27B에서는 recurrent GDN layer의 compact state와 full-attention layer의 KV page가 필요하다. GLM-5.2에서는 MLA latent page와 DSA indexer key가 필요하다. LongStraw는 모든 activation을 보존하는 대신 continuation에 필요한 state만 explicit owner page에 둔다.

### 2) Physical page ownership

Long context에서는 logical sharding만으로 충분하지 않다. 실제 page가 어느 rank에 resident하는지 명확해야 peak memory와 communication을 예측할 수 있다.

Qwen path는 KV page를 CP8에 물리적으로 나누고, local attention 결과를 cross-rank log-sum-exp와 output merge로 결합한다. 이 방식은 각 rank가 전체 prompt KV를 복제하지 않으면서 global attention semantics를 구성한다.

### 3) Recurrent state restoration

GDN layer는 full KV history 대신 compact recurrent state를 유지한다. 각 response replay는 같은 prompt boundary state에서 출발해야 하므로, group member마다 state를 정확히 restore해야 한다. Restore가 in-place로 이전 response의 state를 오염시키면 group member가 독립 trajectory라는 GRPO 가정이 깨진다.

### 4) Blockwise suffix replay

Response 전체를 한 번에 graph로 만들지 않고 block 단위로 replay한다. Blockwise replay는 response activation peak를 줄이지만, block boundary에서 attention state와 recurrent state를 정확히 이어야 한다.

Group member는 serial하게 처리된다.

- Response 1 forward and backward
- Suffix graph release
- Response 2 forward and backward
- Suffix graph release
- ...
- One gradient finalization

이 구조에서는 group size가 늘어도 response graph가 동시에 늘지 않는다. 대신 wall-clock은 response replay 수에 따라 증가한다.

### 5) GLM layer staging

GLM-5.2는 78-layer MLA/DSA stack과 MoE tail을 가진다. 모든 long-context state를 GPU에 resident시키기 어렵기 때문에 CPU에 CP-sharded page를 두고 한 layer씩 stage한다.

DSA global sparse selection은 각 CP rank의 local candidate만 고르면 정확하지 않다. LongStraw는 IndexShare-aware global selection을 CP32 communication으로 복원한다. 선택된 response token은 EP32에서 top-8 expert로 dispatch된다.

### 6) Time-multiplexed rollout and training

GLM run은 같은 32개 H20 GPU를 두 mode로 사용한다.

- Rollout: vLLM TP8 plus PP4
- Training: Megatron TP1 plus CP32 plus EP32

Rollout과 training을 동시에 resident시키는 대신 phase별로 topology를 전환한다. 이는 fixed GPU budget에서 inference engine과 training engine을 함께 운영하기 위한 중요한 engineering choice다.

### 7) One optimizer transaction

Response마다 optimizer step을 수행하면 group-relative objective가 달라지고 communication overhead도 커진다. LongStraw는 response별 backward로 gradient를 누적하되, `finalize_model_grads`와 optimizer step은 group 전체에 대해 한 번 수행한다.

# 4. Training / Data / Recipe

## 4-1. Data

이 논문은 task-learning quality를 보여주는 일반적인 RL paper라기보다 systems feasibility paper에 가깝다. 따라서 workload도 두 execution path를 검증하도록 구성된다.

Qwen path에서는 long synthetic prompt와 supplied response/reward를 사용한 receipt가 중심이다. 별도의 짧은 online canary도 제공하지만, multi-million-token setting에서 반복적인 downstream learning curve를 제시하는 것은 아니다.

GLM path에서는 DAPO-MATH prompt를 사용하고, group size 2의 짧은 completion과 reward `[-1, +1]`로 full online transaction을 검증한다. 이 설정은 2M context에서 rollout부터 optimizer step까지 end-to-end로 실행되는지를 보여주는 데 목적이 있다.

## 4-2. Training strategy

### Qwen3.6-27B exact 2M receipt

논문이 보고한 대표 설정은 다음과 같다.

- Total positions: 2,097,152
- Prompt length: 2,088,960
- Response length: 8,192
- GPUs: 8 H20
- Context parallelism: CP8

Group size별 결과는 다음과 같다.

| Group size | Peak memory per rank | Step time |
| ---: | ---: | ---: |
| 2 | 97.503 GB | 5,198.780 sec |
| 8 | 97.711 GB | 6,785.225 sec |

Group size가 2에서 8로 늘 때 peak memory 증가는 0.208 GB에 그친다. 이는 serial replay가 group response graph의 동시 resident를 막는다는 핵심 claim을 지지한다. 다만 시간은 response replay 수에 따라 늘어난다.

### Qwen prefix-reuse extension

논문은 4,456,448 positions까지 확장한 prefix-reuse mode도 보고한다.

- Group size: 8
- Total replay cycles: 64
- Peak memory per rank: 83.894 GB
- Avoided prefix recapture time: 17,729.8 sec

이 mode는 optimizer update 뒤에도 이전 prompt state를 재사용하므로 prefix state가 stale해질 수 있다. 따라서 exact recapture mode와 같은 의미로 해석하면 안 된다.

### GLM-5.2 exact online transaction

GLM path는 rollout, mixed reward, 두 번의 78-layer backward, distributed gradient finalization, optimizer step을 32개 rank에서 수행한다. Context parallelism과 expert parallelism을 모두 32로 둔다.

Main recipe의 포인트는 hyperparameter보다 execution order다.

1. Rollout topology에서 response 생성
2. Training topology로 전환
3. Prompt state page 구성과 staging
4. Old/reference scoring
5. Response replay backward
6. MoE gradient communication
7. Finalize and optimizer step

## 4-3. Engineering notes

### 1) Receipt에 gradient semantics를 함께 기록해야 한다

Peak memory와 token count만으로는 충분하지 않다. Prompt-side VJP를 생략했는지, full-sequence gradient인지, prefix state가 stale한지 명시해야 한다.

### 2) State ownership audit가 필요하다

Qwen distributed update에는 page-owner의 dK/dV contribution이 replicated adapter에 완전히 synchronize되지 않는 caveat가 companion audit에 남아 있다. 시스템이 실행된다는 것과 모든 gradient path가 정확히 합쳐진다는 것은 별도 검증 항목이다.

### 3) Group scaling은 memory와 time을 분리해 봐야 한다

Serial replay는 memory scaling을 거의 평평하게 만들지만 group member 수만큼 compute는 남는다. Production recipe에서는 reward quality를 위해 group size를 늘릴 때 step time과 GPU utilization을 함께 봐야 한다.

### 4) Prompt reuse policy를 명시해야 한다

Same-step group reuse는 objective에 자연스럽지만, optimizer step을 넘는 reuse는 stale state가 된다. Reuse horizon과 recapture condition을 configuration으로 분리하는 것이 안전하다.

### 5) Architecture adapter가 핵심 abstraction이다

LongStraw를 다른 model에 이식하려면 다음 interface가 필요하다.

- Prompt state capture: prompt 경계 state를 graph 없이 포착한다.
- State serialization and ownership: state의 저장 형식과 rank별 소유권을 정의한다.
- State restoration: response replay 전에 필요한 state를 복원한다.
- Response replay: group member의 response suffix를 하나씩 다시 실행한다.
- Global attention or sparse selection merge: 분산된 attention 또는 sparse selection 결과를 합친다.
- Gradient finalization: 누적된 gradient를 정확한 collective 순서로 확정한다.

# 5. Evaluation

## 5-1. Main results

논문의 strongest evidence는 conventional training의 failure와 LongStraw transaction의 completion receipt다.

### 1) Qwen group size scaling

2M token에서 group size 2와 8의 peak memory가 거의 같게 유지된다. 이는 live graph가 group 전체가 아니라 current suffix에 bounded된다는 설계와 일치한다.

### 2) 4.45M positions

Prefix-reuse mode에서 4,456,448 positions의 resident prefix를 8번의 G=8 optimizer cycle에 재사용하며, 총 64 response replay를 83.894 GB per rank에서 완료한다. 이는 fixed H20 budget에서도 multi-million-token prompt state를 장기간 유지할 수 있음을 보여준다.

### 3) GLM full transaction

GLM-5.2 path는 heterogeneous attention과 MoE를 가진 model에서 exact-2M online transaction을 완료한다. 특히 DSA scratch, MLA page, expert dispatch가 겹치는 conventional full-sequence schedule의 memory peak를 one-layer staging으로 분해한다.

## 5-2. What really matters in the experiments

### 1) 이 논문은 quality SOTA paper가 아니다

LongStraw는 2M context로 RL을 오래 수행하면 reasoning accuracy가 얼마나 오르는지를 보여주지 않는다. Main evidence는 시스템이 transaction을 끝까지 수행하고 memory가 bounded된다는 것이다.

따라서 결과를 읽을 때 다음을 분리해야 한다.

- Feasibility: multi-million-token RL step이 실행되는가
- Correctness: distributed gradient와 state restore가 objective semantics를 지키는가
- Efficiency: conventional alternative보다 memory와 time이 얼마나 나은가
- Learning utility: 실제 task performance가 개선되는가

논문은 첫 번째와 두 번째 일부에 강하고, 네 번째는 후속 검증이 필요하다.

### 2) Response-only closure와 full gradient를 섞으면 안 된다

Qwen main path의 response-only closure는 prompt-side gradient를 생략한다. 이는 memory를 크게 줄이는 대신 optimized objective를 바꾼다. GLM exact transaction과 같은 표에서 token count만 비교하면 오해하기 쉽다.

### 3) Memory flatness는 좋은 결과지만 throughput은 별도다

Group size 8에서도 memory가 거의 늘지 않는 것은 중요한 결과다. 그러나 step time은 5,198.780 sec에서 6,785.225 sec로 증가한다. Memory feasibility와 practical iteration speed는 별도의 축이다.

### 4) Long context가 실제 reward에 필요한지 확인해야 한다

2M token execution이 가능하더라도 task가 그 context를 활용하지 못하면 시스템 비용만 늘어난다. Context utilization, evidence position, reward sensitivity, truncation ablation이 함께 필요하다.

# 6. Limitations

1. Learning quality evidence가 제한적이다.
   - Multi-million-token setting에서 반복적인 RL learning curve와 downstream benchmark improvement가 중심 결과로 제시되지 않는다.
   - 현재 결과는 systems execution receipt로 읽는 것이 안전하다.

2. Qwen response-only mode는 exact full-sequence gradient가 아니다.
   - Prompt boundary를 `stopgrad`로 두어 prompt-side VJP를 생략한다.
   - 이 근사가 어떤 task에서 성능에 영향을 주는지 추가 실험이 필요하다.

3. Prefix-reuse mode에는 staleness가 있다.
   - Optimizer update 이후에도 이전 prefix state를 재사용하면 current parameter와 state가 일치하지 않는다.
   - Recapture frequency와 bias trade-off가 필요하다.

4. Distributed gradient caveat가 남아 있다.
   - Qwen path의 page-owner dK/dV와 replicated adapter synchronization은 별도 audit 항목으로 남는다.
   - Execution completion과 numerical equivalence를 구분해야 한다.

5. Wall-clock cost가 크다.
   - Single step이 수천 초 수준이므로 많은 update를 수행하는 RL run에서는 total training time이 매우 커질 수 있다.
   - Memory를 해결한 뒤 throughput이 다음 병목이 된다.

6. Architecture-specific implementation cost가 높다.
   - GDN, full attention, MLA, DSA, MoE마다 state와 collective가 다르다.
   - 새로운 model을 지원하려면 단순 config 추가가 아니라 kernel과 distributed semantics 수준의 작업이 필요하다.

7. Hardware portability가 확인되지 않았다.
   - H20 topology와 memory capacity에 맞춘 receipt가 중심이다.
   - 다른 interconnect, GPU memory, CPU bandwidth에서는 staging balance가 달라질 수 있다.

# 7. My Take

## 7-1. Why this matters for my work

LongStraw의 가장 중요한 기여는 "2M token이 된다"는 숫자보다 RL step을 database transaction처럼 재설계한 점이다. Prompt capture, read-only scoring, serial replay, gradient accumulation, one-time commit을 분리하면 long-context RL의 memory 문제를 훨씬 명확하게 진단할 수 있다.

특히 agent RL이나 evidence-grounded RLVR에서 하나의 긴 environment trace를 여러 candidate response가 공유하는 경우 이 구조가 직접 연결된다. 모든 history를 graph로 유지하는 대신, reusable environment state와 trainable suffix를 분리할 수 있기 때문이다.

## 7-2. Reuse potential

### 1) Long-trace agent RL

Tool trajectory나 document history를 no-grad shared state로 capture하고, 여러 candidate decision suffix만 replay하는 구조를 적용할 수 있다.

### 2) Evidence-grounded GRPO

같은 retrieved evidence bundle에서 여러 answer와 citation trajectory를 sample할 때 prompt state를 group 단위로 재사용할 수 있다.

### 3) Systems receipt template

Long-context training 결과를 보고할 때 다음을 함께 남기는 형식이 유용하다.

- 전체 token 수
- Prompt와 response의 길이 분할
- Gradient semantics
- State residency와 rank별 ownership
- Rank별 peak memory
- Transaction별 wall-clock
- Collective topology
- State staleness policy

### 4) Conditional-gradient ablation

Prompt-side VJP를 유지한 exact mode와 response-only mode를 짧은 context에서 먼저 비교하면, memory-saving approximation의 quality cost를 측정할 수 있다.

### 5) State adapter abstraction

Model architecture별 prompt state capture와 replay를 plugin interface로 만들면, long-context RL framework에 재사용하기 좋다.

## 7-3. Follow-up papers

- Group Relative Policy Optimization: group-relative advantage objective의 출발점
- DeepSeekMath: GRPO를 reasoning model 학습에 적용한 기반 연구
- DAPO: large-scale RL의 optimization recipe를 다룬 후속 연구
- Ring Attention with Blockwise Transformers for Near-Infinite Context: long-context distributed attention의 대표 구조
- DeepSpeed-Ulysses: sequence parallelism 기반 long-context training system
- HybridFlow: RLHF dataflow와 distributed execution을 결합한 system
- SGLang: rollout serving과 structured generation을 위한 runtime

# 8. Summary

- LongStraw는 long-context RL의 병목을 attention cost가 아니라 state와 gradient lifetime overlap으로 본다.
- Prompt는 graph 없이 capture하고, old/reference scoring 뒤 response를 하나씩 replay해 live graph를 suffix로 제한한다.
- Qwen3.6-27B와 GLM-5.2에 서로 다른 state adapter와 distributed topology를 적용한다.
- 2M token 이상에서 memory-bounded transaction을 보여주지만, response-only gradient와 prefix staleness 같은 semantics caveat가 있다.
- 다음 과제는 execution feasibility를 실제 long-horizon learning gain과 end-to-end throughput으로 연결하는 것이다.
