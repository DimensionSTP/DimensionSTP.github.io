---
layout: single
title: "From AGI to ASI Review"
categories: Study-concept
tag: [AGI, ASI, Superintelligence, UniversalAI, MultiAgent]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2606.12683)

From AGI to ASI를 "AGI 다음에는 더 큰 모델이 나온다"는 미래 전망으로만 읽으면 핵심을 놓치기 쉽다. 이 보고서가 실제로 하려는 일은 AGI 이후의 발전을 단일 timeline이나 단일 scaling curve로 예측하는 것이 아니다. 대신 human-level AGI와 ASI 사이를 하나의 연속적인 system transition으로 보고, 어떤 경로가 이 전환을 만들 수 있는지, 어디에서 속도가 줄어들 수 있는지, 무엇을 측정해야 불확실성을 줄일 수 있는지를 구조화한다.

이 framing은 생각보다 중요하다. AGI를 하나의 threshold로만 보면 논의는 쉽게 "언제 도달하는가"에 머문다. 하지만 human-level capability를 가진 digital system이 만들어진 뒤에는 operation speed, replication, memory, communication bandwidth, parallel deployment 같은 축이 인간 조직과 다르게 확장될 수 있다. 따라서 AGI가 하나의 종착점인지, 아니면 더 빠른 변화가 시작되는 중간 지점인지는 별도의 연구 문제다.

논문은 이 문제를 네 개의 경로로 나눈다. 첫째는 compute, model, data를 더 확장하는 scaling이다. 둘째는 continual learning, memory, world model처럼 현재 paradigm의 구조적 한계를 바꾸는 algorithmic evolution이다. 셋째는 AI가 AI 연구, data generation, hardware design에 기여하는 recursive improvement다. 넷째는 매우 많은 agent가 specialization과 coordination을 통해 collective intelligence를 만드는 multi-agent path다.

> 한 줄 요약: From AGI to ASI는 human-level AGI 이후의 발전을 scaling, paradigm evolution, recursive improvement, multi-agent collective라는 네 경로로 분해하고, 이 경로를 늦출 여섯 가지 friction과 이를 검증하기 위한 research agenda를 제시하는 landscape report다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- AGI를 단일 benchmark threshold가 아니라 post-AGI capability continuum으로 다시 본다.
- ASI가 반드시 하나의 거대한 monolithic model일 필요는 없고, 많은 model instance와 agent가 만든 collective system일 수 있음을 강조한다.
- Scaling, algorithm, self-improvement, coordination을 서로 대체하는 설명이 아니라 동시에 작동할 수 있는 경로로 정리한다.
- Data, resource, paradigm, research difficulty, abstraction, governance를 실제 friction으로 분리해 낙관론과 비관론을 같은 frame 안에 둔다.
- Post-human benchmark, AI R&D automation metric, recursive improvement scaling law, multi-agent scaling law처럼 측정 가능한 후속 연구 질문을 제안한다.

이 보고서는 새로운 model이나 benchmark를 제안하는 실험 논문은 아니다. 더 정확히는 AGI 이후를 연구하려면 어떤 variable을 기록하고, 어떤 experiment를 설계하고, 어떤 claim을 분리해야 하는지를 보여주는 research map에 가깝다.

# 1. Problem Setting

## 1-1. Problem definition

이 논문이 겨냥하는 문제는 "human-level AGI 이후 AI progress를 어떻게 연구할 것인가"다.

논문은 AGI를 대략 대부분의 cognitive task에서 median human 수준에 도달한 general system으로 놓는다. 반면 ASI는 거의 모든 인간 관심 영역에서 대규모로 협력하는 human expert group보다 더 높은 cognitive capability를 가진 system으로 설명한다. 두 정의 사이에는 매우 큰 공간이 있다.

| Term | Working definition | Main question |
| --- | --- | --- |
| AGI | 대부분의 cognitive task에서 median human 수준에 가까운 general system | Human-level generality를 달성했는가 |
| ASI | 광범위한 task에서 큰 human expert collective를 넘어서는 system | Capability, speed, scale, coordination이 인간 조직을 얼마나 넘어서는가 |
| UAI | 모든 computable environment를 포괄하는 formal upper-bound framework | Intelligence의 비인간 중심 이론적 기준점은 무엇인가 |

이 구분에서 중요한 것은 ASI를 "AGI보다 benchmark score가 조금 높은 model"로 정의하지 않는다는 점이다. ASI의 비교 대상은 개인 한 명이 아니라 조직화된 expert collective다. 따라서 평가 단위도 single response accuracy에서 scientific discovery, long-horizon planning, institutional coordination, multi-domain execution으로 확장되어야 한다.

또 하나 중요한 점은 system boundary다. ASI는 하나의 checkpoint일 수도 있지만, 수많은 model instance, memory system, tool, simulator, human collaborator가 결합된 distributed system일 수도 있다. 이 경우 parameter count나 single-agent benchmark만으로는 capability를 설명하기 어렵다.

## 1-2. Why the AGI threshold is not enough

AGI threshold 중심의 논의에는 네 가지 한계가 있다.

첫째, **threshold 이후의 slope를 설명하지 못한다.** Human-level에 도달했다는 사실만으로 그 다음 발전이 느릴지, 빠를지, 특정 domain에서만 진행될지 알 수 없다.

둘째, **digital intelligence의 operating advantage를 놓친다.** 동일한 task accuracy를 가진 system이라도 더 빠르게 실행되고, 거의 손실 없이 복제되고, 여러 instance가 experience를 공유할 수 있다면 aggregate capability는 크게 달라질 수 있다.

셋째, **single-agent benchmark가 collective capability를 과소평가할 수 있다.** 실제 organization-level intelligence는 specialization, parallel work, communication protocol, verification, division of labor에서 나온다. Model 하나의 score가 낮아도 agent system 전체의 throughput과 reliability는 높을 수 있다.

넷째, **timeline forecast와 mechanism forecast를 혼동한다.** "ASI가 몇 년에 오는가"를 맞히는 것보다, progress를 만드는 mechanism과 이를 늦추는 friction을 따로 측정하는 편이 더 검증 가능하다.

## 1-3. Digital intelligence has a different scaling surface

논문은 digital system이 biological human과 다른 확장 축을 가진다고 본다.

| Digital advantage | Why it matters |
| --- | --- |
| Fast input and output | 더 많은 experiment, interaction, tool call을 같은 시간에 수행할 수 있음 |
| Faster internal processing | 가능한 경우 wall-clock time당 reasoning step을 늘릴 수 있음 |
| Larger working memory | 긴 context와 많은 intermediate state를 유지할 수 있음 |
| Substrate independence | Hardware와 serving configuration을 바꾸며 배치할 수 있음 |
| Lossless replication | Skill을 가진 instance를 낮은 marginal cost로 복제할 수 있음 |
| High-bandwidth sharing | Experience, gradient, memory, tool trace를 여러 instance가 빠르게 공유할 수 있음 |

이 축들은 human-level competence와 별개다. 예를 들어 한 agent가 한 명의 researcher와 비슷한 quality를 내더라도, 동일한 agent를 대규모로 복제하고, 병렬로 hypothesis를 탐색하고, 결과를 중앙 memory에 합칠 수 있다면 organization-level output은 달라질 수 있다.

다만 digital advantage가 곧 무한한 capability를 뜻하지는 않는다. 논문은 physics, real-time latency, physical manipulation, observability, controllability, computational complexity, formal undecidability 같은 hard limit도 함께 강조한다. ASI를 omniscient하거나 omnipotent한 존재로 읽으면 안 된다.

# 2. Core Idea

## 2-1. Universal AI as a formal reference point

논문은 AGI와 ASI를 설명하기 전에 Universal AI, 즉 UAI를 이론적 기준점으로 사용한다. Legg-Hutter 계열의 universal intelligence measure는 agent policy가 여러 computable environment에서 얻는 expected reward를 environment complexity에 따라 가중하는 방식으로 생각할 수 있다.

$$
\Upsilon(\pi) = \sum_{\mu \in \mathcal{E}} 2^{-K(\mu)} V_{\mu}^{\pi}
$$

여기서 $\pi$는 agent policy, $\mu$는 computable environment, $K(\mu)$는 environment의 description complexity, $V_{\mu}^{\pi}$는 해당 environment에서 policy가 얻는 expected value다.

이 formalism의 역할은 현재 LLM score를 직접 계산하는 데 있지 않다. 더 중요한 목적은 intelligence를 특정 human task set에 고정하지 않고, 다양한 environment에서 goal을 달성하는 일반적인 capacity로 정의하는 것이다.

AIXI는 이 관점의 theoretical endpoint지만 computable하지 않다. 따라서 논문도 AIXI를 practical architecture로 제안하지 않는다. 대신 이 upper bound를 통해 두 가지를 얻는다.

- Intelligence에는 human benchmark를 넘어서는 non-anthropocentric direction이 있을 수 있다.
- 현재 system이 어떤 task에서 human을 넘었다고 해서 universal intelligence에 가까워졌다고 바로 말할 수는 없다.

즉 UAI는 roadmap의 목적지가 아니라, AGI와 ASI 정의가 human benchmark saturation에만 묶이지 않게 만드는 formal anchor다.

## 2-2. Four pathways from AGI to ASI

논문의 핵심 기여는 AGI 이후의 발전 경로를 네 개로 분해한 것이다.

| Pathway | Main lever | Central question | Representative risk |
| --- | --- | --- | --- |
| Scaling AGI | Compute, model size, data, test-time search, deployment scale | 현재 paradigm을 더 밀면 capability가 계속 증가하는가 | Diminishing return과 resource bottleneck |
| Algorithmic evolution | Continual learning, memory, world model, new architecture | 현재 neural paradigm의 structural limit를 바꿀 수 있는가 | Breakthrough timing을 예측하기 어려움 |
| Recursive improvement | AI-assisted algorithm, data, hardware, research automation | AI가 다음 세대 AI의 improvement rate를 얼마나 높이는가 | Feedback loop가 plateau하거나 불안정해질 수 있음 |
| Multi-agent collective | Specialization, coordination, parallelism, shared memory | 많은 agent가 group-level intelligence를 만들 수 있는가 | Communication cost와 coordination failure |

이 네 경로는 mutually exclusive하지 않다. 오히려 서로를 강화할 가능성이 높다.

- Scaling은 더 많은 agent, 더 긴 search, 더 큰 memory를 가능하게 한다.
- Algorithmic improvement는 같은 compute에서 더 높은 capability를 만든다.
- Recursive improvement는 scaling과 algorithmic progress의 속도를 높일 수 있다.
- Multi-agent system은 research task를 분할하고 parallel exploration을 늘려 recursive improvement를 가속할 수 있다.

이 상호작용 때문에 ASI progress를 하나의 scaling curve로 줄이기 어렵다. Model-level gain이 작아도 deployment scale이나 coordination gain이 커질 수 있고, 반대로 model score가 좋아져도 research automation이나 physical execution이 병목이면 system-level progress는 제한될 수 있다.

## 2-3. Pathway 1: Scaling AGI

첫 번째 경로는 가장 익숙하다. 더 많은 compute, 더 좋은 hardware, 더 큰 investment, 더 효율적인 algorithm, 더 많은 inference-time search를 결합해 capability를 높이는 방식이다.

논문은 effective compute를 대략 다음과 같은 곱셈 구조로 본다.

$$
G_{\mathrm{effective}} = G_{\mathrm{hardware}} \times G_{\mathrm{investment}} \times G_{\mathrm{algorithm}}
$$

핵심은 각 factor가 독립적인 작은 gain이어도 곱해지면 큰 변화가 될 수 있다는 점이다. 논문은 하나의 aggressive illustration에서 연간 약 $10x$ 수준의 effective compute 증가 가능성을 언급하지만, 이를 안정적인 forecast로 제시하지는 않는다. Hardware trend, capital deployment, algorithmic efficiency estimate가 모두 불확실하기 때문이다.

Scaling path는 다음 요소를 포함한다.

- Pretraining compute와 data scale 증가
- Test-time compute와 search depth 증가
- Tool use와 external memory 확대
- Model instance 수와 serving throughput 증가
- Synthetic data와 self-generated curriculum 확대
- Specialized model과 router를 포함한 system scale 증가

여기서 주의할 점은 "parameter scaling"과 "system scaling"을 분리해야 한다는 것이다. Post-AGI progress는 checkpoint 하나를 키우는 것보다, 많은 instance를 얼마나 효율적으로 배치하고, 얼마나 긴 task를 수행하게 하고, 결과를 어떻게 검증하는지에 더 크게 좌우될 수 있다.

## 2-4. Pathway 2: Algorithmic evolution

두 번째 경로는 현재 paradigm 안의 missing capability를 보완하거나, paradigm 자체를 바꾸는 방식이다.

논문이 현재 system의 구조적 한계로 보는 축은 다음과 같다.

- Deployment 중 지속적으로 배우는 continual learning 부족
- Long-horizon experience를 안정적으로 유지하는 memory 부족
- Real-world dynamics를 예측하고 intervention effect를 계산하는 world model 부족
- Open-ended environment에서 robust exploration과 adaptation 부족
- Fixed-weight model과 external tool 사이의 느슨한 integration
- Planning, verification, execution을 하나의 persistent loop로 묶는 architecture 부족

이 경로의 포인트는 새로운 attention block 하나를 예측하는 데 있지 않다. Paradigm shift는 사전에 구체적으로 예측하기 어렵고, breakthrough가 실제로 어떤 형태일지도 불확실하다. 그래서 논문은 특정 architecture를 정답으로 제시하기보다, current system이 반복적으로 실패하는 capability boundary를 추적해야 한다고 본다.

실무적으로는 "새 paradigm이 필요한가"라는 큰 질문을 다음처럼 쪼개는 편이 낫다.

- 같은 task에서 context를 늘려도 failure가 유지되는가.
- More test-time compute가 error를 고치지 못하는가.
- External memory가 있어도 retrieval과 consolidation이 무너지는가.
- Tool use가 늘어날수록 reliability가 떨어지는가.
- Environment interaction에서 model update가 필요한가.

이런 failure가 scaling으로 계속 줄어드는지, 아니면 특정 floor에 걸리는지가 paradigm evolution의 필요성을 판단하는 더 직접적인 signal이다.

## 2-5. Pathway 3: Recursive improvement

Recursive improvement는 AI system이 AI progress를 만드는 process 자체에 기여하는 경로다. 이때 self-improvement를 model이 자신의 source code를 한 번에 다시 쓰는 장면으로만 생각하면 너무 좁다.

논문이 다루는 loop는 더 넓다.

1. AI가 algorithm idea를 제안한다.
2. Experiment code와 evaluation을 자동화한다.
3. Training data와 curriculum을 생성한다.
4. Hardware design과 compiler optimization을 지원한다.
5. Research result를 다음 model이나 agent에 distill한다.
6. 개선된 system이 다시 더 높은 quality의 R&D를 수행한다.

이 경로의 핵심 variable은 capability 자체보다 improvement rate다. AI가 researcher의 일부 task를 자동화하는 것과, 전체 research cycle의 throughput을 높이는 것은 다르다. 더 나아가 throughput 증가와 scientific novelty 증가도 구분해야 한다.

Recursive loop가 빠르게 가속될 수 있는 이유는 digital research의 많은 부분이 code, paper, simulation, benchmark처럼 machine-readable artifact 위에서 진행되기 때문이다. 반대로 다음 요인들은 loop를 늦출 수 있다.

- Physical experiment와 fabrication latency
- High-quality evaluation의 부족
- Research problem이 점점 어려워지는 현상
- Generated idea의 verification cost
- Model이 만든 artifact를 사람이 oversight하는 비용
- Same-distribution self-distillation에서 생기는 capability ceiling

따라서 "AI가 AI를 개선한다"는 binary statement보다, research process의 어느 stage가 얼마나 자동화되고, 그 자동화가 end-to-end cycle time과 discovery rate를 얼마나 바꾸는지 측정해야 한다.

## 2-6. Pathway 4: Multi-agent collective intelligence

네 번째 경로는 ASI가 single agent가 아니라 large-scale collective에서 나올 수 있다는 관점이다.

Human organization의 intelligence는 개인 IQ의 단순 합이 아니다. 역할 분리, communication protocol, memory, hierarchy, market mechanism, peer review, redundancy, conflict resolution이 group-level output을 만든다. AI system도 비슷한 구조를 더 빠른 communication과 replication 위에서 구현할 수 있다.

Multi-agent path의 가능한 advantage는 다음과 같다.

- Task decomposition과 parallel search
- Domain별 specialist routing
- Independent verification과 adversarial review
- Shared memory를 통한 experience aggregation
- Dynamic team composition
- Large hypothesis space에 대한 distributed exploration

하지만 agent 수를 늘린다고 자동으로 intelligence가 증가하지는 않는다. Coordination overhead, duplicated work, error propagation, correlated failure, communication bottleneck, incentive mismatch가 함께 커질 수 있다.

그래서 중요한 연구 대상은 agent count가 아니라 multi-agent scaling law다. 추가 agent 한 명이 group performance에 주는 marginal gain이 언제 줄어드는지, communication budget과 role diversity가 어떤 영향을 주는지, centralized planner와 decentralized market-style coordination 중 어떤 구조가 task별로 유리한지를 측정해야 한다.

# 3. Architecture / Method

## 3-1. Overview

이 논문은 neural architecture를 제안하지 않는다. 대신 post-AGI progress를 분석하기 위한 conceptual architecture를 만든다.

| Item | Description |
| --- | --- |
| Goal | Human-level AGI 이후의 progress mechanism과 bottleneck을 연구 가능한 형태로 분해 |
| Formal anchor | Universal AI와 AIXI를 통한 non-anthropocentric upper bound |
| Capability unit | Single model뿐 아니라 deployed system과 multi-agent collective 포함 |
| Main pathways | Scaling, algorithmic evolution, recursive improvement, multi-agent collective |
| Main frictions | Data, resource, paradigm, research difficulty, abstraction, deliberate slowdown |
| Main output | Forecast 하나가 아니라 open research question과 measurement agenda |
| Difference from timeline forecast | Arrival date보다 mechanism, interaction, bottleneck severity를 따로 측정 |

## 3-2. Analytical pipeline

보고서의 논리 구조는 다음 순서로 읽을 수 있다.

1. **Define the continuum**
   - AGI, ASI, UAI를 분리한다.
   - Human individual과 human organization을 서로 다른 comparison baseline으로 둔다.

2. **Identify digital asymmetries**
   - Speed, replication, memory, communication, substrate independence를 정리한다.
   - Biological limit와 digital deployment surface가 다르다는 점을 명시한다.

3. **Set a formal upper reference**
   - UAI를 통해 intelligence를 human benchmark 밖에서 정의한다.
   - 동시에 AIXI가 incomputable하다는 theory-practice gap을 인정한다.

4. **Decompose progress mechanisms**
   - Scaling, paradigm evolution, recursive improvement, multi-agent coordination을 나눈다.
   - 각 경로가 서로 결합될 수 있음을 본다.

5. **Attach countervailing frictions**
   - 각 경로를 늦출 수 있는 bottleneck을 여섯 유형으로 분류한다.
   - Friction마다 이를 상쇄할 counterforce도 함께 본다.

6. **Translate uncertainty into research questions**
   - Timeline prediction을 바로 내놓지 않는다.
   - Measurement, benchmark, scaling law, governance research로 변환한다.

이 구조가 좋은 이유는 낙관론과 비관론을 같은 object에 연결하기 때문이다. 예를 들어 data wall을 말할 때 synthetic data와 environment interaction이라는 counterforce를 같이 보고, resource bottleneck을 말할 때 hardware efficiency와 algorithmic gain을 같이 본다.

## 3-3. The six frictions

논문은 ASI progress를 늦출 수 있는 friction을 여섯 가지로 분류한다.

| Friction | Core concern | Possible counterforce | What should be measured |
| --- | --- | --- | --- |
| Data wall | High-quality human data가 부족해짐 | Synthetic data, simulation, self-play, active data generation | New data의 marginal capability gain |
| Economic and natural resource demand | Compute, energy, capital, chip supply가 너무 빠르게 증가 | Hardware efficiency, sparse compute, better utilization | Capability gain per dollar and per joule |
| Neural paradigm insufficiency | Current architecture가 continual learning이나 robust agency에 부족 | New memory, world model, online adaptation, hybrid system | Scaling 후에도 남는 capability floor |
| Research gets harder | 쉬운 improvement가 먼저 소진될 수 있음 | AI-assisted search, automation, larger experiment portfolio | Research effort 대비 discovery yield |
| Abstraction barrier | 더 복잡한 system을 인간이 이해하고 통제하기 어려움 | Automated interpretability, modular verification, formal interface | Oversight cost와 undetected failure rate |
| Deliberate slowdown | Safety, governance, social choice로 deployment를 제한 | International coordination, staged deployment, better assurance | Capability와 deployment 사이의 gap |

이 표에서 가장 중요한 것은 friction의 존재 자체가 아니다. 핵심 질문은 각 friction이 progress driver보다 빠르게 커지는가다.

예를 들어 data shortage가 있어도 synthetic environment에서 얻는 training signal이 더 빨리 좋아지면 data wall의 영향은 작을 수 있다. 반대로 compute efficiency가 좋아져도 energy, chip fabrication, capital concentration이 더 큰 bottleneck이 될 수 있다. 결국 필요한 것은 friction과 counterforce의 relative growth rate다.

## 3-4. ASI is a system property

논문의 여러 argument를 하나로 묶으면 ASI는 model property보다 system property에 가깝다.

System-level capability에는 다음 요소가 같이 들어간다.

- Base model competence
- Inference-time search budget
- Tool and environment access
- Long-term memory
- Number of parallel instances
- Agent specialization
- Coordination protocol
- Verification and recovery loop
- Data generation capacity
- Research and deployment infrastructure

이 관점은 benchmark 해석도 바꾼다. Single model의 exam score가 높다고 ASI인 것은 아니고, 반대로 single model이 완벽하지 않아도 collective system이 large human organization보다 높은 throughput과 reliability를 보일 수 있다.

따라서 model card만으로는 post-AGI progress를 추적하기 어렵다. System card, deployment topology, agent count, tool budget, wall-clock time, human intervention, failure recovery까지 함께 기록해야 한다.

# 4. Training / Data / Recipe

## 4-1. Evidence base

이 논문에는 model training, dataset release, loss function, benchmark leaderboard가 없다. Evidence base는 다음 요소의 synthesis로 구성된다.

- Scaling trend와 compute forecast에 관한 기존 연구
- Universal intelligence와 AIXI에 관한 theoretical work
- Current model의 memory, reasoning, tool use, continual learning limitation
- AI-assisted research와 automation에 관한 early evidence
- Multi-agent coordination과 collective intelligence literature
- Economic growth, resource constraint, governance에 관한 interdisciplinary research

따라서 이 논문을 empirical result paper처럼 읽으면 안 된다. 각 section의 claim은 evidence strength가 다르다.

- Digital replication이나 faster communication은 비교적 구조적인 argument다.
- Scaling trend는 historical data가 있지만 extrapolation uncertainty가 크다.
- Recursive improvement의 acceleration rate는 아직 direct evidence가 제한적이다.
- Large-scale multi-agent ASI는 plausible mechanism이지만 scaling law가 거의 정립되지 않았다.
- Deliberate slowdown은 technical variable이 아니라 governance와 social choice에 의존한다.

## 4-2. Forecasting strategy

논문은 하나의 arrival date를 예측하지 않는다. 대신 scenario ensemble에 가까운 접근을 취한다.

1. 가능한 progress pathway를 여러 개 둔다.
2. 각 pathway의 driver를 정의한다.
3. Driver를 늦출 friction을 연결한다.
4. Friction을 완화할 counterforce를 찾는다.
5. 관측 가능한 metric을 설계한다.
6. New evidence가 들어올 때 scenario weight를 갱신한다.

이 방식의 장점은 forecast가 틀렸을 때 어디서 틀렸는지 분석할 수 있다는 것이다. Scaling estimate가 틀린 것인지, algorithmic breakthrough가 있었던 것인지, multi-agent coordination이 예상보다 어려웠던 것인지 분리할 수 있다.

반대로 단점도 명확하다. Variable이 많고 interaction이 강해서 precise probability를 주기 어렵다. 특히 recursive improvement와 paradigm shift는 historical base rate가 약하다.

## 4-3. A practical measurement recipe

이 보고서의 research agenda를 실제 engineering workflow로 옮기면 다음과 같은 observability table을 만들 수 있다.

| Research object | Observable metric | Example experiment |
| --- | --- | --- |
| Scaling return | Capability gain per training FLOP and inference FLOP | 동일 task에서 train-time and test-time compute sweep |
| AI R&D automation | Human time saved, cycle time, accepted contribution rate | Idea to experiment to review 전체 loop 측정 |
| Recursive improvement | Generation-to-generation improvement multiplier | AI-generated algorithm을 next agent가 다시 개선하는 repeated loop |
| Multi-agent gain | Group score minus best single agent score | Agent count, role diversity, communication budget sweep |
| Research difficulty | Improvement per experiment and per dollar | 동일 domain에서 time-series로 marginal discovery yield 추적 |
| Oversight burden | Human review time and escaped error rate | Capability 증가에 따라 verification cost가 어떻게 변하는지 측정 |
| Post-human capability | Human expert saturation 이후의 progress | Setter-solver, compression, open-ended discovery task 사용 |

여기서 가장 중요한 engineering rule은 total system budget을 고정하는 것이다. Agent 수를 늘리고 token budget도 늘리고 tool access도 넓힌 뒤 single-agent와 비교하면 coordination gain을 분리할 수 없다. Compute, wall-clock time, tool call, human intervention을 함께 기록해야 한다.

## 4-4. Benchmark design beyond human saturation

Human expert benchmark가 포화되면 단순 accuracy는 progress를 잘 보여주지 못한다. 논문은 다음과 같은 평가 방향을 제안한다.

### 1) Setter-solver benchmark

한 system이 새 문제를 만들고, 다른 system이 이를 푼다. Solver가 기존 static benchmark를 memorization하는 문제를 줄이고, task generation quality와 solution capability를 함께 볼 수 있다.

다만 setter와 solver가 같은 blind spot을 공유하면 benchmark가 쉽게 닫힌 생태계가 될 수 있다. Human audit와 independent verification이 필요하다.

### 2) Multi-agent cooperative benchmark

개별 정답률보다 division of labor, communication, conflict resolution, shared memory를 평가한다. Agent count가 늘어날 때 performance가 어떻게 변하는지 측정해야 한다.

### 3) Compression benchmark

Broad knowledge와 predictive structure를 얼마나 compact하게 encode하는지 본다. Intelligence를 next-token score 하나가 아니라 world regularity를 포착하는 capacity로 읽는 방향이다.

### 4) Economic and scientific output

Real-world productivity, accepted research contribution, verified discovery, cycle-time reduction 같은 indirect metric을 사용한다. 다만 이런 metric은 organization design, capital, human quality, deployment policy에 영향을 많이 받으므로 causal attribution이 어렵다.

# 5. Evaluation

## 5-1. Main results

이 보고서에는 conventional main result가 없다. 대신 다음 네 가지 analytical output을 제공한다.

1. **ASI characterization**
   - ASI를 large human organization을 넘어서는 broad cognitive system으로 정의한다.
   - Single model과 collective system을 모두 포함한다.

2. **Four-pathway map**
   - Scaling, algorithmic evolution, recursive improvement, multi-agent collective를 분리한다.
   - 경로 사이의 compounding effect를 강조한다.

3. **Six-friction map**
   - Data, resource, paradigm, research difficulty, abstraction, deliberate slowdown을 구분한다.
   - 각 friction이 negligible한지 substantial한지는 open question으로 남긴다.

4. **Research agenda**
   - Post-human benchmark, quantitative forecast, recursive improvement dynamics, multi-agent scaling law, theory, safety, sociotechnical research를 제안한다.

따라서 이 논문의 quality는 benchmark win이 아니라 decomposition quality로 평가해야 한다.

## 5-2. What really matters in the analysis

### 1) ASI를 monolithic model과 분리한다

가장 중요한 포인트는 ASI를 "아주 큰 한 개의 model"로 고정하지 않는다는 점이다. 실제 deployment에서는 여러 model, tool, memory, human, infrastructure가 결합된다. 이 system boundary를 넓히면 scaling과 coordination이 architecture만큼 중요한 연구 대상이 된다.

### 2) Progress driver와 friction을 같은 수준에서 비교한다

Scaling만 말하면 낙관론으로 흐르고, bottleneck만 말하면 정체론으로 흐르기 쉽다. 이 논문은 driver와 friction의 relative speed를 보라고 요구한다. Data wall이 있는가보다 synthetic data quality가 data shortage를 얼마나 빠르게 상쇄하는지가 중요하다.

### 3) Recursive improvement를 process metric으로 바꾼다

Recursive improvement를 신비로운 self-rewrite로 다루지 않고 algorithm, hardware, data, division of labor의 improvement loop로 분해한다. 이 framing은 AI R&D automation을 실제 workflow metric으로 측정하게 만든다.

### 4) Multi-agent scaling law의 빈 공간을 드러낸다

Current benchmark는 single-agent capability를 중심으로 설계되어 있다. 하지만 collective path가 중요하다면 agent count, communication bandwidth, specialization, coordination topology에 따른 scaling law가 필요하다. 이 부분은 아직 empirical evidence가 가장 부족한 영역 중 하나다.

### 5) Low confidence를 숨기지 않는다

논문은 AGI 이후의 progress를 높은 confidence로 예측하지 않는다. Human-level AGI가 가능하다고 가정할 때 정확히 human level에서 발전이 멈추는 상황은 다소 특이할 수 있다고 보지만, AGI 이전 plateau, gradual transition, faster recursive dynamics를 모두 열어 둔다.

이 불확실성을 weakness로만 볼 필요는 없다. 오히려 report의 목적은 한 시나리오를 확정하는 것이 아니라, 어떤 observation이 시나리오 사이의 weight를 바꿀지 정리하는 데 있다.

## 5-3. Strong claims and weak claims should be separated

이 논문을 읽을 때 claim의 강도를 분리하는 것이 중요하다.

비교적 강한 구조적 claim은 다음과 같다.

- Digital system은 replication, speed, memory, communication에서 human organization과 다른 scaling surface를 가진다.
- AGI threshold만으로 post-AGI progress rate를 알 수 없다.
- ASI는 single model이 아니라 collective system으로 구현될 수 있다.
- Human-level benchmark가 포화되면 새로운 evaluation design이 필요하다.

더 약하고 empirical validation이 필요한 claim은 다음과 같다.

- Effective compute가 앞으로도 매우 빠르게 증가할 것이다.
- Recursive improvement가 sustained acceleration을 만들 것이다.
- Multi-agent collective가 coordination cost보다 큰 intelligence gain을 낼 것이다.
- Current neural paradigm이 ASI까지 충분하거나 충분하지 않을 것이다.
- Friction이 progress driver보다 느리게 증가할 것이다.

이 구분을 유지해야 report를 prophecy로 과대해석하지 않을 수 있다.

# 6. Limitations

1. **Quantitative probability와 timeline이 없다**
   - 네 경로와 여섯 friction을 제시하지만, 각 scenario의 probability를 계산하지 않는다.
   - 어떤 pathway가 dominant할지 비교할 quantitative model도 아직 없다.

2. **AGI와 ASI definition이 넓다**
   - "Most cognitive tasks"와 "large human organizations"는 operationalization이 어렵다.
   - Human baseline도 tool, institution, education에 따라 계속 변한다.

3. **Scaling estimate의 uncertainty가 크다**
   - Hardware, investment, algorithmic efficiency를 곱하는 decomposition은 유용하지만 factor 사이의 independence가 보장되지 않는다.
   - Historical trend를 post-AGI regime까지 extrapolate하면 distribution shift가 클 수 있다.

4. **UAI와 practical system 사이의 gap이 크다**
   - AIXI는 formal reference로는 유용하지만 incomputable하다.
   - Current LLM, agent, world model의 engineering bottleneck을 직접 해결해 주지는 않는다.

5. **Recursive improvement의 end-to-end evidence가 부족하다**
   - AI가 code나 idea를 돕는 것과 sustained research acceleration을 만드는 것은 다르다.
   - Physical experiment, verification, organizational adoption이 loop를 끊을 수 있다.

6. **Collective intelligence가 coordination을 자동으로 해결하지 않는다**
   - Agent 수가 늘수록 communication cost와 correlated error도 늘 수 있다.
   - Large-scale agent collective가 human organization보다 안정적으로 작동한다는 evidence는 아직 제한적이다.

7. **Alignment를 충분히 해결된 working assumption으로 두는 부분이 있다**
   - Capability pathway를 분석하기 위해 alignment problem을 어느 정도 분리하지만, 실제로는 oversight와 control이 progress path를 직접 바꿀 수 있다.
   - Recursive improvement와 multi-agent scale이 커질수록 alignment는 external constraint가 아니라 architecture constraint가 될 수 있다.

8. **Political economy와 distribution problem은 더 깊게 다룰 수 있다**
   - Compute ownership, labor substitution, institutional concentration, international competition은 deployment speed를 크게 바꿀 수 있다.
   - Aggregate capability가 커져도 benefit과 control이 어떻게 분배되는지는 별도 문제다.

9. **Pathway interaction을 정성적으로만 다룬다**
   - 네 경로가 서로 강화할 수 있다는 설명은 설득력이 있지만, interaction coefficient를 측정하는 방법은 아직 구체적이지 않다.
   - System-level causal model이 필요하다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문의 가장 큰 가치는 ASI를 예측했다는 데 있지 않다. 더 중요한 가치는 "AGI 이후"라는 추상적인 주제를 observable system variable로 바꾸는 데 있다.

LLM과 agent system을 실제로 만들 때 이미 비슷한 현상이 보인다. Capability는 backbone score 하나로 결정되지 않는다. Test-time compute, retrieval, memory, tool, verifier, agent topology, serving throughput, human review가 함께 결과를 만든다. From AGI to ASI는 이 system view를 훨씬 큰 scale로 확장한다.

특히 다음 두 가지는 실무적으로도 바로 연결된다.

첫째, **model improvement와 system improvement를 분리해야 한다.** 동일한 backbone이라도 better tool routing, memory, parallel execution, verification으로 output이 크게 달라질 수 있다.

둘째, **automation rate보다 closed-loop improvement rate를 봐야 한다.** AI가 research task 몇 개를 수행하는지보다, idea, experiment, evaluation, revision, deployment cycle 전체를 얼마나 줄이고 quality를 얼마나 높이는지가 중요하다.

## 7-2. Reuse potential

이 논문의 framework는 다음과 같이 재사용할 수 있다.

### 1) Capability observability dashboard

Model score뿐 아니라 다음 metric을 함께 기록한다.

- Training compute
- Test-time compute
- Tool calls
- Parallel agent count
- Human intervention
- Wall-clock completion time
- Verification cost
- Recovery rate
- Accepted output rate

이렇게 해야 architecture gain과 system orchestration gain을 분리할 수 있다.

### 2) AI R&D loop instrumentation

Research agent를 평가할 때 final paper score만 보지 않는다.

- Idea generation quality
- Experiment success rate
- Debugging time
- Reviewer rejection reason
- Evidence-to-claim consistency
- Human correction time
- Next iteration improvement

이 metric은 recursive improvement가 실제로 일어나는지 확인하는 최소 단위가 된다.

### 3) Multi-agent scaling experiment

Agent count만 늘리지 말고 다음 axis를 sweep한다.

- Agent count
- Role diversity
- Communication bandwidth
- Shared memory size
- Centralized vs. decentralized coordination
- Independent verifier 수
- Total compute budget

핵심 metric은 best single agent 대비 group gain과 marginal gain per additional agent다.

### 4) Bottleneck stress test

각 friction을 separate stress test로 만든다.

- Data를 제한했을 때 synthetic data가 얼마나 보완하는가.
- Compute를 늘렸을 때 capability floor가 남는가.
- Memory와 tool을 늘렸을 때 error가 줄지 않는가.
- Human review budget을 줄였을 때 escaped error가 얼마나 증가하는가.
- Physical feedback latency가 long-horizon planning을 얼마나 제한하는가.

### 5) Scenario update process

한 번 만든 forecast를 고정하지 않는다. Quarterly or yearly basis로 pathway evidence와 friction evidence를 갱신한다.

- Scaling return이 유지되는가.
- AI R&D automation이 실제 cycle time을 줄이는가.
- Multi-agent gain이 agent count와 함께 증가하는가.
- Oversight cost가 capability보다 빠르게 증가하는가.
- New paradigm이 기존 failure floor를 낮추는가.

이렇게 보면 ASI forecast는 한 번의 숫자가 아니라 지속적으로 update되는 evidence system이 된다.

## 7-3. Follow-up papers

- [Levels of AGI for Operationalizing Progress on the Path to AGI](https://arxiv.org/abs/2311.02462)
  - AGI를 performance와 generality level로 operationalize하는 framework를 제안한다.
  - From AGI to ASI의 broad definition을 benchmark design 관점에서 보완하기 좋다.

- [One Decade of Universal Artificial Intelligence](https://arxiv.org/abs/1202.6153)
  - UAI와 AIXI의 formal motivation을 더 깊게 이해하는 데 유용하다.
  - 이론적 upper bound와 practical approximation의 차이를 정리할 수 있다.

- [Measuring AI R&D Automation](https://arxiv.org/abs/2603.03992)
  - AI R&D automation을 capability benchmark가 아니라 spending, researcher time, oversight, incident metric으로 측정하려는 논문이다.
  - Recursive improvement pathway를 empirical monitoring으로 연결하기 좋다.

# 8. Summary

- From AGI to ASI는 AGI 이후를 하나의 timeline이 아니라 scaling, algorithmic evolution, recursive improvement, multi-agent collective라는 네 경로로 분해한다.
- ASI는 single model이 아니라 많은 instance, tool, memory, verifier가 결합된 system property일 수 있다.
- Digital system은 speed, replication, memory, communication에서 human organization과 다른 scaling surface를 가진다.
- Data, resource, paradigm, research difficulty, abstraction, deliberate slowdown이 progress를 늦출 수 있으며, 핵심은 friction과 counterforce의 relative growth rate다.
- 이 보고서의 실용적인 가치는 ASI를 예언하는 것이 아니라 post-human benchmark, AI R&D metric, recursive improvement dynamics, multi-agent scaling law라는 측정 가능한 research agenda를 제공하는 데 있다.
