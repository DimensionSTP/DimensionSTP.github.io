---
layout: single
title: "The Latent Space: Foundation, Evolution, Mechanism, Ability, and Outlook Review"
categories: Study-concept
tag: [LLM, LatentSpace, Survey]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2604.02029)

[Resource link](https://github.com/YU-deep/Awesome-Latent-Space)

The Latent Space는 "latent reasoning survey" 정도로 읽으면 핵심을 놓치기 쉬운 논문이다. 이 survey의 진짜 흥미로운 지점은 latent space를 chain-of-thought 압축의 한 하위 기법으로 보지 않고, language-based model 전체가 점점 token-level explicit computation에서 machine-native latent computation으로 이동하고 있다는 관점으로 재정리한다는 데 있다.

특히 최근에는 latent reasoning, latent memory, latent collaboration, latent perception이 서로 다른 커뮤니티에서 각자 다른 문제처럼 다뤄지고 있었다. 그런데 이 논문은 그 분절된 흐름을 Foundation, Evolution, Mechanism, Ability, Outlook이라는 5개 질문으로 다시 묶는다. 그래서 이 글은 "새 알고리즘 하나"를 읽는 리뷰가 아니라, 앞으로 어떤 latent-space paper를 어떤 기준으로 읽어야 하는지에 대한 map을 읽는 리뷰에 가깝다.

> 한 줄 요약: 이 논문은 latent space를 token-level reasoning shortcut이 아니라 차세대 language-based model의 machine-native computation substrate로 해석하고, 이를 Foundation, Evolution, Mechanism, Ability, Outlook의 5개 축으로 정리한 survey다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- latent space를 reasoning 한 분야로만 보지 않고, **architecture / representation / computation / optimization** 과 **reasoning / planning / modeling / perception / memory / collaboration / embodiment** 로 동시에 분해한다.
- explicit space의 구조적 한계를 "linguistic redundancy", "discretization bottleneck", "sequential inefficiency", "semantic loss"로 선명하게 정리한다.
- 이후 paper를 읽을 때 "어디에 latent variable을 두는가", "무엇을 latent로 옮기는가", "그 결과 어떤 능력이 열리는가"라는 공통 질문으로 비교할 수 있게 해준다.

이 논문의 핵심 가치는 성능 claim이 아니라 framing이다.

latent space를 "CoT를 더 짧게 만드는 요령"으로 보는 순간 이 survey의 절반만 읽게 된다. 이 논문은 오히려 반대로, latent space를 future model systems의 내부 작업 공간으로 다시 정의한다.

# 1. Problem Setting

## 1-1. Problem definition

이 논문이 겨냥하는 핵심 문제는 latent space 관련 연구가 너무 빠르게 커지면서, 무엇을 latent space라고 불러야 하고 그것이 explicit token computation과 정확히 어떻게 다른지에 대한 공통 좌표계가 부족해졌다는 점이다.

기존 autoregressive generation을 개념적으로 쓰면 아래처럼 볼 수 있다.

$$
p(y|x) = \prod_t p(y_t | x, y_{<t})
$$

여기서 입력과 출력, 그리고 중간 계산의 외부 인터페이스는 모두 token space에 묶여 있다. 반면 latent-space method는 내부 계산에 연속 표현 $z$를 더 적극적으로 도입한다.

$$
p(y|x,z) = \prod_t p(y_t | x, y_{<t}, z)
$$

물론 실제 논문들은 이보다 훨씬 다양한 형태를 갖는다. 어떤 방법은 hidden state 자체를 latent representation으로 쓰고, 어떤 방법은 learnable latent token을 두고, 어떤 방법은 external latent prior를 backbone에 주입한다. 그래서 이 survey의 첫 번째 과제는 "latent space"를 하나의 구현으로 고정하는 것이 아니라, language-based model 안에서 latent computation이 들어가는 위치와 역할을 체계적으로 다시 정의하는 것이다.

## 1-2. Why previous approaches are insufficient

기존 방식의 한계는 크게 세 가지로 보인다.

1. **token-centric framing** 이 너무 강했다. explicit language는 사람이 보기에는 좋지만, 중간 계산을 매 step token으로 바꾸는 과정 자체가 구조적 비용을 만든다.
2. latent-space 관련 기존 리뷰는 latent reasoning이나 implicit reasoning 쪽에 더 집중되어 있었다. 그런데 최근 흐름은 이미 reasoning을 넘어 planning, memory, collaboration, embodiment까지 확장되고 있다.
3. visual generative model의 latent space와 language model의 latent space를 같은 말로 묶어버리기 쉽다. 하지만 visual latent는 reconstruction objective와 spatial topology에 더 강하게 묶여 있고, language model hidden state는 next-token prediction으로 조직된다는 차이가 있다.

핵심은 단순하다. 앞으로의 질문은 "latent reasoning이 explicit CoT보다 좋은가"가 아니다. 더 중요한 질문은 "어떤 계산은 explicit로 남기고, 어떤 계산은 latent로 옮겨야 하는가"에 가깝다.

# 2. Core Idea

## 2-1. Main contribution

이 논문의 핵심 기여는 새로운 latent architecture를 제안하는 것이 아니라, 흩어진 문헌을 읽기 위한 **공통 해석 프레임**을 제시한 데 있다.

첫째, 논문은 전체 문헌을 Foundation, Evolution, Mechanism, Ability, Outlook의 5개 질문으로 정리한다.

둘째, 기술 축을 두 개로 나눈다.

- **Mechanism axis**: latent space가 어떻게 만들어지고 쓰이는가
- **Ability axis**: latent space가 무엇을 가능하게 하는가

셋째, Mechanism axis를 다시 네 가지로 쪼갠다.

- Architecture
- Representation
- Computation
- Optimization

넷째, Ability axis를 일곱 가지로 정리한다.

- Reasoning
- Planning
- Modeling
- Perception
- Memory
- Collaboration
- Embodiment

다섯째, Outlook에서는 이 분야의 병목을 evaluability, controllability, interpretability로 압축한다. 즉 latent space의 장점만 정리하는 survey가 아니라, 왜 이 분야가 아직 바로 production-ready하다고 말하기 어려운지도 같이 정리한다.

## 2-2. Design intuition

이 논문의 설계 직관은 다음과 같다.

- explicit language는 앞으로도 instruction, reporting, verification의 인터페이스로 남을 가능성이 높다.
- 반면 model 내부의 실제 계산은 점점 더 latent space로 이동할 가능성이 높다.
- 따라서 latent space를 단순한 hidden implementation detail로 보면 안 되고, model systems의 내부 작업 공간으로 봐야 한다.
- 이때 중요한 것은 특정 task label이 아니라, latent variable이 어디에 놓이고, 어떻게 계산에 참여하며, 어떤 능력을 열어주는지다.

explicit language는 interface이고, latent space는 workspace라는 관점이다. 이 framing이 잡히면 reasoning paper, memory paper, multimodal paper, agent paper를 한 좌표계에서 읽을 수 있다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | latent space 연구를 language-based model 관점에서 하나의 공통 taxonomy로 정리 |
| Core narrative | Foundation, Evolution, Mechanism, Ability, Outlook |
| Mechanism axis | Architecture, Representation, Computation, Optimization |
| Ability axis | Reasoning, Planning, Modeling, Perception, Memory, Collaboration, Embodiment |
| Difference from prior work | latent reasoning에만 머물지 않고 multimodal, agentic, embodied system까지 포함 |
| Associated resource | continuously updated Awesome-Latent-Space repository |

## 3-2. Module breakdown

### 1) Foundation

Foundation 파트는 latent space의 개념 경계를 잡는다. 여기서 중요한 것은 두 가지다.

- explicit space와 비교하면, latent space는 linguistic redundancy, representational transformation cost, sequential decoding overhead를 우회할 가능성이 있다.
- visual generative model의 latent와 language model의 latent는 같은 "continuous representation" 이라는 말만 공유할 뿐, 학습 objective와 구조적 inductive bias가 다르다.

이 구분을 먼저 해두지 않으면, 이후 문헌을 읽을 때 latent token, hidden state, KV cache, multimodal feature, external latent prior가 모두 한데 섞여 버린다.

### 2) Evolution

Evolution 파트는 문헌을 4단계로 나눈다.

| Stage | Period | What changed |
| --- | --- | --- |
| Prototype | Previous - Mar 2025 | latent reasoning의 가능성을 보여준 초기 탐색 |
| Formation | Apr 2025 - Jul 2025 | 이론화와 systematic evaluation이 본격화 |
| Expansion | Aug 2025 - Nov 2025 | vision, embodied action, multi-agent collaboration으로 확장 |
| Outbreak | Dec 2025 - Present | architecture, optimization, theory, cross-modal integration이 빠르게 성숙 |

이 timeline이 중요한 이유는, 지금 이 분야를 읽을 때 어떤 paper가 prototype 성격이고 어떤 paper가 systems-level maturity를 노리는지 감을 잡게 해주기 때문이다.

### 3) Mechanism

Mechanism 파트는 latent space를 "어떻게 구현하는가"라는 질문으로 정리한다.

- **Architecture**: backbone 안에 latent computation을 내장하는가, 별도 component를 붙이는가, auxiliary model을 쓰는가
- **Representation**: internal hidden state인가, external latent prior인가, learnable latent token인가, hybrid인가
- **Computation**: compressed, expanded, adaptive, interleaved 중 어떤 계산 패턴을 택하는가
- **Optimization**: pre-training, post-training, inference 중 어디서 latent behavior를 유도하는가

이 네 축은 실무적으로도 유용하다. 새로운 paper를 읽을 때 task 이름보다 먼저 이 네 질문으로 분해하면 구조가 더 빨리 보인다.

### 4) Ability

Ability 파트는 latent space가 실제로 어떤 능력을 여는지 정리한다. 여기서 survey의 폭이 가장 잘 드러난다.

- **Reasoning**: implicit inference, compact trace, continuous refinement, branching path
- **Planning**: controllable exploration, search efficiency, adaptive budget, sequential decision
- **Modeling**: self inspection, robust control, scalable computation
- **Perception / Memory / Collaboration / Embodiment**: text 밖의 modality와 agent setting으로 latent computation이 확장되는 축

핵심은 latent space가 reasoning 하나의 efficiency trick이 아니라는 점이다. 이 survey는 latent space를 internal deduction, persistent memory, multimodal grounding, inter-agent communication, grounded action까지 연결한다.

### 5) Outlook

Outlook 파트는 이 분야의 미래를 낙관적으로만 쓰지 않는다. 오히려 latent space가 강력해질수록 인간이 직접 보지 못하는 내부 계산이 늘어나기 때문에 evaluability, controllability, interpretability 문제가 더 중요해진다고 본다.

이 파트가 특히 중요하다. latent space는 능력 확장만의 언어로 읽기 쉽지만, 실제 연구와 배포에서는 관찰 가능성과 감사 가능성이 같이 따라와야 한다.

# 4. Training / Data / Recipe

## 4-1. Data

이 논문은 model training paper가 아니라 survey다. 따라서 여기서 말하는 "data"는 학습 데이터가 아니라 survey가 포괄하는 문헌과 associated repository에 가깝다.

특히 GitHub의 Awesome-Latent-Space repository는 survey와 함께 업데이트되는 companion resource로 제시된다. 그래서 이 논문은 정적인 PDF 하나로 끝나는 review보다, 계속 확장되는 living map에 더 가깝다.

또 하나 중요한 점은 scope다. 이 survey는 text-only latent reasoning만 보지 않는다. language-based model을 기준으로 text, vision, action, multi-agent, embodied setting까지 함께 본다. 그래서 breadth는 넓지만, 반대로 각 하위 분야를 깊이 따라가려면 결국 원문 paper로 다시 들어가야 한다.

## 4-2. Training strategy

survey 자체의 "recipe"는 문헌을 정리하는 방식에 있다. 아래 순서가 이 논문의 실제 분석 recipe다.

1. 개념을 정의한다. Foundation
2. 역사적 변화를 시간축으로 정리한다. Evolution
3. 구현 방식을 mechanism으로 분해한다. Mechanism
4. capability를 ability로 분해한다. Ability
5. 남은 병목을 governance 관점으로 정리한다. Outlook

특히 Evolution 파트의 4-stage 구분은 단순 timeline 이상이다. prototype에서 formation으로 가면서 이론과 benchmark가 생기고, expansion에서 multimodal과 agent setting으로 퍼지고, outbreak에서 architecture와 optimization이 본격적으로 고도화된다는 흐름이 보인다. 이 구분은 이후 후속 논문을 읽을 때 maturity level을 판단하는 기준으로도 쓸 수 있다.

## 4-3. Engineering notes

실무적으로 이 survey를 읽을 때 유용한 포인트는 다음과 같다.

1. 새로운 latent-space paper를 보면 먼저 **Mechanism** 으로 분류해 보는 편이 좋다.
2. 이후 실제 적용 가치가 궁금하면 **Ability** 축으로 옮겨서 어떤 capability를 여는지 확인하면 된다.
3. latent space를 visual diffusion latent와 바로 같은 개념으로 놓으면 해석이 꼬이기 쉽다.
4. 이 survey는 breadth가 넓기 때문에, 세부 수식이나 figure를 인용할 때는 HTML보다 PDF를 다시 보는 편이 안전하다.

가장 실용적인 reading order는 Mechanism 먼저, Ability 나중이다.

시스템 설계 관점에서는 "어떻게 latent를 넣는가"가 먼저 보이고, 제품 관점에서는 "그 latent가 무엇을 열어주는가"가 나중에 보인다.

# 5. Evaluation

## 5-1. Main results

이 논문은 benchmark paper가 아니므로, 여기서의 "main result"는 수치 표가 아니라 taxonomy와 synthesis의 완성도다. 핵심 결과는 아래 네 가지다.

| What the survey delivers | Why it matters |
| --- | --- |
| 5-question narrative | 넓은 문헌을 읽을 때 길을 잃지 않게 해준다 |
| 2-axis taxonomy | 서로 다른 task의 paper를 공통 좌표계로 비교하게 해준다 |
| 4-stage evolution timeline | 이 분야가 prototype인지 maturity 단계인지 구분하게 해준다 |
| evaluability / controllability / interpretability framing | latent space의 병목을 accuracy 밖으로 끌어낸다 |

논문이 가장 강하게 던지는 메시지는 latent space가 더 이상 latent reasoning만의 이야기가 아니라는 점이다. reasoning에서 시작했지만, 지금은 planning, perception, memory, collaboration, embodiment로 이미 번지고 있다. 즉 분야 자체가 "text-only hidden reasoning" 에서 "general computational substrate" 로 이동하고 있다는 판단이 이 survey의 중심이다.

## 5-2. What really matters in the experiments

이 survey에서 정말 중요한 것은 "몇 점 올랐다"가 아니다. 오히려 아래 세 가지가 더 중요하다.

### 1) 이 논문은 benchmark가 아니라 map이다

이 survey를 읽고 "어떤 방법이 제일 좋나"만 찾으면 큰 의미가 줄어든다. 이 논문의 진짜 용도는 새로운 latent paper를 만났을 때, 그 paper가 mechanism 어디에 속하고 ability 어디를 겨냥하는지 빠르게 파악하는 데 있다.

### 2) explicit vs latent를 대체 관계로만 읽으면 안 된다

논문 전반의 흐름을 보면, explicit language가 완전히 사라진다고 보기보다 instruction, reporting, verification의 인터페이스로 남고, latent space가 내부 계산을 더 많이 담당하는 방향을 상정한다. 즉 "explicit or latent"가 아니라 "explicit interface + latent workspace" 조합으로 읽는 편이 맞다.

### 3) challenge framing이 성능 주장보다 더 중요하다

이 survey는 latent space의 장점을 operability, expressiveness, scalability, generalization으로 설명하지만, 동시에 evaluability, controllability, interpretability 문제를 같은 무게로 제시한다. 이 균형이 좋다. latent space를 미래 방향으로 보더라도, 그 내부 계산을 어떻게 관찰하고 통제할 것인지 없이는 실제 연구 축이 성숙하기 어렵다.

# 6. Limitations

1. 이 논문은 breadth가 넓은 대신, 각 하위 분야의 depth는 uneven하다. latent reasoning, latent memory, latent collaboration을 한 paper 안에서 함께 다루는 장점이 있지만, 개별 분야의 세부 실험까지는 결국 원문으로 다시 내려가야 한다.

2. taxonomy가 유용하긴 하지만 category overlap도 많다. 실제로 하나의 method가 architecture이면서 representation이고, reasoning이면서 planning인 경우가 적지 않다. 그래서 이 분류는 strict ontology라기보다 reading aid에 가깝다.

3. field 자체가 너무 빠르게 움직인다. survey는 arXiv v1 시점의 문헌을 잘 묶었지만, 이후 몇 달만 지나도 새 method가 많이 추가될 수 있다. 다행히 associated repo가 이 문제를 일부 완화한다.

4. 논문이 말하는 가장 큰 open challenge도 여전히 unresolved다. evaluability, controllability, interpretability는 아직 정리된 benchmark나 protocol이 부족하다.

5. 추가로 조심할 점도 있다. latent space를 너무 넓게 정의하면 거의 모든 hidden-state manipulation이 latent-space research로 들어와 버릴 수 있다. 앞으로는 taxonomy의 경계도 함께 다듬어질 필요가 있다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문이 중요한 이유는 latent paper를 읽는 질문 자체를 바꿔주기 때문이다. 예전에는 "이게 explicit CoT보다 효율적인가"를 먼저 봤다면, 이 survey를 읽고 나면 오히려 아래 질문이 먼저 떠오른다.

- latent variable은 어디에 놓였는가
- 그것은 internal hidden state인가, external prior인가, learnable token인가
- 계산은 compressed인가, adaptive인가, interleaved인가
- 그리고 그 결과 reasoning 말고 어떤 능력이 열리는가

이 질문들은 LLM post-training, long-context system, VLM memory, agent collaboration 같은 내 실무 관심사와도 잘 맞는다. 특히 memory와 collaboration 파트는 앞으로 service system에서 더 빨리 중요해질 축이라고 본다.

## 7-2. Reuse potential

실무적으로 재사용 가능한 포인트는 다음과 같다.

1. latent-space paper reading rubric으로 바로 쓸 수 있다.
2. explicit interface와 latent workspace를 분리해 시스템을 설계할 때 좋은 사고 틀을 준다.
3. long-context, multimodal, agent system에서 text bottleneck이 심한 부분을 어디서 latent로 치환할지 고민할 때 기준점이 된다.
4. pure reasoning보다 latent memory, latent collaboration, latent perception처럼 더 시스템적인 축을 탐색할 때 entry point 역할을 한다.

가장 재사용 가치가 높은 부분은 Outlook이 아니라 Mechanism 파트다.

실제 설계에서는 "latent가 좋다"보다 "어떤 형태의 latent를 어디에 심을 것인가"가 훨씬 더 중요한 질문이기 때문이다.

## 7-3. Follow-up papers

- Training Large Language Models to Reason in a Continuous Latent Space
- SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs
- Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach
- CLaRa
- MemGen

# 8. Summary

- The Latent Space는 latent reasoning survey라기보다, language-based model이 token-level explicit computation에서 **machine-native latent computation** 으로 이동하는 흐름을 정리한 map에 가깝다.
- 핵심 기여는 Foundation, Evolution, Mechanism, Ability, Outlook의 5-question narrative와, Mechanism / Ability의 2-axis taxonomy를 제시한 데 있다.
- 특히 Architecture, Representation, Computation, Optimization으로 분해하는 Mechanism 파트는 이후 latent-space paper를 읽을 때 가장 실용적인 reading rubric으로 바로 쓸 수 있다.
- 이 논문의 중요한 framing은 **explicit language는 interface로 남고, latent space는 internal workspace가 된다** 는 관점이다.
- 다만 latent space를 미래 방향으로 보더라도, evaluability, controllability, interpretability가 같이 풀리지 않으면 이 축은 research hype를 넘어 stable system design으로 바로 이어지기 어렵다.
