---
layout: single
title: "The Topological Trouble With Transformers Review"
categories: Study-concept
tag: [Transformer, Recurrence, StateTracking, Architecture]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2604.17121)

The Topological Trouble With Transformers는 Transformer를 단순히 더 긴 context와 더 깊은 layer로 확장하면 state tracking 문제가 해결될 것이라는 가정을 정면으로 다시 보는 글이다. 여기서 state tracking은 과거 token을 다시 찾아오는 retrieval이 아니라, 시간에 따라 변하는 latent state를 계속 갱신하고 유지하는 능력이다.

> 한 줄 요약: 이 논문은 feedforward Transformer의 causal topology에서는 state update가 sequence가 진행될수록 더 깊은 layer로 밀려 올라가며, 이 때문에 긴 상호작용에서 동적인 belief state를 안정적으로 유지하기 어렵다고 주장한다. 해결 방향은 더 긴 explicit thought trace가 아니라, recurrent activation dynamics를 다시 architecture 안으로 가져오는 것이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같다.

- Long context와 chain-of-thought가 커질수록, 모델이 과거를 읽는 능력과 현재 state를 유지하는 능력을 구분해야 한다.
- Agent, multi-turn reasoning, tool-use, multi-agent coordination에서는 static retrieval보다 dynamic state update가 더 자주 병목이 된다.
- 논문은 새로운 benchmark score보다 architecture topology 관점의 framing을 제공한다.
- Recurrent Transformer, SSM, latent thought, looped depth, block recurrence를 한 taxonomy 안에서 비교한다.
- 효율적인 reasoning model을 만들려면 explicit token thought만 늘리는 전략의 비용을 다시 봐야 한다.

이 글은 Transformer를 비판하기 위한 글이라기보다, Transformer가 잘하는 일과 구조적으로 어색한 일을 분리하려는 글에 가깝다. 핵심은 "attention이 과거를 볼 수 있다"와 "모델이 세계 상태를 갱신하고 유지한다"가 같은 말이 아니라는 점이다.

# 1. Problem Setting

## 1-1. Problem definition

논문이 겨냥하는 문제는 state tracking이다. 상태 추적은 다음과 같은 반복 갱신 문제로 볼 수 있다.

$$
s_t = f(s_{t-1}, x_t)
$$

여기서 $s_t$는 현재까지의 belief state이고, $x_t$는 새로 들어온 input이다. 중요한 점은 $s_t$가 단순한 memory slot이 아니라, 이전 state와 현재 input을 합쳐 새로 갱신된 동적 표현이라는 점이다.

Transformer는 긴 context window와 attention을 통해 과거 token을 잘 검색한다. 하지만 검색은 state update와 다르다. 예를 들어 대화 중 특정 단어의 의미가 앞 문맥 때문에 하나로 정해졌다면, 모델은 그 결정을 다음 token 처리에서도 유지해야 한다. 단순히 이전 token을 다시 볼 수 있다는 사실만으로는, 그 결정이 얕은 layer에서 바로 사용 가능하다는 보장이 없다.

논문은 이 차이를 20 questions 게임과 polysemous word 예시로 설명한다. 모델이 이전 답변들과 일관된 범위를 유지하지 못하거나, river bank로 해석한 문맥을 나중에 financial bank처럼 바꿔버리는 현상은 static lookup failure라기보다 state tracking failure에 가깝다.

## 1-2. Why previous approaches are insufficient

기존 Transformer 확장 전략은 대체로 세 방향으로 간다.

| 방식 | 핵심 아이디어 | 이 논문 관점의 문제 |
| --- | --- | --- |
| Longer context | 더 많은 과거 token을 attention으로 볼 수 있게 만든다 | 과거를 볼 수 있어도 state를 얕은 layer에서 즉시 사용할 수 있는 것은 아니다 |
| Deeper model | 더 많은 layer로 복잡한 computation을 허용한다 | state update가 token step마다 더 깊은 곳으로 밀리면 depth budget을 소모한다 |
| Explicit thinking | intermediate thought를 token으로 출력해 다시 입력으로 넣는다 | state를 외부 token으로 재표현하므로 compute와 context budget을 많이 쓴다 |

이 논문의 문제의식은 특히 두 번째와 세 번째에 있다. Feedforward Transformer에서 이전 state와 새 input을 합쳐 다음 state를 만들려면, 새 state representation은 이전 state보다 더 깊은 layer에 있어야 한다. sequence가 길어질수록 이 update chain이 위로 올라가고, 결국 model depth라는 물리적 한계에 닿는다.

Chain-of-thought는 이 한계를 우회한다. 깊은 layer에서 만든 representation을 text token으로 출력하고, 다음 step에서 그 token을 다시 얕은 layer부터 읽게 만들기 때문이다. 하지만 논문은 이 방식이 micro-state tracking까지 모두 담당하기에는 비싸고 부자연스러운 workaround라고 본다. 사람이 매번 단어 의미나 대화 상태를 긴 문장으로 자기 자신에게 설명하지 않듯, 모델도 모든 state update를 visible thought token으로 외부화할 필요는 없어야 한다.

# 2. Core Idea

## 2-1. Main contribution

이 논문의 핵심 기여는 두 가지다.

1. Transformer의 state tracking 문제를 depth-limited topology 문제로 설명한다.
2. Recurrent and continuous-thought Transformer 계열을 recurrence axis와 input tokens per recurrence step이라는 두 축으로 정리한다.

첫 번째 기여는 특히 중요하다. 논문은 Transformer가 state를 아예 못 가진다고 말하지 않는다. 오히려 Transformer는 context retrieval, shortcut computation, compositional state representation을 통해 많은 일을 잘 처리한다. 그러나 arbitrary and indefinite state update가 필요한 경우, 완전히 병렬화 가능한 feedforward topology가 구조적 제약을 만든다고 본다.

두 번째 기여는 recurrence라는 말을 더 엄밀하게 나누는 것이다. 많은 architecture가 iterative processing을 쓰지만, 그것이 곧 state tracking에 충분한 recurrence를 뜻하지는 않는다. 논문은 recurrence step을 sequence length 방향의 병렬화를 막는 sequential dependency로 엄격하게 정의한다. 이 정의를 쓰면 depth recurrence, step recurrence, depth plus step recurrence가 서로 다른 design space로 보인다.

## 2-2. Design intuition

핵심 직관은 단순하다. 어떤 모델이 다음처럼 state를 갱신해야 한다고 하자.

$$
s_3 = f(f(f(s_0, x_1), x_2), x_3)
$$

Feedforward Transformer가 이 computation을 token dimension 전체에 병렬로 펼치면, 각 update는 더 깊은 layer로 이동한다. 따라서 $s_1$, $s_2$, $s_3$가 같은 depth에서 반복적으로 갱신되는 것이 아니라, depth 방향으로 점점 올라가는 cascade가 된다.

반면 recurrent architecture는 같은 function을 step마다 다시 쓰면서 state를 유지할 수 있다. 물론 recurrence를 넣으면 training과 inference의 병렬성이 약해진다. 이 논문의 실질적 메시지는 그래서 "Transformer를 RNN으로 되돌리자"가 아니다. 더 정확히는 다음 trade-off를 정직하게 보자는 것이다.

- Parallel feedforward computation은 static context retrieval과 많은 pattern composition에 강하다.
- Arbitrary state tracking은 sequential dependency를 요구할 수 있다.
- Explicit thought는 이 sequential dependency를 token으로 외부화하지만, 비용이 크다.
- Implicit recurrent activation은 더 자연스러운 state channel일 수 있지만, training efficiency가 어렵다.

# 3. Architecture / Method

## 3-1. Overview

| 항목 | 설명 |
| --- | --- |
| Goal | Transformer의 state tracking 한계를 topology 관점에서 설명 |
| Key claim | Feedforward causal Transformer에서는 state update chain이 depth budget을 소모한다 |
| Main distinction | Context lookup과 dynamic state maintenance는 다르다 |
| Proposed direction | Explicit thought trace보다 implicit recurrent activation dynamics에 주목 |
| Taxonomy 축 | Recurrence axis, input tokens per recurrence step |
| Practical tension | Sequential state update는 expressivity를 주지만 parallel training efficiency를 줄인다 |

이 논문은 새로운 model block을 바로 제안하지 않는다. 대신 어떤 architecture family가 어떤 종류의 recurrence를 제공하는지 분류하고, state tracking 관점에서 어떤 cell이 비어 있는지 보여준다.

## 3-2. Module breakdown

### 1) Feedforward Transformer의 depth cascade

Feedforward decoder는 input step을 horizontal axis로, layer를 vertical axis로 놓고 볼 수 있다. Causal attention 때문에 각 position의 block은 아래쪽과 왼쪽 아래쪽의 정보를 본다. 하지만 activation은 기본적으로 shallow layer에서 deep layer로 흐른다.

state update가 $s_t = f(s_{t-1}, x_t)$ 형태라면, $s_t$는 $s_{t-1}$와 $x_t$를 모두 반영해야 한다. 이때 feedforward 구조에서는 새 state가 이전 state보다 더 깊은 곳에 만들어진다. 그래서 sequence가 길어질수록 state representation이 위로 밀리고, 얕은 layer의 다음 token processing에서는 그 state를 바로 쓰기 어렵다.

이 지점이 논문 제목의 topological trouble이다. 문제는 단순히 parameter가 부족하다는 것이 아니라, 정보가 흐르는 방향과 state update가 요구하는 방향이 잘 맞지 않는다는 것이다.

### 2) Explicit thought as a workaround

Chain-of-thought나 latent thought는 깊은 곳에서 만든 representation을 다음 step의 input으로 되돌리는 방식이다. Natural language thought라면 실제 token으로 출력하고, latent thought라면 hidden representation을 다시 input처럼 사용한다.

이 방식은 state propagation 관점에서 효과적이다. Deep representation을 다시 shallow layer에서 접근 가능하게 만들기 때문이다. 하지만 비용이 있다.

- Output token이 늘어나 context window를 소비한다.
- Micro-state까지 전부 verbalize하면 불필요한 compute가 커진다.
- Thought trace가 길어질수록 memory와 latency가 커진다.
- 내부 activation으로 충분할 수 있는 state까지 외부 sequence로 빼내게 된다.

논문은 explicit thought를 부정하지 않는다. 다만 모든 state tracking을 explicit thought로 해결하는 것은 구조적 결함을 비싼 방식으로 우회하는 것이라고 본다.

### 3) Recurrence taxonomy

논문은 recurrent Transformer 계열을 아래처럼 나눈다.

| Recurrence 축 | Ratio > 1 | Ratio = 1 | Ratio < 1 |
| --- | --- | --- | --- |
| Depth | Looped Transformer, Universal Transformer, RINS | 빈 cell 또는 탐색 필요 영역 | 빈 cell 또는 탐색 필요 영역 |
| Step | Block-recurrent Transformer | Linear attention, DeltaNet, Mamba, RWKV-7, PaTH attention | DeltaProduct |
| Depth plus step | Recurrent memory Transformer, RINs, sentence gestalt | Feedback Transformer | COCONUT, hierarchical reasoning model, CYB |

여기서 ratio는 input tokens per recurrence step이다. Ratio > 1은 여러 token을 한 recurrence step에서 처리한다는 뜻이고, ratio < 1은 token 하나를 처리하기 전에 여러 recurrence step을 돈다는 뜻이다.

이 표에서 중요한 것은 recurrence가 어느 축에 있느냐다. Depth recurrence는 같은 layer 또는 layer group을 반복해 compute depth를 늘릴 수 있다. 하지만 그것만으로는 indefinite state tracking이 보장되지 않는다. Step recurrence는 sequence step 사이에 state를 전달한다. Depth plus step recurrence는 두 축을 함께 쓰며, 더 풍부한 state dynamics를 만들 수 있다.

### 4) Promising directions

논문이 제시하는 유망한 방향은 대략 다섯 가지로 요약할 수 있다.

1. Enhanced SSM
   - DeltaNet, RWKV-7, PaTH attention, gated linear attention, gated DeltaNet 같은 흐름을 state tracking 관점에서 본다.

2. Feedforward Transformer에 state tracking bias 넣기
   - 특수 training objective나 structural prior로 native lookback ability를 강화할 수 있다.

3. Coarse recurrence
   - token마다 recurrence를 두는 대신 chunk, sentence, thought 단위로 recurrence를 둬 compute burden을 낮춘다.

4. Representation alignment 활용
   - residual connection 때문에 layer 간 representation이 어느 정도 alignment되어 있다는 점을 이용한다.

5. 효율적인 recurrent training
   - 처음에는 parallelizable feedforward pretraining을 하고, 이후 recurrent mechanism을 도입하는 multi-stage training이 가능하다.

# 4. Training / Data / Recipe

## 4-1. Data

이 논문은 새로운 training dataset을 제시하지 않는다. 대신 여러 기존 연구와 failure case를 바탕으로 architecture-level argument를 만든다. 그래서 이 글을 읽을 때는 benchmark table보다 figure와 taxonomy를 더 중요하게 봐야 한다.

검토할 evidence는 다음과 같다.

- 20 questions와 hidden number consistency 예시
- bank라는 polysemous word의 interpretation flip-flop 예시
- Transformer layer-depth schematic
- 기존 mechanistic analysis가 보여준 upward activation flow
- Recurrent Transformer taxonomy table

## 4-2. Training strategy

논문이 직접 제안하는 training recipe는 없다. 다만 architecture training 관점에서 중요한 원칙은 분명하다.

첫째, arbitrary state propagation을 원하면 sequence length 방향의 완전 병렬성을 일부 포기해야 할 수 있다. 논문은 recurrence step을 sequential dependency로 정의하므로, 진짜 recurrence는 training efficiency와 충돌한다.

둘째, recurrence를 처음부터 대규모 pretraining 전체에 넣는 것이 유일한 길은 아니다. 논문은 parallel feedforward pretraining 이후 recurrent stage를 도입하는 multi-stage strategy를 가능성으로 언급한다. 이는 실무적으로 중요하다. 처음부터 모든 token step을 recurrent하게 학습하면 비용이 너무 커질 수 있기 때문이다.

셋째, recurrence granularity를 잘 잡아야 한다. Token-level recurrence는 expressive하지만 비싸다. Chunk-level, sentence-level, thought-level recurrence는 비용과 state tracking 사이에서 더 현실적인 operating point가 될 수 있다.

## 4-3. Engineering notes

실제 연구나 시스템 설계에서 이 논문을 가져가려면 다음을 확인해야 한다.

1. State variable이 무엇인지 먼저 정의한다.
   - 대화 상태, tool state, hidden target, plan state, entity location, constraint set처럼 갱신되어야 하는 변수가 있는지 봐야 한다.

2. Retrieval 문제와 update 문제를 분리한다.
   - 관련 문장을 찾아오는 문제라면 long context나 RAG가 맞을 수 있다.
   - 매 turn마다 state를 갱신해야 한다면 별도 memory 또는 recurrence가 필요할 수 있다.

3. Explicit thought를 무조건 늘리지 않는다.
   - Reasoning token이 늘어도 state consistency가 자동으로 해결되는 것은 아니다.
   - 어떤 state는 visible text보다 compact internal state로 유지하는 편이 낫다.

4. Coarse recurrence를 먼저 실험한다.
   - Token-level recurrence보다 episode, block, sentence, tool-call boundary 단위 recurrence가 product setting에서는 더 다루기 쉽다.

5. Evaluation은 state consistency 중심이어야 한다.
   - 단일 정답 accuracy보다, 여러 step 뒤에도 이전 state를 유지하는지 보는 protocol이 필요하다.

# 5. Evaluation

## 5-1. Main results

이 논문은 실험 논문이라기보다 position paper와 taxonomy paper에 가깝다. 따라서 main result는 특정 benchmark score가 아니라 다음 argument chain이다.

1. State tracking은 static token retrieval과 다르다.
2. Feedforward causal Transformer에서는 state update chain이 depth 방향으로 이동한다.
3. 이 구조는 long interaction에서 depth budget과 shallow-layer accessibility 문제를 만든다.
4. Explicit thought는 이를 우회하지만 compute와 context 비용을 키운다.
5. Recurrence, SSM, coarse recurrence, latent thought는 이 문제를 다루는 design space다.

특히 Figure 1은 이 논문을 이해하는 핵심이다. State가 새 input과 결합될 때마다 더 깊은 layer로 이동한다는 schematic을 보여주기 때문이다. Table 1은 그 다음 단계다. 어떤 recurrent architecture가 depth axis와 step axis 중 어디에 recurrence를 두는지 정리한다.

## 5-2. What really matters in the experiments

이 논문을 benchmark number가 아닌 evaluation 관점으로 읽으면, 앞으로 필요한 metric은 다음과 같다.

| 평가 대상 | 중요한 이유 |
| --- | --- |
| State consistency | 긴 interaction에서 belief state가 뒤집히지 않는지 본다 |
| Delayed usability | deep layer에서 만든 state가 이후 shallow computation에 쓰일 수 있는지 본다 |
| Multihop dependency | 여러 step의 state update를 압축 shortcut 없이 수행하는지 본다 |
| Cost per state update | explicit thought token 대비 internal recurrence가 효율적인지 본다 |
| Generalization length | train length보다 긴 sequence에서 state tracking이 유지되는지 본다 |

이 관점에서는 단순히 long-context benchmark 점수가 높다고 state tracking을 잘한다고 말하기 어렵다. Long context benchmark가 과거 fact retrieval 위주라면, 이 논문이 말하는 topology problem을 제대로 stress test하지 못할 수 있다.

# 6. Limitations

1. 직접 실험보다 개념적 argument에 가깝다.
   - 논문은 taxonomy와 failure interpretation을 제공하지만, 새 architecture를 구현해 대규모 실험으로 입증하지는 않는다.

2. Transformer가 실제로 학습하는 shortcut을 완전히 배제하기 어렵다.
   - 논문도 parity처럼 feedforward shortcut이 가능한 문제를 인정한다.
   - 따라서 모든 state tracking 문제에 depth-linear update가 필요하다고 단정하면 안 된다.

3. Recurrence는 비용과 병렬성 문제를 만든다.
   - 완전한 state propagation을 위해 sequential dependency를 넣으면 pretraining throughput이 떨어질 수 있다.
   - 실제 frontier-scale training에서는 이 비용이 가장 큰 제약이 될 가능성이 높다.

4. SSM과 hybrid model의 실전 성능은 task마다 다를 수 있다.
   - Enhanced SSM이 promising direction이라고 해서 모든 reasoning task에서 Transformer를 대체한다고 볼 수는 없다.

5. Cognitive analogy는 조심해서 읽어야 한다.
   - 뇌가 dynamical system이라는 설명은 좋은 직관을 주지만, foundation model architecture 설계로 바로 옮길 때는 engineering constraint가 더 중요하다.

6. Evaluation protocol이 아직 더 필요하다.
   - 이 논문의 주장은 설득력이 있지만, 실제로 어떤 benchmark가 topology-level state tracking failure를 가장 깨끗하게 측정하는지는 별도 연구가 필요하다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문이 흥미로운 이유는 reasoning model의 병목을 token budget이 아니라 state channel의 위치 문제로 바꿔 보기 때문이다. 최근 reasoning model은 더 길게 생각하고, 더 많은 scratchpad를 쓰고, 더 많은 tool call을 하도록 설계되는 경우가 많다. 하지만 이 방식은 state를 계속 text로 외부화한다.

이 논문의 핵심 메시지는 "생각을 길게 만들자"보다 "어떤 state는 생각으로 말하지 않아도 유지되어야 한다"에 가깝다. Agent system에서는 특히 그렇다. User goal, tool result, environment constraint, intermediate plan, failed attempt를 모두 자연어 로그로만 들고 가면 context가 커지고, state가 반복 재해석되며, 작은 consistency error가 누적된다.

따라서 이 논문은 long-context architecture, memory-augmented agent, recurrent reasoning model을 볼 때 좋은 기준점을 준다. 어떤 시스템이 state를 어디에 저장하고, 언제 갱신하고, 어떤 layer 또는 step에서 다시 접근하는지 물어보게 만든다.

## 7-2. Reuse potential

재사용해볼 만한 포인트는 다음과 같다.

1. Agent memory 설계 기준
   - Memory를 단순 retrieval store로 두지 말고, evolving state와 archival evidence를 분리한다.

2. Long-context benchmark 재해석
   - Retrieval score와 state update score를 별도로 본다.

3. Coarse recurrence wrapper
   - Tool call, paragraph, dialogue turn, episode boundary마다 compact state를 갱신하는 module을 실험한다.

4. Thought budget control
   - 모든 state를 chain-of-thought로 풀지 않고, visible reasoning이 필요한 state와 internal update로 충분한 state를 나눈다.

5. Hybrid architecture 읽기
   - Mamba, DeltaNet, RWKV, recurrent memory Transformer, latent thought model을 같은 taxonomy 안에서 비교한다.

## 7-3. Follow-up papers

- Recurrent Memory Transformer
- Universal Transformer
- Mamba
- DeltaNet and gated DeltaNet
- COCONUT latent reasoning model
- RWKV-7
- Training-free looped transformers
- LLMs cannot play Hangman: On the necessity of a private working memory for language agents

# 8. Summary

- 이 논문은 Transformer의 한계를 context length 문제가 아니라 state tracking topology 문제로 읽는다.
- Attention-based lookup과 dynamic belief state update는 다른 능력이다.
- Feedforward Transformer에서는 state update가 sequence가 진행될수록 더 깊은 layer로 밀려 depth budget을 소모할 수 있다.
- Explicit thought는 이 문제를 우회하지만 compute와 context cost를 늘린다.
- 앞으로 중요한 방향은 recurrent activation dynamics, enhanced SSM, coarse recurrence, state-aware evaluation이다.
