---
layout: single
title: "Attention to Mamba: A Recipe for Cross-Architecture Distillation Review"
categories: Study-concept
tag: [LLM, Mamba, Distillation, LinearAttention, EfficientInference]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2604.14191)

Attention to Mamba는 Mamba를 새로운 sequence mixer로 소개하는 논문이면서도, 더 본질적으로는 이미 잘 학습된 Transformer를 어떻게 더 저렴한 아키텍처로 "옮길 것인가"를 다루는 distillation recipe 논문이다. 이 논문의 핵심은 Mamba 자체보다 architecture alignment에 있다. 저자들은 Transformer의 softmax attention을 바로 Mamba로 distill하면 거의 무너진다고 보고, 그 사이에 learned linear attention bridge를 끼워 넣는다.

> 한 줄 요약: Attention to Mamba는 Transformer의 softmax attention을 바로 Mamba로 옮기지 않고, 먼저 Hedgehog 기반 linear attention으로 distill한 뒤 그 표현을 Mamba initialization으로 옮기는 2-stage recipe를 제안해, 1B scale에서 teacher Pythia perplexity 13.86에 가까운 14.11까지 복구한 논문이다.

## 0-1. 왜 지금 읽을 가치가 있는가

- 이미 잘 학습된 Transformer checkpoint를 버리지 않고, 더 효율적인 sequence mixer로 옮길 수 있는지 묻는다.
- "direct distillation"이 아니라 "bridge distillation"이 왜 필요한지를 이론과 실험으로 같이 설명한다.
- hybrid attention-SSM block을 남기는 대신, attention block 없이도 teacher 성능에 가까워질 수 있음을 보여준다.
- long-context serving, memory cost, inference throughput 같은 실무 이슈와 직접 연결된다.

## 0-2. 먼저 말하고 싶은 핵심 결론

1. **Transformer -> Mamba direct distillation은 사실상 recipe가 아니다.** 저자들은 naive baseline이 `PPL > 100`으로 무너졌다고 적는다.

2. **핵심은 linear attention 자체보다 architecture alignment다.**
softmax attention과 Mamba 사이를 바로 잇지 않고, Hedgehog feature map으로 한번 정렬한 뒤 Mamba를 initialize한다.

3. **stage 2가 더 길어야 하지만, stage 1은 짧아도 반드시 필요하다.**
10B token budget에서 최적 split은 `10 / 90`이고, `0 / 100`이나 `100 / 0`은 둘 다 크게 나빠진다.

4. **Mamba component 중에서는 gate branch의 기여가 가장 크다.**
SSM과 short conv만으로는 개선 폭이 제한적이고, gate를 여는 순간 성능이 크게 올라간다.

## 0-3. 이 글의 관점

이 글은 이 논문을 "새로운 Mamba 아키텍처"라기보다, "Transformer installed base를 효율적인 student로 옮기는 변환 recipe"로 읽은 정리다. 그래서 attention linearization 자체보다, 어떤 부분을 그대로 두고 어떤 부분만 바꾸는지, 그리고 distillation budget을 어떻게 stage마다 배분하는지가 더 중요하다고 본다.

# 1. Problem Setting

## 1-1. Problem definition

- SSM 계열, 특히 Mamba는 generation 시점의 memory footprint와 throughput 측면에서 Transformer보다 유리하다.
- 반면 실제 오픈 생태계에는 이미 잘 학습된 Transformer가 많고, 그 위에 쌓인 학습 노하우도 훨씬 크다.
- 그렇다면 질문은 "Mamba를 처음부터 더 잘 학습할 수 있는가"가 아니라, "이미 있는 Transformer를 Mamba-like architecture로 옮길 수 있는가"가 된다.
- 하지만 이 문제는 생각보다 어렵다. softmax attention과 Mamba는 sequence mixing 방식이 근본적으로 다르고, naive distillation은 teacher behavior를 거의 보존하지 못한다.

## 1-2. Why previous approaches are insufficient

기존 방향은 크게 세 가지다.

### 1) Mamba를 scratch부터 다시 학습

- 가장 정직한 방법이지만, 이미 큰 비용으로 학습된 Transformer 자산을 거의 재사용하지 못한다.
- 실제로는 pretraining budget과 recipe maturity가 부족하면 Transformer와의 gap이 남기 쉽다.

### 2) Transformer -> Mamba direct distillation

- 가장 단순해 보이지만 구조 mismatch가 너무 크다.
- 이 논문에서도 naive direct distillation은 `PPL > 100`으로 실패했다고 적는다.
- 결국 teacher output을 잘 맞추는 것보다, teacher가 수행하던 sequence mixing을 student가 받아들일 수 있는 intermediate form으로 바꾸는 것이 더 중요해진다.

### 3) hybrid attention + SSM block

- prior work가 자주 택한 방향이다.
- 성능 보존에는 유리할 수 있지만, attention block을 남기면 "fully attention-free student"라는 목적과는 거리가 생긴다.
- inference cost를 줄이려는 동기와, 최종 아키텍처가 여전히 attention hybrid라는 결과가 충돌할 수 있다.

이전 접근의 한계는 단순히 distillation objective가 약하다는 것이 아니다. 더 정확하게는 **teacher와 student 사이에 parameter-level, function-level bridge가 부족했다**는 점이다.

# 2. Core Idea

## 2-1. Main contribution

이 논문의 핵심 기여는 2-stage cross-architecture distillation recipe다.

1. softmax Attention -> Hedgehog linear Attention
2. Hedgehog linear Attention -> HedgeMamba

첫 번째 단계에서는 Hedgehog를 사용해 softmax attention을 learned linear attention으로 바꾼다. 저자들은 Mercer theorem과 kernel trick을 이용해 softmax 안의 exponential kernel을 feature map의 inner product로 다시 쓰고, 이 feature map을 MLP로 학습한다.

$$
\phi(x) \approx \phi_{MLP}(x) := \sigma(Wx + b)
$$

두 번째 단계에서는 이렇게 얻은 linear attention을 Mamba initialization으로 옮긴다. 핵심 대응은 아래처럼 설명된다.

$$
B(X) -> \hat K(X) := \phi_{MLP}(K(X)),
\quad
C(X) -> \hat Q(X) := \phi_{MLP}(Q(X)),
\quad
\Lambda -> I,
\quad
X -> \hat V(X) := V(X)
$$

그리고 여기서 끝내지 않고, Mamba 쪽의 SSM, short convolution, gate branch를 열어 full fine-tuning으로 들어간다.

## 2-2. Design intuition

내가 이해한 설계 직관은 아래와 같다.

### 1) 바로 옮기지 말고 먼저 "비슷한 계산 형태"로 만든다

softmax attention과 Mamba는 둘 다 sequence mixer지만, 실제 계산 구조는 전혀 다르다. 저자들은 이 구조 차이를 brute-force distillation으로 밀지 않는다. 대신 attention을 먼저 linear attention 형태로 바꿔서, Mamba와 function-level correspondence가 생기도록 한다.

### 2) student는 random init이 아니라 principled init이 필요하다

이 논문의 가장 좋은 문장은 아마 이것이다. architecture alignment through principled initialization. 즉 Mamba가 teacher를 흉내 내게 하려면, 그냥 loss만 주는 것이 아니라 student의 초기 동작이 teacher와 비슷하도록 만드는 과정이 먼저 필요하다.

### 3) linear attention은 최종 목적지가 아니라 bridge다

Hedgehog 자체도 의미 있는 linearization이지만, 이 논문에서 Hedgehog는 중간 지점이다. stage 1의 목표는 linear attention으로 끝내는 것이 아니라, stage 2에서 Mamba가 출발할 수 있는 좋은 initialization을 만드는 것이다.

### 4) expressivity는 stage 2에서 회복한다

stage 1은 정렬, stage 2는 복구에 가깝다. stage 2에서 cross-entropy fine-tuning을 하며 SSM, conv, gate를 열어 주면, teacher와의 gap을 더 줄일 수 있다. 특히 gate branch가 가장 큰 개선을 만든다는 ablation이 이 직관을 잘 뒷받침한다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | pretrained Transformer를 Mamba-like student로 옮기면서 teacher 성능을 최대한 유지 |
| Teacher | pretrained Pythia Transformer |
| Stage 1 | softmax attention을 Hedgehog linear attention으로 distill |
| Stage 2 | Hedgehog를 초기값으로 삼아 HedgeMamba를 fine-tune |
| What stays from teacher | MLP, layer norm, input-output embeddings, overall Pythia block skeleton |
| What is swapped | softmax attention sequence mixer만 HedgeMamba mixer로 교체 |
| Key claim | direct distillation보다 2-stage bridge가 훨씬 안정적이고 성능 보존이 좋음 |

## 3-2. Module breakdown

### 1) Stage 1: softmax attention to Hedgehog

- softmax attention의 exponential kernel을 feature map inner product로 다시 쓴다.
- 단순 Taylor truncation linear attention 대신, Hedgehog처럼 MLP feature map을 학습한다.
- 학습 objective는 teacher layer output과 student layer output 사이의 cosine embedding matching이다.
- 이 단계에서는 feature map 관련 파라미터만 scratch에서 학습하고, 나머지 파라미터는 teacher에서 복사한 뒤 고정한다.

이 설정이 중요한 이유는 stage 1이 사실상 "교체 가능한 attention surrogate"를 만드는 단계이기 때문이다. MLP, layer norm, embedding, residual path는 teacher 그대로 두고, attention block만 linearized version으로 바꾼다.

### 2) Stage 2: Hedgehog to HedgeMamba

- 이제 learned Hedgehog feature map을 Mamba parameter initialization으로 옮긴다.
- 논문은 `B`, `C`, `Lambda`, `X`를 key, query, value와 identity state로 대응시킨다.
- 단, vanilla Mamba는 value transform을 바로 받지 않으므로 구현을 수정해 value projection을 추가한다.
- gate branch와 convolution은 처음에는 identity처럼 동작하도록 초기화한다.
- 그리고 normalization이 없는 linear attention output을 보정하기 위해 아래 normalization을 추가한다.

$$
Y_{\phi} -> Y_{\phi} / \bar Y_{\phi}
$$

이를 한 번의 SSM pass로 계산하려고 value에 all-one tensor를 concat하고, state matrix도 duplicate한다.

$$
V -> concat[V; 1],
\quad
\Lambda -> concat[\Lambda; \Lambda]
$$

이 부분이 꽤 중요하다. 이 논문은 단순히 Mamba에 attention weight를 복사하는 것이 아니라, **attention normalization까지 student 안에서 재현 가능하도록 Mamba implementation을 바꾼다.**

### 3) Final HedgeMamba는 무엇이 다른가

HedgeMamba는 아래 요소를 가진다.

- learned Hedgehog feature map
- SSM mixer parameter
- short convolution
- gate branch

저자들은 이를 pure softmax attention 대체 mixer로 본다. 즉 최종 모델은 hybrid model이라기보다, Pythia block 안의 softmax attention만 HedgeMamba mixer로 치환한 구조다.

### 4) 왜 gate branch가 중요했는가

Table 2 ablation이 가장 깔끔하다.

| Model | Params | PPL |
| --- | ---: | ---: |
| Hedgehog | 1,014M | 14.89 |
| +SSM | 1,020M | 14.89 |
| +Conv | 1,020M | 14.89 |
| +Gate, final HedgeMamba | 1,087M | 14.58 |

논문 텍스트도 가장 큰 improvement는 gate branch에서 나온다고 직접 말한다. downstream 쪽에서도 `BoolQ 54.80 -> 57.61`, `LogiQA 21.66 -> 24.42`처럼 gate를 붙였을 때 이득이 커진다. 이 결과는 Mamba의 선형 recurrence 자체보다 **가볍지만 expressive한 gating**이 teacher behavior 복구에 더 중요할 수 있다는 신호다.

# 4. Training / Data / Recipe

## 4-1. Data

- distillation dataset은 OpenWebText다.
- 저자들은 OpenWebText를 GPT-2 training corpus의 open-source reproduction으로 설명한다.
- same GPT-NeoX tokenizer를 사용해 teacher Pythia와 student HedgeMamba 결과를 직접 비교 가능하게 만든다.
- validation split은 0.0005 percent, 약 4M tokens다.
- 총 distillation budget은 10B tokens이며, OpenWebText 기준 약 1.1 epoch라고 적는다.

## 4-2. Training strategy

### Stage 1

- stage 1에서는 attention block만 Hedgehog linearization으로 교체한다.
- feature map 파라미터만 scratch에서 학습한다.
- 나머지 파라미터는 teacher에서 복사하고 고정한다.
- layer output에 대해 cosine embedding matching loss를 사용한다.
- 1B tokens를 사용한다.
- batch size는 48, sequence length는 1024다.
- 총 20K training steps다.

### Stage 2

- stage 2에서는 Hedgehog initialization을 HedgeMamba로 옮긴다.
- input-output embedding layer는 계속 고정한다.
- 그 외 나머지 아키텍처는 standard cross-entropy loss로 fine-tuning한다.
- 9B tokens를 추가로 사용한다.
- 총 180K training steps다.

### Optimizer and infra

- appendix 기준 optimizer는 AdamW다.
- `beta1 = 0.9`, `beta2 = 0.95`를 사용한다.
- linear warm-up 뒤 cosine decay to `0.1x` peak LR schedule을 쓴다.
- 1B model 기준 peak learning rate는 `0.01`이 좋았다고 적는다.
- 실험은 8 x NVIDIA A100 node에서 수행된다.

## 4-3. Engineering notes

### 1) selective scan hard-cap이 wall-clock을 왜곡한다

논문 appendix가 꽤 솔직한 부분이다. Mamba selective scan implementation은 model dimension 256 이상에서 serialization이 걸리고, 이 논문 실험은 2048 dimension까지 가므로 training time이 `> 8x` inflated된다고 적는다. 그래서 저자들은 wall-clock보다 distillation token budget을 더 reliable한 cost metric으로 보자고 말한다.

### 2) 이 recipe는 token budget 효율이 중요하다

teacher Pythia-1B는 훨씬 큰 pretraining budget을 썼지만, 이 논문은 distillation에 10B tokens만 사용한다. 텍스트 기준으로는 teacher budget의 약 2.7 percent 수준이다. 즉 이 논문이 강조하는 것은 절대 wall-clock speedup보다, 이미 존재하는 teacher를 비교적 작은 budget으로 student로 옮길 수 있다는 점이다.

### 3) final architecture는 "full native Mamba"와는 다르다

이 부분도 중요하다. 최종 student는 모든 것을 새로 설계한 native Mamba LLM이 아니다. Pythia block skeleton을 유지하고, sequence mixer만 HedgeMamba로 바꾼다. 따라서 이 논문은 full architecture redesign보다 **teacher-preserving conversion recipe**에 더 가깝다.

# 5. Evaluation

## 5-1. Main results

대표 결과는 Table 1이다.

| Model | PPL | Arc-C | Arc-E | SIQA | PiQA | Lambada | BoolQ | HSwag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pythia teacher | 13.86 | 27.04 | 56.98 | 39.86 | 70.72 | 42.07 | 60.82 | 47.16 |
| Hedgehog baseline | 14.89 | 26.45 | 52.74 | 38.38 | 68.01 | 30.60 | 54.80 | 40.79 |
| HedgeMamba | 14.11 | 27.13 | 53.66 | 39.76 | 68.72 | 32.31 | 55.20 | 41.87 |

이 표를 어떻게 읽어야 할까.

- **upstream perplexity는 상당히 잘 복구된다.** `13.86 -> 14.11`이면 꽤 가깝다.
- 일부 downstream metric은 teacher와 거의 비슷하거나 약간 넘는다. 예를 들어 `Arc-C 27.04 -> 27.13`.
- 하지만 모든 task가 teacher parity는 아니다. `BoolQ`, `HSwag`, `Lambada`는 여전히 teacher보다 아래다.
- 그래도 same 10B budget에서 Hedgehog baseline보다 upstream, downstream 모두 낫다.

가장 중요한 보조 결과는 direct distillation failure다. 저자들은 naive Transformer -> Mamba distillation이 `PPL > 100`으로 계속 실패했다고 적는다. 즉 이 논문의 gain은 단순한 tuning gain이 아니라, **원래 거의 안 되던 문제를 되는 recipe로 바꿨다**는 쪽에 가깝다.

## 5-2. What really matters in the experiments

### 1) stage 2가 더 중요하지만 stage 1은 절대 빼면 안 된다

Table 3가 이 논문의 핵심 결과다.

| S1 / S2 split | PPL | Arc-C | BoolQ | HSwag |
| --- | ---: | ---: | ---: | ---: |
| 100 / 0 | 25.71 | 25.85 | 61.47 | 26.14 |
| 50 / 50 | 14.58 | 26.19 | 57.61 | 41.81 |
| 10 / 90 | 14.11 | 27.13 | 55.20 | 41.87 |
| 0 / 100 | 17.08 | 26.11 | 54.01 | 40.25 |

여기서 읽을 포인트는 명확하다.

- stage 2 fine-tuning이 PPL을 내리는 데 가장 큰 비중을 갖는다.
- 하지만 stage 1을 완전히 빼면 `17.08`까지 나빠진다.
- stage 1만 하고 끝내면 `25.71`로 훨씬 더 나쁘다.
- 즉 stage 1은 teacher-student alignment를 만들고, stage 2는 그 위에서 성능을 복구한다.
- best split이 `10 / 90`이라는 점은 alignment 자체는 짧게 끝낼 수 있지만, **좋은 initialization이 있는지 없는지**가 최종 성능을 크게 좌우한다는 뜻이다.

### 2) token budget은 아직 안 찼다

Table 4도 중요하다.

| Distillation tokens | PPL | Arc-C | BoolQ | HSwag |
| --- | ---: | ---: | ---: | ---: |
| 1B | 16.56 | 26.19 | 57.49 | 40.67 |
| 2B | 15.61 | 25.94 | 56.45 | 40.29 |
| 3B | 15.15 | 25.09 | 56.57 | 41.03 |
| 10B | 14.11 | 27.13 | 55.20 | 41.87 |

저자들의 해석은 간단하다. student perplexity는 token budget이 늘수록 계속 개선되고, 10B에서도 saturation이 보이지 않는다. 즉 이 recipe는 "작은 toy budget에서만 좋아 보이는 trick"이 아니라, 더 많은 distillation budget을 주면 더 좋아질 가능성이 남아 있다.

### 3) gate branch는 생각보다 더 큰 역할을 한다

Table 2 ablation에서 SSM과 conv를 붙였을 때는 PPL이 거의 그대로인데, gate를 붙인 final HedgeMamba에서 유의미한 하락이 생긴다. 이 결과는 Mamba를 쓸 때 recurrence 그 자체만 보지 말고, gating이 만드는 expressivity도 같이 봐야 한다는 점을 보여준다.

### 4) scale이 커질수록 recipe가 더 자연스럽게 먹힌다

Appendix Table 6은 160M, 410M, 1B를 같이 본다. 텍스트 설명은 아주 간단하지만 중요하다.

- scale이 커질수록 PPL은 내려가고 downstream 성능은 올라간다.
- HedgeMamba는 각 scale에서 Hedgehog보다 일관되게 낫다.

이 결과는 "bridge distillation"이 충분히 작은 모델에서는 capability bottleneck에 걸릴 수 있지만, teacher/student scale이 커질수록 더 자연스럽게 작동할 가능성을 시사한다.

# 6. Limitations

1. **teacher family가 Pythia에 한정된다.**
Llama, Qwen, DeepSeek 계열처럼 더 최근 backbone에서 같은 recipe가 얼마나 잘 먹히는지는 아직 모른다.

2. **distillation dataset quality는 분리해서 분석하지 않는다.**
논문도 OpenWebText 하나만 사용했다고 적는다. 더 깨끗한 corpus, code-heavy corpus, long-context corpus에서 차이가 나는지는 열려 있다.

3. **teacher parity를 완전히 달성한 것은 아니다.**
PPL은 꽤 가깝지만, downstream task 중 일부는 여전히 teacher보다 눈에 띄게 낮다.

4. **최종 student는 full native Mamba redesign과는 다르다.**
Pythia block skeleton을 유지한 채 sequence mixer를 교체하는 방식이라, pure Mamba LLM을 처음부터 설계하는 문제와는 다르다.

5. **wall-clock efficiency는 구현 제약에 영향을 받는다.**
selective scan hard-cap 때문에 실험 wall-clock이 왜곡된다고 저자들이 직접 적는다. 그래서 실제 end-to-end system gain은 kernel과 implementation 상태에 따라 달라질 수 있다.

6. **평가가 mostly lm-eval common-sense suite에 집중된다.**
긴 문맥 retrieval, code agent, tool use, serving throughput까지 같이 보지는 않는다. 따라서 "실서비스에서 무조건 더 좋다"로 해석하면 과하다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문이 중요한 이유는 Mamba가 좋으냐 Transformer가 좋으냐를 다시 논하는 데 있지 않다. 더 중요한 질문은 **이미 있는 pretrained Transformer 자산을 어떻게 cheaper student로 바꿀 것인가**다.

실무에서는 새 아키텍처가 좋아 보여도, 이미 있는 teacher checkpoint와 학습 pipeline을 버리고 다시 pretrain하기 어렵다. 그래서 이 논문의 메시지는 꽤 현실적이다.

- distillation은 objective보다 alignment가 먼저다.
- teacher와 student가 너무 다르면 loss만 줘서는 안 된다.
- intermediate architecture를 하나 두고 넘어가는 것이 오히려 싸고 안정적일 수 있다.

## 7-2. Reuse potential

재사용 포인트로 보는 것은 네 가지다.

### 1) architecture bridge를 먼저 설계한다

Transformer -> Mamba뿐 아니라, attention -> linear mixer, dense -> recurrent, even multimodal adapter conversion에서도 intermediate bridge를 둘 수 있다.

### 2) short alignment + long recovery 구조

이 논문은 `10 / 90` split이 best다. 즉 stage 1은 길지 않아도 되고, stage 2가 길어야 한다. 이 패턴은 다른 cross-architecture distillation에도 꽤 일반적인 힌트가 될 수 있다.

### 3) student initialization이 곧 recipe의 절반이다

random init student에 distillation loss를 거는 것보다, 초기 동작이 teacher에 가깝도록 parameter mapping을 먼저 주는 것이 중요하다.

### 4) gate branch를 가볍게 보지 않는다

SSM이나 conv 같은 "주인공" 모듈보다, gate처럼 단순해 보이는 보조 branch가 성능 회복에 더 크게 기여할 수 있다. Efficient architecture를 설계할 때 자주 놓치기 쉬운 포인트다.

## 7-3. Follow-up papers

- Hedgehog
- Transformers to SSMs: Distilling Quadratic Knowledge to Subquadratic Models
- The Mamba in the Llama: Distilling and Accelerating Hybrid Models
- Llamba

# 8. Summary

- Attention to Mamba의 핵심은 Transformer를 바로 Mamba로 옮기는 것이 아니라, Hedgehog linear attention을 bridge로 두는 2-stage distillation이다.
- stage 1은 짧아도 필요하고, stage 2는 길게 가져가야 한다. best split은 `10 / 90`이다.
- final HedgeMamba는 Hedgehog baseline보다 upstream perplexity와 downstream 성능이 전반적으로 낫다.
- 특히 gate branch가 가장 큰 성능 회복을 만든다는 ablation이 인상적이다.
- 이 논문은 "좋은 Mamba 논문"이라기보다, pretrained Transformer installed base를 cheaper sequence mixer로 옮기는 recipe 논문으로 읽는 것이 더 정확하다.
