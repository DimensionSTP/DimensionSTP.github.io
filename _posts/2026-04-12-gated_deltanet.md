---
layout: single
title: "Gated Delta Networks: Improving Mamba2 with Delta Rule Review"
categories: Study-concept
tag: [DeltaNet, Mamba-2, LinearAttention]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://openreview.net/forum?id=r8H7xhYPwz)

Gated Delta Networks는 제목만 보면 “Mamba2를 조금 개선한 후속작”처럼 보이지만, 실제로는 **memory management를 어떻게 할 것인가**에 대한 꽤 좋은 설계 논문이다. 이 논문이 흥미로운 이유는 단순히 새 recurrence 하나를 제안한 데 있지 않다. 저자들은 **Mamba2의 gating이 잘하는 일**과 **DeltaNet의 delta rule이 잘하는 일**이 서로 다르다고 보고, 두 메커니즘을 하나의 update rule 안에서 결합한다. 그리고 거기서 끝나지 않고, 그 결합이 실제로 GPU-friendly하게 학습될 수 있도록 **chunkwise parallel algorithm**까지 같이 제시한다.

최근 efficient mixer 계열을 읽다 보면 결국 같은 질문으로 돌아오게 된다.  
“무엇을 오래 기억하게 할 것인가?”  
“무엇을 빨리 지울 것인가?”  
“그 선택을 hardware-efficient하게 구현할 수 있는가?”  

이 논문은 그 질문들에 대해 꽤 명확한 답을 준다. 특히 Mamba2, DeltaNet, Kimi Linear, Qwen3.5 같은 흐름을 같이 보고 있다면, Gated DeltaNet은 그 사이를 이어주는 좋은 기준점이다.

> 한 줄 요약: Gated DeltaNet은 Mamba2의 selective forgetting과 DeltaNet의 targeted memory update를 결합한 **gated delta rule**을 제안하고, 이를 chunkwise 병렬 학습과 hybrid architecture까지 확장해 **retrieval·memorization·long-context**를 동시에 개선하려는 ICLR 2025 논문이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Mamba2와 DeltaNet을 **무엇이 다른가** 수준이 아니라, **각각 어떤 memory failure mode를 갖는가**까지 설명해 준다.
- S-NIAH case study가 좋아서, gating과 delta rule의 역할 분담을 실험적으로 이해하기 쉽다.
- pure recurrent 설계에서 끝나지 않고, **hybrid model(H1/H2)** 과 **chunkwise training algorithm**까지 이어져서 실무적으로도 읽을 가치가 있다.

내가 보기엔 이 논문의 핵심은 “Mamba2보다 조금 더 잘 나온 recurrent block”이 아니다.  
오히려 **fixed-state recurrent model이 retrieval과 memorization 사이에서 왜 흔들리는지**, 그리고 그 균형을 어떤 update rule로 맞출 수 있는지를 보여주는 논문에 가깝다.

# 1. Problem Setting

## 1-1. Problem definition

- 이 논문이 겨냥하는 표면적 문제는 Transformer attention의 비효율이지만, 더 직접적인 문제는 **linear recurrent / linear attention 계열이 retrieval과 long-context에서 자주 약해진다**는 점이다.
- 저자들은 특히 최근 efficient sequence mixer들이 두 방향으로 발전해 왔다고 본다.
  - **gating / decay 계열**: Mamba2처럼 과거 상태를 selective하게 줄여서 memory를 비우는 방식
  - **delta rule 계열**: DeltaNet처럼 특정 key-value association을 정밀하게 수정하는 방식
- 문제는 이 둘이 잘하는 일이 다르다는 점이다.
  - gating은 **빠른 memory clearing**에는 좋지만, 지나친 decay는 **장기 기억 유지**를 해칠 수 있다.
  - delta rule은 **정밀한 associative update**에는 좋지만, robust한 forgetting이 없으면 **state saturation**과 **memory collision**이 생기기 쉽다.
- 따라서 이 논문의 목표는 “더 빠른 linear model”이 아니라, **forgetting과 targeted overwrite를 동시에 갖는 update rule**을 만들고, 그 rule을 실제 hardware-friendly training path 위에 올리는 것이다.

## 1-2. Why previous approaches are insufficient

- vanilla linear attention은 상태를 누적하는 방식이 단순해서 language modeling 품질이나 retrieval에서 softmax attention보다 약한 경우가 많았다.
- Mamba2는 adaptive decay를 도입해 **irrelevant information filtering**에는 강하지만, S-NIAH-1 같은 설정에서는 decay가 너무 강하게 작동해 **memory retention**이 빠르게 무너질 수 있다.
- DeltaNet은 delta rule을 통해 **associative recall**과 **memorization**에 강하지만, fixed-size state 안에 정보가 계속 겹쳐 쌓이면 **무엇을 지워야 하는가**를 잘 다루지 못한다.
- 그리고 pure recurrent model은 결국 retrieval, local comparison, local shift modeling에서 한계를 보이기 때문에, 최근 Griffin / Samba 같은 흐름처럼 **hybrid architecture**가 자연스럽게 등장한다.
- 즉 기존 방식들의 한계는 단순 성능 부족이 아니라, **memory update의 성격 자체가 한쪽으로 치우쳐 있었다**는 데 있다.

# 2. Core Idea

## 2-1. Main contribution

- **Gated delta rule**: Mamba2의 gate(감쇠/forgetting)와 DeltaNet의 delta update(선택적 overwrite)를 하나의 rule로 결합한다.
- **Online learning view**: 논문은 Table 1에서 여러 linear RNN을 online learning objective로 다시 정리하고, Gated DeltaNet을 **adaptive weight decay가 들어간 delta-rule SGD update**처럼 해석한다.
- **Hardware-efficient chunkwise training**: recurrence를 그대로 scan하는 대신, DeltaNet 계열의 병렬화 아이디어를 gating까지 확장해 chunkwise 병렬 학습이 가능하도록 만든다.
- **Hybrid models**: pure recurrent Gated DeltaNet만 제안하는 것이 아니라, Gated DeltaNet + SWA(H1), Mamba2 + Gated DeltaNet + SWA(H2) 구조까지 함께 제시한다.

## 2-2. Design intuition

이 논문의 설계 직관은 꽤 선명하다.

1. **Mamba2가 잘하는 것**  
   - memory를 빠르게 줄이거나 비우는 것  
   - 즉, irrelevant context를 필터링하는 것

2. **DeltaNet이 잘하는 것**  
   - 특정 key-value association을 더 정확하게 덮어쓰는 것  
   - 즉, memorization과 associative recall을 잘하는 것

3. **둘을 합치면 기대되는 것**  
   - 지워야 할 건 빨리 지우고  
   - 남겨야 할 건 더 정확하게 덮어쓰는 update

논문이 좋은 이유는 이 intuition을 그냥 말로 끝내지 않고, S-NIAH benchmark에서 세 가지 관찰로 풀어준다는 점이다.

- **Decay hurts memory retention**  
  Mamba2류 decay만 강하면 긴 길이에서 기억 유지가 빨리 나빠질 수 있다.
- **Gating facilitates filtering**  
  forgetting이 없으면 state saturation이 일어나 retrieval이 무너질 수 있다.
- **Delta rule helps memorization**  
  복잡한 value pattern을 저장하고 되살리는 능력은 delta rule 쪽이 더 낫다.

내가 보기엔 이 paper의 핵심은 “둘을 섞었다”가 아니라,  
**forgetting과 overwrite가 서로 다른 failure mode를 고친다**는 점을 분리해서 보여준 데 있다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | selective forgetting과 targeted update를 동시에 갖는 recurrent token mixer 설계 |
| Key module | gated delta rule 기반 Gated DeltaNet block |
| Core update | decay gate `α` + delta-rule step size `β`를 함께 사용하는 state update |
| Parallelization | gating을 포함한 recurrence를 chunkwise matrix form으로 바꿔 hardware-efficient training 가능하게 함 |
| Hybrid variants | H1: Gated DeltaNet + SWA, H2: Mamba2 + Gated DeltaNet + SWA |
| Difference from prior work | Mamba2의 gating만 쓰지도 않고, DeltaNet의 delta rule만 쓰지도 않음. 둘을 하나의 online-learning view로 묶고 병렬 알고리즘까지 제안 |

## 3-2. Module breakdown

### 1) Gated delta rule

- Table 1 기준으로 보면 Gated DeltaNet의 update는 **Mamba2의 adaptive decay**와 **DeltaNet의 key-aware corrective update**를 동시에 포함한다.
- 직관적으로는:
  - `α_t`가 상태를 얼마나 남길지 정하고
  - `β_t`가 현재 key-value pair를 얼마나 강하게 반영할지 정한다.
- 이 구조 덕분에 모델은
  - 필요할 때는 `α_t`를 작게 두어 과거 정보를 빠르게 줄일 수 있고
  - 필요할 때는 delta rule을 통해 특정 association만 선택적으로 수정할 수 있다.
- 논문은 이를 단순 recurrence가 아니라 **online regression을 푸는 SGD update + adaptive weight decay** 관점으로 해석한다는 점이 중요하다.

### 2) Token mixer block design

Fig. 1과 공식 구현을 같이 보면 block design은 비교적 명확하다.

- `q`, `k` 경로:
  - linear projection
  - short convolution
  - SiLU
  - L2 normalization
- `v` 경로:
  - linear projection
  - short convolution
  - SiLU
- `α`, `β`:
  - linear projection으로 생성
- 출력부:
  - normalization + output gate를 거친 뒤 output projection

이 설계는 꽤 의도적이다.

- q/k에 **L2 norm**을 넣어 학습 안정성을 높이고
- short convolution으로 local mixing을 조금 보완하고
- output gate를 둬서 recurrence 출력을 그대로 내보내지 않고 한 번 더 조절한다

즉 Gated DeltaNet은 단지 “state update 식”만 새로운 모델이 아니라, **projection / local mixing / normalization / gating을 한 묶음으로 설계한 token mixer**다.

### 3) Hardware-efficient chunkwise training

- recurrent form은 이론적으로 선형이더라도, 실제 GPU에서 빠르게 돌리려면 **scan-friendly recurrence를 matmul-heavy chunkwise form으로 바꾸는 과정**이 필요하다.
- 논문은 DeltaNet의 병렬화 아이디어를 가져와, gating이 들어간 경우에도 recurrence를 chunk 단위로 부분 전개하고 matrix form으로 다시 쓴다.
- 여기서 핵심 메시지는 수식 자체보다도, **좋은 update rule은 좋은 병렬화 전략과 같이 설계되어야 한다**는 점이다.
- 이 논문이 설득력 있는 이유는 “메모리 업데이트 아이디어”와 “현대 하드웨어에서의 실제 학습 경로”를 분리하지 않았기 때문이다.

### 4) Hybrid models: H1 / H2

- 저자들은 pure recurrent model의 한계를 이미 인정하고 들어간다.
- 그래서 다음 두 hybrid를 추가로 제안한다.
  - **Gated DeltaNet-H1**: Gated DeltaNet + sliding window attention
  - **Gated DeltaNet-H2**: Mamba2 + Gated DeltaNet + sliding window attention
- 이 hybrid는 retrieval, local comparison, local shift modeling을 recurrent state 하나에 전부 맡기지 않고, attention branch를 부분적으로 섞어 부담을 분산한다.
- Appendix 기준 hybrid ablation에서는 **Mamba2 → Gated DeltaNet → SWA** 순서가 가장 좋은 결과를 보였다.  
  이 점은 꽤 중요하다. hybrid는 “뭘 섞느냐” 못지않게 **어떤 순서로 섞느냐**도 성능에 영향을 준다는 뜻이기 때문이다.

# 4. Training / Data / Recipe

## 4-1. Data

- 공정 비교를 위한 main experiment는 **1.3B parameter 모델을 100B tokens** 위에서 동일 조건으로 학습한다.
- 데이터는 **FineWeb-Edu**에서 샘플링한 100B tokens다.
- 모든 모델은 **Llama2 tokenizer**를 사용하며, vocabulary size는 32,000이다.
- Appendix의 block ablation은 **400M parameter / 15B tokens** 설정에서 진행된다.
- 즉 이 논문은 “우리 모델이 더 좋다”를 말할 때, 최소한 **학습 데이터량과 파라미터 규모를 맞춘 controlled comparison**을 의식하고 있다.

## 4-2. Training strategy

- optimizer: **AdamW**
- peak learning rate: **4e-4**
- weight decay: **0.1**
- gradient clipping: **1.0**
- scheduler: **cosine annealing**
- warm-up: **1B tokens**
- batch size: **0.5M tokens**
- training length: **4K tokens**
- Samba 및 hybrid 계열의 SWA window size: **2K**

이 recipe에서 중요한 건 화려한 트릭이 아니라 **비교 조건의 일관성**이다.  
Mamba, Mamba2, DeltaNet, RetNet, HGRN2, Samba까지 같이 놓고 비교할 때 동일한 1.3B/100B 설정을 맞췄다는 점이 이 논문의 실험을 읽기 편하게 만든다.

## 4-3. Engineering notes

- 공식 PyTorch 구현이 공개되어 있어 block 구조를 코드로 역추적하기 쉽다.
- README 수준에서 봐도 `q_proj`, `k_proj`, `v_proj`, `a_proj`, `b_proj`, depthwise short convolution, output gate(`g_proj`), output norm, output projection이 드러난다.
- 즉 이 논문은 **paper figure와 실제 code path가 꽤 가깝게 대응되는 편**이다.
- 또 README는 추가 기능(예: varlen training / inference support)은 FLA 구현을 참고하라고 안내한다.  
  실무 관점에서 보면, 이런 부분은 “아이디어 제안”을 넘어 **실제 kernel ecosystem과 연결되는가**를 가늠할 수 있는 단서다.

# 5. Evaluation

## 5-1. Main results

### 1) S-NIAH case study가 이 논문의 핵심 증거다

개인적으로 이 논문의 strongest evidence는 broad benchmark 평균보다 **S-NIAH Table 2**다.  
왜냐하면 이 표가 gate와 delta rule의 역할을 가장 직접적으로 보여주기 때문이다.

- **S-NIAH-1 (pass-key retrieval)**  
  - 8K에서 DeltaNet은 **98.8**
  - Gated DeltaNet은 **91.8**
  - Mamba2는 **30.4**
- 해석:
  - 순수 기억 유지 자체는 DeltaNet이 강하고
  - decay가 강한 Mamba2는 long retention에서 크게 무너진다
  - Gated DeltaNet은 decay의 단점을 완화하지만, 순수 retention만 보면 DeltaNet보다는 약하다

반대로 filtering이 중요한 task에서는 패턴이 달라진다.

- **S-NIAH-2 (number in haystack)**  
  - 8K에서 DeltaNet **14.4**
  - Mamba2 **17.0**
  - Gated DeltaNet **29.6**
- 해석:
  - state saturation과 irrelevant memory filtering이 필요한 상황에서는 gating이 중요하고
  - Gated DeltaNet이 두 장점을 함께 가져간다

또 memorization 쪽에서는 delta rule의 효과가 다시 드러난다.

- **S-NIAH-3 (uuid in haystack)**  
  - 4K에서 DeltaNet **22.4**
  - Mamba2 **4.6**
  - Gated DeltaNet **27.6**
- 해석:
  - 복잡한 value pattern을 기억하고 복원하는 능력은 delta rule이 강하고
  - Gated DeltaNet은 그 장점을 gating과 함께 유지한다

즉 Table 2는 Gated DeltaNet이 “무조건 둘보다 낫다”보다,  
**어떤 failure mode를 어떤 메커니즘이 보완하는지**를 보여주는 diagnostic experiment다.

### 2) Language modeling / commonsense reasoning

- Table 3에서 저자들은 Gated DeltaNet이 recurrent 계열 중에서 전반적으로 더 좋은 language modeling / common-sense reasoning 성능을 보인다고 주장한다.
- 중요한 것은 단순 평균 수치보다도, **Mamba2와 DeltaNet을 둘 다 이긴 recurrent variant**라는 위치다.
- 그리고 hybrid variant(H1/H2)는 pure recurrent Gated DeltaNet보다 더 나은 결과를 보여준다.
- 이 메시지는 분명하다.  
  **좋은 recurrent update 하나만으로 모든 문제가 해결되지는 않으며, hybrid가 여전히 강하다.**

## 5-2. What really matters in the experiments

### 1) Real-world retrieval에서는 hybrid의 의미가 더 크다

- Table 4에서 pure recurrent Gated DeltaNet은 real-world retrieval average가 **30.6**으로 Mamba2(**29.8**)와 DeltaNet(**26.2**)보다 높다.
- 하지만 hybrid는 여기서 더 크게 뛴다.
  - H1: **39.0**
  - H2: **40.1**
- 즉 retrieval-heavy task에서는 “좋은 recurrent rule”도 중요하지만, **attention/local modeling을 적절히 섞는 것**이 더 결정적이다.

### 2) LongBench에서도 recurrent보다 hybrid가 더 강하다

- Table 5 average를 보면
  - Gated DeltaNet: **16.6**
  - Mamba2: **13.5**
  - DeltaNet: **13.6**
  - H1: **17.8**
  - H2: **18.4**
- 논문이 직접 강조하는 포인트도 recurrent setting에서의 **single-doc QA / few-shot ICL / code** 이점이다.
- 하지만 최종 메시지는 역시 같다.  
  **pure recurrent 개선은 분명 의미 있지만, 최종 승자는 hybrid 쪽**이다.

### 3) Throughput 결과는 “거의 공짜는 아니지만 꽤 괜찮다”

- Fig. 3 해석에 따르면 Gated DeltaNet은 **DeltaNet과 essentially 같은 throughput**을 달성하고, Mamba2보다는 약간 느리다.
- 저자들은 그 차이를 **더 expressive한 transition matrix** 때문이라고 설명한다.
- 반면 hybrid 계열은 2K SWA를 섞으면서 오히려 더 좋은 throughput 특성을 보이기도 한다.
- 특히 논문은 **Gated DeltaNet-H1이 짧은 sequence에서도 설득력 있는 training throughput**을 유지한다고 본다.

내가 보기엔 이 throughput 결과도 꽤 중요하다.  
좋은 recurrence를 제안했는데 실제 kernel path가 너무 무거우면 의미가 반감된다. 이 논문은 최소한 **“좋아졌는데 너무 느려졌다”는 비판은 피한다**는 점에서 설계가 균형적이다.

# 6. Limitations

1. **pure recurrent의 한계는 여전히 남아 있다**  
   Gated DeltaNet이 Mamba2와 DeltaNet을 recurrent setting에서 개선하는 것은 맞지만, retrieval과 long-context 전체 관점에서는 H1/H2 같은 hybrid가 더 강하다. 즉 이 논문의 결론은 “attention은 필요 없다”가 아니다.

2. **실제 retrieval gap의 일부는 update rule 바깥 문제일 수 있다**  
   논문도 real-world retrieval에서 improvement margin이 synthetic task보다 작다고 설명한다. instruction-unaligned small LM의 repetition error가 주요 원인이라면, update rule만으로 해결되지 않는 오차가 섞여 있다는 뜻이다.

3. **스케일 해석에는 주의가 필요하다**  
   주 실험은 1.3B / 100B, ablation은 400M / 15B 설정이다. 결과는 충분히 의미 있지만, 이를 그대로 10B~100B class LLM의 최종 거동으로 일반화하는 것은 조심해야 한다. 원문에서 더 큰 scale 실험은 제공하지 않는다.

4. **시스템적 우위도 “공짜”는 아니다**  
   Gated DeltaNet은 DeltaNet 대비 marginal overhead 수준이라고 하지만, Mamba2보다 완전히 빠른 것은 아니다. transition matrix 표현력이 좋아진 만큼 약간의 비용은 감수한다.

# 7. My Take

## 7-1. Why this matters for my work

- 이 논문은 efficient token mixer를 볼 때 좋은 해석 틀을 준다.
- 앞으로 비슷한 구조를 볼 때 “attention이냐 Mamba냐”보다
  - 얼마나 잘 잊는가
  - 얼마나 잘 덮어쓰는가
  - retrieval burden을 어디서 처리하는가
  - 그걸 어떤 병렬화 경로로 학습하는가
  로 분해해서 보게 만든다.
- 특히 long-context, in-context retrieval, agentic loop 같은 workload를 생각하면, **memory clearing과 targeted overwrite를 따로 설계해야 한다**는 메시지는 꽤 실무적이다.
- 최근 Kimi Linear나 Qwen3.5 같은 후속 구조를 볼 때도, Gated DeltaNet을 먼저 읽어 두면 “왜 finer-grained gating이나 hybrid layout이 또 나왔는가”가 훨씬 잘 보인다.

## 7-2. Reuse potential

- **아키텍처 관점**  
  pure recurrent block을 설계할 때 forgetting과 associative update를 한 update rule 안에서 분리해 설계하는 관점이 재사용 가능하다.
- **실험 설계 관점**  
  broad benchmark 평균만 보지 않고, S-NIAH처럼 memory failure mode를 직접 때리는 diagnostic benchmark를 같이 넣는 방식이 좋다.
- **시스템 관점**  
  recurrent idea는 scan complexity만으로 평가하면 안 되고, chunkwise matmul path와 kernel ecology까지 같이 봐야 한다는 점이 중요하다.
- **실무 관점**  
  pure recurrent가 아니라 hybrid를 기본 operating point로 보는 태도가 더 현실적이다. 이 논문도 결국 그 방향을 지지한다.

## 7-3. Follow-up papers

- Transformers are SSMs
- Parallelizing Linear Transformers with the Delta Rule over Sequence Length
- Kimi Linear: An Expressive, Efficient Attention Architecture
- Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling

# 8. Summary

- Gated DeltaNet은 Mamba2의 gating과 DeltaNet의 delta rule이 서로 보완적이라는 관찰에서 출발한다.
- 핵심은 **forgetting(α)** 과 **targeted overwrite(β)** 를 하나의 update rule 안에서 동시에 다루는 것이다.
- S-NIAH 실험은 이 조합이 왜 필요한지 가장 잘 보여주며, retention / filtering / memorization의 trade-off를 분해해서 해석하게 만든다.
- 하지만 최종 empirical message는 “pure recurrent의 완전한 승리”가 아니라, **좋은 recurrent rule + hybrid layout** 이 현실적이라는 쪽에 가깝다.
- 따라서 이 논문은 Mamba2의 minor improvement라기보다, 이후 hybrid efficient mixer들을 읽기 위한 **핵심 기준 논문**으로 보는 편이 맞다.
