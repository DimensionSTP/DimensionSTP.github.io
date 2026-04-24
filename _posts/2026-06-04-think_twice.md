---
layout: single
title: "ThinkTwice Review"
categories: Study-concept
tag: [LLM, RLVR, GRPO, Self-Refinement, Reasoning]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2604.01591)

[Code link](https://github.com/CSSLab/ThinkTwice)

ThinkTwice는 "self-refinement prompt를 잘 쓰는 법" 정도로 읽으면 핵심을 놓치기 쉬운 논문이다. 이 논문이 진짜 흥미로운 이유는 RLVR이 보통 첫 답변의 정답률을 올리는 쪽에 집중하는 반면, ThinkTwice는 같은 sparse reward를 이용해 "문제를 푸는 능력"과 "자기 답안을 다시 고치는 능력"을 한 프레임 안에 넣으려 한다는 점이다.

보통 self-refinement는 inference time trick으로 소비되기 쉽다. 하지만 reasoning model 관점에서는 그보다 더 중요한 질문이 있다. 모델이 틀린 답을 낸 뒤, 그 오답을 다시 학습 신호로 바꿔 쓸 수 있는가? ThinkTwice는 바로 그 질문에 답한다. critique annotation도 없고, process label도 없고, "너는 지금 틀렸다" 같은 correctness hint도 주지 않는다. 대신 같은 binary correctness reward를 reasoning phase와 refinement phase에 모두 걸고, 같은 문제를 두 번 보게 만든다.

특히 이 논문이 좋은 이유는 결과를 단순 평균 점수로만 밀지 않는다는 점이다. 저자들은 왜 ThinkTwice가 동작하는지를 training dynamics로 풀어낸다. early stage에서는 refinement가 틀린 해를 고치는 "rectify" 역할을 하고, later stage에서는 이미 맞은 해를 더 짧고 안정적으로 다듬는 "fortify" 역할로 이동한다. 즉, refinement가 그냥 generation을 한 번 더 길게 하는 기능이 아니라, reward signal 자체를 더 유용하게 만드는 장치라는 해석이 가능해진다.

> 한 줄 요약: ThinkTwice는 GRPO 기반 RLVR 위에서 reasoning phase와 self-refinement phase를 번갈아 학습시키고, 두 phase에 **같은 binary correctness reward**를 걸어 reasoning과 refinement를 함께 강화하는 2-phase post-training 방법이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- reasoning RL을 first-pass 정답률 문제로만 보지 않고, **correctable error를 다시 학습 신호로 회수하는 문제**로 바꿔 본다.
- critique annotation, process supervision, correctness hint 없이도 self-refinement policy를 학습할 수 있다는 점을 보여준다.
- direct reasoning 성능과 refinement 성능을 따로 보고, cross-model refinement와 training dynamics까지 분석해서 설계 의도를 비교적 설득력 있게 보여준다.

ThinkTwice의 핵심 메시지는 꽤 선명하다. 좋은 reasoning post-training은 reward를 더 복잡하게 만드는 것만이 아니라, **같은 sparse reward를 더 많이 일하게 만드는 학습 구조**를 설계하는 문제일 수 있다.

# 1. Problem Setting

## 1-1. Problem definition

이 논문이 겨냥하는 문제는 RLVR로 reasoning 성능을 올린 뒤에도 남는 "정답 직전의 오답"이다.

수학 reasoning에서는 final answer verification이 가능하므로, GRPO 같은 outcome-based RL이 잘 작동한다. 하지만 실제 rollout을 보면 문제를 거의 풀었는데 계산 실수로 틀리거나, 좋은 풀이 방향을 잡고도 끝까지 밀고 가지 못하거나, 중간에 비효율적인 search path로 흘러가면서 마지막 답이 틀리는 경우가 많다. 이런 샘플은 완전히 무의미한 실패는 아니다. 오히려 refinement가 잘 붙으면 유용한 학습 신호가 될 수 있다.

문제는 기존 RLVR recipe가 이 부분을 직접 다루지 않는다는 점이다. 보통은 첫 답변을 생성하고, verifier가 정답 여부만 판정한 뒤, 그 signal로 policy를 업데이트한다. 이 구조에서는 "틀렸지만 조금만 더 보면 고칠 수 있는 답안"과 "애초에 전혀 방향이 다른 답안"이 outcome level에서는 같은 0 reward로 묶인다.

ThinkTwice는 이 간극을 줄이려 한다. 같은 문제에 대해 모델이 먼저 풀이를 내고, 그 풀이를 다시 검토해서 고치게 만든 뒤, 그 refinement 결과에도 같은 correctness reward를 건다. 즉, 문제 설정 자체를 "one-pass reasoning"에서 "solve and revise"로 확장한다.

## 1-2. Why previous approaches are insufficient

기존 접근의 한계는 크게 세 가지로 정리할 수 있다.

1. **training-free refinement는 policy를 학습하지 못한다.**
   Self-Refine, Reflexion 같은 방법은 inference time에 반성이나 재시도를 붙일 수는 있지만, 그 행동 자체가 reusable policy로 축적되지는 않는다. 논문 Figure 1(A)은 prompt-only reflection이 frontier LLM에서도 AIME24 성능을 소폭 떨어뜨릴 수 있다고 보여준다.

2. **training-based refinement는 외부 supervision 의존성이 크다.**
   process supervision, critique annotation, explicit correctness signal, stronger verifier, stronger teacher 같은 외부 채널이 들어가는 경우가 많다. 하지만 frontier 수준 문제에서는 더 강한 critique source가 항상 존재하지 않는다.

3. **one-pass RLVR는 correctable error를 따로 다루지 않는다.**
   GRPO, DAPO, Dr. GRPO 같은 strong baseline도 first-pass reasoning을 더 잘하게 만들 수는 있다. 하지만 refinement를 학습 목표에 직접 넣지 않기 때문에, "한 번 더 보면 고칠 수 있는 샘플"을 별도 policy skill로 축적하지 않는다.

결국 이 논문의 문제 설정은 self-refinement를 inference에 덧붙일까가 아니라, **self-refinement를 학습 가능한 post-training target으로 끌어올릴 수 있는가**에 가깝다.

# 2. Core Idea

## 2-1. Main contribution

ThinkTwice의 핵심 기여는 shared policy 기반 2-phase RLVR다.

- Phase 1에서는 일반적인 reasoning rollout을 생성하고, correctness reward로 policy를 업데이트한다.
- Phase 2에서는 Phase 1에서 나온 base solution을 refinement prompt에 넣고, 같은 policy가 자기 답안을 다시 고치게 만든다.
- refinement 결과에도 **같은 binary correctness reward**를 적용한다.
- 이 과정에서 correctness hint, critique annotation, process label, external verifier는 추가로 쓰지 않는다.

GRPO backbone 위에서 생각하면 구조는 단순하다. 각 rollout의 advantage는 기존처럼 group normalization으로 계산된다.

$$
A_i = \frac{r_i - mean(R)}{std(R)}
$$

ThinkTwice가 추가하는 것은 reward 자체가 아니라, **reward를 받는 입력 구조**다. refinement phase의 입력은 대략 아래처럼 구성된다.

$$
x_{refine} = [problem,\ base\_solution,\ review\_instruction]
$$

여기서 중요한 점은 `review_instruction`이 task-agnostic하다는 것이다. 즉 특정 오류 타입이나 정답 여부를 알려주지 않는다. 그저 "이전 풀이를 차근차근 검토하고, 틀렸으면 고치고, 맞았으면 더 명확하게 다듬으라"는 generic instruction만 준다.

## 2-2. Design intuition

설계 직관은 꽤 좋다. reasoning RL에서 sparse reward가 약한 이유는 정보량이 적어서이기도 하지만, "거의 맞았던 실패"를 충분히 활용하지 못해서이기도 하다. ThinkTwice는 refinement phase를 통해 그 실패를 다시 들여다보게 만든다.

특히 인상적인 부분은 random base-solution sampling이 만든 emergent curriculum이다.

- early training에서는 모델이 틀린 base solution을 많이 만들기 때문에, refinement phase가 자연스럽게 **error correction**에 집중한다.
- later training에서는 base solution의 정답률이 올라가므로, refinement phase가 **solution preservation and polishing** 쪽으로 이동한다.

저자들은 이를 "rectify-then-fortify" dynamic으로 해석한다. 이 부분이 ThinkTwice의 진짜 포인트다. hand-crafted curriculum 없이도, 같은 문제를 두 번 보게 하는 구조만으로 refinement difficulty가 모델의 capability boundary에 맞춰 이동한다.

또 하나 중요한 설계 포인트는 **reward를 더 복잡하게 만들지 않았다는 점**이다. ThinkTwice는 refinement adherence를 따로 보상하지 않는다. boxed answer format이나 Final Answer marker 같은 formatting도 별도 reward가 없다. 그럼에도 refinement가 진행되면서 output format과 concise answer가 함께 좋아진다. 즉, self-refinement를 너무 세밀한 rule-based target으로 만들지 않고도 behavior를 유도할 수 있다는 주장이다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | reasoning과 self-refinement를 한 shared policy 안에서 함께 학습 |
| Backbone | GRPO |
| Phase 1 | 문제를 처음부터 풀고 correctness reward로 업데이트 |
| Phase 2 | 자기 base solution을 다시 검토하고 수정한 뒤 같은 reward로 업데이트 |
| Extra supervision | 없음 |
| Extra verifier / teacher | 없음 |
| Difference from prior work | process label, critique data, correctness hint 없이 solve and revise를 모두 RL target으로 사용 |

## 3-2. Module breakdown

### 1) Reasoning phase

첫 번째 phase는 표준 GRPO와 크게 다르지 않다.

- 문제 배치를 샘플링한다.
- 현재 policy로 여러 reasoning rollout을 생성한다.
- Math-Verify exact match로 correctness reward를 준다.
- GRPO objective로 policy를 업데이트한다.

이 단계만 놓고 보면 baseline reasoning RL과 비슷하다. 하지만 ThinkTwice는 여기서 끝나지 않는다.

### 2) Refinement prompt construction

Phase 1에서 생성된 rollout들 중 각 문제마다 base solution 하나를 **random**하게 뽑는다. 그리고 아래처럼 multi-turn chat 형식의 refinement prompt를 만든다.

1. user: 원래 문제
2. assistant: base solution
3. user: generic review instruction

Appendix에 나온 refinement instruction은 4단계로 구성된다.

1. 이전 풀이를 step-by-step으로 다시 본다.
2. 계산, 논리, 문제 이해에 오류가 있으면 무엇이 틀렸는지 명시하고 올바른 접근을 설명한다.
3. 이미 정답이면 각 단계를 검토하고 더 명확하게 설명한다.
4. 마지막에는 refined solution과 answer를 제공한다.

핵심은 이 instruction이 **정답 여부를 알려주지 않는다는 점**이다.

### 3) Shared sparse reward

refinement phase에서도 reward는 달라지지 않는다. base solution이 틀렸는지 맞았는지에 대한 추가 label 없이, refined solution의 final answer correctness만 본다. ThinkTwice는 revise phase에 dense feedback이 반드시 필요하다는 가정을 버리고, **same sparse reward in both phases**를 밀어붙인다.

이 설계는 단순하지만 꽤 의미가 있다. prior RL-based refinement 계열은 explicit verification objective나 critique objective를 추가하는 경우가 많다. 반면 ThinkTwice는 Table 3 기준으로 RL-based refinement 계열 중에서도 가장 보수적인 축에 가깝다.

### 4) Why this can improve first-pass reasoning

재미있는 점은 ThinkTwice가 refinement만 좋아지게 만드는 것이 아니라, **direct reasoning도 같이 좋아지게 만든다는 점**이다. 논문 Table 1에서 ThinkTwice는 self-refinement를 쓰지 않은 direct prompting 평가에서도 strongest baseline들을 이긴다.

저자들의 해석은 자연스럽다. refinement phase가 "거의 맞았던 실패"에서 추가 learning signal을 회수하므로, shared policy 전체가 더 좋은 reasoning policy로 수렴한다. 즉 revise skill이 solve skill과 분리되지 않고 같이 학습된다.

# 4. Training / Data / Recipe

## 4-1. Data

학습 데이터는 MATH training split이다.

- training: MATH 7,500 problems
- evaluation total: 1,526 problems
- benchmarks:
  - MATH500: 500
  - OlympiadBench: 581
  - Minerva Math: 272
  - AIME 2022-2024: 90
  - AMC 10/12: 83

모든 문제는 final answer를 `\boxed{}` 형식으로 내도록 포맷되고, Hugging Face Math-Verify exact match로 채점한다. 이 점 때문에 ThinkTwice는 현재로서는 **verifiable outcome task**에 가장 잘 맞는 recipe로 읽는 편이 맞다.

## 4-2. Training strategy

모델은 두 family를 사용한다.

- Qwen3-4B-Instruct-2507
- OLMo3-7B-Instruct

선정 이유도 납득 가능하다. refinement phase가 multi-turn instruction following을 필요로 하기 때문에, base model이 아니라 instruct model을 쓴다.

Appendix Table 4 기준 핵심 hyperparameter는 아래와 같다.

| Category | Hyperparameter | Value |
| --- | --- | --- |
| GRPO Training | Learning rate | 1e-6 |
| GRPO Training | PPO clip ratio | 0.2 |
| GRPO Training | Max response length | 3000 |
| GRPO Training | Train batch size | 32 |
| GRPO Training | PPO mini batch size | 8 |
| GRPO Training | Group size | 8 |
| GRPO Training | Entropy coefficient | 0.0 |
| GRPO Training | KL penalty in reward | Disabled |
| Refinement Training | Refinement steps | 2 |
| Refinement Training | Refinement selection mode | random |
| Generation | Temperature | 1.0 train, 0.0 val |
| Generation | Top-p | 1.0 |
| Generation | Top-k | -1 |
| Generation | Max model length | 8192 |

여기서 두 포인트가 중요하다.

- **selection mode가 random**이라는 점. ThinkTwice는 아직 base solution quality에 따라 우선순위를 주는 hard selection을 쓰지 않는다.
- **refinement steps가 2**라는 점. 즉 main setting은 one reasoning + one refinement다. multi-step iterative refinement를 본격적으로 다루는 논문은 아니다.

## 4-3. Engineering notes

실무 관점에서 눈여겨볼 포인트는 아래와 같다.

1. **training-free baselines도 같은 refinement instruction으로 맞춘다.**  
   Self-Refine, Reflexion, one-step refinement 비교에서 prompt engineering 차이가 섞이지 않도록 같은 instruction을 사용한다.

2. **initial reasoning은 sampling, refinement는 greedy decoding이다.**  
   Appendix 설명에 따르면 branch별 initial sample을 만들고, second-stage generation은 greedy로 수행한다. 그래서 refinement gains를 "추가 sampling budget"이 아니라 refinement policy 자체의 차이로 읽기 쉽다.

3. **timing은 2 x H100 80GB 기준이다.**  
   Figure 5의 wall-clock 비교는 2 x H100 80GB에서 측정되었다. absolute training time은 환경 의존적이지만, relative overhead는 참고할 만하다.

4. **boxed answer exact match라서 metric이 명확하다.**  
   장점은 judge bias가 적다는 점이고, 단점은 open-ended task로 그대로 옮기기 어렵다는 점이다.

# 5. Evaluation

## 5-1. Main results

논문에서 가장 중요한 결과는 **direct reasoning과 self-refinement 둘 다 좋아졌다**는 점이다.

아래는 핵심 수치만 압축한 표다.

| Model | Method | AIME reasoning pass@4 | Avg reasoning | AIME refine pass@4 | Avg refine |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-4B | GRPO | 39.06 | 62.22 | 48.91 | 67.42 |
| Qwen3-4B | DAPO | 42.54 | 64.53 | 49.86 | 69.01 |
| Qwen3-4B | ThinkTwice | 44.11 | 65.57 | 60.43 | 71.88 |
| OLMo3-7B | GRPO | 39.38 | 62.45 | 46.04 | 66.08 |
| OLMo3-7B | DAPO | 36.72 | 62.12 | 44.26 | 66.69 |
| OLMo3-7B | ThinkTwice | 39.24 | 64.22 | 49.33 | 69.35 |

Qwen3-4B 기준으로 보면 메시지가 더 선명하다.

- reasoning AIME pass@4: 39.06 -> 44.11, GRPO 대비 +5.05
- refinement AIME pass@4: 48.91 -> 60.43, GRPO 대비 +11.52
- direct reasoning average: 62.22 -> 65.57
- self-refinement average: 67.42 -> 71.88

즉 ThinkTwice는 "refinement용 후처리 모델"이 아니라, shared policy 전체를 더 강하게 만든다.

또 하나 중요한 결과는 cross-model refinement다. Figure 3에 따르면 ThinkTwice를 refinement model로 사용할 때, 어떤 backbone reasoning model이 base solution을 만들었는지와 무관하게 가장 높은 average pass@4를 보인다. 이건 refinement capability가 자기 rollout에만 overfit되지 않았다는 의미다.

## 5-2. What really matters in the experiments

### 1) refinement training이 first-pass reasoning도 올린다

이 논문을 읽을 때 가장 먼저 봐야 할 지점이다. ThinkTwice는 test time refinement를 붙이기 전 direct prompting Table 1에서도 strongest average를 기록한다. 즉 refinement phase는 부가 기능이 아니라, **reasoning policy 자체를 더 잘 학습시키는 보조 신호**로 작동한다.

### 2) training-free baselines를 꽤 확실하게 이긴다

Table 2에서 ThinkTwice는 Qwen3-4B 기준 Self-Refine 66.83, Reflexion 60.98을 넘어서 71.88 average를 기록한다. OLMo3-7B에서도 같은 패턴이 나온다. 이건 inference prompt tuning보다, **self-refinement를 policy로 학습시키는 것**이 더 중요할 수 있음을 보여준다.

### 3) rectify-then-fortify dynamic이 숫자로도 보인다

Figure 4의 핵심 metric은 두 가지다.

- `fix-wrong`: 틀린 base solution이 refinement 후 맞는 답으로 바뀌는 비율
- `damage-correct`: 맞은 base solution이 refinement 후 틀리게 망가지는 비율

저자들에 따르면 ThinkTwice는 early stage에서 더 높은 `fix-wrong`을 유지하고, later stage에서는 `damage-correct`가 거의 0에 가까워진다. 동시에 correct-only base solution에 대한 refinement output 길이도 훈련 후반으로 갈수록 눈에 띄게 짧아진다. 이건 refinement가 단순히 verbose second pass가 아니라, **already-correct solution을 압축하고 안정화하는 역할**도 한다는 뜻이다.

### 4) format reward 없이 formatting도 좋아진다

Figure 4 상단은 boxed answer와 Final Answer marker 사용률을 보여준다. 논문은 format reward를 따로 주지 않았는데도 ThinkTwice가 vanilla GRPO보다 더 높은 boxed-answer rate와 final-answer marker rate를 보인다고 설명한다. 즉 revise phase가 chain을 정리하는 과정에서 출력 format도 안정화된 셈이다.

### 5) cost overhead는 생각보다 작다

Figure 5의 cost analysis도 꽤 설득력 있다.

- same step count up to step 300: 9.42h vs 9.15h, total wall-clock 기준 약 3% overhead
- best checkpoint wall-clock: 7.2h vs 8.6h, GRPO보다 16% 빠르게 best checkpoint 도달
- best checkpoint step: 220 vs 280

한 단계 더 돌리는 구조치고는 overhead가 작다. refinement phase가 early training에서는 비싸지만, refined response가 점점 짧아지면서 cost gap이 줄어드는 그림도 일관적이다.

# 6. Limitations

1. **현재 검증은 수학 reasoning에 집중되어 있다.**  
   outcome verification이 쉬운 math benchmark에서는 잘 맞지만, open-ended writing, agent planning, tool use, long-form QA에 그대로 옮기려면 reward design이 다시 필요하다.

2. **main recipe는 one refinement step에 머문다.**  
   ThinkTwice는 multi-turn format을 지원할 수 있다고 말하지만, 논문의 중심 결과는 one reasoning + one refinement다. iterative refinement를 깊게 확장한 결과는 아직 없다.

3. **random base-solution sampling은 가장 단순한 선택이다.**  
   논문은 일부러 simple strategy를 택했지만, hard-case prioritization이나 uncertainty-aware selection이 더 좋을 가능성은 충분하다.

4. **모델 스케일과 범위가 제한적이다.**  
   실험은 Qwen3-4B와 OLMo3-7B 두 family에 집중되어 있다. larger model, MoE model, code-specialized model, multimodal model에서 같은 dynamic이 얼마나 유지되는지는 미지수다.

5. **reward는 여전히 sparse하다.**  
   ThinkTwice의 장점은 sparse reward를 잘 재활용한 데 있지만, 동시에 그 한계도 있다. refinement adherence나 local correction quality를 따로 보지 않기 때문에, open-ended task에서는 신호가 약해질 수 있다.

6. **evaluation target이 꽤 clean하다.**  
   boxed exact match가 가능한 수학에서는 self-refinement skill이 비교적 깔끔하게 측정된다. 하지만 실제 서비스에서는 답이 부분적으로만 맞거나, reasoning은 좋아도 format이 틀리거나, answer span이 애매한 경우가 많다. 그런 setting에서는 additional verifier나 rubric이 필요할 수 있다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문이 내 관점에서 중요한 이유는 post-training target을 다시 보게 만들기 때문이다. reasoning RL을 할 때 우리는 보통 "정답률을 올리는 첫 답변 policy"에만 집중한다. 그런데 실제 서비스나 agent pipeline에서는 first pass가 틀렸더라도, 그 답안을 기반으로 second pass에서 복구할 수 있으면 전체 quality가 많이 좋아진다.

ThinkTwice는 그 second pass를 inference trick이 아니라 training target으로 끌어올린다. 이건 코드 생성, 문서 파싱 후 correction, symbolic reasoning, theorem-style derivation, tool-call repair처럼 초기 답변을 다시 고치는 능력이 중요한 워크플로우에 꽤 직접적으로 연결된다.

## 7-2. Reuse potential

재사용 포인트는 아래 4가지다.

1. **verifiable code generation으로 확장하기 좋다.**  
   unit test나 compiler signal이 있으면, reasoning 대신 code solve, refinement 대신 patch generation 구조로 거의 그대로 옮길 수 있다.

2. **document AI 후처리에도 응용 가능하다.**  
   예를 들어 표 추출이나 field extraction에서 first pass 결과를 넣고 "다시 검토해서 고쳐라"를 시킨 뒤, exact-match 또는 schema-level verifier를 붙일 수 있다.

3. **distillation 앞단의 data engine으로도 가치가 있다.**  
   ThinkTwice 자체를 그대로 서비스 inference에 쓰지 않더라도, solve / revise pair를 많이 만들어 student model이나 reward model 학습 재료로 쓸 수 있다.

4. **reward redesign 없이 curriculum을 얻는다는 점이 좋다.**  
   많은 RL recipe가 reward engineering을 먼저 고민하는데, ThinkTwice는 input structuring만으로 curriculum을 얻는다. 이건 생각보다 실무적이다.

## 7-3. Follow-up papers

- DeepSeekMath / GRPO
- DAPO
- Dr. GRPO
- Self-Refine
- Reflexion
- PAG / Self-Verify 계열 RL-based refinement 논문

# 8. Summary

- ThinkTwice는 reasoning과 self-refinement를 한 shared policy 안에서 jointly optimize하는 2-phase RLVR 방법이다.
- solve phase와 revise phase 모두에 **같은 binary correctness reward**를 사용하고, critique annotation이나 correctness hint는 쓰지 않는다.
- direct reasoning 성능도 좋아지고, self-refinement 성능은 더 크게 좋아진다. Qwen3-4B에서는 AIME refine pass@4가 GRPO 48.91에서 ThinkTwice 60.43으로 오른다.
- training dynamics는 early correction, late preservation으로 이동하는 **rectify-then-fortify** 흐름을 보여준다.
- 이 논문의 가장 큰 가치는 reward를 더 복잡하게 만드는 대신, **같은 sparse reward가 더 많은 useful signal을 만들도록 학습 구조를 설계했다는 점**이다.
