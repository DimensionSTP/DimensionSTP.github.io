---
layout: single
title: "Function-Aware Fill-in-the-Middle as Mid-Training for Coding Agent Foundation Models Review"
categories: Study-concept
tag: [Coding-Agent, Mid-Training, Fill-in-the-Middle]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2607.12463)

[Code link](https://github.com/TIGER-AI-Lab/FIM-Midtraining)

> 한 줄 요약: 이 논문은 coding agent의 action-observation-continuation 구조가 code의 function call-return-downstream usage와 닮았다는 관찰에서 출발해, function 단위 FIM을 agentic post-training 직전의 mid-training objective로 사용한다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Coding agent 성능을 trajectory SFT나 RL 데이터만의 문제로 보지 않고, base model이 가진 structural prior의 문제로 본다.
- Internet-scale code에 이미 존재하는 function dependency를 agent tool interaction의 self-supervision으로 재해석한다.
- Random-span FIM 대신 program dependency graph와 complexity-inferability criterion으로 학습 target을 고른다.
- Qwen2.5-Coder 7B와 14B, Qwen3-8B, 세 post-training pipeline에서 일관된 개선을 검증한다.
- Agent benchmark뿐 아니라 LiveCodeBench, tau-bench, BFCL의 capability preservation까지 함께 본다.

Coding agent는 매 step에서 history를 읽고 action을 만들며, external tool이 반환한 observation을 받아 다음 reasoning을 이어간다. Standard left-to-right code pretraining도 function call과 return을 보지만, 대부분 완성된 code를 순서대로 읽을 뿐이다. 이미 존재하는 caller와 downstream usage를 보고 중간 function body를 역으로 복원하는 훈련은 상대적으로 약하다.

논문은 이 차이를 다음 구조적 대응으로 설명한다.

| Coding agent | Function call site |
| --- | --- |
| History | Pre-call context |
| Action | Function call |
| Observation | Externally computed return |
| Continuation | Downstream code |

이 대응이 문자 그대로 동일하다는 주장은 아니다. FIM은 suffix를 이미 보고 middle을 복원하지만, agent inference는 observation 이후 continuation을 생성한다. 논문의 가치는 이 structural analogy가 post-training 이후 실제 agent behavior에 transfer되는지 실험으로 확인한다는 데 있다.

# 1. Problem Setting

## 1-1. Problem definition

현재 coding agent pipeline은 보통 다음 세 단계로 구성된다.

1. Large-scale code pretraining
2. Instruction tuning or coding SFT
3. Agentic post-training with tool trajectories

실제 agent capability는 세 번째 단계에서 주로 주입된다고 가정하기 쉽다. 하지만 trajectory data는 비싸고, synthetic agent trace는 특정 harness와 teacher behavior를 모방할 수 있다. 또한 agentic post-training이 SWE task를 개선하는 대신 general coding이나 non-coding tool use를 떨어뜨리는 capability erosion도 생길 수 있다.

이 논문이 묻는 질문은 다음과 같다.

> Agent가 나중에 요구받을 conditioning structure를 ordinary code에서 미리 학습시킬 수 있는가?

Agent step은 다음처럼 쓸 수 있다.

$$
a_t \sim \pi(a_t \mid h_t), \qquad
o_{t+1} \sim p(o_{t+1} \mid h_t,a_t)
$$

Model은 $h_t$, $a_t$, $o_{t+1}$을 바탕으로 다음 continuation을 만든다. Function-aware FIM은 surrounding file의 prefix와 suffix를 주고, 그 사이의 function body와 rationale을 생성하게 한다. 즉 local completion보다 broader dependency를 읽게 만든다.

## 1-2. Why previous approaches are insufficient

### 1) Random-span FIM은 boundary가 임의적이다

Random FIM은 expression 중간이나 partial statement를 mask할 수 있다. 이런 target도 code completion에는 유용하지만, caller, callee, return usage를 연결하는 function-level reasoning signal은 약하다.

### 2) Direct code reconstruction에는 reasoning supervision이 없다

Agent는 observation을 받은 뒤 무엇을 수정하고 어떤 tool을 다시 호출할지 reasoning해야 한다. Random FIM이 code span만 복원하면 think-then-act structure를 직접 학습시키지 못한다.

### 3) Pretraining에 섞인 FIM signal은 희석된다

Trillion-token pretraining에서 일부 random FIM을 섞어도, agentic post-training 직전까지 그 inductive bias가 얼마나 남아 있는지 불분명하다. 논문은 별도의 mid-training stage로 signal을 집중한다.

### 4) Agent trajectory만 늘리면 target capability에 과적합할 수 있다

Post-training은 SWE issue solving을 높이면서 LiveCodeBench나 tool-use benchmark를 떨어뜨릴 수 있다. Base representation이 brittle한 상태에서 narrow trajectory를 반복하면 specialization cost가 커질 수 있다.

# 2. Core Idea

## 2-1. Main contribution

논문의 pipeline은 네 단계로 정리할 수 있다.

1. Python repository 수집과 decontamination
2. Program dependency graph construction
3. Complexity plus inferability 기반 function target selection
4. Rationale plus function body를 FIM middle로 생성하는 mid-training

그 다음 기존 agentic post-training pipeline은 수정하지 않는다. 즉 비교는 다음 두 경로다.

- Baseline: base model -> agentic post-training
- Proposed: base model -> function-aware FIM mid-training -> same agentic post-training

이 설계 덕분에 최종 gain이 새로운 agent harness나 RL recipe가 아니라 mid-training prior에서 왔는지 비교하기 쉬워진다.

## 2-2. Design intuition

좋은 FIM target은 두 조건을 동시에 만족해야 한다.

- 충분히 복잡해서 reasoning signal이 있어야 한다.
- Surrounding context만으로 어느 정도 inferable해야 한다.

너무 쉬운 function은 boilerplate completion이 되고, 너무 어려운 function은 external knowledge가 없으면 복원 불가능한 noise가 된다. 논문은 이를 complexity score $\hat{H}$와 inferability score $\hat{I}$로 분리한다.

Complexity score는 lines of code, cyclomatic complexity, nesting depth를 결합한다.

$$
\hat{H}(v) =
w_l\phi(\mathrm{LoC}(v),c_l)
+w_c\phi(\mathrm{CC}(v),c_c)
+w_d\phi(\mathrm{D}(v),c_d)
$$

Inferability score는 caller, callee, signature, docstring, class sibling signal을 모은다.

$$
\hat{I}(v) =
\alpha C_{caller}(v)
+\beta C_{callee}(v)
+\gamma C_{sig}(v)
+\delta C_{doc}(v)
+\varepsilon C_{class}(v)
$$

최종 score는 두 값이 모두 높을 때만 커지도록 harmonic-mean-like form을 쓴다.

$$
\mathrm{FIM}(v) =
\frac{\hat{H}(v)\hat{I}(v)}{\hat{H}(v)+\hat{I}(v)+\epsilon}
\rho(\Delta(v))
$$

$\rho(\Delta(v))$는 full context를 줘도 지나치게 어려운 target을 낮추는 difficulty penalty다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Agentic post-training 전에 tool-return-like conditioning prior를 주입 |
| Data source | 968 decontaminated Python repositories |
| Unit | Single function 또는 연결된 2-3 function group |
| Structure | AST-based program dependency graph |
| Selection | Complexity plus inferability plus difficulty penalty |
| Target | CoT rationale followed by function body |
| Training position | Base model과 agentic post-training 사이의 mid-training |
| Evaluated bases | Qwen2.5-Coder-Instruct 7B/14B, Qwen3-8B |

## 3-2. Module breakdown

### 1) Repository collection and decontamination

논문은 약 2,000개 candidate repository에서 quality filtering을 거쳐 968개 Python repository를 남긴다.

SWE-Bench source repository와 이름 및 known fork가 겹치는 repository를 제거한다. 또한 각 repository의 commit을 SWE-Bench-Verified와 SWE-Bench-Lite의 가장 이른 base commit 이전으로 제한한다.

최종 corpus는 다음과 같다.

- About 400K FIM samples
- About 2.6B tokens
- 320K single-function targets
- 60K function-pair targets
- 20K function-triple targets

### 2) Program dependency graph

각 file의 AST에서 top-level function과 class method를 node로 만든다. Edge는 두 종류다.

- Call edge: caller와 callee를 연결
- Sibling edge: 같은 class의 method를 연결

Sibling edge가 필요한 이유는 method가 직접 호출하지 않더라도 `self` state를 통해 강하게 결합될 수 있기 때문이다. Call resolution은 direct invocation, class instantiation, `self`와 `cls` method call을 처리하고, qualified name index를 사용한다.

### 3) Function target selection

각 function에 complexity와 inferability score를 계산한 뒤 threshold를 통과한 target만 남긴다. Dunder method, 지나치게 짧거나 긴 body 등은 hard filter로 제거한다.

Hand-designed score를 쓴 이유도 중요하다. Learned predictability model을 사용하면 selection policy가 특정 reference model에 종속되고, 다른 base model로 transfer되는지 해석하기 어려워진다.

### 4) Multi-function masking

Real code patch는 한 function에만 닫히지 않는 경우가 많다. 논문은 caller-callee, co-callee, sibling-coupled, mutual-call, call-chain, hub, fan-in, class-triad 같은 topology를 사용해 2-3 function group을 만든다.

Group inferability는 joint masking 상태에서 다시 계산한다. Mask될 function끼리 서로의 context를 제공해 score가 부풀려지는 것을 막기 위해서다.

### 5) Rationale generation and filtering

Gemini-3-Flash가 masked file만 보고 rationale와 candidate body를 만든다. Ground-truth body는 generation model에 주지 않는다.

별도의 Gemini-3-Flash judge가 candidate를 ground truth와 비교해 feasibility와 quality를 평가한다. Ground truth는 filter anchor일 뿐 training target으로 직접 들어가지 않는다.

최종 FIM sample은 다음 순서다.

```text
<fim_prefix> prefix
<fim_suffix> suffix
<fim_middle> rationale
body
```

Rationale와 body를 함께 middle에 넣어 reasoning과 action이 일치하도록 학습한다.

### 6) Existing post-training pipeline reuse

Mid-training 뒤에는 기존 pipeline을 그대로 사용한다.

- Qwen2.5-Coder 7B/14B plus R2E-Gym
- Qwen2.5-Coder 7B plus SWE-Smith
- Qwen3-8B plus SWE-Lego

이 점은 method의 practical value를 높인다. 새로운 agentic RL stack을 요구하지 않고 base checkpoint를 더 적합하게 만든다.

# 4. Training / Data / Recipe

## 4-1. Data

Corpus는 Python-only다. Tool call trace, shell interaction, customer support dialogue는 포함하지 않는다. 따라서 tau-bench와 BFCL improvement는 direct domain matching이 아니라 function-call inductive bias의 transfer evidence로 해석된다.

Training sample은 single, pair, triple target을 80:15:5 비율로 구성한 setting이 가장 좋았다. Single-only보다 multi-function group을 일부 섞는 것이 real patch의 cross-function dependency를 더 잘 반영한다.

## 4-2. Training strategy

FIM mid-training hyperparameter는 세 model family에 공통적으로 다음과 같다.

| Hyperparameter | Value |
| --- | --- |
| Optimizer | AdamW |
| Learning rate | $1\times10^{-5}$ |
| Schedule | Cosine |
| Warmup ratio | 0.1 |
| Weight decay | 0.05 |
| Epochs | 1 |
| Per-device batch | 1 |
| Gradient accumulation | 16 |
| Global batch | 128 |
| Sequence length | 32,768 |
| Precision | bf16 |

Post-training은 official pipeline을 가능한 한 유지한다. Qwen3-8B의 SWE-Lego만 overfitting을 피하기 위해 4 epoch 대신 2 epoch를 사용한다.

전체 실험은 8 H100 80GB를 사용했고, 논문은 모든 실험을 합친 compute를 약 5,760 GPU-hours로 보고한다. 이는 2.6B-token mid-training 하나만의 비용이 아니라 model, pipeline, ablation을 포함한 총량으로 읽어야 한다.

## 4-3. Engineering notes

### 1) FIM-only checkpoint를 바로 agent로 평가하면 안 된다

논문은 mid-training-only model이 instruction following을 잃을 수 있어 최종 비교에서 제외한다. 반드시 동일한 agentic post-training을 거친 checkpoint끼리 비교한다.

### 2) Decontamination은 repository와 commit time을 함께 봐야 한다

Repository name만 제거하면 fork나 이후 commit leakage가 남을 수 있다. Original source, fork relation, benchmark base commit timestamp를 함께 관리하는 것이 좋다.

### 3) Target selection audit가 필요하다

Complexity와 inferability score가 특정 code style에 편향될 수 있다. Sample별 score decomposition, selected/rejected example, topology distribution을 저장해야 한다.

### 4) Teacher rationale를 맹신하면 안 된다

Gemini-generated rationale는 training signal이지만, 실제 ground-truth implementation의 유일한 reasoning path는 아니다. No-CoT와 self-CoT ablation을 함께 보는 이유다.

### 5) Post-training recipe를 고정해야 한다

Mid-training effect를 보려면 downstream trajectory data, harness, optimizer, epoch가 같아야 한다. Qwen3 comparison은 base와 post-training pipeline이 함께 바뀌므로 cross-family generalization claim을 제한적으로 읽어야 한다.

# 5. Evaluation

## 5-1. Main results

SWE benchmark 결과는 다음과 같다. 모든 값은 final pipeline checkpoint를 세 evaluation seed로 평가한 평균이다.

| Base and post-training | SWE-Bench-Verified | SWE-Bench-Lite | Gain on Verified/Lite |
| --- | ---: | ---: | --- |
| Qwen2.5-Coder-7B plus R2E-Gym | 15.00 -> 17.80 | 11.33 -> 15.00 | +2.80 / +3.67 |
| Qwen2.5-Coder-7B plus SWE-Smith | 12.30 -> 17.60 | 14.20 -> 14.70 | +5.30 / +0.50 |
| Qwen2.5-Coder-14B plus R2E-Gym | 26.20 -> 29.20 | 18.00 -> 22.00 | +3.00 / +4.00 |
| Qwen3-8B plus SWE-Lego | 31.80 -> 35.00 | 27.30 -> 32.70 | +3.20 / +5.40 |

Gain이 모든 model과 pipeline에서 양수라는 점이 중요하다. 다만 R2E-Gym reproduced baseline과 official reported number가 다르므로, 논문은 동일 setup의 reproduced baseline을 primary comparison으로 사용한다.

### Capability preservation

14B setting에서 agentic post-training만 수행하면 일부 non-target capability가 크게 떨어진다. FIM mid-training을 앞에 넣으면 post-training-only checkpoint 대비 다음 회복이 보고된다.

- LiveCodeBench: +11.1
- OJBench: +1.94
- FullStackBench-EN: +0.53
- Terminal-Bench 2.0: +1.25
- tau-bench: +3.9
- BFCL: +2.4

Python-only corpus가 non-coding tool-use benchmark까지 개선했다는 점이 논문의 central hypothesis와 연결된다.

## 5-2. What really matters in the experiments

### 1) No-CoT FIM도 이미 효과가 있다

Ablation에서 rationale를 제거한 function-aware FIM도 baseline보다 좋아진다. 즉 gain 전체가 Gemini CoT distillation에서만 온 것은 아니다.

Self-CoT variant는 teacher rationale 대신 model 자체 rationale를 사용해 상당한 gain을 회복한다. Function selection 자체가 strongest component이고, PDG, complexity, inferability를 모두 사용한 full recipe가 가장 좋다.

### 2) Gain은 multi-function task에 집중된다

Gold patch shape별 분석에서 single-function task improvement는 상대적으로 작고, same-file multi-function task에서 gain이 크다.

- Single-function: +2.1 points
- Same-file multi-function: 13.6 -> 22.7, +9.1 points

반면 multi-file task에서는 차별적 gain이 나타나지 않는다. 이는 training unit이 file-local function dependency이므로 자연스러운 결과다.

### 3) Agent가 더 오래 시도하고 회복한다

Negative observation 이후 recovery는 24.8에서 28.8로 오른다. Solved task의 edit count는 3.3에서 7.4로, step count는 15.1에서 23.6으로 늘어난다.

이 수치는 무조건 효율이 좋아졌다는 뜻은 아니다. 더 많은 step과 edit를 사용해 iterate-and-verify behavior가 강화되었고, 그 결과 더 많은 task를 해결했다는 해석이 적절하다.

### 4) Capability preservation이 중요한 supporting evidence다

단순히 SWE-Bench gain만 보면 synthetic rationale distillation이나 extra token 효과로 설명할 수 있다. 하지만 non-agent coding과 non-coding tool use의 회복은 function-call prior가 narrow benchmark를 넘어 transfer될 수 있다는 추가 근거다.

# 6. Limitations

1. Python-only corpus다.
   - Static AST와 function boundary가 명확한 Python에 최적화되어 있다.
   - C++, Java, JavaScript, dynamically generated code로 확장할 때 parser와 dependency resolution을 다시 설계해야 한다.

2. Gemini teacher와 judge에 의존한다.
   - Rationale style과 filtering preference가 teacher model bias를 반영할 수 있다.
   - Self-CoT ablation이 일부 완화하지만 fully teacher-free pipeline은 아니다.

3. Qwen3 comparison은 confounded되어 있다.
   - Base model뿐 아니라 post-training pipeline도 SWE-Lego로 바뀐다.
   - Cross-family transfer의 증거지만 family-independent guarantee로 보기는 어렵다.

4. Function modularity 가정이 항상 맞지 않는다.
   - Monolithic script, generated code, notebook, global side effect가 많은 project에서는 function body가 적절한 semantic unit이 아닐 수 있다.

5. Multi-file dependency는 직접 해결하지 못한다.
   - Gain이 same-file multi-function task에 집중되고 multi-file task에는 differential gain이 약하다.
   - Repository-level retrieval이나 cross-file graph가 후속 과제다.

6. Mid-training compute가 작지 않다.
   - 2.6B tokens와 teacher generation/filtering, 여러 post-training run이 필요하다.
   - Gain당 data and compute cost를 production pipeline에서 따로 계산해야 한다.

7. Rationale correctness는 완전히 검증되지 않는다.
   - Candidate body가 feasible하더라도 rationale가 causal explanation인지, plausible narration인지 분리하기 어렵다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문의 가장 좋은 점은 agent capability를 더 많은 trajectory의 문제로만 보지 않았다는 데 있다. Tool-use structure와 닮은 supervision을 ordinary code에서 찾아 base representation에 먼저 넣고, 이후 기존 post-training을 그대로 적용한다.

이는 agent data가 부족한 환경에서 특히 실용적이다. Domain-specific tool trace를 대량 생성하기 전에, domain artifact 자체에서 action-observation-like dependency를 찾아 mid-training할 수 있다는 일반화된 방향을 제시하기 때문이다.

## 7-2. Reuse potential

### 1) Document workflow mid-training

문서의 field dependency, formula reference, cross-section citation을 mask하고 surrounding context에서 복원하게 하면 document agent의 observation integration prior로 확장할 수 있다.

### 2) Tool schema-aware FIM

API call, return schema, downstream variable usage가 있는 code를 target으로 골라 function-call and tool-call bridge를 더 직접적으로 만들 수 있다.

### 3) Failure-aware target mining

Agent trajectory에서 자주 실패하는 action type을 분석한 뒤, 그 구조와 대응하는 code function topology를 더 많이 sample할 수 있다.

### 4) Cross-file extension

Import graph, symbol index, call graph를 repository level로 확장해 multi-file target을 만들 수 있다. 현재 논문의 가장 명확한 다음 단계다.

### 5) Capability preservation benchmark

Agentic post-training 전후에 target benchmark뿐 아니라 general coding, tool use, instruction following을 regression suite로 고정하는 방식은 바로 재사용할 수 있다.

## 7-3. Follow-up papers

- Fill-in-the-Middle: Enabling Language Models to Fill in Arbitrary Text Infills
- SWE-Gym
- R2E-Gym
- SWE-Smith
- SWE-Lego
- Code2LoRA
- SWE-Bench

# 8. Summary

- 이 논문은 function call-return-downstream usage를 coding agent의 action-observation-continuation과 대응시킨다.
- Random FIM 대신 PDG와 complexity-inferability criterion으로 function-level target을 고른다.
- 968개 Python repository, 400K samples, 2.6B tokens로 mid-training한 뒤 기존 agentic post-training을 그대로 적용한다.
- Qwen2.5-Coder 7B/14B와 Qwen3-8B에서 SWE benchmark가 일관되게 개선되고 capability erosion도 완화된다.
- Gain이 same-file multi-function task에 집중된다는 점은 method의 강점과 현재 범위를 동시에 보여준다.
