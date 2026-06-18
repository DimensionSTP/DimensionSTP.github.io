---
layout: single
title: "VibeThinker-3B: Exploring the Frontier of Verifiable Reasoning in Small Language Models Review"
categories: Study-concept
tag: [VibeThinker, SmallLanguageModel, ReinforcementLearning, Reasoning]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2606.16140)

[Hugging Face model](https://huggingface.co/WeiboAI/VibeThinker-3B)

[GitHub repository](https://github.com/WeiboAI/VibeThinker)

> 한 줄 요약: VibeThinker-3B는 Qwen2.5-Coder-3B를 기반으로 curriculum SFT, capability-boundary 중심의 multi-domain RL, offline self-distillation, instruction RL을 순차적으로 결합해 3B dense model의 verifiable reasoning 성능을 밀어붙인다. 이 논문의 핵심은 작은 model 하나의 leaderboard 기록보다, verifier가 강한 domain에서 어떤 post-training loop가 parameter scale의 일부를 대체할 수 있는지 보여주는 데 있다.

이 논문을 단순히 3B model이 대형 model을 이겼다는 이야기로 읽으면 두 가지를 놓치게 된다. 첫째, 성능의 상당 부분은 architecture innovation이 아니라 여러 training stage를 연결한 recipe에서 나온다. 둘째, 저자들이 주장하는 frontier는 모든 task에 대한 general intelligence가 아니라 정답 검증이 가능한 math, competitive programming, STEM reasoning에 집중되어 있다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같다.

- Small language model의 성능 한계를 parameter count가 아니라 post-training signal design 관점에서 분석한다.
- Broad SFT에서 hard reasoning RL로 바로 넘어가지 않고, diversity, capability boundary, trajectory consolidation, instruction controllability를 각각 별도 stage로 다룬다.
- 64K context를 RL 시작부터 유지하고, accuracy를 먼저 확보한 뒤 response length를 줄이는 방식처럼 long reasoning model을 학습할 때 실무적으로 중요한 선택을 제시한다.
- AIME26 94.3, LiveCodeBench v6 80.2, IFEval 93.4를 함께 보고해 reasoning 강화와 instruction following 사이의 trade-off를 확인한다.
- Claim-Level Reliability Assessment, 이하 CLR을 통해 base model 성능과 test-time compute 성능을 분리해 볼 수 있다.

# 1. Problem Setting

## 1-1. Small model에서 무엇을 frontier라고 부를 것인가

Small model 연구에서는 흔히 parameter 수 대비 benchmark score를 비교한다. 하지만 이 비교에는 서로 다른 능력이 섞여 있다.

- Multi-step deduction과 constraint satisfaction
- 계산 결과나 code execution으로 검증할 수 있는 reasoning
- Open-domain fact와 long-tail knowledge recall
- 자연스러운 dialogue와 broad instruction handling
- Tool use, API orchestration, repository-scale coding

VibeThinker-3B가 집중하는 영역은 첫 두 항목이다. 정답을 rule, symbolic checker, test case, option match로 검증할 수 있다면 training signal의 noise를 낮출 수 있고, 많은 rollout 중 유효한 trajectory를 선별할 수 있다. 저자들은 이 조건에서 reasoning capability가 비교적 compact한 parameter space에 압축될 가능성을 탐색한다.

반대로 open-domain knowledge는 model 내부에 넓은 fact coverage가 필요하다. 같은 3B model이라도 reasoning procedure를 잘 학습하는 것과 세계 지식을 폭넓게 저장하는 것은 다른 scaling law를 가질 수 있다는 것이 논문의 문제의식이다.

## 1-2. 기존 small reasoning model의 병목

작은 model에 SFT와 RL을 그대로 적용하면 다음 문제가 생긴다.

### 1) Narrow SFT spectrum

한 종류의 정답 trajectory만 모으면 model은 특정 template를 잘 따라 하지만, 다른 solution path나 문제 변형에 취약해질 수 있다. Small model은 capacity가 제한되어 있으므로 data diversity를 늘릴 때 유효 pattern과 noise를 구분하는 작업도 더 중요하다.

### 2) Weak or saturated RL signal

현재 policy가 거의 항상 실패하는 prompt는 reward가 대부분 0이고, 거의 항상 맞히는 prompt는 reward가 대부분 1이다. 두 경우 모두 group-based policy optimization에서 sample 간 advantage가 작아진다. 학습 가능한 경계에 있는 prompt를 구분하지 않으면 많은 rollout budget이 signal이 약한 문제에 쓰인다.

### 3) Long reasoning truncation

Long-context capability를 점진적으로 늘리는 방식은 약한 initialization에는 도움이 될 수 있다. 그러나 이미 긴 reasoning path를 생성할 수 있는 model에서는 짧은 context warmup이 오히려 valid trajectory를 자르고, truncated answer를 reward signal에 섞을 수 있다.

### 4) Cross-domain interference

Math, code, STEM은 verifier와 오류 구조가 다르다. 모든 domain을 하나의 RL mixture로 동시에 넣으면 특정 domain이 reward scale이나 sampling 빈도를 지배할 수 있고, 이전 stage에서 얻은 능력이 다음 stage에서 약해질 수 있다.

### 5) Accuracy와 efficiency의 충돌

Reasoning model은 더 오래 생각할수록 정답률이 오르는 경향이 있다. 학습 초기에 length penalty를 강하게 주면 아직 형성되지 않은 reasoning chain을 잘라 accuracy ceiling을 낮출 수 있다. 반대로 accuracy만 최적화하면 inference token cost가 커진다.

### 6) Reasoning 강화 후 instruction degradation

Math와 code reward만 반복해서 최적화하면 format, item count, response order, concise answer 같은 user constraint를 무시할 수 있다. Benchmark score가 높아져도 product-facing model로는 쓰기 어려울 수 있다.

# 2. Core Idea

## 2-1. Spectrum-to-Signal Principle

VibeThinker-3B의 전체 pipeline은 Spectrum-to-Signal Principle, 이하 SSP로 요약된다.

- SFT는 가능한 reasoning path의 spectrum을 넓힌다.
- RL은 verifier가 확인한 correct signal을 강화한다.
- Offline self-distillation은 domain별 RL stage에서 발견한 유효 trajectory를 하나의 model에 다시 통합한다.
- Instruct RL은 reasoning optimization 과정에서 약해질 수 있는 controllability를 복구한다.

이 순서는 중요하다. RL이 새로운 reasoning style을 무에서 만들어내기를 기대하기보다, SFT가 여러 valid path를 먼저 policy support 안에 넣고 RL이 그중 reward가 높은 path를 선택하도록 만든다.

## 2-2. Curriculum-based two-stage SFT

SFT는 broad coverage와 hard reasoning을 한 번에 처리하지 않는다.

### Stage 1: broad spectrum

Math, code, STEM, general dialogue, instruction following을 함께 학습한다. Seed query를 확장하고 여러 teacher response를 생성한 뒤, majority voting과 verifier를 사용해 valid trajectory를 선별한다. Intermediate reasoning step을 유지해 final answer만 맞는 short label보다 다양한 solution skeleton을 model에 노출한다.

### Stage 2: hard and long reasoning

Stage 1 model이 이미 잘 푸는 짧은 문제보다 긴 reasoning이 필요한 prompt에 집중한다. 논문은 5K token보다 짧은 trajectory를 버리고, VibeThinker-1.5B로 prompt마다 8개 rollout을 생성한 뒤 error rate가 0.75보다 낮은 쉬운 문제를 제외한다.

이 curriculum은 단순한 난이도 정렬이 아니다. Stage 1은 coverage를 만들고, Stage 2는 model의 현재 boundary 밖에 가까운 trajectory를 policy support 안으로 가져오는 역할을 한다.

## 2-3. Diversity-Exploring Distillation

저자들은 validation loss나 Pass@1이 가장 높은 단일 checkpoint만 고르지 않는다. Training 중 여러 checkpoint를 domain별 Pass@K로 probe하고, 서로 다른 valid solution을 더 많이 생성하는 specialist checkpoint를 찾는다. 그 뒤 선택한 checkpoint parameter를 merge해 다음 stage의 unified SFT model을 만든다.

이 설계의 직관은 다음과 같다.

- Pass@1은 가장 자주 나오는 한 path의 품질에 민감하다.
- Pass@K는 policy support 안에 얼마나 다양한 correct path가 있는지 더 잘 보여준다.
- RL 이전에는 한 path의 confidence보다 valid trajectory coverage가 중요할 수 있다.

즉 SFT checkpoint selection 자체를 diversity optimization 문제로 바꾼다.

## 2-4. Capability boundary를 노리는 MGPO

Multi-domain RL은 MaxEnt-Guided Policy Optimization, 이하 MGPO를 사용한다. Prompt $q$에 대해 $G$개의 response를 sampling하고, group accuracy를 다음처럼 계산한다.

$$
p(q) = \frac{1}{G}\sum_{i=1}^{G}\mathbf{1}[r_i = 1]
$$

$p(q)$가 0에 가까우면 현재 policy가 거의 풀지 못하고, 1에 가까우면 이미 충분히 쉽다. 가장 informative한 prompt는 성공과 실패가 섞이는 $p(q) = 0.5$ 부근이다. 논문은 이 capability boundary에 가까운 prompt의 contribution을 크게 만든다.

$$
w(q) = \exp\left(-\gamma D_{\mathrm{ME}}\left(p(q)\|0.5\right)\right)
$$

여기서 $D_{\mathrm{ME}}$는 논문에서 사용하는 max-entropy guided distance이고, $w(q)$는 GRPO-style clipped objective에 곱해지는 prompt weight다.

핵심은 hard example mining과 조금 다르다. 항상 어려운 문제를 고르는 것이 아니라, 현재 policy가 일부는 성공하고 일부는 실패하는 학습 가능한 문제에 rollout budget을 집중한다.

## 2-5. Sequential multi-domain RL

RL은 Math, Code, STEM을 동시에 섞지 않고 순차적으로 진행한다.

1. Math RL은 final answer verifier를 사용한다.
2. Code RL은 sandbox execution과 test case를 사용한다.
3. STEM RL은 answer extraction과 option verification을 사용한다.

각 stage 시작 시 accuracy가 정확히 0 또는 1인 prompt를 제외한다. 모두 실패하거나 모두 성공하는 group은 relative advantage가 약하기 때문이다.

또한 64K context를 RL 처음부터 유지한다. 저자들은 더 짧은 context에서 시작해 늘리는 progressive schedule이 이미 강한 3B initialization에서는 long trajectory truncation을 늘리고 성능을 떨어뜨렸다고 보고한다.

## 2-6. Accuracy first, efficiency second

Math RL 뒤에는 Long2Short stage를 둔다. 먼저 accuracy를 충분히 끌어올리고, 그 다음 correct response끼리만 length를 비교한다.

Correct response 집합을 $C(q)$라고 하고 response length를 $L_i$라고 하면, brevity signal은 다음처럼 둘 수 있다.

$$
s_i = \frac{1}{L_i}
$$

논문은 correct group 안에서 $s_i$를 중심화해 짧은 correct path에는 positive shift, 긴 correct path에는 negative shift를 준다. Reward shift의 합은 0이 되도록 구성한다.

$$
\sum_{i \in C(q)} \Delta r_i = 0
$$

Incorrect response의 reward는 그대로 두며, length reward coefficient는 0.2를 사용한다. 이 방식은 짧은 오답을 장려하지 않고, 이미 맞힌 trajectory 사이에서만 efficiency preference를 만든다.

## 2-7. Offline self-distillation as consolidation

Sequential RL은 각 domain에 강한 checkpoint를 만들지만 하나의 model에 모든 능력이 안정적으로 남는다는 보장은 없다. 저자들은 Math, Code, STEM checkpoint에서 verified trajectory를 다시 수집해 unified student에 SFT한다.

이때 단순히 모든 correct response를 넣지 않고 learning potential score를 사용한다.

$$
S_{\mathrm{LP}}(q,y) = -\frac{1}{|y|}\sum_{t=1}^{|y|}\log \pi_{\mathrm{student}}\left(y_t \mid q,y_{<t}\right)
$$

Score가 높다는 것은 response가 verifier 기준으로는 맞지만 현재 student가 아직 낮은 probability를 주고 있다는 뜻이다. 논문은 domain별 length bucket 안에서 score를 비교하고, 지나치게 짧거나 극단적인 outlier는 제외한 뒤 middle-to-high learning potential trajectory를 우선한다.

이 stage는 teacher output을 압축하는 일반 distillation보다, 여러 RL specialist가 발견한 correct behavior를 하나의 model에 replay하는 consolidation에 가깝다.

## 2-8. Instruct RL

마지막 stage는 user-facing constraint를 다시 강화한다.

- Format, order, item count, keyword inclusion처럼 명시적인 constraint는 rule-based validator로 평가한다.
- Helpfulness, coherence, adherence, redundancy처럼 open-ended한 품질은 rubric-based reward model로 평가한다.
- Long-context instruction과 general alignment prompt도 포함한다.

Reasoning score를 유지하면서 IFEval 93.4를 얻었다는 결과는 이 stage의 필요성을 보여준다. 다만 stage별 ablation이 충분히 공개되지 않아 어느 component가 instruction following을 얼마나 복구했는지는 분리하기 어렵다.

## 2-9. Claim-Level Reliability Assessment

CLR은 training method가 아니라 answer-verifiable task를 위한 test-time scaling procedure다.

1. 문제마다 32개 candidate trajectory를 생성한다.
2. 각 trajectory에서 decision-relevant claim 5개와 final answer를 추출한다.
3. Model이 각 claim을 다시 검증해 binary verdict를 만든다.
4. Claim 하나의 실패도 크게 반영하도록 nonlinear reliability score를 계산한다.
5. Equivalent final answer를 cluster하고 cluster별 reliability를 합산한다.
6. 전체 procedure를 8회 반복한 뒤 평균 Pass@1을 보고한다.

따라서 AIME26 94.3과 97.1은 같은 inference budget의 결과가 아니다. 94.3은 base evaluation이고, 97.1은 32-way sampling과 repeated verification을 포함한 CLR 결과다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Base model | Qwen2.5-Coder-3B lineage의 3B dense causal LM |
| Main target | Math, code, STEM처럼 answer verification이 가능한 reasoning |
| SFT design | Broad coverage Stage 1 + hard and long Stage 2 |
| Diversity device | Pass@K-based specialist checkpoint selection and parameter merge |
| RL design | MGPO with capability-boundary weighting |
| Domain schedule | Math RL -> Code RL -> STEM RL |
| Long reasoning | Single 64K context from RL start |
| Efficiency stage | Long2Short after accuracy optimization |
| Consolidation | Learning-potential filtered offline self-distillation |
| Alignment | Rule and rubric based Instruct RL |
| Test-time scaling | CLR with candidate sampling and claim verification |

## 3-2. End-to-end pipeline

전체 흐름을 model state 관점에서 보면 다음과 같다.

### 1) Base initialization

Qwen2.5-Coder-3B를 시작점으로 사용한다. 이미 code와 structured generation에 강한 initialization을 선택했기 때문에 결과를 일반 3B base model의 평균적 한계로 해석해서는 안 된다.

### 2) Broad SFT model

Multiple domain과 instruction data로 valid response space를 넓힌다. Query expansion, teacher sampling, majority voting, verifier filtering, n-gram decontamination을 통해 training candidate를 만든다.

### 3) Hard reasoning SFT model

Long trace와 high-error prompt로 distribution을 이동한다. 이 stage는 model이 RL에서 탐색할 수 있는 long reasoning path를 미리 확보한다.

### 4) Diversity-preserving merge

Periodic checkpoint를 domain별 Pass@K로 평가하고 specialist를 고른다. Parameter merge를 통해 single checkpoint selection보다 넓은 solution support를 다음 stage에 전달한다.

### 5) Sequential RL specialists

Math, Code, STEM 순서로 on-policy RL을 수행한다. 각 stage는 domain-specific verifier를 사용하고, capability boundary에 가까운 prompt를 강조한다.

### 6) Long2Short math specialist

정답률을 먼저 확보한 뒤 correct response 사이의 length preference만 추가한다. Accuracy와 token efficiency를 같은 시점에 최적화하지 않는 것이 핵심이다.

### 7) Unified distilled model

각 specialist가 생성한 verified trajectory 중 student가 아직 잘 설명하지 못하는 example을 골라 다시 SFT한다. RL stage 사이의 forgetting을 줄이고 domain capability를 하나의 checkpoint에 모은다.

### 8) Instruction-aligned model

Rule validator와 rubric reward를 사용해 user constraint와 general response quality를 보강한다.

### 9) Optional CLR inference

Single response를 바로 내지 않고 candidate generation, claim extraction, self-verification, answer aggregation을 추가한다. 이는 model weight의 개선이 아니라 inference-time search와 verification이다.

## 3-3. Why on-policy matters here

논문은 RL stage를 on-policy로 유지한다. 이유는 train-time policy와 inference-time sampling distribution이 달라질 때 long reasoning trajectory의 token probability mismatch가 누적되기 때문이다.

Offline response만 반복해서 학습하면 다음 문제가 생길 수 있다.

- Teacher가 생성한 trace는 현재 student policy가 실제로 방문하지 않는 state를 포함한다.
- 긴 trajectory에서는 작은 token distribution 차이가 뒤쪽 reasoning branch를 크게 바꾼다.
- Verifier가 final answer만 확인하면 intermediate path의 off-policy error를 놓칠 수 있다.

VibeThinker pipeline은 RL exploration은 current policy로 수행하고, offline self-distillation은 verified specialist behavior를 consolidation하는 별도 stage로 제한한다. 이 역할 분리가 설계상 중요한 부분이다.

# 4. Training / Data / Recipe

## 4-1. Data construction

논문은 domain별 exact sample count를 공개하지 않지만, data construction 원칙은 비교적 자세히 설명한다.

### Query generation

- Reliable seed query를 준비한다.
- Concept composition, solving skeleton, constraint, evaluation objective를 바꿔 query를 확장한다.
- Math, code, STEM, general dialogue, instruction following을 포함한다.

### Response generation

- 여러 teacher sample을 생성한다.
- Answer-level majority voting으로 pseudo-label reliability를 높인다.
- Final answer뿐 아니라 complete intermediate reasoning을 유지한다.
- 서로 다른 valid solution path를 가능한 한 남긴다.

### Quality control

- Evaluation set과의 overlap을 줄이기 위해 n-gram filtering을 사용한다.
- LLM-based query quality filter를 적용한다.
- Math와 STEM은 answer verifier를 사용한다.
- Code는 execution과 test case로 확인한다.
- Majority vote와 trace-level correctness check를 함께 사용한다.

다만 teacher model의 구체적 identity, domain별 생성량, filtering pass rate, deduplication threshold는 technical report에서 충분히 공개되지 않는다.

## 4-2. SFT schedule

### Stage 1

- Global batch size: 128
- Initial learning rate: $5 \times 10^{-5}$
- Final learning rate: $8 \times 10^{-8}$
- Scheduler: cosine decay
- Epochs: 5
- Warmup: first 5 percent
- Sequence packing: enabled

### Stage 2

- 5K token보다 짧은 trajectory를 제외한다.
- VibeThinker-1.5B로 prompt당 8개 rollout을 생성한다.
- Error rate가 0.75보다 낮은 prompt를 제외한다.
- Stage 1과 같은 optimizer schedule을 사용한다.
- Additional epochs: 2

Stage 2의 filtering은 모든 hard example을 넣는다는 뜻이 아니다. 매우 어려워 현재 policy가 전혀 signal을 만들지 못하는 prompt보다, long correct trace가 일부 존재하면서 model이 자주 실패하는 영역을 겨냥한다.

## 4-3. RL recipe

### Prompt filtering

각 RL stage 시작 시 group accuracy가 0 또는 1인 prompt를 제외한다. Relative reward를 만들 수 있는 prompt에 sampling budget을 집중하기 위한 선택이다.

### Reward source

| Domain | Main verification signal | Main risk |
| --- | --- | --- |
| Math | Final answer verification | Correct answer with flawed intermediate logic |
| Code | Sandbox execution and tests | Weak or incomplete test coverage |
| STEM | Answer extraction and option verification | Knowledge recall and reasoning conflation |
| Instruction | Rule validators and rubric reward models | Reward model bias and rubric gaming |

### Context schedule

RL 시작부터 64K context를 사용한다. Progressive context extension은 이 model initialization에서 truncation을 늘려 long reasoning을 해쳤다고 보고한다.

이 결과는 모든 model에 64K-from-start가 최선이라는 뜻은 아니다. Base model이 이미 long generation을 안정적으로 수행하는지, optimizer와 memory budget이 충분한지에 따라 달라질 수 있다.

### Long2Short

Math accuracy optimization 뒤에 별도 stage로 적용한다. Incorrect response에는 brevity reward를 주지 않고, correct response group 안에서만 short path를 선호한다. Coefficient는 0.2다.

## 4-4. Offline self-distillation recipe

1. Math, Code, STEM RL checkpoint에서 response를 다시 sampling한다.
2. Domain verifier로 correct trajectory만 남긴다.
3. Student negative log-likelihood를 length-normalized learning potential로 계산한다.
4. Domain별 length bucket 안에서 trajectory를 rank한다.
5. 지나치게 쉬운 example과 extreme outlier를 제외한다.
6. Middle-to-high learning potential data로 unified student를 SFT한다.

이 stage의 장점은 현재 student에게 정보량이 낮은 already-mastered response를 줄이고, specialist가 알지만 unified model은 아직 약한 behavior에 training budget을 쓰는 데 있다.

## 4-5. Engineering notes

### 1) Small parameter count와 cheap training은 같은 말이 아니다

3B model은 weight memory와 serving footprint가 작다. 그러나 64K on-policy rollout, multiple domain verifier, periodic checkpoint probe, candidate distillation, CLR까지 포함하면 전체 research compute는 작지 않을 수 있다. Technical report는 total token count, GPU type, wall-clock time, end-to-end training cost를 공개하지 않는다.

### 2) Verifier quality가 upper bound를 만든다

Math answer checker와 code test가 잘못된 response를 통과시키면 RL은 그 오류를 빠르게 증폭한다. Product domain에 recipe를 재사용하려면 model architecture보다 먼저 verifier precision과 coverage를 측정해야 한다.

### 3) Evaluation sampling과 serving policy를 분리해야 한다

Paper evaluation은 temperature 1.0, top-p 0.95, top-k -1을 사용한다. 실제 service에서는 latency, determinism, token budget이 다르므로 같은 score가 그대로 재현되지 않을 수 있다.

### 4) CLR은 별도 system component다

CLR의 32 candidates, claim extraction, self-verification, answer clustering, 8-repeat evaluation을 single-pass model score와 섞으면 안 된다. Deployment에서는 latency와 verifier failure mode를 별도로 계산해야 한다.

# 5. Evaluation

## 5-1. Evaluation protocol

Technical report는 vLLM에서 다음 decoding 설정을 사용한다.

- Temperature: 1.0
- Top-p: 0.95
- Top-k: -1
- Additional output cap: 없음

Reported Pass@1은 한 번의 deterministic run이 아니라 여러 sampled generation의 평균이다.

- Math: 64 generations
- IMO-AnswerBench: 16 generations
- Knowledge: 16 generations
- Coding: 8 generations

Math는 rule-based verifier와 LLM judge를 함께 사용하고, code는 execution 결과로 평가한다. 비교 model score 일부는 각 model report, public leaderboard, official record에서 수집되므로 모든 baseline이 동일 inference stack과 동일 sampling budget으로 재평가된 것은 아니다.

## 5-2. Main results

아래 표는 논문의 핵심 비교 중 일부만 정리한 것이다.

| Model or setting | Params | AIME25 | AIME26 | IMO-AnswerBench | LiveCodeBench v6 | OJBench | GPQA-Diamond | IFEval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | 4B | 79.8 | 84.0 | 48.7 | 62.0 | 23.5 | 76.2 | 89.8 |
| VibeThinker-3B | 3B | 91.4 | 94.3 | 76.4 | 80.2 | 38.6 | 70.2 | 93.4 |
| VibeThinker-3B + CLR | 3B + test-time compute | 96.7 | 97.1 | 80.6 | - | - | 72.9 | - |

이 표에서 먼저 볼 것은 AIME score 하나가 아니다.

### 1) Verifiable math and code

AIME26 94.3과 LiveCodeBench v6 80.2는 3B model의 parameter scale을 고려하면 매우 강하다. Math와 competitive programming처럼 answer verification이 명확한 domain에서 SSP pipeline이 효과적이라는 주장을 뒷받침한다.

### 2) Instruction following

IFEval 93.4는 reasoning RL 이후에도 explicit instruction control이 유지되었음을 보여준다. 단순히 math specialist를 만든 것이 아니라 final Instruct RL을 통해 user-facing constraint를 일부 복구했다는 근거다.

### 3) Knowledge-heavy reasoning

GPQA-Diamond는 70.2로 강하지만 Qwen3.5-4B의 76.2보다 낮다. CLR을 적용해도 72.9다. 저자들의 Parametric Compression-Coverage Hypothesis와 맞는 패턴이지만, 이 한 benchmark만으로 reasoning과 knowledge의 scaling law를 확정할 수는 없다.

### 4) Base model and CLR

AIME26 94.3에서 97.1로 오른 결과는 weight improvement가 아니라 test-time sampling과 verification의 효과다. CLR은 accuracy를 올리지만 generation count, claim extraction, verification, clustering을 추가한다. Model efficiency를 논할 때는 base score와 CLR score를 분리해야 한다.

## 5-3. Recent LeetCode evaluation

저자들은 2026-04-25부터 2026-05-31까지의 unseen LeetCode weekly와 biweekly contest를 사용한다.

- 8 contests
- Contest당 4 problems
- Problem당 4 rollouts
- 총 128 first-attempt Python submissions
- Passed: 123
- Acceptance rate: 96.1 percent

동일 표에서 GPT-5.3-Codex는 128, Gemini 3.1 Pro는 127, Gemini 3 Flash는 124를 통과한다.

이 실험의 장점은 paper release 직전 contest로 temporal overlap risk를 줄였다는 점이다. 다만 competitive programming에 한정된 small sample이며, repository navigation, dependency management, tool calling, test repair 같은 software engineering 능력은 측정하지 않는다. Official model card도 tool calling과 agent-based programming 용도로 추천하지 않는다고 명시한다.

## 5-4. What the experiments actually establish

### Strong evidence

- 3B dense model도 verifier-rich math and code domain에서 높은 reasoning accuracy를 낼 수 있다.
- Broad-to-hard SFT, capability-boundary RL, specialist consolidation, instruction RL을 연결한 recipe가 practical value를 가진다.
- Long reasoning과 instruction following을 반드시 zero-sum trade-off로 볼 필요는 없다.
- Test-time claim verification은 answer-verifiable task의 accuracy를 추가로 높일 수 있다.

### Evidence that is still incomplete

- 각 training stage가 최종 score에 기여한 양을 분리하는 ablation이 부족하다.
- Total training tokens와 compute가 공개되지 않아 parameter efficiency와 training efficiency를 동시에 주장하기 어렵다.
- Baseline의 inference budget과 evaluator가 완전히 통일되지 않았다.
- Verifiable benchmark 성능이 open-domain assistant, agent, real-world coding으로 전이된다는 증거는 없다.
- CLR improvement를 low-cost inference improvement로 해석할 수 없다.

# 6. Limitations

## 6-1. Stage-level causal evidence가 부족하다

Pipeline에는 two-stage SFT, diversity distillation, MGPO, sequential RL, Long2Short, offline self-distillation, Instruct RL이 포함된다. 하지만 full ablation matrix가 없어 AIME26 94.3 중 어느 stage가 얼마나 기여했는지 알기 어렵다.

특히 다음 비교가 필요하다.

- Standard GRPO vs MGPO under matched rollout budget
- Mixed multi-domain RL vs sequential RL
- Progressive context vs 64K-from-start under matched truncation rate
- Random correct trace vs learning-potential filtering
- With and without checkpoint diversity merge
- With and without Instruct RL while holding reasoning score constant

## 6-2. Training compute와 data scale가 불투명하다

Model은 3B지만 training pipeline은 compute-intensive하다. Technical report에는 다음 정보가 충분히 없다.

- Domain별 SFT sample count
- Teacher model identity와 generation budget
- RL prompt count와 rollout count
- Total training tokens
- GPU type, GPU hours, wall-clock time
- Verifier rejection rate
- Stage별 checkpoint merge coefficient

이 정보가 없으면 다른 team이 recipe를 재현하거나 cost-effectiveness를 비교하기 어렵다.

## 6-3. Benchmark comparison이 완전한 apples-to-apples는 아니다

Paper는 baseline score 일부를 official report와 public leaderboard에서 수집한다. Model마다 prompt format, system prompt, reasoning mode, generation count, output length, judge가 다를 수 있다. Parameter 대비 score 비교는 인상적이지만 exact rank는 보수적으로 읽어야 한다.

## 6-4. Verifiable domain에 강하게 특화되어 있다

Official model card는 tool calling, API orchestration, autonomous coding agent에 사용하지 말라고 명시한다. Coding 결과도 competitive programming 중심이다.

따라서 다음 능력은 별도 평가가 필요하다.

- Repository-scale code search and edit
- Multi-file dependency reasoning
- Tool selection and function calling
- Web or database grounded research
- Long conversation memory
- Multilingual and culturally grounded instruction
- Safety-critical refusal and uncertainty calibration

## 6-5. Knowledge coverage의 한계

GPQA-Diamond gap은 small model이 reasoning procedure를 잘 배워도 broad factual coverage가 부족할 수 있음을 보여준다. Retrieval augmentation이나 external tool을 붙이지 않은 closed-book setting에서 이 차이는 parameter scale과 pretraining corpus의 영향을 함께 받는다.

## 6-6. Long output cost를 무시할 수 없다

3B model은 per-token inference가 가볍지만 reasoning token이 길어지면 end-to-end latency와 energy가 커진다. Model card example은 매우 큰 `max_new_tokens`를 허용하며, paper evaluation도 additional output cap을 두지 않는다.

Small model의 practical advantage를 입증하려면 accuracy뿐 아니라 다음 지표가 필요하다.

- Correct answer당 generated tokens
- Time to first token
- End-to-end latency
- Energy per solved problem
- Accuracy under fixed token budget
- Accuracy under fixed dollar or GPU-second budget

## 6-7. CLR의 compute와 self-verification bias

CLR은 32 candidate trajectory와 5 claims per trajectory를 처리하고 전체 procedure를 8회 반복한다. Candidate generation뿐 아니라 claim extraction, claim verification, answer clustering도 필요하다.

또한 같은 model family가 answer를 만들고 claim을 검증하면 correlated error가 남을 수 있다. 그럴듯한 but shared misconception을 모든 candidate가 반복하면 self-verification도 실패할 수 있다.

## 6-8. Parametric Compression-Coverage는 hypothesis다

논문의 결론은 verifiable reasoning이 parameter-dense and compressible하고, open-domain knowledge는 broad coverage를 위해 더 큰 parameter가 필요하다는 가설이다. 현재 결과는 이 가설과 일관되지만 다음 alternative explanation도 가능하다.

- Qwen2.5-Coder-3B initialization이 structured reasoning에 이미 강하다.
- Training data와 verifier가 benchmark distribution에 매우 잘 맞는다.
- Open-domain score 차이는 parameter가 아니라 pretraining data mix에서 온다.
- Long sampling budget이 small model capacity를 보완한다.

더 다양한 base model, data budget, parameter scale, domain으로 controlled scaling study가 필요하다.

## 6-9. Public artifact의 범위

Model weight는 Hugging Face에 MIT license로 공개되어 있다. 반면 technical report 수준의 full training data, verifier suite, end-to-end training code, 3B evaluation reproduction package는 repository에서 완전하게 확인하기 어렵다. Weight release와 recipe reproducibility는 분리해 평가해야 한다.

# 7. My Take

## 7-1. 왜 이 논문이 중요한가

이 논문의 가장 중요한 메시지는 small model도 frontier가 될 수 있다는 문장이 아니다. 더 중요한 것은 verifier-rich domain에서 capability ceiling을 결정하는 병목이 parameter count만이 아니라 training signal density일 수 있다는 점이다.

VibeThinker-3B는 다음 loop를 반복한다.

1. SFT로 correct behavior의 support를 넓힌다.
2. Current policy boundary에 있는 prompt를 찾는다.
3. Reliable verifier로 correct signal을 증폭한다.
4. Domain specialist가 발견한 behavior를 unified student에 다시 압축한다.
5. Product-facing constraint를 별도 reward로 복구한다.

이 구조는 math model을 넘어 tool result를 명확히 검증할 수 있는 domain에 재사용할 수 있다. 예를 들어 SQL generation, compiler repair, structured extraction, unit-testable code transformation, theorem proving처럼 outcome verifier가 강한 task가 후보가 된다.

## 7-2. 실무에서 재사용할 수 있는 설계 원칙

### 1) Spectrum before signal

RL을 먼저 늘리기보다 policy가 다양한 valid trajectory를 생성할 수 있는지 확인해야 한다. Pass@1만 보지 말고 Pass@K, solution cluster count, verifier-approved path diversity를 같이 측정할 필요가 있다.

### 2) Train on the capability boundary

너무 쉬운 prompt와 불가능한 prompt를 같은 비율로 sampling하지 않는다. Online rollout에서 success rate가 0.5 부근인 prompt에 budget을 집중하는 것은 label difficulty보다 current policy difficulty를 사용하는 curriculum이다.

### 3) Separate accuracy and efficiency stages

Reasoning을 형성하는 동안 length penalty를 걸지 말고, accuracy가 안정된 뒤 correct trajectory 안에서만 token efficiency를 최적화한다. 이 원칙은 agent step count와 tool call cost에도 적용할 수 있다.

### 4) Treat specialist checkpoints as data generators

Sequential domain RL에서 forgetting이 생기더라도 각 specialist가 만든 verified behavior를 다시 unified model에 distill할 수 있다. Checkpoint를 최종 model candidate로만 보지 않고 high-quality trajectory generator로 사용하는 관점이다.

### 5) Keep model score and system score separate

CLR처럼 test-time search를 붙이면 accuracy는 올라가지만 latency profile이 완전히 달라진다. Single-pass, self-consistency, verifier reranking, claim-level aggregation을 각각 별도 operating point로 보고해야 한다.

### 6) Pair the model with retrieval or tools for coverage

Reasoning core가 강해도 knowledge coverage는 제한된다. Product system에서는 3B model 단독보다 retrieval, calculator, code executor, domain database를 붙이고, tool-use data로 별도 alignment하는 방향이 더 현실적이다.

## 7-3. 다음으로 해볼 실험

### Experiment A: Stage attribution

동일 base model과 동일 data budget에서 SFT only, SFT + standard GRPO, SFT + MGPO, full pipeline을 비교한다. Accuracy뿐 아니라 Pass@K, response length, verifier false positive, instruction score를 함께 본다.

### Experiment B: Cost-normalized evaluation

Model size가 아니라 fixed GPU-second, fixed generated tokens, fixed dollar budget 아래에서 3B model과 larger model을 비교한다. CLR도 같은 budget 안에 포함해야 한다.

### Experiment C: Verifier robustness

Weak test case, ambiguous answer, adversarial formatting을 넣어 reward hacking rate를 측정한다. Final answer verifier와 intermediate claim verifier가 같은 오류를 공유하는지도 확인한다.

### Experiment D: Real software engineering

LiveCodeBench 외에 repository navigation, issue localization, multi-file patch, test repair를 평가한다. Tool calling data를 추가하기 전과 후를 비교하면 competitive programming skill의 transfer boundary를 볼 수 있다.

### Experiment E: Retrieval-augmented small reasoning core

GPQA처럼 knowledge coverage가 필요한 task에서 retrieval을 붙이고, 3B reasoning core가 larger closed-book model과 어느 cost point에서 교차하는지 측정한다.

## 7-4. Follow-up papers

- Tiny Model, Big Logic: Diversity-Driven Optimization Elicits Large-Model Reasoning Ability in VibeThinker-1.5B
- DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
- DAPO: An Open-Source LLM Reinforcement Learning System at Scale
- Group Sequence Policy Optimization

# 8. Summary

- VibeThinker-3B는 Qwen2.5-Coder-3B lineage 위에 multi-stage post-training pipeline을 쌓은 3B dense reasoning model이다.
- SSP는 SFT에서 solution spectrum을 넓히고, MGPO 기반 RL에서 verifier-confirmed signal을 capability boundary 중심으로 강화한다.
- Math, Code, STEM RL을 순차적으로 진행하고 64K context를 처음부터 사용하며, accuracy 이후 Long2Short로 response efficiency를 최적화한다.
- Offline self-distillation은 specialist checkpoint의 verified trajectory를 unified student에 통합하고, Instruct RL은 format과 user constraint를 보강한다.
- AIME26 94.3과 LiveCodeBench v6 80.2는 강하지만, CLR 97.1은 추가 test-time compute 결과이며 open-domain knowledge, agent coding, reproducibility에는 분명한 한계가 있다.
