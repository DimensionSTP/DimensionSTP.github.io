---
layout: single
title: "Self-Improving Pretraining Review"
categories: Study-concept
tag: [LLM, Pretraining, PostTraining, RL, Alignment]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2601.21343)

Self-Improving Pretraining은 LLM 학습 pipeline을 읽는 관점을 꽤 근본적으로 바꾸는 논문이다. 보통 우리는 pretraining을 raw text next-token prediction으로 보고, instruction following, factuality, safety, reasoning 같은 성질은 post-training에서 덧붙이는 것으로 생각한다. 이 논문은 그 분리를 문제로 본다.

"post-training에서 고칠 행동이라면, 왜 pretraining 단계에서는 아무 신호도 주지 않는가"

이 질문이 이 논문의 출발점이다. 저자들은 강한 post-trained model을 teacher로 사용해 pretraining data를 rewrite하고, policy model rollout을 judge하며, safety, factuality, quality, reasoning을 더 이른 training stage에 넣는다. 즉 이 논문은 좋은 dataset filtering 논문이라기보다, **pretraining objective 자체를 behavior shaping objective로 바꾸려는 시도**에 가깝다.

> 한 줄 요약: Self-Improving Pretraining은 강한 post-trained model을 rewriter와 judge로 사용해 raw text next-token prediction 중심의 pretraining을 sequence generation plus RL training으로 바꾸고, 추가로 thinking mid-training을 통해 reasoning trace를 pretraining과 post-training 사이에 주입하는 논문이다.

## 0-1. 왜 지금 읽을 가치가 있는가

- LLM 학습이 pretraining, mid-training, post-training으로 세분화되는 상황에서, 각 stage의 역할을 다시 묻는 논문이다.
- safety, factuality, quality를 post-training 보정 문제가 아니라 pretraining objective 문제로 다룬다.
- rollouts, original suffix, rewritten suffix를 한 candidate pool에 넣고 judge로 고르는 구조가 RLHF와 pretraining 사이의 경계를 흐린다.
- thinking mid-training 파트는 reasoning을 question-answer post-training에서만 학습하는 것이 아니라, pretraining corpus 안에 interleaved thought로 심는 방향을 보여준다.
- 실무적으로는 synthetic data, judge model, rewrite model, online DPO, RF-NLL, DrGRPO를 어떻게 연결할지에 대한 recipe 관점에서 볼 가치가 있다.

## 0-2. 먼저 말하고 싶은 핵심 결론

1. **이 논문의 핵심은 더 좋은 text를 고르는 것이 아니라 training signal의 위치를 바꾸는 것이다.**

2. **original suffix, rewritten suffix, policy rollout을 동시에 비교한다는 점이 중요하다.**

3. **초기에는 rollout을 믿지 않고, 모델이 좋아질수록 rollout을 더 많이 학습 신호로 사용한다.**

4. **thinking mid-training은 CoT를 prompt trick으로 쓰는 것이 아니라 pretraining style corpus에 reasoning trace를 삽입하는 쪽이다.**

5. **결과 수치는 강하지만, teacher judge 의존성과 compute cost를 같이 봐야 한다.**

## 0-3. 이 글의 관점

이 글은 Self-Improving Pretraining을 단순한 alignment 논문이 아니라, training pipeline design 논문으로 읽은 정리다. 그래서 논문의 두 축을 분리해서 본다.

- 첫 번째 축은 safety, factuality, quality를 pretraining 단계에 넣는 sequence pretraining plus RL recipe다.
- 두 번째 축은 reasoning을 mid-training 단계에 넣는 thinking augmentation plus SFT plus RLMT recipe다.

두 축은 모두 같은 메시지를 공유한다.

"강한 post-trained model에서 얻은 판단력과 reasoning trace를 다음 세대 model의 더 이른 학습 단계로 되돌려 보내자"

# 1. Problem Setting

## 1-1. Problem definition

이 논문이 겨냥하는 문제는 classical LLM training pipeline의 stage separation이다. 일반적으로 LLM은 다음 순서로 학습된다.

1. raw text corpus로 pretraining을 수행한다.
2. instruction data로 SFT를 수행한다.
3. preference data, verifier, rule-based reward 등을 사용해 post-training을 수행한다.
4. safety, factuality, helpfulness, reasoning behavior를 뒤늦게 강화한다.

문제는 pretraining에서 이미 model의 core behavior와 representation이 많이 결정된다는 점이다. unsafe text, low-quality text, hallucination-prone continuation, shallow reasoning pattern을 그대로 next-token prediction으로 학습하면, post-training은 이미 몸에 밴 패턴을 뒤늦게 보정하는 단계가 된다.

이 논문은 특히 다음 네 가지 성질을 pretraining 단계에서부터 다루고 싶어 한다.

- overall generation quality
- safety
- factuality
- reasoning ability

여기서 중요한 차이는 data filtering과 Self-Improving Pretraining의 차이다. Filtering은 나쁜 example을 제거한다. 하지만 나쁜 prefix가 들어왔을 때 안전하고 유용한 suffix로 steering하는 능력은 filtering만으로 학습하기 어렵다. 이 논문은 unsafe or low-quality context를 아예 제거하는 대신, 그 context에서 더 나은 continuation을 만들도록 학습시킨다.

## 1-2. Why previous approaches are insufficient

기존 접근의 한계는 크게 네 가지로 정리할 수 있다.

### 1) Raw next-token prediction은 quality preference를 표현하지 못한다

표준 pretraining objective는 주어진 token sequence를 맞추는 데 집중한다. 어떤 suffix가 더 안전한지, 더 factual한지, 더 coherent한지에 대한 구분은 objective에 직접 들어가지 않는다.

수식으로 단순화하면 표준 pretraining은 다음 objective에 가깝다.

$$
\max_\theta \sum_t \log p_\theta(x_t | x_{<t})
$$

이 objective는 corpus의 token distribution을 잘 맞추도록 만든다. 하지만 corpus 안에 low-quality continuation이 있으면, 그것도 맞춰야 할 target이 된다.

### 2) Dataset filtering은 steering을 가르치지 못한다

Unsafe data를 제거하면 model이 unsafe pattern을 덜 보게 할 수는 있다. 하지만 실제 deployment에서는 unsafe or low-quality input이 들어올 수 있다. 그런 input에서 안전하고 유용한 response로 이동하는 능력은, unsafe prefix를 보존한 상태에서 safe suffix를 학습해야 생긴다.

"나쁜 데이터를 안 보는 것"과 "나쁜 context에서 좋은 방향으로 빠져나오는 것"은 다른 문제다.

### 3) Post-training은 이미 형성된 base behavior를 완전히 지우기 어렵다

Post-training은 강력하지만, base model이 pretraining에서 형성한 distributional habit 위에서 작동한다. Safety나 factuality를 나중에 reward로 주입하더라도, pretraining 단계에서 생긴 hallucination tendency나 unsafe completion tendency가 남을 수 있다.

### 4) Reasoning은 post-training question-answer format에 갇히기 쉽다

RLVR이나 math post-training은 특정 QA format에서 reasoning을 강화한다. 하지만 pretraining corpus 자체에는 많은 implicit reasoning opportunity가 있다. 수학적 전개, 설명문, 원인과 결과, narrative logic, factual passage 안의 inferential relation이 모두 reasoning trace를 넣을 수 있는 지점이다.

Self-Improving Pretraining은 이 gap을 줄이기 위해 pretraining과 post-training 사이에 thinking mid-training을 둔다.

# 2. Core Idea

## 2-1. Main contribution

논문의 핵심 기여는 두 개의 recipe로 나눌 수 있다.

### 1) Self-Improving Pretraining for safety, factuality, and quality

기존 pretraining data stream을 prefix와 suffix로 나눈다. 그리고 suffix를 그대로 맞추는 대신, 강한 post-trained model을 사용해 다음 candidate들을 비교한다.

- original suffix
- rewritten suffix
- current policy rollout

Teacher model은 rewriter로도 쓰이고 judge로도 쓰인다. Rewriter는 suffix를 더 안전하거나 더 높은 품질의 continuation으로 바꾼다. Judge는 candidate completion들을 scoring하고, policy model은 그 reward를 사용해 online DPO 또는 RF-NLL로 학습된다.

**즉 target은 더 이상 corpus suffix 하나가 아니다.**

### 2) Thinking Mid-training for reasoning

논문의 두 번째 축은 reasoning을 pretraining과 post-training 사이에 넣는 것이다. Teacher model이 raw pretraining chunk에 interleaved thought를 삽입한다. 그 다음 student는 augmented corpus로 SFT mid-training을 하고, 이후 RL mid-training에서 generated thought가 subsequent text prediction에 실제로 도움이 되는지 judge reward를 받는다.

이 흐름은 다음처럼 볼 수 있다.

1. raw chunk를 teacher가 thought-augmented chunk로 바꾼다.
2. student가 augmented chunk를 next-token prediction으로 학습한다.
3. student가 prefix에서 thought plus suffix를 생성한다.
4. judge가 generated suffix와 ground truth suffix를 비교해 reward를 준다.
5. DrGRPO로 thought가 prediction에 도움이 되도록 강화한다.

## 2-2. Design intuition

이 논문의 설계 직관은 단순하지만 강하다.

"pretraining은 model이 deploy될 때 할 일을 연습해야 한다"

Deployment에서 model은 다음 token 하나를 맞추는 것이 아니라, prefix를 보고 quality, safety, factuality, reasoning이 있는 sequence를 생성해야 한다. 그렇다면 pretraining도 token-level imitation만이 아니라 sequence-level generation과 preference judgment를 포함해야 한다는 논리다.

또 하나의 핵심은 teacher의 위치다. 보통 teacher model은 distillation에서 output을 제공하거나, post-training에서 reward model처럼 쓰인다. 이 논문에서는 teacher가 pretraining data stream 안으로 들어간다.

- rewriter는 raw suffix를 더 좋은 training target으로 바꾼다.
- judge는 suffix, rewrite, rollout 중 무엇을 학습할지 고른다.
- annotator는 corpus 안에 thought를 삽입한다.
- judge는 generated thought가 suffix prediction에 도움이 되는지 평가한다.

이 논문의 핵심은 self-training이라기보다 **stage feedback loop**다. 이미 post-trained model이 가진 instruction following, safety, factuality, reasoning 능력을 다음 model의 earlier training stage로 되돌려 보내는 구조이기 때문이다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | pretraining 단계에서 safety, factuality, quality, reasoning을 직접 학습 |
| Main teacher | strong post-trained model |
| Candidate set | original suffix, rewritten suffix, policy rollout |
| Reward source | suffix judge, quality judge, safety judge, factuality judge |
| Policy update | online DPO, RF-NLL, DrGRPO |
| First recipe | Self-Improving Pretraining for quality, safety, factuality |
| Second recipe | Thinking Mid-training for reasoning |
| Key difference | next-token target 하나가 아니라 teacher-judged sequence candidate를 사용 |

## 3-2. Module breakdown

### 1) Prefix-conditioned suffix generation

논문은 pretraining stream을 chunk로 자르고, 현재 chunk를 suffix, 앞선 context를 prefix로 본다. Policy model은 prefix를 조건으로 길이 $K$의 suffix를 생성한다.

$$
y \sim p_\theta(\cdot | x_{prefix})
$$

표준 next-token prediction과 달리, 여기서는 generated suffix가 original suffix와 정확히 같을 필요가 없다. 오히려 original suffix가 unsafe or low-quality라면 그대로 맞추는 것이 바람직하지 않을 수 있다.

그래서 이 논문은 suffix generation을 다음처럼 본다.

- high-quality original suffix는 mimic할 수 있다.
- low-quality original suffix는 rewrite로 대체할 수 있다.
- policy rollout이 충분히 좋아지면 rollout 자체를 학습 신호로 사용할 수 있다.

### 2) Suffix Rewriter

Suffix rewriter는 prefix와 suffix를 보고 더 나은 suffix를 생성한다. 역할은 task에 따라 다르다.

- quality setting에서는 low-quality suffix를 더 coherent and useful continuation으로 바꾼다.
- safety setting에서는 unsafe prefix를 유지한 채, suffix만 안전한 방향으로 바꾼다.
- augmentation setting에서는 suffix를 다양한 방식으로 다시 써서 training signal의 폭을 넓힌다.

여기서 가장 중요한 점은 unsafe prefix를 지우지 않는다는 것이다. Prefix는 그대로 두고 suffix를 안전하게 바꾸면, model은 unsafe input에 노출되면서도 safe continuation으로 steering하는 방법을 학습한다.

### 3) Suffix Judge

Suffix judge는 candidate completion을 평가한다. Candidate set은 다음과 같다.

- original suffix
- rewritten suffix
- one or more rollouts from current policy

Judge는 quality, safety, factuality를 따로 평가할 수 있다. Quality는 pairwise comparison으로 더 coherent and high-quality continuation을 고르고, safety는 pointwise safe or unsafe decision을 내린다. Factuality는 original suffix를 reference로 삼아 hallucination 여부를 평가한다.

Policy training에서는 이 judge score를 reward로 사용한다.

### 4) Online DPO and RF-NLL

논문은 policy update로 online DPO와 RF-NLL을 사용한다. Online DPO에서는 highest scoring candidate를 chosen, lowest scoring candidate를 rejected로 둔다. RF-NLL은 highest scoring candidate에 대해 NLL update를 수행한다.

Online DPO가 이 setting에 적합한 이유는 off-policy candidate를 다룰 수 있기 때문이다. Original suffix나 rewritten suffix는 현재 policy가 생성한 sequence가 아니다. GRPO처럼 on-policy rollout 중심인 방법보다, suffix와 rewrite를 한 candidate pool에 넣는 데 DPO가 더 자연스럽다.

이 구조를 간단히 쓰면 다음과 같다.

$$
y^+ = \arg\max_{y \in C} r(x, y)
$$

$$
y^- = \arg\min_{y \in C} r(x, y)
$$

여기서 $C$는 original suffix, rewritten suffix, policy rollouts로 구성된 candidate set이다.

### 5) Training dynamics

초기 policy rollout은 품질이 낮다. 따라서 training 초반에는 original suffix와 rewritten suffix가 주된 supervision이 된다. 그러나 model이 개선되면 judge가 policy rollout을 점점 더 자주 선택한다.

"처음에는 teacher가 고쳐 준 target을 따라가고, 나중에는 policy 자신의 좋은 rollout이 학습 target이 된다"

이 dynamic이 self-improving이라는 이름의 실제 의미에 가깝다. 단순히 model이 자기 output을 계속 학습하는 것이 아니라, teacher judge가 좋은 rollout을 걸러내는 조건에서 자기 output이 training signal로 승격된다.

### 6) Thinking data augmentation

Thinking mid-training에서는 raw pretraining chunk에 interleaved thought를 넣는다. Teacher model은 chunk 안에서 semantic position을 고르고, 그 위치에 intermediate reasoning을 삽입한다.

예를 들어 raw corpus가 다음과 같다고 하자.

$$
x = [s_1, s_2, s_3]
$$

Teacher는 이를 다음처럼 바꾼다.

$$
x' = [s_1, t_1, s_2, t_2, s_3]
$$

여기서 $t_i$는 reasoning thought다. 중요한 점은 thought가 question 끝에만 붙는 것이 아니라, corpus 내부에 interleaved된다는 점이다.

### 7) Thinking SFT and Thinking RL Mid-training

Thinking SFT는 augmented sequence 전체를 next-token prediction으로 학습한다. 이때 original content token과 thought token이 모두 loss에 들어간다. 목적은 model이 interleaved thought format 자체를 배워 cold start를 얻는 것이다.

Thinking RL Mid-training은 한 단계 더 나아간다. Model은 prefix에서 thought와 suffix를 생성하고, judge는 generated suffix가 ground truth suffix를 충분히 잘 예측했는지 평가한다. 이 reward는 thought가 실제 prediction에 도움이 되는지 보도록 설계된다.

즉 SFT는 thought를 모방하게 만들고, RLMT는 thought가 쓸모 있게 만들도록 압력을 준다.

# 4. Training / Data / Recipe

## 4-1. Data

논문은 두 recipe에서 서로 다른 data setting을 사용한다.

### 1) Self-Improving Pretraining side

- Policy model baseline은 주로 Llama2 1.4B pretrained model이다.
- Continued pretraining과 from-scratch pretraining을 모두 실험한다.
- Quality and factuality 실험에는 SlimPajama를 사용한다.
- Safety 실험에는 RedPajama의 unsafe content를 filtering해 사용한다.
- Policy, judge, rewriter는 서로 겹치지 않는 data subset으로 학습 및 평가된다.

구체적으로 policy training에는 SlimPajama 983,520 samples, RedPajama unsafe filtered 257,154 samples가 사용된다. Judge training에서는 quality task에 75,432 training samples와 4,096 validation samples를 만들고, safety task에 3,192 training samples와 512 validation samples를 만든다. Safety rewriter training에는 73,080 safe and unsafe suffix samples가 사용된다.

### 2) Thinking Mid-training side

Thinking mid-training에서는 DCLM과 FineMath 같은 general reasoning corpus를 사용한다. GPT-OSS-120B가 annotator로 사용되어 raw data에 interleaved thoughts를 넣고, RL training 쪽에서는 thought position만 teacher가 주고 thought token과 continuation은 policy model이 생성한다.

Post-training에는 DAPO-Math-14k가 사용되며, answer verification에는 Math-Verify가 사용된다. Main experiment는 Llama-3-8B 계열에서 진행되고, Qwen3-8B에서도 ablation을 수행한다.

## 4-2. Training strategy

### 1) Judge and rewriter training

Quality and safety judge는 Llama3.1-8B-Instruct에서 fine-tune한 model과 GPT-OSS-120B prompting을 함께 사용한다. Quality task에서는 Llama3.3-70B-Instruct로 original suffix를 corrupt하여 positive and negative pair를 만든다. Safety task에서는 RedPajama suffix를 safe and unsafe로 filtering하고, judge prompt로 감싸 training data를 만든다.

Judge and rewriter training은 GRPO로 수행된다. 논문은 global batch size 256, 16 generations per prompt, 64 GPUs, 500 steps setting을 사용했다고 적는다. Prompt length와 generation length는 judge와 rewriter에서 다르게 둔다. Rewriter는 128 new tokens를 생성하도록 맞춘다.

### 2) Policy training for Self-Improving Pretraining

Continual pretraining에서는 online DPO를 기본으로 사용한다. Global batch size는 256이고, prompt마다 16 rollouts를 sampling한다. 64 GPUs에서 2000 steps를 돌리고, maximum sequence length는 2048 tokens다. Safety task에서는 fine-tuned Llama3.1-8B-Instruct judge를 사용하고, quality and factuality task에서는 GPT-OSS-120B judge를 사용한다.

From-scratch pretraining에서는 training steps를 21,000으로 늘리고, 1 rollout만 사용한다.

### 3) Thinking mid-training

Thinking SFT는 augmented corpus의 일부에서 수행된다. 이후 RLMT에서는 model이 generated thoughts plus predicted suffix를 만들고, GPT-OSS-120B judge가 generated suffix와 original continuation을 비교한다. Optimization은 DrGRPO로 수행된다. Training framework는 fairseq2다.

## 4-3. Engineering notes

이 논문을 재현하거나 응용하려면 다음 지점이 중요하다.

1. **Judge prompt 품질이 성능을 크게 좌우한다.**

Appendix에서는 medium-sized post-trained model들이 quality judge task에서 충분히 강하지 않을 수 있음을 보여준다. 특히 judge가 context coherence보다 completion 느낌을 선호하는 문제가 있었다고 설명한다.

2. **Candidate pool 설계가 중요하다.**

Original suffix, rewrite, rollout을 어떤 조합으로 넣는지에 따라 결과가 크게 바뀐다. Rollout만 SFT하는 방식은 collapse를 만들 수 있다.

3. **Rollout 수는 성능과 compute cost를 동시에 바꾼다.**

1 rollout보다 16 rollouts가 더 강한 결과를 보이지만, pairwise comparison cost도 커진다. Pivot comparison으로 비용을 줄이는 실험도 했지만 성능 deterioration이 보고된다.

4. **Objective를 섞는다고 자동으로 multi-objective generalization이 생기지는 않는다.**

논문은 safety를 optimize한다고 factuality가 자동으로 좋아지는 것이 아니며, 반대도 마찬가지라고 설명한다. 따라서 reward design에서 어떤 behavior를 넣을지 명시해야 한다.

5. **Thinking mid-training은 format learning과 utility optimization을 분리한다.**

SFT는 thought format을 배우게 하고, RLMT는 thought가 prediction에 실제로 도움이 되도록 만든다. 이 분리는 reasoning data augmentation에서 꽤 중요하다.

# 5. Evaluation

## 5-1. Main results

### 1) Continued pretraining for quality, factuality, and safety

논문 Table 1은 continued pretraining setting에서 Self-Improving Pretraining이 standard pretraining baseline보다 크게 개선된다고 보고한다.

| Setting | Baseline | Self-Improving Pretraining | Main reading |
| --- | --- | --- | --- |
| Quality generation win rate | 49.0 | 86.3 | quality judge로 sequence-level preference를 넣을 때 큰 차이 |
| Quality standard eval avg | 46.8 | 50.8 | 표준 eval도 일부 회복 또는 개선 |
| Coherence eval | 49.4 | 87.9 | generation quality 축에서 가장 선명한 개선 |
| Factuality eval avg | 44.0 | 57.6 | factuality reward가 hallucination 관련 지표에 직접 반영 |
| Safety eval avg | 75.5 | 91.1 | RedPajama unsafe setting에서도 safety score 개선 |

여기서 가장 흥미로운 수치는 quality setting의 generation quality 86.3이다. 단순 next-token continued pretraining baseline이 49.0인 것과 비교하면, judge-based sequence objective가 output quality를 훨씬 직접적으로 건드린다는 것을 보여준다.

Factuality setting에서는 FActScore와 HaluEval, TruthfulQA 계열 지표에서 개선이 보고된다. Safety setting에서는 RealToxicityPrompts, RedPajama test, XSTest, Toxigen 같은 지표에서 개선이 보고된다.

### 2) From-scratch safety pretraining

From-scratch setting에서도 효과가 있다. RedPajama에서 21k steps로 학습한 1.4B model 기준, baseline generation quality win rate는 1.3이고 safety eval average는 85.2다. Rewrite만으로 pretraining하면 safety는 96.7까지 올라가지만 generation quality는 1.6에 머문다.

반면 Self-Improving Pretraining RF-NLL with rollout vs rewrite는 generation quality 32.4, safety eval 97.5를 보고한다. 즉 rewrite target 자체가 safety에는 도움을 주지만, model rollout을 judge로 걸러 학습시키는 구조가 quality 측면에서 더 큰 차이를 만든다.

### 3) Objective ablation

Table 6은 꽤 중요하다. 단순히 rewrite로 SFT하면 safety는 86.5로 올라가지만 generation quality는 52.7 정도다. 1 rollout으로 SFT하면 safety eval은 99.5로 높아 보이지만, generation quality가 2.0과 0.2로 붕괴한다. 논문은 이 model이 meaningless but safe sequence를 생성하는 collapse를 보였다고 설명한다.

따라서 중요한 결론은 다음이다.

**Rollout을 쓰는 것이 중요한 것이 아니라, judge가 좋은 rollout을 고르는 구조가 중요하다.**

Online DPO에서 suffix vs 16 rollouts를 사용하면 generation quality 73.6, unsafe prefix 77.7, safety eval 91.1을 얻는다. 또한 rollouts 수를 늘릴수록 quality, factuality, safety의 결과가 좋아지는 경향이 보고된다.

### 4) Thinking mid-training results

Thinking mid-training 결과도 큰 메시지를 준다. Llama3-8B-Base에서 base average score는 0.0475다. Raw data SFT 10k steps는 0.0999까지 올리지만, SFT on thinking-augmented data는 0.2221까지 올라간다. 여기에 RLMT를 붙이면 0.3390까지 올라간다.

Post-training까지 본 Table 9에서는 차이가 더 분명하다. Base model에 바로 RL post-training을 하면 average 0.1197이다. 반면 SFT(think) 10k plus RLMT 5k plus RLPT는 0.3837이다. 논문은 이를 3.2 improvement로 해석한다.

Qwen3-8B ablation은 더 조심스럽게 읽어야 한다. Base average가 0.3572로 이미 높고, SFT(raw)와 SFT(think)는 오히려 성능을 낮춘다. 그러나 SFT(think) plus RLMT 1k는 0.3660으로 base를 약간 넘는다. 이는 stronger base model에서는 SFT mid-training이 단순히 항상 좋다고 말하기 어렵고, RLMT가 더 중요한 조정 역할을 할 수 있음을 보여준다.

## 5-2. What really matters in the experiments

이 논문의 실험에서 진짜 봐야 하는 것은 absolute benchmark score만이 아니다. 더 중요한 것은 training signal이 어느 stage에 들어갔을 때 어떤 behavior가 변하는가다.

### 1) Quality, safety, factuality는 서로 자동 전이되지 않는다

논문은 safety를 optimize한다고 factuality가 자동으로 좋아지지 않고, factuality를 optimize한다고 safety가 자동으로 해결되지 않는다고 말한다. 이는 reward design에서 behavior별 objective를 명시해야 한다는 뜻이다.

### 2) Judge 없는 rollout 학습은 위험하다

Self-training류 방법의 큰 위험은 model이 자기 output을 무비판적으로 다시 학습하면서 collapse하는 것이다. 이 논문의 ablation은 이 위험을 직접 보여준다. 좋은 rollout을 고르는 judge가 없으면 rollout 자체는 학습 신호가 아니라 noise source가 될 수 있다.

### 3) More rollouts is not just exploration, it is candidate quality control

Rollout 수를 늘리면 policy가 더 다양한 후보를 만들고, judge가 그중 더 좋은 후보를 고를 수 있다. 이 구조에서는 rollout 수가 RL exploration budget이면서 동시에 candidate selection quality를 높이는 장치다.

### 4) Thinking trace는 imitation만으로는 부족하다

SFT on thinking data만으로도 큰 개선이 있지만, RLMT가 추가 개선을 만든다. 이는 thought가 그럴듯한 문장으로 보이는 것과, 실제로 suffix prediction에 도움이 되는 것이 다르다는 점을 보여준다.

# 6. Limitations

1. Teacher model dependence
   - 이 방법은 강한 post-trained model이 이미 존재한다는 가정에 의존한다.
   - Teacher가 rewriter, judge, annotator 역할을 모두 수행하므로 teacher bias와 teacher failure가 training signal에 들어갈 수 있다.

2. Judge reliability
   - Quality judge는 context coherence와 completion completeness를 혼동할 수 있다.
   - 논문도 judge training 전에 model들이 suffix 품질 판단에서 충분하지 않은 결과를 보였다고 설명한다.

3. Compute and serving cost
   - Candidate rollouts, pairwise comparisons, teacher judging, rewriter generation은 모두 비용이 크다.
   - 16 rollouts와 all-pair comparison은 research setting에서는 가능하지만, large-scale pretraining에 바로 넣기에는 cost modeling이 필요하다.

4. Scale limitation
   - Safety, factuality, quality 실험은 주로 1.4B policy model 기준이다.
   - Thinking mid-training은 8B class model에서 강하게 보이지만, 100B scale 이상에서도 같은 dynamics가 유지되는지는 추가 확인이 필요하다.

5. Objective interference
   - Safety, factuality, quality는 같은 방향으로만 움직이지 않는다.
   - Multi-objective reward를 잘못 섞으면 한 behavior가 다른 behavior를 희생할 수 있다.

6. Evaluation circularity risk
   - Strong post-trained model이 teacher and judge로 쓰이고, 평가에서도 LLM judge나 judge-based metric이 포함된다.
   - Human evaluation, independent evaluator, downstream deployment metric으로 교차 검증해야 한다.

7. Data wall claim은 추가 검증이 필요하다
   - 논문은 compute를 활용해 data wall에 덜 영향을 받는 방향을 제시하지만, 실제 large-scale data mixture에서 얼마나 안정적인지는 아직 열린 문제다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문의 가장 큰 의미는 pretraining을 더 이상 raw text compression만으로 보지 않는다는 점이다. 최근 LLM training은 post-training recipe 경쟁처럼 보이지만, base model이 가진 behavior distribution이 post-training의 ceiling을 결정하는 경우가 많다.

Self-Improving Pretraining은 이 ceiling을 올리는 방법을 제안한다. 안전한 답변, factual한 continuation, coherent sequence, useful reasoning trace를 post-training에서만 기대하지 않고, pretraining과 mid-training 단계에서부터 rewardable behavior로 만든다.

특히 실무 관점에서는 다음 아이디어가 중요하다.

- dataset filtering만으로는 steering behavior를 학습하기 어렵다.
- bad context를 남겨두고 good continuation을 학습하는 방식이 필요하다.
- rollout self-training은 judge 없이 쓰면 collapse할 수 있다.
- reasoning trace는 final QA format이 아니라 raw corpus 안에도 삽입할 수 있다.

## 7-2. Reuse potential

이 논문에서 바로 재사용할 만한 포인트는 다섯 가지다.

### 1) Prefix-suffix rewrite pipeline

Unsafe, low-quality, hallucination-prone suffix를 단순 제거하지 않고, prefix는 유지한 채 suffix만 개선하는 data pipeline으로 응용할 수 있다.

### 2) Candidate pool based training

Original answer, rewritten answer, model rollout을 한 pool에 넣고 judge로 선택하는 구조는 instruction tuning, tool-use trajectory, code generation에도 옮겨갈 수 있다.

### 3) Judge-before-self-training

Self-training에서 가장 중요한 것은 model output을 다시 먹이는 것이 아니라, 어떤 output을 먹일지 고르는 judge다. 이 관점은 synthetic data generation pipeline에도 그대로 적용된다.

### 4) Thinking augmentation for non-QA corpus

Reasoning data를 question-answer pair로만 만들 필요는 없다. Long document, code explanation, math derivation, scientific passage, policy document 안에 interleaved thought를 넣는 방식도 가능하다.

### 5) Stage-specific reward design

Pretraining, mid-training, post-training에서 같은 reward를 쓰기보다 stage별로 reward target을 다르게 두는 것이 더 자연스럽다. Pretraining에서는 suffix quality, mid-training에서는 thought utility, post-training에서는 answer correctness가 될 수 있다.

## 7-3. Follow-up papers

- Quiet-STaR: arbitrary text에서 rationale을 생성해 reasoning을 강화하는 방향.
- RLPT: pretraining data에서 reward를 추출해 RL을 수행하는 방향.
- DAPO: math RL post-training recipe와 long-horizon RLVR 안정화.
- Self-Instruct: model generated data를 instruction tuning에 사용하는 초기 self-improvement 방향.
- Constitutional AI: safety behavior를 preference and critique based training으로 넣는 방향.

# 8. Summary

- Self-Improving Pretraining은 safety, factuality, quality를 post-training에서만 고치지 말고 pretraining 단계에서부터 sequence-level objective로 학습하자는 논문이다.
- 핵심 구조는 prefix-suffix stream에서 original suffix, rewritten suffix, policy rollout을 candidate로 두고, strong post-trained model judge가 학습 target을 고르는 방식이다.
- Online DPO가 중요한 이유는 original suffix와 rewrite처럼 off-policy sequence를 candidate pool에 넣을 수 있기 때문이다.
- Thinking mid-training은 raw pretraining corpus에 interleaved thought를 넣고, SFT와 RLMT를 통해 reasoning trace를 더 이른 stage에 주입한다.
- 결과는 인상적이지만, teacher judge 의존성, rollout compute cost, objective interference, scale generalization을 같이 봐야 한다.
