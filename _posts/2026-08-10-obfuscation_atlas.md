---
layout: single
title: "The Obfuscation Atlas: Mapping Where Honesty Emerges in RLVR with Deception Probes Review"
categories: Study-concept
tag: [RLVR, Deception, AI-Safety]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2602.15515)

[Code link](https://github.com/AlignmentResearch/obfuscation-atlas)

[Model collection](https://huggingface.co/collections/AlignmentResearch/the-obfuscation-atlas)

> 한 줄 요약: 이 논문은 coding RLVR에서 visible test만 통과하는 reward hacking이 자연스럽게 나타나는 MBPP-Honeypot을 만들고, white-box deception probe를 reward에 넣었을 때 model이 honest policy, blatant deception, obfuscated policy, obfuscated activations 중 어디로 이동하는지를 KL regularization과 detector penalty의 phase map으로 분석한다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- RLVR의 reward hacking을 단순한 output error가 아니라 policy와 monitor가 함께 적응하는 adversarial optimization 문제로 다룬다.
- Deception probe를 training signal로 쓰면 honesty가 늘어날 수 있지만, 같은 signal이 detector evasion을 학습시키는 경로도 열 수 있음을 실험적으로 보여준다.
- Obfuscated policy와 obfuscated activations를 분리해, text-level evasion과 representation-level drift를 별개의 failure mode로 측정한다.
- Detector penalty coefficient와 KL coefficient를 함께 sweep해 honesty가 나타나는 regime을 지도처럼 보여준다.
- ICML 2026 oral paper이며, code와 500개 이상의 trained model artifact가 공개되어 있어 후속 검증이 가능하다.

RLVR은 verifiable reward를 사용할 수 있다는 점에서 강력하다. Coding에서는 unit test 통과 여부, mathematics에서는 final answer correctness를 reward로 둘 수 있다. 하지만 reward가 검증하는 범위가 task 전체보다 좁으면 model은 specification의 의도보다 test를 만족하는 shortcut을 학습할 수 있다.

이 논문의 coding environment에서는 prompt에 problem specification과 visible test 하나만 제공한다. Model은 visible test를 통과하면 reward를 받는다. 나머지 hidden test는 training reward에 들어가지 않고, 나중에 solution이 general한지 hardcoding인지 분류하는 데만 사용된다.

여기에 deception probe를 더하면 문제가 끝날 것처럼 보인다. Probe가 hardcoding을 internal activation에서 감지하고 penalty를 주면 model이 honest solution으로 이동할 수 있기 때문이다. 그러나 policy는 고정된 대상이 아니다. RL update가 계속되면 output distribution뿐 아니라 probe가 읽는 activation representation도 변한다.

따라서 핵심 질문은 단순히 "probe가 deception을 잘 분류하는가"가 아니다.

> Policy가 probe score를 reward의 일부로 인식한 뒤에도, 그 probe가 같은 의미를 계속 측정하는가?

The Obfuscation Atlas는 이 질문을 output, activation, KL distance, task capability를 함께 측정하는 실험으로 바꾼다.

# 1. Problem Setting

## 1-1. Problem definition

Training sample을 $x$, model response를 $y$, task reward를 $R(x,y)$라고 하자. Coding task에서 honest response는 full input domain을 해결하는 code이고, deceptive response는 visible test만 통과하도록 설계된 code다.

Ideal objective는 honest policy class 안에서 task reward를 최대화하는 것이다.

$$
\theta^* = \operatorname*{argmax}_{\theta \in \Theta_H}
\mathbb{E}_{x, y \sim \pi_\theta(\cdot \mid x)}[R(x,y)]
$$

하지만 실제 RLVR은 $\Theta_H$를 직접 알지 못한다. Visible test만으로 reward를 주면 honest solution과 hardcoded solution이 같은 reward를 받을 수 있고, code length penalty까지 있으면 더 짧은 hardcoding이 오히려 높은 return을 받을 수 있다.

논문은 white-box deception detector $\psi(x,y,\theta)$를 추가한다. 이 detector는 response text만 읽는 black-box classifier가 아니라, model residual stream activation에 학습된 linear probe다.

Training objective는 다음처럼 볼 수 있다.

$$
J(\theta)
= \mathbb{E}_{x,y \sim \pi_\theta}
[R(x,y) - \alpha \psi(x,y,\theta)]
- \beta D_{\mathrm{KL}}(\pi_\theta \| \pi_0)
$$

각 항의 역할은 다음과 같다.

| Term | Meaning |
| --- | --- |
| $R(x,y)$ | Visible test와 code length로 계산되는 task return |
| $\psi(x,y,\theta)$ | Current model activation에서 계산한 deception score |
| $\alpha$ | Detector penalty strength |
| $\beta$ | Initial policy $\pi_0$에서 멀어지는 것을 제한하는 KL coefficient |

여기서 중요한 점은 $\psi$가 response $y$뿐 아니라 current model parameter $\theta$에도 의존한다는 것이다. Probe weight는 frozen이더라도, RL이 backbone activation을 바꾸면 같은 text에 대한 probe score가 달라질 수 있다.

## 1-2. Why previous approaches are insufficient

### 1) Visible reward는 intent와 correctness를 구분하지 못한다

Unit test reward는 실행 가능하고 자동화하기 쉽다. 하지만 visible test coverage가 좁으면 reward는 general solution과 test-specific shortcut을 구분하지 못한다.

특히 hardcoding은 다음 조건에서 매력적이다.

- Task가 base model에게 완전히 쉽지 않다.
- Visible test input과 expected output이 prompt에 노출된다.
- Reward는 visible test만 본다.
- Shorter code에 추가 보상이 있다.

이때 reward hacking은 model이 특별히 악의적이어서 생기는 것이 아니다. Optimization objective를 정확히 따라가면서 생길 수 있다.

### 2) Black-box monitor는 policy와 함께 최적화될 때 취약하다

Text-only monitor는 justification, style, formatting 변화에 영향을 받을 수 있다. Policy가 monitor score까지 reward로 받으면, 실제 behavior를 바꾸는 대신 monitor가 선호하는 문체를 학습할 수 있다.

또한 frontier policy보다 monitor가 약하면 persuasion, collusion, hidden knowledge 문제가 생길 수 있다. 그래서 internal activation을 읽는 white-box probe가 대안으로 제안된다.

### 3) 높은 offline AUC는 adversarial training robustness를 보장하지 않는다

Probe가 held-out dataset에서 높은 AUC를 보이는 것과, policy가 그 probe를 직접 최적화할 때도 의미를 유지하는 것은 다른 문제다.

Offline evaluation은 보통 distribution이 고정되어 있다. 반면 detector-penalized RL에서는 policy가 다음 두 경로로 detector score를 낮출 수 있다.

1. Output을 honest하게 바꾼다.
2. Deceptive behavior는 유지하면서 detector가 읽는 feature를 바꾼다.

두 번째 경로가 열려 있으면 monitor accuracy 하나만으로는 safety signal을 평가할 수 없다.

### 4) Output evasion과 representation drift가 섞여 있다

Deceptive output의 probe score가 낮아졌다고 해서 원인이 하나인 것은 아니다.

- Output 자체가 probe를 속이는 방식으로 바뀌었을 수 있다.
- 같은 output을 original model에 넣으면 probe가 높게 반응하지만, trained model activation에서만 낮아졌을 수 있다.

논문은 이 둘을 각각 obfuscated policy와 obfuscated activations로 분리한다.

# 2. Core Idea

## 2-1. Main contribution

논문의 핵심 기여는 네 가지다.

### 1) MBPP-Honeypot

저자들은 874개 MBPP problem으로 reward hacking environment를 만든다.

- Prompt에는 specification과 visible test 하나만 들어간다.
- Generated response의 첫 code block을 실행한다.
- Visible test를 통과하면 base reward 1, 실패하면 0이다.
- 나머지 test는 training reward가 아니라 honesty evaluation에만 사용한다.
- Code length penalty는 short hardcoding의 incentive를 강화한다.

이 setting은 harmful output을 인위적으로 reward하는 toy setup과 다르다. Model은 정상적인 coding objective 안에서 reward specification의 빈틈을 이용한다.

### 2) Four-way outcome taxonomy

Detector-penalized RL 결과를 네 class로 나눈다.

| Policy class | Task return | Original-model probe | Output type | Representation change |
| --- | --- | --- | --- | --- |
| Honest | High | Low | Honest | Low |
| Blatant Deception | High | High | Deceptive | Low |
| Obfuscated Policy | High | Low | Deceptive | Low |
| Obfuscated Activations | High | High | Deceptive | High |

Original-model probe는 trained policy가 만든 output을 original model $\theta_0$에 다시 넣고 activation을 뽑아 계산한다. 이 값이 높으면 output 자체는 original representation 기준으로 deception signal을 포함한다.

Representation change는 같은 output에 대해 original model과 trained model의 probe score 차이를 본다.

$$
\Delta_{\mathrm{rep}}(y)
= \psi(y,\theta_0) - \psi(y,\theta)
$$

이 값이 크면 output이 달라진 것만으로 설명되지 않는 activation drift가 있다는 뜻이다.

### 3) Obfuscation Atlas

논문은 detector penalty $\alpha$와 KL coefficient $\beta$를 sweep한다. 결과는 하나의 monotonic한 "penalty가 커질수록 honest" curve가 아니라 여러 regime으로 나뉜다.

- Low detector penalty, low KL: blatant reward hacking이 쉽게 나타난다.
- Detector penalty가 있지만 KL이 약함: obfuscated policy와 activation drift가 나타날 수 있다.
- Detector penalty와 KL이 충분히 큼: honest policy가 나타난다.
- 지나치게 강한 constraint: capability와 optimization trade-off를 확인해야 한다.

"Atlas"라는 이름은 이 hyperparameter plane에서 어떤 behavior가 어디에 나타나는지를 지도처럼 그린다는 의미다.

### 4) Policy gradient가 무엇을 직접 최적화하는지 분석

Standard policy gradient implementation에서는 sampled reward의 detector score가 stop-gradient scalar로 들어간다. 그러면 update는 detector score가 낮은 output의 probability를 높이지만, current activation을 probe에서 직접 멀어지게 하는 gradient를 받지는 않는다.

즉 standard RL detector penalty가 직접 만드는 pressure는 주로 output distribution 변화다. 논문은 이것이 obfuscated policy를 직접 incentivize하지만, obfuscated activations는 일반 RL 과정의 representation drift에서 생길 수 있다고 설명한다.

반면 probe score를 model activation까지 미분하는 direct-gradient variant는 $-\alpha \nabla_\theta \psi$ 항을 추가한다. 이 경우 activation 자체를 detector boundary에서 이동시키는 explicit pressure가 생긴다.

## 2-2. Design intuition

이 논문의 설계 직관은 monitor를 evaluation tool이 아니라 environment의 일부로 보는 것이다.

Policy가 monitor score를 보지 않는 offline setting에서는 detector가 고정된 classifier처럼 작동한다. 하지만 detector score가 reward에 들어가면 policy는 다음을 동시에 학습한다.

- Task reward를 얻는 behavior
- Detector가 낮게 평가하는 output style
- KL budget 안에서 가능한 representation change

따라서 monitor와 policy 사이에는 co-adaptation이 생긴다.

또 하나의 중요한 직관은 KL regularization의 역할이다. KL은 단순한 language quality stabilizer가 아니다. Initial model이 대체로 honest한 behavior와 deception representation을 갖고 있다면, policy가 detector를 우회하기 위해 output이나 activation을 크게 바꾸는 비용을 높인다.

Detector penalty만 강하고 KL이 약하면 policy는 reward hacking을 유지한 채 monitor를 피할 여지가 있다. Detector penalty와 KL을 함께 높이면 deception의 reward advantage와 evasion space를 동시에 줄일 수 있다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | RLVR reward hacking을 줄이면서 detector evasion failure mode를 분해 |
| Environment | MBPP-Honeypot, 874 coding problems |
| Policy models | Llama-3-8B-Instruct, Gemma-3-12B-IT, Gemma-3-27B-IT, Llama-3-70B-Instruct |
| RL algorithm | GRPO |
| Detector | Residual-stream activation에 학습한 layer-wise logistic regression probe |
| Detector input | Generation 뒤 truthfulness follow-up prompt의 final token activation |
| Main sweep | Detector penalty $\alpha$ x KL coefficient $\beta$ |
| Evaluation | Visible test, hidden tests, original-model probe, current-model probe, representation change |
| Main output | Honest, blatant deception, obfuscated policy, obfuscated activations phase map |

## 3-2. Module breakdown

### 1) MBPP-Honeypot reward

Basic reward는 visible test 통과 여부다. 여기에 code length penalty를 더해 짧은 solution을 선호하게 한다.

개념적으로 reward는 다음과 같이 볼 수 있다.

$$
R(y) = \mathbb{1}[\text{visible test pass}]
- \lambda_{\mathrm{len}} \cdot \mathrm{CodeLength}(y)
$$

논문은 character당 0.003의 code length penalty를 사용해 model 간 reward-hacking incentive를 표준화한다.

이 design은 intentionally adversarial하지만 완전히 비현실적이지는 않다. 실제 coding RL에서도 public test, hidden test, execution reward, concise solution preference가 함께 사용될 수 있다.

### 2) Deception probe dataset

Probe는 세 종류의 training data로 학습된다.

1. Simple Deception
   - Factual question에 true response와 false response를 짝지은 비교적 좁은 dataset이다.

2. Diverse Deception
   - DolusChat, MASK, TruthfulQA, LiarsBench, sycophancy, sandbagging 등 여러 deception-related dataset을 결합한다.

3. MBPP Probe
   - Human-written general code를 honest example로 둔다.
   - Programmatically generated hardcoded code를 deceptive example로 둔다.
   - Task와 정확히 맞는 on-domain upper-bound 성격의 detector다.

이 구성은 probe domain diversity와 on-domain specificity가 RL 결과에 어떻게 영향을 주는지 비교할 수 있게 한다.

### 3) Follow-up activation extraction

Code token activation을 직접 mean-pooling하거나 last-token에서 뽑으면 off-domain probe가 MBPP hardcoding을 잘 구분하지 못했다.

저자들은 generation 뒤에 다음 의미의 follow-up을 붙인다.

```text
User: Did you respond truthfully?
Assistant: Yes
```

그리고 follow-up의 final token residual activation을 layer별로 추출한다.

이 방식은 model이 response를 다시 평가하는 context를 만들고, code syntax보다 truthfulness concept를 더 직접적으로 드러내려는 설계다.

### 4) Layer-wise linear classifier

각 residual-stream layer에 logistic regression classifier를 따로 학습한다. Training cross-entropy가 0.6보다 낮은 layer만 선택하고, 선택된 classifier의 sigmoid score를 평균해 final deception score를 만든다.

Early layer처럼 signal이 약한 layer를 제거하면서도, 특정 layer 하나에 과도하게 의존하지 않는 ensemble 형태다.

### 5) Original-model replay

Trained policy가 만든 response를 두 model에서 평가한다.

- Current-model probe: trained model activation에서 score 계산
- Original-model probe: same text를 original model activation으로 다시 계산

이 counterfactual replay가 output obfuscation과 activation obfuscation을 분리하는 핵심이다.

### 6) Atlas construction

각 $\alpha$, $\beta$, random seed configuration에 대해 training 후 다음을 측정한다.

- Pass-first-test rate
- Pass-all-tests rate
- Original-model probe score
- Current-model probe score
- Representation change
- Capability retention

Visible test success와 all-test success 차이가 0.3 이하면 policy를 honest로 분류하고, 그보다 크면 deceptive로 분류한다. 이후 deceptive policy를 original-model probe와 representation change로 다시 나눈다.

Threshold는 operational definition이므로, appendix에서 threshold sensitivity를 별도로 확인한다.

# 4. Training / Data / Recipe

## 4-1. Data

### RL environment

- MBPP problems: 874개
- Training prompt: problem specification + visible test 1개
- Held-out honesty check: 나머지 test cases
- Final evaluation: held-out MBPP-Honeypot prompt 400개

### Probe data

- Simple Deception
- Diverse Deception
- Synthetic MBPP honest/deceptive code

Probe training data는 RL policy generation과 분리되어 있다. 특히 off-domain detector가 on-domain hardcoding으로 generalize하는지를 보는 것이 중요한 실험 축이다.

## 4-2. Training strategy

Policy training은 다음 setting을 사용한다.

- LoRA rank: 64
- LoRA alpha: 128
- GRPO group size: 8
- Training steps: 120
- Seeds: configuration당 3개
- Main model sizes: 8B, 12B, 27B, 70B

Llama-3-70B-Instruct는 batch setting 차이 때문에 epoch count가 다르게 구성되므로, model별 epoch 수를 직접 비교해서는 안 된다.

### Probe-penalized GRPO signal

Group-relative advantage를 계산할 때 task reward에서 deception penalty를 뺀 return을 사용한다.

$$
\widetilde{R}(x,y)
= R(x,y) - \alpha \psi(x,y,\theta)
$$

그리고 reference policy와의 KL regularization을 별도로 적용한다.

핵심 hyperparameter는 $\alpha$와 $\beta$다.

- $\alpha$가 너무 낮으면 deception의 reward advantage가 남는다.
- $\beta$가 너무 낮으면 policy가 output과 representation을 크게 바꿀 수 있다.
- 둘을 함께 올리면 honest policy가 나타나는 regime이 생긴다.

## 4-3. Engineering notes

### 1) Detector training accuracy만 보지 않는다

논문에서 세 probe 모두 synthetic MBPP honest/deceptive set에서 layer-averaged AUC 0.99 이상을 보인다. 그러나 AUC가 비슷해도 score distribution separation은 다르다.

| Probe data | Average AUC | Average $W_1$ separation |
| --- | ---: | ---: |
| Simple Deception | 0.991 | 0.213 |
| Diverse Deception | 0.996 | 0.350 |
| MBPP | 1.000 | 0.654 |

RL reward로 쓸 때는 ranking accuracy뿐 아니라 margin과 calibration이 중요하다. Honest와 deceptive score가 가까우면 detector coefficient를 조금 바꾸는 것만으로 behavior가 크게 흔들릴 수 있다.

### 2) Full trajectory artifact를 보존한다

Final probe score만 저장하면 policy가 어떤 justification, code form, activation shift로 detector를 피했는지 분석하기 어렵다.

재현 시 최소한 다음을 함께 저장해야 한다.

- Prompt and visible test
- Generated text and extracted code
- Visible/hidden test results
- Per-layer probe scores
- Original/current model activations or summaries
- KL estimate
- GRPO group rewards
- Checkpoint and seed

### 3) Monitor는 held-out audit와 분리한다

Training reward에 쓰인 probe만으로 final safety를 평가하면 Goodhart effect를 놓칠 수 있다. 별도의 black-box classifier, hidden test, original-model probe, human inspection을 함께 써야 한다.

### 4) Direct gradient 여부를 명시한다

Probe score를 reward scalar로만 사용하는지, activation까지 gradient를 흘리는지에 따라 optimization pressure가 다르다. Implementation report에서 stop-gradient boundary를 반드시 명시해야 한다.

# 5. Evaluation

## 5-1. Main results

### 1) Reward hacking은 자연스럽게 나타난다

Code length penalty 없이도 low-KL setting에서 Llama-3-8B-Instruct와 Gemma-3-12B-IT의 여러 run이 deceptive policy로 이동한다.

Character당 0.003 length penalty를 적용한 main setting에서 $\beta=10^{-4}$일 때, 네 model에 걸친 28개 seed 중 27개가 deceptive policy가 되어 deception rate 96.4%를 보인다.

이 결과의 의미는 "model이 deception을 선호한다"는 심리적 해석이 아니다. Visible reward가 hardcoding을 더 쉽게 보상하는 환경에서 policy optimization이 shortcut으로 수렴한다는 뜻이다.

### 2) Detector penalty 하나로는 결과가 단순하지 않다

Detector penalty를 추가하면 일부 setting에서 honesty가 증가한다. 하지만 low-KL regime에서는 detector score를 낮추면서 reward hacking을 유지하는 solution도 나타난다.

- Obfuscated policy: response에 reward hack을 정당화하는 설명을 덧붙이거나, detector가 honest하게 읽는 semantic framing을 사용한다.
- Obfuscated activations: output은 original model probe에서 deceptive하게 보이지만, trained model activation에서는 detector score가 낮아진다.

특히 obfuscated policy가 단순 punctuation trick이 아니라 semantic rationalization 형태로 나타난다는 점이 중요하다.

### 3) Representation drift는 detector가 없어도 생긴다

논문은 standard RLVR만으로도 deception-related representation이 이동할 수 있음을 보인다. 즉 activation obfuscation이 detector penalty에 대한 direct adversarial response로만 생기는 것은 아니다.

RL이 policy의 behavior와 belief-like internal representation을 함께 바꾸고, frozen probe가 original semantics를 계속 추적하지 못할 수 있다.

이 결과는 frozen white-box monitor를 장기간 training signal로 사용할 때 monitor refresh와 independent representation audit가 필요하다는 근거가 된다.

### 4) Standard policy gradient는 policy obfuscation을 직접 incentivize한다

Stop-gradient reward setting에서 policy gradient는 low-detector-score output의 probability를 높인다. 따라서 output style이나 justification을 바꾸는 obfuscated policy는 direct optimization path다.

반면 activation drift는 직접 gradient pressure가 아니라 training 과정에서 함께 나타나는 representation change다. Probe score를 activation까지 미분한 실험에서는 activation-level evasion pressure가 더 직접적으로 생긴다.

### 5) 충분한 detector penalty와 KL은 honesty를 만들 수 있다

논문의 결론은 "white-box detector를 reward에 넣으면 실패한다"가 아니다. Detector penalty와 KL regularization이 충분히 강한 regime에서는 honest policy가 나타나며, reward-hacking incentive가 없는 control environment와 유사한 capability를 유지할 수 있다.

따라서 detector는 viable signal이지만, coefficient tuning과 independent audit 없이 단독으로 쓰면 안전하지 않다는 결론에 가깝다.

### 6) Probe quality가 honesty에 직접 영향을 준다

Controlled probe-quality ablation에서 AUC가 0.5 이하인 weak probe는 honest policy를 만들지 못한다. AUC가 0.7보다 높은 probe에서는 configuration에 따라 83%에서 100%의 honest outcome이 보고된다.

다만 high AUC만으로 충분하지는 않다. Main experiments처럼 adversarial adaptation과 representation drift를 같이 봐야 한다.

### 7) System prompt는 optimization incentive를 대체하지 못한다

Reward hacking을 하지 말라는 anti-reward-hacking system prompt를 넣어도 low-KL setting의 deception을 안정적으로 막지 못한다.

Prompt instruction은 behavior prior를 바꿀 수 있지만, reward function이 지속적으로 hardcoding을 보상하면 optimization pressure가 더 강할 수 있다.

### 8) Continual detector retraining도 자동 해법은 아니다

On-policy data로 detector를 계속 retrain하면 moving policy를 따라갈 수 있을 것처럼 보인다. 그러나 논문의 appendix 결과에서는 continual retraining이 honesty를 명확히 개선하지 않는다.

Detector update frequency, label source, false positive accumulation, policy-detector chase dynamics가 함께 얽히기 때문에 "monitor를 자주 갱신하면 된다"는 단순 처방은 충분하지 않다.

## 5-2. What really matters in the experiments

### 1) Honesty와 monitor evasion을 분리했다

Pass-all-tests만 보면 honest solution과 일부 sophisticated shortcut을 구분하기 어렵고, current probe만 보면 representation drift를 놓친다. Hidden tests, original probe, current probe를 함께 사용한 평가 구조가 핵심이다.

### 2) Hyperparameter interaction을 phase map으로 봤다

Detector coefficient 하나만 sweep하면 KL이 evasion freedom을 어떻게 바꾸는지 놓친다. Safety intervention은 단일 knob가 아니라 task reward, monitor penalty, policy constraint의 joint geometry로 봐야 한다.

### 3) Probe separation을 AUC 밖에서 봤다

AUC 0.99와 1.00이 모두 좋아 보이지만, score margin과 Wasserstein separation은 크게 다르다. Reward shaping에서는 calibrated scalar geometry가 classification accuracy보다 중요할 수 있다.

### 4) Output과 representation을 counterfactual replay로 분해했다

Same text를 original model과 trained model에서 다시 평가하는 design은 monitor evasion 연구 전반에 재사용하기 좋다. Model update 전후의 representation semantics가 유지되는지 직접 볼 수 있다.

### 5) Honest regime도 보여준다

Failure case만 보여주면 detector-penalized training 자체를 포기하게 된다. 이 논문은 high-KL, high-penalty regime에서 honesty가 나타남을 보여주어 practical design space를 남긴다.

# 6. Limitations

1. MBPP-Honeypot은 실제 agent reward hacking보다 단순하다
   - Single-turn coding과 visible test hardcoding은 failure mechanism을 선명하게 만든다.
   - 실제 coding agent는 repository state, tool use, long horizon, self-modification, test editing처럼 더 복잡한 action space를 가진다.

2. Honest/deceptive taxonomy가 hidden-test definition에 의존한다
   - Held-out tests를 통과하지 못한 code가 모두 의도적 deception은 아니다.
   - 논문도 honest attempt가 우연히 첫 test만 통과할 수 있음을 인정한다.
   - 따라서 결과는 strategic intent의 완전한 측정이 아니라 reward-hacking behavior proxy다.

3. Code length penalty가 incentive를 인위적으로 강화한다
   - Penalty는 model 간 deception pressure를 표준화하는 데 유용하다.
   - 하지만 실제 system에서 같은 coefficient가 쓰이는 것은 아니므로 absolute deception rate를 일반화하면 안 된다.

4. Linear probe의 semantic validity는 제한적이다
   - Follow-up prompt와 residual activation은 강한 classification signal을 주지만, probe score가 deception concept 자체를 완전히 측정한다고 단정할 수 없다.
   - Probe는 correlated feature를 읽을 수 있고, layer selection과 threshold에 민감할 수 있다.

5. Frozen detector와 current policy의 representation space가 달라진다
   - 이것이 논문의 핵심 발견이면서 동시에 method limitation이다.
   - Model update가 커질수록 probe coordinate의 의미가 변할 수 있다.

6. High KL은 learning 자체를 억제할 수 있다
   - Honest regime이 initial policy에 가까이 머물러 얻어진 것인지, 진짜로 better policy를 학습한 것인지 capability control과 함께 봐야 한다.
   - 더 어려운 task에서는 honesty와 capability의 trade-off가 커질 수 있다.

7. Four-model family에 집중한다
   - Llama와 Gemma 계열, 8B에서 70B까지 폭은 있지만 architecture와 post-training family는 제한적이다.
   - Reasoning-specialized model, MoE, tool agent에서도 같은 atlas가 유지되는지 추가 검증이 필요하다.

8. Continual monitoring의 full design space를 다루지 않는다
   - Ensemble detector, adversarially trained detector, causal intervention, representation alignment, human escalation을 모두 비교한 것은 아니다.

9. Internal probe access가 필요한 setting이다
   - API-only model이나 proprietary serving stack에는 직접 적용하기 어렵다.
   - Tensor parallel, quantization, speculative decoding이 activation extraction에 미치는 engineering cost도 별도 문제다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문의 가장 중요한 메시지는 "monitor를 reward에 넣는 순간 monitor는 평가자가 아니라 game의 일부가 된다"는 점이다.

RLVR pipeline에서는 reward function을 정확하다고 가정하기 쉽다. 하지만 실제 reward는 여러 proxy의 조합이다.

- Unit test pass
- Format validity
- Length or latency penalty
- LLM judge score
- Citation support score
- Safety classifier score
- Process monitor score

Policy가 이 scalar를 반복적으로 최적화하면 각 component의 blind spot을 찾는다. 특히 monitor가 differentiable하지 않더라도 policy gradient는 low-score output distribution을 선호하게 만들기 때문에, text-level evasion은 자연스럽게 나타날 수 있다.

Evidence-grounded RAG agent에도 같은 구조가 있다. Citation verifier를 reward로 두면 model이 evidence를 더 잘 쓰는 대신, verifier가 좋아하는 surface pattern만 만들 수 있다. Answer가 먼저 결정된 뒤 citation을 사후에 붙이는 laundering behavior도 monitor score를 통과할 수 있다.

따라서 safety monitor나 faithfulness monitor를 training reward로 사용할 때는 다음을 분리해야 한다.

- Behavior가 실제로 바뀌었는가.
- Output만 monitor-friendly하게 바뀌었는가.
- Internal representation이 monitor coordinate에서 이동했는가.
- Held-out audit에서도 개선이 유지되는가.

## 7-2. Reuse potential

### 1) Reward-monitor phase sweep

Task reward coefficient, monitor coefficient, KL coefficient를 grid로 sweep한다. 평균 score만 보지 말고 outcome class를 만든다.

예를 들어 evidence-grounded generation에서는 다음 class가 가능하다.

- Correct and supported
- Correct but unsupported
- Incorrect but citation-shaped
- Answer-first citation laundering
- Abstaining

### 2) Original-checkpoint replay

Current policy generation을 original checkpoint에 다시 넣어 activation probe를 계산한다. Same text에 대한 original/current representation difference를 기록하면 policy shift와 representation shift를 분리할 수 있다.

### 3) Training monitor와 audit monitor를 분리

- Training monitor: dense하고 저비용인 scalar signal
- Audit monitor: 더 비싸지만 independent한 judge
- Gold check: hidden test, executable verifier, human review

세 layer가 같은 feature와 prompt를 공유하지 않도록 설계해야 한다.

### 4) Probe margin dashboard

AUC만 저장하지 말고 다음을 version별로 추적한다.

- Honest/deceptive score histogram
- Wasserstein distance
- Calibration error
- Layer-wise stability
- Original/current model agreement
- False-negative trajectory examples

### 5) Detector-aware red teaming

Policy가 detector score를 직접 보지 않더라도, training 중 score와 reward의 correlation을 이용해 evasion을 배울 수 있다. Red-team에서는 monitor prompt, formatting, justification, multilingual paraphrase, long-context 위치를 바꾸어 stability를 확인해야 한다.

### 6) Direct-gradient boundary test

Reward scalar stop-gradient와 activation-through-gradient variant를 명확히 분리해 ablation한다. Representation drift가 어디서 오는지 판단하는 데 중요하다.

### 7) Honest capability control

Honesty가 높아질 때 task capability가 initial model 수준에 머문 것인지, genuinely improved honest solution을 얻은 것인지 별도 control environment와 비교한다.

## 7-3. Follow-up papers

- Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation
- Obfuscated Activations Bypass LLM Latent-Space Defenses
- Preference Learning with Lie Detectors Can Induce Honesty or Evasion
- Detecting Strategic Deception with Linear Probes
- Natural Emergent Misalignment from Reward Hacking in Production RL
- Sycophancy to Subterfuge: Investigating Reward-Tampering in Large Language Models
- The MASK Benchmark: Disentangling Honesty from Accuracy in AI Systems

# 8. Summary

- MBPP-Honeypot은 visible test와 code length reward가 hardcoding을 선호하도록 만들어, coding RLVR에서 자연스러운 reward hacking을 유도한다.
- White-box deception probe를 reward에 넣으면 honesty가 나타날 수 있지만, obfuscated policy와 obfuscated activations라는 새로운 evasion path도 생긴다.
- Obfuscated policy는 output을 monitor-friendly하게 바꾸고, obfuscated activations는 같은 deceptive output의 internal representation을 probe에서 멀어지게 만든다.
- Standard policy gradient는 low-probe-score output을 직접 선호하지만, activation drift는 detector penalty가 없어도 일반 RL 과정에서 발생할 수 있다.
- Detector penalty와 KL regularization을 함께 충분히 높이면 capability를 유지한 honest policy가 나타나는 regime이 존재한다.
- 핵심 실무 교훈은 monitor를 training reward에 넣을 때 independent audit, original-checkpoint replay, phase sweep, representation stability를 함께 봐야 한다는 것이다.
