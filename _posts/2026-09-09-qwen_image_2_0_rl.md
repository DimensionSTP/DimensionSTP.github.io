---
layout: single
title: "Qwen-Image-2.0-RL Technical Report Review"
categories: Study-concept
tag: [Qwen-Image-2.0-RL, RLHF, OPD, ImageGeneration, DiffusionModel]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2606.27608)

Qwen-Image-2.0-RL Technical Report는 image generation model post-training을 꽤 실용적인 형태로 정리한 보고서다. 이 논문이 겨냥하는 핵심은 단순히 Qwen-Image-2.0의 score를 더 올리는 것이 아니라, **T2I generation과 image editing이라는 서로 다른 task-specialized policy를 RL로 강화한 뒤 하나의 deployable student로 다시 합치는 pipeline**이다.

Image generation에서 RLHF는 말은 쉽지만 실제로는 reward design이 어렵다. Text-to-image에서는 prompt alignment, aesthetics, portrait fidelity가 중요하다. Image editing에서는 instruction following, source image preservation, face identity preservation이 더 중요하다. 하나의 scalar reward로 이들을 모두 대충 섞으면 task별 failure mode를 잡기 어렵고, 한 task에 맞춘 RL policy가 다른 task의 prior를 깨뜨릴 수도 있다.

Qwen-Image-2.0-RL은 이를 세 단계로 나눠 다룬다.

1. Task-specific composite reward models를 만든다.
2. GRPO 기반 RL로 T2I와 editing policy를 각각 강화한다.
3. On-policy distillation, 이하 OPD로 specialized policies를 하나의 student로 consolidate한다.

여기서 흥미로운 지점은 final stage가 단순 model merging이 아니라 trajectory-level velocity matching이라는 점이다. Diffusion/flow model에서 policy behavior는 final image만이 아니라 denoising trajectory의 vector field에 있다. 따라서 student는 teacher의 output image만 따라 하는 것이 아니라, teacher가 trajectory에서 어떤 velocity direction을 냈는지 따라간다.

> 한 줄 요약: Qwen-Image-2.0-RL은 Qwen-Image-2.0을 대상으로 task-specific reward model, GRPO-based RL, hybrid CFG, prompt range filtering, per-category reward calibration, OPD trajectory-level velocity matching을 결합해 T2I와 editing RL policy를 하나의 stronger image generation model로 consolidate하는 post-training pipeline이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Text-to-image와 image editing을 하나의 RLHF recipe로 뭉개지 않고 reward dimension을 분리한다.
- GRPO-style RL을 image diffusion model post-training에 적용할 때 필요한 prompt filtering, CFG preservation, reward calibration을 구체적으로 제시한다.
- OPD를 diffusion velocity field consolidation 문제로 사용한다.
- Qwen-Image-Bench, T2I arena, image edit arena 수치를 통해 base model 대비 post-training gain을 보고한다.
- Image generation model post-training에서 reward engineering과 policy merging이 얼마나 중요한지 보여준다.
- Qwen-Image-Agent 같은 agentic generation pipeline의 renderer backbone quality와도 직접 연결된다.

이 글에서는 Qwen-Image-2.0-RL을 "image generation RL report"보다, **reward-specialized RL policies를 diffusion trajectory distillation으로 다시 하나의 model에 합치는 post-training recipe paper**로 읽는다.

# 1. Problem Setting

## 1-1. Problem definition

Image generation model $G_{\theta}$는 prompt 또는 editing condition $c$를 받아 image $x$를 생성한다.

$$
x
\sim
G_{\theta}(\cdot \mid c)
$$

Post-training objective는 human preference와 task-specific quality를 반영하는 reward $R$를 높이는 것이다.

$$
\max_{\theta}
\mathbb{E}_{x \sim G_{\theta}(\cdot \mid c)}
[
R(x,c)
]
$$

하지만 text-to-image와 image editing은 reward structure가 다르다.

| Task | Important reward dimensions |
| --- | --- |
| Text-to-image | Alignment, aesthetics, portrait fidelity |
| Image editing | Instruction following, source preservation, face identity preservation |

한 model이 두 task를 모두 지원하려면 다음 문제가 생긴다.

1. Reward model이 task-specific이어야 한다.
2. RL policy가 base model knowledge를 과도하게 깨뜨리지 않아야 한다.
3. T2I와 editing policy를 따로 강화한 뒤, deployment에서는 하나의 model로 합쳐야 한다.
4. Diffusion model에서는 final image뿐 아니라 sampling trajectory behavior가 중요하다.

Qwen-Image-2.0-RL의 문제 설정은 바로 이 네 가지를 post-training pipeline으로 다루는 것이다.

## 1-2. Why previous approaches are insufficient

### 1) Single holistic reward

하나의 preference score로 T2I와 editing을 모두 평가하면 failure attribution이 어렵다. Prompt alignment 문제인지, aesthetics 문제인지, face identity 문제인지, instruction-following 문제인지 분리되지 않는다.

### 2) RL without prior preservation

RL이 reward hacking이나 over-optimization으로 base model prior를 망가뜨릴 수 있다. Image generation에서 이는 diversity loss, artifact 증가, style collapse, instruction overfitting으로 나타날 수 있다.

### 3) Separate specialized policies

T2I-specific RL policy와 editing-specific RL policy를 따로 배포하면 serving과 product complexity가 커진다. 가능하다면 하나의 model이 두 task를 모두 처리해야 한다.

### 4) Output-level distillation only

Diffusion model behavior는 one-shot output distribution만이 아니라 denoising trajectory의 velocity field에 있다. Output image matching만으로는 teacher policy의 generation dynamics를 충분히 internalize하기 어렵다.

# 2. Core Idea

## 2-1. Main contribution

논문의 핵심 기여는 세 단계로 볼 수 있다.

1. **Task-specific composite reward model**
   - VLM-based pointwise scoring paradigm을 사용한다.
   - Chain-of-thought reasoning을 포함해 평가 signal을 더 설명적으로 만든다.
   - T2I와 editing task에 맞는 reward dimension을 따로 구성한다.

2. **Scalable GRPO-based RL**
   - Qwen-Image-2.0 diffusion model을 RL로 post-train한다.
   - Hybrid CFG strategy로 pretrained knowledge를 preserve한다.
   - Intra-group reward range filtering으로 useful prompt를 curate한다.
   - Per-category reward weight calibration으로 reward scale imbalance를 줄인다.

3. **On-policy distillation**
   - T2I policy와 editing policy를 final student로 consolidate한다.
   - Distillation objective는 trajectory-level velocity matching이다.
   - Deployment model은 task-specialized RL teachers의 behavior를 한 model에 흡수한다.

## 2-2. Design intuition

이 pipeline의 design intuition은 다음과 같다.

```text
reward dimension은 task-specific해야 한다
RL은 base prior를 깨뜨리지 않으면서 task behavior를 개선해야 한다
specialized RL teacher는 diffusion trajectory matching으로 합쳐야 한다
```

일반 LLM post-training에서 여러 preference dataset을 섞는 것과 비슷해 보이지만, image diffusion에서는 두 가지가 다르다.

- Reward는 final image quality를 다차원으로 봐야 한다.
- Policy behavior는 denoising trajectory와 velocity field에 나타난다.

따라서 OPD가 중요한 역할을 한다. OPD는 checkpoint를 단순 평균하는 것이 아니라, policy-generated state에서 student가 teacher의 velocity trajectory를 맞추도록 요구한다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Qwen-Image-2.0의 visual quality와 instruction following 개선 |
| Base model | Qwen-Image-2.0 diffusion model |
| Reward modeling | Task-specific composite VLM reward models |
| T2I reward dimensions | Alignment, aesthetics, portrait fidelity |
| Editing reward dimensions | Instruction following, face identity preservation |
| RL method | GRPO-based scalable RL training |
| Prior preservation | Hybrid CFG strategy |
| Prompt selection | Intra-group reward range filtering |
| Calibration | Per-category reward weight calibration |
| Final merge | OPD trajectory-level velocity matching |
| Main result | 57.84 Qwen-Image-Bench overall, T2I Elo 1193, edit Elo 1349 |

## 3-2. Module breakdown

### 1) Composite reward model

Reward model은 generated image와 condition을 pointwise score로 mapping한다.

$$
r_d
=
R_d(x,c)
$$

여기서 $d$는 reward dimension을 가리킨다. T2I에서는 alignment, aesthetics, portrait fidelity가 될 수 있고, editing에서는 instruction-following accuracy나 face identity preservation이 될 수 있다.

Total reward는 weighted sum으로 쓸 수 있다.

$$
R(x,c)
=
\sum_d
w_d R_d(x,c)
$$

논문은 per-category reward weight calibration을 보고한다. Reward dimension마다 score range와 optimization sensitivity가 다를 수 있기 때문이다.

### 2) GRPO-based RL

GRPO는 각 prompt마다 output group을 sampling하고 group-relative reward로 policy를 update한다. Conceptually, prompt $c$에 대해 다음 group을 sample한다.

$$
\{x_1,\ldots,x_K\}
\sim
G_{\theta_{\mathrm{old}}}(\cdot \mid c)
$$

Reward와 advantage는 다음처럼 계산한다.

$$
A_i
=
\frac{
R(x_i,c)-\mathrm{mean}_j R(x_j,c)
}{
\mathrm{std}_j R(x_j,c)+\epsilon
}
$$

그 다음 old policy와의 divergence를 제어하면서 higher-reward sample 쪽으로 model을 update한다. 정확한 image diffusion GRPO implementation detail은 paper에서 확인해야 한다.

### 3) Hybrid CFG

Classifier-free guidance, CFG는 conditional prediction과 unconditional prediction을 섞어 sampling behavior를 바꾼다. Pure RL은 pretrained behavior를 왜곡할 수 있다. Hybrid CFG는 RL이 task-specific quality를 개선하게 하면서 pretrained knowledge를 보존하기 위해 사용된다.

핵심은 CFG 자체가 새롭다는 것이 아니라, guidance strategy가 RL stabilization의 일부가 된다는 점이다.

### 4) Prompt curation via intra-group reward range

Group 안의 모든 sample이 비슷한 reward를 받으면 그 prompt는 약한 learning signal만 제공한다. 논문은 model candidate 사이에 meaningful reward difference가 나타나는 prompt를 고르기 위해 intra-group reward range filtering을 사용한다.

Conceptually:

$$
\Delta R(c)
=
\max_i R(x_i,c)
-
\min_i R(x_i,c)
$$

충분히 큰 $\Delta R(c)$를 가진 prompt는 GRPO에 더 informative하다. 이는 RL을 위한 practical data curation trick이다.

### 5) OPD with trajectory-level velocity matching

Teacher policy $T$와 student $S$가 diffusion sampling 동안 velocity field를 예측한다고 하자.

$$
v_T(x_t,t,c)
$$

$$
v_S(x_t,t,c)
$$

OPD는 student가 teacher의 trajectory-level velocity를 맞추도록 학습한다.

$$
\mathcal{L}_{\mathrm{OPD}}
=
\mathbb{E}_{x_t,t,c}
\left[
\left\|
v_S(x_t,t,c)
-
v_T(x_t,t,c)
\right\|_2^2
\right]
$$

위 notation은 단순화한 것이다. 중요한 점은 teacher behavior가 final image level만이 아니라 trajectory level에서 distill된다는 점이다.

# 4. Training / Data / Recipe

## 4-1. Data

Abstract는 full training data composition을 공개하지 않는다. Post-training data는 T2I와 editing task prompt를 포함하고, reward model이 generated output을 평가하는 구조일 것으로 보인다.

Key data-related components:

| Component | Role |
| --- | --- |
| T2I prompts | Generation quality와 alignment에 대한 RL |
| Editing prompts | Edit instruction following과 identity preservation에 대한 RL |
| Group samples | Reward normalization을 위한 GRPO candidate set |
| Prompt filter | Useful reward spread가 있는 prompt 선택 |
| Arena/evaluation prompts | Human 또는 model-judged evaluation setting |

Full prompt source, curation policy, filtering threshold는 paper table과 appendix에서 확인해야 한다.

## 4-2. Training strategy

Training은 다음 순서로 진행된다.

1. Task-specific reward model을 만든다.
2. GRPO와 T2I reward dimension으로 T2I RL policy를 학습한다.
3. GRPO와 editing reward dimension으로 editing RL policy를 학습한다.
4. Hybrid CFG와 calibrated reward weight로 training을 안정화한다.
5. OPD를 통해 specialized policy를 single student로 distill한다.

## 4-3. Engineering notes

1. **Reward dimension separation이 필수다**
   - Alignment, aesthetics, identity preservation은 서로 바꿔 쓸 수 없다.

2. **Prompt filtering은 RL signal density를 높인다**
   - Reward spread가 낮은 group은 약한 gradient signal만 제공한다.

3. **CFG는 preservation tool이다**
   - RL post-training은 base prior를 망가뜨리지 않아야 한다.

4. **OPD는 product complexity를 줄인다**
   - Specialized teacher를 하나의 inference model로 합칠 수 있다.

5. **Velocity matching은 response matching보다 diffusion에 잘 맞는다**
   - Diffusion model의 behavior는 trajectory 안에 있다.

6. **Arena result는 human protocol 확인이 필요하다**
   - Elo score는 arena setup과 rater distribution에 의존한다.

# 5. Evaluation

## 5-1. Main results

ArXiv abstract는 다음을 보고한다.

| Metric | Qwen-Image-2.0-RL | Gain |
| --- | ---: | ---: |
| Qwen-Image-Bench overall | 57.84 | +2.61 over base |
| T2I arena Elo | 1193 | +78 |
| Image edit arena Elo | 1349 | +93 |

Abstract는 gain이 aesthetic quality, prompt adherence, editing accuracy 전반에서 일관적이라고 설명한다.

## 5-2. What really matters in the experiments

### 1) Reward model reliability가 중요하다

전체 RL pipeline은 reward model quality에 의존한다. VLM reward가 style에 overfit되거나 identity failure를 놓치면 RL은 잘못된 signal을 optimize할 수 있다.

### 2) OPD result는 product-relevant하다

Separate RL teacher는 training에는 유용하지만, deployment에는 하나의 student가 더 쉽다. 따라서 OPD는 단순 실험 기법이 아니라 deployment consolidation stage다.

### 3) Qwen-Image-Bench와 arena는 상호 보완적이다

Benchmark score는 structured rubric 관점을 주고, Arena Elo는 preference competition 관점을 준다. 둘은 함께 해석해야 한다.

### 4) Per-task breakdown이 필요하다

Overall 57.84는 유용한 지표지만 T2I와 editing의 failure mode는 다르다. 강한 결론을 내리기 전에 category-level table을 확인해야 한다.

# 6. Limitations

1. **Public detail이 아직 abstract 수준이다**
   - 작성 시점에는 arXiv abstract가 main pipeline과 headline result를 제공하지만, full table은 다시 확인해야 한다.

2. **Reward model bias가 있다**
   - VLM reward model은 preference, style, demographic bias를 물려받을 수 있다.

3. **RL over-optimization risk가 있다**
   - 강한 reward optimization은 diversity를 줄이거나 reward hacking을 만들 수 있다.

4. **Hybrid CFG가 complexity를 추가한다**
   - Guidance strategy가 training recipe의 일부가 되므로 세심한 tuning이 필요할 수 있다.

5. **OPD는 teacher quality에 묶인다**
   - Student는 specialized teacher가 제공한 behavior 안에서만 consolidate할 수 있다.

6. **Arena Elo는 evaluation protocol에 의존한다**
   - Rater distribution, prompt set, comparison pool이 중요하다.

7. **Editing identity preservation은 어렵다**
   - Face identity reward가 모든 real-world identity와 privacy concern을 포괄하지 못할 수 있다.

8. **아직 independent reproduction이 없다**
   - Code, checkpoint, reward model release status를 확인해야 한다.

9. **Application safety 문제가 있다**
   - 더 나은 portrait fidelity와 editing accuracy는 misuse risk를 높일 수 있다.

10. **Renderer-only improvement다**
    - 이 pipeline은 model post-training을 개선하지만 context gap을 직접 해결하지는 않는다. Agentic context construction은 별도 문제로 남는다.

# 7. My Take

## 7-1. Why this matters for my work

Qwen-Image-2.0-RL의 핵심은 "RLHF를 image generation에도 적용했다"가 아니다. 더 중요한 점은 **reward specialization과 policy consolidation을 분리했다는 것**이다.

Product image generation에서는 specialized improvement와 deployable single model이 동시에 필요하다. 이 논문은 그럴듯한 recipe를 제시한다.

```text
task별로 specialize한다
task-specific reward로 reinforce한다
trajectory-level OPD로 merge한다
하나의 model로 serve한다
```

이 pattern은 image generation뿐 아니라 video generation, editing, multimodal generation에도 재사용 가능하다.

## 7-2. Reuse potential

### Image editing post-training

Instruction following과 identity preservation은 T2I aesthetics와 다르게 reward되어야 한다. Separate RL teacher와 OPD merge는 깔끔한 design이다.

### Video generation

Video model도 motion realism, prompt adherence, subject preservation, editing consistency에 대해 separate reward policy를 쓰고, 이후 trajectory-level distillation으로 consolidate할 수 있다.

### Agentic generation

Qwen-Image-Agent에는 강한 renderer가 필요하다. Qwen-Image-2.0-RL은 renderer side를 강화하고, Qwen-Image-Agent는 context construction을 다룬다.

### Reward engineering

Per-category calibration을 가진 composite reward model은 하나의 monolithic image preference model보다 유용할 가능성이 높다.

## 7-3. Production considerations

- Total score뿐 아니라 per-category reward를 monitoring한다.
- Human preference audit와 reward model score를 분리한다.
- RL 이후 diversity와 failure mode를 확인한다.
- Editing task는 identity, text, layout, style preservation을 분리해 test한다.
- Reward model과 calibration weight를 versioning한다.
- 강해진 portrait/editing capability의 safety impact를 audit한다.

## 7-4. Follow-up papers

- Qwen-Image-2.0 Technical Report
- Qwen-Image-Bench
- Qwen-Image-Agent
- DanceOPD
- On-Policy Distillation
- GRPO
- RLHF for diffusion and image generation
- Classifier-Free Guidance

# 8. Summary

- Qwen-Image-2.0-RL은 Qwen-Image-2.0을 위한 post-training pipeline이다.
- T2I와 editing을 위한 task-specific composite reward model을 만든다.
- Hybrid CFG, reward-range prompt filtering, reward calibration을 결합한 GRPO-based RL을 사용한다.
- OPD velocity matching으로 specialized RL teacher를 하나의 student에 consolidate한다.
- 핵심 교훈은 image generation RL에 reward specialization과 deployment-oriented policy consolidation이 모두 필요하다는 점이다.
