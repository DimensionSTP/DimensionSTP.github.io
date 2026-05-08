---
layout: single
title: "Visual Generation in the New Era: An Evolution from Atomic Mapping to Agentic World Modeling Review"
categories: Study-concept
tag: [VisualGeneration, WorldModel, MultimodalAI, Diffusion, Agent]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2604.28185)

[Project page](https://evolvinglmms-lab.github.io/Evolving-Visual-Generation)

[GitHub link](https://github.com/EvolvingLMMs-Lab/Evolving-Visual-Generation)

Visual Generation in the New Era는 "요즘 image generation model이 좋아졌다"를 정리하는 평범한 survey로 읽으면 핵심을 놓치기 쉽다. 이 논문이 진짜로 하려는 일은 visual generation의 진보를 perceptual quality나 benchmark score가 아니라 **capability level**로 다시 분해하는 것이다. 즉 더 예쁜 이미지를 만드는가보다, 모델이 구조를 이해하는가, 상태를 유지하는가, 피드백을 받아 고치는가, 그리고 결국 world simulator처럼 동작할 수 있는가를 묻는다.

최근 image/video generation은 이미 많은 면에서 인상적이다. photorealism은 좋아졌고, typography도 상당히 좋아졌고, instruction following과 interactive editing도 빠르게 올라왔다. 하지만 실제로 복잡한 요청을 던져 보면 failure mode는 여전히 선명하다. 지하철 노선도처럼 topology가 중요한 이미지는 그럴듯해 보이지만 연결이 틀어지고, 지도나 좌표계에서는 spatial relation이 깨지고, multi-turn editing에서는 처음 정한 identity나 layout이 조용히 drift한다. 영상 쪽에서는 object permanence, physical causality, long-horizon consistency가 아직 약하다.

이 논문은 이 gap을 하나의 문장으로 정리한다. visual generation은 appearance synthesis에서 intelligent visual generation으로 넘어가야 한다. 여기서 intelligent visual generation은 단순히 보기 좋은 output이 아니라, structure, dynamics, domain knowledge, causal relation에 grounded된 visual output을 뜻한다.

> 한 줄 요약: 이 논문은 visual generation의 진화를 Atomic Generation, Conditional Generation, In-Context Generation, Agentic Generation, World-Modeling Generation이라는 5단계 taxonomy로 정리하고, 앞으로의 병목이 photorealism보다 structure, memory, feedback loop, causal simulation에 있다고 주장하는 roadmap paper다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- image generation과 video generation을 하나의 capability ladder로 묶어 읽을 수 있다.
- open model과 closed model의 차이를 raw visual quality가 아니라 data engine, post-training, long context, verification loop 관점으로 다시 본다.
- diffusion, flow matching, autoregressive model, hybrid AR plus diffusion/flow, unified multimodal model을 기술 조각이 아니라 capability transition을 만드는 building block으로 정리한다.
- current benchmark가 perceptual quality를 과대평가하고 structural, temporal, causal failure를 놓친다는 점을 꽤 명확하게 짚는다.
- visual generation을 단발성 rendering task가 아니라 agentic system과 world model로 확장하는 관점을 준다.

이 논문은 새로운 generator를 제안하는 논문이 아니다. 오히려 최근 Qwen-Image, Seedream, HunyuanImage, Wan-Image, Z-Image, GPT-Image 계열을 읽을 때 어떤 축으로 비교해야 하는지 알려주는 **review frame**에 가깝다. 특히 visual generation을 실서비스나 multimodal agent에 붙이려는 사람에게는, "좋은 이미지"와 "쓸 수 있는 visual reasoning output" 사이의 차이를 정리해 준다는 점에서 꽤 유용하다.

# 1. Problem Setting

## 1-1. Problem definition

이 논문이 겨냥하는 문제는 visual generation field가 여전히 output appearance 중심으로 평가되고 있다는 점이다. 현재 많은 model comparison은 다음 질문에 머문다.

- 이미지가 선명한가.
- 사람이 보기에 아름다운가.
- prompt의 표면적 키워드를 잘 반영하는가.
- text rendering이나 layout이 그럴듯한가.
- 한 번의 edit instruction을 잘 따르는가.

물론 이 질문들은 중요하다. 그러나 논문은 이 정도로는 다음 generation model의 핵심 능력을 평가하기 어렵다고 본다. 실제 intelligent visual generation에서는 다음 질문이 더 중요해진다.

- 물체 간 spatial relation이 정확히 유지되는가.
- reference identity와 state가 multi-turn interaction 동안 보존되는가.
- 모델이 visual input과 text instruction을 함께 해석해 plan을 세울 수 있는가.
- generation 결과를 다시 보고 self-correction할 수 있는가.
- physical dynamics와 intervention effect를 어느 정도 일관되게 simulate할 수 있는가.

즉 문제는 단순히 image quality가 아니라 **visual output이 world constraint를 얼마나 지키는가**다. 이 관점에서 visual generation은 text-to-image mapping 문제가 아니라, 점점 더 visual reasoning, memory, tool use, world simulation에 가까워진다.

이 논문은 이 진화를 5단계로 정리한다.

| Level | Name | Core capability | Main failure mode |
| --- | --- | --- | --- |
| L1 | Atomic Generation | one-shot distribution matching | controllability 부족 |
| L2 | Conditional Generation | explicit condition 기반 controllable generation | attribute binding과 spatial precision 부족 |
| L3 | In-Context Generation | rich context와 history를 한 번에 흡수 | identity/state drift |
| L4 | Agentic Generation | generation을 closed-loop action으로 사용 | verification과 self-correction 부족 |
| L5 | World-Modeling Generation | dynamics와 causal effect를 internalize | causal faithfulness와 physical plausibility 부족 |

이 표가 중요한 이유는, 각 level이 단순히 더 큰 모델을 뜻하지 않기 때문이다. L1에서 L5로 갈수록 필요한 것은 더 많은 parameter만이 아니라, condition interface, memory mechanism, verifier, tool loop, simulation state 같은 system-level component다.

## 1-2. Why previous approaches are insufficient

기존 visual generation 접근은 크게 세 가지 한계를 갖는다.

첫째, **single-pass mapping bias**다. 많은 text-to-image model은 prompt에서 image로 바로 mapping한다. 이 구조는 photorealistic sample을 만들기에는 강하지만, 요청이 compositional하거나 step-wise verification이 필요할 때 약하다. 예를 들어 "이 지도에서 A와 B가 같은 노선에 있고 C는 환승역이어야 한다" 같은 요청은 보기 좋은 그림보다 topology correctness가 중요하다.

둘째, **implicit memory의 한계**다. In-context generation은 여러 reference image, edit history, instruction을 입력으로 넣어 context를 흡수한다. 하지만 이 memory는 대부분 input context에 implicit하게 들어 있을 뿐, external state나 verifier가 보장하는 memory가 아니다. 그래서 multi-turn editing에서는 초기 identity, style, spatial layout이 점진적으로 사라질 수 있다.

셋째, **evaluation mismatch**다. current benchmark는 perceptual quality, prompt alignment, aesthetic preference를 많이 본다. 하지만 논문이 강조하듯 이런 metric은 structural integrity, temporal coherence, causal reasoning failure를 잘 잡지 못한다. 모델이 그림을 예쁘게 만들수록 사람이나 evaluator가 topology error를 놓칠 수도 있다.

이 문제는 특히 최근 closed model과 open model 비교에서 중요하다. raw image quality만 보면 open model도 빠르게 따라오고 있다. 그러나 multi-turn editing, long-form instruction adherence, domain-specific structured image, reasoning-augmented generation에서는 gap이 남아 있다. 논문은 이 gap이 architecture 하나보다 data curation, post-training, long context, tool use, verification loop에서 생긴다고 본다.

# 2. Core Idea

## 2-1. Main contribution

이 논문의 핵심 기여는 새로운 loss나 architecture가 아니라 **capability-centered taxonomy와 roadmap**이다.

구체적으로는 네 가지로 정리할 수 있다.

1. **5-level taxonomy 제안**
   - Atomic Generation에서 World-Modeling Generation까지 visual generation의 진화를 capability level로 정리한다.
   - 각 level은 대표 model family, characteristic, failure mode를 갖는다.

2. **technical driver 정리**
   - GAN, diffusion, flow matching, autoregressive generation, hybrid AR plus diffusion/flow를 하나의 generative paradigm map 안에서 정리한다.
   - U-Net에서 DiT, MM-DiT, AR backbone, SSM, MoE backbone으로 이동하는 architecture trend를 설명한다.
   - condition module, multimodal fusion, tokenizer, representation space를 separate component로 본다.

3. **training and inference recipe 정리**
   - pre-training, continued training, SFT, RL, reward modeling, synthetic annotation, data curation, distillation, sampling acceleration을 한 pipeline으로 묶는다.
   - 특히 VLM relabeling, structured caption, defect-aware filtering, active curation이 visual generator quality에 미치는 의미를 강조한다.

4. **evaluation frame 제안**
   - benchmark review만으로는 부족하고, in-the-wild stress test와 expert-constrained case study가 필요하다고 주장한다.
   - jigsaw reconstruction, metro-map topology, coordinate map, physical causality, multi-turn editing, image-text reasoning 같은 failure probe를 제안한다.

이 논문에서 가장 중요한 메시지는 다음과 같다.

> visual generation의 다음 병목은 rendering fidelity가 아니라, generated visual output이 구조적이고 동적인 world constraint를 얼마나 지키는가다.

## 2-2. Design intuition

이 논문의 설계 직관은 L1에서 L5로 갈수록 generation이 점점 더 "행동"에 가까워진다는 데 있다.

L1에서는 모델이 prompt를 받아 한 번에 image를 만든다. 이때 핵심은 distribution matching이다. 보기 좋고 그럴듯하면 어느 정도 충분하다. 하지만 L2로 가면 depth, sketch, pose, reference identity 같은 condition을 받아야 하므로 controllability가 중요해진다. L3에서는 context와 history가 들어오므로 memory와 consistency가 중요해진다. L4에서는 generation이 agent loop 안의 action이 된다. 즉 모델은 만들고, 보고, 평가하고, 다시 고쳐야 한다. L5에서는 모델이 visual output을 넘어 world transition 자체를 simulate해야 한다.

이 흐름을 간단히 쓰면 다음과 같다.

$$
\text{rendering} \to \text{controlled rendering} \to \text{contextual editing} \to \text{closed-loop generation} \to \text{world simulation}
$$

여기서 중요한 것은 각 단계가 이전 단계를 완전히 대체하지 않는다는 점이다. L4 agentic generation도 여전히 좋은 renderer가 필요하고, L5 world modeling도 여전히 photorealistic rendering이 필요하다. 다만 상위 level로 갈수록 renderer 위에 추가적인 state, verifier, planner, dynamics model이 필요해진다.

이 taxonomy의 장점은 visual generation model을 단일 score로 줄이지 않는다는 점이다. 어떤 model은 L1/L2에서는 강하지만 L3 multi-turn consistency에서 약할 수 있다. 어떤 model은 edit instruction은 잘 따르지만 physical causality는 약할 수 있다. 그래서 앞으로는 "이 모델이 좋은가"보다 "어떤 level의 어떤 capability가 좋은가"를 물어야 한다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | visual generation의 진화를 appearance synthesis에서 intelligent generation으로 재정의 |
| Core framework | L1 Atomic, L2 Conditional, L3 In-Context, L4 Agentic, L5 World-Modeling |
| Main method type | survey, taxonomy, roadmap, evaluation analysis |
| Technical drivers | flow matching, DiT/MM-DiT, unified multimodal architecture, data curation, post-training, reward modeling, sampling acceleration |
| Evaluation focus | perceptual quality보다 structure, state, dynamics, causality, multi-turn consistency |
| Difference from model paper | new architecture가 아니라 field-level capability decomposition을 제안 |

## 3-2. Module breakdown

### 1) L1 Atomic Generation

Atomic Generation은 가장 기본적인 generation setting이다. input prompt 또는 latent condition에서 visual output을 한 번에 생성한다. 이 level의 핵심은 stochastic plausibility, 즉 training distribution 위에서 그럴듯한 sample을 만드는 것이다.

대표적인 흐름은 GAN, DDPM, LDM/Stable Diffusion, DiT, LlamaGen, VAR 같은 계열로 볼 수 있다. 중요한 점은 이 level의 모델이 "아무것도 모른다"는 뜻은 아니다. 오히려 photorealism과 style diversity는 매우 강할 수 있다. 다만 control interface가 약하기 때문에, 위치, 관계, identity, causal state를 사용자가 정밀하게 지정하기 어렵다.

GAN 계열은 빠른 inference와 latent control이 장점이지만, stability와 scaling에서 어려움이 있었다. Diffusion 계열은 denoising objective를 통해 품질과 diversity를 크게 끌어올렸다. 기본적인 diffusion training objective는 다음처럼 볼 수 있다.

$$
L_{diff}(\theta) = E_{x,\epsilon,t}[||\epsilon - \epsilon_\theta(x_t,t,c)||_2^2]
$$

여기서 핵심은 noised sample $x_t$에서 noise 또는 score를 예측해 reverse process를 학습하는 것이다. L1에서는 이 objective가 image distribution을 잘 복원하는 데 집중한다.

### 2) L2 Conditional Generation

Conditional Generation은 explicit condition을 generation process에 넣는다. depth map, edge map, pose, bounding box, reference image, identity embedding, segmentation map 같은 조건이 들어올 수 있다. 이 level의 핵심은 compositional controllability다.

대표적인 방법은 ControlNet, IP-Adapter, GLIGEN, SD3 계열로 볼 수 있다. 여기서 중요한 설계는 condition을 어떻게 추출하고, backbone에 어떻게 주입할지다.

condition module은 대략 두 부분으로 나뉜다.

| Component | Role | Example |
| --- | --- | --- |
| Feature extraction | external signal을 model feature로 변환 | ControlNet copied backbone, IP-Adapter, VLM feature |
| Feature injection | feature를 generation backbone에 주입 | addition, cross-attention, in-context self-attention, AdaLN, CFG |

L2의 한계는 condition이 있어도 generation이 여전히 single-pass라는 점이다. 즉 한 번의 condition을 따라 그리는 능력은 생기지만, 실패한 결과를 보고 다시 계획하거나, multi-turn state를 external memory로 유지하는 능력은 제한적이다.

### 3) L3 In-Context Generation

In-Context Generation은 reference image, edit history, multiple examples, visual memory를 한 번의 context 안에 넣고 생성한다. 이 level의 핵심은 contextual coherence다. 모델은 단순히 prompt를 따르는 것이 아니라, 주어진 visual context와 history를 해석해야 한다.

예를 들어 character identity를 유지한 채 여러 장면을 만들거나, 이전 edit 결과를 바탕으로 다음 edit를 수행하거나, 특정 style/persona를 계속 유지하는 작업이 여기에 들어간다. 논문은 SEED-Data-Edit, ImgEdit, StoryMaker, Visual Persona 같은 흐름을 예로 든다.

하지만 L3의 memory는 주로 input context에 들어 있다. 외부 planner나 verifier가 state를 관리하는 것은 아니다. 그래서 context가 길어지고 edit가 반복되면 silent drift가 생기기 쉽다. 초반에 정한 얼굴, 의상, 배경, 물체 관계가 점진적으로 바뀌는데, 모델은 이를 명시적으로 detect하지 못할 수 있다.

L3는 현재 많은 multimodal generation model이 가장 치열하게 경쟁하는 영역이다. 사용자는 점점 단발 이미지보다 "이 캐릭터를 유지하면서 여러 장면을 만들어줘", "이 로고는 유지하고 배경만 바꿔줘", "방금 결과에서 왼쪽 물체만 수정해줘" 같은 요청을 한다. 이때 memory와 edit locality가 핵심 품질 조건이 된다.

### 4) L4 Agentic Generation

Agentic Generation에서는 generation이 하나의 action이 된다. 시스템은 생성하고, 관찰하고, 평가하고, 다음 action을 결정한다. 즉 visual generator는 더 이상 passive renderer가 아니라 closed-loop system 안의 actuator가 된다.

이 level에서는 다음 구성 요소가 중요해진다.

- planner: task를 subgoal로 나눈다.
- generator: visual candidate를 만든다.
- verifier: output이 constraint를 만족하는지 평가한다.
- refiner: failure를 수정하기 위한 next action을 만든다.
- toolset: segmentation, OCR, retrieval, physics checker, renderer, editor 등을 호출한다.

논문은 closed-source frontier model의 multi-turn editing과 structured content robustness를 이런 L4 stack으로 해석한다. 물론 closed model 내부를 직접 알 수는 없으므로 이 부분은 speculative reading에 가깝다. 하지만 system design 관점에서는 설득력이 있다. 한 번에 완벽한 이미지를 만들기보다, 여러 번 보고 고치는 loop가 복잡한 visual task에서 더 자연스럽기 때문이다.

Agentic Generation의 핵심은 self-correction이다. 좋은 generator라도 한 번의 sample에서 topology나 text rendering을 틀릴 수 있다. 하지만 verifier가 오류를 잡고, refiner가 수정 instruction을 만들고, generator가 다시 실행된다면 system-level capability는 올라갈 수 있다.

### 5) L5 World-Modeling Generation

World-Modeling Generation은 논문에서 future level에 가깝다. 이 level의 목표는 model이 stable dynamics, physics, intervention effect를 internalize해 world simulator처럼 동작하는 것이다.

대표적인 방향은 Genie 2, GameNGen, Oasis, UniSim, GAIA-1 같은 interactive world model 계열로 볼 수 있다. 여기서 generation은 image 하나를 만드는 것이 아니라, action이나 intervention에 따라 world state가 어떻게 변하는지 예측하는 문제가 된다.

이를 단순히 conditional video generation으로 보면 부족하다. 중요한 것은 causal faithfulness다. 예를 들어 object를 밀면 어디로 움직이는지, 물체가 가려졌다가 다시 나타날 때 identity가 유지되는지, agent action이 environment state를 어떻게 바꾸는지까지 일관되어야 한다.

L5는 아직 해결된 문제가 아니다. 그러나 이 level을 명시적으로 두는 것은 중요하다. 그래야 video generation, embodied AI, robotics, game simulation, interactive media를 같은 장기 목표 아래에서 비교할 수 있다.

### 6) Generative paradigm: diffusion, flow, AR, hybrid

논문은 visual generation의 backbone evolution도 capability 관점에서 정리한다.

Flow Matching과 Rectified Flow는 최근 visual generation에서 매우 중요한 driver다. 기본적인 flow matching objective는 다음처럼 쓸 수 있다.

$$
L_{FM}(\theta) = E_{x_0,x_1,t}[||v_\theta(x_t,t,c) - (x_1 - x_0)||_2^2]
$$

이 접근은 noise에서 data로 가는 velocity field를 직접 학습한다. diffusion과 비교해 더 straight한 path, simulation-free training, fewer-step inference와 연결될 수 있다. SD3 계열이나 최신 DiT/flow 기반 image generator를 이해할 때 중요한 축이다.

Autoregressive model은 visual token을 순차적으로 생성한다.

$$
p_\theta(x_{1:T}|c) = \prod_{t=1}^{T} p_\theta(x_t|x_{1:t-1},c)
$$

AR approach는 planning, token-level control, unified multimodal sequence와 잘 맞는다. LlamaGen, VAR, Chameleon, Emu3 같은 흐름이 여기에 들어간다. 다만 high-fidelity dense rendering에서는 diffusion/flow 계열의 장점이 여전히 크다.

그래서 최근에는 hybrid approach가 자연스럽다. AR이 semantic plan이나 discrete intermediate representation을 만들고, diffusion/flow가 final visual rendering을 담당한다.

$$
p_\theta(x|c) = \int p_\phi(z|c)p_\psi(x|z,c)dz
$$

여기서 $z$는 layout, semantic plan, visual token, scene representation 같은 intermediate plan으로 볼 수 있다. 이 factorization은 agentic generation이나 world modeling으로 갈수록 더 중요해진다. 이유는 모델이 먼저 생각하거나 계획한 뒤 render해야 하는 task가 늘어나기 때문이다.

### 7) Multimodal fusion: late fusion에서 early fusion으로

논문이 중요하게 보는 또 하나의 축은 multimodal fusion이다. 예전 LDM 계열에서는 text encoder와 image generator의 결합이 상대적으로 late fusion에 가까웠다. 이후 MM-DiT, Bridge, Janus-Pro 계열은 text/image stream을 더 적극적으로 섞고, Chameleon, Emu3, BAGEL, OmniMamba 같은 흐름은 shared parameter와 unified multimodal token stream으로 간다.

이 변화는 단순 architecture 취향이 아니다. L3/L4 capability를 위해서는 input image, reference, text instruction, edit history, generated intermediate state가 일찍부터 같은 reasoning space에서 상호작용해야 한다. 즉 early fusion은 visual generation을 multimodal reasoning problem으로 바꾸는 방향이다.

# 4. Training / Data / Recipe

## 4-1. Data

이 논문은 특정 dataset을 새로 만들기보다, 최신 visual generation system들이 어떤 data engine을 쓰는지 정리한다. 핵심은 data quality가 단순 aesthetic filtering보다 훨씬 넓어졌다는 점이다.

주요 data recipe는 다음과 같이 볼 수 있다.

| Data stage | Goal | Typical mechanism |
| --- | --- | --- |
| Filtering | low-quality sample 제거 | waterfall filtering, defect-aware filtering, active curation |
| Relabeling | image-text alignment 강화 | VLM captioning, structured JSON metadata, Visual CoT caption |
| Balancing | semantic/domain coverage 확보 | category balancing, text-heavy sample staging |
| Curriculum | resolution/task 난이도 조절 | low-to-high resolution, editing/multi-view/spatial task staging |
| Synthetic distillation | scarce capability data 보강 | stronger VLM/generator로 synthetic instruction/edit pair 생성 |

이 부분이 중요한 이유는 open/closed gap과 직접 연결된다. 모델 architecture가 공개되어도, 어떤 데이터를 어떤 순서로 걸러서 어떤 caption으로 다시 붙였는지는 공개되지 않는 경우가 많다. visual generation에서는 이 data engine이 capability를 크게 좌우한다.

특히 structured image, text rendering, diagram generation, domain-specific visual reasoning에서는 일반 caption보다 더 dense한 annotation이 필요하다. 예를 들어 단순히 "a subway map"이라고 caption을 붙이는 것과, node, line, transfer, label, topology constraint를 structured하게 설명하는 것은 완전히 다르다.

## 4-2. Training strategy

논문이 정리하는 training pipeline은 대략 다음 흐름으로 볼 수 있다.

1. **Pre-training**
   - large-scale noisy data에서 broad visual prior를 학습한다.
   - data density, caption quality, resolution curriculum이 중요하다.

2. **Continued Training, CT**
   - curated high-quality data로 production resolution, editing, multi-view, spatially grounded capability를 강화한다.
   - base distribution matching에서 controllable generation으로 넘어가는 bridge phase로 볼 수 있다.

3. **Supervised Fine-Tuning, SFT**
   - instruction following과 task-specific behavior를 강화한다.
   - edit instruction, structured prompt, reference adherence 같은 데이터를 사용한다.

4. **Preference optimization / RL**
   - human preference, reward model, DPO, GRPO 등을 통해 aesthetic, semantic alignment, compositional correctness를 개선한다.
   - L4로 갈수록 verifier와 reward signal이 중요해진다.

5. **Inference acceleration**
   - fewer-step sampling, distillation, pruning, caching, quantization으로 deployment cost를 줄인다.

이를 압축하면 다음과 같다.

$$
\text{PT} \to \text{CT} \to \text{SFT} \to \text{Preference/RL} \to \text{Acceleration}
$$

여기서 CT가 특히 흥미롭다. LLM에서도 midtraining이나 continued pretraining이 특정 capability를 키우는 bridge 역할을 하듯, visual generator에서도 CT는 base generative prior와 task-aligned behavior 사이를 잇는다. 단순 SFT만으로는 model이 spatial grounding, edit locality, multi-view consistency를 충분히 얻기 어렵기 때문이다.

## 4-3. Engineering notes

이 논문에서 실무적으로 가져갈 engineering note는 다섯 가지다.

1. **caption은 더 이상 짧은 문장이 아니다**
   - VLM relabeling, structured metadata, Visual CoT caption이 중요해진다.
   - generation model이 무엇을 그려야 하는지뿐 아니라, 관계와 constraint까지 배워야 하기 때문이다.

2. **condition interface를 task별로 분리하지 말고 unified flow로 봐야 한다**
   - text-to-image, editing, reference-based generation, inpainting, multi-view generation이 따로따로 있으면 system integration이 어렵다.
   - 최신 model은 점점 unified generation/editing architecture로 간다.

3. **post-training은 visual quality만 맞추는 단계가 아니다**
   - compositional correctness, layout adherence, identity consistency, text fidelity, edit locality를 reward나 preference signal로 넣어야 한다.

4. **verification loop가 capability를 만든다**
   - L4 agentic generation에서는 generator 하나보다 verifier와 refiner가 중요해질 수 있다.
   - OCR, segmentation, detector, geometry estimator, VLM judge, physics checker 같은 external tool이 generation quality의 일부가 된다.

5. **inference cost는 capability 설계와 같이 봐야 한다**
   - closed-loop generation은 여러 번 sample하고 evaluate하므로 비용이 커진다.
   - sampling acceleration과 distillation은 단순 serving optimization이 아니라 agentic workflow를 가능하게 하는 조건이다.

# 5. Evaluation

## 5-1. Main results

이 논문은 새 모델의 benchmark score를 제시하는 논문이 아니다. 따라서 "어떤 모델이 몇 점 올랐다"가 main result가 아니다. main result는 current evaluation이 visual generation의 중요한 failure mode를 충분히 측정하지 못한다는 진단이다.

논문이 강조하는 evaluation gap은 다음과 같다.

| Evaluation target | 기존 metric이 잘 보는 것 | 놓치기 쉬운 것 |
| --- | --- | --- |
| Perceptual quality | sharpness, realism, aesthetic | 구조적 오류 |
| Prompt alignment | keyword coverage | relation, counting, binding |
| Text rendering | visible text quality | semantic consistency, layout logic |
| Editing | single-turn instruction following | multi-turn drift, edit locality |
| Video quality | frame-level quality | object permanence, causal dynamics |
| World modeling | plausible motion | intervention consistency, physics |

논문은 이를 보완하기 위해 benchmark review뿐 아니라 in-the-wild stress tests와 expert-constrained case studies를 제안한다. 예시는 다음과 같다.

- jigsaw reconstruction
- metro-map topology
- coordinate maps
- physical causality
- multi-turn editing
- image text reasoning
- low-level vision task consistency
- cross-disciplinary application constraints

이런 task는 일반적인 aesthetic metric으로는 잘 잡히지 않는다. 예를 들어 metro map은 예쁘게 보이는 것보다 node와 edge의 topology가 맞는지가 중요하다. physical causality task에서는 frame이 자연스럽게 보이는 것보다 intervention 후 state transition이 맞는지가 중요하다.

## 5-2. What really matters in the experiments

이 논문에서 진짜 중요한 evaluation 관점은 세 가지다.

### 1) 예쁜 output은 correctness를 가릴 수 있다

이미지가 고품질일수록 사람은 그 안의 structural error를 덜 의심할 수 있다. 특히 diagram, map, UI, chart, document, scientific visualization에서는 이 문제가 크다. visual generation이 실무에 들어가려면 aesthetic quality보다 correctness가 먼저인 task가 많다.

### 2) single-turn metric은 multi-turn drift를 숨긴다

한 번의 edit는 잘해도, 5번, 10번 edit하면 초기 identity, object attribute, spatial layout이 무너질 수 있다. L3/L4 generation을 평가하려면 누적된 edit history 위에서 state preservation을 봐야 한다.

### 3) world model은 video realism과 다르다

video가 자연스럽게 보여도 world model이라고 부르기에는 부족하다. action이나 intervention에 대해 consistent transition을 예측해야 하고, object permanence와 physics가 유지되어야 한다. 즉 L5 evaluation은 video quality benchmark보다 embodied interaction, counterfactual intervention, long-horizon state tracking에 가까워야 한다.

이 논문의 evaluation message는 꽤 실용적이다. 앞으로 visual generation system을 product에 넣을 때도 "예쁜가"를 보는 QA만으로는 부족하다. domain-specific verifier를 붙여야 한다. OCR verifier, geometry verifier, topology checker, physics constraint checker가 visual generation eval stack의 일부가 될 가능성이 크다.

# 6. Limitations

1. **이 논문은 taxonomy paper이지 new model paper가 아니다.**
   - 따라서 benchmark improvement나 ablation으로 claim을 검증하는 구조는 아니다.
   - 독자는 이 글을 empirical proof보다 roadmap으로 읽어야 한다.

2. **L5 World-Modeling Generation은 아직 aspirational level에 가깝다.**
   - Genie, GameNGen, Oasis 같은 방향이 있지만, physical causality와 intervention consistency를 일반적으로 해결했다고 보기는 어렵다.
   - world model이라는 용어가 너무 넓게 쓰일 위험도 있다.

3. **closed-source frontier 분석은 speculative하다.**
   - Nano Banana, GPT-Image 계열 같은 closed system이 실제로 어떤 내부 stack을 쓰는지는 공개되어 있지 않다.
   - planner, verifier, refiner loop는 plausible한 해석이지만, 확인된 architecture라고 쓰면 안 된다.

4. **5-level taxonomy는 깔끔하지만 경계가 항상 명확하지 않다.**
   - 어떤 system은 L2 condition과 L3 context, L4 tool loop를 동시에 갖는다.
   - 따라서 level은 discrete class라기보다 capability axis로 읽는 편이 안전하다.

5. **evaluation stress test는 중요하지만 standardized benchmark는 아니다.**
   - jigsaw, map, causality probe는 failure를 잘 보여주지만, reproducible scoring protocol이 필요하다.
   - 특히 expert-constrained case study는 해석력이 높지만 scale과 자동화가 어렵다.

6. **data curation과 post-training recipe가 너무 broad하게 정리될 수 있다.**
   - survey 특성상 각 system의 실제 data mixture, filtering threshold, reward model detail은 대부분 공개되지 않는다.
   - 따라서 실무 재현 관점에서는 여전히 많은 부분이 black box다.

7. **agentic generation은 model capability와 system wrapper를 섞어 말할 수 있다.**
   - verifier와 tool loop가 좋아서 system이 잘하는 것인지, generator 자체가 잘하는 것인지 분리하기 어렵다.
   - 앞으로 evaluation은 single model capability와 agent system capability를 따로 보고해야 한다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문은 visual generation을 image/video model 논문으로만 읽지 않고, multimodal system design 관점으로 읽게 만든다. 특히 document AI, OCR, visual reasoning, UI agent, video generation, 3D/world model을 모두 다루는 입장에서는 taxonomy가 꽤 유용하다.

가장 중요하게 본 포인트는 세 가지다.

첫째, **visual generation의 병목이 rendering에서 verification으로 이동한다**는 점이다. 예쁜 이미지를 만들 수 있는 모델은 많아지고 있다. 하지만 map, document, chart, UI, scientific figure처럼 correctness가 중요한 영역에서는 verifier 없이는 production quality를 보장하기 어렵다.

둘째, **context와 state가 generation quality의 핵심이 된다**는 점이다. 같은 character, 같은 object, 같은 layout을 유지하면서 multi-turn으로 수정하려면 단순 prompt adherence보다 memory architecture가 중요하다. 이는 LLM agent memory 문제와도 닮아 있다.

셋째, **world model은 video generator의 자연스러운 다음 단계지만, evaluation은 완전히 달라져야 한다**는 점이다. video가 자연스러워 보이는 것과 controllable simulator가 되는 것은 다르다. embodied AI나 robotics에 쓰려면 intervention consistency와 causal state tracking이 필요하다.

## 7-2. Reuse potential

이 논문에서 바로 재사용할 수 있는 것은 model component보다 평가와 설계 checklist다.

1. **generation task를 level로 분류하기**
   - 지금 만들고 있는 task가 L1인지, L2인지, L3인지, L4인지 먼저 정한다.
   - 그러면 필요한 component가 달라진다. L2는 condition interface, L3는 memory, L4는 verifier/tool loop가 필요하다.

2. **visual QA에 verifier를 붙이기**
   - OCR, object count, topology, geometry, edit locality, identity consistency를 별도 verifier로 본다.
   - aesthetic judge 하나로 모든 것을 평가하지 않는다.

3. **multi-turn drift를 별도 metric으로 보기**
   - single edit success만 보지 말고, 누적 edit 후 초기 constraint가 살아 있는지 본다.

4. **data curation을 capability별로 설계하기**
   - text rendering, structured diagram, reference preservation, physical dynamics는 서로 다른 data engine이 필요하다.
   - 하나의 broad captioning recipe로 모든 capability를 얻기 어렵다.

5. **closed-loop generation을 default architecture 후보로 보기**
   - generator 단독 개선보다 planner, verifier, refiner, toolset을 붙이는 것이 더 빠른 practical path일 수 있다.

## 7-3. Follow-up papers

- ControlNet
- IP-Adapter
- GLIGEN
- Stable Diffusion 3
- DiT
- LlamaGen
- VAR
- Chameleon
- Emu3
- JanusFlow
- Transfusion
- Qwen-Image
- Seedream
- HunyuanImage
- Wan-Image
- Genie 2
- GameNGen
- Oasis
- UniSim

# 8. Summary

- 이 논문은 visual generation을 Atomic, Conditional, In-Context, Agentic, World-Modeling이라는 5단계 capability ladder로 정리한다.
- 핵심 메시지는 photorealism 이후의 병목이 structure, memory, feedback loop, causality라는 점이다.
- 최신 training stack은 PT, CT, SFT, RL, reward modeling, data curation, synthetic annotation, sampling acceleration이 결합된 full pipeline으로 가고 있다.
- current benchmark는 perceptual quality를 잘 보지만 topology, state drift, physical causality 같은 failure를 놓치기 쉽다.
- 이 논문은 새 모델 논문이라기보다, visual generation system을 설계하고 평가할 때 쓸 수 있는 capability-centered roadmap으로 읽는 편이 맞다.
