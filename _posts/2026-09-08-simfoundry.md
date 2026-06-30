---
layout: single
title: "SimFoundry: Modular and Automated Scene Generation for Policy Learning and Evaluation Review"
categories: Study-concept
tag: [SimFoundry, Robotics, Simulation, DigitalTwin, PolicyLearning]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2606.28276)

[Project page](https://research.nvidia.com/labs/gear/simfoundry/)

SimFoundry는 "real-to-sim reconstruction system"으로만 보면 부족하다. 이 논문이 실제로 겨냥하는 문제는 더 넓다. **한 번 촬영한 real-world video를 sim-ready digital twin으로 만들고, 거기서 object, scene, task variation을 자동 생성해 policy training과 evaluation에 쓸 수 있는가**다.

Robot policy learning에서 simulation은 오래된 도구지만, 실제 scene을 simulation으로 옮기는 일은 여전히 느리고 비싸다. Object mesh를 만들고, pose를 맞추고, physical parameter를 주고, background를 재구성하고, task initial condition을 정의하고, simulator에서 sanity check까지 해야 한다. 대부분 hand-engineering이 많이 들어간다.

SimFoundry는 이를 modular pipeline으로 자동화하려 한다. Input은 real-world scene video다. Pipeline은 foreground object와 background를 분리하고, per-object masks, depth, 3D mesh, pose, physical parameters를 구성해 interactive simulation scene을 만든다. 여기서 끝나지 않는다. SimFoundry는 reconstructed digital twin을 바탕으로 object cousin, scene cousin, task cousin을 생성한다. 즉 원래 scene을 그대로 복제하는 것을 넘어, affordance-preserving variation을 만들어 policy generalization data로 쓴다.

> 한 줄 요약: SimFoundry는 single real-world video에서 sim-ready digital twin을 만들고, object, scene, task cousin을 자동 생성해 robot policy training과 evaluation에 사용하는 modular real-to-sim system이며, 7개 manipulation task와 5개 policy architecture에서 sim evaluation이 real performance를 strongly predict한다고 보고한다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Real-to-sim을 manual asset authoring이 아니라 foundation-model-assisted modular pipeline으로 다시 설계한다.
- Digital twin reconstruction과 digital cousin generation을 하나의 policy learning pipeline으로 연결한다.
- Simulation을 단순 training data source가 아니라 real-world policy evaluator로 검증한다.
- 7 manipulation tasks, 5 policy architectures에서 sim-real ranking correlation을 보고해 evaluation utility를 강조한다.
- Object, scene, task cousins가 각각 zero-shot real-world transfer를 개선할 수 있음을 보여준다.
- Modular component swap design이라 future foundation model improvement를 pipeline에 흡수하기 쉽다.

이 글에서는 SimFoundry를 "scene generation paper"보다, **real-world scene video를 policy learning/evaluation용 simulation factory로 바꾸려는 robotics infrastructure paper**로 읽는다.

# 1. Problem Setting

## 1-1. Problem definition

Robot policy learning은 real-world data와 evaluation 비용이 높다. Simulation은 더 저렴하고 scalable하지만, simulation scene이 real task distribution과 맞지 않으면 policy transfer가 약하다.

Goal은 real scene video $v$에서 simulation environment $\mathcal{S}$를 자동으로 만드는 것이다.

$$
\mathcal{S}
=
F(v)
$$

그리고 이 simulation에서 policy $\pi$를 training하거나 evaluation한다.

$$
\pi
=
\mathrm{Train}(\mathcal{S})
$$

또는

$$
\hat{R}_{sim}(\pi)
=
\mathrm{Eval}(\pi,\mathcal{S})
$$

유용한 system이 되려면 두 조건을 만족해야 한다.

1. Real-to-sim fidelity
   - Reconstructed scene은 task와 관련된 object, layout, geometry, interaction을 보존해야 한다.

2. Diversity generation
   - Simulation은 한 scene을 복사하는 데 그치면 안 된다. Affordance를 보존하면서 training distribution을 넓히는 variation을 만들어야 한다.

SimFoundry는 exact reconstruction을 digital twin, variation을 digital cousin이라고 부른다.

## 1-2. Why previous approaches are insufficient

### 1) Manual simulation authoring

Manual scene authoring은 정확할 수 있지만 느리다. 많은 real scene, object, task로 scale하기 어렵다.

### 2) Pure domain randomization

Domain randomization은 diversity를 만들지만 real scene distribution과 맞지 않을 수 있다. Random variation은 affordance나 task semantics를 깨뜨릴 수 있다.

### 3) Zero-shot 3D reconstruction

3D reconstruction model은 asset을 만들 수 있지만, policy training에는 physics-ready simulation이 필요하다. Object mesh, pose, background, physical parameter, task state, simulator sanity check가 모두 필요하다.

### 4) Sim evaluation without real correlation

Simulator는 sim score가 real performance를 예측할 때만 evaluation에 유용하다. 그렇지 않으면 sim에서의 policy selection이 misleading해질 수 있다.

SimFoundry는 real scene을 reconstruct하고, simulation evaluation이 real-world performance와 correlate하는지 확인함으로써 이 문제를 다룬다.

# 2. Core Idea

## 2-1. Main contribution

SimFoundry의 핵심 아이디어는 세 가지다.

1. **Modular real-to-sim scene construction**
   - Input video에서 physical scene information을 추출한다.
   - Object mesh와 background를 생성한다.
   - Physical parameter를 포함해 simulation scene을 compile한다.
   - Simulator에서 sanity check를 수행한다.

2. **Digital cousin generation**
   - Object cousin: affordance를 보존하는 alternative object
   - Scene cousin: alternative scene layout/configuration
   - Task cousin: 같은 scene family 안의 related task 또는 interaction

3. **Policy learning과 evaluation loop**
   - Generated simulation으로 policy를 evaluate한다.
   - Generated cousin으로 zero-shot real-world variant에 transfer되는 policy를 학습한다.

## 2-2. Design intuition

SimFoundry는 다음 robotics intuition 위에 세워져 있다.

```text
하나의 digital twin은 fidelity를 높인다.
Digital cousin family는 generalization을 높인다.
```

하나의 exact reconstruction에서만 학습한 policy는 특정 object pose나 layout에 overfit될 수 있다. 반대로 random synthetic scene에서 학습한 policy는 real task로 transfer되지 않을 수 있다. Digital cousin은 그 중간 지점이다. Scene을 변화시키되 affordance와 task semantics를 보존한다.

그래서 SimFoundry는 단순 reconstruction이 아니다. Generative part가 핵심이다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Real video에서 policy learning/evaluation용 sim-ready scene 구축 |
| Input | Single real-world scene video |
| Output | Digital twin과 digital cousin |
| Scene representation | 3D Gaussian Splat background와 textured object mesh를 결합한 hybrid scene |
| Main modules | Physical scene extraction, sim scene generation, cousin augmentation |
| Cousin types | Object, scene, task |
| Evaluation | 7개 manipulation task와 5개 policy architecture |
| Main reported sim-real metric | Mean Pearson correlation 0.911과 mean MMRV 0.018 |

## 3-2. Module breakdown

### 1) Physical scene extraction

Pipeline은 real video에서 object별 relevant information을 추출한다.

여기에는 다음이 포함된다.

- segmentation masks
- depth
- object instances
- pose estimates
- background separation

목표는 scene을 simulator에서 표현 가능한 object component와 background component로 분해하는 것이다.

### 2) Object mesh generation

SimFoundry는 2D observation에서 2D-to-3D generation model을 사용해 textured object mesh를 만든다. 이 mesh는 단순 rendering용이 아니라 simulation에 적합해야 한다.

Project page는 SimFoundry가 demo에서 SAM3D보다 더 정확한 object mesh와 pose를 복원한다고 강조한다. 특히 occluded cluttered scene에서 그렇다. 정확한 quantitative value는 paper table에서 확인해야 한다.

### 3) Background reconstruction

Foreground object를 제거해 background-only video를 만들고, 이를 refine한 뒤 3D Gaussian Splat background를 학습하는 데 사용한다.

이 hybrid scene representation은 다음을 결합한다.

- visual fidelity를 위한 3DGS background
- interaction과 physics를 위한 textured object mesh

### 4) Physical parameter annotation과 sanity check

Simulation-ready scene에는 physical metadata가 필요하다.

- collision geometry
- mass or proxy parameter
- articulated state if relevant
- object pose
- task affordance
- simulator consistency

SimFoundry는 scene을 compile한 뒤 physics simulator에서 전체 configuration을 sanity check한다.

### 5) Digital cousins

SimFoundry는 세 가지 variation axis를 지원한다.

| Cousin type | What changes | Purpose |
| --- | --- | --- |
| Object cousin | Affordance를 보존하면서 object appearance나 shape 변경 | Novel object로 generalize |
| Scene cousin | Object configuration과 layout 변경 | New scene으로 generalize |
| Task cousin | Related viable interaction 변경 | Related task로 generalize |

목표는 arbitrary randomization이 아니다. Affordance-preserving variation이다.

# 4. Training / Data / Recipe

## 4-1. Data와 tasks

논문은 다음 유형의 real-world manipulation task를 평가한다.

- multi-step manipulation
- articulated object interaction
- bimanual interaction
- single-arm DROID-style settings
- dual-arm YAM-style settings

Project page는 Clear Table, Marker in Cup, Store Marker, Serve Fruits, Stack Dishware 같은 example task를 제시한다.

## 4-2. Policy evaluation과 training recipe

SimFoundry는 두 가지 use case를 지원한다.

### 1) Sim evaluation

Generated simulation 안에서 real-world policy를 평가하고, sim performance를 real performance와 비교한다.

논문은 7 tasks와 5 policy architectures에서 simulation evaluation이 real-world performance를 강하게 예측한다고 보고한다. Mean Pearson correlation은 0.911이고 mean maximum ranking violation은 0.018이다.

### 2) Sim training

SimFoundry-generated data로 policy를 train 또는 fine-tune한다. 논문은 object, scene, task cousin이 real-world zero-shot task success를 평균적으로 각각 17%, 21%, 40% 개선한다고 보고한다.

## 4-3. Engineering notes

1. **Digital twin에서 멈추지 않는다**
   - Training distribution을 넓히는 핵심은 digital cousin generation이다.

2. **Affordance preservation이 key constraint다**
   - Random object replacement는 task meaning을 깨뜨릴 수 있다.

3. **Sim-real ranking correlation을 평가한다**
   - Sim accuracy는 real performance를 예측할 때에만 유용하다.

4. **Modular component를 사용한다**
   - Segmentation, depth, 3D generation, simulation sanity check는 독립적으로 개선될 수 있다.

5. **Hybrid representation은 실용적이다**
   - Scene realism에는 visual background representation을 쓰고, physical interaction에는 object mesh를 사용한다.

# 5. Evaluation

## 5-1. Real-to-sim evaluation

논문은 강한 sim-real correlation을 보고한다.

| Metric | Reported value |
| --- | ---: |
| Mean Pearson correlation | 0.911 |
| Mean maximum ranking violation | 0.018 |
| Tasks | 7 manipulation tasks |
| Policy architectures | 5 |

이 결과는 SimFoundry가 training data generator일 뿐 아니라 evaluation environment로도 유효할 수 있음을 보여준다는 점에서 중요하다.

## 5-2. Sim-to-real policy transfer

논문은 SimFoundry data로 학습한 policy가 multi-step manipulation, articulated object interaction, bimanual interaction을 포함하는 real task로 zero-shot transfer된다고 보고한다.

보고된 cousin별 average task success improvement는 다음과 같다.

| Cousin type | Average improvement |
| --- | ---: |
| Object cousins | 17% |
| Scene cousins | 21% |
| Task cousins | 40% |

이 수치들은 digital cousin이 generalization을 개선한다는 아이디어를 뒷받침한다.

## 5-3. Project page examples

Project page는 추가로 다음 예시도 보고한다.

- SimFoundry scene은 simulation 안에서 real-world policy를 평가할 수 있고, mean Pearson correlation 0.911을 보인다.
- SimFoundry data는 13개 task에서 real-world VLA performance를 28%에서 46%로 높인다.
- 7개 held-out task에서는 결과가 0%에서 29%로 개선된다.

이들은 유용한 signal이지만, final blog post에서는 project page demo number와 core abstract claim을 구분해서 읽어야 한다.

## 5-4. What really matters in the experiments

### 1) Real performance와의 correlation

Generated simulator는 real ranking을 예측할 때 유용하다. Mean Pearson correlation과 MMRV는 visual reconstruction만 보는 것보다 더 중요하다.

### 2) Cousin을 통한 generalization

Digital cousin 아이디어는 distribution expansion을 직접 겨냥한다. 단순 scene reconstruction이 아니다.

### 3) Modularity가 중요하다

Pipeline이 여러 foundation model에 의존하기 때문에 modular replacement가 strength다. Segmentation, 3D generation, physics annotation이 좋아지면 전체 system도 개선될 수 있다.

### 4) Evaluation은 여전히 어렵다

Sim-to-real success는 policy class, task, robot, object, physical parameter accuracy에 의존한다. 하나의 숫자만으로 general simulation validity가 증명되지는 않는다.

# 6. Limitations

1. **Pipeline complexity가 크다**
   - SimFoundry는 segmentation, depth, 2D-to-3D generation, background reconstruction, physics annotation, simulator check 등 많은 component에 의존한다.

2. **Physical parameter uncertainty가 있다**
   - Mesh와 pose가 좋아도 friction, mass, compliance, articulation은 여전히 근사값일 수 있다.

3. **Affordance preservation은 보장되지 않는다**
   - Digital cousin은 task semantics를 보존해야 하지만, generation은 그럴듯해 보여도 부적절한 variant를 만들 수 있다.

4. **Task scope가 제한적이다**
   - 7개 manipulation task와 5개 policy architecture는 강한 평가지만 exhaustive하지는 않다.

5. **Real-to-sim은 여전히 real video가 필요하다**
   - Pipeline은 pure text-to-sim이 아니라 captured real-world video에서 시작한다.

6. **Simulator bias가 남아 있다**
   - Policy는 generated simulator artifact에 여전히 overfit될 수 있다.

7. **Evaluation hardware와 policy가 중요하다**
   - 결과는 robot platform과 policy architecture에 의존할 수 있다.

8. **Code와 asset release를 확인해야 한다**
   - Reproducibility는 released pipeline, asset, simulator setup에 의존한다.

9. **Scene reconstruction error가 누적될 수 있다**
   - Object mesh, pose, background, physics parameter error가 서로 상호작용하며 누적될 수 있다.

10. **Safety 검증이 필요하다**
    - Zero-shot real-world transfer도 safety constraint 아래에서 test되어야 한다.

# 7. My Take

## 7-1. Why this matters for my work

SimFoundry의 핵심은 "real-to-sim reconstruction"보다, **simulation environment를 policy training과 evaluation factory로 자동 생성하려는 점**이다.

Robot learning에는 data bottleneck이 있다. Real data는 비싸고 hand-built simulation은 scale되지 않는다. SimFoundry는 real video를 reusable simulation asset으로 바꾸고, 이를 cousin으로 증식시키려 한다.

이는 scalable embodied AI에 필요한 infrastructure에 가깝다.

## 7-2. Reuse potential

### Policy evaluation

모든 policy를 real world에서 시험하기 전에, generated digital twin에서 sim evaluation을 수행하고 ranking이 real performance와 correlate하는지 확인할 수 있다.

### Policy training

Object, scene, task cousin을 사용해 하나의 reconstructed scene을 넘어 generalize하는 policy를 학습할 수 있다.

### Benchmark generation

SimFoundry-like pipeline은 sim controllability를 유지하면서 real scene에서 benchmark suite를 만들 수 있다.

### Digital twin workflows

Manufacturing, household robotics, warehouse task에는 모두 더 빠른 scene-to-sim conversion이 필요하다.

## 7-3. Production considerations

- Task family별 sim-real correlation을 측정한다.
- Safety-critical task에서는 physical parameter를 수동 검증한다.
- Arbitrary augmentation이 아니라 affordance를 보존하는 cousin을 사용한다.
- Real robot evaluation을 final gate로 유지한다.
- 각 asset을 어떤 foundation model이 생성했는지 추적한다.
- Scene, object, task, simulator parameter를 모두 versioning한다.

## 7-4. Follow-up papers

- PolaRiS
- DROID
- RoboTwin
- PhysisForcing
- Cosmos world models
- World Action Models survey
- PointWorld
- DreamGen
- FLARE
- Sim2Real and domain randomization literature

# 8. Summary

- SimFoundry는 real-world video를 sim-ready digital twin으로 바꾼다.
- Policy generalization을 위해 object, scene, task cousin을 생성한다.
- 3DGS background와 textured object mesh를 결합한 hybrid scene representation을 사용한다.
- Simulation evaluation은 real-world policy performance와 강한 correlation을 보고한다.
- 핵심 challenge는 automatic reconstruction과 variation 과정에서 physical correctness와 affordance correctness를 보존하는 것이다.
