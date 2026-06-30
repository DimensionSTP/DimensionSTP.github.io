---
layout: single
title: "NatureBench: Can Coding Agents Match the Published SOTA of Nature-Family Papers? Review"
categories: Study-concept
tag: [NatureBench, CodingAgent, AIForScience, Benchmark, NatureGym]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[논문 링크](https://arxiv.org/abs/2606.24530)

[코드 링크](https://github.com/FrontisAI/NatureBench)

[리더보드](https://frontisai.github.io/NatureBench/)

NatureBench는 "AI coding agent가 논문을 재현할 수 있는가"보다 훨씬 더 어려운 질문을 던진다. **Agent가 Nature-family paper에서 나온 real scientific task를 받아, source method 없이 published SOTA를 match or surpass할 수 있는가**다.

기존 paper-based benchmark는 대체로 reproduction을 본다. Agent에게 source paper or implementation을 주고, 방법을 이해해 다시 구현하는지 평가한다. 이는 중요하지만 scientific discovery와는 다르다. Discovery setting에서는 agent가 original method를 모른 채, dataset, task brief, evaluator, held-out test로부터 좋은 방법을 찾아야 한다. 즉 "paper를 따라 하는가"가 아니라 "같은 scientific problem에서 더 나은 method를 찾을 수 있는가"를 묻는다.

NatureBench는 이 문제를 NatureGym pipeline으로 푼다. Nature-family publications에서 ML task, dataset, metric, SOTA anchor를 추출하고, source method를 firewall로 제거한 뒤, containerized task package를 만든다. Final benchmark는 90 tasks, 333 evaluation instances, six scientific domains로 구성된다. Evaluation은 web-search-disabled protocol에서 ten frontier agent configurations를 평가하고, strongest model도 $g>0.1$ criterion에서 only 17.8% tasks를 surpass했다고 보고한다.

> 한 줄 요약: NatureBench는 Nature-family publications에서 90 scientific ML tasks를 containerized package로 만들고, coding agents가 source method 없이 published SOTA를 match or surpass할 수 있는지 평가하는 discovery-oriented AI-for-science benchmark다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Coding agent evaluation을 reproduction에서 discovery-oriented optimization으로 확장한다.
- NatureGym이 paper, dataset, evaluator, environment를 standard package로 바꾸는 pipeline을 제공한다.
- Information firewall을 통해 original method leakage를 줄이고, agent가 method를 새로 찾아야 하는 setting을 만든다.
- Six scientific domains and heterogeneous metrics를 SOTA-normalized relative gap으로 비교한다.
- Strongest model이 SOTA를 surpass하는 비율이 17.8%에 그쳐 current coding agent의 scientific discovery capability gap을 보여준다.
- Success mode가 scientific invention보다 methodological translation에 가깝다는 analysis가 agent research 방향에 중요하다.

# 1. Problem Setting

## 1-1. Problem definition

Scientific task $T$는 dataset $D$, metric $m$, published SOTA score $s^*$, and hidden test set으로 정의된다.

$$
T=(D,m,s^*)
$$

Agent는 task brief and allowed files만 보고 method $A$를 만들어야 한다.

$$
\hat{s}=m(A(D_{\mathrm{test}}),y_{\mathrm{test}})
$$

Benchmark는 $\hat{s}$가 published SOTA $s^*$와 얼마나 가까운지 혹은 이를 넘는지 평가한다.

여기서 핵심은 정보 접근이다. Reproduction setting은 original method를 주거나 강하게 암시한다. NatureBench는 method-specific files and source method details를 제거한다. Agent는 problem, data, evaluator는 보지만 paper의 solution path는 보지 못한다.

따라서 task는 다음에 가깝다.

```text
real scientific dataset에서 competitive method를 발견하라
```

다음과는 다르다.

```text
published method를 다시 구현하라
```

## 1-2. Why previous approaches are insufficient

### 1) Paper reproduction benchmark

PaperBench, CORE-Bench, ReplicationBench 계열 setting은 논문 이해와 구현 능력을 본다. 그러나 agent가 source method 없이 frontier scientific method를 찾을 수 있는지는 별도 질문이다.

### 2) Kaggle or ML-engineering benchmark

Kaggle-style benchmark는 feature engineering, model selection, leaderboard optimization을 볼 수 있다. 하지만 Nature-family task가 요구하는 domain-specific reasoning, scientific tooling, dataset heterogeneity, metric complexity를 충분히 반영하지 못할 수 있다.

### 3) Environment-fragmented tasks

Scientific papers는 dataset, scripts, formats, dependencies, metrics가 모두 다르다. Standardized containerized environment construction 없이는 benchmark 결과가 쉽게 흔들린다.

### 4) Raw score comparison

Scientific tasks는 metric scale이 다르다. AUC, accuracy, RMSE, correlation, domain-specific error 등이 섞이면 raw score average는 의미가 약하다. Cross-task benchmark에는 normalized metric이 필요하다.

# 2. Core Idea

## 2-1. Main contribution

NatureBench는 두 layer로 구성된다.

1. **NatureGym**
   - Nature-family paper를 containerized task package로 변환하는 pipeline이다.
   - Task brief, dataset, held-out test set, automated evaluator, SOTA anchor를 만든다.
   - Source method artifacts를 제거해 discovery setting을 구성한다.
   - Review-gated verify-repair stage를 포함한다.

2. **NatureBench**
   - 90 tasks and 333 evaluation instances로 구성된다.
   - Six scientific domains를 포함한다.
   - Web-search-disabled protocol에서 frontier coding agents를 평가한다.
   - SOTA-normalized relative gap, Match-SOTA, Surpass-SOTA, validity judge를 사용한다.

## 2-2. Design intuition

NatureBench의 design intuition은 다음이다.

```text
Discovery를 평가하려면 method는 숨기고, problem, data, evaluator, SOTA anchor는 보존해야 한다.
```

이것이 paper-sourced discovery benchmark의 핵심이다. Original method가 code, preprocessing files, intermediate outputs, paper text에서 새면 benchmark는 reproduction이 된다. 반대로 evaluator or data reconstruction이 불완전하면 benchmark는 unreliable해진다. NatureGym은 이 두 위험 사이를 조절한다.

| 위험 | NatureGym 설계 |
| --- | --- |
| Method leakage | File-level firewall and source method removal |
| Non-runnable environment | Containerized task package |
| Broken metric | Automated evaluator and calibration |
| Missing data | Dataset acquisition and verification |
| Task ambiguity | Task brief and human-confirmed critical corrections |

# 3. Architecture / Method

## 3-1. Overview

| 항목 | 설명 |
| --- | --- |
| 목표 | Scientific discovery capability of coding agents 평가 |
| Pipeline | NatureGym |
| Benchmark | NatureBench |
| Source | Nature-family publications from 2022 to 2025 |
| Source journals | Ten Nature-family journals |
| Final tasks | 90 |
| Evaluation instances | 333 |
| Domains | Six scientific task domains |
| Protocol | Web-search-disabled, source-method-firewalled |
| 주요 결과 | Strongest model surpasses SOTA on 17.8% tasks under $g>0.1$ |

## 3-2. NatureGym pipeline

NatureGym은 각 task를 세 단계로 만든다.

### 1) Paper filtering

Filtering은 task, evaluation, data 세 차원에서 진행된다.

| Level | 기준 |
| --- | --- |
| Task | Extractable ML task or algorithmic contribution이 있어야 한다. |
| Evaluation | Main metric이 deterministic automated evaluation을 허용해야 한다. |
| Data | Dataset이 publicly accessible and complete해야 한다. |

Pipeline은 ML이 auxiliary인 task, non-computational wet-lab work, pure theory, hardware-only task, physical interaction이 필요한 task를 제외한다.

### 2) Dataset acquisition and verification

Pipeline은 dataset metadata만 보는 것이 아니라 실제 files를 확인한다. 그리고 어떤 file이 task-defining artifact이고 어떤 file이 source method leakage인지 구분한다.

이 file-level firewall이 핵심이다.

- Task-defining files는 유지한다.
- Method-specific preprocessing, intermediate outputs, final outputs, irrelevant files는 제거한다.

### 3) Task package construction

각 task package는 다음을 포함한다.

| 구성요소 | 역할 |
| --- | --- |
| Task brief | Agent가 볼 scientific problem 정의 |
| Dataset | Input data and splits |
| Held-out test set | Evaluation용 hidden or separated data |
| Automated evaluator | Metric 계산 |
| SOTA anchor | Published method의 target score |
| Container | Reproducible environment |

각 단계는 independent review and repair loop를 갖고, critical override는 human-confirmed 방식으로 처리된다.

## 3-3. NatureBench construction

Source corpus는 ten Nature-family journals, 2022-2025 publications에서 가져온다. Final 90-task set은 이 중 six journals and six scientific domains를 커버한다.

Pipeline funnel은 다음과 같다.

| 단계 | 남은 paper/task 수 |
| --- | ---: |
| Initial crawl from 10 Nature-family journals | 5,500 |
| Article-type filter | 2,500 |
| Three-level filtering | 200 |
| Dataset acquisition and verification | 180 |
| Task construction | 160 |
| Final calibration | 90 |

Final benchmark는 cellular omics, protein biology, biomedical modeling, physical modeling, molecular design, relational reasoning 등 다양한 scientific domain을 포함한다고 설명된다.

## 3-4. Evaluation protocol

NatureBench는 SOTA-normalized relative gap $g$를 primary metric으로 사용한다. 정확한 formula는 원문 table and method section에서 재확인해야 한다. 의도는 task-specific score scale을 published SOTA 기준으로 normalize해 heterogeneous tasks를 비교 가능하게 만드는 것이다.

Evaluation category는 다음과 같다.

| 범주 | 의미 |
| --- | --- |
| Match-SOTA | Agent가 published SOTA에 근접한 경우 |
| Surpass-SOTA | Agent가 threshold 이상으로 SOTA를 넘은 경우 |
| Validity judge | Shortcut behavior and invalid output 감지 |
| Completion/validity | Evaluable artifact를 생성했는지 여부 |

Benchmark는 strict web-search-disabled protocol을 사용해 source paper or online code leakage를 줄인다.

# 4. Training / Data / Recipe

## 4-1. Data

NatureBench는 training dataset이 아니라 evaluation benchmark다.

| 항목 | 값 |
| --- | ---: |
| Final tasks | 90 |
| Evaluation instances | 333 |
| Source period | 2022-2025 |
| Source journals | Ten selected Nature-family journals |
| Final domain count | Six scientific domains |

Data curation에서 가장 중요한 것은 method firewall이다. 같은 problem and dataset은 남기되, solution path는 제거해야 한다.

## 4-2. Benchmark calibration

Construction 이후 authors는 evaluation-time quality calibration을 수행한다.

1. Claude Opus 4.6을 base mode로 전체 tasks에 실행한다.
2. Exposed task package defects를 diagnose and repair or drop한다.
3. Reproduction-mode audit에서 agent에게 source paper를 주고 method reproduction이 가능한지 본다.
4. Claude Opus 4.6 and DeepSeek-V4-Pro를 사용해 source-aware setting에서 package가 paper method reproduction을 support하는지 확인한다.
5. Human review로 systematic defects를 제거하고 final 90 tasks를 확정한다.

Reproduction-mode final check에서는 Claude Opus 4.6이 30 tasks, DeepSeek-V4-Pro가 21 tasks를 reproduce한다고 보고된다. 두 model이 모두 성공한 16 tasks에서는 SOTA 근처 deviation이 tight하다고 설명한다. 이는 package calibration을 지지하는 근거다.

## 4-3. Engineering notes

1. **Firewall이 benchmark의 핵심이다**
   - Source method artifacts가 남아 있으면 discovery benchmark가 reproduction benchmark가 된다.

2. **Containerization이 필수다**
   - Scientific task는 dependency, data format, evaluator가 너무 다양하다.

3. **Task-specific evaluator and normalized metric을 같이 써야 한다**
   - 각 domain에는 고유 metric이 필요하지만, leaderboard에는 comparable scale이 필요하다.

4. **Reproduction mode로 package sanity를 확인해야 한다**
   - Source-aware agent도 reproduce하지 못하면 package defect일 수 있다.

5. **Validity judge가 필요하다**
   - Agent는 output fabrication, evaluator gaming, shortcut exploitation을 시도할 수 있다.

# 5. Evaluation

## 5-1. Main results

논문 초록은 다음을 보고한다.

| 항목 | 값 |
| --- | ---: |
| Tasks | 90 |
| Frontier agent configurations evaluated | 10 |
| Strongest model Surpass-SOTA under $g>0.1$ | 17.8% |
| Protocol | Web-search-disabled |

이 결과는 조심해서 읽어야 한다. Current coding agent가 scientific coding에 쓸모없다는 뜻은 아니다. Strict discovery setting에서 published SOTA를 match or surpass하는 일이 여전히 드물다는 뜻이다.

## 5-2. Success mechanism

논문은 validated success가 주로 methodological translation에서 나온다고 분석한다. 즉 agent가 scientific task를 familiar supervised prediction template으로 바꿀 수 있을 때 성공한다. 이는 현재 agent가 완전히 새로운 scientific modeling idea를 발명한다기보다, known ML pattern을 scientific dataset에 이식하는 데 강하다는 해석으로 이어진다.

## 5-3. Failure mechanism

논문 초록은 failure가 task misunderstanding보다 wrong method choice and insufficient compute budget에서 주로 나온다고 설명한다. 이는 중요하다. Comprehension만 개선한다고 NatureBench 성능이 자동으로 좋아지는 것은 아니다. Method search, compute allocation, experiment planning, scientific prior가 함께 개선되어야 한다.

## 5-4. What really matters in the experiments

### 1) Discovery setting이 핵심이다

Source method가 보이면 reproduction이다. NatureBench는 same problem, same data, same metric, hidden method를 구성하려 한다.

### 2) Environment quality가 큰 기여다

NatureGym이 없으면 NatureBench가 성립하기 어렵다. Scientific benchmark는 task package construction 자체가 연구 기여다.

### 3) SOTA surpassing은 harsh하지만 의미 있다

Scientific progress는 valid solution만으로 충분하지 않다. Published metric에서 competitive해야 한다.

### 4) Current success는 translation 중심이다

Agent는 domain-specific scientific invention보다 known ML recipe adaptation에 더 강한 것으로 보인다.

# 6. Limitations

1. **Source corpus filter**
   - Public data, automated metric, manageable dataset size 조건을 만족하는 papers만 포함된다.
   - 중요한 Nature-family studies가 많이 제외될 수 있다.

2. **Physical or wet-lab task 제외**
   - Physical interaction이 필요한 task는 benchmark 범위 밖이다.

3. **SOTA anchor 불완전성**
   - Published SOTA는 compute, preprocessing, undocumented details에 의존할 수 있다.

4. **Compute budget sensitivity**
   - 일부 failure는 idea 부족보다 resource 부족일 수 있다.

5. **Validity judge dependency**
   - Shortcut detection and invalid output 판단이 judge model quality에 의존한다.

6. **Discovery 범위 제한**
   - Fixed task and evaluator 안에서의 discovery이지, 완전한 open-ended science는 아니다.

7. **Data leakage risk**
   - Web-search-disabled protocol이 pretraining contamination을 완전히 제거하지는 못한다.

8. **Containerization overhead**
   - NatureGym package를 만들고 유지하는 비용이 크다.

9. **Cross-domain normalization**
   - SOTA-normalized gap은 유용하지만 모든 scientific metric을 완벽히 비교 가능하게 만들지는 않는다.

10. **Leaderboard drift**
    - New agents, compute budget, inference scaffold가 바뀌면 결과가 빠르게 변할 수 있다.

# 7. My Take

## 7-1. Why this matters for my work

NatureBench의 핵심은 "coding agents are weak"가 아니다. 더 중요한 점은 **scientific agent evaluation needs method-firewalled, containerized, SOTA-normalized tasks**라는 것이다.

AI-for-science benchmark가 reproduction에 머물면, agent가 paper understanding을 잘하는지 볼 수는 있다. 하지만 discovery는 다른 protocol이 필요하다. Same problem, same data, same metric, hidden method가 필요하다.

NatureBench는 그 construction path를 꽤 구체적으로 보여준다.

## 7-2. Reuse potential

### Scientific agent evaluation

NatureGym-style packaging은 internal AI-for-science benchmark에도 재사용 가능하다. 핵심은 method를 숨기되 problem definition을 보존하는 것이다.

### Coding agent training

Failure analysis는 method search, compute budgeting, experiment planning을 더 훈련해야 함을 시사한다. 단순 code implementation 능력만으로는 부족하다.

### Benchmark governance

Verify-repair loop and reproduction-mode audit는 credible agent benchmark를 만드는 좋은 template이다.

### Research automation

NatureBench는 scientific discovery capability를 주장하는 agent를 stress test하는 데 유용하다.

## 7-3. Production considerations

- Source method firewall을 엄격히 유지해야 한다.
- Preprocessing artifacts leakage를 audit해야 한다.
- Reproduction mode and discovery mode를 분리해야 한다.
- Run당 compute budget을 기록해야 한다.
- Suspicious improvement는 validity judge and human spot-check를 같이 써야 한다.
- Final score뿐 아니라 method pathway를 기록해야 한다.

## 7-4. Follow-up papers

- PaperBench
- CORE-Bench
- ReplicationBench
- MLE-bench
- PostTrainBench
- NatureBench leaderboard updates
- AI Scientist and AI Scientist-v2
- Towards Automating Scientific Review with PAT
- NatureBench-related agent evaluation papers

# 8. Summary

- NatureBench는 coding agents가 Nature-family scientific tasks에서 published SOTA를 match or surpass할 수 있는지 평가한다.
- NatureGym은 papers를 task brief, dataset, evaluator, held-out test, SOTA anchor가 있는 containerized tasks로 바꾼다.
- Source method는 firewall로 제거되어 agent가 reproduction이 아니라 discovery에 가까운 문제를 풀게 된다.
- Final benchmark는 90 tasks and 333 evaluation instances로 구성된다.
- Current agents는 published SOTA를 넘는 경우가 아직 제한적이며, 성공은 주로 methodological translation에서 나온다.
