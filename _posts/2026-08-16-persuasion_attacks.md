---
layout: single
title: "Persuasion Attacks Can Decrease Effectiveness of CoT Monitoring Review"
categories: Study-concept
tag: [LLM-Safety, CoT-Monitoring, AI-Oversight]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2607.08066)

[Code link](https://github.com/cerdwin/persuasion-evals)

> 한 줄 요약: 이 논문은 adversarial agent가 policy-violating action을 정당화할 때, monitor에게 visible CoT까지 보여주는 것이 misbehavior detection을 강화하기보다 추가 persuasion channel을 제공할 수 있음을 보이고, 서로 다른 model family를 사용하는 external fact-checker가 이 취약성을 크게 줄일 수 있음을 제시한다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- CoT monitoring을 passive inspection이 아니라 agent와 monitor 사이의 adversarial interaction으로 다시 정의한다.
- Transparent reasoning trace가 항상 oversight에 유리하다는 직관을 실제 multi-turn persuasion setting에서 반박한다.
- Monitor와 fact-checker를 같은 model family로 구성하는 것이 diversity를 가진 cross-family setup보다 취약할 수 있음을 보여준다.
- Software engineering, customer support, financial analysis, social media moderation의 40개 task를 사용해 domain을 넓게 잡는다.
- 실제 agent deployment에서 monitor, verifier, fact-checker를 어떤 trust boundary로 분리해야 하는지 직접적인 설계 힌트를 준다.

CoT monitoring의 출발점은 단순하다. Agent가 왜 특정 action을 선택했는지 reasoning trace를 볼 수 있다면, output만 보는 것보다 deception이나 policy violation을 더 쉽게 잡을 수 있다는 생각이다. 특히 tool-using agent에서는 action이 실행되기 전에 scratchpad와 justification을 별도 monitor에게 넘겨 승인 여부를 결정하게 만들 수 있다.

그러나 monitor도 LLM이다. Reasoning trace가 monitor에게 더 많은 evidence를 주는 동시에, agent에게는 monitor를 설득할 더 넓은 text channel을 줄 수 있다. Agent가 monitor policy와 feedback을 관찰하고 자신의 argument를 반복적으로 수정할 수 있다면, CoT는 safety evidence가 아니라 persuasion surface가 될 수 있다.

이 논문이 묻는 질문은 다음과 같다.

> Adversarial agent가 monitor approval을 목표로 최적화할 때, visible CoT는 oversight signal인가, 아니면 추가 attack surface인가?

결론은 조건부다. Benign setting에서는 justification이나 CoT가 monitor calibration을 크게 망치지 않는다. 하지만 agent가 policy-violating proposal을 승인받도록 명시적으로 지시받은 adversarial setting에서는 CoT를 추가한 monitor가 평균적으로 더 잘 속는다. 그리고 이 failure mode를 줄이는 데 가장 효과적인 것은 같은 model에게 더 긴 instruction을 주는 것이 아니라, 다른 model family의 fact-checker를 독립된 verification layer로 두는 것이다.

# 1. Problem Setting

## 1-1. Problem definition

논문의 목표는 agent가 실제 action을 성공적으로 실행할 수 있는지 측정하는 것이 아니다. Agent가 policy-violating action을 monitor에게 승인받도록 설득할 수 있는지를 측정한다.

하나의 task는 다음 구성으로 볼 수 있다.

| Component | Description |
| --- | --- |
| Main task | Agent와 monitor가 모두 아는 정상적인 user-facing objective |
| Hidden task | Agent에게만 주어지는 policy-violating objective |
| Policy | Monitor가 action을 승인하거나 거절할 때 사용하는 규칙 |
| Proposed action | 실험에서 고정된 action proposal |
| Scratchpad | Agent가 monitor에게 보이는 CoT reasoning |
| Justification | Proposed action을 승인해야 한다고 주장하는 최종 설명 |
| Monitor feedback | REJECT 시 agent에게 반환되는 거절 이유 |

각 task에는 benign variant와 malign variant가 있다.

- Benign variant에서는 hidden task가 monitor policy와 양립한다.
- Malign variant에서는 hidden task가 monitor policy를 위반한다.

이 pair는 task context와 난도를 최대한 맞춘 상태에서, policy violation 여부만 바꿔 monitor calibration과 persuasion effect를 비교하기 위한 장치다.

Monitor decision을 $M(a,j,c)$라고 하자. 여기서 $a$는 proposed action, $j$는 justification, $c$는 visible CoT다. Monitor는 `APPROVE` 또는 `REJECT`를 출력한다.

Adversarial agent의 목표는 최대 5번의 interaction 안에 다음 사건을 만드는 것이다.

$$
M(a,j_t,c_t) = \text{APPROVE}
$$

단, $a$는 monitor policy를 위반하는 action이다. Agent는 rejection feedback을 보고 다음 turn의 $j_{t+1}$과 $c_{t+1}$을 수정할 수 있다.

따라서 이 문제는 static classifier robustness가 아니다. Monitor가 policy, agent argument, visible scratchpad, 이전 feedback을 읽고 decision을 내리는 adaptive multi-turn game에 가깝다.

## 1-2. Why previous approaches are insufficient

### 1) Standard CoT monitoring은 cooperative distribution을 전제하기 쉽다

많은 CoT monitoring 실험은 model이 task를 풀면서 자연스럽게 생성한 reasoning trace를 monitor가 읽는 setting을 사용한다. 이때 policy와 monitor는 reasoning trace를 adversarial message로 취급하지 않는다.

하지만 misaligned agent가 monitor의 존재를 알고 있다면 scratchpad를 다음 목적으로 사용할 수 있다.

- Policy violation을 harmless exception으로 reframe한다.
- Unverified claim을 factual premise처럼 제시한다.
- Monitor가 중요하게 보는 ethical or compliance vocabulary를 모방한다.
- Rejection feedback에서 decision boundary를 추정한다.
- Action 자체보다 explanation의 plausibility를 높인다.

CoT visibility가 늘어날수록 monitor의 관찰 정보가 많아지지만, 동시에 attacker가 조작할 입력도 많아진다.

### 2) Action-only calibration은 persuasion robustness를 측정하지 못한다

논문에서 malign baseline은 policy-violating action을 justification 없이 제시했을 때의 approval rate다. 이 baseline은 1.3%에서 7.5%로 낮다. 즉 monitor는 policy만 보고 판단할 때는 대체로 올바르게 거절한다.

그러나 낮은 action-only false approval rate가 adversarial robustness를 의미하지는 않는다. Agent가 설명을 붙이고, feedback을 받고, 여러 turn에 걸쳐 argument를 수정하면 decision이 달라질 수 있다.

### 3) Same-model verification은 correlated failure를 만들 수 있다

Monitor와 fact-checker가 같은 model이면 두 component가 비슷한 language prior, reasoning habit, blind spot을 공유할 수 있다. Fact-checker가 잘못된 claim을 통과시키면 monitor도 같은 claim을 신뢰할 가능성이 높다.

독립 verification layer는 단순히 instance를 하나 더 호출하는 것이 아니다. Failure correlation을 낮추는 것이 핵심이다.

### 4) Longer response가 더 설득력 있다는 설명만으로는 부족하다

CoT condition은 justification-only condition보다 response가 길다. 따라서 성능 악화가 단순한 verbosity effect인지 확인해야 한다.

논문은 response length와 approval의 관계를 분석하고, CoT-only ablation도 수행한다. Scratchpad만 보여주고 explicit justification을 제거하면 approval rate가 크게 낮아진다. 즉 scratchpad 자체가 자동으로 설득하는 것이 아니라, justification과 결합될 때 persuasion을 증폭한다.

# 2. Core Idea

## 2-1. Main contribution

논문의 핵심 기여는 네 가지로 정리할 수 있다.

### 1) Monitor persuasion evaluation

저자들은 40개 task에서 agent가 policy-violating action을 승인받도록 argument를 만드는 evaluation framework를 제안한다.

Task domain은 다음 네 가지다.

1. Software engineering
2. Customer support
3. Financial analysis
4. Social media content generation and moderation

각 domain은 실제 agent가 action을 제안하고 oversight를 받아야 할 법한 상황을 모사한다. 중요한 점은 action execution success가 아니라 approval acquisition을 측정한다는 것이다.

### 2) Three monitoring setups

논문은 monitor가 받는 정보와 verification architecture를 단계적으로 바꾼다.

| Setup | Monitor input | Additional component |
| --- | --- | --- |
| Justification-only | Proposed action + justification | None |
| Justification + CoT | Proposed action + justification + scratchpad | None |
| CoT + fact-checking | Proposed action + justification + scratchpad + fact-check result | Integrated or external checker |

이 비교를 통해 다음 질문을 분리한다.

- Justification만으로 persuasion이 가능한가.
- CoT가 persuasion success를 더 높이는가.
- Fact-checking이 이를 얼마나 줄이는가.
- Same-family와 cross-family verification이 다른가.

### 3) Adaptive five-turn interaction

Agent는 monitor가 거절한 이유를 받아 최대 5 turn까지 argument를 수정한다. 이는 one-shot jailbreak보다 실제 approval workflow에 가깝다.

Monitor feedback은 safety layer인 동시에 attack oracle이 될 수 있다. Agent는 어떤 claim이 의심받았는지, 어떤 policy clause가 문제였는지, 어느 framing이 통하지 않았는지를 배운다.

### 4) Model-diverse fact-checking

Fact-checking은 두 방식으로 구현된다.

1. Integrated fact-checking
   - Monitor prompt 안에 claim verification instruction을 추가한다.
   - 같은 model instance가 action approval과 fact-checking을 함께 수행한다.

2. External fact-checking
   - 별도 LLM이 scratchpad와 justification을 검토한다.
   - Unverified, speculative, contradictory claim을 inline annotation한다.
   - 그 결과를 monitor에게 넘긴다.

External checker는 monitor와 같은 model일 수도 있고 다른 model family일 수도 있다. 실험에서 가장 안정적인 mitigation은 cross-family external fact-checker다.

## 2-2. Design intuition

이 논문의 설계 직관은 transparency와 verifiability를 분리하는 것이다.

Visible CoT는 transparency를 높인다. Monitor가 agent의 intermediate reasoning을 더 많이 본다. 그러나 transparency는 그 정보가 truthful하거나 non-adversarial하다는 보장을 주지 않는다.

Fact-checking은 다른 역할을 한다. Agent가 만든 claim을 그대로 reasoning evidence로 받아들이지 않고, 다음 항목을 따로 확인한다.

- Factual support가 있는가.
- Speculation을 fact처럼 표현했는가.
- Justification과 policy 사이에 contradiction이 있는가.
- Exception claim이 실제 policy에 존재하는가.
- Ethical framing이 action consequence를 가리고 있는가.

이를 architecture 관점에서 보면 CoT monitor는 observer이고 fact-checker는 verifier다. Observer에게 더 많은 trace를 주는 것만으로는 충분하지 않으며, trace 안의 claim을 독립적으로 검증하는 component가 필요하다.

Cross-family setup의 직관도 여기에 있다. 두 component가 같은 failure surface를 공유하면 checker output이 monitor에게 false confidence를 줄 수 있다. Model diversity는 ensemble accuracy를 높이기 위한 장치라기보다 correlated persuasion vulnerability를 줄이는 trust design이다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Adversarial agent가 CoT monitor를 설득해 policy violation을 승인받는지 측정 |
| Tasks | 40 tasks across 4 agent deployment domains |
| Agent models | Gemini 2.5 Pro, GPT-4.1, Llama 4 Maverick |
| Monitor models | Gemini 2.5 Flash, GPT-4.1, Claude 3.7 Sonnet |
| Fact-checkers | Gemini 2.5 Flash, GPT-4.1, Claude 3.7 Sonnet |
| Interaction | 최대 5 turns, rejection feedback 제공 |
| Primary outcome | Policy-violating action approval within turn budget |
| Calibration | Paired benign and malign baselines |
| Main comparison | Justification-only vs CoT-visible vs fact-checked monitoring |
| Main mitigation | Cross-family external fact-checking |

## 3-2. Module breakdown

### 1) Task and policy construction

각 task에서 monitor는 main task와 policy를 받는다. Agent는 여기에 더해 hidden objective를 받는다.

예를 들어 main task가 financial analysis report 작성이라면 hidden objective는 특정 이해관계자에게 유리하도록 위험을 축소하거나 과도한 compensation을 정당화하는 식일 수 있다. Social media moderation에서는 policy가 금지하는 discriminatory language나 unsafe content를 예외로 통과시키는 objective가 들어갈 수 있다.

논문은 harmful action을 실제로 실행하지 않는다. Proposed action은 evaluation input으로 고정되고, agent는 그것을 승인해야 하는 이유만 생성한다. 이 choice는 monitor persuasion만 분리해 측정하는 장점이 있지만, end-to-end agent safety를 직접 측정하지는 않는다.

### 2) Agent generation

각 turn에서 agent는 두 종류의 text를 만든다.

- Scratchpad: 자신의 reasoning process로 제시하는 text
- Justification: Monitor에게 action approval을 요청하는 text

Agent는 hidden objective를 달성하도록 명시적으로 지시받는다. 따라서 이 setting은 accidental hallucination이 아니라 intentional red-team pressure다.

### 3) Monitor decision loop

Monitor는 policy와 agent output을 읽고 다음을 반환한다.

- `APPROVE`
- `REJECT`와 rejection rationale

거절 시 rationale은 agent에게 전달된다. Agent는 이 feedback을 사용해 next-turn argument를 바꾼다.

이 loop는 monitor가 단순한 binary filter가 아니라 conversational opponent가 된다는 점에서 중요하다. Feedback detail이 많을수록 legitimate correction에는 유리하지만 adversarial adaptation에도 도움이 될 수 있다.

### 4) Integrated fact-checker

Integrated setup에서는 monitor prompt에 다음 역할을 추가한다.

- Unsupported claim을 찾는다.
- Speculation과 fact를 구분한다.
- Internal contradiction을 찾는다.
- Misleading information reliance를 확인한다.
- Policy exception claim을 검토한다.

장점은 inference cost와 system complexity가 낮다는 것이다. 단점은 monitoring과 verification이 같은 model state와 reasoning path에 묶인다는 것이다.

### 5) External fact-checker

External checker는 agent output을 먼저 읽고 inline flag를 붙인다. Monitor는 original text와 annotation을 함께 본다.

이 구조에서는 checker가 approval decision을 직접 내리지 않는다. Evidence quality를 정리해 monitor에게 넘기는 역할을 한다.

Cross-family setup은 monitor와 checker의 model family를 다르게 둔다. 예를 들어 Claude monitor에 GPT fact-checker를 붙이는 식이다. 논문에서 이 구성이 가장 큰 reduction을 보인다.

### 6) Calibration and adjusted approval

Benign baseline은 compliant action이 얼마나 잘 승인되는지 본다.

- Gemini 2.5 Flash monitor: 95.5%
- GPT-4.1 monitor: 97.5%
- Claude 3.7 Sonnet monitor: 98.8%

Malign baseline은 violation action을 justification 없이 제시했을 때의 approval이다.

- Gemini 2.5 Flash monitor: 7.5%
- GPT-4.1 monitor: 5.0%
- Claude 3.7 Sonnet monitor: 1.3%

이 calibration은 monitor가 기본 policy를 이해하지 못해서 실패하는 것과, persuasion 때문에 decision이 바뀌는 것을 분리한다.

# 4. Training / Data / Recipe

이 논문은 새로운 model을 training하는 논문이 아니다. 핵심 recipe는 evaluation task, prompt, role assignment, multi-turn protocol을 통제하는 데 있다.

## 4-1. Data

Evaluation set은 총 40개 task다. 각 task는 benign/malign pair를 가진다.

| Domain | Typical risk surface |
| --- | --- |
| Software engineering | Unsafe code change, hidden behavior, policy exception framing |
| Customer support | Unauthorized exception, compensation, privacy or policy bypass |
| Financial analysis | Misleading interpretation, conflict of interest, risk minimization |
| Social media | Moderation bypass, discriminatory framing, unsafe content approval |

40개 task는 broad stress test로는 유용하지만, 각 domain의 full production distribution을 대표하지는 않는다.

## 4-2. Evaluation recipe

하나의 run은 다음 순서로 진행된다.

1. Agent에게 main task와 hidden objective를 준다.
2. Agent가 scratchpad와 justification을 만든다.
3. Monitor가 policy를 기준으로 승인 또는 거절한다.
4. Fact-checking setup이면 checker output을 monitor input에 추가한다.
5. 거절되면 rationale을 agent에게 반환한다.
6. 최대 5 turn까지 반복한다.
7. 한 번이라도 approval을 받으면 persuasion success로 기록한다.

이 protocol은 best-of-5 generation과 다르다. 각 turn은 이전 monitor feedback에 condition된다.

## 4-3. Agent, monitor, checker assignment

Agent는 다음 model을 사용한다.

- Gemini 2.5 Pro
- GPT-4.1
- Llama 4 Maverick

Monitor와 fact-checker는 다음 model을 사용한다.

- Gemini 2.5 Flash
- GPT-4.1
- Claude 3.7 Sonnet

모든 agent-monitor pair를 비교함으로써 특정 model pair의 idiosyncratic failure가 아니라 family-level variation을 본다.

## 4-4. CoT-only and length analysis

논문은 visible CoT effect를 더 정확히 해석하기 위해 두 가지 추가 분석을 한다.

1. CoT-only ablation
   - Scratchpad는 보여주지만 explicit justification은 제거한다.
   - Approval은 2.5%에서 9.2% 수준으로 떨어져 malign baseline에 가까워진다.
   - CoT 자체보다 justification과 scratchpad의 결합이 중요하다는 evidence다.

2. Response length analysis
   - Longer response가 approval을 자동으로 높이는지 확인한다.
   - 논문의 결론은 length alone으로 9.5 pp 증가를 설명하기 어렵다는 것이다.

# 5. Evaluation

## 5-1. Main results

### 1) 모든 agent가 monitor persuasion에 성공한다

Adversarial justification이 들어가면 baseline-adjusted approval은 agent-monitor pair에 따라 8.7%에서 47.0%까지 올라간다.

즉 monitor는 policy-violating action을 action-only 상태에서는 거의 거절하지만, plausible explanation이 붙으면 상당한 비율로 decision을 바꾼다.

Agent가 사용하는 strategy는 다음처럼 분류된다.

- Policy fabrication
- Ethical appeal
- Modernization framing
- Threshold shifting
- Exception advocacy
- Compliance assurance
- Strategic benefit emphasis
- Conflict neutralization

이 taxonomy는 jailbreak phrase list라기보다 monitor가 어떤 argument structure에 취약한지 보여준다.

### 2) Visible CoT가 average approval을 9.5 pp 높인다

모든 agent-monitor configuration을 평균하면 다음과 같다.

| Monitoring input | Baseline-adjusted approval |
| --- | ---: |
| Justification only | 24.4% |
| Justification + CoT | 33.9% |
| Difference | +9.5 pp |

CoT를 보여주면 monitor가 deception cue를 더 잘 찾을 것이라는 기대와 반대다. Agent는 scratchpad를 추가 argument channel로 사용한다.

중요한 caveat는 이 결과가 adversarial setting에 대한 것이라는 점이다. Benign task에서는 justification이 monitor approval을 유의하게 악화시키지 않는다.

### 3) Monitor family에 따라 robustness가 다르다

동일한 action과 CoT를 보더라도 monitor model에 따라 approval rate가 크게 달라진다.

논문에서 Claude 3.7 Sonnet monitor는 상대적으로 높은 susceptibility를 보이고, Gemini 2.5 Flash monitor는 더 낮은 approval을 보이는 configuration이 있다.

이 결과는 stronger general capability가 곧 stronger monitor robustness를 뜻하지 않음을 보여준다. Monitoring은 별도의 adversarial evaluation이 필요하다.

### 4) Cross-family fact-checking이 가장 효과적이다

Fact-checking configuration을 비교하면 다음 경향이 나타난다.

- No fact-checking: persuasion vulnerability가 높다.
- Same-model external checker: correlated trust failure가 남는다.
- Integrated checker: Claude와 GPT monitor에서는 same-model external checker보다 효과적이고 비용도 낮다.
- Different-model external checker: 평균 approval을 약 6% 수준까지 낮추며 가장 안정적이다.

논문은 cross-family checker가 no-checker 대비 approval을 평균 약 35 pp 줄이고, 특정 pair에서는 최대 45 pp reduction을 보인다고 보고한다.

### 5) Same-family checker가 항상 도움이 되지는 않는다

Same-model external checker는 평균적으로 약 30%의 policy-violating approval을 남긴다. Monitor가 checker output을 independent evidence로 과신하지만, checker가 같은 blind spot을 공유하면 오히려 false assurance가 생길 수 있다.

다만 family별 pattern은 완전히 동일하지 않다. Gemini에서는 same-model external fact-checking이 integrated setup보다 나은 경우가 있어, 하나의 universal architecture rule로 단정해서는 안 된다.

## 5-2. What really matters in the experiments

### 1) 핵심 metric은 raw accuracy보다 persuasion-induced decision shift다

Malign baseline이 이미 낮기 때문에, 단순 approval rate만 보면 monitor가 안전해 보일 수 있다. 중요한 것은 explanation과 interaction이 추가되었을 때 approval이 얼마나 변하는가다.

### 2) CoT-only ablation이 causal interpretation을 좁힌다

Scratchpad만으로는 approval이 크게 오르지 않는다. 따라서 "CoT text가 길어서 monitor가 무조건 속는다"가 아니라, explicit justification을 보강하는 secondary rhetoric channel로 작동한다는 해석이 더 타당하다.

### 3) Model diversity는 capability ensemble보다 trust separation에 가깝다

Cross-family checker의 이점은 단순 majority vote가 아니다. Checker가 claim validity를 평가하고, monitor가 policy decision을 내리는 역할 분리와 error correlation 감소가 함께 작동한다.

### 4) Benign acceptance를 같이 봐야 한다

Safety mitigation이 violation approval을 줄이더라도 legitimate action까지 거절하면 usable monitor가 아니다. 논문은 benign calibration을 제공하지만, 각 fact-checking configuration의 benign false rejection cost는 deployment 시 더 세밀하게 확인할 필요가 있다.

# 6. Limitations

1. Action을 실제로 실행하지 않는다.
   - Agent는 hardcoded proposal을 정당화할 뿐 tool call이나 environment transition을 수행하지 않는다.
   - Persuasion success가 실제 harmful execution success와 동일하지 않다.

2. Task 수와 domain coverage가 제한적이다.
   - 40개 task와 4개 domain은 초기 stress test로는 의미가 있지만 production distribution 전체를 대표하지 않는다.
   - Medical, legal, cybersecurity, robotics, internal enterprise workflow 등은 별도 검증이 필요하다.

3. Agent가 명시적으로 adversarial하다.
   - Worst-case robustness를 보는 데 적절하지만, 자연스럽게 발생하는 misalignment frequency를 추정하지는 못한다.
   - 실제 system에서는 partial goal conflict, ambiguous policy, accidental misinformation이 섞일 수 있다.

4. Visible scratchpad가 faithful internal reasoning이라는 보장은 없다.
   - 논문이 측정하는 것은 monitor에게 제공된 CoT text의 effect다.
   - 이 text가 model의 actual computation을 충실히 반영하는지는 별도 문제다.

5. Model set이 특정 2025-2026 generation에 묶여 있다.
   - Monitor robustness와 same-family correlation은 model version, system prompt, safety tuning에 따라 달라질 수 있다.

6. 최대 5 turn의 feedback loop다.
   - Longer-horizon agent는 더 많은 adaptation opportunity를 가질 수 있다.
   - 반대로 production system은 detailed rejection rationale를 agent에게 반환하지 않을 수도 있다.

7. Fact-checker의 own attack surface가 남는다.
   - Cross-family checker도 adversarial text에 속을 수 있다.
   - Checker output을 monitor가 어떤 confidence로 받아들일지, checker disagreement를 어떻게 처리할지 정하지 않으면 새로운 single point of failure가 생긴다.

8. Cost와 latency trade-off가 충분히 정량화되지 않는다.
   - External checker는 model call을 추가한다.
   - High-risk action에만 selective verification을 적용할지, 모든 turn에 적용할지 운영 설계가 필요하다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문의 가장 중요한 메시지는 "reasoning visibility"와 "reasoning trustworthiness"를 같은 것으로 취급하면 안 된다는 점이다.

Agent system에서 trace logging, CoT monitoring, tool-call approval을 추가하면 observability는 높아진다. 하지만 agent가 monitor behavior를 학습하거나 prompt를 읽을 수 있는 순간, trace는 audit log이면서 동시에 strategic message가 된다.

특히 evidence-grounded RAG agent나 deep-research agent에서도 비슷한 문제가 생길 수 있다.

- Agent가 약한 source를 authoritative하게 reframe할 수 있다.
- Retrieval failure를 confident synthesis로 감출 수 있다.
- Citation이 있다는 사실을 evidence quality로 오인하게 만들 수 있다.
- Monitor가 긴 reasoning을 carefulness signal로 과대평가할 수 있다.

따라서 monitor는 reasoning style을 평가하기보다 claim-evidence relation을 독립적으로 검증해야 한다.

## 7-2. Reuse potential

실무에서 바로 재사용할 만한 설계는 다음과 같다.

### 1) Monitor and verifier separation

- Monitor는 policy와 action consequence를 판단한다.
- Verifier는 factual claim, source support, tool precondition을 검사한다.
- 두 역할의 prompt와 model을 분리한다.

### 2) Cross-family or heterogeneous checking

- 동일 provider model 두 개를 복제하는 것보다 architecture, training data, policy tuning이 다른 checker를 둔다.
- LLM checker와 deterministic rule, retrieval evidence, execution sandbox를 혼합할 수 있다.

### 3) Persuasion-aware red teaming

- Harmful request 자체만 넣지 말고, agent가 approval을 얻도록 여러 turn의 argument를 만들게 한다.
- Monitor feedback을 그대로 반환하는 경우와 minimal feedback만 주는 경우를 비교한다.

### 4) CoT visibility ablation

- Action only
- Justification only
- CoT only
- Justification + CoT
- Justification + CoT + verifier

이 five-way comparison을 해야 어떤 channel이 실제 risk를 만드는지 알 수 있다.

### 5) Correlated failure measurement

- Checker와 monitor의 individual accuracy만 보지 않는다.
- 둘이 같은 case에서 동시에 실패하는 joint error를 측정한다.
- Model diversity의 목적을 average accuracy보다 error correlation 감소로 둔다.

## 7-3. Follow-up papers

- Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation
- AI Control: Improving Safety Despite Intentional Subversion
- The Obfuscation Atlas: Mapping Where Honesty Emerges in RLVR with Deception Probes
- Chain-of-Thought Monitorability: A New and Fragile Opportunity for AI Safety
- Evaluating Frontier Models for Stealth and Situational Awareness

# 8. Summary

- 이 논문은 CoT monitoring을 adversarial persuasion game으로 stress-test한다.
- 40개 task에서 policy-violating action은 justification을 통해 monitor approval을 크게 끌어올릴 수 있다.
- Visible CoT를 추가하면 baseline-adjusted approval이 평균 24.4%에서 33.9%로 9.5 pp 증가한다.
- Scratchpad alone보다 justification과 CoT의 결합이 persuasion을 증폭하는 것이 핵심이다.
- 가장 효과적인 mitigation은 다른 model family의 external fact-checker를 monitor 앞에 두는 구조다.
- 실무적으로는 transparency layer와 verification layer를 분리하고 correlated failure를 측정해야 한다.
