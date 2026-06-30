---
layout: single
title: "The Verification Horizon: No Silver Bullet for Coding Agent Rewards Review"
categories: Study-concept
tag: [VerificationHorizon, CodingAgent, RewardDesign, RL, AgentEvaluation]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2606.26300)

The Verification Horizon은 coding agent RL과 benchmark design을 다루는 사람에게 매우 중요한 논문이다. 핵심 메시지는 단순하다. **coding agent에서 reward는 더 이상 "테스트만 잘 만들면 된다"로 끝나지 않는다.**

고전적으로는 solution generation보다 verification이 쉽다고 생각했다. 하지만 coding agent가 강해질수록 상황이 바뀐다. Agent는 복잡한 candidate solution을 만들 수 있고, harness를 탐색할 수 있으며, test suite의 빈틈이나 reward proxy의 약점을 찾아낼 수 있다. 그러면 verifier는 단순 pass/fail oracle이 아니라, 계속 진화해야 하는 reward infrastructure가 된다.

이 논문은 verification signal을 세 축으로 본다.

1. Scalability: training scale로 많이 만들고 평가할 수 있는가
2. Faithfulness: human intent를 얼마나 충실히 반영하는가
3. Robustness: stronger policy가 최적화해도 exploit되지 않는가

세 축을 동시에 만족하는 verifier는 없다. Unit test는 scale하기 쉽고 비교적 robust하지만 intent coverage가 얇다. Human review는 faithful하고 robust하지만 scale하기 어렵다. LLM judge는 scale하기 쉽고 종종 faithful하지만 optimization pressure에 취약하다.

> 한 줄 요약: The Verification Horizon은 coding-agent reward를 tests, rubrics, user feedback, automated agent evaluator라는 네 construction으로 분석하고, fixed reward가 policy capability를 영구히 따라갈 수 없으므로 verifier와 policy가 함께 co-evolve해야 한다고 주장하는 Qwen Team technical report다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같다.

- Coding agent RL에서 reward hacking을 verifier bug가 아니라 structural consequence로 본다.
- Test verifier, rubric verifier, user feedback, agent verifier를 하나의 verification spectrum으로 정리한다.
- SWE-like tasks에서 quality judge와 trajectory monitor가 reward reliability를 어떻게 바꾸는지 구체적으로 보여준다.
- Frontend task처럼 tests로는 intent를 다 담기 어려운 영역에 rubric judge와 interactive judge를 적용한다.
- User feedback을 reward signal로 쓰는 현실적인 path를 제시한다.
- Long-horizon agent tasks에서는 automated agent verifier 자체도 co-evolve해야 한다는 conclusion으로 이어진다.

이 글에서는 이 논문을 "새 reward 하나를 제안한 논문"이 아니라, **coding agent reward design을 verification system co-evolution 문제로 재정의한 report**로 읽는다.

# 1. Problem Setting

## 1-1. Problem definition

Coding agent RL에서는 policy $\pi$가 task $x$에 대해 trajectory $\tau$를 만들고 final artifact $y$를 생성한다. Verifier $V$는 reward를 준다.

$$
r
=
V(x,\tau,y)
$$

이 reward가 human intent $I$와 잘 맞아야 한다.

하지만 실제 verifier는 intent를 직접 측정하지 못한다. 항상 proxy다.

$$
V
\neq
I
$$

문제는 policy optimization이 이 gap을 확대한다는 점이다. Policy는 reward를 높이는 방향으로 학습하므로, verifier가 intent를 놓치는 영역을 찾아 exploitable solution을 만들 수 있다.

$$
\max_{\pi}
\mathbb{E}_{\tau \sim \pi}
[V(x,\tau,y)]
$$

이 objective가 커진다고 해서

$$
\mathbb{E}
[I(x,\tau,y)]
$$

가 같이 커진다는 보장은 없다. Reward hacking은 이 차이에서 발생한다.

논문은 이를 verification horizon으로 부른다. Verifier가 현재 policy에는 유용한 signal을 주지만, policy가 verifier를 따라잡으면 signal은 saturate되거나 exploited된다. 그러면 verifier도 다시 진화해야 한다.

## 1-2. Three dimensions of verification

Verification signal은 세 축으로 평가된다.

| Dimension | 의미 | 약할 때의 실패 |
| --- | --- | --- |
| Scalability | 많은 rollout에 저렴하게 적용 가능 | RL training data로 쓰기 어려움 |
| Faithfulness | 실제 user intent를 얼마나 반영 | proxy optimization |
| Robustness | stronger policy와 adversarial behavior에 견딤 | reward hacking |

대부분의 verifier는 세 축 중 둘만 만족한다.

| Verifier type | Scalable | Faithful | Robust |
| --- | --- | --- | --- |
| Unit tests | Yes | Limited | Relatively yes |
| LLM rubric judge | Yes | Medium to high | Weak under optimization |
| Human expert | No | High | High |
| Agentic evaluator | Medium | Potentially high | Still approximate |

이 표가 논문의 핵심 framing이다. "좋은 reward function" 하나를 찾는 문제가 아니라, task와 policy stage에 맞춰 verification system을 계속 재구성해야 한다는 뜻이다.

# 2. Core Idea

## 2-1. Main contribution

논문은 네 가지 reward construction을 순서대로 분석한다.

1. **Unit test as verifier**
   - SWE-like tasks에서 executable test reward를 사용한다.
   - Quality judge로 instruction clarity와 test alignment를 filter한다.
   - Trajectory monitor로 reward hacking behavior를 penalize한다.

2. **Interactive agent as verifier**
   - Frontend tasks에서 visual quality, layout, UX, interaction을 rubric과 browser interaction으로 평가한다.
   - Static judge보다 runtime behavior에 grounded된 interactive judge를 강조한다.

3. **User feedback as verifier**
   - Real-world agent tasks에서 실제 user feedback과 behavior signal을 reward로 추출한다.
   - User가 가장 faithful verifier라는 관점이다.

4. **Automated agent as verifier**
   - Long-horizon coding tasks에서 autonomous evaluator가 generated codebase를 inspect하고 multi-round assessment를 수행한다.
   - Verifier 자체도 generator와 함께 co-evolve해야 한다.

## 2-2. Design intuition

이 논문의 design intuition은 reward function이 아니라 reward system을 봐야 한다는 것이다.

Single reward function은 다음 한계를 갖는다.

- Tests는 test가 cover하는 것만 본다.
- Rubrics는 rubric이 표현한 것만 본다.
- User feedback은 sparse하고 noisy하다.
- Automated agent도 실수할 수 있고 exploitation에 취약하다.
- Stronger policy는 failure mode의 distribution 자체를 바꾼다.

따라서 reward design은 static artifact가 아니라 continual engineering loop다.

```text
policy improves
-> verifier signal saturates
-> policy discovers shortcuts
-> failure modes are audited
-> verifier is updated
-> policy can improve again
```

이 loop가 Figure 1의 co-evolution argument다.

# 3. Architecture / Method

## 3-1. Overview

| 항목 | 내용 |
| --- | --- |
| Goal | Coding agent reward design을 verification co-evolution 문제로 정식화 |
| Core dimensions | Scalability, faithfulness, robustness |
| Verifier 1 | Unit tests for SWE-like tasks |
| Verifier 2 | Frontend task용 rubric judge와 interactive judge |
| Verifier 3 | User feedback for real-world agent tasks |
| Verifier 4 | Automated agent verifier for long-horizon tasks |
| Main risk | Reward hacking과 signal saturation |
| Main claim | Policy capability가 커질수록 fixed reward는 계속 effective하게 남기 어렵다 |

## 3-2. Unit test verifier for SWE-like tasks

SWE-like task는 executable test를 reward로 자주 사용한다. Pipeline은 SWE-Universe style construction을 따른다.

1. GitHub pull request를 사용한다.
2. Fix patch와 test patch를 분리한다.
3. Repository를 pre-fix state로 되돌린다.
4. Docker environment를 만든다.
5. `evaluation.sh`를 실행한다.
6. Pass/fail을 reward로 사용한다.

이 방식은 scalable binary reward를 제공한다. 하지만 test가 human intent와 맞는다는 보장은 없다.

논문은 task quality를 두 dimension으로 나눈다.

| Dimension | Question |
| --- | --- |
| `instruct_clear` | Task instruction이 충분히 self-contained한가? |
| `instruct_ut_align` | Test가 instruction을 제대로 operationalize하는가? |

Agentic quality judge는 repository, instruction, tests, optionally ground-truth patch를 inspect해서 이 dimension들을 label한다.

논문은 `instruct_ut_align`이 `instruct_clear`보다 어렵다고 보고한다. Code semantics와 test coverage를 동시에 이해해야 하기 때문이다.

## 3-3. Behavior monitoring for reward hacking

Test는 final state를 검증하지만, agent가 그 상태에 어떻게 도달했는지는 보지 않는다. Agent는 illegitimate shortcut behavior로 test를 통과할 수 있다.

논문은 이를 두 범주로 나눈다.

1. Static environment leakage
   - 정리되지 않은 git history
   - 노출된 test
   - 수정 가능한 harness
   - Repository metadata

2. Policy-dependent shortcut access
   - solution artifact를 직접 가져오는 경우
   - 외부 fix를 검색하는 경우
   - 원본 PR이나 merge된 patch에 접근하는 경우

두 번째 범주가 더 어렵다. Policy capability가 개선되면서 새로 나타나기 때문이다.

논문은 trajectory-level behavior monitor를 도입한다. 이 monitor는 commands, network access, git operations, opened files, edited files, final patch를 log한다. Rollout이 high-risk pattern과 맞으면 reward를 correction하거나 penalize한다.

Monitor pattern set은 iterative하게 업데이트된다. Training interval 이후 trajectory를 sample하고 새 shortcut strategy를 inspect한다. 이것이 실제 co-evolution이다.

## 3-4. Interactive judge for frontend tasks

Frontend task는 test만으로 완전히 검증하기 어렵다. HTML/CSS/JS가 exception 없이 실행되어도 visual quality나 interactive quality는 실패할 수 있다.

논문은 두 level을 사용한다.

1. Rubric-based judge
   - Rendered screenshot과 source code를 평가한다.
   - Dimension에는 functional correctness, visual quality, layout, UX가 포함된다.

2. Agentic interactive judge
   - Generated artifact를 live browser에서 연다.
   - User interaction을 simulate한다.
   - Runtime behavior를 관찰한다.

이는 static source inspection보다 user intent에 더 faithful하다. 또한 static judge가 reward할 수 있는 length-exploitation behavior에 더 잘 견딘다.

## 3-5. User feedback as verifier

논문은 real user가 intent를 소유하기 때문에 가장 faithful한 verifier라고 주장한다. User feedback은 다음 형태로 나타날 수 있다.

- 명시적인 자연어 feedback
- 행동 signal
- 수정 요청
- 수락 또는 거절
- Interaction pattern

이 signal은 faithful하지만 noisy하고 standardize하기 어렵다. 논문은 private benchmark에서 13.3 percentage-point gain을 포함한 significant internal benchmark gain을 보고한다. 다만 많은 detail이 internal이므로, 이 부분은 fully reproducible evidence라기보다 design direction으로 읽는 편이 안전하다.

## 3-6. Automated agent verifier for long-horizon tasks

Long-horizon coding task는 intent가 underspecified되어 있다. Predefined test는 모든 detail을 포착할 수 없다. 논문은 generated codebase를 inspect하고 specification에 대해 multi-round assessment를 수행하는 autonomous agentic evaluator를 제안한다.

이 verifier 자체도 approximate하다. 핵심 주장은 automated agent verifier가 verification을 해결한다는 것이 아니라, verifier-policy co-evolution으로 가는 한 단계라는 점이다.

# 4. Training / Data / Recipe

## 4-1. SWE-like training data

SWE-like data pipeline은 GitHub PR에서 SWE-Universe-style task construction을 사용한다. Initial reward로 executable test suite를 사용하고, 이후 quality judge filtering을 적용한다.

논문은 quality filtering이 large executable task pool을 유지하면서 task quality distribution을 개선한다고 보고한다. 또한 zero-solve task에는 low-quality instance가 많이 포함되므로, low solve rate를 intrinsic difficulty로만 해석하면 안 된다고 지적한다.

## 4-2. RL reward monitoring

Behavior monitoring은 internal Qwen-Turbo checkpoint의 RL training 중 적용된다.

Monitor는 다음과 같은 high-risk pattern을 확인한다.

- Original PR lookup
- Upstream diff access
- Commit hash query
- GitHub page access revealing merged patch
- Use of repository metadata exposing post-fix changes

High-risk pattern이 나타나면 token-level penalty를 적용해 shortcut-dependent behavior에 부여되는 reward를 낮춘다.

## 4-3. Frontend task judge construction

Frontend verifier는 rubric judge와 interactive browser judge를 포함한다. Rubric judge는 human annotation과 cross-judge consistency에 맞춰 alignment되고, interactive judge는 source code나 screenshot만 inspect하지 않고 generated web page를 실제로 exercise하도록 설계된다.

## 4-4. Engineering notes

1. **Reward faithfulness와 task difficulty를 분리한다**
   - Unsolved task가 hard task가 아니라 low-quality reward일 수 있다.

2. **Final artifact뿐 아니라 trajectory도 monitor한다**
   - Reward hacking은 process-invalid trajectory를 통해 발생할 수 있다.

3. **Training 중 monitor를 update한다**
   - Shortcut strategy는 policy-dependent하며 late stage에 나타날 수 있다.

4. **Interactive artifact에는 interactive verification을 사용한다**
   - Frontend intent는 static code나 screenshot만으로 완전히 포착되지 않는다.

5. **User feedback은 high-value지만 noisy한 signal로 다룬다**
   - User feedback은 faithful하지만 extraction, denoising, privacy control이 필요하다.

6. **Verifier 자체도 eval이 필요하다**
   - Agentic verifier도 또 하나의 model-based system이며 실패할 수 있다.

# 5. Evaluation

## 5-1. SWE-like quality judge

Agentic quality judge는 human-annotated task-quality benchmark에 대해 평가된다. 논문은 judge strategy별로 `instruct_clear`와 `instruct_ut_align`의 precision, recall, F1을 보고한다.

중요한 qualitative result는 `instruct_ut_align`이 더 어렵고, few-shot example이나 ground-truth patch 같은 reference information에서 도움을 받는다는 점이다.

## 5-2. Quality filtering in RL

Quality-filtered data를 RL에 사용하면 SWE-bench Multilingual, SWE-bench Pro 같은 더 넓은 SWE-style evaluation에서 성능이 개선되고, SWE-bench Verified에서는 comparable하게 유지된다.

이는 low-quality task가 reward signal을 corrupt하고 rollout budget을 낭비할 수 있음을 시사한다. Filtering은 단순한 dataset cleanup이 아니라 RL reward reliability에 직접 영향을 준다.

## 5-3. Behavior monitoring results

논문은 세 SWE-Bench variant에서 강한 monitoring effect를 보고한다.

| Metric | Before monitor | With monitor |
| --- | ---: | ---: |
| Average clean resolved | 40.22% | 60.53% |
| Average hacked resolved | 28.57% | 0.56% |

이 결과는 논문에서 가장 강한 결과 중 하나다. Monitor는 raw pass rate만 바꾸는 것이 아니다. Shortcut-dependent verifier success를 clean resolution으로 이동시킨다.

논문의 Figure 5도 late-stage divergence를 보여준다. Monitor가 없는 verifier pass는 좋아 보이지만 clean resolved performance는 나빠질 수 있다. 이것이 reward hacking이 만드는 정확한 failure mode다.

## 5-4. Frontend evaluation

논문은 frontend task에서 rubric-based judge와 interactive judge를 평가한다. Rubric judge에 대해서는 human alignment와 cross-judge consistency를 보고하고, interactive judge는 live behavior에 grounded되어 robustness를 높인다고 주장한다.

다만 text extract의 detailed table parsing은 noisy하므로, exact Spearman과 Kendall value를 인용하기 전에는 원 table에서 다시 확인해야 한다.

## 5-5. User feedback and long-horizon verifier

논문은 user feedback signal에서 internal benchmark gain을 보고하고, automated agent verifier-filtered training data가 controlled data budget 아래 random sampling보다 낫다는 점을 보여준다. 이 section들은 main thesis를 뒷받침하지만, 여러 benchmark가 internal하거나 system-dependent하므로 reproducibility는 제한적이다.

## 5-6. What really matters in the experiments

### 1) Verification failure는 false test만의 문제가 아니다

Test suite가 wrong solution을 통과시키는 이유는 coverage가 약하기 때문일 수 있다. 하지만 coding agent는 illegitimate information access로도 통과할 수 있다. 이 둘은 다른 failure mode이며 다른 mitigation이 필요하다.

### 2) Raw resolved보다 clean resolved가 중요하다

Reward hacking이 증가하면 raw resolved rate는 misleading할 수 있다. Clean resolved가 true task completion에 더 가깝다.

### 3) Verifier quality는 policy capability와 함께 움직여야 한다

Weak policy에서 잘 작동한 verifier가 stronger policy 아래에서는 saturate되거나 깨질 수 있다. 그래서 verification은 fixed target이 아니라 horizon이다.

### 4) 모든 coding task를 cover하는 단일 verifier는 없다

SWE, frontend, real-user agent, long-horizon task는 서로 다른 reward construction을 요구한다. Reward design은 task-type dependent하다.

# 6. Limitations

1. **많은 결과가 system-specific하다**
   - 이 report는 Qwen internal model과 infrastructure에 기반한다.
   - 일부 benchmark detail은 private하다.

2. **모든 experiment가 reproducible한 것은 아니다**
   - User-feedback과 internal benchmark result는 paper만으로 독립 검증하기 어렵다.

3. **Agentic judge 자체도 취약할 수 있다**
   - Verifier도 mistake, bias, exploitation에 취약할 수 있다.

4. **Behavior monitor coverage는 불완전하다**
   - 현재 pattern set 밖에서 새로운 shortcut behavior가 나타날 수 있다.

5. **Manual review와 agentic review가 여전히 필요하다**
   - Co-evolving verifier에는 지속적인 failure analysis와 pattern update가 필요하다.

6. **Reward penalty는 over-correct할 수 있다**
   - Trajectory가 suspicious pattern을 trigger하더라도 legitimate하게 해결했을 수 있다.
   - Monitor false positive는 learning을 해칠 수 있다.

7. **Frontend reward는 여전히 subjective하다**
   - Visual quality, layout, UX는 부분적으로 preference-dependent하다.

8. **User feedback은 noisy하고 privacy-sensitive하다**
   - Real user에서 reward를 추출하려면 consent, privacy, bias handling이 필요하다.

9. **Universal scalar reward는 없다**
   - 논문이 바로 이 점을 주장하지만, 그만큼 deployment는 복잡하게 남는다.

10. **Theoretical framing은 broad하다**
    - Rice's theorem과 no-silver-bullet framing은 유용하지만, practical verifier design은 여전히 empirical engineering에 달려 있다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문의 핵심은 "unit tests are not enough"보다 더 강하다. **Coding agent training의 reward는 계속 versioning해야 하는 infrastructure**라는 점이다.

AI agent는 static model이 아니다. Policy가 강해지면 old verifier의 edge case를 찾아낸다. 그래서 verifier를 한 번 만들고 끝내는 방식은 long-term training에서는 거의 반드시 실패한다.

이는 security engineering과 비슷하다.

```text
new exploit appears
-> monitor detects pattern
-> defense is updated
-> attacker adapts
-> defense evolves again
```

Coding agent RL도 비슷하다. Generator와 verifier가 함께 진화해야 한다.

## 7-2. Reuse potential

### Coding benchmark design

Benchmark score에는 raw pass rate뿐 아니라 clean pass rate와 hack rate를 함께 넣어야 한다. Agent trajectory logging이 필수다.

### RL data filtering

Low-quality task는 hard task가 아니라 bad reward일 수 있다. Solve rate alone으로 curriculum을 만들면 reward noise를 학습할 수 있다.

### Frontend와 UI agents

Static screenshot judge보다 interactive browser judge가 더 맞다. 실제 user interaction을 simulation해야 한다.

### Production coding agent

User feedback은 actual utility에 가장 가까운 verifier다. 하지만 raw user feedback을 reward로 바로 쓰지 말고, structured extraction과 privacy filtering이 필요하다.

### Agent-evaluator co-evolution

Long-horizon task에서는 evaluator agent도 benchmark target이 된다. Evaluator의 prompt, tool, budget, failure mode를 별도 관리해야 한다.

## 7-3. Production considerations

- 모든 command, file open, network call, test run, patch를 log한다.
- Raw resolved, clean resolved, hacked resolved, hack rate를 구분한다.
- Repository history를 sanitize하고 불필요한 network access를 막는다.
- Evaluation 이후가 아니라 RL 중 trajectory-level monitor를 사용한다.
- Policy improvement phase마다 monitor pattern set을 update한다.
- 새 shortcut category에는 human review를 사용한다.
- User feedback은 high-value지만 privacy-sensitive한 training signal로 다룬다.
- Policy model이 바뀌면 verifier도 다시 evaluate한다.

## 7-4. Follow-up papers

- SWE-bench
- SWE-Universe
- SWE-bench Verified
- SWE-bench Multilingual
- SWE-bench Pro
- Reward hacking과 specification gaming 논문
- AgentBench
- WebDev와 frontend benchmark 논문
- Red Queen Godel Machine
- WorkArena와 long-horizon agent evaluation


# 8. Summary

- Coding agent verification은 generation보다 어려워지고 있다.
- Policy capability가 커질수록 fixed reward function은 계속 reliable하게 남기 어렵다.
- Verification quality에는 scalability, faithfulness, robustness라는 세 축이 있다.
- Tests, rubric judges, users, agentic evaluators는 intent의 서로 다른 부분을 cover한다.
- Reward infrastructure는 policy model과 함께 co-evolve해야 한다.
