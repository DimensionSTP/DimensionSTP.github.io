---
layout: single
title: "RetireOPD: Self-Retiring On-Policy Distillation for Agentic Reinforcement Learning Review"
categories: Study-concept
tag: [RetireOPD, OnPolicyDistillation, AgenticRL, GRPO, KnowledgeDistillation]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2609.20784)

[Official repository](https://github.com/ZJU-REAL/SDAR)

[Retirement scheduler](https://github.com/ZJU-REAL/SDAR/blob/308153bb5f7d63cd57f301f432d616a072220e1b/verl/trainer/ppo/retire_opd.py)

> 한 줄 요약: RetireOPD는 agentic RL에서 teacher의 도움을 무조건 끝까지 유지하지 않고, teacher 정렬의 진전과 student의 과제 수행 능력을 함께 관찰해 OPD를 종료한 뒤 환경 보상 기반 학습을 계속한다.

On-policy distillation은 student가 실제로 방문한 상태와 생성한 token에서 teacher의 신호를 받을 수 있다는 장점이 있다. 그런데 student가 충분히 학습한 뒤에도 같은 teacher를 계속 따라가야 할까? 초기 학습을 도운 신호가 이후의 환경 보상 최적화와 항상 같은 방향이라고 보장할 수는 없다.

RetireOPD는 이 질문을 distillation coefficient를 언제 줄일지의 스케줄 문제로만 다루지 않는다. **Teacher를 따라가는 과정이 여전히 진전되고 있는지와 student가 이미 충분한 과제 수행 능력을 확보했는지를 함께 본다.** 두 조건이 충족되면 OPD 항을 끄고, student는 RL을 계속한다. [S1], [S2]

이 리뷰는 저자 초록과 공식 저장소의 결과 표, 학습 스크립트, retirement scheduler 및 trainer 코드를 기반으로 한다. 논문 PDF 전문은 직접 확보하지 못했으므로, 아래의 정확한 종료 기준 수식은 원문 식을 전사한 것이 아니라 명시한 공개 코드의 계산을 정리한 것이다.

# 1. Problem Setting

## 1-1. 환경 보상과 teacher 모방은 같은 목표가 아니다

Agentic RL에서는 여러 turn의 관찰과 행동을 거쳐 환경 보상을 얻는다. OPD는 같은 학습 과정에 teacher의 token-level 신호를 더한다. 두 목적의 결합을 설명하기 위한 일반적인 손실 표기는 다음과 같다.

<div class="math-block" markdown="0" style="overflow-x: auto;">
$$
\mathcal{L}_{\mathrm{total}}(k)
=
\mathcal{L}_{\mathrm{RL}}(k)
+
\beta_k\mathcal{L}_{\mathrm{OPD}}(k)
$$
</div>

여기서 $k$는 학습 step, $\beta_k$는 OPD의 기여를 조절하는 계수다. 이 식은 두 목적의 역할을 구분하기 위한 개념식이며, 공개 구현의 모든 clipping 및 advantage 계산을 생략한 표기다.

환경 보상은 실제 과제 성공을 지향한다. 반면 OPD는 teacher와 student 사이의 차이를 줄이는 방향을 제공한다. 초기에는 이 차이가 유용한 지침이 될 수 있지만, 이후에는 고정된 teacher를 따라가는 것이 더 나은 행동을 찾는 것과 일치하지 않을 수 있다.

핵심은 teacher가 나쁘다는 주장이 아니다. **Teacher의 유용성이 학습 전 구간에서 일정하다고 가정하지 않는 것**이다.

## 1-2. 고정 스케줄이 놓치는 것

미리 정한 step에서 OPD를 끄면 구현은 간단하다. 하지만 student가 학습하는 속도와 task 난이도가 달라져도 같은 시점에 종료된다. 너무 일찍 끄면 유용한 신호를 버리고, 너무 늦게 끄면 더 이상 필요하지 않은 정렬 압력과 계산 비용을 유지한다.

Teacher와의 차이가 더 이상 줄지 않는다는 조건 하나만으로 종료하는 것도 충분하지 않다. Student가 아직 과제를 잘 수행하지 못하는 상태에서 학습이 정체될 수도 있기 때문이다. 반대로 성공률이 어느 수준에 도달했다는 이유만으로 종료하면, 아직 유용한 distillation의 진전을 버릴 수 있다.

RetireOPD는 이 두 종류의 오판을 함께 줄이기 위해 alignment와 competence를 결합한다. [S1], [S3]

# 2. Core Idea

## 2-1. Teacher를 먼저 준비하고 종료 판단은 student 학습 중에 한다

공식 README의 실행 절차는 두 단계다. 먼저 skill-conditioned teacher를 환경에서 학습한다. 이후 teacher checkpoint와 해당 환경의 validation 성능을 지정해 RetireOPD student 학습을 시작한다. Student의 RL과 OPD가 함께 진행되다가 종료 조건을 만족하면 OPD를 제거한다. [S2]

이때 teacher 성능은 단순한 참고 숫자가 아니다. Student가 어느 정도 competence를 확보했는지 판단하는 기준에 직접 들어간다. Teacher와 student의 validation metric이 일치해야 하는 이유다.

이 구조를 동일 모델의 현재 출력으로 자기 자신을 계속 따라가는 방식과 혼동하지 않는 것이 좋다. 공개 workflow에서는 먼저 준비한 teacher checkpoint를 별도로 지정한다.

## 2-2. 종료는 성능 정체에 대한 단순 반응이 아니다

현재 구현의 종료 판단은 두 질문의 결합이다. 최근 두 관찰 구간 사이에서 teacher 정렬이 진전되었는가? 최근 validation에서 student는 teacher 기준으로 충분한 성능에 도달했는가?

첫 질문은 distillation을 계속할 이유가 남아 있는지 본다. 둘째 질문은 distillation을 종료해도 될 만큼 student가 준비되었는지 본다. 둘 다 충족될 때만 종료한다. [S3]

## 2-3. OPD의 종료와 학습의 종료를 구분한다

Retirement 이후 RL은 계속된다. 종료되는 것은 학습 전체가 아니라 OPD 보조 항이다. 현재 trainer는 retirement 상태에서 training teacher forward도 건너뛴다. 계수만 0으로 만들고 teacher 계산을 그대로 수행하는 구현과 구분되는 부분이다. [S4]

이것이 효율에 기여할 수 있는 직접적인 경로다. 다만 teacher 준비 단계와 retirement 이전의 비용은 그대로 남는다. 최종 비용을 비교할 때는 종료 이후의 절약만이 아니라 전체 실험의 예산을 계산해야 한다.

# 3. Architecture / Method

## 3-1. 공개 구현의 구성 요소

| 구성 요소 | 역할 |
| --- | --- |
| Skill-conditioned teacher | 별도 학습한 checkpoint로 student에 OPD 신호를 제공한다. |
| Student policy | 환경 rollout을 만들고 RL 및 OPD로 업데이트된다. |
| Gap 관찰 | Teacher-student log-probability 차이의 평균을 기록한다. |
| Window 집계 | 고정 길이의 겹치지 않는 구간별 통계를 만든다. |
| Competence 측정 | Student validation 성능을 teacher 기준과 비교한다. |
| Retirement scheduler | 두 조건이 충족되면 OPD 계수를 0으로 고정한다. |
| Trainer hook | Teacher forward 생략과 상태 저장 및 복원을 연결한다. |

이 역할 분담은 [공식 README][S2], [scheduler][S3], [trainer][S4]를 기준으로 정리한 것이다.

## 3-2. Alignment 통계를 어떻게 계산하는가

Scheduler에 전달되는 step별 teacher-student gap 평균을 $g_k$라고 하자. 공개 구현은 먼저 그 부호를 바꿔 다음 통계를 만든다.

<div class="math-block" markdown="0" style="overflow-x: auto;">
$$
K_k = -g_k
$$
</div>

기본 window 크기는 5다. $w$개의 step으로 구성한 겹치지 않는 구간 $\mathcal{W}_m$에 대해 다음 평균을 계산한다.

<div class="math-block" markdown="0" style="overflow-x: auto;">
$$
\bar{K}_m = \frac{1}{w}\sum_{k\in\mathcal{W}_m}K_k
$$
</div>

이후 validation 시점에 이전 구간과 현재 구간을 비교한다. 공개 코드의 alignment progress는 다음과 같다. [S3]

<div class="math-block" markdown="0" style="overflow-x: auto;">
$$
P_m = \frac{\bar{K}_{m-1}-\bar{K}_m}{\bar{K}_{m-1}}
$$
</div>

이전 값이 양의 거리처럼 해석되는 구간에서는 통계가 줄면 양의 progress가 된다. 기본 alignment threshold는 0이므로, 더 이상 감소하지 않는 경우가 retirement의 한 조건이다.

여기서 $K_k$를 매 step의 정확한 KL divergence라고 단정하지 않는 것이 중요하다. 코드가 사용하는 것은 sampled token의 log-probability gap에서 만든 관측 통계다. Finite sample과 masking, 정책 시점에 따라 값의 해석에 주의가 필요하다. 구현은 분모의 절댓값이 `1e-8`보다 큰지와 계산 결과가 유한한지도 확인한다. [S3]

## 3-3. Competence를 어떻게 판단하는가

이전과 현재 validation 성능을 각각 $S_{m-1}$, $S_m$, 지정한 teacher 성능을 $S_T$라고 하자. 공개 구현은 최근 두 성능의 평균을 teacher 성능으로 나눈 값을 사용한다.

<div class="math-block" markdown="0" style="overflow-x: auto;">
$$
C_m = \frac{S_{m-1}+S_m}{2S_T}
$$
</div>

기본 competence threshold는 0.9다. 즉 이 비율이 0.9 이상인지 확인한다. 이는 "student 성공률이 90% 이상"이라는 조건이 아니다. Teacher 성능에 대한 상대 기준이다. [S3]

예를 들어 teacher의 같은 metric 값이 0.8이라면, 이 비율의 기준 0.9는 최근 두 student 성능의 평균이 0.72 이상인지에 대응한다. 이 숫자는 조건의 의미를 설명하기 위한 가상 예시이며 논문에서 보고한 teacher 성능이 아니다.

## 3-4. 두 조건을 결합한 상태 전환

기본 설정에서 retirement는 유효한 통계를 얻은 상태에서 다음 두 조건을 동시에 만족할 때 발생한다.

<div class="math-block" markdown="0" style="overflow-x: auto;">
$$
P_m \le 0
\qquad\text{and}\qquad
C_m \ge 0.9
$$
</div>

한 번 발생하면 `retired=True`가 되고 OPD 계수는 0으로 설정된다. 현재 scheduler에는 이후 성능이 떨어졌을 때 teacher를 자동으로 다시 활성화하는 전환이 없다. 이 점에서 매 step마다 켜졌다 꺼지는 gate와 다르다. [S3]

또한 window는 겹치지 않는 구간으로 집계한다. 매 step마다 같은 표본을 대부분 공유하는 이동평균을 갱신하는 방식으로 설명하면 구현과 다르다. 구간의 완성과 validation의 주기를 함께 기록해야 언제 어떤 관측이 종료 판단에 쓰였는지 재현할 수 있다.

## 3-5. Retirement 이후 teacher forward를 생략한다

공개 trainer의 `_get_training_teacher_log_probs`는 retirement가 일어나면 teacher 모델을 다시 호출하지 않는다. 대신 후속 계산 경로를 유지하기 위한 placeholder를 반환하고, teacher forward를 생략했다는 metric을 기록한다. [S4]

따라서 논문의 아이디어를 재현할 때는 coefficient schedule뿐 아니라 실행 경로도 확인해야 한다. 보조 손실을 수학적으로 0으로 만든 것과, 그 손실 계산을 위한 비용을 실제로 없앤 것은 다르다.

## 3-6. Scheduler 상태도 checkpoint의 일부다

Scheduler는 window의 진행 상태, 완료된 구간, 이전 평균과 validation 성능, retirement 여부, 발생 step, 현재 계수를 저장한다. 복원할 때 configuration의 일치 여부도 검사한다. [S3]

이는 단순한 운영 편의가 아니다. 모델 가중치만 복원하고 window 통계를 초기화하면, 중단하지 않은 학습과 다른 시점에 teacher를 종료할 수 있다. 따라서 이 방법의 재현 단위는 policy checkpoint만이 아니라 policy와 retirement 상태의 결합이다.

# 4. Training / Data / Recipe

## 4-1. Teacher 준비 단계의 비용과 기준

공식 workflow는 먼저 `skill_grpo_trainer`의 스크립트로 teacher를 학습하고, 필요한 경우 sharded checkpoint를 Hugging Face 형식으로 병합하도록 안내한다. 그다음 선택한 환경에서 teacher의 validation score를 기록한다. [S2]

README의 `TEACHER_PERFORMANCE=0.50`은 사용자가 실제 값으로 바꿔야 하는 예제다. 이를 논문에서 얻은 teacher 성능으로 인용하면 안 된다. 이 값은 scheduler의 competence 계산 분모로 사용되므로, 잘못 입력하면 종료 시점 자체가 바뀐다.

Student와 teacher가 다른 데이터나 다른 채점 규칙으로 평가되었다면 상대 기준의 의미가 약해진다. 실제 적용에서는 평가 split, 환경 버전, sampling 조건, metric 방향을 함께 고정하는 편이 안전하다. 이는 구현의 입력 의존성에서 도출되는 재현 주의점이다.

## 4-2. 공개 ALFWorld 3B 설정

아래는 현재 공개된 ALFWorld 3B 실행 스크립트의 기본값이다. 논문 전체의 모든 환경 설정을 대표하지 않는다. [S5]

| 항목 | 기본값 |
| --- | --- |
| Student 초기 모델 | Qwen2.5-3B-Instruct |
| Actor learning rate | `1e-6` |
| OPD coefficient | 0.01 |
| Window 크기 | 5 |
| Alignment threshold | 0.0 |
| Competence threshold | 0.9 |
| Validation metric | `val/success_rate` |
| 최대 prompt 길이 | 2,048 tokens |
| 한 번의 응답 생성 최대 길이 | 512 tokens |
| 환경의 최대 step 수 | 50 |
| 기본 GPU 수 | 8 |

응답의 최대 길이 512와 환경의 최대 50 steps는 다른 제한이다. 전체 agent trajectory가 512 tokens라는 뜻으로 읽지 않는다. 여러 turn을 거치는 환경에서는 각 응답 길이와 전체 episode 길이를 따로 계산해야 한다.

## 4-3. 실험 로그에 남길 정보

최종 성공률만 저장하면 retirement가 왜 일어났는지 알기 어렵다. 최소한 구간별 $\bar{K}_m$, alignment progress, student와 teacher의 validation 성능, competence ratio, trigger step, teacher forward 생략 여부를 함께 기록해야 한다.

Retirement 시점을 비교할 때는 step뿐 아니라 누적 token과 환경 interaction 수도 보는 것이 좋다. 같은 step 수라도 response length가 바뀌면 실제 학습 및 추론 비용이 달라질 수 있다.

이 두 문단은 재현을 위한 제안이다. 논문이 모든 비용 축을 이미 완전하게 보고했다고 주장하는 것은 아니다.

# 5. Evaluation

## 5-1. 공식 저장소의 RetireOPD 결과

아래는 공식 README에서 RetireOPD 행으로 명시한 성공률이다. SDAR의 다른 실험 결과와 섞지 않았다. [S2]

| 초기 모델 | ALFWorld 성공률 (%) | WebShop 성공률 (%) |
| --- | ---: | ---: |
| Qwen2.5-1.5B-Instruct | 89.8 | 75.8 |
| Qwen2.5-3B-Instruct | 93.8 | 77.3 |
| Qwen2.5-7B-Instruct | 95.3 | 84.4 |

이 표는 공개된 절대 성능을 보여준다. 하지만 baseline의 동일 조건 점수와 오차 범위를 이 초안에서 원문 대조하지 못했으므로, 여기서 상대 향상률이나 percentage-point 개선량을 만들어내지 않는다.

저장소가 Search-QA를 지원한다는 사실 역시 RetireOPD가 같은 표에서 Search-QA 결과를 보고했다는 뜻은 아니다. Framework의 지원 범위와 특정 방법의 평가 범위를 구분해야 한다.

## 5-2. 최종 성능 외에 분리해야 할 효과

RetireOPD의 성능에는 적어도 teacher 준비, 초기 OPD, 종료 기준, 종료 이후 RL이 함께 관여한다. 최종 점수가 좋다는 사실만으로 두 조건의 결합이 전부 기여했다고 결론 내릴 수 없다.

그 기여를 보기 위해서는 계속 OPD를 유지한 조건, 미리 정한 시점에 종료한 조건, alignment만으로 종료한 조건, competence만으로 종료한 조건을 비교할 필요가 있다. 이 목록은 method의 효과를 분리하기 위한 검토 틀이며, 원문 ablation의 모든 행을 확인했다는 의미는 아니다.

특히 같은 total training budget에서의 비교와 같은 teacher 호출 예산에서의 비교는 다른 질문에 답한다. 전자는 최종적인 학습 효율을, 후자는 teacher 신호를 얼마나 유효하게 사용했는지 보여준다.

## 5-3. Teacher를 종료한 뒤 얼마나 좋아지는가

가장 흥미로운 관찰 구간은 trigger step 전후다. Retirement 직후 성능이 유지되는지, 잠깐 떨어졌다가 회복되는지, 이후 teacher의 성능을 넘는지, seed에 따라 종료 시점이 크게 달라지는지를 함께 보아야 한다.

단순히 student가 teacher 수준에 도달했다고 해서 종료가 원인이 되어 이후 성능이 개선되었다고 단정할 수는 없다. 같은 시점에서 OPD를 유지했을 반사실적 비교가 필요하다. 이런 비교가 있어야 종료 전략의 효과와 원래 RL의 진행 효과를 구분할 수 있다.

# 6. Limitations

첫째, competence 기준은 teacher의 validation 성능에 의존한다. Teacher 기준값이 부정확하거나 환경 및 채점 규칙이 달라지면 종료 판단이 왜곡될 수 있다. 코드도 teacher 성능이 양의 유한한 값이어야 한다고 검사한다. [S3]

둘째, alignment 통계의 정체가 teacher의 유용성 소진만을 의미하는 것은 아니다. Sampling 변동, 데이터 변화, optimization의 일시적 정체가 섞일 수 있다. 구간 집계와 competence 조건은 이런 위험을 완화하려는 구조지만, 모든 분포 변화에 대한 보장은 아니다.

셋째, 현재 공개 구현의 retirement는 되돌리지 않는다. 이후 task distribution이 바뀌어 teacher가 다시 유용해지는 상황까지 자동으로 처리하는 일반적인 teacher lifecycle 관리와는 범위가 다르다. [S3]

넷째, 종료 이후 teacher forward를 줄이는 것은 분명한 실행 경로 변화지만, teacher를 먼저 학습한 비용까지 사라지는 것은 아니다. 새 프로젝트에 적용할 때는 전체 비용과 teacher 재사용 횟수를 함께 보아야 한다.

마지막으로 PDF 전문을 확보하지 못해 baseline, error bar, 세부 ablation 및 비용 비교표를 직접 대조하지 못했다. 이 한계 때문에 이 글의 성능 해석은 공식 README에서 확인 가능한 절대 결과와 공개 구현의 동작으로 제한했다.

# 7. My Take

## 7-1. 증류는 항상 켜 둘 정규화가 아닐 수 있다

RetireOPD를 읽는 유용한 관점은 teacher를 영구적인 정답 기준이 아니라 학습 단계에 따라 가치가 달라지는 보조 신호로 보는 것이다. 초기에는 탐색과 학습을 돕지만, student가 준비된 뒤에는 환경 보상에 더 집중하는 선택이 가능하다.

이는 단순한 coefficient tuning과 닮았지만 동일하지 않다. 종료 결정을 외부에서 미리 정하지 않고, 정렬의 진행 상황과 과제 수행 능력을 함께 사용한다는 점이 method의 중심이다.

## 7-2. 재사용하기 쉬운 부분은 상태기계다

전체 agent 환경을 그대로 가져오지 않더라도 scheduler의 입력과 전환 구조는 참고할 수 있다. 다만 다른 task에 적용할 때는 로그의 부호, 평균의 단위, metric의 방향과 teacher 기준값부터 다시 정의해야 한다.

특히 성능이 낮을수록 좋은 loss나 error metric을 success rate처럼 그대로 competence 비율에 넣으면 의미가 뒤집힐 수 있다. 공개 스크립트가 성공률을 쓰는 맥락을 유지하고, 다른 metric으로 바꿀 때는 기준 자체를 다시 설계해야 한다.

## 7-3. 다른 OPD 연구와의 연결

[SDAR][S6]는 같은 저장소의 기반 연구이며, [공식 README][S2]는 여러 agentic distillation 후속 연구를 구분해 나열한다. 그중 RetireOPD의 질문은 "어떤 teacher 신호를 줄 것인가"보다 "그 신호를 언제 그만 사용할 것인가"에 가깝다.

다음 리뷰의 Cal-OPD는 teacher-student discrepancy에 섞인 teacher 자체의 편차를 교정하려는 연구다. 하나는 신호의 사용 기간, 다른 하나는 신호의 내용을 다룬다. 두 방법을 결합할 수 있다는 것은 흥미로운 후속 가설이지만, 이 글에서 결합 실험이 검증되었다고 주장하지는 않는다.

# 8. Summary

- RetireOPD는 agentic RL에서 OPD를 종료할 시점을 학습 중 관측으로 결정한다.
- Alignment 진전이 멈췄는지와 student가 teacher 대비 충분한 competence를 확보했는지를 함께 본다.
- 공개 구현은 겹치지 않는 window를 사용하며, 두 조건을 만족하면 OPD 계수를 0으로 고정한다.
- Retirement 이후 RL은 계속되고 training teacher forward는 생략된다.
- 재현에서는 teacher 기준값, validation 조건, scheduler checkpoint와 전체 학습 비용이 중요하다.

[S1]: https://arxiv.org/abs/2609.20784
[S2]: https://github.com/ZJU-REAL/SDAR/blob/308153bb5f7d63cd57f301f432d616a072220e1b/README.md
[S3]: https://github.com/ZJU-REAL/SDAR/blob/308153bb5f7d63cd57f301f432d616a072220e1b/verl/trainer/ppo/retire_opd.py
[S4]: https://github.com/ZJU-REAL/SDAR/blob/308153bb5f7d63cd57f301f432d616a072220e1b/verl/trainer/ppo/retire_opd_trainer.py
[S5]: https://github.com/ZJU-REAL/SDAR/blob/308153bb5f7d63cd57f301f432d616a072220e1b/examples/retireopd_trainer/run_alfworld_3b.sh
[S6]: https://arxiv.org/abs/2605.15155
