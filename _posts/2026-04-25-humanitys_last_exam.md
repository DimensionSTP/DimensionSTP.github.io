---
layout: single
title: "Humanity's Last Exam Review"
categories: Study-concept
tag: [HLE, Benchmark, Evaluation]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://www.nature.com/articles/s41586-025-09962-4)

요즘 frontier LLM을 평가할 때 점점 더 애매해지는 지점이 있다. **쉬운 벤치마크는 이미 너무 쉬워졌고**, 반대로 실제로 어려운 문제를 보려고 하면 **평가가 비공개이거나, 특정 도메인에 치우치거나, judge 설계에 크게 의존**하는 경우가 많다. 그래서 이제 중요한 질문은 "모델이 MMLU를 몇 점 맞았나"가 아니라, **지금 시점의 모델이 인간 전문가의 지식 frontier와 얼마나 떨어져 있나를 어떻게 측정할 것인가**에 더 가깝다.

Humanity's Last Exam(HLE)은 바로 그 질문을 정면으로 다루는 논문이다. 이 논문의 진짜 가치는 "세상에서 제일 어려운 문제를 모았다"는 데 있지 않다. 오히려 **전문가가 문제를 쓰고, 현재 frontier 모델로 난이도를 먼저 거르고, 다단계 human review를 거친 뒤, private holdout과 calibration 측정까지 포함한 평가 체계**를 만들었다는 데 있다. 즉 HLE는 단순 benchmark release라기보다, **expert-level closed-ended benchmark를 어떻게 운영할 것인가에 대한 설계 문서**에 가깝다.

> 한 줄 요약: HLE는 2,500개의 expert-level closed-ended question을 over-100 subject에 걸쳐 모으고, frontier-model difficulty filtering + expert review + private holdout + calibration-aware evaluation을 결합해 **현재 LLM과 인간 전문가 frontier 사이의 gap**을 측정하려는 benchmark다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- 기존 broad benchmark가 너무 빨리 포화되는 시점에서, **"어려움 자체"가 아니라 어려움을 만드는 수집/검수/운영 구조**를 보여주기 때문이다.
- 단순 정확도만이 아니라 **confidence를 같이 받아 calibration까지 측정**한다. 즉 "맞히는가"뿐 아니라 "모를 때 모른다고 아는가"를 같이 본다.
- Nature 공개본과 공식 사이트 운영 흐름을 같이 보면, HLE는 고정된 시험지가 아니라 **post-release refinement와 HLE-Rolling까지 붙는 living benchmark**로 읽는 편이 더 맞다.

내가 보기엔 이 논문은 "모델이 박사급 문제를 풀 수 있는가"를 넘어서, **frontier eval은 결국 question engineering + review ops + post-release maintenance의 문제**라는 사실을 잘 보여준다. 동시에 HLE가 측정하는 것은 어디까지나 **closed-ended academic capability**이지, open-ended research나 agentic scientific discovery 그 자체는 아니라는 점도 같이 봐야 한다.

# 1. Problem Setting

## 1-1. Problem definition

- 이 논문이 겨냥하는 핵심 문제는 **기존 LLM benchmark가 frontier capability를 측정하기에 너무 쉬워지고 있다**는 점이다.
- broad benchmark는 여전히 유용하지만, state-of-the-art 모델이 높은 점수를 내기 시작하면 **누가 얼마나 더 강한지**를 분해해서 보기 어려워진다.
- 반대로 정말 어려운 문제를 평가하려고 하면 보통 다음 셋 중 하나가 된다.
  1. 특정 도메인에만 강하게 치우친 benchmark
  2. 내부 안전평가처럼 외부 재현이 어려운 private eval
  3. open-ended 판단이 필요해서 judge 설계에 많이 의존하는 평가
- HLE는 이 중간을 노린다. 즉 **broad subject coverage를 유지하면서도**, **expert-level difficulty를 확보하고**, **자동 채점 가능한 형식**으로 benchmark를 만드는 것이 목표다.
- 그래서 HLE의 문제 설정은 "어려운 시험지 만들기"가 아니라, **현대 모델이 쉽게 외워 풀 수 없고, 인터넷 검색으로도 즉시 해결되지 않으며, 그래도 자동 채점은 가능한 문제를 대규모로 모으는 것**에 가깝다.

## 1-2. Why previous approaches are insufficient

- 기존 broad academic benchmark는 범위는 넓지만, 이미 공개된 지 오래되었거나 웹 전반에 널리 퍼져 있어 **data contamination**을 피하기 어렵다.
- GPQA 같은 harder benchmark는 유익하지만, 상대적으로 **subject coverage가 좁거나 benchmark 성격이 한쪽으로 기운다**.
- 반대로 open-ended evaluation은 실제 사용 시나리오와 가깝지만, 사람 평가나 LLM judge를 써야 해서 **정답성보다 스타일, 길이, 표현 방식이 점수에 개입**하기 쉽다.
- 단순히 "더 어렵게 만들자"만으로도 충분하지 않다. closed-ended frontier question은 본질적으로 만들기 어렵고, **문제가 모호해지거나 정답 검증이 힘들어지는 순간 benchmark 신뢰성 자체가 흔들리기 때문**이다.
- 결국 기존 접근의 한계는 단순히 난도가 낮다는 것이 아니라, **breadth / difficulty / objectivity / freshness / maintainability**를 동시에 설계한 benchmark가 드물다는 데 있다.

# 2. Core Idea

## 2-1. Main contribution

- HLE의 핵심 기여는 하나의 모델 성능표가 아니라, **expert-written frontier benchmark construction recipe**를 제시했다는 점이다.
- 구체적으로는 다음 다섯 축이 핵심이다.
  1. **광범위한 subject coverage**: 100개가 넘는 subject, 8개 high-level category
  2. **전문가 중심 문제 수집**: 전 세계 nearly 1,000명 규모의 subject expert contributor 참여
  3. **frontier model filtering**: 현재 강한 모델들이 먼저 풀어보게 하고, 쉽게 맞히는 문제는 탈락
  4. **다단계 human review**: graduate-level reviewer와 organizer approval을 거쳐 closed-ended quality를 높임
  5. **정확도 + calibration 측정**: answer뿐 아니라 confidence까지 받아 model의 uncertainty handling도 측정
- 즉 이 논문은 "정답률 낮은 benchmark"를 만든 게 아니라, **정답률이 낮을 수밖에 없도록 benchmark construction pipeline을 설계한 것**에 가깝다.

## 2-2. Design intuition

- 이 논문의 설계 직관은 꽤 분명하다. frontier eval에서 진짜 중요한 건 단순히 obscure fact를 묻는 것이 아니라, **전문가가 보기에 분명히 답이 있는 문제를 모델이 아직 잘 못 푼다**는 점을 보여주는 것이다.
- 그래서 HLE는 자유서술형 연구 제안이나 창의적 에세이를 평가하지 않는다. 대신 **exact-match와 multiple-choice**처럼 채점 가능한 형식을 택한다.
- 하지만 형식이 닫혀 있다고 해서 쉬운 문제를 의미하지는 않는다. 오히려 HLE는 research-level background, niche domain knowledge, hard math reasoning, image understanding을 묻는 식으로 **문제의 본질적 난도는 높게 유지하고, 답 형식만 닫힌 형태로 정리**한다.
- 여기에 confidence를 요구하는 것도 인상적이다. 단순 accuracy만 보면 "찍어서 맞힌 것"과 "정말 아는 것"을 분리하기 어렵지만, calibration을 보면 **틀리는데도 자신만만한 모델**을 드러낼 수 있다.
- 내 해석으로는 HLE의 진짜 기여는 문제 bank가 아니라, **frontier benchmark를 closed-ended form으로 압축하는 질문 설계 감각**이다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | broad subject coverage를 유지하면서 expert-level closed-ended academic capability를 측정하는 것 |
| Question interface | exact-match + multiple-choice + 일부 multi-modal image question |
| Difficulty control | frontier LLM pre-check + two-stage expert review |
| Evaluation harness | structured answer format + confidence elicitation + LLM judge for equivalence checking |
| Maintenance design | public release + private holdout + post-release bug bounty + HLE-Rolling |
| Difference from prior work | harder question set 자체보다 **benchmark construction and maintenance pipeline**를 드러냄 |

## 3-2. Module breakdown

### 1) Closed-ended question interface

- HLE는 두 가지 question type을 사용한다.
  - **exact-match**: 모델이 짧고 정확한 문자열 형태의 답을 내야 하는 문제
  - **multiple-choice**: 다섯 개 이상 보기 중 하나를 고르는 문제
- 전체 문제 중 약 **14%는 text + image 이해가 함께 필요한 multi-modal question**이고, 약 **24%는 multiple-choice**, 나머지는 exact-match다.
- 중요한 건 이 형식이 benchmark를 쉽게 만들기 위한 타협이 아니라는 점이다. HLE는 오히려 **research-level question을 자동 채점 가능한 형식으로 재정의**하는 쪽에 가깝다.
- exact-match 답을 짧고 검증 가능하게 유지하려는 정책도 일관적이다. 모델이 길게 서술해서 어물쩍 넘어가는 것을 막고, **최종 정답 판정이 가능한 closed-ended interface**를 강제한다.

### 2) Contributor and incentive design

- 이 benchmark는 소수 저자가 직접 전부 작성한 문제가 아니다.
- nearly 1,000명의 subject expert contributor가 500개 이상 기관, 50개국에 걸쳐 참여한다.
- 이 구조의 장점은 분명하다. broad benchmark를 유지하면서도 각 question의 domain depth를 확보할 수 있다.
- 여기에 **USD 500,000 prize pool**을 걸어 고품질 기여를 유도한 것도 독특하다.
  - top 50 question: 5,000달러
  - next 500 question: 500달러
- 이 설계는 benchmark를 빠르게 키우는 데 효과적이지만, 동시에 **"정말 좋은 question"과 "모델이 맞히기 어렵지만 검증도 어려운 question" 사이의 긴장**을 낳을 수 있다는 점도 같이 읽어야 한다.

### 3) Frontier-model difficulty filter

- HLE의 가장 특징적인 부분 중 하나는 **사람 review 전에 모델이 먼저 문제를 푼다**는 점이다.
- 질문이 제출되면 frontier LLM으로 난이도를 점검하고, 모델이 쉽게 맞히는 문제는 다음 단계로 가지 못한다.
- exact-match question은 매우 엄격하게, multiple-choice는 **lucky guess floor를 감안해 조금 더 완화된 기준**으로 걸러낸다.
- 이 방식의 의미는 단순하다. benchmark difficulty를 추상적으로 논하지 않고, **현재 frontier model 기준으로 실제로 어려운 문제만 남긴다**는 것이다.
- 논문 기준으로 이 단계에서만 **70,000회 이상의 시도**가 기록됐고, 그 결과 약 **13,000개 문제**가 human expert review 단계로 넘어갔다.

### 4) Two-stage expert review

- LLM filter를 통과했다고 해서 바로 dataset에 들어가지는 않는다.
- review는 최소 두 층으로 이루어진다.
  1. **초기 피드백 라운드**: graduate-level reviewer들이 question quality, closed-endedness, ambiguity를 점검
  2. **최종 승인 라운드**: organizer와 trained expert reviewer가 채택 여부를 최종 결정
- 첫 라운드에서 각 question은 보통 **1~3회의 리뷰**를 받는다.
- 이 단계의 핵심은 단순 fact check가 아니다. 오히려 "이 질문이 정말 closed-ended인가?", "정답이 객관적인가?", "web search로 너무 쉽게 풀리지 않는가?" 같은 **benchmarkability** 자체를 가다듬는 데 있다.
- 즉 HLE는 question writing보다 **question shaping**에 더 많은 에너지를 쓴 benchmark라고 보는 편이 맞다.

### 5) Evaluation harness and answer checking

- 모델 평가 시 HLE는 단순 자유형 응답을 받지 않는다.
- 시스템 프롬프트를 통해 모델 출력 형식을 아래처럼 고정한다.
  - Explanation
  - Answer
  - Confidence
- 이렇게 하는 이유는 두 가지다.
  1. **최종 답 파싱**을 쉽게 하기 위해서
  2. **calibration 측정**을 위해 confidence를 수집하기 위해서
- 정답 판정은 단순 string exact match로 끝나지 않는다. Nature 공개본 기준으로는 **o3-mini를 judge로 사용해**, fraction과 decimal처럼 표현이 다른 경우나 소규모 수치 오차를 고려해 equivalence를 판단한다.
- 이건 꽤 중요한 설계다. HLE는 exact-match benchmark이지만, 실제 판정은 **"표현이 달라도 같은 답인가"를 추론할 수 있는 judge model**을 끼워 넣어 false negative를 줄이려 한다.

### 6) Post-release refinement and HLE-Rolling

- 이 논문의 최근 버전에서 특히 인상적인 부분은 **post-release maintenance를 method의 일부로 받아들인다**는 점이다.
- 공개 이후 benchmark team은 다음을 수행한다.
  - community feedback bug bounty
  - searchable question audit
  - 추가 late contribution review
  - future overfitting 점검용 second held-out private set 구축
- 더 나아가 논문은 **HLE-Rolling**이라는 dynamic fork를 예고한다. 이는 community feedback과 새 question을 반영해 계속 갱신되는 버전이다.
- 내 해석으로는 여기서 HLE는 더 이상 "정적인 시험지"가 아니라, **frontier eval dataset을 운영하는 서비스**에 가까워진다.

# 4. Training / Data / Recipe

이 논문은 모델 학습 논문이 아니므로, 여기서는 일반적인 training recipe 대신 **dataset construction recipe**로 읽는 편이 맞다.

## 4-1. Data

- HLE의 공개 benchmark는 **총 2,500문항**이다.
- subject는 100개가 넘고, high-level category는 8개다.
- category 분포는 아래처럼 꽤 치우쳐 있다.

| Category | Share |
| --- | ---: |
| Math | 41% |
| Biology / Medicine | 11% |
| Computer Science / Artificial Intelligence | 10% |
| Physics | 9% |
| Humanities / Social Science | 9% |
| Other | 9% |
| Chemistry | 7% |
| Engineering | 4% |

- 이 표만 봐도 HLE의 character가 선명하다. **broad benchmark이지만, 완전히 균등한 분포는 아니고 math/STEM 쪽 비중이 높다.**
- question style 측면에서는,
  - 약 14%가 multimodal
  - 약 24%가 multiple-choice
  - 나머지가 exact-match다.
- 즉 HLE는 VLM capability도 약간 포함하지만, 본질적으로는 **expert-level text-centric academic reasoning benchmark**에 더 가깝다.

## 4-2. Dataset construction strategy

- contributor pool은 매우 넓지만, question acceptance 기준은 보수적이다.
- 제출된 question은 다음 원칙을 충족해야 한다.
  - precise
  - unambiguous
  - solvable
  - non-searchable
  - original work 혹은 non-trivial synthesis
- 특히 흥미로운 건 **직접적인 연구 경험에서 나온 질문도 허용한다**는 점이다.
- 이 설계는 benchmark를 단순 textbook QA보다 훨씬 frontier 쪽으로 밀어주지만, 동시에 외부 검증을 어렵게 만드는 원인이기도 하다.
- construction pipeline을 거칠게 정리하면 이렇다.
  1. 전문가가 문제 제출
  2. frontier model difficulty check
  3. human review와 refinement
  4. organizer / expert reviewer approval
  5. public set과 private held-out set으로 분리
- 이 과정은 benchmark를 단순 static dataset이 아니라, **difficulty-filtered and peer-reviewed question funnel**로 만든다.

## 4-3. Engineering notes

- HLE의 engineering에서 가장 중요한 포인트는 **답 형식과 정답 판정 인터페이스를 강하게 통제한다**는 점이다.
- exact-match benchmark는 사소한 formatting 차이에도 오답이 나기 쉬운데, HLE는 이를 o3-mini judge로 완화한다.
- 또 평가 프롬프트가 Explanation / Answer / Confidence를 강제하기 때문에, 모델 성능은 pure capability만이 아니라 **structured output compliance**의 영향을 일부 함께 받는다.
- evaluation table을 읽을 때도 주의할 점이 있다.
  - Nature 공개본은 **post-release model**을 따로 분리한다.
  - 이유는 benchmark가 이미 public이 된 뒤의 모델은, 직접적 contamination이 아니더라도 **모델 빌더가 benchmark 존재를 알고 최적화할 가능성**을 배제하기 어렵기 때문이다.
- 이 분리는 HLE를 해석할 때 매우 중요하다. 같은 leaderboard라도 **공개 전과 공개 후 score는 같은 의미가 아니다.**

# 5. Evaluation

## 5-1. Main results

- HLE의 headline result는 단순하다. **강한 모델도 여전히 낮은 점수에 머문다.**
- 이것은 benchmark가 잘 만들어졌다는 뜻이기도 하고, 동시에 score를 해석할 때 주의가 필요하다는 뜻이기도 하다. HLE는 collection pipeline 자체가 기존 강한 모델을 떨어뜨리도록 설계되어 있기 때문이다.
- Nature 공개본에서 일부 대표 결과를 추리면 아래와 같다.

| Model | Accuracy (%) | Calibration error (%) | Note |
| --- | ---: | ---: | --- |
| GPT-4o | 2.7 +/- 0.6 | 89 | pre-release |
| Claude 3.5 Sonnet | 4.1 +/- 0.8 | 84 | pre-release |
| Gemini 1.5 Pro | 4.6 +/- 0.8 | 88 | pre-release |
| o1 | 8.0 +/- 1.1 | 83 | pre-release |
| DeepSeek-R1* | 8.5 +/- 1.2 | 73 | text-only subset |
| Claude 4 Sonnet | 7.8 +/- 1.1 | 75 | post-release |
| Gemini 2.5 Pro | 21.6 +/- 1.6 | 72 | post-release |
| GPT-5 | 25.3 +/- 1.7 | 50 | post-release |

- 이 표에서 눈에 띄는 건 두 가지다.
  1. **점수가 생각보다 매우 낮다.**
  2. **calibration이 나쁘다.** 즉 틀리는데도 confidence가 높다.
- paper의 메시지는 단순 accuracy gap이 아니다. HLE에서는 많은 모델이 **모를 때도 모른다고 말하지 못한다.**
- 그리고 Nature 버전은 post-release model을 별도 구획으로 두는데, 이건 아주 좋은 판단이다. public benchmark가 된 뒤의 성능은 여전히 의미가 있지만, **순수한 blind test signal과는 다르게 읽어야 한다.**

- 또 한 가지 흥미로운 실험은 **reasoning token budget**이다.
- 논문은 reasoning model의 output token 수를 분석했는데, token budget이 늘수록 정확도가 로그 선형적으로 좋아지다가, 대략 **2^14 수준 이후에는 오히려 꺾이는 패턴**을 보고한다.
- 이건 "더 오래 생각하면 무조건 낫다"는 직관이 frontier benchmark에서는 성립하지 않을 수 있음을 보여준다.

## 5-2. What really matters in the experiments

내가 보기엔 이 논문에서 진짜 중요한 실험은 "누가 1등인가"보다 아래 네 가지다.

### 1) HLE의 낮은 점수는 benchmark failure가 아니라 construction result다

- HLE는 원래 strong model이 틀리는 문제를 남기도록 설계됐다.
- 그래서 여기서 낮은 score는 곧바로 "모델이 아무것도 못한다"는 뜻이 아니라, **benchmark의 entry condition이 이미 frontier model failure를 포함하고 있다**는 뜻이다.
- 논문도 작은 점수 차이를 과하게 해석하지 말라고 경고한다. multiple-choice에는 non-zero floor가 있고, near-zero 구간의 작은 변화는 진짜 진전인지 noise인지 애매할 수 있다.

### 2) HLE는 accuracy보다 calibration 이야기가 더 중요하다

- HLE는 confidence를 함께 수집하기 때문에, 단순히 "틀렸다"가 아니라 **얼마나 자신 있게 틀렸는가**를 본다.
- closed-ended expert question에서 confidence가 높게 틀리는 모델은, 실제 scientific assistance나 decision support 환경에서 특히 위험하다.
- 그래서 HLE는 그냥 hard QA benchmark가 아니라, **hallucination under uncertainty를 드러내는 calibration benchmark**로도 읽을 수 있다.

### 3) public benchmark 이후의 성능은 반드시 분리해서 봐야 한다

- Nature 공개본은 post-release model을 table에서 따로 구분한다.
- 이건 leaderboard 해석에서 아주 중요한 hygiene다.
- 같은 20점대 score라도,
  - benchmark 공개 전 blind evaluation에서 나온 값인지
  - benchmark 공개 후에 나온 값인지
  는 의미가 다르다.
- 내 관점에서는 HLE의 숫자 그 자체보다 **이 구분을 정직하게 유지한 태도**가 더 중요하다.

### 4) HLE는 closed-ended academic frontier의 지표이지, AGI 검증기는 아니다

- 논문도 분명히 말하듯, HLE 고득점은 expert-level closed-ended academic capability를 시사할 수는 있다.
- 하지만 그게 곧 **autonomous research**, **creative discovery**, **agentic tool use**, **open-ended scientific workflow**를 의미하지는 않는다.
- 즉 HLE는 매우 유용하지만, 어디까지나 **정해진 답이 있는 어려운 시험**이다.
- 이것을 open-ended real-world capability 전체의 proxy로 쓰는 순간 과해진다.

# 6. Limitations

1. **math/STEM 편향이 분명하다.**
   - HLE는 broad benchmark이지만 category 비중을 보면 Math 41%, Bio/Med 11%, CS/AI 10%, Physics 9%, Chemistry 7%로 STEM 쪽 비중이 크다.
   - humanities/social science도 포함되지만, 전체 구조는 여전히 **technical academic reasoning 중심**이다.

2. **closed-ended benchmark의 한계가 있다.**
   - HLE는 의도적으로 exact-match와 multiple-choice로 제한된다.
   - 이 덕분에 자동 채점이 가능하지만, 동시에 **open-ended research judgement, multi-step tool use, exploratory thinking** 같은 능력은 직접 측정하지 못한다.

3. **질문 검증 자체가 여전히 어렵다.**
   - Nature 공개본도 post-release refinement를 별도로 다루고, public set의 estimated expert disagreement rate를 15.4%로 제시한다.
   - biology/chemistry/health subset에서는 targeted peer review 기준 약 18% 수준의 disagreement도 보고된다.
   - 즉 HLE의 어려움은 모델만의 어려움이 아니라, **문항 검증의 어려움**이기도 하다.

4. **후속 검증 논문이 보여주듯 annotation noise는 실제 score를 흔들 수 있다.**
   - 2026년 follow-up인 *HLE-Verified*는 original HLE에 non-trivial noisy item이 있고, 검증/수정된 버전에서 평균 정확도가 7~10 point 정도 올라갈 수 있다고 주장한다.
   - 이건 HLE가 쓸모 없다는 뜻이 아니라, frontier benchmark일수록 **benchmark maintenance와 auditing이 model evaluation만큼 중요하다**는 뜻이다.

5. **public benchmark가 된 이후엔 contamination / gaming 해석을 피할 수 없다.**
   - HLE team도 이 문제를 알고 private set, second held-out set, HLE-Rolling을 도입하려 한다.
   - 하지만 일단 유명해진 benchmark는 모델 개발 생태계의 목표가 되기 쉽고, 그 시점부터는 score 자체보다 **evaluation protocol transparency**가 더 중요해진다.

# 7. My Take

## 7-1. Why this matters for my work

- 내 관점에서 HLE의 가장 큰 가치는 **preference leaderboard로는 못 보는 capability ceiling을 보여준다**는 데 있다.
- 요즘 서비스형 모델 평가는 보통 user preference, helpfulness, style, coding convenience 쪽으로 많이 기운다.
- 그런데 research나 enterprise high-stakes setting에서는 여전히 **정답이 있는 어려운 문제를 틀리는가**, **모를 때 과신하는가**가 매우 중요하다.
- HLE는 바로 그 축을 밀어준다.
- 특히 confidence를 같이 받는 설계는 사내 eval에도 바로 참고할 만하다. 정확도만 보면 놓치는 failure mode가 너무 많기 때문이다.

## 7-2. Reuse potential

이 논문에서 실무적으로 바로 훔쳐올 수 있는 건 surprisingly 많다.

### 1) Frontier-model pre-filter

- internal benchmark를 만들 때도, 먼저 현재 strongest model에 돌려보고 너무 쉬운 문제는 빼는 방식이 매우 유효하다.
- benchmark를 static QA set이 아니라 **current frontier-aware canary set**으로 바꾸는 첫걸음이다.

### 2) Public / private split

- 일부는 공개하고, 일부는 private holdout으로 남겨두는 구조는 contamination을 완전히 막진 못해도 매우 실용적이다.
- 특히 internal eval에서는 이 구조가 거의 필수라고 본다.

### 3) Answer + confidence interface

- 모델에게 답만이 아니라 confidence를 내게 하면, 같은 정확도에서도 훨씬 많은 정보가 생긴다.
- 잘 모를 때 낮게 말하는 모델과, 항상 높게 말하는 모델은 실제 배치 환경에서 전혀 다르다.

### 4) Post-release bug bounty and rolling revision

- benchmark를 한 번 만들고 끝내지 않고, community feedback와 audit를 붙이는 방식은 매우 중요하다.
- 특히 도메인 지식이 깊은 question bank일수록 **dataset governance**가 model governance 못지않게 중요해진다.

내가 보기엔 HLE의 핵심은 "마지막 시험"이라는 브랜딩보다, **frontier benchmark도 배포 후 운영해야 한다**는 메시지다.

## 7-3. Follow-up papers

- **HLE-Verified: A Systematic Verification and Structured Revision of Humanity's Last Exam**
  - HLE의 noisy item 문제를 정면으로 다루는 후속 검증 논문. HLE를 계속 볼 생각이면 거의 필수다.
- **LiveBench: A Challenging, Contamination-Limited LLM Benchmark**
  - HLE가 expert-level closed-ended benchmark라면, LiveBench는 dynamic contamination-limited benchmark ops 쪽에 더 가깝다. 둘을 같이 보면 static vs rolling eval 관점이 선명해진다.
- **GPQA: A Graduate-Level Google-Proof Q&A Benchmark**
  - broadness는 HLE보다 좁지만, domain-expert hard question benchmark의 출발점을 이해하기 좋다.

# 8. Summary

- HLE는 단순히 어려운 문제를 모은 게 아니라, **expert writing + frontier-model filtering + multi-stage review + private holdout**을 결합한 benchmark construction pipeline이다.
- 공개 benchmark는 2,500문항, 100개 이상의 subject, 8개 category로 구성되며, 약 14%는 multimodal, 약 24%는 multiple-choice다.
- 정확도만 낮은 것이 아니라 calibration도 나쁘다는 점이 핵심이다. 즉 많은 모델이 **모를 때도 과신한다.**
- 다만 HLE는 closed-ended academic eval이며, open-ended research나 agentic capability 전체를 대신하지는 못한다.
- 결국 이 논문이 남기는 가장 큰 메시지는 **frontier eval도 dataset release가 아니라 운영 문제**라는 점이다.
