---
layout: single
title: "LiveBench: A Challenging, Contamination-Limited LLM Benchmark Review"
categories: Study-concept
tag: [LiveBench, LLM, Evaluation]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2406.19314)

요즘 LLM 평가에서 점점 더 불편해지는 지점이 있다. **정적인 벤치마크는 너무 빨리 오염되고**, 반대로 **LLM-as-a-judge나 human preference 기반 평가는 스타일 편향**을 피하기 어렵다. 그래서 이제 중요한 질문은 "새 벤치마크가 하나 더 나왔는가"가 아니라, **벤치마크를 어떤 운영 원리로 설계해야 앞으로도 의미가 남는가**에 더 가깝다.

LiveBench는 바로 그 질문에 답하려는 논문이다. 이 논문의 진짜 가치는 "상위권 모델 순위를 새로 세웠다"는 데 있지 않다. 오히려 **recent source 기반 문제 수집, judge-free objective scoring, 월 단위 갱신, private slice 유지, scoring function 유지보수**를 하나의 시스템으로 묶어, 평가를 고정 데이터셋이 아니라 **지속적으로 운영되는 파이프라인**으로 본다는 데 있다.

> 한 줄 요약: LiveBench는 최근 정보원에서 가져온 문제를 월 단위로 갱신하고, 객관적 정답 기반 자동 채점을 사용하며, 수학, 코딩, 추론, 언어, 지시이행, 데이터분석을 함께 다루는 **동적 contamination-limited LLM benchmark 운영 프레임워크**에 가깝다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- 정적인 leaderboard를 보는 것만으로는 더 이상 모델 능력을 깔끔하게 비교하기 어렵고, **평가 설계 자체가 결과를 크게 바꾸는 시기**이기 때문이다.
- LiveBench는 단순 benchmark release가 아니라, **objective eval을 실제로 굴리는 운영 디테일**까지 공개한다.
- 실험 결과도 흥미롭다. LiveBench 점수는 ChatBot Arena, Arena-Hard와 대체로 상관되지만, **일부 모델은 스타일 편향 때문에 상대 위치가 꽤 달라진다**는 점을 보여준다.

내가 보기엔 이 논문은 "또 하나의 benchmark paper"가 아니다. **LLM eval을 static dataset에서 dynamic benchmark ops로 바꾸는 문서**에 더 가깝다.

# 1. Problem Setting

## 1-1. Problem definition

- 이 논문이 겨냥하는 핵심 문제는 **현대 LLM을 공정하게 평가하기가 점점 더 어려워지고 있다**는 점이다.
- 첫째, 정적인 벤치마크는 인터넷에 공개되고, 최신 모델들은 대규모 웹 데이터를 학습하기 때문에 **test set contamination** 위험이 빠르게 커진다.
- 둘째, contamination을 피하려고 LLM judge나 human crowd judging으로 가면, 이번에는 **정답성보다 스타일, 길이, 표현 방식이 점수에 개입**하기 쉽다.
- 셋째, 실제 서비스에서 중요한 능력은 수학이나 코딩뿐 아니라 instruction following, structured transformation, practical data analysis까지 넓게 퍼져 있는데, 이런 능력을 **한 번에 objective하게** 보기 어렵다.
- 그래서 이 논문의 문제 설정은 "오염되지 않은 문제 몇 개를 새로 만든다"가 아니라, **계속 갱신되면서도 자동 채점 가능한 broad LLM benchmark를 설계하는 것**에 가깝다.

## 1-2. Why previous approaches are insufficient

- 기존의 정적인 benchmark suite는 유용하지만, 공개된 지 시간이 지날수록 **학습 데이터 유입에 따른 점수 inflation**을 피하기 어렵다.
- LLM-as-a-judge 기반 평가는 빠르고 싸지만, **모델이 자기 답변을 더 선호하거나 장황한 답변을 더 좋게 보는 편향**이 있다는 점이 여러 차례 지적되어 왔다.
- human preference 기반 평가는 실제 사용자 선호를 반영한다는 장점이 있지만, formatting, tone, verbosity 같은 요소가 섞이면서 **정답성 자체를 분리해 보기 어려워진다**.
- contamination을 줄이는 최근 벤치마크들도 존재하지만, 예를 들어 LiveCodeBench처럼 **특정 도메인에 한정된 경우가 많다**.
- 결국 기존 접근의 한계는 단순히 benchmark가 낡았다는 것이 아니라, **freshness / objectivity / breadth**를 동시에 잡는 설계가 드물다는 데 있다.

# 2. Core Idea

## 2-1. Main contribution

- LiveBench의 핵심 기여는 세 가지 desiderata를 동시에 만족하려고 한 점이다.
  1. **최근 정보원 기반의 자주 갱신되는 문제**
  2. **LLM judge 없이 objective ground truth로 자동 채점**
  3. **6개 카테고리로 넓게 퍼진 task coverage**
- 이를 위해 벤치마크를 **18개 task / 6개 category / 총 1000문항 규모**로 구성한다.
- 문제 소스도 두 갈래로 나뉜다.
  - 최근 대회, 기사, 데이터셋, arXiv abstract, 영화 synopsis처럼 **시간 민감한 source 기반 task**
  - 기존 benchmark task를 더 어렵고 더 다양하게 바꾼 **harder / procedurally-generated task**
- 또한 월 단위 업데이트 정책을 두어, 평균적으로 **매 업데이트마다 1/6 문항을 교체**하고, public leaderboard에는 항상 **1/6 private question slice**를 유지한다.

## 2-2. Design intuition

- 이 논문의 설계 직관은 꽤 실용적이다. **오염을 완전히 없애는 것**보다 **오염을 계속 밀어내는 운영 구조를 만드는 것**이 더 현실적이라는 판단이다.
- 동시에 judge-free를 유지하려면, free-form generative task를 그냥 채점할 수는 없기 때문에, 문제 자체를 **"정답이 검증 가능한 형태"로 다시 설계**해야 한다.
- 그래서 LiveBench는 단순히 질문을 모으는 데서 끝나지 않고, proof task를 masked equation ordering으로 바꾸고, instruction following을 verifiable constraint task로 바꾸고, plot unscrambling을 edit-distance 기반 ordering 문제로 바꾸는 식으로 **open-ended 능력을 objective surrogate task로 변환**한다.
- 내 해석으로는, LiveBench의 진짜 기여는 dataset보다도 **evaluation interface design**이다. "모델이 무엇을 할 수 있나?" 못지않게 "그 능력을 어떤 형식으로 물어야 자동 채점이 가능해지나?"를 같이 푼다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | contamination risk를 줄이면서도 broad capability를 objective하게 비교할 수 있는 benchmark를 만드는 것 |
| Key design axes | frequent updates + objective scoring + six-category coverage |
| Freshness mechanism | 최근 source 기반 문제 + 월 단위 갱신 + private question slice |
| Scoring principle | question -> task -> category -> overall score로 집계되는 judge-free 자동 채점 |
| Difference from prior work | 고정 데이터셋이 아니라 **계속 운영되는 benchmark pipeline**으로 설계됨 |

## 3-2. Module breakdown

### 1) Dynamic task sourcing

- LiveBench는 문제를 한 번에 고정하지 않는다.
- 각 task는 크게 두 부류 중 하나다.
  - **recent information source 기반 task**: 최근 math competition, 최근 LeetCode/AtCoder 문제, The Guardian 기사, 최근 Kaggle/Socrata 데이터셋, 최근 arXiv abstract, 2024년 이후 영화 synopsis 등
  - **harder / synthetic task**: AMPS_Hard, Web of Lies v2, Zebra Puzzle, 일부 instruction-following reformulation 등
- 중요한 건 "최신 뉴스 몇 개 넣었다"가 아니다. **fresh source를 정기적으로 교체할 수 있는 구조**를 만들었다는 점이 핵심이다.

아래 표처럼 카테고리 구성이 꽤 명확하다.

| Category | Representative tasks | Source / construction | 왜 의미 있나 |
| --- | --- | --- | --- |
| Math | math_comp, olympiad, AMPS_Hard | 최근 수학경시 + synthetic hard math | symbolic reasoning을 contamination-limited하게 측정 |
| Coding | LCB_generation, coding_completion | LiveCodeBench + 부분 솔루션 completion | 코드 생성과 보완 능력을 분리해서 봄 |
| Reasoning | web_of_lies_v2, zebra_puzzle, spatial | synthetic / manually created | judge-free 논리 추론 평가에 적합 |
| Language | connections, typos, plot_unscrambling | NYT puzzle, recent arXiv abstracts, IMDb/Wikipedia | 단순 QA보다 언어 조작 능력을 더 잘 봄 |
| Instruction Following | summarize, paraphrase, simplify, story_generation | Guardian 기사 + verifiable instruction set | 객관적 채점이 가능한 instruction following |
| Data Analysis | cta, tablejoin, tablereformat | recent Kaggle/Socrata datasets | 실용적 structured transformation 능력 측정 |

### 2) Objective reformulation of hard tasks

- LiveBench에서 특히 인상적인 부분은 **채점이 어려운 능력을 어떻게 채점 가능한 task로 바꿀 것인가**다.
- 예를 들어 proof-based olympiad 문제는 자연어 proof를 judge하지 않는다. 대신 proof의 일부 수식을 가린 뒤, 그 수식을 **올바른 순서로 다시 끼워 넣는 문제**로 바꾼다.
- instruction following도 자유로운 creative writing 전체를 평가하지 않고, **검증 가능한 constraint**를 여러 개 붙여 prompt-level / instruction-level로 점수화한다.
- plot unscrambling은 문장 재배열 문제로 바꾸고, 단순 exact match 대신 fuzzy transcription matching과 normalized edit-distance를 사용한다.
- typos task도 마찬가지다. "글을 더 좋게 써라"가 아니라 **오탈자만 고치고 나머지 스타일은 유지하라**는 문제로 바꾼다.
- 즉, LiveBench는 정답이 있는 문제만 모은 게 아니라, **정답이 있게 보이도록 문제를 다시 표현한 benchmark**다.

### 3) Prompt and answer interface design

- 각 task prompt는 보통 다음 요소를 갖는다.
  - zero-shot chain-of-thought 유도
  - 모르면 **best guess**를 하라는 지시
  - 정답을 XML tag나 **double asterisks**처럼 **파싱하기 쉬운 형식**으로 내라는 지시
- 이 설계는 실용적이지만 trade-off도 있다.
- 저자들도 인정하듯, 이렇게 structured answer format을 요구하면 일부 task는 **원래 능력 + instruction following**을 함께 측정하게 된다.
- 다만 그 대신 scoring은 훨씬 자동화 가능해지고, 다양한 모델을 안정적으로 비교하기 쉬워진다.

### 4) Scoring and aggregation pipeline

- LiveBench는 각 질문에 대해 **0에서 1 사이 점수**를 부여하고,
- task score는 문항 평균,
- category score는 task 평균,
- 최종 LiveBench score는 6개 category 평균으로 계산한다.
- 중요한 건 자동 채점이 단순 exact match만은 아니라는 점이다.
- 여러 task에서 flexible regex, fuzzy match, edit-distance 등을 써서 **형식 차이 때문에 정답이 오답 처리되는 false negative**를 줄이려 한다.
- 저자들이 제시한 원칙도 좋다. "사람이 읽어 정답임을 이해할 수 있다면, 가능한 한 맞게 처리해야 한다"는 쪽이다.

### 5) Maintenance and update policy

- LiveBench는 benchmark lifecycle 자체를 method로 포함한다.
- 평균적으로 **매월 1/6 문항을 교체**해서, 전체 benchmark가 약 6개월 주기로 한 번씩 refresh되도록 설계했다.
- 새 문항은 공개 leaderboard에 즉시 공개하지 않고 **한 달 뒤 공개**해서, 항상 private slice가 남게 한다.
- 어떤 task를 먼저 업데이트할지는 주로 두 기준으로 고른다.
  1. **가장 오래된 task**
  2. **가장 쉬워진 task**
- 거의 모든 task는 script 기반으로 갱신 가능하지만, spatial reasoning처럼 손으로 만드는 task도 있고, olympiad / math_comp처럼 source 자체가 자주 바뀌지 않는 task도 있다.

# 4. Training / Data / Recipe

## 4-1. Data

- LiveBench의 current problem set은 **총 1000문항**으로 구성된다.
- 각 task는 보통 **40~100문항** 규모이며, top model 기준 성공률이 대체로 **30~70% 범위**에 오도록 난도를 맞추려 한다.
- task별 문항 수는 완전히 균일하지 않다. 예를 들면,
  - plot_unscrambling은 40문항,
  - olympiad는 36문항,
  - AMPS_Hard는 100문항,
  - math_comp는 96문항,
  - LCB_generation은 78문항이다.
- 이 불균일성 자체가 큰 문제는 아니다. 최종 점수를 task 평균 -> category 평균 -> 전체 평균으로 집계하기 때문에, 단순 문항 수가 전체 점수를 직접 지배하지는 않는다.

- source 설계도 꽤 세심하다.
  - instruction following은 **The Guardian 기사 200개**를 기반으로 만들고,
  - 16개의 verifiable instruction 중 일부를 **2~5개 샘플링**한 뒤 충돌을 제거해 prompt를 구성한다.
  - typos는 최근 arXiv abstract를 사람이 먼저 정리한 뒤 synthetic misspelling을 주입한다.
  - plot unscrambling은 **2024년 1월 1일 이후 개봉 영화**의 synopsis를 사용한다.
  - data analysis는 Kaggle / Socrata의 최근 dataset을 사용한다.
- 즉, LiveBench의 데이터 설계는 "최신 자료를 쓰자"가 아니라, **지속적으로 refresh 가능한 public source pool을 찾자**는 쪽이다.

## 4-2. Training strategy

이 논문은 모델 학습 논문이 아니라 benchmark 논문이므로, 여기서는 **benchmark 운영 전략**을 보는 편이 맞다.

- 실험에서는 총 **40개 모델**을 평가한다.
- paper v2 기준으로 모델 범위는 **0.5B에서 405B급**까지 걸쳐 있다.
- 실행 프로토콜은 꽤 엄격하다.
  - 기본적으로 **single-turn evaluation**
  - temperature 0
  - 각 모델 family의 example code에 맞춘 chat template / system message / inference hyperparameter 사용
  - open-source 모델은 bfloat16 실행
- 새로운 모델을 붙일 때도 그냥 한 번 돌려보고 끝내지 않는다.
- 먼저 example code와 템플릿을 맞추고, 비슷한 성능대 모델과 비교해 **이상하게 낮은 task가 없는지** 본 다음, 잘못된 parsing이나 scoring 문제를 점검한다.
- 평가를 "한 번의 batch run"이 아니라 **계속 보정되는 execution pipeline**으로 다루는 태도가 좋다.

## 4-3. Engineering notes

- LiveBench의 engineering에서 내가 가장 높게 보는 부분은 **scoring fairness**다.
- 자동 채점을 쓰면 두 가지 위험이 생긴다.
  1. 문제 자체가 instruction-following test가 되어버리는 것
  2. 특정 출력 형식에 맞춘 모델만 유리해지는 것
- 저자들은 이를 줄이기 위해 flexible regex와 permissive scoring을 쓴다. 예를 들어 수학 문제는 boxed/fbox 같은 표기 차이를 허용하고, 객관식 문제는 letter answer뿐 아니라 raw answer도 허용한다.

- 또 scoring function을 고정 불변으로 두지 않는다.
- 새 모델이 들어오면
  - 비슷한 모델 대비 이상한 task를 찾고,
  - incorrect sample을 수동 검토해 false negative를 확인하고,
  - 필요하면 scoring function을 수정한 뒤,
  - **기존 모델 전체를 다시 채점**한다.
- benchmark engineering에서는 이게 매우 중요하다. 모델이 바뀌면 parser edge case도 바뀌기 때문이다.

- 유지보수 비용도 paper에 꽤 솔직하게 적혀 있다.
- 저자들은 leaderboard에 **약 40~50개 모델**을 유지하고, 매달 **200개 새 문항 정도**를 평가하는 정도는 감당 가능한 수준이라고 본다.
- 또 1000문항 기준 평균 입력 토큰은 약 **1612**, 평균 출력 토큰은 약 **395**다.
- 2024-10-01 기준 API 비용 예시도 제시하는데, 가장 싼 쪽은 Claude 3 Haiku 약 **$0.90**, 비싼 쪽은 Claude 3 Opus 약 **$53.80**, o1-preview는 약 **$47.87** 수준이다.
- 이 숫자들이 중요한 이유는 단순 가격 정보가 아니라, **dynamic benchmark를 운영하려면 평가비도 design variable**라는 걸 보여주기 때문이다.

# 5. Evaluation

## 5-1. Main results

- 가장 먼저 눈에 띄는 건 LiveBench가 실제로 꽤 어렵다는 점이다.
- paper v2의 Table 2 기준으로 **어떤 모델도 70점을 넘지 못한다**.
- 상위권은 다음과 같다.

| Model | LiveBench score | 메모 |
| --- | --- | --- |
| o1-preview-2024-09-12 | 64.7 | 데이터분석, 언어, 수학에서 강세 |
| claude-3.5-sonnet-20241022 | 58.5 | 코딩에서 매우 강함 |
| claude-3.5-sonnet-20240620 | 58.2 | 여전히 상위권 |
| o1-mini-2024-09-12 | 56.7 | 추론 카테고리에서 강세 |

- category specialization도 분명하다.
  - **o1-preview**는 data analysis / language / math에서 강하고,
  - **Claude 3.5 Sonnet 계열**은 coding에서 강하며,
  - **o1-mini**는 reasoning에서 가장 두드러진다.
- 즉 LiveBench는 "overall one-number ranking"을 주지만, 실제로는 **모델이 어디서 강한지 꽤 다르게 드러나는 benchmark**다.

- 흥미로운 건 open 계열의 위치다.
- Table 2 기준으로 meta-llama-3.1-405b-instruct-turbo가 **51.1**로 상위 open 계열에 위치하고, 일부 강한 open 계열 모델도 50점 전후까지 올라온다.
- 다만 proprietary 상위권과는 여전히 차이가 있다. LiveBench는 이 격차를 단순 chat preference가 아니라 **ground-truth task**에서 드러낸다는 점에서 의미가 있다.

- benchmark 비교 실험도 중요하다.
- LiveBench 점수는 **ChatBot Arena와 0.91**, **Arena-Hard와 0.88**의 상관을 보인다.
- 하지만 완전히 같은 순서를 주지는 않는다.
- 논문은 GPT-4 계열이 Arena-Hard에서 상대적으로 더 강하게 보일 수 있고, Gemini 1.5 계열은 ChatBot Arena에서 상대적으로 더 좋아 보일 수 있다고 해석한다. 전자는 **judge bias**, 후자는 **human style preference**의 가능성을 시사한다.
- 이건 LiveBench의 가장 강한 메시지 중 하나다. **비슷해 보이는 leaderboard도 무엇을 채점하느냐에 따라 다른 모델을 띄운다.**

- monthly update 분석도 좋다.
- 두 번의 월간 업데이트 사이에서 모델 rank correlation은 모두 **0.997 초과**였고, 가장 최근 question set으로 갈수록 평균/중앙 score가 약 **1.2% 하락**했다.
- 즉, benchmark는 **점점 더 어려워지지만 순위 자체는 함부로 흔들리지 않는다**는 뜻이다.

## 5-2. What really matters in the experiments

- 내가 보기엔 이 논문에서 진짜 중요한 실험은 "누가 1등인가"가 아니다.

### 1) LiveBench는 style leaderboard와 겹치지만 같지 않다

- Arena류 benchmark와 높은 상관을 보이면서도, 특정 모델은 꽤 다른 상대 위치를 가진다.
- 즉 LiveBench는 완전히 엉뚱한 benchmark가 아니라, **기존 leaderboard와 공통된 signal은 잡되 style bias를 덜 먹는 방향**에 있다.

### 2) instruction following은 다른 능력과 꽤 분리되어 있다

- category correlation table을 보면 instruction_following은 전체 LiveBench score와의 상관이 약 **0.817**로, 다른 카테고리보다 가장 낮다.
- 이건 꽤 중요한 메시지다. **"말 잘 듣는 모델"과 "정답을 잘 맞히는 모델"은 꽤 다른 축일 수 있다.**
- 서비스 모델을 평가할 때 one-number score만 보지 말아야 하는 이유가 여기 있다.

### 3) 일부 task는 overall ability proxy로 더 좋다

- task correlation 기준으로는 **math_comp**가 전체 평균 점수와 가장 높은 상관(약 **0.904**)을 보인다.
- 빠른 internal eval을 설계할 때 이런 task-level proxy는 유용할 수 있다.
- 물론 이것이 곧 "math_comp만 보면 된다"는 뜻은 아니고, **짧은 smoke test용 대표 task 후보**로 읽는 편이 좋다.

### 4) hard math / logic에서는 LLM judge가 실제로 꽤 불안정하다

- Appendix의 예비 실험은 재미있다.
- GPT-4-Turbo judge를 사용했을 때 AMC12, AIME, SMC, Zebra Puzzles에서 **오류율이 생각보다 높게** 나온다.
- 예를 들어 GPT-4-Turbo judge의 오류율은 AMC12에서 **0.380**, Zebra Puzzles에서 **0.420**까지 올라간다.
- 저자들도 이 실험을 definitive하다고 말하진 않지만, 적어도 이 결과만 보면 **어려운 추론 문제는 judge model도 못 푼다**는 직관이 꽤 설득력 있게 드러난다.

# 6. Limitations

1. **완전한 contamination-free benchmark는 아니다.**
   - 저자들도 appendix에서 인정하듯, 일부 coding question은 2023년 11월 source를 포함하고, 일부 math competition 문제는 비교적 낮은 수준의 변형만 거쳤다.
   - 그래서 LiveBench는 title 그대로 **contamination-limited**로 읽는 편이 맞다.

2. **objective scoring이 가능한 task만 다룰 수 있다.**
   - 여행 가이드를 써 보라거나, 긴 open-ended essay quality를 보라거나, 복합적 agent loop를 보라거나 하는 문제는 ground truth 정의 자체가 어렵다.
   - 즉 LiveBench는 broad하지만, 여전히 **judge-free evaluation이 가능한 영역으로 문제를 투영한 benchmark**다.

3. **prompt / format bias를 완전히 제거하지는 못한다.**
   - structured answer format은 자동 채점을 가능하게 하지만, 동시에 일부 task를 instruction-following과 섞이게 한다.
   - 저자들도 특정 LLM family가 특정 prompt type을 선호하는 편향 가능성을 인정한다.

4. **운영형 benchmark라서 유지보수 자체가 연구다.**
   - scoring function 유지, 새로운 모델 온보딩, task refresh, private slice 운영까지 모두 필요하다.
   - 논문은 이를 장점으로 포장할 수 있지만, 실제로는 **benchmark를 계속 돌릴 팀의 운영 역량**이 있어야만 유지된다.

# 7. My Take

## 7-1. Why this matters for my work

- 내 관점에서 LiveBench의 가장 큰 가치는 **평가도 제품처럼 운영해야 한다**는 메시지다.
- 실제 서비스에서는 static test set 하나로 모델을 오래 비교할 수 없다. 특히 post-training이 빠르게 반복되고, 모델들이 benchmark-friendly behavior를 익혀가는 시기엔 더 그렇다.
- LiveBench는 이 문제를 잘 짚는다. 평가를 한 번의 dataset release가 아니라,
  - source refresh,
  - private holdout,
  - scoring maintenance,
  - model onboarding QC,
  - versioned leaderboard
  의 묶음으로 본다.
- 이 시각은 연구에도 좋지만, **사내 internal eval 설계**에도 바로 쓸 수 있다.

## 7-2. Reuse potential

- 이 논문에서 바로 재사용 가능한 아이디어는 많다.

### 1) Public slice + private slice 구조
- 전체 eval 중 일부는 공개하고, 일부는 private canary set으로 남겨두는 방식은 contamination을 완전히 막지 못해도 꽤 실용적이다.

### 2) Open-ended task의 objective reformulation
- 자유서술을 그대로 judge하지 말고, 가능한 한 **정답 검증 가능한 하위 문제**로 쪼개는 방식은 document AI, data extraction, UI evaluation에도 잘 이식된다.

### 3) Scoring function 유지보수 문화
- parser와 scoring code를 한 번 짜고 끝내지 않고, 새로운 모델 output을 보며 계속 보정하는 습관은 internal benchmark에서도 매우 중요하다.

### 4) Easiest / oldest task 우선 갱신
- 어떤 task를 새로 만들지 막막할 때, "가장 쉬워진 것"과 "가장 오래된 것"을 먼저 바꾸는 정책은 단순하지만 강력하다.

내가 보기엔 LiveBench의 핵심은 수치가 아니라 **eval ops recipe**다.

## 7-3. Follow-up papers

- **LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code**
  - contamination-free / continuously updated benchmark를 코드 도메인에 더 깊게 가져간 사례.
- **MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark**
  - static benchmark를 더 어렵고 더 robust하게 만드는 흐름을 비교해서 보기 좋다.
- **Humanity's Last Exam**
  - objective, contamination, frontier difficulty라는 축이 어디까지 확장될 수 있는지 비교하기 좋다.

# 8. Summary

- LiveBench는 정적인 dataset이 아니라 **월 단위로 갱신되는 benchmark pipeline**으로 읽어야 한다.
- 이 논문의 핵심은 최근 source, objective scoring, six-category coverage를 함께 만족시키려는 설계다.
- 특히 proof, instruction following, plot unscrambling 같은 문제를 **자동 채점 가능한 surrogate task**로 다시 표현한 점이 좋다.
- 실험은 current models가 여전히 70점 아래에 머물고, style-judged benchmark와는 다른 상대 위치를 보일 수 있음을 보여준다.
- 다만 LiveBench는 contamination-free가 아니라 contamination-limited이며, open-ended generation 전체를 대체하는 만능 평가 프레임워크는 아니다.
