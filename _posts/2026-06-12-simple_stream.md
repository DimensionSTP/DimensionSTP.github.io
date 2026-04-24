---
layout: single
title: "A Simple Baseline for Streaming Video Understanding Review"
categories: Study-concept
tag: [Video, VLM, Streaming, VideoUnderstanding, Multimodal]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2604.02317)

A Simple Baseline for Streaming Video Understanding은 "streaming video understanding에는 memory bank나 retrieval이 꼭 필요하다"는 최근 흐름을 정면으로 다시 묻는 논문이다. 이 논문의 진짜 흥미로운 지점은 새 memory architecture를 제안하는 데 있지 않다. 오히려 최근 프레임 몇 장만 보는 sliding-window baseline이 이미 얼마나 강한지, 그리고 우리가 benchmark score를 너무 쉽게 "memory progress"로 읽고 있지는 않은지 를 드러낸다는 데 있다.

특히 최근 streaming VLM 논문들은 external memory, retrieval, compression, latent memory 같은 모듈을 거의 기본값처럼 붙인다. 그런데 이 논문은 반대로 간다. off-the-shelf Qwen2.5-VL과 Qwen3-VL에 visible prefix에서 최근 N개 프레임만 넣고, 나머지는 과감히 버린다. 그 결과가 단순한 sanity check 수준이 아니라, 기존 streaming 계열과 정면 비교해도 꽤 강하다.

> 한 줄 요약: SimpleStream은 off-the-shelf VLM에 최근 N개 프레임만 넣는 training-free sliding-window baseline인데, 복잡한 memory module 없이도 OVO-Bench와 StreamingBench에서 매우 강한 streaming baseline을 만든다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- **memory를 더 붙이면 무조건 좋아진다**는 가정을 가장 단순한 방식으로 다시 검증한다.
- **recent perception과 long-range memory를 분리해서 봐야 한다**는 평가 관점을 꽤 선명하게 제시한다.
- **training-free baseline**인데도 강하다는 점에서, 새 streaming method를 볼 때 먼저 깔아야 할 control baseline을 제공한다.
- **benchmark가 무엇을 보상하는지**까지 같이 분석해서, leaderboard 해석 자체를 다시 생각하게 만든다.

이 논문의 핵심 메시지는 단순하다. 이 논문은 "더 강한 memory"를 만드는 paper라기보다, 현재 streaming benchmark가 실제로 무엇을 측정하고 있는지 드러내는 강한 baseline paper에 가깝다. 그래서 결과 자체보다도, 어떤 실험을 추가로 해야 memory method의 진짜 가치를 말할 수 있는지 를 보여준다는 점이 더 중요하다.

# 1. Problem Setting

## 1-1. Problem definition

- 이 논문이 겨냥하는 문제는 causal streaming setting에서 비디오를 길게 보면서도, 현재 시점의 질문에 정확히 답하는 것이다.
- streaming video understanding은 본질적으로 두 가지를 동시에 요구한다.
  - 지금 막 보이는 장면을 정확히 인식하는 능력
  - 이전에 봤던 사건이나 상태를 필요할 때 끌어오는 능력
- 최근 방법들은 이 두 번째 요구 때문에 memory bank, retrieval, compression, latent memory 같은 설계를 점점 더 많이 붙여 왔다.
- 하지만 이 논문은 더 근본적인 질문을 던진다. 정말로 그런 복잡성이 필요한가. 아니면 이미 강한 VLM backbone 위에서는 최근 프레임 몇 장만 잘 보존해도 충분히 강한 성능이 나오는가.

## 1-2. Why previous approaches are insufficient

- 기존 streaming 방법들은 method마다 backbone, prompt, frame budget, retrieval policy가 제각각이라서, 실제로는 **memory module의 기여**와 **base VLM의 기여**가 섞여 보이기 쉽다.
- offline long-context VLM과 online streaming VLM의 비교도 종종 불공정하다. 같은 visible prefix를 쓰더라도, 어떤 방법은 수십 프레임을 보고 어떤 방법은 1 fps streaming 조건을 따른다.
- 또 aggregate score만 보면 recent perception과 memory recall이 섞여 버린다. 그러면 memory를 늘려 일부 backward task를 올린 방법이, 동시에 current-scene perception을 망가뜨려도 전체 해석이 흐려질 수 있다.
- 결국 기존 흐름의 한계는 memory를 더 잘 쓰는가 이전에, **강한 recent-context baseline을 먼저 놓고 비교했는가**가 불분명했다는 데 있다.

# 2. Core Idea

## 2-1. Main contribution

- 이 논문의 핵심 기여는 의외로 단순하다. streaming video understanding을 위해 복잡한 memory architecture를 새로 만드는 대신, **SimpleStream**이라는 최소 baseline을 정의한다.
- 방법은 다음 한 줄로 요약된다. visible prefix를 1 fps로 샘플링하고, query 시점에서 **최근 N개 프레임만** off-the-shelf VLM에 넣는다.
- 기본 설정에서는 별도 memory bank, retrieval, vision compression, KV compression을 쓰지 않는다.
- 저자들은 이 baseline을 Qwen2.5-VL-7B-Instruct와 Qwen3-VL-8B-Instruct 위에 올려, OVO-Bench와 StreamingBench에서 offline VLM 6개와 streaming VLM 7개를 포함한 총 13개 major baseline과 비교한다.
- 추가로 window size ablation, model scaling, Visual-RAG, latency, peak GPU memory까지 같이 분석해 왜 이 단순 baseline이 강한지 해부한다.

## 2-2. Design intuition

- 이 논문의 설계 직관은 꽤 현실적이다. 현재 강한 VLM backbone은 이미 OCR, object recognition, short-horizon action understanding, query-conditioned reasoning을 잘한다.
- 그렇다면 성능의 병목이 "기억이 부족해서"가 아니라, 최근 장면을 얼마나 선명하게 보존하느냐 일 수도 있다.
- memory, retrieval, compression은 과거 정보를 더 많이 넣어주지만, 동시에 noise, redundancy, attention dilution을 같이 넣을 수 있다.
- 그래서 이 논문은 "history를 더 많이 넣자"보다 "recent signal을 망치지 말자" 쪽에 가깝다.
- 이 설계가 중요한 이유는, streaming VLM에서 memory가 공짜가 아니라는 사실을 baseline 차원에서 보여주기 때문이다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | causal streaming setting에서 최소 상태만으로 강한 video understanding baseline 만들기 |
| Backbone | Qwen2.5-VL-7B-Instruct, Qwen3-VL-8B-Instruct |
| Input policy | visible prefix를 1 fps로 샘플링하고 최근 N개 프레임만 사용 |
| Extra module | 기본 설정에서는 없음 |
| Training | 없음. off-the-shelf VLM을 그대로 사용 |
| Controlled probes | recent-window ablation, model scaling, Visual-RAG, latency, peak GPU memory |
| Main claim | stronger memory가 아니라 stronger recency baseline을 먼저 넘어서야 진전이라고 말할 수 있음 |

## 3-2. Module breakdown

### 1) Recent-window baseline

- query가 들어오면 해당 시점까지의 visible prefix만 사용할 수 있다.
- SimpleStream은 이 prefix를 1 fps로 샘플링하고, 거기서 **최근 N개 프레임만** 남긴다.
- 즉 method의 상태는 fixed-size recent window 하나뿐이다.
- 이 설계는 극단적으로 단순하지만, 그만큼 baseline 역할을 명확하게 한다. 무엇이 성능을 만드는지 해석하기 쉽다.

### 2) Training-free instantiation

- SimpleStream은 backbone을 fine-tune하지 않는다.
- Qwen2.5-VL과 Qwen3-VL 같은 off-the-shelf open-source VLM 위에 그대로 얹는다.
- 이 점이 중요하다. 성능 개선이 추가 데이터나 streaming-specific training에서 온 것이 아니라, **input policy 자체**에서 온다는 뜻이기 때문이다.

### 3) Recent-window size as the main control knob

- 이 논문에서 사실상 유일한 중요한 hyperparameter는 최근 프레임 개수다.
- main setting에서는 2-frame, 4-frame, 8-frame variant를 비교하고, 추가 실험에서는 16-frame까지 확장한다.
- 이 단순한 knob 하나만으로도 "more history = better" 가설을 꽤 날카롭게 테스트할 수 있다.

### 4) Visual-RAG as a counterfactual probe

- 저자들은 "그렇다면 retrieval을 쓰면 memory 쪽이 좋아지지 않나"라는 반론도 직접 확인한다.
- 이를 위해 CLIP-based historical chunk retrieval을 붙인 Visual-RAG variant를 만든다.
- 구체적으로는 top-5 retrieved chunks를 recent window 뒤에 append한다.
- 이 ablation이 중요한 이유는, SimpleStream이 retrieval을 안 써서 이긴 것이 아니라, **retrieval을 붙여도 전체 score가 꼭 좋아지지는 않는다**는 점을 보여주기 때문이다.

### 5) Efficiency as part of the method story

- SimpleStream은 method 자체가 단순해서 peak GPU memory가 거의 flat하게 유지된다.
- state가 누적되지 않기 때문에, stream이 길어져도 memory curve가 크게 자라지 않는다.
- TTFT도 HERMES를 제외하면 대부분의 streaming baseline보다 낮다.
- 즉 이 논문은 accuracy만이 아니라, "단순한 설계가 latency와 memory에도 실제로 유리하다"는 점을 함께 보여준다.

# 4. Training / Data / Recipe

## 4-1. Data

- 이 논문은 새 training dataset을 만드는 논문이 아니다. 핵심은 unified evaluation이다.
- OVO-Bench는 총 1,640개 question과 12개 task로 구성된다.
- 저자들은 이 중 **Real-Time Visual Perception**과 **Backward Tracing** category를 중심으로 본다.
- StreamingBench는 official real-time visual understanding subset을 사용하며, 총 2,500개 question과 10개 task type으로 구성된다.
- 두 benchmark를 함께 보는 이유는 OVO-Bench에서 본 경향이 다른 streaming benchmark에서도 유지되는지 확인하기 위해서다.

## 4-2. Training strategy

- **학습은 없다.**
- backbone은 Qwen2.5-VL-7B-Instruct와 Qwen3-VL-8B-Instruct를 그대로 사용한다.
- visible prefix를 1 fps로 샘플링하고 최근 N개 프레임만 넣는 것이 전부다.
- 비교군은 각 원 논문이나 official implementation에서 보고한 best inference setting을 최대한 따르되, visible prefix 조건은 unified protocol로 맞춘다.
- 이 점 때문에 이 논문은 새 model recipe라기보다, **evaluation control을 잘 설계한 baseline paper**라고 읽는 편이 맞다.

## 4-3. Engineering notes

- 공개 코드 기준으로 OVO-Bench와 StreamingBench 평가 스크립트가 따로 공개되어 있다.
- StreamingBench 스크립트에서는 `--top-k 0`으로 retrieval을 끄고 recent-only baseline을 유지한다.
- efficiency benchmark는 TTFT, throughput, peak GPU memory를 같이 측정하게 되어 있다.
- 실무적으로는 이 부분이 꽤 중요하다. accuracy만 올리고 state가 계속 커지는 streaming method는 실제 배포에서 불리할 수 있는데, SimpleStream은 fixed recent window라서 이 문제가 훨씬 단순하다.
- 이 논문이 method보다도 **deployment-friendly control baseline**으로 유용한 이유가 여기 있다.

# 5. Evaluation

## 5-1. Main results

가장 먼저 볼 표는 main benchmark result다.

| Method | StreamingBench | OVO RT Avg. | OVO Bwd Avg. | OVO Avg. |
| --- | ---: | ---: | ---: | ---: |
| StreamForest-7B | 77.26 | 61.2 | 52.0 | 56.60 |
| HERMES-7B | 79.44 | 69.0 | 49.4 | 59.20 |
| SimpleStream Qwen2.5-VL + 4f | 78.47 | 78.4 | 51.9 | 65.13 |
| SimpleStream Qwen3-VL + 4f | 80.59 | 81.4 | 54.0 | 67.70 |
| SimpleStream Qwen3-VL + 8f | 78.83 | 79.9 | 54.9 | 67.37 |

이 표만 봐도 메시지는 꽤 선명하다.

- Qwen3-VL + 4f는 OVO Avg. 67.70으로 HERMES 59.20을 8.5 point 앞선다.
- Real-Time Visual Perception에서는 81.4로 HERMES 69.0보다 훨씬 높다.
- Backward Tracing은 8f가 54.9로 더 높지만, 전체 평균은 4f가 더 좋다.
- StreamingBench에서도 Qwen3-VL + 4f가 80.59로 HERMES 79.44를 넘는다.

즉 이 논문은 memory-heavy baseline을 완전히 무의미하다고 말하는 것이 아니다. 오히려 memory 쪽 점수는 8f나 일부 streaming baseline이 더 낫거나 비슷한 경우도 있다. 다만 aggregate score와 real-time category를 포함해서 보면, **4 recent frames**라는 단순한 operating point가 이미 매우 강하다.

## 5-2. What really matters in the experiments

### 1) 4프레임이 8프레임보다 더 낫다는 점

이 논문에서 제일 중요한 결과는 단순 leaderboard보다도 recent-window ablation이다.

- 2f에서 4f로 가면 Overall accuracy가 66.4에서 67.7로 오르고, Real-Time accuracy도 79.3에서 81.4로 오른다.
- 하지만 8f에서는 Overall이 67.4, Real-Time이 79.9로 다시 떨어진다.
- 16f에서는 Overall 67.1, Real-Time 77.9로 더 내려간다.

즉 "more history = better"가 아니라, 조금 늘리면 도움되지만 그 이후는 비단조적이라는 것이다. 이 한 줄만으로도 많은 memory-centric 설계를 다시 보게 만든다.

### 2) longer context의 효용은 backbone-dependent하다

저자들은 이 현상이 작은 모델에서만 그런지 보기 위해 Qwen2.5-VL과 Qwen3-VL의 여러 scale을 비교한다.

- Qwen2.5-VL-72B는 16f에서 최고점을 찍는다.
- 그런데 Qwen2.5-VL-32B는 4f가 최고다.
- Qwen3-VL-32B는 8f가 최고인데, Qwen3-VL-30B-A3B는 4f가 최고다.

즉 larger model이 항상 longer window를 더 잘 쓰는 것도 아니다. 이 논문은 이걸 clean scaling law로 해석하지 않고, **backbone family와 benchmark structure가 함께 결정하는 operating point**로 본다. 이 해석이 꽤 타당하다.

### 3) retrieval은 memory를 올리지만 perception을 깎는다

Visual-RAG ablation도 꽤 중요하다.

- EPM은 52.5에서 59.6으로 오른다.
- ASI는 58.8에서 64.9로 오른다.
- 하지만 OJR은 81.5에서 72.3으로 떨어진다.
- OCR은 94.0에서 85.9로 떨어진다.
- ACR은 78.9에서 71.6으로 떨어진다.
- 전체 정확도는 66.0에서 63.7로 내려간다.

즉 retrieval이 "memory를 조금 올리고 perception을 조금 희생"하는 정도가 아니라, memory-side gain과 perception-side loss가 꽤 뚜렷하게 같이 간다는 것이다. 이게 바로 저자들이 말하는 perception-memory trade-off다.

### 4) benchmark interpretation이 이 논문의 진짜 메시지다

이 논문의 가장 큰 공헌은 여기다.

- HLD는 저자들 주장대로 pure memory task라기보다 hallucination robustness에 더 가깝다.
- OVO macro-average는 capability type이 균형 잡혀 있지 않다.
- Real-Time Visual Perception track이 6개이고, Backward Tracing은 3개다.
- 그러면 recent-scene perception이 좋은 방법이 구조적으로 유리해질 수 있다.

이 말은 SimpleStream 결과를 깎아내리자는 뜻이 아니다. 오히려 반대다. **왜 이 baseline이 강한지 제대로 이해하자**는 뜻이다. benchmark leadership과 long-horizon memory 해결은 같은 말이 아니라는 점을 이 논문은 꽤 설득력 있게 보여준다.

### 5) latency와 memory도 같이 봐야 한다

- TTFT 비교에서 SimpleStream-4f는 16, 64, 256 observed-frame setting 모두에서 HERMES 다음으로 빠르다.
- peak GPU memory도 가장 낮고 가장 flat한 curve를 보인다.
- 이건 "정확도는 좋지만 streaming system으로 쓰기엔 무거운 방법"과 구분되는 포인트다.

실제로 streaming system에서는 state가 늘어날수록 운영 비용이 커진다. 그래서 이 논문은 accuracy만 좋다고 끝나는 게 아니라, **정확도와 운영 단순성의 Pareto point**도 꽤 좋다는 점이 중요하다.

# 6. Limitations

1. 이 결론은 강한 backbone family에 묶여 있다.

- 저자들도 명시하듯, 실험은 Qwen2.5-VL과 Qwen3-VL 위에서 수행된다.
- 따라서 다른 visual encoder, 다른 pretraining mixture, 다른 temporal reasoning 특성을 가진 backbone에서도 같은 정도로 강할지는 추가 검증이 필요하다.

2. 이 논문은 deliberately strong baseline paper다.

- 저자들 스스로도 새 memory-centric architecture를 제안하는 paper가 아니라고 분명히 말한다.
- 즉 이 논문을 "streaming video understanding을 해결했다"로 읽으면 과장이다.

3. benchmark가 recent perception에 유리한 구조를 갖고 있다.

- OVO의 macro-average가 capability-balanced metric이 아니라는 지적은 꽤 중요하다.
- 그래서 SimpleStream의 강점은 method 자체의 강함과 benchmark 구조의 보상이 함께 만든 결과로 읽어야 한다.

4. memory recall과 hallucination robustness가 아직 섞여 있다.

- HLD를 backward tracing에 넣는 현재 구성은 long-range memory를 깨끗하게 재는 데 적합하지 않다.
- future benchmark에서는 perception, memory recall, hallucination robustness를 분리해 볼 필요가 있다.

5. 실무적으로는 "recent-first"만으로는 부족할 수 있다.

- 이 논문은 최근 프레임 보존이 강한 baseline임을 보여주지만, 실제 product에서는 특정 use case가 정말 long-range recall을 많이 요구할 수 있다.
- 따라서 이 baseline은 memory를 포기하자는 뜻이 아니라, **memory를 on-demand로 붙여야 한다**는 방향에 더 가깝다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문은 streaming VLM 자체보다도, multimodal system 설계를 볼 때 중요한 기준을 준다.

- 첫째, 복잡한 memory module을 붙이기 전에 **강한 recency baseline**을 먼저 깔아야 한다.
- 둘째, aggregate score만 보지 말고 **current-scene perception과 memory recall을 분리**해서 봐야 한다.
- 셋째, retrieval이나 memory injection이 일부 backward task를 올렸다면, 그 대가로 OCR, object recognition, action recognition이 얼마나 깎였는지도 같이 봐야 한다.

이 관점은 video understanding뿐 아니라 document AI나 multimodal agent에도 그대로 이어진다. 예를 들어 문서 스트림이나 UI stream을 보는 agent에서도, 무턱대고 history를 길게 넣기보다 최근 증거를 얼마나 또렷하게 보존할지 가 먼저 중요할 수 있다.

## 7-2. Reuse potential

재사용하고 싶은 포인트는 아래 4가지다.

1. **recent-first, history-on-demand** 설계 원칙

- 기본은 최근 컨텍스트만 쓰고, 과거는 정말 필요할 때만 retrieval하는 방향이 더 실용적이다.

2. 강한 baseline부터 깔기

- 새 memory module을 붙였으면, 같은 backbone 위에서 recent-window baseline과 먼저 비교해야 한다.

3. metric 분해

- perception, episodic recall, hallucination robustness를 한 숫자로 합치지 말고 따로 본다.

4. 운영 비용까지 같이 보기

- TTFT와 peak GPU memory를 accuracy 옆에 같이 놓아야 실제 streaming system trade-off가 보인다.

## 7-3. Follow-up papers

- HERMES
- StreamForest
- OVO-Bench
- StreamingBench
- Qwen3-VL technical report

# 8. Summary

- SimpleStream은 최근 N개 프레임만 보는 training-free sliding-window baseline인데도 streaming VLM benchmark에서 매우 강하다.
- 핵심 메시지는 "more history = better"가 아니라, recent perception을 망치지 않는 작은 window가 종종 더 낫다는 것이다.
- longer context의 효용은 model scale에 따라 달라지고, clean scaling law처럼 단조롭게 늘지 않는다.
- retrieval은 EPM, ASI 같은 memory-oriented track을 올릴 수 있지만, OCR, OJR, ACR 같은 current-scene perception을 자주 깎는다.
- 이 논문은 새로운 memory architecture보다, memory method를 어떻게 평가해야 하는지를 다시 정리한 strong baseline paper로 읽는 편이 맞다.
