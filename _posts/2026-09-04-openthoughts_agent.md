---
layout: single
title: "OpenThoughts-Agent: Data Recipes for Agentic Models Review"
categories: Study-concept
tag: [OpenThoughtsAgent, AgenticModels, DataCuration, SFT, AgentBenchmarks]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2606.24855)

[Project page](https://www.openthoughts.ai/)

OpenThoughts-Agent는 "agentic model을 위한 데이터셋을 하나 공개했다" 정도로 읽으면 핵심을 놓치기 쉽다. 이 논문이 실제로 다루는 문제는 훨씬 구체적이다. **agentic model을 넓은 benchmark suite에서 잘 일반화시키려면 SFT data pipeline의 어느 단계가 가장 중요하고, 어떤 data recipe가 실제로 scale하는가**다.

Open agent 연구에서 code, terminal, web, search task는 많이 늘고 있지만, training data recipe는 여전히 불투명하다. SWE-Smith, SERA, Nemotron-Terminal 같은 공개 노력은 존재하지만, 대개 특정 benchmark나 좁은 agent domain에 집중한다. 반면 frontier agent model은 여러 종류의 computer-use task를 동시에 잘해야 한다. 이 gap 때문에 open community가 agentic post-training을 체계적으로 개선하기 어렵다.

OpenThoughts-Agent는 이를 data curation 문제로 본다. Task source selection, task mixing, task augmentation, task filtering, teacher selection, trajectory filtering, scaling strategy를 stage별로 ablation하고, 각 stage가 downstream agent benchmark 성능에 어떤 영향을 주는지 본다. 논문은 100개 이상의 controlled ablation experiment를 수행하고, 최종적으로 100K SFT example을 구성해 Qwen3-32B를 fine-tune한다.

> 한 줄 요약: OpenThoughts-Agent는 broad agentic SFT data pipeline을 공개적으로 ablation한 논문으로, task source diversity, teacher choice, long trajectory filtering, synthetic task augmentation이 agent benchmark generalization에 얼마나 중요한지 분석하고, 100K example으로 Qwen3-32B 기반 OpenThinkerAgent-32B를 학습해 7개 agentic benchmark 평균 44.8%를 보고한다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Agentic model training에서 model architecture보다 data recipe가 어디서 성능을 좌우하는지 보여준다.
- 95개 task generation strategy, teacher model choice, trace filtering, scaling strategy를 단계별로 ablation한다.
- Benchmark-specific overfitting을 피하려고 core benchmarks와 OOD benchmarks를 분리한다.
- 강한 teacher가 항상 좋은 teacher가 아니라는 실무적으로 중요한 결과를 보여준다.
- Long trajectory filter가 agentic supervision quality를 높일 수 있음을 보고한다.
- Pipeline, training data, experimental data, models를 공개한다고 밝힌 점에서 open agent training 연구의 재현 기반을 만든다.

이 글에서는 OpenThoughts-Agent를 "agent SFT dataset paper"보다, **agentic data curation에서 어떤 choices가 실제 downstream agent capability를 만든다는 것을 실험적으로 분해한 recipe paper**로 읽는다.

# 1. Problem Setting

## 1-1. Problem definition

Agentic model training data는 일반 instruction tuning data와 다르다. 단순히 prompt-response pair가 아니라, environment와 상호작용하며 여러 turn에 걸쳐 행동하고, tool output을 보고, 실패를 수정하는 trajectory가 필요하다.

SFT dataset을 다음처럼 볼 수 있다.

$$
D
=
\{(x_i,\tau_i)\}_{i=1}^{N}
$$

여기서 $x_i$는 task description이고, $\tau_i$는 agent trajectory다. Agentic SFT의 핵심은 좋은 $x_i$와 좋은 $\tau_i$를 동시에 만드는 것이다.

OpenThoughts-Agent의 central question은 다음이다.

> 어떤 task source, teacher, filter, mixture, scaling strategy가 broad agentic benchmark 성능을 가장 안정적으로 높이는가?

이 질문이 어려운 이유는 agentic benchmark가 하나가 아니기 때문이다. SWE-Bench에서 좋은 data가 Terminal-Bench에 항상 좋은 것은 아니고, terminal infrastructure question이 software issue resolution에 항상 좋은 것도 아니다. Data source는 benchmark-specific skill을 강하게 만든다.

## 1-2. Why previous approaches are insufficient

### 1) Single-benchmark data recipes

SWE-Smith, SERA, Nemotron-Terminal 같은 기존 공개 effort는 매우 유용하지만, 특정 benchmark나 task family에 집중하는 경향이 있다. 그러나 실제 배포된 agent는 coding, terminal, function calling, finance, search, web interaction을 함께 처리해야 한다.

### 2) Task source와 teacher choice가 충분히 문서화되지 않는다

많은 model release는 architecture와 training compute는 비교적 자세히 설명하지만, data curation 과정은 덜 투명하다. 하지만 agentic capability는 task source와 trajectory teacher에 크게 의존할 수 있다. Ablation이 없으면 어떤 요소가 실제로 중요한지 알기 어렵다.

### 3) 강한 teacher가 자동으로 좋은 teacher는 아니다

Benchmark score가 높은 teacher라도 student SFT에 덜 유용한 trajectory를 만들 수 있다. 너무 짧게 풀거나, 특정 model에만 맞는 shortcut을 쓰거나, student가 모방하기 어려운 trace를 만들 수 있기 때문이다. OpenThoughts-Agent는 이 지점을 명시적으로 실험한다.

### 4) Dataset을 키우는 것만으로 충분하지 않다

같은 task description에서 rollout 수만 늘리는 scaling은 plateau에 걸릴 수 있다. 병목이 task diversity라면, 같은 task에 대한 trajectory를 더 많이 뽑아도 문제가 해결되지 않는다. 논문의 scaling analysis는 바로 이 문제를 다룬다.

# 2. Core Idea

## 2-1. Main contribution

OpenThoughts-Agent의 기여는 네 가지다.

1. **Open agentic SFT data pipeline**
   - Task-trajectory pair를 만들기 위한 6단계 pipeline을 제시한다.
   - 각 stage를 독립적으로 ablation한다.

2. **Controlled ablation study**
   - 100개 이상의 ablation experiment를 수행한다.
   - Stage별 실험은 10K data scale에서 진행한다.
   - 비용 효율적인 비교를 위해 Qwen3-8B를 fine-tune한다.

3. **OpenThinkerAgent dataset and model**
   - 100K example final dataset을 만든다.
   - Qwen3-32B를 fine-tune한 model을 공개한다.
   - 7개 agentic benchmark 평균 44.8%를 보고한다.

4. **Scaling and RL investigation**
   - SFT data scaling을 분석한다.
   - 새로 만든 RL data와 two-stage 8B result도 논의한다. 다만 세부 내용은 추가 확인이 필요하다.

## 2-2. Design intuition

설계 직관은 agentic training data quality가 여러 단계의 조합으로 결정된다는 것이다. Dataset은 어느 한 stage가 약해도 실패할 수 있다.

| Stage | Failure mode |
| --- | --- |
| Task source | 필요한 skill을 덮지 못하거나 특정 benchmark에만 치우침 |
| Task mixing | 한 domain에 과도하게 특화됨 |
| Task augmentation | Learnability를 높이지 않는 constraint를 추가함 |
| Task filtering | Trivial하거나 noisy한 task를 남김 |
| Teacher model | 품질이 낮거나 student로 transfer되지 않는 trajectory를 생성함 |
| Trajectory filtering | 너무 짧거나 얕거나 timeout/low-value trace를 남김 |
| Scaling | Diversity를 늘리지 않고 같은 task만 반복함 |

Pipeline approach가 유용한 이유는 data를 black box로 보지 않기 때문이다. 각 stage를 따로 평가할 수 있다.

Ranking method는 세 core benchmark의 average z-score를 사용한다. Benchmark마다 raw accuracy range가 다르기 때문에 이 점이 중요하다. 한 benchmark를 크게 올리지만 다른 benchmark를 망치는 strategy가 단순 raw scale 때문에 이기는 것을 막는다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | 넓은 범위의 agentic model을 위한 open SFT/RL data recipe 구축 |
| Main artifact | OpenThoughts-Agent SFT pipeline과 100K dataset |
| Base models | Ablation에는 Qwen3-8B, final SFT model에는 Qwen3-32B 사용 |
| Teacher baseline | terminus-2 harness에서 GLM-4.7-AWQ 사용 |
| Core ablation scale | Strategy마다 10K trajectory |
| Core benchmarks | OpenThoughts-TB-Lite, SWE-Bench Verified-100, Terminal-Bench 2 |
| OOD benchmarks | Aider Polyglot, BFCL-Parity, MedAgentBench, GAIA-127, FinanceAgent-Terminal |
| Final result | 7개 agentic benchmark 평균 44.8% |

## 3-2. Module breakdown

### 1) Task sourcing

논문은 95개 task generation strategy를 테스트한다. Source에는 synthetic issue-resolution task, 사람이 작성한 computer-use question, infrastructure question이 포함된다.

Top-performing strategy는 다음을 포함한다.

- SWE-Smith
- StackExchange SuperUser
- StackExchange Tezos
- IssueTasks

중요한 finding은 task source choice의 영향이 매우 크다는 점이다. 논문은 task generation strategy가 SWE-Bench Verified-100에서 downstream accuracy를 최대 30 percentage point, Terminal-Bench 2.0에서 10 point까지 바꿀 수 있다고 보고한다.

### 2) Task mixing

Top source 하나만 쓰지 않고 top-ranked source를 섞는다. 10K scale에서는 Top-4와 Top-8 mix가 가장 좋다. Single benchmark over-specialization을 피하기 때문이다.

이것은 agent data를 만들 때 중요한 교훈이다.

```text
best single source != best broad agent dataset
```

### 3) Task augmentation

논문은 requirement clarification이나 task description hardening 같은 여러 LLM-driven task augmentation strategy를 테스트한다. 놀랍게도 어떤 augmentation도 original task description 대비 안정적인 개선을 만들지 못한다.

이는 유용한 negative result다. Task text가 더 복잡해진다고 supervision이 자동으로 좋아지지는 않는다.

### 4) Task filtering

LLM-based difficulty signal로 task description을 filtering하면 성능이 개선된다. 특히 GPT-5가 더 긴 response를 생성하는 task를 선택하는 heuristic은 core benchmark 평균을 약 3 percentage point 높인다.

해석은 strong model의 긴 response가 trivial prompt보다 non-trivial multi-step task를 가리킬 수 있다는 것이다.

### 5) Teacher model choice

논문은 trajectory generation을 위한 여러 teacher model을 테스트한다. 놀라운 결과는 benchmark에서 가장 강한 model이 꼭 best teacher는 아니라는 점이다. GPT-5.3-Codex는 benchmark에서는 강하지만 이 setup에서는 teacher로 약하고, Terminal-Bench 2.0에서 GLM-4.7-AWQ보다 약 5 point 낮다고 보고된다.

이는 SFT teacher quality가 teacher solve rate와 같지 않다는 뜻이다.

### 6) Trajectory filtering

5 turn 미만 trace를 제거하는 filter가 trajectory filter 중 가장 좋다. 논문은 appendix에서 token budget을 맞춰 gain이 단순히 token 수가 늘어난 효과가 아님을 확인한다.

긴 trajectory는 tool use, feedback, correction, state update, intermediate observation에서의 recovery 같은 더 풍부한 decision supervision을 담을 가능성이 높다.

### 7) Scaling strategy

같은 task description에 대해 rollout을 더 많이 뽑는 방식의 scaling은 31.6K에서 100K 사이에 plateau된다. 논문은 이후 synthetic task augmentation으로 이 plateau를 넘긴다. 결론은 task description diversity가 병목이 될 수 있다는 것이다.

# 4. Training / Data / Recipe

## 4-1. Data

Final SFT data는 pipeline에서 만든 100K example을 사용한다. 논문은 training set, pipeline, experimental data, model을 공개했다고 보고한다.

Core source mix는 ablation 이후 고른 상위 task source를 포함한다. 정확한 final data composition은 재사용 전 released data card와 appendix에서 확인해야 한다.

## 4-2. Training strategy

Ablation fine-tuning setup은 다음과 같다.

| Item | Value |
| --- | --- |
| Ablation model | Qwen3-8B |
| Data per ablation | 10K trajectories |
| Training | Full-parameter SFT |
| Learning rate | $4e-5$ |
| Schedule | Cosine |
| Global batch size | 96 |
| Epochs | 7 |
| Context length | 32,768 |
| Compute per 10K finetune | 160 GPU-hours on GH200s |

Final model은 다음과 같다.

| Item | Value |
| --- | --- |
| Base model | Qwen3-32B |
| SFT examples | 100K |
| Reported model | OpenThinkerAgent-32B |

## 4-3. Evaluation setup

Core development benchmark는 다음과 같다.

| Benchmark | Size |
| --- | ---: |
| OpenThoughts-TB-Lite | 100 tasks |
| SWE-Bench Verified-100 | 100 tasks |
| Terminal-Bench 2 | 89 tasks |

OOD evaluation benchmark는 다음과 같다.

| Benchmark |
| --- |
| Aider Polyglot |
| BFCL-Parity |
| MedAgentBench |
| GAIA-127 |
| FinanceAgent-Terminal |

Evaluation은 Daytona sandbox와 terminus-2 harness를 사용하며, pipeline experiment에서는 task마다 3번의 stochastic re-run을 사용한다.

## 4-4. Engineering notes

1. **초기에는 data source가 성능을 크게 좌우한다**
   - 모든 synthetic task를 서로 바꿔 써도 된다고 보면 안 된다.

2. **Normalized ranking으로 benchmark를 균형 있게 본다**
   - Raw average는 score range가 넓은 benchmark를 과가중할 수 있다.

3. **더 강한 teacher가 더 좋은 teacher라고 가정하지 않는다**
   - Teacher trajectory는 student가 배울 수 있고 다른 task로 옮겨갈 수 있어야 한다.

4. **의미 있는 multi-turn trace를 우선한다**
   - 짧은 trace는 agentic decision supervision이 부족할 수 있다.

5. **Task diversity가 scaling 병목이 될 수 있다**
   - 같은 task의 rollout만 늘리면 plateau될 수 있다.

6. **OOD benchmark는 recipe search 밖에 둔다**
   - 그렇지 않으면 data recipe가 core development suite에 overfit될 수 있다.

# 5. Evaluation

## 5-1. Main final result

논문은 100K SFT example로 학습한 OpenThinkerAgent-32B가 7개 agentic benchmark 평균 44.8%를 달성하고, Nemotron-Terminal-32B의 40.9%를 넘는다고 보고한다.

Introduction에 보고된 주요 table value는 다음과 같다.

| Model | Average | SWE-Bench Verified | Terminal-Bench 2.0 |
| --- | ---: | ---: | ---: |
| OpenThinkerAgent-32B | 44.8 | 54.0 | 26.2 |
| Nemotron-Terminal-32B | 40.9 | 41.9 | 25.1 |

논문은 Aider Polyglot, BFCL-Parity, MedAgentBench, GAIA-127, FinanceAgent-Terminal에서도 좋은 generalization을 보고한다.

이 결과는 논문이 비교한 setting 안에서, open-data 기반의 Qwen3-or-earlier <=32B agentic model 중 강하다는 의미로 읽어야 한다. Universal agent SOTA claim으로 읽으면 안 된다.

## 5-2. Ablation findings

중요한 ablation takeaway는 다음과 같다.

| Stage | Finding |
| --- | --- |
| Task source | 가장 큰 성능 차이를 만들며, source choice가 score를 크게 바꿀 수 있음 |
| Task mixing | Top-4에서 Top-8까지의 mix가 single-source over-specialization보다 나음 |
| Augmentation | LLM-driven task description augmentation은 안정적으로 도움이 되지 않음 |
| Filtering | GPT-5 longer-response filter가 평균 약 3 percentage point를 개선 |
| Teacher | 강한 benchmark model이 best SFT teacher는 아닐 수 있음 |
| Trajectory filter | 5 turn 미만 trace를 제거하는 것이 도움됨 |
| Scaling | 같은 task의 rollout 증가는 plateau되며, synthetic task augmentation이 추가 도움을 줌 |

## 5-3. Scaling

논문은 같은 task description당 rollout을 추가로 upsample하는 방식이 31.6K에서 100K 사이에 plateau된다고 보고한다. 이후 synthetic task augmentation이 세 core benchmark 모두에서 추가 개선을 만든다.

이는 agentic SFT data scaling이 단순 trajectory count가 아니라 task diversity에 의해 제한될 수 있음을 시사한다.

## 5-4. What really matters in the experiments

### 1) Data recipe는 측정 가능한 대상이다

OpenThoughts-Agent는 SFT data pipeline choice도 model architecture처럼 ablation할 수 있음을 구체적으로 보여준다.

### 2) Broad agentic capability에는 source mixture가 필요하다

Single benchmark data는 한 benchmark를 개선하면서 generalization에는 실패할 수 있다. 그래서 상위 source를 섞는 과정이 필요하다.

### 3) Trajectory shape이 중요하다

Agentic data는 final answer만이 아니다. Path length, observation, tool call, feedback이 learnability에 영향을 준다.

### 4) Teacher selection은 별도의 optimization 문제다

Best solver가 best teacher는 아니다. Distillation quality는 outcome뿐 아니라 trajectory style에 의존한다.

### 5) Open artifact가 중요하다

논문의 가장 중요한 생태계 기여는 final model만이 아니라 pipeline과 experimental data release일 수 있다.

# 6. Limitations

1. **Evaluation은 여전히 benchmark-dependent이다**
   - 7개 benchmark는 넓지만 모든 경우를 포괄하지는 않는다.

2. **Harness effect가 있다**
   - Score는 terminus-2와 original harness choice에 의존한다.
   - Agent harness가 측정된 capability를 바꿀 수 있다.

3. **Teacher와 model version은 바뀔 수 있다**
   - GPT-5.3-Codex, GLM, Kimi, Qwen version은 최종 확인이 필요하다.

4. **Data contamination을 확인해야 한다**
   - Public source의 agent task는 benchmark-like task와 overlap될 수 있다.

5. **SFT 중심 논문이다**
   - 논문은 RL data도 다루지만 headline pipeline은 SFT 중심이다.

6. **Open data quality는 균일하지 않다**
   - Pipeline이 공개되어 있더라도 task legality, licensing, benchmark leakage를 audit해야 한다.

7. **Ablation scale과 final scale이 다르다**
   - Stage ablation은 대부분 10K/Qwen3-8B이고, final result는 100K/Qwen3-32B다.

8. **LLM teacher trace는 shortcut behavior를 담을 수 있다**
   - Trajectory quality는 length와 success만으로 결정되지 않는다.

9. **Average score는 benchmark trade-off를 숨길 수 있다**
   - Data recipe가 SWE를 올리면서 terminal을 떨어뜨리거나 그 반대일 수 있다.

10. **Compute cost가 작지 않다**
   - 10K ablation fine-tuning당 160 GPU-hours는 broad recipe search가 비싸다는 뜻이다.

# 7. My Take

## 7-1. Why this matters for my work

OpenThoughts-Agent의 가장 중요한 포인트는 "100K SFT data로 좋은 agent를 만들었다"보다, **agentic training data를 단계별 engineering object로 다뤘다는 점**이다.

Agent capability는 model size만이 아니다. 다음이 모두 중요하다.

- 어떤 task를 생성하는가
- task source를 어떻게 섞는가
- 어떤 teacher가 trajectory를 만드는가
- 어떤 trajectory를 남기는가
- data scaling이 diversity를 늘리는가, 반복만 늘리는가
- evaluation이 한 benchmark overfitting을 어떻게 피하는가

이 논문은 그 pipeline을 실제로 ablation한 드문 공개 사례다.

## 7-2. Reuse potential

### Internal agent data curation

Internal coding or terminal agent를 만든다면 같은 stage를 사용할 수 있다.

1. Task source를 모은다.
2. Core eval에서 source를 ranking한다.
3. 상위 source를 섞는다.
4. Difficulty proxy로 task description을 filtering한다.
5. Teacher leaderboard가 아니라 student transfer 기준으로 teacher를 고른다.
6. 의미 있는 multi-turn behavior가 있는 trajectory를 남긴다.
7. Rollout 반복만 늘리지 말고 new task로 scale한다.

### Benchmark design

OpenThoughts-TB-Lite 같은 fast proxy benchmark는 유용하다. Full Terminal-Bench 2.0은 비쌀 수 있으므로 curated lite benchmark가 data recipe iteration을 가속한다.

### Teacher evaluation

Teacher는 teacher solve rate가 아니라 downstream student performance로 평가해야 한다. Code, search, tool-use data generation에도 그대로 적용된다.

## 7-3. Production considerations

- 모든 generated task와 trajectory의 provenance를 추적한다.
- Held-out OOD benchmark는 recipe search 밖에 완전히 둔다.
- Trajectory shortcut behavior와 unsafe tool use를 audit한다.
- Average score를 과최적화하지 말고 per-benchmark regression을 확인한다.
- Base model이 바뀌면 teacher choice를 다시 평가한다.
- Source task와 generated trace의 license를 확인한다.

## 7-4. Follow-up papers

- OpenThoughts
- SWE-Smith
- SERA
- Nemotron-Terminal
- SWE-bench
- Terminal-Bench
- Aider Polyglot
- BFCL
- GAIA
- Agentic RL data curation papers

# 8. Summary

- OpenThoughts-Agent는 agentic SFT와 RL을 위한 open data recipe를 다루는 논문이다.
- Task source, mixing, augmentation, filtering, teacher choice, trajectory filtering, scaling을 ablation한다.
- Final 100K SFT data는 Qwen3-32B를 7개 agentic benchmark 평균 44.8%까지 fine-tune한다.
- 가장 강한 insight는 broad agent capability가 task diversity와 trajectory quality에 크게 의존한다는 점이다.
- Strong solver model이 자동으로 best SFT teacher가 되지는 않는다.
