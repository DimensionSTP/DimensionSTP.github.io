---
layout: single
title: "2 OLMo 2 Furious Review"
categories: Study-concept
tag: [OLMo2, LLM, TrainRecipe]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2501.00656)

2 OLMo 2 Furious는 "좋은 open model을 만들었다"는 수준에서 읽으면 안되는 논문이다. 이 논문의 진짜 가치는 **현대 open LLM을 어떻게 안정적으로 학습시키고, 어떤 데이터 커리큘럼으로 능력을 끌어올리고, 어떤 post-training loop로 assistant 성격을 붙이는가**를 꽤 정직하게 공개했다는 데 있다.

특히 요즘은 open-weight 모델이 많아졌지만, 여전히 많은 경우에 공개되는 것은 **최종 체크포인트**이지 **만드는 과정**은 아니다. 그런데 실제 연구나 실무에서 재사용 가치가 높은 것은 결과물 그 자체보다도, 어떤 실패를 겪었고 어떤 선택이 왜 유효했는가를 보여주는 recipe다. OLMo 2는 그 점에서 모델 소개 논문이라기보다 **foundation LLM full-stack recipe report**에 가깝다.

> 한 줄 요약: OLMo 2는 안정성 중심의 Transformer 수정, 두 단계 base training, Dolmino Mix 1124 기반 late-stage curriculum, Tulu 3식 SFT -> DPO -> RLVR post-training을 하나의 재현 가능한 fully-open recipe로 묶은 논문이다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- 요즘 open model 성능 차이는 아키텍처 한두 줄보다 **데이터 품질, mid-training 설계, post-training 운영**에서 더 크게 갈리는 경우가 많다.
- 이 논문은 base model, data mix, training code, eval suite, training logs까지 함께 공개해서 **어떻게 만들었는가**를 따라가기 좋다.
- 특히 foundation LLM을 만들 때 중요한 **pretrain / mid-train / post-train의 역할 분리**가 아주 선명하게 드러난다.

내가 보기엔 이 논문은 "OLMo 2가 잘한다"보다, **현대 LLM recipe는 결국 stage design의 문제**라는 사실을 잘 보여준다. broad coverage를 위한 pretraining, capability patching을 위한 mid-training, assistant behavior를 위한 post-training이 뒤섞이면 설명도 재현도 어려워지는데, OLMo 2는 그 경계를 꽤 잘 나눠서 보여준다.

# 1. Problem Setting

## 1-1. Problem definition

- 이 논문이 겨냥하는 핵심 문제는 **fully open language model이 강한 성능과 재현성을 동시에 달성하기 어렵다**는 점이다.
- open ecosystem에는 이미 많은 open-weight 모델이 있지만, 실제 연구 관점에서 중요한 데이터, 코드, 중간 체크포인트, 실험 로그까지 모두 공개하는 경우는 드물다.
- 또 base model 개발은 단순히 "더 많은 토큰을 학습시킨다"로 끝나지 않는다. **학습 안정성**, **per-token efficiency**, **late-stage capability shaping**, **post-training controllability**를 동시에 다뤄야 한다.
- 즉 이 논문의 문제 설정은 "좋은 open model 하나 만들기"가 아니라, **성능이 충분히 강하면서도 fully-open한 end-to-end recipe를 설계하는 것**에 가깝다.

## 1-2. Why previous approaches are insufficient

- 기존 fully open 계열은 투명성 면에서는 강했지만, 성능 면에서는 최신 open-weight 모델들과 격차가 있는 경우가 많았다.
- 반대로 최근 강한 open-weight 모델들은 최종 weights는 공개해도, 실제 성능을 만든 data mixture, training loop, checkpoint selection 과정은 충분히 드러나지 않는 경우가 많다.
- 또 generic pretraining mix만으로는 broad language capability는 얻을 수 있어도, **math, STEM reference, instruction following** 같은 영역을 균형 있게 끌어올리기 어렵다.
- 여기에 학습 스파이크, initialization 불안정, attention logit 폭주 같은 문제까지 겹치면, 대규모 학습은 사소해 보이는 구현 차이에도 민감해진다.
- 결국 기존 접근의 한계는 단일 요소의 부족이 아니라, **아키텍처 안정화 / 데이터 커리큘럼 / 평가 위생 / post-training objective가 하나의 시스템으로 설계되지 않았다**는 데 있다.

# 2. Core Idea

## 2-1. Main contribution

- OLMo 2의 핵심 기여는 하나의 아키텍처 혁신이라기보다 **full-stack recipe의 재구성**이다.
- 첫째, base Transformer에 안정성 중심의 수정들을 넣는다. RMSNorm, reordered norm, QK-norm, z-loss, 더 큰 RoPE θ, tokenizer 변경이 여기에 해당한다.
- 둘째, base training을 **pretraining + mid-training**의 두 단계로 나눈다. pretraining은 broad coverage를 맡고, mid-training은 high-quality text와 math/STEM capability patching을 맡는다.
- 셋째, mid-training을 단순 추가 학습으로 보지 않고, **Dolmino Mix 1124라는 late-stage curriculum**으로 설계한다.
- 넷째, instruct model은 Tulu 3 recipe를 기반으로 **SFT -> DPO -> RLVR**의 stage-aligned pipeline으로 만든다.
- 다섯째, 모델, 데이터, 코드, eval, logs를 함께 공개해 **결과보다 과정이 더 중요한 논문**으로 만든다.

## 2-2. Design intuition

- 이 논문의 설계 직관은 꽤 명확하다. 좋은 foundation model은 한 번의 거대한 pretraining으로 완성되지 않는다.
- broad world knowledge와 token efficiency는 주로 pretraining에서 잡고,
- 남은 capability gap, 특히 math나 고품질 reference exposure는 mid-training에서 보강하고,
- 실제 assistant behavior는 post-training에서 shaping해야 한다.
- 즉, 모든 문제를 한 stage에서 해결하려는 대신, **각 stage가 맡아야 할 역할을 분리하는 것**이 이 논문의 핵심이다.
- 그래서 OLMo 2를 "새 모델"로 보기보다, **stage-separated foundation LLM construction manual**로 읽는 편이 더 낫다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | fully-open 조건을 유지하면서 training stability, per-token efficiency, downstream adaptability를 함께 끌어올리는 것 |
| Key module | 안정성 중심 Transformer 수정 + two-stage base training + Dolmino mid-training + Tulu 3식 post-training |
| Core design principle | broad capability는 pretraining, capability patching은 mid-training, assistant behavior는 post-training에서 해결 |
| Difference from prior work | 새 블록 하나보다 recipe 전체를 공개하고, 데이터/학습/평가/운영을 한 시스템으로 묶음 |

## 3-2. Module breakdown

### 1) Stability-oriented Transformer updates

- OLMo 2는 decoder-only Transformer를 유지한다.
- 다만 핵심은 "Transformer를 바꿨다"가 아니라 **학습이 덜 흔들리게 만드는 세부 수정들**을 넣었다는 점이다.
- 주요 변경점은 다음과 같다.
  - nonparametric LayerNorm 대신 RMSNorm 사용
  - attention/MLP 입력이 아니라 출력 쪽을 정규화하는 reordered norm
  - query / key projection에 RMSNorm을 거는 QK-norm
  - softmax logits 폭주를 줄이기 위한 z-loss
  - RoPE θ를 10,000에서 500,000으로 증가
  - embedding weight decay 제거
- 이 조합은 모델 표현력을 새로 정의한다기보다, **대규모 dense LM이 덜 깨지게 만드는 안전한 operating point**를 찾는 쪽에 가깝다.

### 2) Tokenizer update

- tokenizer도 중요한 변화다.
- OLMo 2는 GPT-3.5 / GPT-4 계열의 cl100k 기반 pre-tokenizer와 vocabulary를 빌려오고, 이전 OLMo의 masking tokens는 유지한다.
- 이 변화는 겉보기엔 작아 보이지만, 작은 규모 실험에서도 OLMES 쪽에서 측정 가능한 개선을 보인다.
- 즉, 이 논문은 architecture만큼이나 **text interface 자체도 recipe의 일부**로 본다.

### 3) Two-stage base training

- base training은 크게 두 단계다.
- Stage 1 pretraining은 전체 학습 FLOPs의 90~95%를 사용하며, mostly web-sourced data 위에서 broad capability를 확보한다.
- Stage 2 mid-training은 남은 5~10% FLOPs를 사용하며, high-quality web / curated non-web / synthetic math 중심의 Dolmino Mix 1124를 투입한다.
- 여기서 중요한 건 mid-training을 단순 fine-tuning이 아니라, **late-stage curriculum**으로 설계했다는 점이다.

### 4) Model soups

- OLMo 2에서 꽤 실용적인 포인트는 mid-training을 여러 번 돌리고 평균 내는 **checkpoint soup** 전략이다.
- 7B는 50B-token mid-training run 3개를 평균하고,
- 13B와 32B는 100B-token run 3개와 300B-token run 1개를 평균한다.
- 이 전략은 "어떤 단일 seed가 잘 나오길 바란다"보다, **고품질 anneal을 여러 번 하고 합쳐서 더 좋은 local minimum을 찾는다**는 관점에 가깝다.

### 5) Tulu 3-aligned post-training pipeline

- instruct 모델은 Tulu 3 recipe를 거의 그대로 계승하되, permissive license와 OLMo 2 특성에 맞게 조정한다.
- pipeline은 SFT -> DPO -> RLVR의 3단계다.
- SFT는 broad assistant prior,
- DPO는 preference shaping,
- RLVR는 correctness를 검증할 수 있는 영역, 특히 수학과 제약 추종을 밀어주는 역할을 맡는다.
- 이 stage 분리는 이후 다른 base model이나 MLLM에도 꽤 잘 이식될 수 있는 생각법이다.

# 4. Training / Data / Recipe

## 4-1. Data

- pretraining data인 **OLMo 2 Mix 1124**는 약 3.9T tokens 규모다.
- 구성은 DCLM 기반 web text가 대부분이고, 여기에 StarCoder code, peS2o academic papers, arXiv STEM papers, OpenWebMath, Algebraic Stack, Wikipedia/Wikibooks 등이 섞인다.
- 논문에서 특히 흥미로운 점은, pretraining mix가 단순히 "인터넷을 크게 긁어온 데이터"가 아니라 **web + code + academic + math reference**를 의도적으로 섞은 구조라는 점이다.

- mid-training data인 **Dolmino Mix 1124**는 훨씬 더 공격적으로 설계된다.
- high-quality subset 쪽에는 다음이 들어간다.
  - 더 강한 품질 필터를 통과한 DCLM web
  - decontaminated FLAN
  - peS2o
  - Wikipedia/Wikibooks
  - curated StackExchange Q&A
- math mix 쪽에는 TuluMath, TinyGSM-MIND, MathCoder2-style synthetic math, filtered Metamath, CodeSearchNet, GSM8K train split 등이 들어간다.
- 핵심은 broad data를 더 넣는 게 아니라, **이미 배운 모델에 어떤 high-quality exposure를 late stage에 다시 주면 능력이 바뀌는가**를 묻는 방식이다.

- post-training 데이터도 stage별로 다르다.
- SFT는 Tulu 3의 high-quality instruction data에 PersonaHub 기반 synthetic SFT를 섞는다.
- DPO는 여러 모델의 응답을 수집하고 synthetic preference를 만든다.
- RLVR는 GSM8K, MATH, prompts with constraints처럼 **정답 검증이 가능한 문제들**에 집중한다.

## 4-2. Training strategy

- base model은 7B, 13B, 32B 세 스케일로 전개된다.
- 논문에서 공개한 핵심 하이퍼파라미터를 간단히 정리하면 아래와 같다.

| Model | Layers | d_model | Attention | Context | Total tokens |
| --- | --- | --- | --- | --- | --- |
| OLMo 2 7B | 32 | 4096 | MHA (32/32) | 4096 | 4.05T |
| OLMo 2 13B | 40 | 5120 | MHA (40/40) | 4096 | 5.6T |
| OLMo 2 32B | 64 | 5120 | GQA (40/8) | 4096 | 6.6T |

- warmup은 공통적으로 2000 steps다.
- peak learning rate와 cosine schedule 길이는 scale마다 다르게 잡는다.
- 이 부분에서 내가 흥미롭게 본 점은, 논문이 "어떤 LR이 정답이다"를 과장하지 않고, 오히려 **상당히 넓은 plateau 안에서 안정적으로 잘 학습되는 설정을 찾는 것**에 초점을 둔다는 점이다.

- mid-training은 OLMo 2의 진짜 하이라이트다.
- 저자들은 high-quality source와 math source를 무작정 넣지 않고, **microanneal**이라는 작은 annealing 실험으로 각각의 데이터 품질을 싸게 평가한다.
- 이 과정에서 꽤 실무적인 인사이트가 나온다.
  - domain-specific data는 많지 않아도 도움이 된다.
  - 고품질 domain data는 약간의 duplication만으로도 추가 이득을 줄 수 있다.
  - code 스타일로 쓰인 math 데이터를 자연어 풀이 스타일로 rewriting하면 성능이 크게 좋아질 수 있다.
- 즉, 중요한 건 "math 데이터를 더 모으자"가 아니라, **모델이 먹기 쉬운 형태로 다시 써주는 것**이다.

- checkpoint soup도 중요한 training 전략이다.
- 7B는 50B mid-training run 3개 평균,
- 13B와 32B는 100B run 3개 + 300B run 1개 평균을 사용한다.
- 이건 연구 관점에서도 흥미롭다. 좋은 late-stage data mix를 찾은 뒤에는, **한 번 더 긴 학습을 하는 것만큼이나 여러 anneal을 평균내는 것이 중요할 수 있다**는 뜻이기 때문이다.

- post-training은 SFT -> DPO -> RLVR 순으로 진행된다.
- SFT mix는 약 939K prompts 규모다.
- DPO는 on-policy synthetic preference를 포함하고,
- 7B / 13B는 PPO 기반 RLVR, 1B / 32B는 GRPO 기반 RLVR을 적용한다.
- RLVR는 한 번으로 끝나지 않고, 13B의 경우 GSM8K와 MATH 쪽 성능을 보면서 추가 stage를 더 밟는다.
- 이 점이 중요하다. RL을 "마지막 만능 비법"처럼 쓰지 않고, **특정 capability gap을 메우는 targeted stage**로 쓴다.

## 4-3. Engineering notes

- 이 논문에서 가장 실무적인 부분은 사실 architecture보다 engineering이다.
- 먼저 repeated n-gram이 training loss spike를 유발할 수 있다고 보고, 데이터 필터링 단계에서 이런 문서를 제거한다.
- 또 initialization을 바꾸고, gradient spike score까지 정의해서 안정성을 수치적으로 추적한다.
- reordered norm + QK-norm + z-loss 조합도 같은 맥락이다. 성능 향상 이전에 **학습을 끝까지 무사히 가져가는 것**이 먼저라는 태도다.

- evaluation hygiene도 좋다.
- 저자들은 development benchmark와 held-out benchmark를 분리해서, recipe가 개발셋 과적합인지 아닌지를 따로 본다.
- 완전한 contamination-free를 보장하는 건 아니더라도, 적어도 "우리는 어떤 평가를 보면서 개발했는가"를 명시하는 태도는 다른 open model technical report보다 낫다.

- 마지막으로 infrastructure 섹션도 무시하기 어렵다.
- Jupiter / Augusta 클러스터, Beaker workload system, GPU health check, restart / quarantine 정책까지 상세히 쓴다.
- 보통 이런 부분은 재미없다고 넘기기 쉬운데, 실제로 foundation model을 만들 때는 **좋은 optimizer보다 먼저 좋은 운영 체계가 필요하다**는 사실을 상기시켜 준다.

# 5. Evaluation

## 5-1. Main results

- base model 관점에서 OLMo 2는 fully-open 조건을 유지하면서도 꽤 강한 성능을 보인다.
- OLMES subset 기준으로 7B는 average 62.9, 13B는 68.3, 32B는 73.3을 기록한다.
- 절대 수치만 보면 일부 강한 open-weight 모델이 더 높은 곳도 있지만, 이 논문의 포인트는 **training FLOPs 대비 성능과 openness를 함께 봤을 때 Pareto frontier에 올라간다**는 점이다.

- instruct 모델도 stage별 개선이 선명하다.
- 예를 들어 13B 기준으로 average score는
  - SFT 56.6
  - DPO 62.0
  - RLVR까지 거친 final Instruct 63.4
  로 올라간다.
- GSM8K, IFEval, MATH 같은 verifiable / structured 영역에서 특히 개선이 크다.
- 13B Instruct는 peer 8B급 open instruction models를 확실히 넘고, 일부 14B급 모델에 근접하는 성능을 보여준다.

- 내가 보기엔 여기서 중요한 건 "Qwen을 이겼나" 같은 headline보다,
  1. fully-open 조건에서
  2. base와 instruct 모두
  3. 여러 scale에 걸쳐
  4. 일관된 recipe improvement를 보여줬다는 점이다.

## 5-2. What really matters in the experiments

- 이 논문에서 진짜 의미 있는 평가는 개별 benchmark 1~2개가 아니다.
- 첫째, **compute 대비 성능**이다. Figure 1은 OLMo 2가 fully-open 모델로서 꽤 좋은 efficiency operating point에 있다는 걸 보여준다.
- 둘째, **development vs held-out 분리**다. 성능 향상이 단순 benchmark fitting이 아니라는 최소한의 근거를 제공한다.
- 셋째, **stage별 ablation**이다. SFT, DPO, RLVR가 각각 무슨 역할을 하는지 분리해서 보여준다.
- 넷째, **mid-training 분석**이다. 이 논문은 단순히 Dolmino가 좋다고 끝내지 않고, 어떤 종류의 source가 왜 도움이 되는지 microanneal로 설명한다.

- 개인적으로 가장 좋았던 실험 해석은 세 가지다.
  - late-stage high-quality exposure는 생각보다 강력하다.
  - math data는 양보다도 **형태와 rewriting 방식**이 중요하다.
  - checkpoint soup는 애매한 보조 기법이 아니라, mid-training recipe의 핵심 일부가 될 수 있다.

# 6. Limitations

1. 이 논문은 매우 많은 개선 요소를 동시에 다루기 때문에, **어떤 요소가 최종 성능에 얼마나 기여했는지 완전히 분리하기 어렵다**. 실무에서는 오히려 이런 논문이 유용하지만, 연구적으로는 attribution이 흐려질 수 있다.

2. fully-open이라고 해서 쉽게 재현 가능한 것은 아니다.  
   7B조차 4T 이상 토큰, 여러 mid-training run, checkpoint soup, RLVR까지 포함한다. 따라서 이 recipe는 "투명한 recipe"이지 "가벼운 recipe"는 아니다.

3. mid-training이 math / STEM / high-quality text에 강하게 최적화되어 있기 때문에, 같은 방식이 coding, multilingual, tool use, agent loop 같은 영역에도 그대로 통할지는 추가 검증이 필요하다.

4. evaluation hygiene는 좋지만 완전무결하진 않다.  
   held-out set을 분리해도, 다른 모델들이 같은 벤치마크를 개발 중 보지 않았다는 보장은 없다. 즉, 상대 비교의 공정성은 여전히 외부 변수의 영향을 받는다.

5. 논문 버전에 따라 서술 범위가 다르다.  
   COLM shorter version과 arXiv 최신 버전은 범위가 조금 다를 수 있어서, 공개 글에서 특정 표나 수치를 인용할 때는 버전을 맞춰 확인하는 편이 안전하다.

# 7. My Take

## 7-1. Why this matters for my work

- 내가 이 논문을 높게 보는 이유는, modern foundation LLM development를 **단일 모델 설계가 아니라 pipeline design**으로 보게 만들기 때문이다.
- 특히 앞으로 foundation MLLM이나 domain-specialized foundation model을 만들 때도, 같은 질문을 하게 된다.
  - 무엇을 broad pretraining에 맡길 것인가?
  - 무엇을 late-stage curriculum으로 밀어줄 것인가?
  - 무엇을 post-training objective로 해결할 것인가?
- OLMo 2는 이 질문에 꽤 깔끔한 기본 틀을 준다.

## 7-2. Reuse potential

- 내가 실제로 재사용하고 싶은 포인트는 아래 쪽이다.
- **microanneal 방식**  
  고비용 full run 전에 data source 품질을 싸게 가늠하는 방식으로 바로 응용 가능하다.
- **late-stage curriculum 분리**  
  pretraining과 capability patching을 한 덩어리로 보지 않고 분리하는 사고방식이 좋다.
- **checkpoint soup**  
  특히 마지막 anneal 단계에서 단일 seed best checkpoint보다 더 실용적일 수 있다.
- **held-out evaluation 선언**  
  사내 실험에서도 dev set과 "절대 안 보는 셋"을 분리하는 습관으로 옮기기 좋다.
- **stability-first engineering**  
  repeated n-gram filter, initialization analysis, spike score 같은 아이디어는 대형 학습뿐 아니라 중형 실험에도 적용할 만하다.

## 7-3. Follow-up papers

- Tulu 3
- OLMoE
- Phi-4 Technical Report

# 8. Summary

- 2 OLMo 2 Furious는 새 아키텍처 하나보다 **foundation LLM recipe 전체**를 공개한 논문이다.
- 핵심은 안정성 중심 Transformer 수정, two-stage base training, Dolmino late-stage curriculum, Tulu 3식 post-training이다.
- 특히 mid-training을 capability patching stage로 분리하고, microanneal과 checkpoint soup로 다듬은 점이 인상적이다.
- evaluation에서도 compute 대비 성능과 held-out hygiene를 함께 보려는 태도가 좋다.
- fully-open foundation model을 만들고 싶다면, 이 논문은 결과보다도 **생산 공정 자체를 해부하기 좋은 문서**다.
