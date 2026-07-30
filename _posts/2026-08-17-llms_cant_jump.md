---
layout: single
title: "Position: LLMs can't jump Review"
categories: Study-concept
tag: [LLM, Scientific-Discovery, World-Model]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://openreview.net/forum?id=klU4737opt)

[ICML 2026 listing](https://icml.cc/Downloads/2026)

> 한 줄 요약: 이 position paper는 current LLM이 induction과 deduction에는 강해지고 있지만, sparse observation에서 새로운 explanatory axiom을 제안하는 abduction의 "jump"는 구조적으로 갖추지 못했다고 주장하며, artificial scientific discovery의 다음 병목을 physically grounded world model에서 formal axiom으로 넘어가는 interface로 본다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Scientific discovery를 benchmark score나 theorem proving 성능이 아니라 hypothesis formation의 구조로 다시 묻는다.
- Induction, deduction, abduction을 분리해 current generative model의 강점과 빈칸을 명확한 논쟁거리로 만든다.
- Einstein의 General Relativity 형성 과정을 단순한 genius narrative가 아니라 sensory simulation, physical intuition, axiom formation, formal deduction의 pipeline으로 읽는다.
- "Creativity as compression"만으로 data-poor scientific leap를 설명할 수 있는지 문제를 제기한다.
- Multimodal world model 연구가 next-token prediction을 넘어 scientific invention과 연결되려면 무엇이 더 필요한지 구체적인 bottleneck을 제시한다.

이 논문은 새로운 model, dataset, benchmark를 제안하지 않는다. 제목처럼 강한 position을 제시하고, 역사적 case와 inference theory를 사용해 current LLM의 한계를 논증한다.

출발점은 Albert Einstein이 Maurice Solovine에게 보낸 letter의 diagram이다. 논문은 이 diagram을 discovery cycle로 읽는다.

1. Sensory experience에서 출발한다.
2. Experience만으로 직접 도출되지 않는 axiom으로 jump한다.
3. Axiom에서 logical consequence를 deduction한다.
4. Consequence를 다시 experience와 비교한다.
5. 실패하면 cycle을 반복한다.

저자가 강조하는 것은 2번이다. Observation을 잘 압축하거나 이미 주어진 premise에서 theorem을 증명하는 것과, 어떤 premise를 세워야 하는지 제안하는 것은 다른 computational problem이라는 주장이다.

이 논문의 핵심 질문은 다음처럼 정리할 수 있다.

> 충분히 큰 language model이 모든 논문과 실험 기록을 읽으면, 아직 표현되지 않은 새로운 explanatory axiom까지 스스로 만들 수 있는가?

논문의 답은 부정적이다. Current LLM은 corpus에 나타난 regularity를 induction하고, formal tool과 search를 결합해 deduction을 수행할 수 있지만, sparse and weak evidence에서 새로운 hypothesis space를 여는 abductive mechanism이 부족하다고 본다.

다만 이 결론은 theorem이 아니다. "Structurally incapable"라는 표현은 position의 중심 주장이지, 모든 possible LLM architecture나 agentic system에 대해 실험적으로 증명된 impossibility result는 아니다. 따라서 이 글은 논문의 논지를 충실히 해설하되, argument의 강도와 evidence의 범위를 분리해서 본다.

# 1. Problem Setting

## 1-1. Problem definition

논문은 scientific reasoning을 세 가지 inference mode로 나눈다.

| Inference mode | Given | Produce | Typical AI framing |
| --- | --- | --- | --- |
| Induction | Cases and observations | General pattern or rule | Statistical learning, compression, prediction |
| Deduction | Rule and premises | Logical consequence | Proof search, symbolic reasoning, theorem proving |
| Abduction | Surprising observation | Plausible explanatory hypothesis | New premise, model, mechanism, or axiom formation |

간단한 notation으로 observation set을 $E$, axiom set을 $A$, deduced consequence를 $C$라고 하자.

Deduction은 대략 다음 mapping을 다룬다.

$$
A \rightarrow C
$$

Induction은 여러 experience에서 regularity를 추정한다.

$$
E \rightarrow \hat{A}_{\mathrm{pattern}}
$$

하지만 논문이 말하는 abductive jump는 단순 empirical fit을 넘는다.

$$
E \xRightarrow{J} A_{\mathrm{new}}
$$

여기서 $J$는 observation에 이미 명시된 pattern을 그대로 요약하는 것이 아니라, observation을 설명할 새로운 conceptual structure를 제안하는 과정이다.

논문은 modern LLM이 first mapping과 second mapping의 많은 부분을 수행할 수 있다고 본다. Large corpus에서 pattern을 학습하고, code or proof tool을 사용하며, 이미 제시된 formal language 안에서 search를 확장할 수 있기 때문이다.

문제는 $J$다. Data가 풍부하고 target regularity가 corpus에 반복되면 induction으로도 새로운 것처럼 보이는 output을 만들 수 있다. 그러나 evidence가 sparse하고, 기존 theory도 대부분 observation을 잘 설명하며, 새로운 conceptual vocabulary 자체가 필요한 경우에는 compression만으로 discovery를 설명하기 어렵다는 것이 저자의 position이다.

## 1-2. Why previous approaches are insufficient

### 1) Better prediction이 곧 new explanation은 아니다

Next-token prediction은 observed distribution의 regularity를 압축하고 재구성하는 데 강하다. Scientific text를 충분히 학습하면 known theory를 설명하고, analogy를 만들고, candidate experiment를 제안할 수 있다.

그러나 predictive adequacy와 explanatory novelty는 다르다. 두 theory가 현재 data를 비슷하게 설명하더라도, one theory가 새로운 mechanism과 counterfactual structure를 제안할 수 있다.

논문은 General Relativity를 이 gap의 case로 사용한다. 당시 available observation이 압도적으로 Einstein의 theory를 강제한 것은 아니며, Newtonian framework도 많은 현상을 잘 설명했다. 따라서 historical evidence를 단순히 더 잘 compress하면 equivalence principle이 자동으로 나온다고 보기 어렵다는 주장이다.

### 2) Formal deduction은 premise selection을 대신하지 못한다

Theorem prover나 code search system은 premise가 주어졌을 때 vast search space를 탐색할 수 있다. AlphaProof류 system이 보여주는 발전도 deduction의 중요성을 분명히 한다.

하지만 theorem이 무엇이어야 하는지, 어떤 primitive concept를 도입할지, 어떤 quantity를 invariant로 볼지 결정하는 것은 다른 문제다.

- 어떤 axiom set을 채택할 것인가.
- 어떤 representation에서 problem이 단순해지는가.
- 어떤 observation을 anomaly로 취급할 것인가.
- 어떤 thought experiment가 informative한가.
- 기존 vocabulary를 버리고 어떤 new concept를 만들 것인가.

Deduction engine이 강해져도 이 selection problem이 자동으로 해결되지는 않는다.

### 3) Creativity as compression은 data-poor discovery를 과소설명할 수 있다

Generative creativity를 learned distribution의 recombination이나 minimum description length 관점에서 설명할 수 있다. 이 관점은 art, design, language variation, many engineering optimizations에 상당한 설명력을 가진다.

논문은 그 설명을 부정하기보다 범위를 제한한다. Strong statistical signal이 없는 상황에서 radically new explanatory structure가 나오는 case는 compression alone으로 충분히 설명되지 않는다고 주장한다.

### 4) Text-only grounding은 physical counterfactual을 제한한다

Scientific abduction은 text pattern만이 아니라 counterfactual manipulation과 embodied constraint를 요구할 수 있다.

예를 들어 다음 질문은 language association만으로 다루기 어렵다.

- Observer가 free fall하면 local measurement는 어떻게 달라지는가.
- Reference frame을 바꾸면 어떤 quantity가 invariant하게 남는가.
- Hypothetical world에서 law를 바꾸면 observed trajectory가 어떻게 변하는가.
- Visual or physical simulation의 regularity를 어떤 formal object로 표현해야 하는가.

논문은 physically consistent multimodal world model을 이 gap을 줄이는 방향으로 제안한다.

# 2. Core Idea

## 2-1. Main contribution

이 position paper의 contribution은 method보다 argument structure에 있다.

### 1) Einstein discovery cycle을 computational decomposition으로 사용

논문은 Einstein의 Solovine diagram을 다음 cycle로 해석한다.

| Stage | Role |
| --- | --- |
| Experience $E$ | Observation, sensation, experiment, anomaly |
| Jump $J$ | Experience에서 explanatory axiom으로 이동 |
| Axiom $A$ | New conceptual premise or theory structure |
| Deduction | Axiom에서 testable consequence를 도출 |
| Verification | Consequence를 experience와 비교 |

이 decomposition은 scientific AI를 하나의 monolithic intelligence score로 보지 않게 만든다. Observation processing, hypothesis generation, formalization, proof, experiment design은 서로 다른 capability다.

### 2) Induction, deduction, abduction의 capability gap 제시

논문의 position은 다음 세 문장으로 압축할 수 있다.

- Generative AI는 induction에 매우 강하다.
- Formal reasoning과 tool use를 통해 deduction도 빠르게 발전하고 있다.
- Abduction, 특히 new axiom formation에는 필요한 mechanism이 없다.

이 구분의 실용적 가치는 benchmark interpretation에 있다. Model이 known theorem을 증명하거나 literature에서 hidden relation을 찾았다고 해서, genuinely new explanatory premise를 만들었다고 결론 내리면 안 된다.

### 3) General Relativity를 computational case study로 사용

논문은 Einstein의 discovery를 다음 요소로 본다.

- Sparse and ambiguous observational evidence
- Free-fall thought experiment
- Equivalence principle
- Physical intuition과 formal mathematics 사이의 gap
- Marcel Grossmann과의 collaboration
- Differential geometry를 통한 formalization
- Deduction과 empirical verification의 반복

핵심은 Einstein이 raw data에서 equation을 direct regression한 것이 아니라, a physically meaningful simulation-like intuition에서 principle을 만들고, 이후 mathematical language를 찾았다는 해석이다.

### 4) Critical bottleneck을 simulation-to-axiom translation으로 정의

World model만 만들면 scientific discovery가 해결된다는 주장은 아니다. 논문이 더 날카롭게 지목하는 bottleneck은 다음 interface다.

$$
\text{Counterfactual simulation} \rightarrow \text{Formal axiom}
$$

World model이 plausible trajectory를 생성하더라도, 그 trajectory에서 invariant, symmetry, causal mechanism, law candidate를 추출해 formal statement로 바꾸지 못하면 deduction system에 넘길 수 없다.

### 5) Physically grounded multimodal world model을 research direction으로 제안

Text corpus만 학습한 model보다 다음 capability를 가진 system이 abductive jump에 가까워질 수 있다고 본다.

- Action-controllable simulation
- Multimodal sensory state
- Physical consistency
- Counterfactual intervention
- Long-horizon consequence prediction
- Formal concept extraction

## 2-2. Design intuition

논문의 intuition은 "discovery는 output novelty가 아니라 explanatory cycle의 completion"이라는 데 있다.

LLM이 novel sentence를 만들거나 unseen molecule을 제안할 수 있다. 하지만 scientific discovery라고 부르려면 최소한 다음 chain이 필요하다.

1. Anomaly or gap을 식별한다.
2. Candidate mechanism을 제안한다.
3. Mechanism을 formal premise로 표현한다.
4. New prediction을 deduction한다.
5. Experiment나 simulation으로 falsify한다.
6. Failure에 따라 hypothesis를 수정한다.

Current LLM system은 이 chain의 일부를 tool orchestration으로 구현할 수 있다. Literature search, code generation, theorem proving, experiment scripting, report writing은 이미 가능하다.

논문은 그럼에도 central bottleneck이 남는다고 본다. Existing representation 안에서 candidate를 search하는 것과, representation 자체를 바꾸는 것은 다르기 때문이다.

이 distinction은 agentic scientific discovery를 설계할 때 중요하다. More rollouts, larger context, better retrieval이 search coverage를 넓힐 수는 있지만, hypothesis language가 잘못되어 있으면 search space 전체가 relevant discovery를 포함하지 않을 수 있다.

# 3. Architecture / Method

이 논문은 model architecture를 제안하지 않는다. 따라서 이 section에서는 논문의 argument architecture와 implied system architecture를 나눠 본다.

## 3-1. Argument overview

| Item | Description |
| --- | --- |
| Paper type | ICML 2026 Position Paper |
| Central claim | Current LLM lacks abductive jump from experience to new axioms |
| Conceptual basis | Einstein's Solovine diagram and induction/deduction/abduction distinction |
| Historical case | Formation of General Relativity |
| Critique target | Creativity as compression and scale-only discovery narratives |
| Proposed direction | Physically consistent multimodal world models |
| Claimed bottleneck | Translation from simulation and intuition into formal axioms |
| Evidence type | Conceptual argument, history of science, capability decomposition |
| Not provided | New benchmark, trained model, controlled ablation, impossibility theorem |

## 3-2. Argument breakdown

### 1) Sense experience is not a token sequence alone

Einstein's diagram starts from sensory experience. 논문은 scientific cognition이 language description보다 richer state를 사용한다고 본다.

Relevant state는 다음을 포함할 수 있다.

- Visual scene
- Motion and force
- Observer perspective
- Action consequence
- Temporal evolution
- Measurement uncertainty
- Counterfactual variation

Text can encode these, but text description은 observation의 full structure를 이미 abstraction한 결과다. Abductive system이 text only로 시작하면 human writer가 선택한 concept와 vocabulary에 강하게 제한될 수 있다.

### 2) The jump proposes explanatory structure

Jump는 random novelty가 아니다. Observation과 연결되면서도 observation에 deductively entailed되지 않는 hypothesis를 만든다.

Candidate axiom $A$는 적어도 다음 property를 가져야 한다.

- Observed anomaly를 설명한다.
- Existing knowledge와 가능한 한 양립한다.
- New testable consequence를 만든다.
- Simplicity or coherence를 가진다.
- Alternative hypothesis와 비교할 수 있다.

즉 abduction은 unconstrained generation이 아니라 structured hypothesis proposal이다.

### 3) Deduction turns intuition into science

Axiom이 제안된 뒤에는 formal consequence를 뽑아야 한다. 이 단계에서 modern theorem prover, symbolic algebra, code execution, simulation engine이 강점을 가진다.

논문은 LLM의 deduction progress를 인정한다. 문제는 이 progress가 jump capability의 evidence로 자동 전환되지 않는다는 점이다.

### 4) Verification closes the loop

Scientific hypothesis는 plausible story가 아니라 falsifiable model이어야 한다. 따라서 world model이 vivid simulation을 만들더라도 physical consistency와 external measurement가 필요하다.

이 점에서 proposed world model은 일반 video generator와 다르다.

- Action에 반응해야 한다.
- Hidden state가 persistent해야 한다.
- Physical constraint를 장기적으로 유지해야 한다.
- Counterfactual intervention 결과를 비교할 수 있어야 한다.
- Uncertainty를 표현해야 한다.

### 5) Simulation-to-axiom interface

논문이 제안하는 future system을 구성 요소로 풀면 다음과 같다.

| Module | Role |
| --- | --- |
| Multimodal encoder | Observation을 structured sensory state로 변환 |
| World model | Action-conditioned future와 counterfactual을 simulation |
| Anomaly detector | Current theory와 observation의 mismatch를 식별 |
| Hypothesis generator | New latent mechanism or concept candidate를 제안 |
| Formalizer | Candidate를 symbolic axiom, equation, program으로 변환 |
| Deduction engine | Consequence와 theorem을 도출 |
| Experiment planner | Hypothesis를 구분할 intervention을 선택 |
| Verifier | Real or simulated evidence로 candidate를 update |

논문이 실제로 이 full architecture를 구현한 것은 아니다. 이 table은 position의 research direction을 system design으로 해석한 것이다.

# 4. Training / Data / Recipe

이 논문에는 model training recipe가 없다. 대신 argument를 구성하는 evidence source와 future research recipe를 구분해야 한다.

## 4-1. Historical and conceptual sources

논문의 main source는 다음과 같다.

- Einstein's letter to Maurice Solovine
- General Relativity discovery narrative
- Induction, deduction, abduction의 철학적 구분
- Modern generative AI의 compression 관점
- Theorem proving and formal reasoning progress
- World model and multimodal simulation research

이 source set은 mechanism hypothesis를 만드는 데는 유용하지만, current model capability를 exhaustive하게 측정하는 dataset은 아니다.

## 4-2. General Relativity case decomposition

논문은 General Relativity를 대략 다음 computational sequence로 읽는다.

1. Existing physics and observation을 학습한다.
2. Free-fall thought experiment를 구성한다.
3. Local equivalence에 대한 intuition을 형성한다.
4. Equivalence principle을 axiom candidate로 둔다.
5. Appropriate mathematics를 탐색한다.
6. Differential geometry로 formalize한다.
7. Consequence를 deduction한다.
8. Observation과 비교한다.

이 sequence에서 LLM이 잘할 수 있는 단계와 부족한 단계를 구분하는 것이 논문의 목적이다.

- Literature synthesis: 강함
- Known mathematics retrieval: 강함
- Equation manipulation: 빠르게 발전
- Formal proof: 빠르게 발전
- Candidate thought experiment generation: 부분적으로 가능
- New principle formation: 논문이 문제 삼는 부분
- Formal axiom extraction from simulation: 핵심 bottleneck

## 4-3. Proposed research recipe

논문의 제안을 engineering recipe로 바꾸면 다음과 같은 research program이 된다.

### Stage 1: Controllable world model

- Passive next-frame prediction이 아니라 action-conditioned transition을 학습한다.
- Multiple sensory modality를 shared state에 묶는다.
- Long-horizon physical consistency를 평가한다.

### Stage 2: Counterfactual search

- Current law 아래의 expected outcome과 observed anomaly를 비교한다.
- Intervention variable을 바꾸어 causal sensitivity를 본다.
- New mechanism candidate가 만드는 distinct prediction을 생성한다.

### Stage 3: Concept and axiom induction

- Repeated simulation pattern에서 invariant를 추출한다.
- Natural language description만이 아니라 equation, program, graph, symbolic rule로 표현한다.
- Competing axiom sets의 explanatory coverage와 complexity를 비교한다.

### Stage 4: Deduction and falsification

- Formal prover와 simulator를 사용해 consequence를 계산한다.
- Information gain이 큰 experiment를 선택한다.
- Failed hypothesis를 archive하고 representation을 update한다.

이 recipe는 논문의 direct implementation이 아니라 position이 요구하는 components를 구체화한 해석이다.

# 5. Evaluation

Position paper이므로 conventional benchmark table은 없다. Evaluation은 argument를 어떤 기준으로 받아들일지에 초점을 둬야 한다.

## 5-1. What the paper establishes

### 1) Useful capability decomposition

Induction, deduction, abduction을 분리하면 current AI achievement를 더 정확히 말할 수 있다.

- Benchmark generalization이 induction인지 new principle formation인지 구분할 수 있다.
- Theorem proving success가 premise invention까지 포함하는지 확인할 수 있다.
- Agentic search가 existing hypothesis space 탐색인지 hypothesis space expansion인지 물을 수 있다.

이 decomposition 자체는 매우 유용하다.

### 2) Data-poor discovery is a hard counterexample to scale-only narratives

Strong data regularity가 없는데도 scientific principle이 나오는 역사적 case는 "more data + better compression" 설명에 부담을 준다.

General Relativity 하나가 모든 discovery를 대표하지는 않지만, 최소한 creativity theory가 sparse evidence와 conceptual reframing을 설명해야 한다는 요구를 만든다.

### 3) Simulation-to-formalization is a concrete bottleneck

"World model이 필요하다"는 주장은 흔하다. 이 논문은 한 단계 더 나아가 world model output을 formal axiom으로 바꾸는 interface를 명시한다.

이 bottleneck은 평가 가능하다.

- Simulation에서 invariant를 찾는가.
- Novel variable을 정의하는가.
- Equation or program으로 표현하는가.
- New prediction을 만드는가.
- Competing explanation을 구분하는가.

## 5-2. What the paper does not establish

### 1) Universal impossibility

Current transformer LLM이 abduction benchmark에서 실패한다고 해도, 모든 scaled LLM이나 tool-augmented system이 structurally incapable하다는 theorem은 아니다.

LLM이 proposal generator이고 simulator, theorem prover, experiment loop가 external module이라면 system-level behavior가 달라질 수 있다.

### 2) A clean operational definition of abductive success

Novel hypothesis가 truly new인지, training data에 있었는지, useful인지, explanatory인지 판단하기 어렵다.

평가에는 최소한 다음 control이 필요하다.

- Knowledge cutoff and contamination control
- Counterfactual scientific world
- Hidden governing law
- Sparse observation budget
- Multiple plausible theories
- Experiment cost
- Formal prediction check

논문은 문제를 제기하지만 full benchmark를 제공하지 않는다.

### 3) General Relativity as representative sample

Einstein의 case는 powerful illustration이지만 $n=1$ historical case에 가깝다. Discovery는 다음처럼 다양한 형태를 가진다.

- Instrument-driven discovery
- Large-scale data anomaly
- Accidental observation
- Engineering optimization
- Mathematical conjecture
- Collaborative theory synthesis
- Automated search in known formal space

Abduction의 필요 정도는 domain마다 다를 수 있다.

### 4) World model sufficiency

Physically consistent simulation은 grounding을 제공할 수 있다. 그러나 simulation이 자동으로 concept invention, relevance judgment, scientific taste, experiment selection을 해결하지 않는다.

World model은 necessary component일 수 있지만 sufficient mechanism이라고 보기 어렵다.

# 6. Limitations

1. Position의 central claim이 매우 강하다.
   - "LLMs can't jump"와 "structurally incapable"는 current evidence보다 넓게 읽힐 수 있다.
   - Architecture class, scale, training objective, tool access를 어디까지 LLM으로 포함하는지 경계가 필요하다.

2. Controlled experiment가 없다.
   - Current models를 abductive benchmark에서 systematic하게 비교하지 않는다.
   - Scaling trend나 world model intervention도 실험하지 않는다.

3. Historical reconstruction은 interpretation에 의존한다.
   - Einstein의 discovery process는 letter, retrospective account, collaborator role을 통해 재구성된다.
   - Actual cognition을 완전히 관찰한 것은 아니다.

4. Abduction과 search의 경계가 불명확하다.
   - Very large hypothesis search가 qualitatively new jump처럼 보일 수 있다.
   - Novel representation formation과 combinatorial recombination을 operationally 구분하기 어렵다.

5. Human discovery도 individual jump만으로 설명되지 않는다.
   - Collaboration, notation, instruments, prior mathematics, community criticism이 중요하다.
   - Grossmann과의 interaction은 오히려 distributed system 관점의 중요성을 보여준다.

6. Physically consistent world model의 요구가 높다.
   - Current video model은 visual plausibility와 physical law를 혼동할 수 있다.
   - Long-horizon intervention, hidden state, uncertainty, causal structure가 모두 필요하다.

7. Formalization bottleneck의 implementation이 열려 있다.
   - Simulation latent를 equation or axiom으로 변환하는 objective가 무엇인지 제시하지 않는다.
   - Symbol grounding, variable discovery, equivalence class selection이 별도 연구 문제다.

8. Text-domain scientific discovery를 과소평가할 수 있다.
   - Mathematics, algorithm, social science처럼 physical sensor grounding이 중심이 아닌 domain도 있다.
   - 이 경우 world model은 physical simulator가 아니라 formal or social environment model이어야 할 수 있다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문의 가장 큰 가치는 "LLM이 scientific discovery를 할 수 있는가"라는 거대한 질문을 answer generation, proof, hypothesis formation으로 분해한 데 있다.

현재 agentic research system은 literature retrieval, code execution, experiment orchestration, report generation을 빠르게 통합하고 있다. 이 system은 known search space 안에서 매우 강해질 수 있다. 하지만 다음 두 경우를 구분해야 한다.

1. Existing paper와 method component를 잘 조합해 새로운 candidate를 만든다.
2. Problem representation 자체를 바꾸는 new explanatory variable을 만든다.

첫 번째도 연구적으로 큰 가치가 있다. 모든 useful discovery가 Einstein-level jump일 필요는 없다. 다만 system capability를 과장하지 않으려면 둘을 같은 단어로 부르지 않는 것이 좋다.

## 7-2. Reuse potential

### 1) Scientific agent benchmark design

Abductive capability를 보려면 hidden-law environment가 필요하다.

- Training corpus에 없는 synthetic world를 만든다.
- Observation budget을 제한한다.
- Existing theory가 대부분 data를 설명하지만 특정 anomaly에서 실패하게 한다.
- Agent가 hypothesis, formal rule, discriminative experiment를 제출하게 한다.
- Final score는 novelty보다 predictive consequence와 falsifiability로 계산한다.

### 2) Research pipeline role separation

- Retriever는 prior knowledge를 모은다.
- World model은 counterfactual을 만든다.
- Hypothesis proposer는 mechanism candidate를 낸다.
- Formalizer는 equation or program으로 바꾼다.
- Prover and simulator는 consequence를 확인한다.
- Critic은 alternative explanation을 찾는다.

단일 LLM에게 모든 역할을 맡기기보다 서로 다른 failure mode를 가진 module로 분리할 수 있다.

### 3) Representation change tracking

Agent가 단순 parameter search만 하는지, variable or ontology를 새로 만드는지 log로 남긴다.

- New state variable
- New causal edge
- New symmetry
- New abstraction boundary
- New measurement proposal

이런 event를 process metric으로 잡으면 final answer correctness보다 jump-like behavior를 더 잘 볼 수 있다.

### 4) Counterfactual and falsification reward

Hypothesis의 reward를 fluent novelty로 주면 storytelling을 강화할 수 있다. 대신 다음을 reward할 수 있다.

- Multiple observation coverage
- Novel testable prediction
- Counterexample resistance
- Experiment information gain
- Simplicity under equal fit
- Successful falsification of alternatives

## 7-3. Follow-up papers

- The Logic of Abduction
- World Models
- Genie: Generative Interactive Environments
- AlphaGeometry and AlphaProof
- AI Scientist and automated experiment systems
- AlphaEvolve: A Coding Agent for Scientific and Algorithmic Discovery
- Scientific Discovery in the Age of Artificial Intelligence

# 8. Summary

- 이 position paper는 scientific discovery를 induction, deduction, abduction으로 분해한다.
- 저자는 current LLM이 pattern induction과 formal deduction에는 강하지만 new axiom을 만드는 abductive jump가 부족하다고 주장한다.
- General Relativity는 sparse evidence에서 physical intuition과 new principle이 형성된 case study로 사용된다.
- 핵심 bottleneck은 counterfactual simulation에서 formal axiom으로 넘어가는 translation이다.
- Physically consistent multimodal world model은 sensory grounding을 제공할 수 있지만, concept invention과 formalization을 자동으로 해결하지는 않는다.
- 논문의 주장은 useful research agenda이지만, universal impossibility theorem이나 controlled benchmark result로 읽으면 안 된다.
