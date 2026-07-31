---
layout: single
title: "BERT-as-a-Judge: A Robust Alternative to Lexical Methods for Efficient Reference-Based LLM Evaluation Review"
categories: Study-concept
tag: [LLM-Evaluation, BERT, Model-Judge]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2604.09497)

[Code link](https://github.com/artefactory/Bert-as-a-Judge)

[Model and data collection](https://huggingface.co/collections/artefactory/bert-as-a-judge)

> 한 줄 요약: BERT-as-a-Judge는 question, candidate answer, reference answer를 함께 입력받는 210M EuroBERT binary classifier를 약 1M synthetic correctness labels로 학습해 regex와 lexical metric의 format sensitivity를 줄이고, 최대 70B 규모 LLM judge와 비슷하거나 더 높은 reference-based evaluation accuracy를 훨씬 낮은 compute로 제공한다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- LLM benchmark score가 model capability뿐 아니라 answer parsing rule에 얼마나 민감한지 정량적으로 보여준다.
- Exact match, regex, ROUGE-L, math verifier의 장점을 유지하면서 자연어 paraphrase와 answer format variation을 처리한다.
- 210M encoder classifier가 36개 candidate model, 15개 task에서 large LLM judge와 경쟁하는 Pareto point를 만든다.
- Synthetic judge label을 human subset으로 검증하고 OOD model family, cross-format, calibration을 함께 분석한다.
- Evaluation pipeline의 비용, latency, reproducibility를 실제 engineering 문제로 다룬다.

LLM benchmark는 보통 다음 pipeline을 가진다.

1. Question을 model에 준다.
2. Model이 reasoning과 final answer를 생성한다.
3. Parser가 final answer를 추출한다.
4. Candidate와 reference를 비교한다.
5. Correct or incorrect label을 만든다.

많은 benchmark는 3번과 4번을 regex and lexical rule에 맡긴다.

예를 들어 multiple-choice task는 `Final answer: A`를 찾고, math task는 boxed expression이나 마지막 숫자를 parse한다. Question answering은 exact match or token F1을 사용한다.

이 방식은 저렴하고 deterministic하다. 하지만 model이 correct answer를 다른 형식으로 쓰면 wrong으로 처리할 수 있다.

- `A` 대신 `The correct option is A.`
- `0.5` 대신 `1/2`
- `12` 대신 `There are twelve items.`
- Equivalent algebraic expression
- Reference보다 더 구체적인 span
- Correct answer plus explanatory text
- Final-answer marker omission

반대로 lexical overlap이 높다고 semantic correctness가 보장되지는 않는다. Candidate가 reference 단어를 반복하면서 negation을 바꿀 수 있다.

LLM-as-a-Judge는 이 문제를 자연어 understanding으로 해결한다. 하지만 large model을 evaluator로 쓰면 다음 cost가 생긴다.

- Evaluation마다 autoregressive generation
- Prompt sensitivity
- Reasoning token cost
- Model version drift
- API dependency
- Reproducibility issue
- Candidate model보다 judge가 더 비쌀 수 있음

BERT-as-a-Judge는 reference-based objective task에서는 generative judge가 과한 solution일 수 있다고 본다.

Question, candidate, reference의 semantic relation을 판별하는 것은 binary classification이다. Small bidirectional encoder가 이 relation을 supervised data로 학습하면 large generative judge 없이 correctness를 판단할 수 있다.

논문은 이를 36개 model과 15개 benchmark에서 검증한다. Candidate model size는 135M부터 70B까지 넓다.

핵심 message는 단순하다.

> Evaluation format robustness를 위해 generative reasoning model이 반드시 필요한 것은 아니다.

# 1. Problem Setting

## 1-1. Problem definition

Reference-based evaluation input을 다음 triplet으로 정의하자.

$$
x
=
(q, c, r)
$$

- $q$: Question or task input
- $c$: Candidate model answer
- $r$: Reference answer

Judge는 binary correctness probability를 출력한다.

$$
p_\theta
=
P_\theta
\left(
y=1
\mid
q,c,r
\right)
$$

Threshold $\delta$를 사용하면 prediction은 다음과 같다.

$$
\hat{y}
=
\mathbb{1}
\left[
p_\theta \geq \delta
\right]
$$

Evaluation target은 human or trusted correctness label $y$와의 agreement다.

논문이 다루는 task는 세 category다.

### 1) Multiple choice

- MMLU
- MMLU-Pro
- TruthfulQA
- ARC-Easy
- ARC-Challenge
- GPQA

### 2) Context extraction and question answering

- SQuAD-v2
- HotpotQA
- DROP
- CoQA

### 3) Open-form math

- GSM8K
- MATH
- ASDiv
- AIME 2024
- AIME 2025

Candidate answer는 zero-shot greedy decoding으로 생성한다. Maximum generation은 2048 tokens이며, prompt는 `Final answer: [answer]` 형식을 요청한다.

그런데 model이 instruction을 완전히 따르지 않으면 regex parser가 fail한다. 이는 reasoning failure가 아니라 format failure다.

논문의 problem setting은 다음과 같다.

| Question | BERT-as-a-Judge의 답 |
| --- | --- |
| Input | Question, candidate, reference |
| Output | Correctness probability |
| Model | 210M EuroBERT encoder |
| Objective | Binary classification |
| Training labels | Synthetic LLM judge labels |
| Human validation | 11 annotators, 3212 labels |
| Evaluation scope | 36 models, 15 tasks |
| Baselines | Regex, lexical metrics, math verifier, LLM judges |
| Main axis | Judge accuracy vs compute |
| Calibration | Temperature scaling |

## 1-2. Why previous approaches are insufficient

### 1) Regex는 model capability와 instruction following을 섞는다

Model이 문제를 풀었지만 final answer marker를 놓치면 wrong이다. Benchmark score는 reasoning and formatting의 joint metric이 된다.

논문 setup에서 open-form math의 format failure는 일부 model에서 매우 크다.

- Llama3 70B는 60%를 넘는 parse failure가 보고되는 setting이 있다.
- Qwen3 32B는 약 20% 수준의 format failure가 나타난다.

이 number는 model, prompt, parser에 특화된 결과다. 그러나 parsing failure가 leaderboard를 실질적으로 바꿀 수 있다는 점은 분명하다.

### 2) Exact match는 equivalent expression을 놓친다

Math에서 다음 answer는 semantic하게 같을 수 있다.

$$
\frac{1}{2}
=
0.5
=
50\%
$$

String exact match는 이를 다른 answer로 본다. Symbolic math verifier는 일부 equivalence를 처리하지만 natural-language answer와 mixed formatting에는 한계가 있다.

### 3) ROUGE-L and token overlap은 correctness를 직접 측정하지 않는다

Context extraction에서 candidate가 reference보다 길거나 paraphrase하면 lexical overlap이 낮아질 수 있다.

반대로 reference term을 포함하면서 wrong relation을 말하면 overlap은 높을 수 있다.

### 4) LLM judge는 expensive하고 nondeterministic할 수 있다

Autoregressive judge는 answer를 생성해야 한다. CoT prompt를 붙이면 token cost와 latency가 더 커진다.

Prompt wording, sampling, system policy, model update가 label을 바꿀 수 있다. Closed API judge는 long-term reproducibility가 약하다.

### 5) Task-specific verifier는 maintenance cost가 크다

Benchmark마다 custom parser를 만들면 precision은 높일 수 있다. 하지만 new task와 new answer format이 등장할 때마다 rule을 추가해야 한다.

BERT-as-a-Judge는 one classifier가 multiple choice, extraction, math를 함께 처리하는 general reference-based judge를 목표로 한다.

# 2. Core Idea

## 2-1. Main contribution

### 1) Question-candidate-reference encoder

Input은 triplet을 separator token으로 연결한다.

```text
[CLS] question [SEP] candidate [SEP] reference [SEP]
```

EuroBERT encoder가 contextual representation을 만들고 classification head가 correctness logit을 출력한다.

$$
z
=
\mathrm{Encoder}_\theta(q,c,r)_{\mathrm{CLS}}
$$

$$
p_\theta
=
\sigma
\left(
w^\top z+b
\right)
$$

Training loss는 binary cross-entropy다.

$$
\mathcal{L}_{\mathrm{BCE}}
=
-
y \log p_\theta
-
(1-y)\log(1-p_\theta)
$$

Bidirectional attention은 candidate and reference를 token-level로 비교하면서 question constraint를 함께 읽는다.

### 2) Large synthetic label corpus

Manual label을 million scale로 만드는 것은 비싸다. 논문은 Nemotron-Super-v1.5를 labeler로 사용한다.

Labeler는 question, candidate, reference를 보고 correct or incorrect를 판단한다.

Synthetic training set은 약 1M labels다. Training task는 explicit train split이 있는 benchmark에서 구성한다.

- MMLU
- ARC-Easy
- ARC-Challenge
- SQuAD-v2
- HotpotQA
- GSM8K
- MATH

Category and candidate model을 balance해 특정 task or model family에 과적합되는 것을 줄인다.

### 3) Human label validation

Synthetic label reliability를 확인하기 위해 11 human annotators가 3212 labels를 검토한다.

Average agreement는 97.5%다.

이 result는 synthetic label이 high quality임을 지지하지만 완전한 gold standard는 아니다.

- Annotator population
- Ambiguous item exclusion
- Category별 agreement
- Labeler and human shared bias

를 함께 봐야 한다.

### 4) Generalization across unseen tasks

Judge는 training에 없는 task에서도 평가된다.

- GPQA
- MMLU-Pro
- TruthfulQA
- CoQA
- DROP
- ASDiv
- AIME 2024
- AIME 2025

이 setup은 task-specific answer parser를 학습한 것이 아니라 semantic correctness pattern을 transfer하는지 본다.

### 5) Generalization across unseen model families

Candidate model family를 training label에서 제외한 뒤 해당 family answer를 평가한다.

Performance degradation이 작아, judge가 특정 model의 response style만 외우지 않았다는 evidence를 제공한다.

### 6) Calibration

Raw classifier probability는 overconfident할 수 있다. Temperature scaling을 사용한다.

$$
p_\tau
=
\sigma
\left(
\frac{z}{\tau}
\right)
$$

논문은 $\tau=1.75$를 사용해 near-perfect calibration을 보고한다.

Calibrated probability는 binary label뿐 아니라 uncertain example sampling and human review routing에 유용하다.

## 2-2. Design intuition

### 1) Reference-based correctness는 generation보다 classification에 가깝다

Reference가 주어지고 answer가 objective하다면 evaluator가 new solution을 생성할 필요가 없다. Candidate and reference relation을 판별하면 된다.

### 2) Question은 answer equivalence의 context를 제공한다

Candidate `42`와 reference `42`가 같아도 question이 multiple-choice label을 요구했다면 format semantics가 달라질 수 있다.

반대로 reference가 short span이고 candidate가 sentence인 경우 question을 보면 same meaning인지 판단하기 쉽다.

Ablation에서 question removal은 context extraction category를 89.2에서 84.2로 낮춘다. Question이 특히 QA context에서 중요하다.

### 3) Small judge는 fixed evaluation protocol에 적합하다

Encoder forward pass는 autoregressive generation보다 빠르고 deterministic하다.

- Fixed checkpoint
- Fixed tokenizer
- Fixed threshold
- Batched inference
- No reasoning output
- Stable versioning

Leaderboard and large-scale model sweep에 유리하다.

### 4) Synthetic labeler는 teacher, BERT judge는 compiled evaluator다

Large LLM이 expensive semantic rule을 training data에 distill한다. 이후 small encoder가 repeated evaluation을 수행한다.

이 구조는 knowledge distillation과 유사하다.

- Teacher cost: One-time labeling
- Student cost: Repeated cheap inference
- Benefit: Evaluation consistency and throughput

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | Reference-based LLM answer correctness evaluation |
| Student | EuroBERT 210M |
| Input | Question, candidate answer, reference answer |
| Output | Binary correctness probability |
| Training labels | Nemotron-Super-v1.5 synthetic judgments |
| Training size | About 1M |
| Human audit | 3212 labels, 11 annotators |
| Task coverage | Multiple choice, context extraction, open math |
| Candidate models | 36 models, 135M-70B |
| Deployment benefit | Batched deterministic encoder inference |

## 3-2. Module breakdown

### 1) Candidate generation

각 benchmark item에 대해 36개 model이 answer를 생성한다.

Generation setup:

- Zero-shot
- Greedy decoding
- Up to 2048 tokens
- Explicit final-answer instruction

이 step에서 response style diversity가 생긴다.

- Short label
- Full sentence
- Chain-of-thought
- Boxed math
- Markdown
- Refusal
- Truncated answer
- Multiple candidate answer

Judge는 이 variation을 처리해야 한다.

### 2) Synthetic labeling prompt

Teacher judge는 question, candidate, reference를 입력받고 binary correctness를 출력한다.

Quality control에는 다음이 중요하다.

- Label schema를 strict하게 만들기
- Teacher reasoning을 label과 분리하기
- Ambiguous reference 처리
- Multiple valid answer 인정
- Unanswerable question handling
- Numerical tolerance
- Unit and sign
- Partial credit exclusion

논문은 objectively verifiable final answer를 대상으로 한다. Open-ended style quality는 scope 밖이다.

### 3) Data balancing

Synthetic corpus가 model family and task distribution에 치우치면 judge가 format prior를 배울 수 있다.

예를 들어 stronger model answer가 항상 longer response이고 correct label이 많다면 length를 shortcut으로 사용할 수 있다.

논문은 task category, candidate model, label balance를 고려해 sampling한다.

### 4) Encoder fine-tuning

Training recipe는 다음과 같다.

- Base encoder: EuroBERT 210M
- Epochs: 1
- Objective: BCE
- Learning rate: $2 \times 10^{-5}$
- Warmup: 5%
- Schedule: Linear decay
- Effective batch size: 32
- Hardware: 8 x MI250X
- Total compute: 약 20 GPU hours

One epoch만으로 strong result를 얻는다. Training cost보다 synthetic label generation and candidate answer collection이 더 큰 part일 수 있다.

### 5) Baselines

#### Lexical and task-specific

- Regex parser
- Exact match
- ROUGE-L
- Math-Verify
- Hybrid regex plus semantic fallback

#### LLM judges

- Qwen3 variants
- Gemma3 variants
- Fine-tuned Qwen3-0.6B
- JudgeLM
- CoT and direct answer prompting

#### Learned metric

- BLEURT

BERT judge의 advantage는 accuracy와 compute Pareto다.

### 6) Cross-format test

Judge를 free-form candidate on training and different answer formatting on test로 평가한다.

Free-form-trained judge가 strict-format-trained judge보다 robust하다. Training data에 natural response variation이 있어야 parser-like shortcut을 덜 배운다.

### 7) Confidence calibration

Validation logits에 scalar temperature를 fit한다. Calibration curve and expected calibration error를 본다.

Probability가 calibrated하면 다음 policy를 만들 수 있다.

- High confidence: Automatic score
- Medium confidence: Second judge
- Low confidence: Human review

# 4. Training / Data / Recipe

## 4-1. Candidate model pool

36개 candidate model은 135M부터 70B까지 포함한다. Different family와 instruction style을 섞는다.

Judge training에서 model diversity는 중요하다. Single family answer만 보면 해당 family의 punctuation, final-answer phrase, verbosity를 shortcut으로 학습할 수 있다.

## 4-2. Training and test task split

Training label source:

| Category | Training tasks |
| --- | --- |
| Multiple choice | MMLU, ARC-Easy, ARC-Challenge |
| Context extraction | SQuAD-v2, HotpotQA |
| Math | GSM8K, MATH |

Evaluation에는 training task와 unseen task를 함께 포함한다.

| Category | Additional evaluation tasks |
| --- | --- |
| Multiple choice | MMLU-Pro, TruthfulQA, GPQA |
| Context extraction | DROP, CoQA |
| Math | ASDiv, AIME 2024, AIME 2025 |

이 분할은 category 내부의 task transfer를 평가한다.

## 4-3. Label scale ablation

Training sample 수를 줄이며 judge accuracy를 본다.

- Multiple choice and math는 약 100K labels에서도 빠르게 saturation한다.
- Context extraction은 더 많은 data에서 계속 benefit을 얻는다.
- 100K setting은 약 2 GPU hours로 training할 수 있다.

Context extraction은 paraphrase, span boundary, unanswerable case가 다양해 sample complexity가 높다.

## 4-4. Hybrid evaluation

Regex parser가 성공하면 lexical score를 사용하고 parse failure에만 BERT judge를 호출하는 hybrid를 고려할 수 있다.

Parsing failure가 20%라면 semantic judge call은 전체의 20%다. Compute는 standalone judge보다 약 5x 줄 수 있다.

그러나 논문에서 hybrid accuracy는 standalone BERT judge보다 낮다. Regex가 confident하게 wrong parsing을 한 case는 fallback에 도달하지 않기 때문이다.

따라서 hybrid는 cost-sensitive deployment option이지 accuracy-optimal setting이 아니다.

## 4-5. Cross-format recipe

Judge training data는 final-answer format을 지나치게 normalize하지 않는 편이 좋다.

- Reasoning 포함
- Full sentence
- Bare label
- Mathematical expression
- Markdown wrapper
- Extra explanation

를 모두 포함해야 actual model response에 robust하다.

## 4-6. Engineering notes

### 1) Reference normalization

Reference answer가 multiple valid form을 가지면 canonicalization or multiple references가 필요하다.

### 2) Input length

Candidate가 2048-token reasoning을 포함하면 question, candidate, reference가 encoder max length를 넘을 수 있다. Truncation policy가 final answer를 보존하도록 해야 한다.

### 3) Threshold versioning

Default 0.5 threshold가 모든 domain에서 optimal하지 않을 수 있다. Domain-specific calibration set을 두되 leaderboard fairness를 위해 fixed threshold를 versioning해야 한다.

### 4) Adversarial candidate

Candidate model이 judge prompt injection, reference repetition, confidence phrase를 사용할 수 있다. Encoder judge도 adversarially robust하다고 자동으로 가정하면 안 된다.

### 5) Label provenance

Synthetic labeler checkpoint, prompt, decoding, timestamp를 저장해야 future reproduction이 가능하다.

# 5. Evaluation

## 5-1. Main results

BERT-as-a-Judge는 task-level correctness agreement에서 다음 result를 보고한다.

| Task | Judge accuracy |
| --- | ---: |
| ARC-Challenge | 99.4 |
| ARC-Easy | 99.7 |
| MMLU | 98.5 |
| GPQA | 93.5 |
| MMLU-Pro | 96.4 |
| TruthfulQA | 98.6 |
| HotpotQA | 90.8 |
| SQuAD-v2 | 89.3 |
| CoQA | 88.2 |
| DROP | 88.2 |
| GSM8K | 98.8 |
| MATH | 93.7 |
| AIME 2024 | 89.8 |
| AIME 2025 | 92.5 |
| ASDiv | 95.1 |

Task마다 ambiguity and answer style가 다르다. Multiple choice는 거의 99%에 가깝고, context extraction과 hard math는 lower but strong accuracy를 보인다.

### 1) Regex and lexical metrics보다 robust하다

Format variation이 큰 math and QA에서 regex failure를 크게 줄인다.

Model leaderboard ranking도 lexical evaluator보다 human label과 더 잘 맞는다.

### 2) Large LLM judge와 경쟁한다

BERT judge는 최대 약 70x larger LLM judge와 비슷하거나 더 높은 accuracy를 보이는 setting이 있다.

특히 reference가 제공되는 objective task에서는 generative CoT가 필요하지 않다. LLM judge에 CoT를 추가해도 accuracy improvement가 거의 없고 compute만 늘어나는 result가 보고된다.

### 3) Judge scale saturation

LLM judge size를 키울 때 performance는 약 10B 부근에서 saturation하는 경향이 있다. 70B가 binary reference comparison에서 proportional gain을 주지 않는다.

### 4) OOD model family robustness

Training label에서 candidate model family를 제외하고 평가해도 degradation이 작다.

이 result는 judge가 model-specific formatting만 memorization한 것이 아니라 question-candidate-reference relation을 배운다는 evidence다.

### 5) Cross-format robustness

Free-form response로 trained judge는 strict final-answer and alternative format에서도 robust하다.

반대로 normalized answer만 학습하면 real response의 explanation and formatting에 약해진다.

### 6) Calibration

Temperature $\tau=1.75$를 적용하면 confidence가 empirical correctness와 잘 맞는다.

Operationally는 low-confidence case를 human review로 보내는 데 useful하다.

## 5-2. What really matters in the experiments

### 1) Judge accuracy와 benchmark score difference를 구분해야 한다

Judge accuracy가 95%라고 model benchmark score도 5 points 틀린다는 뜻은 아니다. Error가 correct and incorrect example에 어떻게 분포하는지에 따라 aggregate score bias가 달라진다.

### 2) Reference-based scope가 performance의 전제다

BERT judge는 reference answer가 있는 objective task에서 strong하다. Summarization quality, translation fluency, open-ended code quality는 다른 evaluation problem이다.

### 3) Synthetic teacher quality가 ceiling을 만든다

Student가 teacher label로 학습되기 때문에 teacher systematic bias를 상속할 수 있다. Human 97.5% agreement는 good sign이지만 category-specific error analysis가 필요하다.

### 4) Question inclusion이 shortcut을 줄인다

Candidate and reference string만 비교하면 semantic equivalence classifier에 머문다. Question이 있어야 requirement, unit, entity, answer scope를 확인할 수 있다.

### 5) Compute Pareto가 practical value다

210M encoder는 batch inference가 가능하다. 36 models x 15 tasks x thousands of examples를 반복 평가할 때 cost difference가 커진다.

# 6. Limitations

1. English and objective reference-based tasks에 집중한다.
   - Multilingual preliminary result가 있어도 main evaluation은 English다.
   - Open-ended summarization, translation, creative writing, instruction following은 scope 밖이다.

2. Multimodal answer를 다루지 않는다.
   - Image-grounded QA, chart reasoning, audio response 평가에는 modality input을 직접 볼 수 없다.

3. Synthetic label bias를 상속한다.
   - Nemotron-Super-v1.5가 특정 reasoning style나 reference interpretation에서 systematic error를 만들 수 있다.
   - Human validation subset이 모든 category를 완전히 대표하지 않을 수 있다.

4. Binary correctness만 제공한다.
   - Partial credit, error type, reasoning quality, citation faithfulness를 분리하지 않는다.
   - Math proof나 multi-part answer에는 richer rubric이 필요하다.

5. Reference answer가 불완전하면 judge도 흔들린다.
   - Candidate가 reference와 다르지만 valid할 수 있다.
   - Multiple reference or answer set representation이 필요하다.

6. Candidate-reference leakage shortcut이 가능하다.
   - Lexical overlap, length, final phrase를 feature로 사용할 수 있다.
   - Counterfactual and adversarial test가 필요하다.

7. Prompt injection and adversarial gaming을 충분히 다루지 않는다.
   - Candidate가 reference를 copy하거나 evaluator를 속이는 instruction을 포함할 수 있다.
   - Encoder는 generative prompt injection에는 덜 민감하지만 text pattern attack은 가능하다.

8. Input truncation이 long reasoning answer를 왜곡할 수 있다.
   - Final answer가 tail에 있는데 head truncation을 하면 judge가 놓친다.
   - Segment-aware truncation이 필요하다.

9. Threshold and calibration이 domain shift에 민감하다.
   - $\tau=1.75$가 new domain and model style에서도 최적이라는 guarantee는 없다.
   - Deployment calibration set이 필요할 수 있다.

10. Benchmark contamination 가능성이 있다.
    - Teacher, student base encoder, candidate model이 benchmark data를 pretraining에서 봤을 수 있다.
    - Judge accuracy와 task-solving contamination은 별개의 문제지만 evaluation ecosystem 전체에 영향을 준다.

# 7. My Take

## 7-1. Why this matters for my work

BERT-as-a-Judge의 가장 큰 의미는 evaluation을 model capability problem이 아니라 compiler and test infrastructure problem으로 본다는 데 있다.

많은 internal benchmark에서 answer parser가 hidden bottleneck이다.

- JSON field가 조금 다르면 fail
- Number formatting이 달라 fail
- Extra explanation 때문에 fail
- Korean and English 표현이 달라 fail
- Correct span이 reference보다 길어 fail

이런 error를 줄이려고 large LLM judge를 붙이면 cost와 reproducibility 문제가 생긴다.

Question-candidate-reference classifier는 objective evaluation에서 좋은 middle ground다.

- Regex보다 semantic
- LLM judge보다 cheap
- Fixed checkpoint라 reproducible
- Confidence calibration 가능
- Batch processing 가능

특히 model iteration이 빠르고 evaluation set이 큰 팀에서 reusable infrastructure가 될 수 있다.

## 7-2. Reuse potential

### 1) Internal document QA judge

Input을 다음처럼 구성한다.

- Question
- Model answer
- Gold answer
- Optional evidence context

Binary correctness뿐 아니라 supported, unsupported, partially correct class로 확장할 수 있다.

### 2) Korean and bilingual judge

Korean document task에는 multilingual encoder 또는 Korean-specialized BERT를 사용할 수 있다.

Training data에는 다음 variation을 넣어야 한다.

- Korean numeral and Arabic numeral
- Spacing variation
- English acronym
- Date and currency format
- OCR typo
- Table cell answer
- Multiple valid honorific or suffix form

### 3) Extraction evaluation

Schema extraction에서 field별 candidate and reference를 judge한다.

- Semantic equivalence
- Unit conversion
- Date normalization
- Null vs missing
- Multiple values
- OCR corruption tolerance

Strict exact match와 semantic judge를 같이 report하면 model error and format error를 분리할 수 있다.

### 4) Cascaded evaluator

Cost and risk에 따라 three-stage evaluator를 만들 수 있다.

1. Deterministic exact verifier
2. BERT judge
3. Large LLM or human review

Confidence and disagreement로 routing한다.

### 5) Error taxonomy head

Binary label 외에 다음 class를 추가할 수 있다.

- Correct
- Wrong entity
- Wrong number
- Missing constraint
- Partial answer
- Unsupported
- Format-only mismatch
- Reference ambiguity

이렇게 하면 model development에 더 useful한 feedback을 얻는다.

### 6) Adversarial robustness test

Judge를 다음 candidate로 stress test한다.

- Reference answer repeated inside a wrong sentence
- Negation flip
- Correct number with wrong unit
- Multiple answers including the reference
- Prompt injection
- Long irrelevant prefix
- Correct answer beyond truncation boundary
- Paraphrase with low lexical overlap

## 7-3. Follow-up papers

- BLEURT
- BERTScore
- JudgeLM
- Prometheus
- G-Eval
- LLM-as-a-Judge evaluation studies
- Math-Verify
- RewardBench
- Critique and verification models
- Calibrated selective prediction

# 8. Summary

- BERT-as-a-Judge는 question, candidate, reference triplet을 입력받는 210M EuroBERT binary evaluator다.
- 약 1M synthetic labels로 1 epoch 학습하고, 11명 human annotator의 3212 labels에서 97.5% agreement를 확인한다.
- 36개 candidate model과 15개 multiple-choice, QA, math task에서 regex and lexical metric보다 format-robust하다.
- Large LLM judge와 비슷하거나 더 높은 accuracy를 훨씬 낮은 compute로 제공하며, CoT judge의 추가 benefit은 작다.
- OOD model family, unseen task, cross-format, calibration analysis에서 robust behavior를 보인다.
- English objective reference task라는 scope, synthetic bias, binary label, adversarial candidate, truncation and calibration shift는 주요 한계다.
