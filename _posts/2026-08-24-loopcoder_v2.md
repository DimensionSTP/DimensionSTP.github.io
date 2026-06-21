---
layout: single
title: "LoopCoder-v2: Only Loop Once for Efficient Test-Time Computation Scaling Review"
categories: Study-concept
tag: [LoopCoder-v2, LLM, ParallelLoopTransformer, TestTimeCompute, CodingAgent]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/abs/2606.18023)

[Code link](https://github.com/CSJianYang/LoopCoder)

[Model link](https://huggingface.co/Multilingual-Multimodal-NLP/LoopCoder-V2)

LoopCoder-v2를 "loop를 더 많이 돌려 7B coding model을 강화한 논문" 정도로 읽으면 핵심을 놓치기 쉽다. 이 논문의 headline은 오히려 반대다. Parallel Loop Transformer, 이하 PLT에서는 loop를 하나 더 추가한 $R=2$가 가장 좋았고, $R=3$과 $R=4$는 여러 benchmark에서 크게 후퇴했다.

여기서 제목의 "Only Loop Once"는 total loop count가 1이라는 뜻이 아니다. $R=1$은 shared block을 한 번 통과하는 non-looped baseline이고, $R=2$는 그 뒤에 shared block을 한 번 더 적용하는 설정이다. 즉 이 논문이 권하는 operating point는 **기본 pass 이후 한 번만 추가 refinement를 수행하는 total two-loop model**이다.

이 결과가 흥미로운 이유는 단순한 hyperparameter sweep이 아니기 때문이다. Looped Transformer는 parameter 수를 늘리지 않고 latent computation depth를 늘릴 수 있다. 하지만 standard loop는 loop count가 늘수록 latency와 KV cache가 같이 커진다. PLT는 Cross-Loop Parallelism, 이하 CLP와 shared-KV Gated Sliding-Window Attention, 이하 G-SWA를 통해 이 비용을 크게 낮춘다. 문제는 이 효율화가 공짜가 아니라는 점이다. CLP는 이전 loop의 같은 token state가 아니라 한 칸 앞 token의 state를 가져오기 때문에 매 loop boundary마다 positional mismatch를 만든다.

LoopCoder-v2는 이 상황을 gain-cost trade-off로 읽는다.

- Gain: 추가 loop가 hidden state, attention routing, output distribution을 얼마나 유의미하게 바꾸는가.
- Cost: CLP offset이 token position 사이에 얼마나 큰 representational mismatch를 만드는가.
- Saturation: marginal gain이 줄어든 뒤에도 offset cost가 유지되면 어느 loop부터 손해가 되는가.

저자들은 7B PLT coder를 $R \in \{1,2,3,4\}$로 각각 처음부터 학습하고, 동일한 data, instruction tuning, evaluation protocol 아래 비교한다. $R=2$는 SWE-bench Verified를 43.0에서 64.4로, Multi-SWE를 14.0에서 31.0으로 올렸다. 반면 $R=3$은 SWE-bench Verified 27.6, $R=4$는 22.4로 baseline 아래까지 떨어졌다.

> 한 줄 요약: LoopCoder-v2는 PLT의 loop count가 단조롭게 성능을 높이지 않으며, 두 번째 loop에서 productive refinement가 집중되고 이후에는 diminishing gain보다 CLP positional mismatch가 커진다는 점을 hidden state, attention, output distribution 진단으로 설명한 7B coding model 연구다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Test-time compute를 더 쓰면 항상 좋아진다는 직관을 실제 large-scale pretraining으로 반박한다.
- Efficient architecture가 줄인 latency와 memory 대신 어떤 information cost를 새로 만드는지 보여준다.
- Loop count를 benchmark sweep만으로 고르지 않고 activation-level diagnostic으로 판단하는 방법을 제안한다.
- 18T token pretraining과 6M instruction example 아래에서 matched comparison을 수행해 loop-count effect를 비교적 깔끔하게 분리한다.
- Explicit chain-of-thought와 latent loop computation이 대체 관계가 아니라 상호 보완적일 수 있음을 보여준다.

이 글에서는 LoopCoder-v2를 coding leaderboard model보다, **recurrence를 병렬화했을 때 생기는 architectural tax와 useful refinement의 경계를 찾는 논문**으로 읽는다.

# 1. Problem Setting

## 1-1. Problem definition

Looped Transformer의 기본 아이디어는 parameter sharing을 depth 방향으로 확장하는 것이다. 일반 Transformer가 layer마다 다른 parameter를 사용한다면, looped model은 하나의 shared block $f_{\theta}$를 반복 적용한다.

$$
\mathbf{h}^{(0)} = \mathrm{Embed}(x)
$$

$$
\mathbf{h}^{(r)} = f_{\theta}\left(\mathbf{h}^{(r-1)}\right), \quad r = 1, \ldots, R
$$

$$
\mathrm{logits} = \mathrm{Head}\left(\mathbf{h}^{(R)}\right)
$$

Shared block 안에 $L$개 layer가 있다면 effective depth는 $R L$이지만 parameter count는 한 block의 크기로 유지된다. 이 구조는 generated reasoning token을 늘리지 않고 latent space에서 추가 computation을 사용할 수 있다는 점에서 test-time compute scaling과 잘 맞는다.

하지만 standard sequential loop는 두 가지 비용을 갖는다.

$$
\mathrm{Latency}_{seq} = O(R C_{block})
$$

$$
\mathrm{KV}_{seq} = O(R L S d)
$$

여기서 $C_{block}$은 shared block 한 번의 실행 cost, $S$는 sequence length, $d$는 hidden width다.

- Loop $r$은 loop $r-1$이 끝나야 시작할 수 있다.
- 각 loop와 layer의 KV state를 보관하면 cache가 $R$에 비례해 증가한다.
- Effective depth는 늘지만 wall-clock latency와 memory도 같이 늘어난다.
- Coding agent나 repository task처럼 context와 output이 긴 setting에서는 이 비용이 특히 크다.

PLT는 loop의 sequential dependency와 KV duplication을 줄여 이 문제를 완화한다. 그러면 새로운 질문이 생긴다.

> Loop를 싸게 만들었다고 해서 loop를 많이 돌리는 것이 항상 좋은가?

이 논문의 problem setting은 바로 이 질문이다. PLT에서 loop count $R$은 단순한 compute knob이 아니라, refinement gain과 offset mismatch를 동시에 바꾸는 architecture choice다.

## 1-2. Why previous approaches are insufficient

### 1) Standard loop는 deployment cost가 너무 직접적이다

Sequential recurrence에서는 loop를 하나 추가할 때마다 shared block을 다시 실행해야 한다. Theoretical parameter efficiency가 좋아도 real-time serving에서는 latency가 그대로 늘어난다. KV cache까지 loop별로 보관하면 long context에서 memory benefit도 약해진다.

따라서 "더 많은 latent reasoning"이라는 아이디어와 "실제로 serve 가능한 inference" 사이에 gap이 생긴다.

### 2) PLT는 cost를 줄이지만 information flow를 바꾼다

PLT는 CLP를 통해 loop를 token 축과 함께 pipeline처럼 병렬화한다. 이를 위해 token $i$의 다음 loop가 token $i$의 이전 loop state를 직접 기다리지 않고, token $i-1$의 state를 사용하게 만든다.

이 shift는 병렬성에는 유리하지만, token position이 정확히 일치하지 않는 state를 주입한다. 즉 system cost를 줄이는 대신 representation path가 달라진다.

### 3) Loop count를 benchmark만으로 고르면 원인을 알 수 없다

$R=2$, $R=3$, $R=4$를 모두 학습하고 benchmark를 돌리면 best setting은 찾을 수 있다. 하지만 다음 질문에는 답하기 어렵다.

- 어느 loop가 실제로 새로운 computation을 하는가.
- Hidden state가 refinement되는가, oscillation하는가.
- Attention head가 새로운 routing을 만드는가, 기존 routing을 반복하는가.
- Output distribution이 계속 유의미하게 바뀌는가.
- Performance drop이 optimization instability인지 CLP mismatch인지 어떻게 구분하는가.

LoopCoder-v2는 loop-wise interpretability metric을 사용해 이 부분을 설명하려 한다.

### 4) "More test-time compute is better"라는 가정이 너무 단순하다

Chain-of-thought, self-consistency, search, recurrent depth는 모두 test-time computation을 늘리는 방법이다. 그러나 additional compute가 useful work인지 확인하지 않으면, model은 다음 중 하나를 할 수 있다.

- 같은 representation을 다시 읽는다.
- Update direction을 뒤집으며 oscillation한다.
- Token representation diversity를 줄인다.
- Frozen global context에 계속 의존한다.
- Output confidence만 높이고 correctness는 개선하지 않는다.

이 논문은 compute quantity보다 **marginal compute quality**를 측정해야 한다고 본다.

# 2. Core Idea

## 2-1. Main contribution

LoopCoder-v2의 핵심 기여는 세 가지다.

1. **PLT loop-count selection을 gain-cost trade-off로 정식화**
   - Extra loop는 latent refinement gain을 줄 수 있다.
   - 동시에 CLP offset은 매 boundary에서 positional mismatch cost를 만든다.
   - Best loop count는 두 항의 balance로 결정된다.

2. **Per-loop diagnostic lens 제안**
   - Hidden-state dynamics
   - Attention evolution
   - Output-distribution shift
   - Intrinsic offset cost

3. **Large-scale matched loop-count study**
   - 7B dense model
   - 18T pretraining tokens
   - $R=1,2,3,4$ variants
   - 동일한 6M instruction-tuning examples
   - Code generation, reasoning, software engineering, tool use benchmark

중요한 점은 PLT 자체가 이 논문의 최초 제안은 아니라는 것이다. CLP와 G-SWA는 앞선 Parallel Loop Transformer 연구에서 제안되었다. LoopCoder-v2의 주된 contribution은 이 architecture에서 loop count가 왜 $R=2$에서 포화되는지 분석하고, 이를 coding model scale에서 검증한 것이다.

## 2-2. Design intuition

이 논문의 직관은 간단하다.

$$
\mathrm{NetValue}^{(r)} =
\mathrm{RefinementGain}^{(r)}
-
\mathrm{OffsetCost}^{(r)}
$$

초기 extra loop는 representation을 크게 바꿀 수 있다. 하지만 shared block을 반복할수록 새로운 정보는 줄어들고, update가 기존 computation과 중복될 수 있다. 반면 CLP의 one-token shift는 매 boundary마다 계속 발생한다.

저자들의 관찰은 다음과 같다.

- Loop 1은 embedding을 contextual representation으로 바꾸는 가장 큰 transformation이다.
- Refinement loop 중에서는 loop 2가 가장 큰 hidden-state change, attention re-routing, output shift를 만든다.
- Effective rank도 loop 2에서 가장 높다.
- Loop 3 이후 output KL과 attention KL은 급격히 줄어든다.
- Hidden-state update direction은 음의 cosine을 보이며 oscillatory해진다.
- Adjacent-token mismatch로 정의한 offset cost는 loop가 깊어져도 크게 줄지 않는다.

따라서 loop 2 이후에는 다음 상태가 된다.

$$
\mathrm{Gain}^{(r)} \downarrow
\quad \mathrm{while} \quad
\mathrm{Cost}^{(r)} \approx \mathrm{constant}
$$

이때 추가 loop는 useful refinement보다 fixed architectural tax를 더 많이 지불하게 된다.

# 3. Architecture / Method

## 3-1. Overview

| Item | Description |
| --- | --- |
| Goal | PLT에서 useful loop count와 saturation point를 진단 |
| Base model | 약 7B dense decoder-only Transformer |
| Shared block | 14 layers |
| Loop variants | $R=1,2,3,4$ |
| Main efficiency module | CLP plus shared-KV G-SWA |
| Global context | Loop 1의 frozen full-context KV cache |
| Local context | Current loop의 sliding-window attention, $w=64$ |
| Main diagnosis | Hidden state, attention, output distribution, offset cost |
| Main finding | $R=2$가 best operating point, $R \ge 3$은 대체로 regression |

## 3-2. Model configuration

논문에 제시된 base configuration은 다음과 같다.

| Hyperparameter | Value |
| --- | --- |
| Shared layers | 14 |
| Hidden size | 5120 |
| Attention heads | 40 |
| KV groups | 8 |
| Head dimension | 128 |
| FFN intermediate size | 27648 |
| Activation | SwiGLU |
| Normalization | RMSNorm, epsilon $10^{-5}$ |
| Position embedding | RoPE, base $5 \times 10^{5}$ |
| Vocabulary size | 76800 |
| Precision | bf16 |
| Total parameters | 약 7B |
| PLT local window | 64 |

Model card는 released checkpoint를 `plt_num_loops=2`로 명시한다. Hugging Face UI에는 8B params로 표시되는 부분이 있어, 실제 parameter count 표기는 publish 전에 config와 weight count를 다시 확인할 필요가 있다.

## 3-3. PLT module breakdown

### 1) Loop 1 full attention

첫 loop는 standard full-context attention을 수행한다.

$$
\mathbf{h}^{(1)} = f_{\theta}\left(\mathbf{h}^{(0)}\right)
$$

이때 생성된 KV cache를 global shared cache로 저장한다.

$$
K_{share}, V_{share} = \mathrm{KV}\left(\mathbf{h}^{(1)}\right)
$$

후속 loop는 이 global cache를 다시 만들지 않는다. 따라서 loop 수가 늘어도 global KV footprint가 반복 저장되지 않는다.

이 design은 memory를 줄이지만 중요한 ceiling도 만든다. Later loop가 보는 global information은 loop 1에서 만들어진 representation에 묶인다. Loop 1의 context encoding이 약하면 이후 loop가 global branch에서 이를 근본적으로 다시 만들기 어렵다.

### 2) Gated Sliding-Window Attention

Loop 2 이후의 attention은 두 branch를 결합한다.

- Global branch: frozen loop-1 KV에 대한 full-context attention
- Local branch: current-loop KV에 대한 sliding-window attention

$$
\tilde{y}^{(r)}
=
g \odot y_{global}^{(r)}
+
(1-g) \odot y_{local}^{(r)}
$$

$$
g
=
\sigma\left(
f_{gate}\left(\mathrm{RMSNorm}(\mathbf{h})\right)
\right)
$$

Gate는 attention head별 scalar를 만든다. 논문은 later loop에서도 mean gate가 0.5보다 높은 상태를 유지한다고 보고한다. 이는 후속 loop가 fresh local representation만 사용하는 것이 아니라, loop 1의 frozen global cache에 계속 크게 의존한다는 뜻이다.

G-SWA가 해결하는 문제는 다음과 같다.

- 모든 loop에서 full KV를 새로 보관하지 않는다.
- Global context는 loop 1 cache로 유지한다.
- Current loop가 만든 변화는 local window 안에서 반영한다.
- Gate가 두 source의 비중을 조절한다.

동시에 이 구조는 later loop의 global update capacity를 제한할 수 있다. Global branch가 늘 같은 cache를 읽기 때문에 loop가 깊어질수록 새로운 global routing이 줄어들 가능성이 있다.

### 3) Cross-Loop Parallelism

PLT의 핵심 latency mechanism은 CLP다. Loop $r \ge 2$에서 input을 다음처럼 만든다.

$$
B^{(r)}
=
\mathrm{Embed}(x)
+
\mathrm{shift}\left(\mathbf{h}^{(r-1)}\right)
$$

$$
\mathbf{h}^{(r)}
=
f_{\theta}\left(B^{(r)}\right)
$$

`shift`는 hidden state를 오른쪽으로 한 token 이동한다.

$$
\mathrm{shift}\left(\mathbf{h}^{(r-1)}\right)_i
=
\mathbf{h}^{(r-1)}_{i-1}
$$

이 구조를 사용하면 token $i$의 loop $r$ 계산과 token $i-1$의 loop $r+1$ 계산을 pipeline할 수 있다. Standard loop처럼 전체 sequence의 이전 loop가 끝나기를 기다리지 않으므로 near-single-pass execution을 목표로 할 수 있다.

하지만 token $i$는 자신의 이전-loop state가 아니라 neighbor token $i-1$의 state를 받는다. 이 정보 mismatch가 CLP의 core cost다.

### 4) Intrinsic offset cost

저자들은 CLP mismatch를 adjacent token representation distance로 측정한다.

$$
\Omega^{(r)}
=
\frac{1}{S}
\sum_i
\left\|
\mathbf{h}^{(r-1)}_i
-
\mathbf{h}^{(r-1)}_{i-1}
\right\|_2
$$

직관은 다음과 같다.

- Neighbor token representation이 비슷하면 one-position shift의 정보 손실이 작다.
- Neighbor representation이 다르면 token $i$가 받아야 할 state와 실제로 받는 state의 차이가 크다.
- 이 distance가 loop마다 비슷하게 유지되면 CLP는 매 loop에 거의 fixed tax를 부과한다.

논문은 $\Omega^{(r)}$가 loop가 깊어져도 roughly constant하다고 보고한다. 반면 output-distribution shift는 loop 2 이후 급격히 줄어든다. 이 둘의 gap을 gain-cost scissors라고 해석한다.

### 5) Hidden-state dynamics

Hidden state 분석에는 네 가지 metric을 사용한다.

#### Step size

$$
\delta^{(r)}
=
\left\|
\mathbf{h}^{(r)}
-
\mathbf{h}^{(r-1)}
\right\|_2
$$

Update magnitude가 얼마나 큰지 본다.

#### Angular change

$$
\cos\theta^{(r)}
=
\frac{
\left\langle
\mathbf{h}^{(r)}-\mathbf{h}^{(r-1)},
\mathbf{h}^{(r-1)}-\mathbf{h}^{(r-2)}
\right\rangle
}{
\left\|
\mathbf{h}^{(r)}-\mathbf{h}^{(r-1)}
\right\|_2
\left\|
\mathbf{h}^{(r-1)}-\mathbf{h}^{(r-2)}
\right\|_2
}
$$

- 값이 1에 가까우면 같은 방향으로 refinement한다.
- 0에 가까우면 update가 orthogonal하다.
- 0보다 작으면 update direction이 뒤집혀 oscillation할 가능성이 있다.

논문은 refinement loop에서 negative cosine을 관찰한다.

#### Effective rank

$$
\mathrm{erank}\left(\mathbf{h}^{(r)}\right)
=
\exp\left(
-\sum_i
\bar{\sigma}_i
\log\bar{\sigma}_i
\right)
$$

RMSNorm-normalized hidden matrix의 singular-value distribution을 사용한다. Effective rank가 높으면 token representation이 더 다양한 subspace를 사용한다고 해석할 수 있다.

논문에서는 loop 2에서 effective rank가 peak를 찍고 이후 감소한다. Later loop가 representation을 풍부하게 하기보다 subspace를 좁힌다는 근거로 사용된다.

#### Fixed-point gap

$$
\Delta_{FP}^{(r)}
=
\left\|
\mathbf{h}^{(r)}
-
f_{\theta}\left(\mathbf{h}^{(r)}\right)
\right\|_2
$$

Current state가 shared block의 fixed point와 얼마나 떨어져 있는지 본다. 추가 recurrence가 아직 바꿀 여지가 있는지 측정하려는 metric이다.

### 6) Attention evolution

Attention 분석은 다음 metric을 사용한다.

- Attention entropy
- Consecutive loop attention distribution의 KL divergence
- Attention head effective rank
- Head 간 cosine similarity
- G-SWA global gate mean

핵심 결과는 다음과 같다.

- Inter-loop attention KL은 loop 2 이후 급격히 낮아진다.
- Head 간 similarity는 loop가 깊어질수록 높아진다.
- Attention head diversity는 감소한다.
- Global gate는 계속 0.5보다 높아 loop-1 cache 의존이 유지된다.

즉 later loop는 새로운 information routing을 만들기보다 비슷한 attention pattern을 반복하는 경향을 보인다.

### 7) Output-distribution shift

Intermediate loop의 hidden state에 output head를 적용해 token probability를 만든다.

$$
p^{(r)}
=
\mathrm{Softmax}
\left(
\mathrm{Head}\left(\mathbf{h}^{(r)}\right)
\right)
$$

논문은 다음을 본다.

- Ground-truth token의 logit-lens rank
- Consecutive loop output distribution의 KL divergence
- Output entropy
- Token별 peak-contribution loop

$$
\Delta p^{(r)}
=
\mathrm{KL}
\left(
p^{(r)}
\|
p^{(r-1)}
\right)
$$

Loop 2가 가장 큰 post-context output change를 만들고, 이후 $\Delta p^{(r)}$는 크게 줄어든다. Final loop에서 small uptick이 보이지만, 저자들은 이를 새로운 representation refinement보다 output readout effect로 해석한다.

# 4. Training / Data / Recipe

## 4-1. Pretraining data

LoopCoder-v2 family는 internal deduplicated mixture 18T token으로 처음부터 학습된다.

- Text to code token ratio: 1:1
- Code language: 100개 이상
- Released paper에는 code portion의 top language share만 공개
- Full corpus와 deduplication pipeline은 공개되지 않음

Code token 중 상위 language share는 다음과 같다.

| Language | Code token share |
| --- | ---: |
| Java | 10.3% |
| Python | 10.1% |
| JavaScript | 9.4% |
| Markdown | 8.7% |
| TypeScript | 8.3% |
| C | 5.2% |
| C++ | 5.0% |
| PHP | 4.7% |
| C# | 4.0% |
| HTML | 3.7% |
| Others, 93 languages | 30.5% |

Markdown 비중이 8.7%라는 점은 code model의 repository and documentation workload를 고려한 것으로 보인다. 다만 raw data source, license distribution, code quality filtering, contamination control은 원문에서 제한적으로만 확인된다.

## 4-2. Matched loop-count training

가장 중요한 experimental control은 loop count별 model을 별도로 학습했다는 점이다.

- Baseline: $R=1$
- PLT variant: $R=2$
- PLT variant: $R=3$
- PLT variant: $R=4$

Training과 inference의 loop count는 일치한다. 즉 $R=2$로 학습한 model을 inference에서 $R=3$으로 늘리는 실험이 아니다.

이 점은 해석에서 매우 중요하다. 논문은 "하나의 checkpoint에 test-time loop를 동적으로 추가하면 어디까지 좋아지는가"보다, **각 loop count를 architecture setting으로 두고 matched training했을 때 어느 setting이 좋은가**를 연구한다.

따라서 LoopCoder-v2의 test-time computation scaling은 parameter-shared effective depth의 관점이지, released checkpoint에서 loop count를 자유롭게 바꾸는 adaptive inference recipe로 이해하면 안 된다.

## 4-3. Optimizer and schedule

Pretraining recipe는 다음과 같다.

| Item | Value |
| --- | --- |
| Optimizer | Adam |
| $\beta_1$ | 0.9 |
| $\beta_2$ | 0.95 |
| Epsilon | $10^{-15}$ |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |
| Peak learning rate | $4 \times 10^{-4}$ |
| Schedule | Cosine decay |
| Warmup | First 5% of training steps |
| Precision | bf16 |
| Memory optimization | Gradient checkpointing |

논문은 $R=1,2,3,4$ family 학습에 총 1M GPU hours가 사용되었다고 보고한다. 이 수치는 matched loop-count study의 규모를 보여주지만, 동시에 method exploration cost가 매우 크다는 점도 드러낸다.

## 4-4. Instruction tuning

네 model은 모두 동일한 6M supervised instruction-tuning examples로 fine-tuning된다. Evaluation은 final SFT checkpoint 기준이다.

Benchmark가 code generation뿐 아니라 agentic software engineering과 tool use까지 포함되므로 instruction data에는 단순 function completion보다 넓은 task가 들어갔을 가능성이 높다. 다만 6M example의 category mixture와 source는 공개 범위를 다시 확인해야 한다.

## 4-5. Training infrastructure

PLT는 weight sharing이 있으므로 naive distributed training과 잘 맞지 않을 수 있다. 논문은 customized Megatron-LM stack을 사용한다.

### Weight-tied unrolling

$R$개의 loop와 $L$개 shared layer를 scheduler 관점에서는 $R L$개 layer처럼 펼친다. 그러나 parameter object는 첫 loop에만 존재하고, 나머지 loop는 같은 module reference를 사용한다.

따라서 다음 항목은 loop count와 무관하게 shared block 기준으로 유지된다.

- Parameter count
- Optimizer state
- Checkpoint footprint

### Pipeline co-location

같은 physical layer의 여러 loop instance를 같은 pipeline stage에 둔다. Weight sharing 때문에 loop 사이에 parameter communication이 발생하지 않도록 하는 설계다.

### Two attention calls in later loops

Loop 2 이후 각 layer는 다음 두 attention을 실행한다.

1. Current-loop KV에 대한 width-64 sliding-window attention
2. Frozen loop-1 KV에 대한 full attention

두 output은 per-head gate로 합친다. Gate는 처음에 local/global 50:50에 가까운 상태가 되도록 zero initialization한다.

### Custom backward hook

CLP는 loop-1 cache와 embedding을 재사용한다. 논문은 scheduling을 위해 tensor를 detach하고 custom backward hook으로 gradient를 다시 accumulate한다고 설명한다.

이 부분은 PLT를 실제 대규모로 학습할 때 중요한 engineering detail이다. Architecture diagram만 구현하면 되는 문제가 아니라, shared tensor의 autograd와 pipeline schedule을 같이 맞춰야 한다.

## 4-6. Engineering notes

실무적으로 가져갈 만한 포인트는 다음과 같다.

1. **Loop count를 model config로 명시**
   - Training loop와 inference loop를 일치시킨다.
   - Released model을 임의의 $R$로 바꾸지 않는다.

2. **Shared global cache와 fresh local cache를 분리**
   - Global information은 reusable cache에 둔다.
   - Later loop update는 local window에서만 새로 만든다.

3. **Parameter sharing과 execution schedule을 분리**
   - Logical depth는 늘리되 parameter object는 공유한다.
   - Distributed scheduler에서는 loop instance를 별도 layer처럼 배치할 수 있다.

4. **Activation diagnostic을 training checkpoint마다 기록**
   - Effective rank
   - Inter-loop KL
   - Update cosine
   - Offset cost
   - Head similarity

5. **Benchmark peak보다 saturation signal을 본다**
   - Effective rank가 오르다 감소하는 지점
   - Attention KL이 급격히 줄어드는 지점
   - Output shift가 collapse하는 지점
   - Offset cost가 marginal gain보다 커지는 지점

6. **Custom architecture serving을 검증**
   - Official model은 `trust_remote_code=True`를 요구한다.
   - Standard vLLM or SGLang support와 exact PLT kernel path를 deployment 전에 확인해야 한다.

# 5. Evaluation

## 5-1. Benchmark suite

Evaluation은 다음 영역을 포함한다.

- HumanEval+
- MultiPL-E
- BigCodeBench-Full
- LiveCodeBench
- SWE-bench Verified
- SWE-bench Multilingual, paper table의 SWE-M
- Terminal-Bench v1
- Terminal-Bench 2.0
- Mind2Web
- BFCL v3

이 조합은 function-level generation부터 repository-level agent, terminal task, browser-style interaction, function calling까지 넓게 본다.

## 5-2. Main loop-count results

| Model | LiveCodeBench | SWE-bench Verified | Multi-SWE | Terminal-Bench 2.0 | BFCL v3 | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline, $R=1$ | 27.4 | 43.0 | 14.0 | 11.2 | 32.2 | 38.0 |
| LoopCoder-v2, $R=2$ | 35.4 | 64.4 | 31.0 | 21.0 | 40.1 | 46.5 |
| LoopCoder-v2, $R=3$ | 28.6 | 27.6 | 11.0 | 12.2 | 36.3 | 36.9 |
| LoopCoder-v2, $R=4$ | 24.5 | 22.4 | 9.3 | 9.0 | 39.5 | 34.3 |

핵심은 non-monotonicity다.

- $R=2$ average는 38.0에서 46.5로 오른다.
- $R=3$ average는 36.9로 baseline보다 낮다.
- $R=4$ average는 34.3으로 더 떨어진다.
- SWE-bench Verified는 43.0 -> 64.4 -> 27.6 -> 22.4로 급격하게 변한다.
- Multi-SWE는 14.0 -> 31.0 -> 11.0 -> 9.3이다.

다만 모든 benchmark가 같은 pattern을 보이는 것은 아니다.

- Mind2Web은 $R=4$가 41.4로 가장 높다.
- BFCL v3는 $R=2$ 40.1, $R=4$ 39.5로 차이가 작다.
- $R=2$의 Mind2Web 34.5는 baseline 35.3보다 약간 낮다.

따라서 "two loops always win"보다는 **broad average와 code-agent benchmark에서 $R=2$가 가장 안정적인 operating point**라고 읽는 편이 정확하다.

## 5-3. Comparison with larger models

논문은 $R=2$의 SWE-bench Verified 64.4가 다음 larger open model과 경쟁적이라고 강조한다.

- Qwen3-235B-A22B-Instruct: 45.2
- Kimi-Dev-72B: 60.4
- Qwen3-Coder-480B-A35B-Instruct: 67.0
- Kimi-K2-Instruct: 69.2

7B model이 agentic coding benchmark에서 large model에 가까운 결과를 낸다는 점은 인상적이다. 다만 이 비교는 model size와 score를 보여주는 leaderboard comparison이다.

다음 조건은 통제되지 않는다.

- Pretraining data
- Post-training data
- Tool scaffold
- Context construction
- Inference budget
- Patch generation protocol
- Model-specific benchmark tuning

Loop count의 causal effect를 보려면 같은 family 안의 $R=1,2,3,4$ 비교가 더 중요하다. Larger model comparison은 capability positioning으로 읽는 편이 안전하다.

논문은 held-out agentic setting인 SWE-bench-CC에서 $R=2$가 33.4를 기록했다고 추가로 보고한다. 이는 loop-2 gain이 SWE-bench Verified 한 benchmark에만 갇히지 않을 가능성을 보여준다.

## 5-4. Per-loop interpretability results

### 1) Effective rank peaks at loop 2

500 held-out sample 분석에서 hidden-state effective rank는 loop 2까지 증가하고 이후 감소한다. 이는 loop 2가 token representation diversity를 가장 넓게 만들고, later loop는 representation을 좁힌다는 해석을 지원한다.

### 2) Update direction becomes oscillatory

Successive hidden-state update의 cosine이 0보다 작아지는 구간이 나타난다. 즉 later loop가 같은 방향으로 convergence하기보다 이전 update를 일부 되돌리는 움직임을 보인다.

Oscillation이 곧 performance drop의 직접 원인이라고 단정할 수는 없지만, 추가 loop가 clean iterative refinement가 아니라는 evidence다.

### 3) Attention routing freezes

Inter-loop attention KL은 loop 2 이후 크게 감소한다. Head similarity는 증가하고 effective rank는 낮아진다. 이는 later loop가 새로운 relation을 찾기보다 비슷한 routing을 반복한다는 뜻이다.

G-SWA gate가 0.5보다 높게 유지된다는 점도 같은 방향이다. Later loop는 loop 1의 global cache를 계속 많이 사용하며, fresh local branch가 global representation을 완전히 갱신하지 못한다.

### 4) Output shift collapses after loop 2

Output distribution KL $\Delta p^{(r)}$도 loop 2 이후 급격히 감소한다. Prediction confidence는 계속 sharpen될 수 있지만, distribution 자체가 유의미하게 이동하는 정도는 작아진다.

이 구분이 중요하다.

- Entropy 감소: 더 확신하는가.
- Ground-truth rank 개선: 맞는 token 쪽으로 가는가.
- Distribution KL: 이전 loop와 다른 computation을 하는가.

Confidence가 높아졌다고 반드시 새로운 reasoning이 일어난 것은 아니다.

### 5) Offset cost stays roughly fixed

Intrinsic offset cost $\Omega^{(r)}$는 loop마다 비슷한 수준을 유지한다. 논문 Figure 3은 loop 2 이후 refinement gain이 collapse하는 동안 offset cost가 유지되고, 추가 loop에서 offset cost가 per-loop gain보다 30x에서 45x 크다고 보고한다.

이 ratio는 서로 다른 diagnostic quantity의 scale에 의존하므로 publish 전 Figure 3의 normalization과 axis를 다시 확인할 필요가 있다. 그러나 qualitative message는 분명하다.

- Marginal gain은 줄어든다.
- CLP mismatch는 자동으로 줄지 않는다.
- Later loop일수록 fixed tax가 net effect를 지배한다.

### 6) Loop 3 is close to a dead pass-through

Four-loop model 분석에서 loop 3은 output shift, attention re-routing, peak-contribution token share 모두 가장 작다. Loop 4는 output-side share가 다시 커지지만 effective rank는 가장 낮다. 저자들은 이를 new representation enrichment보다 final readout에 가깝다고 해석한다.

이 관찰은 "더 깊은 effective depth"와 "더 많은 useful computation"이 같은 개념이 아님을 보여준다.

## 5-5. Explicit and latent chain-of-thought

논문은 optimal $R=2$ model에서 instruction variant와 thinking variant를 비교한다.

| Model, $R=2$ | LiveCodeBench | CRUX | MultiPL-E | FullStackBench | BCB-Hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| Instruct, latent loop only | 35.4 | 86.9 | 73.9 | 47.2 | 23.7 |
| Thinking, explicit CoT plus loop | 62.3 | 93.5 | 77.8 | 49.9 | 26.4 |

LiveCodeBench에서는 35.4에서 62.3으로 26.9 point 상승한다. 저자들은 explicit CoT와 latent loop의 결합이 super-additive하다고 주장한다.

설계 관점에서는 두 compute channel의 granularity가 다르다.

- Explicit CoT: 문제를 여러 textual step으로 분해한다.
- Latent loop: 각 textual step 아래의 representation을 추가 refinement한다.

따라서 latent recurrence가 CoT를 대체한다기보다, CoT가 만든 sequence 위에서 내부 refinement를 반복하는 구조가 될 수 있다.

다만 "explicit CoT alone does not improve the non-looped model"이라는 비교의 exact setting과 full table은 원문에서 다시 확인해야 한다. Thinking SFT data와 inference budget이 instruct model과 어떻게 다른지도 중요한 해석 변수다.

## 5-6. What really matters in the experiments

### 1) Loop count effect는 large-scale matched comparison이다

이 논문의 가장 강한 부분은 $R=1,2,3,4$가 같은 base configuration, pretraining token 수, instruction tuning, benchmark protocol을 사용한다는 점이다. Model family 내부에서는 loop count 차이를 비교적 잘 분리한다.

### 2) Efficiency mechanism은 새로운 inductive bias를 만든다

CLP는 latency를 줄이는 scheduling trick처럼 보이지만, 실제로 token $i$에 neighbor state를 넣는다. 즉 implementation optimization이 information path를 바꾼다.

이 논문은 system optimization과 model semantics를 따로 볼 수 없다는 좋은 사례다.

### 3) Best loop count는 "more compute"가 아니라 "best net refinement"다

$R=2$는 parameter 수를 늘리지 않고 useful depth를 추가한다. $R=3$과 $R=4$는 compute는 늘지만 representation diversity와 attention novelty가 줄어든다.

Compute budget을 늘리는 것과 reasoning capacity를 늘리는 것은 다른 문제다.

### 4) Effective rank는 유용한 early warning signal일 수 있다

저자들은 effective-rank trajectory가 상승 중이면 extra loop가 도움이 될 수 있고, 감소하기 시작하면 representation narrowing이 시작된 것으로 볼 수 있다고 제안한다.

이는 exhaustive benchmark sweep보다 가벼운 diagnostic이 될 수 있다. 다만 candidate model과 activation sample이 이미 있어야 하므로 architecture search cost를 완전히 없애는 것은 아니다.

### 5) Latent and explicit reasoning은 서로 다른 axis다

Looped model research에서 latent computation은 often chain-of-thought의 replacement처럼 소개된다. LoopCoder-v2 결과는 두 방식이 함께 쓸 때 더 강할 수 있음을 보여준다.

향후 중요한 질문은 다음과 같다.

- 어느 token step에 latent loop를 더 배정할 것인가.
- Explicit CoT의 어떤 구간에서 recurrence가 가장 유용한가.
- Easy step은 $R=1$, hard step은 $R=2$로 dynamic routing할 수 있는가.
- CLP offset을 task difficulty에 따라 조정할 수 있는가.

# 6. Limitations

1. **PLT-specific result다**
   - $R=2$ saturation은 one-token CLP offset과 shared loop-1 KV를 사용하는 PLT에서 나온다.
   - Standard sequential loop, MELT, Hyperloop, Mixture-of-Recursions에 그대로 일반화할 수 없다.

2. **각 loop count를 별도로 학습했다**
   - 하나의 checkpoint에서 inference loop를 자유롭게 늘리는 experiment가 아니다.
   - $R=2$ model을 $R=3$으로 바꾸면 같은 결과가 나온다고 볼 수 없다.
   - Dynamic test-time depth claim은 후속 검증이 필요하다.

3. **7B coding domain 하나가 중심이다**
   - General language, math-only model, multimodal model, smaller edge model에서도 loop 2가 best인지 알 수 없다.
   - Code token structure와 repository task가 CLP offset에 특별히 잘 맞거나 안 맞을 수 있다.

4. **Training cost가 매우 크다**
   - 네 variant 전체에 1M GPU hours가 사용되었다.
   - Loop-count diagnostic의 목적이 brute-force sweep을 줄이는 것이지만, 결과를 만들기 위해 이미 expensive sweep을 수행했다.

5. **Pretraining data가 internal이다**
   - 18T token, 1:1 text-code ratio, language share는 공개되지만 raw source와 full filtering recipe는 공개되지 않는다.
   - Contamination과 license composition을 독립적으로 검증하기 어렵다.

6. **Instruction data detail이 제한적이다**
   - 6M example의 task mixture, quality filter, agentic trace source가 결과에 큰 영향을 줄 수 있다.
   - SWE-bench gain이 architecture와 post-training data 중 어디에서 얼마나 오는지 추가 ablation이 필요하다.

7. **Interpretability metric은 causal proof가 아니다**
   - Effective rank decline, negative update cosine, attention redundancy는 performance regression과 함께 움직인다.
   - 이 metric을 직접 intervention해 performance가 회복되는지는 보여주지 않는다.

8. **Offset cost definition은 proxy다**
   - Adjacent hidden-state distance는 intuitive하지만 functional error 전체를 측정하지는 않는다.
   - Token type, syntax boundary, code indentation, long-range dependency에 따른 mismatch를 평균 하나로 압축한다.

9. **CLP and G-SWA ablation이 더 필요하다**
   - Offset size 1, window 64, frozen loop-1 KV, gate design이 고정되어 있다.
   - Alternative offset, learned alignment, refreshed global cache가 saturation point를 바꿀 수 있다.

10. **Detailed wall-clock profile은 main result가 아니다**
    - Near-single-pass latency와 constant KV argument는 PLT architecture에서 온다.
    - LoopCoder-v2 paper는 quality and representation analysis가 중심이며, hardware별 latency, throughput, batch scaling은 원 PLT paper와 implementation에서 다시 확인해야 한다.

11. **Agent benchmark는 scaffold-sensitive하다**
    - SWE-bench, Terminal-Bench, Mind2Web, BFCL은 prompt, tool wrapper, retry policy, context construction에 영향을 받는다.
    - Model-only score로 단순 해석하면 안 된다.

12. **Explicit CoT comparison의 compute matching이 중요하다**
    - Thinking model은 더 많은 output token과 다른 fine-tuning data를 사용할 수 있다.
    - Super-additivity claim은 matched total FLOPs와 token budget 아래에서 재검증할 가치가 있다.

13. **Released checkpoint는 custom code에 의존한다**
    - Hugging Face loading에 `trust_remote_code=True`가 필요하다.
    - Production에서는 custom model code, kernel support, security review가 필요하다.

14. **Code safety는 별도 문제다**
    - Model card도 incorrect, insecure, incomplete code 가능성을 명시한다.
    - Repository-level agent에 사용할 때 test, sandbox, secret isolation, patch review가 필수다.

# 7. My Take

## 7-1. Why this matters for my work

이 논문의 가장 중요한 메시지는 "7B model이 SWE-bench 64.4를 기록했다"가 아니다. 더 중요한 부분은 **efficient recurrence가 만든 architectural distortion을 internal metric으로 찾아낸 것**이다.

AI system 연구에서는 cost optimization을 model behavior와 별개로 보기 쉽다.

- KV cache를 공유한다.
- Layer를 반복 사용한다.
- Loop를 pipeline한다.
- Window attention으로 바꾼다.
- State를 shift해 dependency를 끊는다.

하지만 이런 변경은 모두 information path를 바꾼다. PLT의 CLP도 latency optimization이면서 동시에 token alignment를 바꾸는 inductive bias다. LoopCoder-v2는 이 trade-off를 "효율화했으니 더 많은 loop가 가능하다"에서 끝내지 않고, "효율화를 위해 지불한 mismatch가 언제 gain을 넘는가"까지 추적한다.

이 관점은 attention compression, KV sharing, speculative decoding, recurrent depth, memory routing에도 그대로 적용할 수 있다.

> System cost를 줄인 mechanism이 model semantics에 어떤 fixed tax를 추가하는지 측정해야 한다.

## 7-2. Reuse potential

### 1) Recurrent-depth architecture search

새 looped architecture를 설계할 때 benchmark sweep 전에 다음 curve를 볼 수 있다.

- Effective rank by loop
- Inter-loop attention KL
- Output distribution KL
- Update-direction cosine
- Cache reliance gate
- Architecture-specific mismatch metric

여러 signal이 같은 saturation point를 가리키면 candidate loop count를 빠르게 좁힐 수 있다.

### 2) Dynamic loop allocation

LoopCoder-v2는 fixed $R$ model을 비교하지만 diagnostic은 dynamic depth로 확장할 수 있다.

- Effective rank가 상승 중인 token만 추가 loop
- Output shift가 threshold 아래면 early exit
- High offset cost token은 recurrence skip
- Complex code span에만 $R=2$
- Natural-language wrapper나 boilerplate에는 $R=1$

다만 PLT는 token pipeline dependency가 있으므로 per-token dynamic loop는 scheduling과 cache contract를 다시 설계해야 한다.

### 3) Agentic coding serving

Repository agent는 long prompt, long patch, tool output이 반복되므로 KV memory와 latency가 중요하다. PLT의 shared global cache와 local refinement는 이 workload에 매력적이다.

Production에서 확인할 항목은 다음과 같다.

- Long repository context에서 global loop-1 cache 품질
- Local window 64가 code dependency를 충분히 보존하는지
- Multi-file editing에서 CLP shift가 syntax boundary에 미치는 영향
- Batch serving에서 actual latency
- Tool call 사이 state reuse 가능성
- Patch security and test pass rate

### 4) Explicit plus latent compute

Thinking model 결과는 model serving policy에 다른 선택지를 준다.

- 모든 problem에 긴 CoT를 생성하지 않는다.
- Latent loop만으로 충분한 task는 short answer를 낸다.
- Reasoning-intensive task에는 explicit CoT를 켠다.
- 각 CoT step 아래에서 one extra loop를 사용한다.

이 구조는 output token cost와 latent compute를 별도 knob으로 다루게 한다.

### 5) Training infrastructure

Customized Megatron-LM implementation에서 얻을 수 있는 engineering lesson도 크다.

- Shared parameter와 virtual layer schedule 분리
- Same shared layer의 loop instance co-location
- Frozen global KV plus fresh local KV
- Detached shared tensors에 custom gradient accumulation
- Gate zero initialization
- Loop-aware activation logging

이런 detail이 없으면 architecture idea가 scale training에서 재현되지 않을 수 있다.

## 7-3. Production design considerations

Released LoopCoder-V2를 바로 agent에 붙일 경우 다음을 먼저 확인하는 편이 좋다.

1. **Loop count 고정**
   - Model card의 recommended setting은 $R=2$다.
   - Config를 임의로 늘리지 않는다.

2. **Custom code audit**
   - `trust_remote_code=True`가 필요하다.
   - Model implementation과 generation path를 review한다.

3. **Latency benchmark**
   - Target GPU, batch size, context length, output length에서 직접 측정한다.
   - PLT의 theoretical near-constant cost와 실제 kernel overhead를 구분한다.

4. **Code safety**
   - Generated patch를 sandbox에서 실행한다.
   - Unit test와 static analysis를 적용한다.
   - Secret, network, filesystem permission을 제한한다.

5. **Agent scaffold matching**
   - SWE-bench score만 보고 production task success를 예측하지 않는다.
   - 실제 repository, tool, retry policy로 evaluation한다.

6. **Activation telemetry**
   - Optional debug mode에서 loop-wise output shift와 effective rank를 기록한다.
   - Domain shift에서 $R=2$가 계속 productive한지 확인한다.

## 7-4. Follow-up papers

- Parallel Loop Transformer for Efficient Test-Time Computation Scaling
- Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach
- Universal Transformers
- Looped Transformers for Length Generalization
- Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation
- MELT
- Hyperloop Transformers
- How Much Is a Loop Worth?
- Stabilizing Recurrent-Depth Models with Fixed-Point Regularization

# 8. Summary

- LoopCoder-v2는 PLT loop count를 $R=1,2,3,4$로 matched training한 7B coding model family다.
- $R=2$는 SWE-bench Verified 43.0 -> 64.4, Multi-SWE 14.0 -> 31.0으로 개선하지만 $R \ge 3$은 여러 task에서 regression한다.
- Loop 2에서 hidden-state effective rank, attention re-routing, output shift가 가장 크게 나타난다.
- Later loop는 representation narrowing과 oscillatory update를 보이는 반면 CLP offset cost는 roughly constant하다.
- 핵심 교훈은 더 많은 test-time compute가 아니라, architectural tax를 제외하고 남는 marginal refinement를 측정해야 한다는 점이다.
