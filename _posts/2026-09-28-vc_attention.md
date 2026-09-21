---
layout: single
title: "VC-Attention: Value Smoothing and Softmax Casting for Low-bit Attention Review"
categories: Study-concept
tag: [VideoGeneration, LowBitAttention, GPUKernel]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

[Paper link](https://arxiv.org/pdf/2609.15810)

Low-bit attention 연구는 보통 query-key product를 얼마나 안정적으로 quantize할 것인가에 집중한다. Q와 K의 outlier를 rotation이나 smoothing으로 줄이고, integer 또는 FP8 GEMM으로 score matrix를 계산하는 방식이다.

VC-Attention은 그 다음 병목을 본다. QK path를 충분히 정리하고 나면 실제 output error의 큰 부분은 value path에서 남고, kernel latency의 critical path에는 FP32 softmax exponential과 low-precision cast가 남는다.

즉 이 논문의 질문은 다음과 같다.

> QK quantization만 잘하면 low-bit attention이 완성되는가?

Wan2.2 분석에서 QK smoothing 이후 attention output error의 82%가 value quantization 쪽에서 발생한다. 동시에 H200과 B200에서는 softmax exponential과 cast가 attention kernel의 병목으로 남는다. VC-Attention은 이 두 문제를 각각 V-Smooth와 ExpCast-FP8로 푼다.

> 한 줄 요약: VC-Attention은 token permutation과 block-mean residual quantization으로 value outlier를 줄이는 V-Smooth, 그리고 softmax score를 E4M3 code로 직접 보내는 ExpCast-FP8을 결합해 video DiT attention의 품질과 GPU kernel latency를 함께 개선하는 training-free framework다.

이 논문을 지금 볼 가치가 있는 이유는 다음과 같음.

- Low-bit attention의 error budget을 QK와 PV로 분해하고 value path가 실제 병목임을 측정한다.
- Token reorder를 approximation이 아니라 self-attention의 permutation invariance를 이용한 exact transformation으로 사용한다.
- Mean subtraction 뒤에 사라질 수 있는 low-frequency component를 online softmax row sum으로 정확히 복원한다.
- FP32 exponential과 FP8 cast를 하나의 log-domain mapping으로 합쳐 kernel critical path를 줄인다.
- B200, B300, H200, RTX PRO 6000, RTX 5090에서 quality and latency를 함께 측정한다.

# 1. Problem Setting

## 1-1. Attention quantization error는 어디에서 생기는가

Attention output은 다음과 같이 쓸 수 있다.

$$
O = PV,
\qquad
P = \operatorname{softmax}\left(\frac{QK^T}{\sqrt d}\right)
$$

Quantized probability와 value를 각각 $P_q$, $V_q$라고 하면 output error는 아래처럼 분해된다.

$$
O-O_q
=
(P-P_q)V
+
P_q(V-V_q)
$$

첫 번째 항은 QK and softmax probability error이고, 두 번째 항은 value quantization error다.

기존 low-bit attention은 첫 번째 항을 줄이는 데 강하다. Hadamard rotation, per-block scale, QK smoothing을 사용하면 score quantization은 상당히 안정화된다. 그러나 value tensor에는 channel and token outlier가 남고, $P_q(V-V_q)$가 전체 error를 지배할 수 있다.

논문은 Wan2.2에서 QK smoothing 이후 value-side error가 output error의 82%를 차지한다고 보고한다. 따라서 더 낮은 bit로 내려가기 위해서는 V를 별도로 다뤄야 한다.

## 1-2. Video DiT가 특히 어려운 이유

Video diffusion transformer는 긴 visual sequence와 큰 hidden dimension을 사용한다. Denoising step마다 self-attention을 반복하기 때문에 작은 kernel inefficiency가 전체 generation latency에 누적된다.

또한 video token은 공간과 시간에 걸쳐 correlation을 가진다. 인접한 token이 항상 value space에서 비슷한 것은 아니며, static background와 motion region의 scale도 다르다. 단순 contiguous block quantization은 한 block 안에 서로 다른 value cluster를 섞어 큰 scale을 선택하게 만들 수 있다.

## 1-3. Softmax가 남기는 hardware bottleneck

Low-bit QK GEMM을 빠르게 만들어도 softmax는 FP32 exponential을 수행한다. 그 뒤 probability를 FP8로 cast해야 한다.

Datacenter GPU에서는 GEMM이 빨라질수록 다음 단계가 상대적으로 더 비싸진다.

1. Score normalization
2. FP32 exponential
3. Online row-sum update
4. FP8 conversion
5. PV GEMM

VC-Attention은 exponential과 cast를 별도 단계로 두지 않고, log-domain score에서 E4M3 probability code로 직접 이동하는 방법을 설계한다.

# 2. Core Idea

## 2-1. V-Smooth: 비슷한 value를 같은 block으로 모은다

첫 번째 아이디어는 token을 value similarity 기준으로 cluster한 뒤 같은 cluster가 contiguous하도록 reorder하는 것이다.

Permutation을 $\pi$라고 하면 key와 value를 함께 재배열한다.

$$
K' = K[\pi],
\qquad
V' = V[\pi]
$$

이때 attention probability의 column도 같은 순서로 바뀐다.

$$
P'
=
\operatorname{softmax}
\left(
\frac{QK'^T}{\sqrt d}
\right)
$$

Key and value를 같은 permutation으로 바꾸면 output은 유지된다.

$$
P'V' = PV
$$

따라서 reorder 자체는 approximation이 아니다. Approximation은 reorder 이후 residual value를 low-bit로 quantize하는 단계에서만 생긴다.

논문은 online clustering으로 비슷한 value token을 모은다. Balanced k-means가 error만 보면 더 좋지만 overhead가 크기 때문에, practical trade-off가 좋은 warm-started k-means를 선택한다.

## 2-2. Block mean을 빼고 residual만 quantize한다

Reordered value를 block $j$로 나누고 block mean을 계산한다.

$$
\mu_j
=
\frac{1}{B_v}
V_j'^T\mathbf{1}
$$

Residual은 다음과 같다.

$$
R_j
=
V_j'
-
\mathbf{1}\mu_j^T
$$

Value block의 common component를 제거하면 residual dynamic range가 작아지고 low-bit scale이 outlier에 덜 끌린다.

논문에서 mean subtraction으로 제거되는 energy는 original sequence에서는 약 8%, static cube에서는 약 12%지만, clustering and sorting 이후에는 약 36%까지 커진다. Reorder가 mean residual quantization을 더 효과적으로 만드는 이유다.

## 2-3. Mean restoration을 별도 pass 없이 처리한다

Mean을 뺀 residual만 quantize하면 원래 output의 mean contribution을 복원해야 한다. Naive하게는 full-precision mean path를 별도 GEMM으로 계산해야 한다.

VC-Attention은 online softmax가 이미 유지하는 block probability row sum을 사용한다. Query row $i$가 block $j$에 배정한 probability mass를 $r_{ij}$라고 하면 다음이 성립한다.

$$
P_{ij}V_j'
=
P_{ij}R_j
+
r_{ij}\mu_j^T
$$

Residual path는 low-bit PV GEMM으로 계산하고, mean path는 scalar row sum과 mean vector의 outer-product 형태로 복원한다. 추가 memory pass나 full-precision value buffer가 필요하지 않다.

이 설계가 V-Smooth의 핵심이다. 단순 demeaning이 아니라, online attention state를 이용해 제거한 성분을 정확히 되돌린다.

## 2-4. ExpCast-FP8: exponential과 cast를 합친다

Softmax probability는 score $s$에 대해 $p=\exp(s)$ 형태로 계산된다. ExpCast-FP8은 log-domain score를 E4M3 probability code에 직접 mapping한다.

핵심 목표는 다음 두 연산을 분리하지 않는 것이다.

$$
s
\rightarrow
\exp(s)
\rightarrow
\operatorname{cast}_{\mathrm{FP8}}
$$

대신 exponent and mantissa structure를 이용해 하나의 FMA 중심 mapping으로 FP8 code를 얻는다.

논문 분석에서 exact FP8 code와 같은 code를 선택하는 구간이 약 79.6%이고, 나머지도 대부분 인접 code로 mapping된다. Attention distribution의 total-variation error upper bound는 underflow mass를 제외하면 3.64%이며, 204.8K rows에서 평균 empirical error는 약 1.6%다.

# 3. Architecture / Method

## 3-1. Overview

| Component | Role | Why it matters |
| --- | --- | --- |
| Online clustering | Value-similar token grouping | Block dynamic range 축소 |
| Joint K/V permutation | Exact token reorder | Attention output 보존 |
| Block demeaning | Common component separation | Residual quantization 안정화 |
| Online mean restoration | Row sum으로 mean contribution 복원 | Extra pass 제거 |
| ExpCast-FP8 | Log-domain score to E4M3 | FP32 exp and cast bottleneck 제거 |
| Fused CuTe kernel | Rotation, gather, quantization, GEMM fusion | HBM round trip 최소화 |

## 3-2. Denoising-step schedule

Video diffusion의 모든 denoising step에서 clustering을 새로 수행하면 overhead가 커진다. 논문은 first 25% denoising steps에서 grouping을 수행하고 permutation을 이후 step에 재사용한다.

이 schedule은 diffusion dynamics를 고려한다.

- Early step에서는 representation이 빠르게 바뀌므로 grouping update 가치가 크다.
- Later step에서는 token structure가 상대적으로 안정화되어 permutation reuse가 가능하다.
- 모든 step에서 clustering하면 quality는 조금 더 좋아질 수 있지만 latency benefit이 줄어든다.

Ablation에서 first-quarter grouping이 quality and cost 균형이 가장 좋다. 모든 step grouping은 약 0.5 dB를 더 얻지만 generation time이 약 16 seconds 늘고, uniform quarter schedule은 first-quarter schedule보다 약 2.6 dB 낮다.

## 3-3. Kernel fusion

VC-Attention은 algorithm만 제안하지 않고 실행 경로를 함께 설계한다.

Fused path에는 다음 연산이 포함된다.

- Rotary transform
- Block mean computation
- Token gather
- Hadamard rotation
- Q/K/V quantization
- Low-bit QK and PV
- Online softmax
- Mean restoration

중간 high-precision tensor를 HBM에 기록하지 않고 register and shared-memory path에서 이어 붙인다. Unfused chain 대비 fused implementation은 논문 설정에서 8.74x 짧은 실행 시간을 보인다.

이 수치는 attention 전체 speedup이 아니라 해당 preprocessing and quantization chain의 fused-versus-unfused 비교라는 점을 구분해야 한다.

# 4. Training / Data / Recipe

## 4-1. Training-free setting

VC-Attention은 model retraining이나 QAT를 요구하지 않는다. 기존 checkpoint에 inference-time transformation과 custom kernel을 적용한다.

따라서 이 섹션에서 중요한 것은 training recipe보다 deployment recipe다.

1. Model별 attention tensor shape를 확인한다.
2. Early denoising step에서 value clustering을 실행한다.
3. K and V에 동일 permutation을 적용한다.
4. Value block mean을 계산하고 residual을 quantize한다.
5. QK score는 low-bit path로 계산한다.
6. ExpCast-FP8으로 probability code를 생성한다.
7. Low-bit PV와 exact mean restoration을 fused한다.
8. Later denoising step에서는 permutation을 재사용한다.

## 4-2. Evaluation models

실험은 네 개의 video generation model을 포함한다.

- Wan2.2 at 720p
- LongCat at 480p
- HunyuanVideo-1.5 at 720p
- MiniMax-H3 at 1344x768 with a third-party 8-step acceleration LoRA

Quality comparison은 MovieGen Bench의 100 prompts를 같은 seed로 실행해 BF16 output과 quantized output을 비교한다.

주요 metric은 다음과 같다.

- PSNR
- SSIM
- LPIPS
- VBench subject consistency
- VBench imaging quality

Speed는 attention kernel latency와 end-to-end generation latency를 따로 측정한다.

## 4-3. Hardware scope

Datacenter GPU:

- NVIDIA B200
- NVIDIA B300
- NVIDIA H200

Workstation GPU:

- NVIDIA RTX PRO 6000
- NVIDIA RTX 5090

ExpCast-FP8은 8-bit kernel의 B200 and H200 path에서만 활성화된다. Workstation 4-bit result는 V-Smooth and custom low-bit kernel의 효과를 중심으로 본다.

이 hardware breadth는 중요하다. 같은 low-bit algorithm도 Tensor Core path, memory bandwidth, exponential throughput에 따라 bottleneck이 달라질 수 있기 때문이다.

# 5. Evaluation

## 5-1. Quality results

8-bit setting에서 V-Smooth는 SageAttention2보다 PSNR을 개선한다.

- Wan2.2: +2.3 dB
- HunyuanVideo-1.5: +2.8 dB
- LPIPS reduction: 13% to 29%

ExpCast-FP8을 추가하면 V-Smooth-only보다 PSNR이 0.7 to 2.1 dB 낮아지지만, end-to-end low-bit baseline보다 여전히 높은 fidelity를 유지한다.

4-bit setting에서는 SageAttention3 대비 다음 개선을 보고한다.

- Wan2.2: +2.9 dB
- LongCat: +3.6 dB
- LPIPS reduction: 최대 41%

이 결과는 value-side smoothing이 8-bit뿐 아니라 aggressive 4-bit attention에서 더 중요해질 수 있음을 보여준다.

## 5-2. Kernel and end-to-end speed

Datacenter GPU:

| Hardware | Attention speedup | End-to-end speedup |
| --- | ---: | ---: |
| B200 | 1.59x | 1.19x |
| H200 | 1.46x | 1.13x |

Workstation GPU:

| Hardware | Attention speedup | End-to-end speedup |
| --- | ---: | ---: |
| RTX PRO 6000 | 2.27x | 1.36x |
| RTX 5090 | 3.58x | 1.70x |

Workstation에서 speedup이 더 큰 이유는 FP32 exponential and cast overhead, memory path, native low-precision throughput의 상대 비중이 다르기 때문이다.

B300 비교에서도 naive FP8 attention이 1.31x and 17.1 dB인 반면, VC-Attention은 1.47x and 18.4 dB를 보고한다. 단순 datatype replacement보다 value layout과 softmax path를 함께 설계해야 함을 보여준다.

## 5-3. 무엇을 기준으로 봐야 하는가

이 논문의 headline은 최대 speedup보다 quality-latency frontier다.

주의해서 볼 지표는 다음과 같다.

1. Kernel speedup
   - Attention operator 자체가 얼마나 빨라졌는가.

2. End-to-end speedup
   - DiT 전체 generation에서 attention 비중을 반영한 실제 개선인가.

3. BF16 output fidelity
   - 같은 prompt and seed에서 quantized output이 얼마나 가까운가.

4. Semantic quality
   - Pixel similarity가 낮아도 subject or image quality가 유지되는가.

5. Preprocessing overhead
   - Clustering, reorder, scale computation이 speedup을 상쇄하지 않는가.

VC-Attention은 이 다섯 축을 모두 보고한다는 점이 좋다.

# 6. Limitations

1. **Non-causal video self-attention 중심임**

   K and V permutation이 exact한 이유는 query가 전체 key set을 보는 self-attention 구조에 있다. Causal LLM attention에서는 token order와 mask boundary가 의미를 가지므로 같은 reorder를 그대로 적용할 수 없다.

2. **Custom kernel 의존성이 큼**

   CuTe and CUDA implementation, GPU generation, tensor shape에 따라 speedup이 달라진다. Framework-level portable operator로 바로 옮기기 어렵다.

3. **Clustering overhead and schedule tuning**

   First-quarter schedule이 잘 작동하지만, model and sampler가 달라지면 representation stabilization 시점도 달라질 수 있다.

4. **Datacenter 4-bit path의 미완성**

   논문은 datacenter GPU에서 4-bit scale computation이 critical path로 남기 때문에 4-bit datacenter result를 제공하지 않는다. 4-bit quality gain과 production speedup을 같은 setting에서 확인하지 못한다.

5. **Fidelity metric의 한계**

   PSNR and LPIPS는 BF16 output과의 closeness를 보지만, perceptual acceptability 전체를 의미하지 않는다. 반대로 VBench 일부 dimension은 near-saturation이라 method 차이를 잘 구분하지 못한다.

6. **Video model 범위**

   평가 model은 넓지만 모두 video DiT 계열이다. Image DiT, multimodal transformer, encoder-decoder cross-attention에 대한 일반화는 추가 검증이 필요하다.

# 7. My Take

## 7-1. 핵심은 value quantization보다 error-budget 재배치다

VC-Attention을 단순 V quantization trick으로 읽으면 아쉽다. 더 중요한 기여는 low-bit attention의 병목이 algorithm 개선에 따라 이동한다는 점을 보여준 것이다.

QK error를 줄이면 value error가 지배한다. Low-bit GEMM을 빠르게 만들면 softmax exponential이 지배한다. 따라서 다음 optimization target은 이전 generation kernel과 다르다.

이것은 hardware-aware ML system에서 자주 반복되는 패턴이다.

$$
\text{optimize one stage}
\rightarrow
\text{move the bottleneck}
\rightarrow
\text{redesign the next stage}
$$

## 7-2. Exact transformation과 approximation을 분리한 점이 좋다

V-Smooth에는 exact part와 approximate part가 명확히 나뉜다.

- Joint K/V permutation: exact
- Block mean separation and restoration: exact
- Residual low-bit quantization: approximate
- ExpCast-FP8: approximate

이 구조는 debugging and ablation에 유리하다. Quality loss가 발생했을 때 permutation 때문인지, residual quantization 때문인지, softmax casting 때문인지 분리할 수 있다.

production quantization에서 이 분리는 매우 중요하다. 여러 approximation을 한 kernel에 한꺼번에 넣으면 score는 좋아도 failure provenance가 사라진다.

## 7-3. Blackwell workstation에서 특히 실용적인 이유

RTX PRO 6000 and RTX 5090에서 end-to-end speedup이 datacenter GPU보다 크게 나타난다. High-resolution video generation을 single workstation에서 실행할 때 attention memory traffic과 softmax path가 체감 병목이 될 수 있다는 뜻이다.

다만 실제 적용 전에는 다음을 확인해야 한다.

- Target model의 attention layout
- Sequence length별 clustering overhead
- Torch compile or custom extension integration cost
- FP8 support and kernel fallback
- Multi-GPU sequence parallel과의 compatibility
- Output quality regression을 잡는 prompt subset

## 7-4. Follow-up papers

- SageAttention2
- SageAttention3
- FlashAttention-4
- Attn-QAT
- Native Sparse Attention

함께 읽으면 low-bit datatype, score approximation, sparsity, hardware layout이 각각 attention system의 어느 병목을 겨냥하는지 비교하기 좋다.

# 8. Summary

- QK smoothing 이후에는 value quantization error가 attention output error를 지배할 수 있다.
- V-Smooth는 value-similar token을 모으고 block mean을 분리해 residual dynamic range를 줄인다.
- Joint K/V permutation과 mean restoration은 attention output을 보존하는 exact transformation이다.
- ExpCast-FP8은 FP32 exponential과 FP8 cast를 log-domain mapping으로 합친다.
- Video DiT에서 quality를 유지하면서 datacenter and workstation GPU의 kernel and end-to-end latency를 개선한다.
