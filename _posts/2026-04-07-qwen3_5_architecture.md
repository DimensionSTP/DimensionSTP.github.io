---
layout: single
title: "Qwen3.5 아키텍처 종합 리뷰: Qwen3와 무엇이 달라졌고, Gated DeltaNet은 왜 Mamba가 아닌가"
categories: Study-concept
tag: [LLM, Qwen, Architecture, MoE, DeltaNet, Mamba, VLM]
toc: true
author_profile: false
sidebar:
  nav: "docs"
search: true
typora-root-url: ../
---

# 0. Introduction

Qwen3.5는 이름만 보면 Qwen3의 마이너 업데이트처럼 보인다. 하지만 공개된 model card, checkpoint config, Hugging Face 구현을 따라가 보면, 이 모델은 attention-only Transformer 계열의 소폭 수정판이 아니라 **token mixer, MoE 결합 방식, 장문맥 전략, MTP, multimodal 포지셔닝까지 한 번에 재설계한 세대 교체형 모델**에 가깝다.

이 글은 발표용으로 정리해 두었던 원문을 AI를 활용하여 블로그 포맷으로 다시 다듬은 버전이다. 따라서 발표 자료에서 중요했던 표, 수식, 코드 단서, 참고문헌, 부록까지는 하나도 빼지 않고 유지했고, 대신 블로그에서 처음 읽는 사람도 흐름을 따라가기 쉽도록 서두와 결론부, 섹션 연결, 어투를 조금 더 읽기 좋은 방향으로 정리했다.

> 원 발표 초안의 범위  
> **대상:** Qwen3 <-> Qwen3.5 비교, Gated DeltaNet 해부, MoE/MTP/DeepStack/Nemotron 3 Super 비교    
> **형식:** Notion에 바로 붙여넣기 쉬운 Markdown    
> **기준:** Hugging Face 공식 구현 코드, 공식 model card, 공식 checkpoint config, 공식 기술 보고서/논문/서빙 문서 중심

## 0-1. 핵심 결론

### 3줄 요약
1. **Qwen3.5는 Qwen3의 "마이너 개량판"이 아니라, 하이브리드 토큰 믹서(`Gated DeltaNet + Gated Attention`)를 채택한 새 세대**에 가깝다. 공개 artifact 기준으로는 **Qwen3보다 Qwen3-Next와 구조적으로 더 닮아 있다**.  
2. **Qwen3.5의 linear attention은 Mamba SSM이 아니라 DeltaNet 계열**이다. 구현상 `q/k/v`를 유지하고, `(d_k × d_v)` recurrent memory를 **delta correction**으로 갱신한다는 점에서 **attention memory를 recurrent state로 압축한 구조**에 가깝다.  
3. **MoE는 더 고희소(sparser)해지고, shared expert가 복귀하며, MTP*초장문맥*VLM이 기본 전제**가 되었다. 다만 DeepStack/MTP 일부는 **config에는 보이지만 public HF forward 경로에서는 덜 노출**되어 있다.

### 한 문장으로
> **Qwen3 = standard Transformer-MoE**,  
> **Qwen3.5 = hybrid DeltaNet-Attention-MoE-VLM**.

## 0-2. 이 글을 읽는 법

- `[확인됨]`: 공식 코드/공식 카드/공식 config/공식 논문에서 직접 확인되는 내용
- `[추론]`: 공식 자료를 조합했을 때 가장 자연스러운 해석
- `[보류]`: 공개 자료만으로는 단정하기 어려운 내용

## 0-3. 출처 정책

이 문서는 가능한 한 **1차 자료**를 우선했다.

- **공식 구현:** Hugging Face `transformers`의 `modeling_qwen3_moe.py`, `modeling_qwen3_5_moe.py`, `modeling_qwen3_vl.py`
- **공식 model card / config:** Qwen3, Qwen3.5, Qwen3-Next, DeepSeek-V3, GLM-4.5
- **공식 논문/리포트:** Qwen3 Technical Report, Qwen3-VL Technical Report, Mamba, DeltaNet, Gated Delta Networks, NVIDIA Nemotron 3 Super 자료
- **공식/준공식 서빙 문서:** vLLM, Unsloth, NVIDIA training/deployment docs

문서 끝의 **참고문헌**에 모두 정리했다.

* * *

# 1. 한눈에 보는 비교 지도

## 1-1. Qwen3 vs Qwen3.5 vs Qwen3-Next vs Nemotron 3 Super

| 항목 | Qwen3 | Qwen3.5 | Qwen3-Next | Nemotron 3 Super |
|---|---|---|---|---|
| 큰 틀 | standard Transformer / Transformer-MoE | **Hybrid** (Gated DeltaNet + Gated Attention) + FFN/MoE + Vision Encoder | **Hybrid** (Gated DeltaNet + Gated Attention) + high-sparsity MoE | **Hybrid Mamba-Transformer** + LatentMoE |
| token mixer | 전 층 full attention | `3 × linear_attention + 1 × full_attention` 반복 | `3 × GDN + 1 × GA` 반복 | Mamba branch + Transformer branch |
| attention | GQA + QK-Norm + RoPE | Gated Attention (output gating 추가) | Gated Attention | Transformer attention |
| linear branch | 없음 | **Gated DeltaNet** | **Gated DeltaNet** | **Mamba SSM** |
| FFN/MoE | dense 또는 sparse code path, Qwen3-235B config는 실제로 every layer sparse | dense(27B) 또는 sparse MoE(35B/122B/397B) | sparse MoE | LatentMoE |
| shared expert | Qwen3-MoE는 **제거** | **복귀** | 있음 | 공식 문서상 LatentMoE 강조 |
| MTP | family 차원에선 강조 약함 | **있음** (`mtp_num_hidden_layers=1`) | **있음** | **있음** |
| VLM | 기본 전제 아님 | **기본 VLM** (`Causal Language Model with Vision Encoder`) | 텍스트 모델 | 공식 자료상 unified VLM 포지셔닝 아님 |
| native context | 32K (235B), YaRN 확장 131K | **262K native**, 1.01M까지 확장 | 262K native, 1.01M ext. | 최대 1M (family 자료) |
| 공개 artifact의 느낌 | Qwen2.5의 연장선 | **Qwen3-Next를 VLM/제품군으로 확장한 세대** | Qwen3.5와 가장 닮은 텍스트형 | NVIDIA 최적화 중심 hybrid family |

**핵심 포인트:**  
Qwen3.5는 이름만 보면 Qwen3의 점진적 업데이트처럼 보이지만, **공개 구조는 Qwen3보다 Qwen3-Next와 더 닮아 있다**. 이는 model card의 구조 설명, config의 `layer_types`, HF 구현의 `Qwen3_5MoeGatedDeltaNet`/`Qwen3_5MoeAttention` 분기에서 모두 확인된다. `[확인됨][R5][R6][R9][R10][R11][R12]`

---

## 1-2. 구조 도식(ASCII)

### Qwen3 (전통적 Transformer-MoE)
```text
Token Embedding
   ↓
[L times]
  RMSNorm
    -> Full Self-Attention (GQA + QK-Norm + RoPE)
    -> Residual
  RMSNorm
    -> FFN or Sparse MoE
    -> Residual
   ↓
LM Head
```

### Qwen3.5 (Hybrid DeltaNet-Attention-MoE / VLM)
```text
Text Tokens + Vision Encoder Outputs
   ↓
[repeat]
  3 × (
        RMSNorm
          -> Gated DeltaNet (linear attention)
          -> Residual
        RMSNorm
          -> FFN / Sparse MoE
          -> Residual
      )
  1 × (
        RMSNorm
          -> Gated Attention (full attention + output gate)
          -> Residual
        RMSNorm
          -> FFN / Sparse MoE
          -> Residual
      )
   ↓
(MTP-aware family)
   ↓
LM Head
```

### Nemotron 3 Super (공개 자료 기준 고수준)
```text
Hybrid Mamba-Transformer Backbone
   + LatentMoE
   + MTP
   + NVFP4-oriented training / serving optimizations
```

> **주의:** Nemotron 3 Super의 "정확한 per-layer 반복 패턴"은 내가 사용한 공식 자료 범위에선 Qwen3.5만큼 상세히 공개되어 있지 않다. 따라서 Nemotron은 **고수준 비교**까지만 하는 것이 안전하다. `[확인됨][R22][R23]`

---

# 2. Qwen3에 비해 Qwen3.5는 무엇이 달라졌는가  
(이론적 / 아키텍처적 / 실제 구현 코드)

## 2-1. 요약표

| 차원 | Qwen3 | Qwen3.5 | 왜 중요한가 |
|---|---|---|---|
| backbone 철학 | standard Transformer 계열 | **hybrid linear/full attention** | 컨텍스트 길이, cache 형태, 서빙 전략이 달라진다 |
| attention | standard GQA | **Gated Attention** | attention branch 자체에 output gate가 추가됨 |
| linear branch | 없음 | **Gated DeltaNet** | Mamba가 아니라 DeltaNet 계열 recurrent memory 도입 |
| norm | standard RMSNorm | **1-centered RMSNorm + gated RMSNorm** | 학습/추론 안정화 |
| MoE | 128 experts / 8 active, shared expert 제거 | 256 or 512 experts, shared expert 복귀 | sparse capacity와 fallback path 강화 |
| VLM | 기본 전제 아님 | **Vision encoder 기본 내장** | 아예 family definition이 달라짐 |
| context | 32K native (235B), 131K ext. | **262K native, 1.01M ext.** | 초장문맥 지향성이 강화됨 |
| MTP | 모델 카드상 핵심 전면화는 약함 | **native MTP 흔적 명확** | latency 개선과 serving engine 연동 |
| config signal | Transformer 중심 config | `layer_types`, `linear_*`, `attn_output_gate`, `mtp_*`, VLM config | "기능 흔적"이 config 차원에서 드러남 |

---

## 2-2. 이론적 변화

Qwen3 기술 보고서는 dense 모델을 **GQA + SwiGLU + RoPE + RMSNorm pre-norm + QK-Norm**을 쓰는 Qwen2.5 계열의 진화형으로 설명한다. Qwen3-MoE는 **128 experts 중 8 experts를 token마다 활성화**하며, **Qwen2.5-MoE와 달리 shared expert를 제거**했다고 명시한다. 또한 장문맥 확장은 ABF, YaRN, Dual Chunk Attention 중심이다. `[확인됨][R1]`

반면 Qwen3.5 model card는 첫 줄부터 **"Unified Vision-Language Foundation"**, **"Efficient Hybrid Architecture"**를 강조하고, layout 자체를  
`3 × (Gated DeltaNet -> MoE) -> 1 × (Gated Attention -> MoE)`  
반복으로 적는다. 즉 Qwen3.5는 "Transformer에 MoE만 얹은 것"이 아니라, **토큰 믹서 설계 자체를 바꾼 세대**다. `[확인됨][R5][R6][R7][R8]`

**정리하면:**

- **Qwen3:** "Transformer backbone + MoE"
- **Qwen3.5:** "Hybrid token mixer (GDN + GA) + MoE + VLM + MTP"

---

## 2-3. 아키텍처 변화

### Qwen3-235B-A22B (공식 card/config)
- 235B total / 22B active
- 94 layers
- 64 attention heads / 4 KV heads
- 128 experts / 8 active
- native context 32,768
- YaRN로 131,072까지 확장 `[확인됨][R2][R3]`

### Qwen3.5-35B-A3B (공식 card/config)
- 35B total / 3B active
- 40 layers
- hidden layout: `10 × (3 × (Gated DeltaNet -> MoE) -> 1 × (Gated Attention -> MoE))`
- Gated DeltaNet: 32 value heads / 16 QK heads / 128 dim
- Gated Attention: 16 Q / 2 KV / 256 dim / rotary 64
- MoE: 256 experts / 8 routed + 1 shared
- native context 262,144 / 1,010,000 ext.
- `attn_output_gate: true`, `full_attention_interval: 4`, `mtp_num_hidden_layers: 1` `[확인됨][R5][R9]`

### Qwen3.5-122B-A10B (공식 card/config)
- 122B total / 10B active
- 48 layers
- layout: `12 × (3 × (Gated DeltaNet -> MoE) -> 1 × (Gated Attention -> MoE))`
- Gated DeltaNet: 64 V / 16 QK / 128 dim
- Gated Attention: 32 Q / 2 KV / 256 dim
- MoE: 256 experts / 8 routed + 1 shared
- `mtp_num_hidden_layers: 1` `[확인됨][R6][R10]`

### Qwen3.5-27B dense
- 27B total
- 64 layers
- layout: `16 × (3 × (Gated DeltaNet -> FFN) -> 1 × (Gated Attention -> FFN))`
- 즉 **Qwen3.5 family에는 dense와 MoE가 함께 존재** `[확인됨][R7]`

---

## 2-4. 구현 코드 변화

### Qwen3-MoE decoder
HF `Qwen3MoeDecoderLayer`는 항상 `self.self_attn = Qwen3MoeAttention(...)`를 만들고, FFN 자리에 `Qwen3MoeMLP` 또는 `Qwen3MoeSparseMoeBlock`을 둔다. 공식 235B config의 `decoder_sparse_step=1` 때문에 **실제 checkpoint에선 사실상 every layer sparse MoE**다. `[확인됨][R3][R4]`

### Qwen3.5-MoE decoder
HF `Qwen3_5MoeDecoderLayer`는 `layer_type = config.layer_types[layer_idx]`를 읽고
- `"linear_attention"` -> `Qwen3_5MoeGatedDeltaNet`
- `"full_attention"` -> `Qwen3_5MoeAttention`
으로 분기한다. 즉 **attention 자리에 Gated DeltaNet이 직접 들어온다.** `[확인됨][R11]`

---

# 3. Qwen3.5의 "달라진 부분" 상세 리뷰

## 3-1. 코드 delta 요약표

| 구성요소 | Qwen3-MoE 구현 | Qwen3.5 구현 | 의미 |
|---|---|---|---|
| decoder token mixer | 항상 `Qwen3MoeAttention` | `layer_types`에 따라 `GatedDeltaNet` 또는 `GatedAttention` | backbone이 하이브리드화 |
| `q_proj` | 일반 query 차원 | **2배 차원** -> `[query, gate]` split | output gating용 gate 추가 |
| attention output | 그대로 `o_proj` | `attn_output * sigmoid(gate)` 후 `o_proj` | gated attention |
| RMSNorm | `weight` init = 1, `weight * norm(x)` | `weight` init = 0, `(1 + weight) * norm(x)` | 1-centered norm |
| linear-branch norm | 없음 | `RMSNormGated(x, z)` | output gate를 정규화 뒤 적용 |
| router renorm | config 의존적 | **항상 renorm** | routing stability 강화 |
| sparse MoE output | routed experts only | routed experts + **gated shared expert** | fallback/dense prior 경로 |
| cache | KV cache 중심 | KV + conv/recurrent states | linear branch용 state 도입 |
| external kernels | FlashAttention/SDPA 중심 | `flash-linear-attention`, `causal-conv1d`, fused recurrent rule | 성능/추론 경로 변화 |

---

## 3-2. RMSNorm: 구현과 수식 비교

### Qwen3 standard RMSNorm
HF 구현은 T5-style RMSNorm과 동일한 형태다.

$$
\mathrm{RMSNorm}_{Q3}(x) 
= \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}}
$$

- `weight`는 **ones**로 초기화
- forward는 사실상 `weight * norm(x)` `[확인됨][R4]`

### Qwen3.5 1-centered RMSNorm
HF 구현은 다음처럼 동작한다.

$$
\mathrm{RMSNorm}_{Q3.5}(x) 
= (1+\gamma) \odot \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}}
$$

- `weight`는 **zeros**로 초기화
- forward는 `(1.0 + self.weight) * norm(x)` `[확인됨][R11]`

### 왜 이게 중요한가
초기 상태에서 Qwen3.5는 파라미터가 0이어도 실제 스케일은 1이다.  
즉 **파라미터 표현은 zero-centered, 실제 함수는 identity-centered**인 셈이다.

이건 Qwen3-Next card의 **"zero-centered and weight-decayed layernorm"** 설명과도 방향이 맞는다. `[확인됨][R12]`

### 포인트
- Qwen3: "표준 RMSNorm"
- Qwen3.5: "**1-centered RMSNorm**"
- linear branch: "**gated RMSNorm**까지 추가"

---

## 3-3. Gated RMSNorm: 수식과 코드

Qwen3.5의 linear branch에서는 일반 RMSNorm이 아니라 **`Qwen3_5MoeRMSNormGated`** 또는 fused 대응 모듈을 쓴다. Python fallback 구현은 다음과 같다.

$$
y = \mathrm{RMSNorm}(x) \odot \mathrm{SiLU}(z)
$$

즉 **정규화 후 gate**를 거는 구조다. 코드 주석도 "Norm before gate"라고 명시한다. `[확인됨][R11]`

### 해석
- `RMSNorm`: 수치 안정화
- `SiLU(z)`: 정보 전달량 조절
- 결과적으로 linear branch 출력이 **정규화 + 게이팅**을 동시에 받는다.

---

## 3-4. Full attention: GQA -> Gated Attention

### 표준 full attention (Qwen3)
$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$
$$
A = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}} + M\right)
$$
$$
Y = AV
$$

Qwen3는 여기에 GQA, QK-Norm, RoPE가 붙는 구조다. `[확인됨][R1][R4]`

### Gated Attention (Qwen3.5)
Qwen3.5 full-attention 구현에서 가장 중요한 변화는 `q_proj`가 **두 배 차원**을 내고, 그 결과를 `query`와 `gate`로 split한다는 점이다.

$$
[Q, G] = XW_{QG}
$$
$$
K = XW_K,\quad V = XW_V
$$
$$
H = \mathrm{Attn}(Q,K,V)
$$
$$
Y = \sigma(G) \odot H
$$
$$
O = YW_O
$$

즉 attention core는 유지하되, **attention output branch에 multiplicative gate**를 추가한다. `[확인됨][R11]`

### 구현 비교
- Qwen3: `q_proj -> q`, `k_proj -> k`, `v_proj -> v`, attention, `o_proj`
- Qwen3.5: `q_proj -> [query, gate]`, attention, `attn_output *= sigmoid(gate)`, `o_proj`

### 포인트
> Qwen3.5의 gated attention은 "attention score를 바꾸는 메커니즘"이 아니라, **attention output을 residual에 넘기기 전에 한 번 더 조절하는 메커니즘**이다.

---

## 3-5. MLP / expert 자체는 얼마나 바뀌었는가?

이 부분은 오히려 **많이 안 바뀌었다.**

### 공통된 기본 수식
Qwen3와 Qwen3.5의 dense MLP는 둘 다 전형적 SwiGLU형이다.

$$
\mathrm{MLP}(x) = W_\text{down}\left(\phi(W_\text{gate}x) \odot (W_\text{up}x)\right)
$$

HF 구현도 두 파일 모두 사실상 같은 패턴이다. expert 내부도 `gate_up_proj`를 반으로 나눠 `act(gate) * up` 후 `down_proj`를 거치는 구조다. `[확인됨][R4][R11]`

### 즉, 진짜로 바뀐 곳은
- **expert 내부 수식**이 아니라
- **sparse block의 결합 방식**
- **router renorm**
- **shared expert 경로**
- **하이브리드 backbone과의 결합**

이다.

---

# 4. Linear Attention(Gated DeltaNet)은 full attention, Mamba SSM과 어떻게 다른가  
(구현 / 아키텍처 / 이론)

---

## 4-1. 먼저 한 문장씩 정의

### Full attention
> 모든 과거 토큰 쌍에 대해 score matrix를 만들고, softmax-weighted value 합으로 현재 출력을 만든다.

### Gated DeltaNet
> `q/k/v`를 유지하되, **고정 크기 recurrent memory**를 유지하고, 현재 value와 현재 memory readout의 차이(**delta**)만큼 state를 보정한다.

### Mamba SSM
> 입력 의존적인 selective state-space dynamics로 latent state를 진화시키고, 그 state에서 출력을 읽는다.

---

## 4-2. 고수준 구조 비교

| 항목 | Full Attention | Gated DeltaNet | Mamba SSM |
|---|---|---|---|
| 핵심 표현 | `Q, K, V` + full score matrix | `Q, K, V` + recurrent memory | latent SSM state |
| 전역 상호작용 | 명시적 `QK^T` | 압축된 recurrent memory를 통한 간접 상호작용 | selective state transition |
| 상태 크기 | growing KV cache | fixed-size recurrent/conv state | fixed-size recurrent state |
| 선택성 | softmax weights | `beta`, `g`, `z` gates | input-dependent SSM parameters |
| attention 친화성 | 원형 | **매우 높음** | 낮음 |
| Mamba 친화성 | 낮음 | 중간 | 원형 |

---

## 4-3. Gated DeltaNet의 전체 흐름(공식 구현 기준)

```text
hidden_states
  -> apply_mask_to_padding_states
  -> in_proj_qkv / in_proj_z / in_proj_b / in_proj_a
  -> depthwise causal conv on qkv
  -> split into q, k, v
  -> beta = sigmoid(b)
  -> g = -exp(A_log) * softplus(a + dt_bias)
  -> if prefill:
         chunk_gated_delta_rule(...)
     else if decode:
         recurrent_gated_delta_rule(...)
  -> RMSNormGated(core_out, z)
  -> out_proj
```

이 순서는 HF `Qwen3_5MoeGatedDeltaNet.forward`와 helper 함수들에서 그대로 읽힌다. `[확인됨][R11]`

---

## 4-4. helper 함수까지 포함한 라인 바이 라인 해설

### (a) `apply_mask_to_padding_states`
padding 토큰의 hidden state를 먼저 0에 가깝게 만들어, conv/recurrent state가 padding에 오염되지 않게 한다. 코드 주석은 Mamba issue #66을 직접 가리킨다. `[확인됨][R11]`

**attention과의 차이:**  
attention은 보통 **score-space mask**를 쓰지만, Gated DeltaNet은 먼저 **state update 전에 hidden 자체를 mask**한다.

---

### (b) `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`
입력을 네 갈래로 나눈다.

- `qkv`: 실제 읽기/쓰기 표현
- `z`: output gate
- `b`: update strength gate
- `a`: decay gate 계산용 입력

즉 "gated"라는 말은 단일 gate가 아니라 **세 종류의 gate(`beta`, `g`, `z`)**를 가리킨다. `[확인됨][R11]`

---

### (c) depthwise causal conv
qkv를 그대로 recurrence에 넣지 않고, 먼저 depthwise causal conv를 통과시킨다. `causal-conv1d` fast path가 있으면 그것을 쓰고, 없으면 느린 PyTorch fallback으로 내려간다. `[확인됨][R11]`

**해석:**  
이 conv는 Mamba의 selective scan처럼 "모델의 정체성"이라기보다는 **local causal filtering / qkv 전처리** 역할에 가깝다.

---

### (d) `beta = sigmoid(b)`
$$
\beta_t = \sigma(b_t)
$$

현재 토큰이 state를 얼마나 강하게 수정할지 정하는 **update gate**다. `[확인됨][R11]`

---

### (e) `g = -exp(A_log) * softplus(a + dt_bias)`
$$
g_t = -\exp(A_{\log}) \cdot \mathrm{softplus}(a_t + b_\Delta)
$$

이 값은 나중에 `exp(g_t)`로 실제 decay factor가 된다. 즉 **forget/retention gate** 역할이다. 코드엔 fp16에서 `A_log`를 float로 올리지 않으면 `-inf`가 날 수 있다는 주석까지 있다. `[확인됨][R11]`

---

### (f) `chunk_gated_delta_rule` vs `recurrent_gated_delta_rule`
- **prefill(긴 입력을 한 번에 넣을 때):** chunk 병렬 경로
- **decode(토큰 1개씩 생성할 때):** recurrent 경로

즉 같은 수학을 **훈련/프리필에서는 병렬화하고, 생성에서는 순수 recurrence로 실행**한다. `[확인됨][R11]`

---

## 4-5. Gated DeltaNet의 핵심 수식

decode 시 torch fallback recurrence를 식으로 쓰면 대략 다음과 같다.

$$
S_t^- = \exp(g_t)\odot S_{t-1}
$$

$$
\hat v_t = \langle S_t^-, k_t \rangle
$$

$$
\Delta_t = \beta_t \odot (v_t - \hat v_t)
$$

$$
S_t = S_t^- + k_t \otimes \Delta_t
$$

$$
o_t = \langle S_t, q_t \rangle
$$

여기서 $S_t \in \mathbb{R}^{d_k \times d_v}$ 는 recurrent memory다. `[확인됨][R11]`

### 해석
- 먼저 과거 state를 decay
- 현재 key로 memory를 읽어서 현재 value를 얼마나 이미 설명하는지 확인
- **설명하지 못한 잔차(delta)만** state에 쓴다
- query로 state를 읽어 출력 생성

이 때문에 Gated DeltaNet은 additive linear attention보다 **"correction memory"** 성격이 강하다.

---

## 4-6. Full attention vs Gated DeltaNet vs Mamba SSM: 수식 비교

### Full attention
$$
A = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}} + M\right),\quad Y = AV
$$

- 명시적 $T \times T$ 상호작용
- KV cache가 시퀀스 길이와 함께 커진다

### Gated DeltaNet
$$
S_t = \underbrace{\exp(g_t)\odot S_{t-1}}_{\text{forget}}
+ \underbrace{k_t \otimes \left(\beta_t \odot (v_t - \hat v_t)\right)}_{\text{delta correction}}
$$
$$
o_t = \langle S_t, q_t \rangle
$$

- `q/k/v`는 유지
- full score matrix는 만들지 않음
- state는 고정 크기의 associative memory

### Mamba SSM (개념식)
$$
h_t = \bar A_t h_{t-1} + \bar B_t x_t,\qquad y_t = C_t h_t
$$

- `q/k/v` 없음
- selective state-space dynamics
- latent state evolution이 핵심 `[확인됨][R15]`

---

## 4-7. 구현적인 차이: attention / DeltaNet / Mamba

| 구현 항목 | Full Attention | Gated DeltaNet | Mamba SSM |
|---|---|---|---|
| 입력 투영 | `q_proj`, `k_proj`, `v_proj` | `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a` | SSM parameter/input projections |
| local preprocessing | 보통 없음 또는 RoPE | depthwise causal conv | selective scan / conv-like state-space kernel |
| 핵심 state | past K/V | `(d_k, d_v)` recurrent matrix + conv state | latent SSM state |
| prefill | full/flash attention | chunk_gated_delta_rule | scan parallelization |
| decode | KV cache append | fused_recurrent_gated_delta_rule | recurrent scan |
| fast path 의존성 | FlashAttention/SDPA | `flash-linear-attention`, `causal-conv1d` | Mamba-specific kernels |

---

## 4-8. 결론: Gated DeltaNet은 Mamba인가?

**아니다.**  
정확히는 **"Mamba와 같은 linear-time 진영에 있으나, 계보상으로는 DeltaNet/linear transformer 쪽"**이다.

### 이유 3가지
1. `q/k/v`를 명시적으로 유지한다.
2. state의 shape와 update가 **associative memory**에 가깝다.
3. update 식의 핵심이 **delta correction**이다.

즉 Qwen3.5의 linear branch는  
**"Mamba-like runtime sensitivity를 보일 수는 있지만, 모델 개념으로는 Gated DeltaNet"**이다. `[확인됨][R11][R13][R14][R15]`

---

# 5. MoE 측면에서는 무엇이 개선되었는가

## 5-1. Qwen3-MoE vs Qwen3.5-MoE

| 항목 | Qwen3-MoE | Qwen3.5-MoE | 의미 |
|---|---|---|---|
| experts 수 | 128 | 256 (35B/122B) 또는 512 (397B) | capacity 증가 |
| active experts | 8 | 8 routed + 1 shared / 또는 10 + 1 | routed path + dense prior |
| shared expert | **없음** | **있음** | fallback 경로 |
| router renorm | config 의존 | **항상 on** | 안정성 강화 |
| block output | routed experts only | routed + gated shared | 더 안정적인 expert mixture |
| family 성격 | Transformer-MoE | Hybrid DeltaNet/Attention-MoE | 토큰 믹서와 함께 개선 |

Qwen3 Technical Report는 Qwen3-MoE가 **shared expert를 사용하지 않는다**고 명시한다. 반면 Qwen3.5 model card와 HF 구현은 **shared expert가 실제로 존재**하며, block 출력에 gated shared expert output이 더해진다. `[확인됨][R1][R5][R6][R8][R11]`

---

## 5-2. HF 구현 기준 실제 차이

### Qwen3-MoE sparse block
- routed experts만 계산
- router renorm은 `norm_topk_prob`가 켜졌을 때만 수행 `[확인됨][R4]`

### Qwen3.5 sparse block
- `self.shared_expert`, `self.shared_expert_gate` 추가
- routed expert output + `sigmoid(shared_gate) * shared_expert_output`
- router top-k prob **항상 renorm** `[확인됨][R11]`

---

## 5-3. MoE 비교: Qwen3 / Qwen3.5 / Qwen3-Next / DeepSeek / GLM

| 모델 | 총 파라미터 / 활성 파라미터 | total experts | active per token | shared expert | 비고 |
|---|---|---:|---:|---:|---|
| Qwen3-235B-A22B | 235B / 22B | 128 | 8 | 0 | Qwen3-MoE는 shared expert 제거 `[R1][R2][R3]` |
| Qwen3.5-35B-A3B | 35B / 3B | 256 | 8 routed + 1 shared | 1 | hybrid VLM `[R5][R9]` |
| Qwen3.5-122B-A10B | 122B / 10B | 256 | 8 routed + 1 shared | 1 | hybrid VLM `[R6][R10]` |
| Qwen3.5-397B-A17B | 397B / 17B | 512 | 10 routed + 1 shared | 1 | largest public Qwen3.5 `[R8]` |
| Qwen3-Next-80B-A3B | 80B / 3B | 512 | 10 activated + 1 shared | 1 | 텍스트형 hybrid `[R12]` |
| DeepSeek-V3 | 671B / 37B | 256 routed | 8 | 1 | MLA + DeepSeekMoE + MTP `[R24][R25]` |
| GLM-4.5 | 355B / 32B | 160 routed | 8 | 1 | MTP layers 강조 `[R26][R27]` |

### 해석
- Qwen3.5는 Qwen3보다 **더 많은 experts + shared expert 복귀**로 간다.
- DeepSeek-V3와 GLM-4.5도 shared expert + MTP를 갖고 있어, **업계 공통 추세는 "shared fallback + MTP + 더 세밀한 routing"**로 보인다.
- Qwen3.5의 특이점은 이 MoE를 **Gated DeltaNet/Gated Attention 하이브리드 backbone 위에 얹었다는 것**이다.

---

## 5-4. MoE에서 실제로 개선된 것

### [확인됨]
- experts 수 증가
- shared expert 추가
- unconditional top-k renorm
- family 차원에서 MTP와 결합 `[R5][R6][R8][R9][R10][R11]`

### [추론]
- routed experts만으로 representation을 감당하던 구조보다,  
  **shared expert를 더한 구조가 routing miss / cold-start / sparse gradient 문제에 더 안전**하다.
- hybrid token mixer와 결합했을 때 shared expert는 **linear/full branch의 표현 분포 차이를 흡수하는 완충재** 역할도 할 가능성이 있다.

---

# 6. fused RMSNorm은 무엇이고 왜 쓰는가

## 6-1. 정확히 무엇인가

Qwen3.5에서 말하는 fused RMSNorm은 "모든 RMSNorm을 바꾸는 것"이 아니라, **linear branch의 `RMSNorm + gate activation` 경로를 한 번에 계산하는 fused 커널**을 뜻한다.

HF 구현은
- `flash-linear-attention`이 있으면 `FusedRMSNormGated`
- 없으면 `Qwen3_5MoeRMSNormGated` (Python fallback)
를 쓴다. `[확인됨][R11]`

---

## 6-2. 수식

fused든 unfused든 계산 의미는 동일하다.

$$
y = \mathrm{RMSNorm}(x) \odot \mathrm{SiLU}(z)
$$

즉 **수학을 바꾸는 기능이 아니라, 실행을 빠르게 만드는 기능**이다.

---

## 6-3. 왜 필요한가

Qwen3.5 linear branch는 다음 요소가 한 번에 들어간다.

- qkv projection
- depthwise causal conv
- chunk/recurrent gated-delta kernel
- gated RMSNorm
- output projection

따라서 unfused Python/Torch 경로로 모두 처리하면 **메모리 왕복과 kernel launch overhead**가 커진다. HF 구현도 external fast path가 없으면 "slow implementation"으로 fallback한다고 경고한다. `[확인됨][R11]`

### 한 줄 요약
> fused RMSNorm은 "새 알고리즘"이 아니라, **Gated DeltaNet을 실제 속도로 돌리기 위한 필수 최적화 조각**이다.

---

# 7. Unsloth의 Qwen3.5 파인튜닝 가이드에서 말한 "Mamba SSM 이슈"는 무엇인가

## 7-1. 공식 가이드가 실제로 말하는 것

Unsloth Qwen3.5 fine-tuning guide는 다음을 분명히 말한다.

- 훈련이 평소보다 느리면 **custom Mamba Triton kernels** 컴파일 탓일 수 있음
- 특히 T4에서 컴파일 시간이 길 수 있음
- **QLoRA 4-bit는 권장하지 않음**
- MoE fine-tuning은 bf16 권장
- router-layer fine-tuning은 기본적으로 비활성화(`[stability]`) `[확인됨][R18]`

Unsloth의 GGUF/quantization 문서도 `ssm_out`과 attention 관련 텐서가 hybrid 구조에서 **특히 양자화 민감**하다고 지적한다. `[확인됨][R19]`

---

## 7-2. 이것은 "Qwen3.5 = Mamba"라는 뜻인가?

**아니다.**

### 더 정확한 해석
- Qwen3.5는 **Mamba 모델이 아니라 Gated DeltaNet hybrid 모델**
- 그러나 runtime/backend 관점에서는 linear/recurrent branch가 **Mamba 계열과 비슷한 커널 민감도**를 보일 수 있다.
- 실제 HF 코드도 padding helper 주석에서 **Mamba issue #66**를 직접 참조한다. `[확인됨][R11]`

### 따라서
Unsloth가 말한 "Mamba Triton kernels"는  
**모델 분류의 명칭**이라기보다 **backend/kernel 생태계의 명칭**으로 이해하는 것이 맞다.

---

## 7-3. 왜 QLoRA 4-bit가 특히 문제인가

### [확인됨]
Unsloth는 QLoRA 4-bit를 권장하지 않고, GGUF 문서에서 `ssm_out` 같은 텐서가 양자화에 민감하다고 말한다. `[R18][R19]`

### [추론]
Qwen3.5는
- full attention branch
- recurrent/SSM-like linear branch
- MoE routed/shared experts
를 동시에 갖기 때문에, **양자화 오차가 특정 분기에서 증폭**될 여지가 크다.

---

## 7-4. 생태계 성숙도 이슈

Unsloth 문서는 특정 시점에 "vLLM 0.16.0은 Qwen3.5를 지원하지 않는다"고 적었지만, 현재 공식 vLLM 문서는 `qwen3_5`와 `qwen3_5_mtp`를 지원 모델로 문서화한다. 즉 이 이슈는 **영구 구조 문제라기보다 초기 생태계 성숙도 문제**로 보는 것이 맞다. `[확인됨][R18][R20][R21]`

---

# 8. Qwen3.5는 왜 이번에 바로 VLM으로 냈는가

## 8-1. 공식 자료가 직접 확인해 주는 부분

Qwen3.5 model card는 아예 모델 타입을  
**"Causal Language Model with Vision Encoder"**  
로 정의하고, 첫 번째 강조점으로  
**"Unified Vision-Language Foundation"**  
을 내세운다. 또 `--language-model-only` 옵션으로 **vision encoder를 로드하지 않는 text-only 사용**도 지원한다. `[확인됨][R5][R6][R7][R8]`

즉 공식 포지셔닝만 놓고 보면, Qwen3.5는 "텍스트 모델에 vision을 얹은 파생판"이 아니라 **처음부터 unified VLM family**다.

---

## 8-2. 그럼 "왜 바로 VLM이냐?"에 대한 해석

### [확인됨]
- family definition 자체가 VLM
- model card 핵심 문구가 unified VLM
- training efficiency도 multimodal 관점으로 서술 `[R5][R6][R7][R8]`

### [추론]
Qwen3.5 세대는 이미
- 초장문맥
- hybrid token mixer
- agentic benchmark
- MTP
- multimodal training efficiency
를 함께 묶는 방향으로 설계되었고, 따라서 **language-only -> VLM 파생**이 아니라 **unified multimodal backbone을 기본값으로 삼은 세대**로 보는 편이 자연스럽다.

### 표현(안전한 버전)
> 공식 자료는 Qwen3.5를 **처음부터 unified VLM**으로 소개한다.  
> "왜 VLM으로 바로 냈는가"에 대한 직접 문장은 보지 못했지만, 공개 포지셔닝상 **VLM이 기본이고 text-only는 옵션**이라고 해석하는 것이 가장 자연스럽다.

---

# 9. DeepStack config와 구현은 있는데, 실제로 적용되는가

## 9-1. Qwen3-VL에서는 DeepStack이 명확하다

Qwen3-VL Technical Report는 **DeepStack integration**을 명시적으로 소개한다. 또 HF `modeling_qwen3_vl.py`에는 `deepstack_visual_embeds`를 decoder hidden states에 주입하는 경로가 있다. `[확인됨][R16][R17]`

즉 **Qwen3-VL public stack에서는 DeepStack이 실제 기능**이다.

---

## 9-2. Qwen3.5 public artifact에서는 무엇이 보이나

Qwen3.5 public checkpoint config에는 `vision_config.deepstack_visual_indexes` 키가 있지만,  
내가 확인한 35B-A3B/122B-A10B 공개 config에서는 이 값이 **빈 배열 `[]`**이다. `[확인됨][R9][R10]`

또 현재 public HF `modeling_qwen3_5_moe.py`와 관련 configuration/modeling 경로에서는 Qwen3-VL처럼 **노골적인 DeepStack runtime path**를 찾지 못했다. `[확인됨/보류][R11]`

---

## 9-3. 보수적 결론

### [확인됨]
- key는 있다
- 값은 비어 있다 (`[]`)
- Qwen3-VL과 달리 public Qwen3.5 HF 코드에서 DeepStack 활성 경로가 뚜렷하지 않다

### [보류]
- 내부/비공개 training stack에서 DeepStack이 어떤 형태로 쓰였는지는 공개 자료만으로 단정하기 어렵다

### 가장 안전한 결론
> **public Qwen3.5 HF artifact 기준으로는 DeepStack이 "실제로 활성화되어 있다"고 보기 어렵다.**  
> **Qwen3-VL에서는 명확히 활성 기능이지만, Qwen3.5에서는 compatibility residue 또는 future hook처럼 보인다.**

---

# 10. Qwen3.5의 native MTP 상세 분석  
(원리 / 아키텍처 / 실제 추론 플로우)

## 10-1. 공식적으로 무엇이 확인되는가

Qwen3.5 model card는 전부 **"MTP: Trained with Multi-steps"**를 명시한다.  
Checkpoint config에는 `mtp_num_hidden_layers: 1`, `mtp_use_dedicated_embeddings: false`가 들어 있다.  
또 vLLM은 아예 `qwen3_5_mtp` 구현을 따로 문서화한다. `[확인됨][R5][R6][R7][R8][R9][R10][R21]`

즉 Qwen3.5는 분명히 **native MTP-aware family**다.

---

## 10-2. MTP의 원리

기본 아이디어는 다음과 같다.

- 일반 decoding: 한 번에 다음 토큰 1개 예측
- MTP: 같은 trunk에서 **다음 여러 토큰을 예측할 수 있는 보조 경로/head**를 함께 학습
- 추론에서는 speculative decoding과 결합해, **여러 토큰을 먼저 제안하고** 검증/수락 과정을 거쳐 inter-token latency를 줄인다. `[확인됨][R20][R21]`

### 개념식
일반 LM:
$$
p(x_{t+1}\mid x_{\le t})
$$

MTP-aware LM:
$$
p(x_{t+1:t+k}\mid x_{\le t})
$$
또는 그것을 근사하는 next-N prediction head를 가진다.

---

## 10-3. 추론 플로우(직관적 도식)

```text
context
  ↓
base trunk
  ↓
MTP branch proposes multiple next tokens
  ↓
main decoding loop / verifier checks acceptability
  ↓
accepted tokens are committed
  ↓
repeat
```

vLLM 문서는 Qwen3.5에 대해 **MTP-1 speculative decoding**을 낮은 동시성 환경에서 latency 절감용으로 권장하고, `num_speculative_tokens`를 설정하는 예시를 제공한다. `[확인됨][R20][R21]`

---

## 10-4. 실제 serving 문서에서 보이는 흔적

Qwen3.5-27B card는 vLLM 예시로 다음 계열 설정을 제시한다.

- `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`

즉 family 차원에서 **MTP-aware serving path**가 실제 사용 문서에 들어 있다. `[확인됨][R7]`

---

## 10-5. HF 기본 forward에서는 얼마나 드러나나

### [확인됨]
- config에는 `mtp_*`가 존재
- vLLM 쪽엔 `qwen3_5_mtp`가 존재 `[R9][R10][R21]`

### [보류]
- public HF 기본 forward에서 native MTP가 full-featured하게 노출되는지는 제한적이다.
- 즉 **가중치와 serving engine 차원에선 분명한 기능**이지만, vanilla HF forward만 보면 전면에 드러나지 않는다.

### 가장 안전한 결론
> **Qwen3.5의 MTP는 "구조적 capability"로는 확실하지만, 실제 활용은 vLLM/SGLang 같은 serving engine에서 더 선명하게 드러난다.**

---

# 11. 비슷한 시기의 Nemotron 3 Super와는 무엇이 다른가

## 11-1. 가장 큰 차이: linear branch의 정체성

| 항목 | Qwen3.5 | Nemotron 3 Super |
|---|---|---|
| hybrid branch | **Gated DeltaNet + Gated Attention** | **Mamba + Transformer** |
| 선형 branch 계보 | DeltaNet / linear transformer | Mamba / selective SSM |
| MoE | sparse MoE + shared expert | LatentMoE |
| MTP | 있음 | 있음 |
| multimodal 포지셔닝 | **VLM family** | agent / high-volume workload 중심 |
| training/serving 강조 | multimodal + long-context + agentic | NVFP4 + throughput/serving efficiency |

Nemotron 3 Super 공식 자료는 이를 **"hybrid Mamba-Transformer model"**, **"LatentMoE"**, **"multi-token prediction"**으로 소개한다. `[확인됨][R22][R23]`

---

## 11-2. 구조 도식 비교

### Qwen3.5 (공개 구조가 상세)
```text
3 × (Gated DeltaNet -> FFN/MoE)
1 × (Gated Attention -> FFN/MoE)
repeat
```

### Nemotron 3 Super (공개 자료 기준 고수준)
```text
Mamba branch
   + Transformer attention branch
   + LatentMoE
   + MTP
```

> **차이:** Qwen3.5는 per-layer rhythm이 public artifact에서 매우 명확한 반면, Nemotron 3 Super는 내가 사용한 공식 자료 범위에서 **고수준 hybrid 설명**이 중심이다. `[확인됨][R22][R23]`

---

## 11-3. 이론적 차이

### Qwen3.5
- linear branch가 `q/k/v`를 유지
- associative memory correction 기반
- attention과 연속적인 설계

### Nemotron 3 Super
- hybrid Mamba-Transformer
- 선형 branch가 selective SSM 계열
- NVIDIA는 NVFP4 pretraining, LatentMoE, MTP, throughput을 강하게 강조 `[확인됨][R22][R23]`

---

## 11-4. 비교 한 줄
> **Qwen3.5는 "DeltaNet-based hybrid", Nemotron 3 Super는 "Mamba-based hybrid"**다.

---

# 12. 전체적인 성능 향상은 어떤가

## 12-1. 공식 benchmark 표에서 가장 깔끔한 비교

Qwen3.5-122B-A10B model card는 Qwen3-235B-A22B와 직접 비교할 수 있는 표를 제공한다. `[확인됨][R6]`

| 벤치마크 | Qwen3-235B-A22B | Qwen3.5-122B-A10B | 차이 |
|---|---:|---:|---:|
| MMLU-Pro | 84.4 | **86.7** | +2.3 |
| SuperGPQA | 64.9 | **67.1** | +2.2 |
| IFEval | 87.8 | **93.4** | +5.6 |
| C-Eval | **92.1** | 91.9 | -0.2 |
| IFBench | 51.7 | **76.1** | +24.4 |
| MultiChallenge | 50.2 | **61.5** | +11.3 |
| AA-LCR | 60.0 | **66.9** | +6.9 |

### 해석
- **instruction following / agentic / interactive benchmark**에서 개선 폭이 특히 크다.
- 일부 전통적 지표(C-Eval)는 사실상 동급이다.
- 즉 성능 향상은 "모든 지표가 다 오른다"보다는, **Qwen3.5가 노린 작업에서 특히 크게 오른다**에 가깝다.

---

## 12-2. 작게 보면?
Qwen3.5-35B-A3B와 27B card들도 agentic benchmark, BFCL, TAU2-Bench, BrowseComp 같은 영역에서 경쟁력 있는 결과를 제시한다. 또한 family 차원에선 "Qwen3 수준의 텍스트 능력"을 유지하면서 multimodal과 agentic 영역을 강화했다고 읽힌다. `[확인됨][R5][R7]`

---

## 12-3. 성능 향상을 어떻게 이해해야 하나

### [확인됨]
- 더 긴 native context
- hybrid token mixer
- MTP
- unified VLM
- 고희소 MoE `[R5][R6][R7][R8][R9][R10][R11]`

### [추론]
이 조합은 단순 perplexity 최적화보다는
- long-context interaction
- instruction following
- agentic loop
- multimodal grounding
쪽에서 더 크게 이득을 준다.

---

# 13. 기타 이슈들

## 13-1. 문서-코드-config 간 노출 정도가 다르다

Qwen3.5는 config에 다음 흔적이 분명하다.

- `layer_types`
- `attn_output_gate`
- `linear_*`
- `mtp_*`
- `router_aux_loss_coef`
- `vision_config.deepstack_visual_indexes`
- `mamba_ssm_dtype`

하지만 이 중 일부는 public HF forward에서 **즉시/완전하게 표면화되지 않는다.**  
예: DeepStack, MTP, 일부 aux loss 경로. `[확인됨/보류][R9][R10][R11]`

### 포인트
> **config에 있다고 해서, public HF 기본 forward에서 완전히 활성화된다고 바로 단정하면 안 된다.**

---

## 13-2. 외부 커널 의존성

Qwen3.5 linear branch는 `flash-linear-attention`과 `causal-conv1d`가 없으면 느린 fallback으로 내려간다. 이는 연구적 구조의 장점이 실제 성능으로 나오려면 **전용 커널이 사실상 중요**하다는 뜻이다. `[확인됨][R11]`

---

## 13-3. 수치 안정성 이슈

HF 구현은 `A_log` 관련 계산에서 float 캐스팅을 강제하며, 그렇지 않으면 fp16에서 문제가 생길 수 있다는 주석을 둔다. 즉 Gated DeltaNet branch는 standard attention보다 **수치적으로 예민한 부분**이 있다. `[확인됨][R11]`

---

## 13-4. 양자화 민감도

Unsloth는 `ssm_out`과 hybrid attention 계열 텐서가 양자화에 특히 민감하다고 지적한다. 따라서 "Qwen3.5를 그냥 기존 attention-only 모델처럼 4-bit 처리해도 잘 된다"는 가정은 위험하다. `[확인됨][R19]`

---

## 13-5. reasoning / template 차이

Qwen3.5 family는 older Qwen3처럼 `/think` `/nothink` 인터페이스를 그대로 쓰지 않고, vLLM reasoning parser 문서에 따르면 `<think>`가 prompt template에 들어가며 generation에서는 주로 `</think>`가 관측되는 식으로 처리된다. Qwen3.5-27B card도 **생각 모드가 기본값**에 가깝다고 설명한다. `[확인됨][R7][R29]`

---

## 13-6. text-only 사용 가능하지만, 기본은 VLM

Qwen3.5는 `--language-model-only` 옵션으로 vision encoder를 건너뛸 수 있다. 즉 배포/실험에서는 text-only처럼 쓸 수 있지만, **family identity는 VLM**이다. `[확인됨][R7]`

---

# 14. 비교 포인트

## A. RMSNorm 비교(수식 + 구현)

| 항목 | Qwen3 | Qwen3.5 |
|---|---|---|
| init | `weight = 1` | `weight = 0` |
| 식 | $\gamma \odot \mathrm{norm}(x)$ | $(1+\gamma)\odot \mathrm{norm}(x)$ |
| branch-specific gate | 없음 | linear branch에서 `* SiLU(z)` |
| 메시지 | standard RMSNorm | 1-centered RMSNorm + gated RMSNorm |

---

## B. Full Attention vs Gated Attention

| 항목 | Full Attention (Qwen3) | Gated Attention (Qwen3.5) |
|---|---|---|
| `q_proj` | query만 생성 | **query + gate** 생성 |
| attention core | softmax attention | softmax attention |
| output | `o_proj(H)` | `o_proj(sigmoid(gate) * H)` |
| 메시지 | standard GQA | output gating이 추가된 full attention |

---

## C. Full Attention vs Gated DeltaNet vs Mamba SSM

| 항목 | Full Attention | Gated DeltaNet | Mamba SSM |
|---|---|---|---|
| `q/k/v` | 있음 | **있음** | 없음 |
| full `QK^T` | 있음 | 없음 | 없음 |
| state 의미 | past KV cache | **compressed KV memory** | latent dynamical state |
| update 식 | softmax weighted sum | **delta correction** | selective state transition |
| 계보 | Transformer | linear transformer / DeltaNet | SSM / Mamba |

---

## D. Sparse MoE 비교(Qwen3 / Qwen3.5 / DeepSeek / GLM)

| 모델 | experts | top-k | shared expert | MTP |
|---|---:|---:|---:|---:|
| Qwen3-235B-A22B | 128 | 8 | 0 | (전면 강조 약함) |
| Qwen3.5-35B-A3B | 256 | 8 + shared 1 | 1 | 1 |
| Qwen3.5-397B-A17B | 512 | 10 + shared 1 | 1 | 1 |
| DeepSeek-V3 | 256 routed | 8 | 1 | 1 |
| GLM-4.5 | 160 routed | 8 | 1 | 1 |

---

## E. Hybrid 구조 비교(Qwen3.5 vs Nemotron 3 Super)

| 항목 | Qwen3.5 | Nemotron 3 Super |
|---|---|---|
| hybrid branch | Gated DeltaNet + Gated Attention | Mamba + Transformer |
| MoE | sparse MoE + shared expert | LatentMoE |
| VLM 기본값 | 예 | 공식 포지셔닝상 아님 |
| MTP | 예 | 예 |
| 핵심 차별점 | DeltaNet 계열 associative memory | Mamba 계열 selective SSM |

---

## F. MTP 추론 플로우 도식

```text
1) context 입력
2) base trunk 실행
3) MTP branch가 여러 next token 제안
4) main decoding / verifier가 acceptance check
5) 받아들인 토큰 commit
6) 다음 step 반복
```

---

# 15. Summary & Insights

## 15-1. 가장 중요한 사실 6개

1. **Qwen3.5는 Qwen3보다 Qwen3-Next와 더 닮아 있다.** `[확인됨][R5][R6][R9][R10][R12]`
2. **핵심 변화는 MoE가 아니라 token mixer다.** full attention-only에서 **Gated DeltaNet + Gated Attention hybrid**로 바뀌었다. `[확인됨][R11]`
3. **Gated DeltaNet은 Mamba가 아니다.** `q/k/v`를 유지하고, delta correction으로 associative memory를 갱신하는 **DeltaNet 계열**이다. `[확인됨][R11][R13][R14][R15]`
4. **MoE는 더 sparse해지고, shared expert가 복귀했다.** routed experts만 쓰던 Qwen3보다 **fallback/dense prior**가 강해졌다. `[확인됨][R1][R5][R6][R8][R11]`
5. **Qwen3.5는 처음부터 unified VLM으로 포지셔닝된다.** text-only는 옵션이지 기본 정체성이 아니다. `[확인됨][R5][R6][R7][R8]`
6. **DeepStack/MTP 일부는 config에 보이지만 public HF 기본 경로에서 완전히 표면화되진 않는다.** "공개 artifact에서 확인되는 범위"와 "설계 흔적"을 구분할 것. `[확인됨/보류][R9][R10][R11][R16][R17][R21]`


## 15-2. 구조적 인사이트: 왜 Qwen3.5는 이름보다 설계로 읽어야 하는가

공개 artifact만 기준으로 보면 Qwen3.5의 핵심 변화는 parameter count보다 **layout과 token mixer의 재정의**에 있다. `layer_types`, `full_attention_interval`, `attn_output_gate`, `linear_*`, `mtp_*`, shared expert 복귀, VLM 기본 포지셔닝이 한 번에 등장하는 순간, 이 모델은 더 이상 "Qwen3의 소폭 개량판"으로 읽히지 않는다. 오히려 **Qwen3-Next에서 보이던 hybrid 설계를 multimodal family 전체로 확장한 세대**로 보는 편이 더 자연스럽다.

이 지점이 중요한 이유는 모델 이름이 아키텍처 계보를 완전히 설명해 주지 않기 때문이다. 제품군 명명 관점에서는 `Qwen3 -> Qwen3.5`가 자연스럽지만, 공개된 구조만 놓고 보면 실제 설계 변화의 중심은 **Transformer-MoE에서 hybrid DeltaNet-Attention-MoE-VLM으로의 전환**이다. 다시 말해, Qwen3.5를 이해하려면 이름보다 `layer_types`와 branch 설계를 먼저 봐야 한다.

## 15-3. 실무 인사이트: 왜 이 변화는 연구보다 서빙에서 더 크게 체감되는가

attention-only 모델에서는 좋은 가중치와 좋은 배포 경험이 비교적 직접적으로 연결되는 경우가 많다. 하지만 Qwen3.5는 Gated DeltaNet, depthwise causal conv, recurrent state, fused gated RMSNorm, sparse MoE, MTP를 함께 묶은 구조라서, 모델의 품질만으로는 실제 체감 성능을 설명하기 어렵다. **커널 생태계, 캐시 형식, 양자화 전략, speculative decoding 지원 여부**가 곧바로 실사용 효율로 이어진다.

즉 Qwen3.5는 단순히 "더 똑똑한 모델"이 아니라, **백엔드와 함께 읽어야 하는 모델**이다. Unsloth의 커널/양자화 민감도 이슈, vLLM의 MTP 지원 경로, HF 구현의 fast path 의존성은 모두 같은 방향을 가리킨다. 앞으로 hybrid 계열 모델을 비교할 때는 benchmark 점수만이 아니라 **전용 커널 요구사항, recurrent state 관리, serving engine 지원 상태**까지 함께 보는 것이 더 정확하다.

## 15-4. 앞으로 이 계열 모델을 읽을 때 먼저 볼 것

앞으로 Qwen 계열이나 유사 hybrid 모델을 추적할 때는 "몇 B냐"보다 **무엇이 token mixer인지**, **shared expert가 있는지**, **MTP가 config 수준인지 serving 수준인지**, **multimodal hook가 실제 runtime path에 연결되어 있는지**를 먼저 확인하는 편이 더 유익하다.

Qwen3.5 사례는 공개 model card와 HF config만으로도 연구 논문 못지않게 많은 설계 의도를 읽어낼 수 있다는 점을 잘 보여준다. 동시에, config에 키가 있다고 해서 public forward에서 반드시 활성 기능이라고 단정하면 안 된다는 교훈도 함께 준다.

---

# 16. FAQ

## 16-1. "Qwen3.5의 linear attention은 Mamba인가요?"
**답:** 아니다. runtime 특성은 비슷할 수 있지만, 모델적으로는 `q/k/v`를 유지하는 **Gated DeltaNet**이다.

## 16-2. "가장 큰 구조 변화는 뭐죠?"
**답:** MoE보다 먼저 **token mixer의 하이브리드화**다.

## 16-3. "Qwen3.5는 왜 갑자기 VLM이 됐나요?"
**답:** 공식 자료는 Qwen3.5를 처음부터 **Unified Vision-Language Foundation**으로 소개한다. "왜"에 대한 직접 문장은 못 봤지만, 공개 포지셔닝은 VLM이 기본값이다.

## 16-4. "DeepStack은 실제로 들어가 있나요?"
**답:** Qwen3-VL에선 확실하다. Qwen3.5 public artifact에선 key는 있지만 indexes가 비어 있어, **활성화 증거는 약하다**.

## 16-5. "MTP는 HF에서 바로 쓰나요?"
**답:** config와 family 수준 capability는 확실하지만, 실제 활용은 **vLLM/SGLang 같은 serving engine**에서 더 선명하게 드러난다.

---

# 참고문헌(공식/1차 자료)

## Qwen / Hugging Face / 공식 구현
- [R1] Qwen3 Technical Report - https://arxiv.org/abs/2505.09388  
- [R2] Qwen3-235B-A22B model card - https://huggingface.co/Qwen/Qwen3-235B-A22B  
- [R3] Qwen3-235B-A22B config - https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json  
- [R4] HF `modeling_qwen3_moe.py` - https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py  
- [R5] Qwen3.5-35B-A3B model card - https://huggingface.co/Qwen/Qwen3.5-35B-A3B  
- [R6] Qwen3.5-122B-A10B model card - https://huggingface.co/Qwen/Qwen3.5-122B-A10B  
- [R7] Qwen3.5-27B model card - https://huggingface.co/Qwen/Qwen3.5-27B  
- [R8] Qwen3.5-397B-A17B model card - https://huggingface.co/Qwen/Qwen3.5-397B-A17B  
- [R9] Qwen3.5-35B-A3B config - https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json  
- [R10] Qwen3.5-122B-A10B config - https://huggingface.co/Qwen/Qwen3.5-122B-A10B/blob/main/config.json  
- [R11] HF `modeling_qwen3_5_moe.py` - https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py  
- [R12] Qwen3-Next-80B-A3B-Instruct model card - https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct  
- [R16] Qwen3-VL Technical Report - https://arxiv.org/abs/2511.21631  
- [R17] HF `modeling_qwen3_vl.py` - https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py  
- [R28] HF Qwen3.5 doc page (`under construction`) - https://huggingface.co/docs/transformers/model_doc/qwen3_5_moe  

## DeltaNet / Mamba / 관련 논문
- [R13] Parallelizing Linear Transformers with the Delta Rule over Sequence Length - https://arxiv.org/abs/2406.06484  
- [R14] Gated Delta Networks: Improving Mamba2 with Delta Rule - https://openreview.net/forum?id=r8H7xhYPwz  
- [R15] Mamba: Linear-Time Sequence Modeling with Selective State Spaces - https://arxiv.org/abs/2312.00752  

## 서빙 / 파인튜닝 / 생태계
- [R18] Unsloth Qwen3.5 Fine-tuning Guide - https://unsloth.ai/docs/models/qwen3.5/fine-tune  
- [R19] Unsloth GGUF / quantization notes - https://unsloth.ai/docs/models/qwen3.5/benchmarks  
- [R20] vLLM Qwen3.5 Usage Guide - https://docs.vllm.ai/en/latest/models/supported_models.html#qwen3-5  
- [R21] vLLM `qwen3_5_mtp` docs - https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/qwen3_5_mtp/  
- [R29] vLLM reasoning parser / Qwen3.5 notes - https://docs.vllm.ai/en/latest/features/reasoning_outputs.html  

## 비교 모델
- [R22] NVIDIA Nemotron 3 Super official page - https://research.nvidia.com/labs/nemotron/Nemotron-3-Super/  
- [R23] NVIDIA Nemotron 3 family paper / recipe pages - https://developer.nvidia.com/blog/take-llm-performance-to-the-next-level-with-nemotron-3/  
- [R24] DeepSeek-V3 model card - https://huggingface.co/deepseek-ai/DeepSeek-V3  
- [R25] DeepSeek-V3 config - https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json  
- [R26] GLM-4.5 model card - https://huggingface.co/zai-org/GLM-4.5  
- [R27] GLM-4.5 config - https://huggingface.co/zai-org/GLM-4.5/blob/main/config.json  

---

# 부록 A. "이 슬라이드/문서에서 꼭 남겨야 하는 한 장 요약"

```text
Qwen3
= Standard Transformer-MoE
= Full attention everywhere
= 128 experts / 8 active
= shared expert removed

Qwen3.5
= Hybrid Gated DeltaNet + Gated Attention
= 3 linear + 1 full repeating
= 256~512 experts, shared expert restored
= 262K native context, MTP, unified VLM
= structurally closer to Qwen3-Next than to Qwen3
```

---

# 부록 B. 발표 자료로 바꿀 때 추천 장표 순서

1. 한 장 요약: Qwen3 -> Qwen3.5 핵심 변화  
2. 구조 도식: Qwen3 vs Qwen3.5 vs Nemotron  
3. RMSNorm / Gated RMSNorm 비교  
4. Full Attention vs Gated Attention 비교  
5. Gated DeltaNet line-by-line + 수식  
6. Full Attention vs Gated DeltaNet vs Mamba  
7. MoE 비교표(Qwen3 / Qwen3.5 / DeepSeek / GLM)  
8. MTP 원리 + serving flow  
9. DeepStack / VLM / 공개 artifact의 활성화 범위  
10. 성능 표 + 기타 이슈  



# 부록 C. "직접 확인용" 짧은 원문 / snippet 모음  
> 저작권 이슈를 피하기 위해 **아주 짧은 식별자/한 줄 수준**만 실었다.

## C-1. Qwen3.5 config에서 바로 보이는 핵심 키
```text
attn_output_gate: true
full_attention_interval: 4
mtp_num_hidden_layers: 1
vision_config.deepstack_visual_indexes: []
```
근거: Qwen3.5-35B-A3B / 122B-A10B 공개 config `[R9][R10]`

## C-2. Qwen3.5 layer layout의 핵심 패턴
```text
linear_attention, linear_attention, linear_attention, full_attention
```
근거: `layer_types` in Qwen3.5 공개 config `[R9][R10]`

## C-3. Qwen3.5 attention 구현의 핵심 한 줄
```python
attn_output = attn_output * torch.sigmoid(gate)
```
근거: HF `modeling_qwen3_5_moe.py` `[R11]`

## C-4. Qwen3.5 RMSNorm의 핵심 한 줄
```python
output = output * (1.0 + self.weight.float())
```
근거: HF `modeling_qwen3_5_moe.py` `[R11]`

## C-5. Qwen3.5 gated norm의 핵심 한 줄
```python
hidden_states = hidden_states * F.silu(gate.to(torch.float32))
```
근거: HF `modeling_qwen3_5_moe.py` `[R11]`

## C-6. Qwen3.5 Gated DeltaNet의 decay 계산 핵심 한 줄
```python
g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
```
근거: HF `modeling_qwen3_5_moe.py` `[R11]`

## C-7. Qwen3-VL에서 DeepStack이 실제로 보이는 식별자
```text
deepstack_visual_embeds
```
근거: HF `modeling_qwen3_vl.py` `[R17]`
