---
layout: single
title:  "LLM Train Recipe"
categories: Study-concept
tag: [LLM, Train Recipe, Tulu, SmolLM2, Dataset, Fine-tuning]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
typora-root-url: ../
---





#  LLM Train Recipe

![Logo](/images/2025-03-06-LLM-Train-Recipe/logo.png){: .align-center}



👉🏻[Tulu3 paper link](https://arxiv.org/pdf/2411.15124)

👉🏻[SmolLM2 paper link](https://arxiv.org/html/2502.02737v1)

최근 AI 연구 커뮤니티에서는 모든 학습 데이터와 파라미터를 투명하게 공개하는 모델들이 주목받고 있다.
**Tülu 3**는 Llama 3.1 기반의 post-trained models로, base 모델을 가져와서 지도학습(SFT), Preference 튜닝(DPO), 그리고 (8B 모델에 한정한) 강화학습(RLVR) 단계를 통해 성능을 극대화했다.
**SmolLM2**는 Pre-training부터 Fine-tuning까지 전 과정을 오픈 데이터로 진행한 모델로, Pre-training 데이터와 Fine-tuning 데이터가 모두 공개되어 있는 초경량 영어 모델이다.
본 리뷰에서는 두 모델의 주요 구성 요소를 세세하게 살펴보고, 각 단계마다 dataset, method, hyper-parameters를 비교해 본다.



# Model Overviews

**Tülu 3**와 **SmolLM2**의 모델 구조에 대해 간략히 정리해본다.



## Tülu 3

- **Base models:**
  - Llama 3.1 기반
  - Model parameters: 8B, 70B, 405B

- **Key points:**

  - 자체 구축한 post-train dataset으로 SFT, DPO, RLVR을 단계별로 적용
  - 각 post-train 단계에서의 hyper-parameters 및 reconstruction script 공개

  - 각 단계별로 체크포인트가 공개되어 있어, 연구자들이 쉽게 재현 및 확장이 가능



## SmolLM2

- **Base models:**
  - Llama(>3.1) Architectures 기반
  - Model parameters: 135M, 360M, 1.7B
  - Pre-train부터 진행

- **Key points:**
  - 자체 구축한 pre-train, post-train dataset으로 학습
  - 각 학습 단계에서의 hyper-parameters 공개




# Data Overviews

**Tülu 3**와 **SmolLM2**의 학습 데이터셋에 대해 알아본다.



## Tülu 3

- **Supervised Fine-Tuning (SFT) 데이터:**
  - 공개 인스트럭션 데이터셋과 synthetic 데이터의 혼합
  - 다양한 태스크(대화, 수학, 코딩, 안전성 등)를 포함
  - 👉🏻[allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)

- **Preference Tuning(DPO) 데이터:**
  - SFT 단계의 출력과 타 모델의 응답을 비교하여 구성된 on-policy 데이터
  - 👉🏻[allenai/llama-3.1-tulu-3-405b-preference-mixture](https://huggingface.co/datasets/allenai/llama-3.1-tulu-3-405b-preference-mixture)
  - 👉🏻[allenai/llama-3.1-tulu-3-70b-preference-mixture](https://huggingface.co/datasets/allenai/llama-3.1-tulu-3-70b-preference-mixture)
  - 👉🏻[allenai/llama-3.1-tulu-3-8b-preference-mixture](https://huggingface.co/datasets/allenai/llama-3.1-tulu-3-8b-preference-mixture)

- **RLVR(PPO) 데이터 (8B 모델 전용):**
  - GSM8K, MATH, IFEval 등 "검증 가능한 정답" 여부에 따른 보상 체계 데이터
  - 👉🏻[allenai/RLVR-GSM-MATH-IF-Mixed-Constraints](https://huggingface.co/datasets/allenai/RLVR-GSM-MATH-IF-Mixed-Constraints)



## SmolLM2

- **Pre-training 데이터:**
  - **구성:**
    - **FineWeb-Edu**: 1.3조 토큰 규모의 필터링된 교육 콘텐츠 기반 웹 데이터셋  
    - **DCLM (DataComp-LM baseline)**: 300조+ 원시(raw) 토큰에서 필터링한 CommonCrawl 데이터셋  
    - **The Stack**: 오픈소스 코드 데이터셋  
    - **FineMath**: 새롭게 구축한 수학 데이터셋  
    - **Stack-Edu**: 코딩 학습을 위한 필터링된 코드 데이터  
  - **요약**

| **Models**       | **Sources**                                       | **Amounts** |
|------------------|---------------------------------------------------|-------------|
| **SmolLM2-135M** | FineWeb-Edu, DCLM, The Stack, FineMath, Stack-Edu | ~2T tokens  |
| **SmolLM2-360M** | FineWeb-Edu, DCLM, The Stack, FineMath, Stack-Edu | ~4T tokens  |
| **SmolLM2-1.7B** | FineWeb-Edu, DCLM, The Stack, FineMath, Stack-Edu | ~11T tokens |

- **Supervised Fine-Tuning (SFT) 데이터:**
  - **구성:**
    - **SmolTalk**: LLaMA 3.1 기반의 Instruction-Tuning 데이터셋으로, 공개 데이터 및 자체 생성 데이터(Smol-Magpie-Ultra, Smol-Rewrite 등)를 포함  
    - **UltraFeedback**: GPT-4가 채점한 64k 프롬프트 및 선호도 비교 데이터를 포함하여 학습
  - **요약**

| **Models**       | **SFT(Supervised Fine-Tuning)**       | **DPO(Direct Preference Optimization)** | **Sources**            |
|------------------|---------------------------------------|-----------------------------------------|------------------------|
| **SmolLM2-135M** | SmolTalk(filtered, 약 0.5M 개 samples) | UltraFeedback (~61k prompts pair)      | SmolTalk, UltraFeedback |
| **SmolLM2-360M** | SmolTalk(filtered, 약 0.5M 개 samples) | UltraFeedback (~61k prompts pair)      | SmolTalk, UltraFeedback |
| **SmolLM2-1.7B** | SmolTalk(entire, 약 1.1M 개)           | UltraFeedback (~61k prompts pair)      | SmolTalk, UltraFeedback |

  - **추가 정보:**
    - 데이터셋 소스 및 정제 관련 세부 정보는 리포트 부록 및 공개 저장소에서 확인 가능
    - 👉🏻[smollm2 Dataset Repository](https://github.com/smollm2/dataset)




## 학습 데이터 비교 및 공통점

- **공개성:**
  - 두 모델 모두 모든 데이터셋을 공개하여, 연구자들이 동일 조건에서 실험 재현 및 커스터마이징이 가능하도록 지원

- **데이터 믹싱 전략:**

  - Tülu 3는 포스트 트레이닝을 위한 단계별 데이터 큐레이션에 집중

  - SmolLM2는 대규모 Pre-training 데이터를 기반으로 언어 모델의 범용성을 극대화한 후, 태스크별 Fine-tuning 데이터로 성능을 보완

- **목표:**
  - 두 모델 모두 다양한 태스크를 커버하기 위해, 여러 도메인과 문체의 데이터를 혼합하는 전략을 취함



# Train Pipelines & Hyper-Parameters

**Tülu 3**와 **SmolLM2**의 학습 파이프라인 및 하이퍼 파라미터에 대해 알아본다.



## Tülu 3

👉🏻[Tülu 3 scripts](https://github.com/allenai/open-instruct/blob/main/docs/tulu3.md)

 **Tülu 3**의 경우 위 링크에 각 모델 파라미터, 그리고 단계별 reproduction script가 자세히 나와있으며, 아래 내용은 해당 링크의 내용을 옮겨 적은 것이다.



### SFT

**Llama-3.1-Tulu-3-8B-SFT Reproduction**

Below is (almost) the exact command which produced [Llama-3.1-Tulu-3-8B-SFT](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-SFT). We deployed the command across 8 machines, each equipped with 8 NVIDIA H100 GPUs, for a total of 64 GPUs in the our setup.

```bash
# modify the following `MACHINE_RANK`, `MAIN_PROCESS_IP`,
# `NUM_MACHINES`, `NUM_PROCESSES`, `PER_DEVICE_TRAIN_BATCH_SIZE`,
# `GRADIENT_ACCUMULATION_STEPS` according to your setup
MACHINE_RANK=0
MAIN_PROCESS_IP=localhost
NUM_MACHINES=8
NUM_PROCESSES=64
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=2
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 8 \
    --num_processes 64 \
    --machine_rank $MACHINE_RANK \
    --main_process_ip $MAIN_PROCESS_IP \
    --main_process_port 29400 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard open_instruct/finetune.py \
    --model_name_or_path meta-llama/Llama-3.1-8B \
    --tokenizer_name meta-llama/Llama-3.1-8B \
    --use_slow_tokenizer \
    --use_flash_attn \
    --max_seq_length 4096 \
    --preprocessing_num_workers 128 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate 5e-06 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --output_dir output/sft_8b \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --reduce_loss sum \
    --model_revision main \
    --dataset_mixer_list allenai/tulu-3-sft-mixture 1.0 \
    --checkpointing_steps epoch \
    --dataset_mix_dir output/sft_8b \
    --exp_name tulu-3-8b-sft \
    --seed 123
# For Ai2 internal members, this was the experiment URL: https://beaker.org/ex/01JBNTPW8TKG09B2XR832YB5S8
```



**Llama-3.1-Tulu-3-70B-SFT Reproduction**

This is (almost) the exact command which produced [allenai/Llama-3.1-Tulu-3-70B-SFT](https://huggingface.co/allenai/Llama-3.1-Tulu-3-70B-SFT)

```bash
# modify the following `MACHINE_RANK`, `MAIN_PROCESS_IP`,
# `NUM_MACHINES`, `NUM_PROCESSES`, `PER_DEVICE_TRAIN_BATCH_SIZE`,
# `GRADIENT_ACCUMULATION_STEPS` according to your setup
MACHINE_RANK=0
MAIN_PROCESS_IP=localhost
NUM_MACHINES=8
NUM_PROCESSES=64
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=2
accelerate launch \
    --mixed_precision bf16 \
    --num_machines $NUM_MACHINES \
    --num_processes $NUM_PROCESSES \
    --machine_rank $MACHINE_RANK \
    --main_process_ip $MAIN_PROCESS_IP \
    --main_process_port 29400 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard open_instruct/finetune.py \
    --model_name_or_path meta-llama/Llama-3.1-70B \
    --tokenizer_name meta-llama/Llama-3.1-70B \
    --use_slow_tokenizer \
    --use_flash_attn \
    --max_seq_length 4096 \
    --preprocessing_num_workers 128 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate 2e-06 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --output_dir output/sft_70B \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --reduce_loss sum \
    --model_revision main \
    --dataset_mixer_list allenai/tulu-3-sft-mixture 1.0 \
    --dataset_mix_dir output/sft_70B \
    --checkpointing_steps 1000 \
    --keep_last_n_checkpoints 20 \
    --gradient_checkpointing \
    --exp_name tulu-3-70b-sft \
    --seed 456
# For Ai2 internal members, this was the experiment URL: https://beaker.org/ex/01JC5J4R80M18XQTDH47JSFRJY/
```



### DPO

**Llama-3.1-Tulu-3-8B-DPO Reproduction**

This is (almost) the exact command which produced [allenai/Llama-3.1-Tulu-3-8B-DPO](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-DPO)

```bash
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf open_instruct/dpo_tune.py \
    --model_name_or_path allenai/Llama-3.1-Tulu-3-8B-SFT \
    --use_flash_attn \
    --tokenizer_name allenai/Llama-3.1-Tulu-3-8B-SFT \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-07 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --output_dir output/dpo_8b \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --model_revision main \
    --gradient_checkpointing \
    --dataset_mixer_list allenai/llama-3.1-tulu-3-8b-preference-mixture 1.0 \
    --use_slow_tokenizer \
    --use_lora False \
    --dpo_loss_type dpo_norm \
    --dpo_beta 5 \
    --checkpointing_steps 1000 \
    --exp_name tulu-3-8b-dpo
# For Ai2 internal members, this was the experiment URL: https://beaker.org/ex/01JCRXP0AR5312S8MD3XGCN0J7/
```



**Llama-3.1-Tulu-3-70B-DPO Reproduction**

This is (almost) the exact command which produced [allenai/Llama-3.1-Tulu-3-70B-DPO](https://huggingface.co/allenai/Llama-3.1-Tulu-3-70B-DPO)

```bash
# modify the following `MACHINE_RANK`, `MAIN_PROCESS_IP`,
# `NUM_MACHINES`, `NUM_PROCESSES`, `PER_DEVICE_TRAIN_BATCH_SIZE`,
# `GRADIENT_ACCUMULATION_STEPS` according to your setup
MACHINE_RANK=0
MAIN_PROCESS_IP=localhost
NUM_MACHINES=8
NUM_PROCESSES=64
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=2
accelerate launch \
    --mixed_precision bf16 \
    --num_machines $NUM_MACHINES \
    --num_processes $NUM_PROCESSES \
    --machine_rank $MACHINE_RANK \
    --main_process_ip $MAIN_PROCESS_IP \
    --main_process_port 29400 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard open_instruct/dpo_tune_cache.py \
    --model_name_or_path allenai/Llama-3.1-Tulu-3-70B-SFT \
    --tokenizer_name allenai/Llama-3.1-Tulu-3-70B-SFT \
    --use_flash_attn \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate 2e-07 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --output_dir output/dpo_70b \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --model_revision main \
    --gradient_checkpointing \
    --dataset_mixer_list allenai/llama-3.1-tulu-3-70b-preference-mixture \
    --use_slow_tokenizer \
    --use_lora False \
    --dpo_loss_type dpo_norm \
    --dpo_beta 5 \
    --checkpointing_steps epoch \
    --exp_name tulu-3-70b-dpo
# For Ai2 internal members, this was the experiment URL: https://beaker.org/ex/01JCSAYYHQYF9QDQDCV6KJ53M9/
```



**Llama-3.1-Tulu-3-405B-DPO Reproduction**

This is (almost) the exact command which produced [allenai/Llama-3.1-Tulu-3-405B-DPO](https://huggingface.co/allenai/Llama-3.1-Tulu-3-405B-DPO)

```bash
# modify the following `MACHINE_RANK`, `MAIN_PROCESS_IP`,
# `NUM_MACHINES`, `NUM_PROCESSES`, `PER_DEVICE_TRAIN_BATCH_SIZE`,
# `GRADIENT_ACCUMULATION_STEPS` according to your setup
MACHINE_RANK=0
MAIN_PROCESS_IP=localhost
NUM_MACHINES=8
NUM_PROCESSES=64
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=2
accelerate launch --mixed_precision bf16 \
    --num_machines 32 \
    --num_processes 256 \
    --machine_rank $BEAKER_REPLICA_RANK \
    --main_process_ip $BEAKER_LEADER_REPLICA_HOSTNAME \
    --main_process_port 29400 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard open_instruct/dpo_tune_cache.py \
    --model_name_or_path allenai/Llama-3.1-Tulu-3-405B-SFT \
    --tokenizer_name allenai/Llama-3.1-Tulu-3-70B-SFT \
    --use_flash_attn \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-07 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --output_dir output_405b \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --model_revision main \
    --gradient_checkpointing \
    --dataset_mixer_list ai2-adapt-dev/405b_preference_mix 1.0 \
    --use_slow_tokenizer \
    --use_lora False \
    --dpo_loss_type dpo_norm \
    --dpo_beta 5 \
    --checkpointing_steps 1000
# For Ai2 internal members, this was the experiment URL: https://beaker.org/ex/01JJ4QRZ31SH79AHVM6WWDVJB4/
```



### RLVR

**Llama-3.1-Tulu-3-8B-RM Reproduction**

This is (almost) the exact command which produced [allenai/Llama-3.1-Tulu-3-8B-RM](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-RM)

```bash
accelerate launch \
    --config_file configs/ds_configs/deepspeed_zero3.yaml open_instruct/reward_modeling.py \
    --dataset_mixer '{"allenai/llama-3.1-tulu-3-8b-preference-mixture": 1.0}' \
    --dataset_train_splits train \
    --dataset_eval_mixer '{"allenai/ultrafeedback_binarized_cleaned": 1.0}' \
    --dataset_eval_splits test_prefs \
    --model_name_or_path allenai/Llama-3.1-Tulu-3-8B-SFT \
    --chat_template tulu \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --num_train_epochs 1 \
    --output_dir output/rm_8b \
    --gradient_checkpointing \
    --push_to_hub \
    --with_tracking
# For Ai2 internal members, this was the experiment URL: https://beaker.org/ex/01JCS01RFBQGFE5F1W3W96FFVM/
```



**Llama-3.1-Tulu-3-8B Reproduction**

This is (almost) the exact command which produced [allenai/Llama-3.1-Tulu-3-8B](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B)

```bash
python open_instruct/ppo_vllm_thread_ray_gtrl.py \
    --exp_name tulu-3-8b-rlvr \
    --dataset_mixer_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 16 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --model_name_or_path allenai/Llama-3.1-Tulu-3-8B-DPO \
    --reward_model_path allenai/Llama-3.1-Tulu-3-8B-RM \
    --non_stop_penalty \
    --stop_token eos \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --chat_template_name tulu \
    --sft_messages_key messages \
    --learning_rate 3e-7 \
    --total_episodes 10000000 \
    --penalty_reward_value -10.0 \
    --deepspeed_stage 3 \
    --per_device_train_batch_size 2 \
    --local_rollout_forward_batch_size 2 \
    --local_mini_batch_size 32 \
    --local_rollout_batch_size 32 \
    --actor_num_gpus_per_node 7 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.05 \
    --apply_verifiable_reward true \
    --output_dir output/rlvr_8b \
    --seed 3 \
    --num_evals 3 \
    --save_freq 100 \
    --reward_model_multiplier 0.0 \
    --gradient_checkpointing \
    --with_tracking
# For Ai2 internal members, this was the experiment URL: https://beaker.org/ex/01JCVTA10BQDVGGQKFYWEZ6KCQ/
```



**Llama-3.1-Tulu-3-70B Reproduction**

This is (almost) the exact command which produced [allenai/Llama-3.1-Tulu-3-70B](https://huggingface.co/allenai/Llama-3.1-Tulu-3-70B)

Couple of notes:
* Make sure to modify `configs/beaker_configs/ray_node_setup.sh` in our own cluster setup. The idea is to have the replicas join the main machines via `ray`.
* We had to use `--vllm_tensor_parallel_size 4` because `--vllm_tensor_parallel_size 8` errors out for some strange reason. This is a temporary workaround.
* Here the effective batch size is `sum(actor_num_gpus_per_node) * local_mini_batch_size = 40 * 16 = 640`. If you have less GPUs, you can adjust `actor_num_gpus_per_node` and `local_mini_batch_size` accordingly.

```bash
source configs/beaker_configs/ray_node_setup.sh && python open_instruct/ppo_vllm_thread_ray_gtrl.py \
    --dataset_mixer_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 16 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --model_name_or_path allenai/Llama-3.1-Tulu-3-70B-DPO \
    --exp_name tulu-3-70b-rlvr \
    --reward_model_path allenai/Llama-3.1-Tulu-3-8B-RM \
    --beta 0.07 \
    --warmup_ratio 0.1 \
    --seed 8 \
    --output_dir output/rlvr_70b \
    --non_stop_penalty \
    --stop_token eos \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --chat_template_name tulu \
    --sft_messages_key messages \
    --learning_rate 1e-7 \
    --total_episodes 400000 \
    --penalty_reward_value -10.0 \
    --deepspeed_stage 3 \
    --per_device_train_batch_size 1 \
    --local_rollout_forward_batch_size 1 \
    --local_mini_batch_size 16 \
    --local_rollout_batch_size 16 \
    --actor_num_gpus_per_node 8 8 8 8 8 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 4 \
    --apply_verifiable_reward true \
    --reward_model_multiplier 0.0 \
    --no_gather_whole_model \
    --num_evals 3 \
    --save_freq 40 \
    --gradient_checkpointing \
    --with_tracking
# For Ai2 internal members, this was the experiment URL: https://beaker.org/ex/01JD3YEM4XGH2F2H10Y49GK441/
```



**Llama-3.1-Tulu-3-405B Reproduction**

This is (almost) the exact command which produced [allenai/Llama-3.1-Tulu-3-405B](https://huggingface.co/allenai/Llama-3.1-Tulu-3-405B)

Couple of notes:
* We had to set `TORCH_NCCL_ENABLE_MONITORING=0` to turn off NCCL heartbeat monitoring and avoid timeouts. Feel free to remove this.
* Make sure to modify `configs/beaker_configs/ray_node_setup.sh` in our own cluster setup. The idea is to have the replicas join the main machines via `ray`.
* Here the effective batch size is `sum(actor_num_gpus_per_node) * local_mini_batch_size = 40 * 16 = 640`. If you have less GPUs, you can adjust `actor_num_gpus_per_node` and `local_mini_batch_size` accordingly.

```bash
TORCH_NCCL_ENABLE_MONITORING=0 python mason.py \
    --cluster ai2/jupiter-cirrascale-2 --pure_docker_mode \
    --workspace ai2/tulu-3-dev \
    --priority urgent \
    --preemptible \
    --num_nodes 32 \
    --image nathanl/open_instruct_auto \
    --budget ai2/oe-adapt \
    --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh \&\& TORCH_DISTRIBUTED_DEBUG=DETAIL python open_instruct/ppo_vllm_thread_ray_gtrl.py \
    --dataset_mixer_list allenai/RLVR-MATH 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list allenai/RLVR-MATH 128 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 1024 \
    --model_name_or_path /weka/oe-adapt-default/hamishi/405b_dpo_v4 \
    --exp_name "405b_rlvr_math_only_8b_valu_on_v4" \
    --reward_model_path allenai/Llama-3.1-Tulu-3-8B-RM \
    --beta 0.05 \
    --output_dir "/weka/oe-adapt-default/hamishi/405b_rlvr_math_only_8b_valu_on_v4" \
    --non_stop_penalty \
    --stop_token eos \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --chat_template tulu \
    --sft_messages_key messages \
    --learning_rate 1e-7 \
    --total_episodes 400000 \
    --num_epochs 4 \
    --penalty_reward_value -10.0 \
    --deepspeed_stage 3 \
    --per_device_train_batch_size 1 \
    --local_rollout_forward_batch_size 1 \
    --local_mini_batch_size 8 \
    --local_rollout_batch_size 8 \
    --actor_num_gpus_per_node 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 16 \
    --vllm_enforce_eager true \
    --apply_verifiable_reward true \
    --reward_model_multiplier 0.0 \
    --no_gather_whole_model \
    --seed 3 \
    --num_evals 3 \
    --no_try_launch_beaker_eval_jobs \
    --save_freq 25 \
    --try_launch_beaker_eval_jobs_on_weka \
    --gradient_checkpointing \
    --with_tracking
# For Ai2 internal members, this was the experiment URL: https://beaker.org/ex/01JJA31S20XAFR82YPFKSMMYZV/
```



## SmolLM2

Pre-training, Fine-tuning에 대해 표로 정리해본다.



### Pre-training

모든 SmolLM2 모델은 **AdamW Optimizer**와 **Warmup-Stable-Decay(WSD) LR Scheduler**를 사용하여 학습시켰다.

- **요약**

| **Models** | **Trained Tokens** | **Optimizer** | **Learning Rate(Peak)** | **LR Scheduler**    | **Warmup Steps** | **Sequence Length** | **Global Batch Size** |
|------------|--------------------|---------------|-------------------------|---------------------|------------------|---------------------|-----------------------|
| **135M**   | ~2T                | AdamW         | ~3.0×10^-3              | WSD (Stable, Decay) | 2000             | 2048                | 2M tokens             |
| **360M**   | ~4T                | AdamW         | ~3.0×10^-3              | WSD (Stable, Decay) | 2000             | 2048                | 2M tokens             |
| **1.7B**   | ~11T               | AdamW         | ~5.0×10^-4              | WSD (Stable, Decay) | 2000             | 2048                | 2M tokens             |

- **설명**
  - AdamW (β₁=0.9, β₂=0.999) Optimizer 사용  
  - LR Scheduling: **초기 2000 스텝 Warmup -> 일정 유지(Stable) -> 마지막 10% Decay**  
  - 1.7B 모델은 **4단계 학습(0-6T, 6-8T, 8-10T, 10-11T) 진행**  
  - 135M 및 360M 모델은 **수학 및 코딩 데이터가 후반부에 집중적으로 투입됨**  



### Supervised Fine-Tuning(SFT)

SmolTalk 데이터셋을 사용하여 Instruction-Following 능력을 학습시켰다.

- **요약**

| **Models** | **Trained Samples** | **Optimizer** | **Learning Rate(Peak)** | **LR Scheduler**    | **Epochs**  | **Sequence Length** | **Global Batch Size** |
|------------|---------------------|---------------|-------------------------|---------------------|-------------|---------------------|-----------------------|
| **135M**   | ~0.5M               | AdamW         | ~3.0×10^-4              | WSD (Stable, Decay) | 2           | 8192                | 128                   |
| **360M**   | ~0.5M               | AdamW         | ~3.0×10^-4              | WSD (Stable, Decay) | 2           | 8192                | 128                   |
| **1.7B**   | ~1.1M               | AdamW         | ~3.0×10^-4              | WSD (Stable, Decay) | 2           | 8192                | 128                   |



### Direct Preference Optimization(DPO)

UltraFeedback 데이터셋을 사용하여 모델 출력을 인간 선호도에 맞게 최적화시켰다.

- **요약**

| **Models** | **DPO Pairs** | **Optimizer** | **Learning Rate(Peak)** | **LR Scheduler**    | **Epochs**  | **Sequence Length** | **Global Batch Size** | **DPO beta** |
|------------|---------------|---------------|-------------------------|---------------------|-------------|---------------------|-----------------------|--------------|
| **135M**   | ~61k          | AdamW         | ~1.0×10^-6              | WSD (Stable, Decay) | 2           | 1024                | 128                   | 0.5          |
| **360M**   | ~61k          | AdamW         | ~1.0×10^-6              | WSD (Stable, Decay) | 2           | 1024                | 128                   | 0.5          |
| **1.7B**   | ~61k          | AdamW         | ~1.0×10^-6              | WSD (Stable, Decay) | 2           | 1024                | 128                   | 0.5          |



## 비교 및 인사이트

- **Tülu 3:**

  - Llama 3.1 기반의 모델에 대해 후처리(post-training)로 SFT, Preference 튜닝, RLVR 단계를 적용

  - 모델 크기에 따른 세밀한 파라미터 조정이 돋보임

- **SmolLM2:**

  - Pre-training부터 시작하여, 대규모 데이터로 범용 언어 모델을 구축한 후 Fine-tuning으로 특정 태스크에 맞춤

  - Pre-training 단계의 데이터 규모(약 50B 토큰)와 긴 학습 스텝이 특징

- **공통점:**

  - 모든 단계에서 AdamW optimizer 사용, warm up 및 weight decay 적용

  - 학습 데이터와 하이퍼 파라미터가 전 과정 공개되어 있음

  - 모델의 파라미터 사이즈, 각 학습 단계에 따라 하이퍼 파라미터 조정이 있음



# Model Checkpoints Information

**Tülu 3**와 **SmolLM2**의 모델 체크포인트들을 기록함.



## Tülu 3

- **8B:**

  - SFT: `allenai/Llama-3.1-Tulu-3-8B-SFT`

  - DPO: `allenai/Llama-3.1-Tulu-3-8B-DPO`

  - RLVR 최종: `allenai/Llama-3.1-Tulu-3-8B`

  - 보상 모델(RM): `allenai/Llama-3.1-Tulu-3-8B-RM`

- **70B:**

  - SFT: `allenai/Llama-3.1-Tulu-3-70B-SFT`

  - DPO: `allenai/Llama-3.1-Tulu-3-70B-DPO`

  - 최종 모델: `allenai/Llama-3.1-Tulu-3-70B`

- **405B:**

  - SFT: `allenai/Llama-3.1-Tulu-3-405B-SFT`

  - DPO: `allenai/Llama-3.1-Tulu-3-405B-DPO`
  
  - 최종 모델: `allenai/Llama-3.1-Tulu-3-405B`



## SmolLM2

- **135M:**

  - Base: `HuggingFaceTB/SmolLM2-135M`

  - Instruct: `HuggingFaceTB/SmolLM2-135M-Instruct`

- **360M:**

  - Base: `HuggingFaceTB/SmolLM2-360M`

  - Instruct: `HuggingFaceTB/SmolLM2-360M-Instruct`

- **1.7B:**

  - Base: `HuggingFaceTB/SmolLM2-1.7B`

  - Instruct: `HuggingFaceTB/SmolLM2-1.7B-Instruct`




# 요약 및 결론

Tülu 3와 SmolLM2 모두 데이터 투명성과 세밀한 학습 레시피를 바탕으로 재현 가능하고 확장성이 뛰어난 모델 구축 사례다.

- **Tülu 3:** Llama 3.1 기반 포스트 트레이닝 모델로, SFT, DPO, RLVR 단계를 통해 최종 모델 완성
- **SmolLM2:** Llama architecture 기반 모델로, Pre-train, task 별 SFT를 통해 최종 모델 완성



앞으로도 이러한 공개 자료를 바탕으로 다양한 실험과 연구가 이루어지길 기대한다.

또한, 두 모델의 레시피를 실제 학습에 적용해보는 경험이 큰 도움이 될 것이라 믿는다.

