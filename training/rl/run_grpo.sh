#!/bin/bash
# FrontCoder GRPO (RL) Training Script
#
# This script trains the RL model using GRPO algorithm with
# vision-grounded rewards for front-end code generation.
#
# Key features based on the paper:
# - Composite reward: R = I_rep * I_render * (α*S_chk + β*S_sim + γ*S_len)
# - Checklist-based scoring with VLM (Qwen2.5-VL-72B)
# - Sandboxed HTML rendering for visual feedback
#
# Prerequisites:
# 1. Start the render service first: bash render_service/start_render_service.sh
# 2. Configure VLM service endpoint
#
# Usage:
#   bash run_grpo.sh

set -x

echo "========================================"
echo "FrontCoder GRPO Training"
echo "========================================"

# Configuration
MODEL_PATH="${MODEL_PATH:-path/to/sft_checkpoint}"
RENDER_SERVICE_URL="${RENDER_SERVICE_URL:-http://localhost:8768}"
VLM_BASE_URL="${VLM_BASE_URL:-http://localhost:8000/v1}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=path/to/rl_train.parquet \
    data.val_files=path/to/rl_val.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.prompt_key=prompt \
    data.reward_fn_key=data_source \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.rollout.engine_kwargs.enable_prefix_caching=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=20000 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    custom_reward_function.path=reward/html_reward.py \
    custom_reward_function.name=compute_html_reward_batch \
    reward_model.reward_manager=batch \
    +reward_model.reward_kwargs.use_custom_scoring=true \
    trainer.project_name='frontcoder-grpo' \
    trainer.experiment_name='qwen25-7b-grpo' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=15 \
    trainer.test_freq=1 \
    trainer.total_epochs=15 $@

echo "========================================"
echo "Training Complete"
echo "========================================"
