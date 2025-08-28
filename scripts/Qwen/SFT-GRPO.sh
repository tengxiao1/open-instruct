export integration_mix="saurabh5/rlvr_acecoder 56878 hamishivi/rlvr_orz_math_57k_collected 56878 hamishivi/tulu_3_rewritten_400k_string_f1_only_v2 56878 allenai/IF_multi_constraints_upto5 56878"
# export orz_only_mix="hamishivi/rlvr_orz_math_57k_collected 1.0"

# export polaris_53k="hamishivi/polaris_53k 1.0"

# testing:
# logic data only
# multi subject only.
# olmo2_lc qwen3 
for model in qwen2_5; do
for split_var in integration_mix; do
    split_value="${!split_var}"
    exp_name=sft_grpo_main_${model}_${split_var}_${RANDOM}
    if [ "$model" == "qwen3" ]; then
        model_name_or_path=hamishivi/qwen3_openthoughts2
        chat_template_name=tulu_thinker
        add_bos=False
    elif [ "$model" == "olmo2_lc" ]; then
        model_name_or_path=hamishivi/olmo2_lc_ot2
        chat_template_name=tulu_thinker
        add_bos=True
    elif [ "$model" == "qwen2_5" ]; then
        model_name_or_path=hamishivi/qwen2_5_openthoughts2
        # model_name_or_path=Qwen/Qwen2.5-7B
        # chat_template_name=tulu_thinker_r1_style
        chat_template_name=tulu_thinker
        add_bos=False
    fi

# export VLLM_ENFORCE_EAGER=1
# export TORCHDYNAMO_DISABLE=1

python mason.py \
        --cluster  ai2/augusta-google-1  \
        --workspace  ai2/tulu-thinker  \
        --priority high \
        --image tengx/open_instruct_main_google --pure_docker_mode   \
        --preemptible \
        --num_nodes 5 \
        --max_retries 0 \
        --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
        --budget ai2/oe-adapt \
        --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh  \&\& source configs/beaker_configs/code_api_setup.sh  \&\&  python open_instruct/grpo_fast.py \
        --exp_name ${exp_name} \
        --beta 0.0 \
        --add_bos ${add_bos} \
        --num_samples_per_prompt_rollout 8 \
        --num_unique_prompts_rollout 128 \
        --num_mini_batches 1 \
        --num_epochs 1 \
        --learning_rate 1e-6 \
        --per_device_train_batch_size 1 \
        --output_dir /output \
        --kl_estimator kl3 \
        --dataset_mixer_list ${split_value} \
        --dataset_mixer_list_splits train \
        --dataset_mixer_eval_list hamishivi/tulu_3_rewritten_100k 32 \
        --dataset_mixer_eval_list_splits train \
        --max_token_length 10240 \
        --max_prompt_token_length 2048 \
        --response_length 16384 \
        --pack_length 20480 \
        --model_name_or_path ${model_name_or_path} \
        --chat_template_name ${chat_template_name} \
        --stop_strings "</answer>" \
        --non_stop_penalty False \
        --temperature 1.0 \
        --ground_truths_key ground_truth \
        --sft_messages_key messages \
        --total_episodes 10000000 \
        --deepspeed_stage 3 \
        --num_learners_per_node 8 8 \
        --vllm_num_engines 24 \
        --vllm_tensor_parallel_size 1 \
        --lr_scheduler_type constant \
        --apply_verifiable_reward true \
        --seed 1 \
        --record_entropy False \
        --num_evals 10 \
        --save_freq 100 \
        --try_launch_beaker_eval_jobs_on_weka True \
        --gradient_checkpointing \
        --with_tracking \
        --oe_eval_max_length 32768 \
        --oe_eval_tasks minerva_math::hamish_zs_reasoning,gsm8k::zs_cot_latex,gsm8k::hamish_zs_reasoning,minerva_math_500::hamish_zs_reasoning,zebralogic::hamish_zs_reasoning,aime::hamish_zs_reasoning,agi_eval_english:0shot_cot::hamish_zs_reasoning,gpqa:0shot_cot::hamish_zs_reasoning,ifeval::hamish_zs_reasoning,popqa::hamish_zs_reasoning,mmlu:cot::hamish_zs_reasoning,bbh:cot::hamish_zs_reasoning,mbppplus:0-shot-chat::tulu-thinker,codex_humanevalplus:0-shot-chat-v1::tulu-thinker \
        --wandb_project_name OLMO-2 
done
done