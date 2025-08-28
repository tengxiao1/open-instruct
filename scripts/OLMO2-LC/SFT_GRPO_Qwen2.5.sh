export orz_mix="hamishivi/rlvr_orz_math_57k_collected 56878"
export MathSub_mix="TTTXXX01/MathSub-30K 1.0"
export Math_Filtered_mix="TTTXXX01/MathSub_Filtered_LC 1.0 TTTXXX01/Omega_Filtered_LC 1.0 TTTXXX01/Orz_Filtered_LC 1.0"
for model in qwen25; do
for split_var in Math_Filtered_mix; do
    split_value="${!split_var}"
    exp_name=sft_qwen2.5_ot3_32768_${model}_${split_var}_${RANDOM}
    if [ "$model" == "qwen25" ]; then
        model_name_or_path=/weka/oe-adapt-default/jacobm/rl-sft/checkpoints/qwen_2p5_7b-open_thoughts_3__8__1751916783
        chat_template_name=olmo_thinker
        add_bos=False
    fi

python mason.py \
        --cluster  ai2/augusta-google-1  \
        --workspace ai2/tulu-thinker \
        --priority high \
        --image tengx/open_instruct_main_google2 --pure_docker_mode   \
        --preemptible \
        --num_nodes 4 \
        --max_retries 0 \
        --budget ai2/oe-adapt \
        --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
        --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh  \&\& source configs/beaker_configs/code_api_setup.sh  \&\&  python open_instruct/grpo_fast.py \
        --exp_name ${exp_name} \
        --beta 0.0 \
        --add_bos ${add_bos} \
        --num_samples_per_prompt_rollout 8 \
        --num_unique_prompts_rollout 32 \
        --num_mini_batches 1 \
        --num_epochs 1 \
        --learning_rate 1e-6 \
        --per_device_train_batch_size 1 \
        --output_dir /output \
        --kl_estimator kl3 \
        --dataset_mixer_list ${split_value} \
        --dataset_mixer_list_splits train \
        --dataset_mixer_eval_list TTTXXX01/MathSub_Filtered_LC 8  TTTXXX01/Omega_Filtered_LC 8   TTTXXX01/Orz_Filtered_LC 8 \
        --dataset_mixer_eval_list_splits train \
        --max_token_length 10240 \
        --max_prompt_token_length 2048 \
        --response_length 32768 \
        --pack_length 34816 \
        --model_name_or_path ${model_name_or_path} \
        --chat_template_name ${chat_template_name} \
        --stop_strings "</answer>" \
        --non_stop_penalty False \
        --temperature 1.0 \
        --ground_truths_key ground_truth \
        --sft_messages_key messages \
        --total_episodes 10000000 \
        --deepspeed_stage 3 \
        --num_learners_per_node 8 8\
        --vllm_num_engines 16 \
        --vllm_tensor_parallel_size 1 \
        --lr_scheduler_type constant \
        --apply_verifiable_reward true \
        --seed 1 \
        --fill_completions True \
        --record_entropy False \
        --save_freq 50 \
        --num_evals 200 \
        --eval_priority high \
        --try_launch_beaker_eval_jobs_on_weka True \
        --gradient_checkpointing \
        --with_tracking \
        --fill_completions True \
        --oe_eval_max_length 32768 \
        --clip_higher 0.2 \
        --oe_eval_tasks minerva_math::hamish_zs_reasoning,gsm8k::zs_cot_latex,gsm8k::hamish_zs_reasoning,minerva_math_500::hamish_zs_reasoning,zebralogic::hamish_zs_reasoning,aime::hamish_zs_reasoning,agi_eval_english:0shot_cot::hamish_zs_reasoning,gpqa:0shot_cot::hamish_zs_reasoning,ifeval::hamish_zs_reasoning,popqa::hamish_zs_reasoning,mmlu:cot::hamish_zs_reasoning,bbh:cot::hamish_zs_reasoning,mbppplus:0-shot-chat::tulu-thinker,codex_humanevalplus:0-shot-chat::tulu-thinker \
        --wandb_project_name OLMO-2 
done
done

