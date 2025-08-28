# #!/bin/bash

python scripts/submit_eval_jobs.py \
  --model_name sft_olmo2_new_full8_32768_olmo2_lc_Mathsub_mix_929__1__1755256609_checkpoints_step_50 \
  --location gs://ai2-llm/post-training//hamishivi//output/hamish_test_crash_32768_olmo2_lc_MathSub_mix_2866__1__1755381424_checkpoints/step_50 \
  --cluster ai2/saturn-cirrascale ai2/neptune-cirrascale \
  --is_tuned \
  --workspace "tulu-3-results" \
  --priority high \
  --preemptible \
  --use_hf_tokenizer_template \
  --run_oe_eval_experiments \
  --skip_oi_evals \
  --oe_eval_max_length 32768 \
  --step 100 \
  --gpu_multiplier 2 \
  --oe_eval_tasks aime:zs_cot_r1::pass_at_32_2024_temp1,aime:zs_cot_r1::pass_at_32_2025_temp1,aime::hamish_zs_reasoning


# #!/bin/bash
# python scripts/submit_eval_jobs.py \
#   --model_name  sft_olmo2_lc_wildchat_8192_olmo2_lc_orz_mix_24175__1__1754343934_checkpoints\
#   --location gs://ai2-llm/post-training//tengx//output/sft_olmo2_lc_wildchat_8192_olmo2_lc_orz_mix_24175__1__1754343934_checkpoints/step_100 \
#   --cluster ai2/saturn-cirrascale ai2/neptune-cirrascale \
#   --is_tuned \
#   --workspace "tulu-3-results" \
#   --priority high \
#   --preemptible \
#   --use_hf_tokenizer_template \
#   --run_oe_eval_experiments \
#   --skip_oi_evals \
#   --oe_eval_max_length 8192 \
#   --step 250 \
#   --gpu_multiplier 2 \
#   --oe_eval_tasks mbppplus:0-shot-chat::tulu-thinker

