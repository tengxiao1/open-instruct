# python /weka/oe-adapt-default/tengx/main/open-instruct/mason.py \
#   --cluster ai2/jupiter-cirrascale-2 \
#   --image michaeln/open_instruct_olmo2_retrofit --pure_docker_mode   \
#   --workspace  ai2/olmo-instruct \
#   --priority urgent \
#   --preemptible \
#   --env  VLLM_TORCH_COMPILE=0 \
#   --gpus 8 \
#   --budget ai2/oe-adapt \
#   --num_nodes 1 \
#   --max_retries 0 \
#   -- python scripts/data/rlvr/filtering_vllm.py \
#   --model /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/rl-sft/olmo2-7B-FINAL-lc-OT3-full-regen-wc-oasst-ccn-pif-qif-wgwj-syn2-aya-tgpt-ncode-scode \
#   --dataset TTTXXX01/MathSub-30K  \
#   --split train \
#   --chat_template olmo_thinker \
#   --offset 0 \
#   --size 1000 \
#   --output-file /weka/oe-adapt-default/tengx/main/open-instruct/scripts/data/MathSub-30K_0_1000.jsonl \
#   --number_samples 8

# python /weka/oe-adapt-default/tengx/main/open-instruct/mason.py \
#   --cluster ai2/jupiter-cirrascale-2 \
#   --image tengx/open_instruct_main_google2   \
#   --workspace  ai2/olmo-instruct \
#   --priority urgent \
#   --preemptible \
#   --gpus 8 \
#   --budget ai2/oe-adapt \
#   --num_nodes 1 \
#   --max_retries 0 \
#    -- python scripts/data/rlvr/filtering_sglang.py \
#   --model /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/rl-sft/olmo2-7B-FINAL-lc-OT3-full-regen-wc-oasst-ccn-pif-qif-wgwj-syn2-aya-tgpt-ncode-scode \
#   --dataset TTTXXX01/MathSub-30K  \
#   --split train \
#   --chat_template olmo_thinker \
#   --offset 500 \
#   --size 1000 \
#   --tensor_parallel_size 8 \
#   --output-file /weka/oe-adapt-default/tengx/main/open-instruct/scripts/data/MathSub-30K_500_1000_sglang.jsonl \
#   --number_samples 8


# python /weka/oe-adapt-default/tengx/main/open-instruct/mason.py \
#   --cluster ai2/jupiter-cirrascale-2  \
#   --workspace  ai2/olmo-instruct \
#   --image tengx/open_instruct_main_google2 \
#   --priority urgent \
#   --preemptible \
#   --gpus 8 \
#   --budget ai2/oe-adapt \
#   --num_nodes 1 \
#   --max_retries 0 \
#    -- python scripts/data/rlvr/filtering_sglang.py \
#   --model /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/rl-sft/olmo2-7B-FINAL-lc-OT3-full-regen-wc-oasst-ccn-pif-qif-wgwj-syn2-aya-tgpt-ncode-scode \
#   --dataset hamishivi/rlvr_orz_math_57k_collected  \
#   --split train \
#   --chat_template olmo_thinker \
#   --offset 13000 \
#   --size 1000 \
#   --tensor_parallel_size 8 \
#   --output-file /weka/oe-adapt-default/tengx/Data/filter_MathSub/orz_sglang_13000_14000.jsonl \
#   --number_samples 8

# python /weka/oe-adapt-default/tengx/main/open-instruct/mason.py \
#   --cluster ai2/jupiter-cirrascale-2  \
#   --workspace  ai2/olmo-instruct \
#   --image tengx/open_instruct_main_google2 \
#   --priority urgent \
#   --preemptible \
#   --gpus 8 \
#   --budget ai2/oe-adapt \
#   --num_nodes 1 \
#   --max_retries 0 \
#    -- python scripts/data/rlvr/filtering_sglang.py \
#   --model /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/rl-sft/olmo2-7B-FINAL-lc-OT3-full-regen-wc-oasst-ccn-pif-qif-wgwj-syn2-aya-tgpt-ncode-scode \
#   --dataset hamishivi/rlvr_orz_math_57k_collected \
#   --split train \
#   --chat_template olmo_thinker \
#   --offset 14000 \
#   --size 1000 \
#   --tensor_parallel_size 8 \
#   --output-file /weka/oe-adapt-default/tengx/Data/filter_MathSub/orz_14000_15000.jsonl \
#   --number_samples 8

# python /weka/oe-adapt-default/tengx/main/open-instruct/mason.py \
#   --cluster ai2/jupiter-cirrascale-2  \
#   --workspace  ai2/olmo-instruct \
#   --image tengx/open_instruct_main_google2 \
#   --priority urgent \
#   --preemptible \
#   --gpus 8 \
#   --budget ai2/oe-adapt \
#   --num_nodes 1 \
#   --max_retries 0 \
#    -- python scripts/data/rlvr/filtering_sglang.py \
#   --model /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/rl-sft/olmo2-7B-FINAL-lc-OT3-full-regen-wc-oasst-ccn-pif-qif-wgwj-syn2-aya-tgpt-ncode-scode \
#   --dataset hamishivi/rlvr_orz_math_57k_collected  \
#   --split train \
#   --chat_template olmo_thinker \
#   --offset 15000 \
#   --size 1000 \
#   --tensor_parallel_size 8 \
#   --output-file /weka/oe-adapt-default/tengx/Data/filter_MathSub/orz_sglang_15000_16000.jsonl \
#   --number_samples 8




# python /weka/oe-adapt-default/tengx/main/open-instruct/mason.py \
#   --cluster ai2/jupiter-cirrascale-2  \
#   --workspace  ai2/olmo-instruct \
#   --image tengx/open_instruct_main_google2 \
#   --priority urgent \
#   --preemptible \
#   --gpus 8 \
#   --budget ai2/oe-adapt \
#   --num_nodes 1 \
#   --max_retries 0 \
#    -- python scripts/data/rlvr/filtering_sglang.py \
#   --model /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/rl-sft/olmo2-7B-FINAL-lc-OT3-full-regen-wc-oasst-ccn-pif-qif-wgwj-syn2-aya-tgpt-ncode-scode \
#   --dataset hamishivi/rlvr_orz_math_57k_collected  \
#   --split train \
#   --chat_template olmo_thinker \
#   --offset 16000 \
#   --size 1000 \
#   --tensor_parallel_size 8 \
#   --output-file /weka/oe-adapt-default/tengx/Data/filter_MathSub/orz_sglang_16000_17000.jsonl \
#   --number_samples 8


# python /weka/oe-adapt-default/tengx/main/open-instruct/mason.py \
#   --cluster ai2/jupiter-cirrascale-2  \
#   --workspace  ai2/olmo-instruct \
#   --image tengx/open_instruct_main_google2 \
#   --priority urgent \
#   --preemptible \
#   --gpus 8 \
#   --budget ai2/oe-adapt \
#   --num_nodes 1 \
#   --max_retries 0 \
#    -- python scripts/data/rlvr/filtering_sglang.py \
#   --model /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/rl-sft/olmo2-7B-FINAL-lc-OT3-full-regen-wc-oasst-ccn-pif-qif-wgwj-syn2-aya-tgpt-ncode-scode \
#   --dataset hamishivi/rlvr_orz_math_57k_collected  \
#   --split train \
#   --chat_template olmo_thinker \
#   --offset 17000 \
#   --size 1000 \
#   --tensor_parallel_size 8 \
#   --output-file /weka/oe-adapt-default/tengx/Data/filter_MathSub/orz_sglang_17000_18000.jsonl \
#   --number_samples 8




# python /weka/oe-adapt-default/tengx/main/open-instruct/mason.py \
#   --cluster ai2/jupiter-cirrascale-2  \
#   --workspace  ai2/olmo-instruct \
#   --image tengx/open_instruct_main_google2 \
#   --priority urgent \
#   --preemptible \
#   --gpus 8 \
#   --budget ai2/oe-adapt \
#   --num_nodes 1 \
#   --max_retries 0 \
#    -- python scripts/data/rlvr/filtering_sglang.py \
#   --model /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/rl-sft/olmo2-7B-FINAL-lc-OT3-full-regen-wc-oasst-ccn-pif-qif-wgwj-syn2-aya-tgpt-ncode-scode \
#   --dataset hamishivi/rlvr_orz_math_57k_collected  \
#   --split train \
#   --chat_template olmo_thinker \
#   --offset 19000 \
#   --size 1000 \
#   --tensor_parallel_size 8 \
#   --output-file /weka/oe-adapt-default/tengx/Data/filter_MathSub/orz_sglang_19000_20000.jsonl \
#   --number_samples 8


#   python /weka/oe-adapt-default/tengx/main/open-instruct/mason.py \
#   --cluster ai2/jupiter-cirrascale-2  \
#         --workspace  ai2/olmo-instruct \
#         --image tengx/open_instruct_main_google2 \
#         --priority urgent \
#   --preemptible \
#   --gpus 8 \
#   --budget ai2/oe-adapt \
#   --num_nodes 1 \
#   --max_retries 0 \
#    -- python scripts/data/rlvr/filtering_sglang.py \
#   --model /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/rl-sft/olmo2-7B-FINAL-lc-OT3-full-regen-wc-oasst-ccn-pif-qif-wgwj-syn2-aya-tgpt-ncode-scode \
#   --dataset hamishivi/rlvr_orz_math_57k_collected  \
#   --split train \
#   --chat_template olmo_thinker \
#   --offset 20000 \
#   --size 1000 \
#   --tensor_parallel_size 8 \
#   --output-file /weka/oe-adapt-default/tengx/Data/filter_MathSub/orz_sglang_20000_21000.jsonl \
#   --number_samples 8


#   python /weka/oe-adapt-default/tengx/main/open-instruct/mason.py \
#   --cluster ai2/jupiter-cirrascale-2  \
#         --workspace  ai2/olmo-instruct \
#         --image tengx/open_instruct_main_google2 \
#         --priority urgent \
#   --preemptible \
#   --gpus 8 \
#   --budget ai2/oe-adapt \
#   --num_nodes 1 \
#   --max_retries 0 \
#    -- python scripts/data/rlvr/filtering_sglang.py \
#   --model /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/rl-sft/olmo2-7B-FINAL-lc-OT3-full-regen-wc-oasst-ccn-pif-qif-wgwj-syn2-aya-tgpt-ncode-scode \
#   --dataset hamishivi/omega-combined  \
#   --split train \
#   --chat_template olmo_thinker \
#   --offset 3000 \
#   --size 1000 \
#   --tensor_parallel_size 8 \
#   --output-file /weka/oe-adapt-default/tengx/Data/filter_MathSub/omega_sglang_3000_4000.jsonl \
#   --number_samples 8




#   python /weka/oe-adapt-default/tengx/main/open-instruct/mason.py \
#   --cluster ai2/jupiter-cirrascale-2  \
#         --workspace  ai2/olmo-instruct \
#         --image tengx/open_instruct_main_google2 \
#         --priority urgent \
#   --preemptible \
#   --gpus 8 \
#   --budget ai2/oe-adapt \
#   --num_nodes 1 \
#   --max_retries 0 \
#    -- python scripts/data/rlvr/filtering_sglang.py \
#   --model /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/rl-sft/olmo2-7B-FINAL-lc-OT3-full-regen-wc-oasst-ccn-pif-qif-wgwj-syn2-aya-tgpt-ncode-scode \
#   --dataset hamishivi/omega-combined  \
#   --split train \
#   --chat_template olmo_thinker \
#   --offset 4000 \
#   --size 1000 \
#   --tensor_parallel_size 8 \
#   --output-file /weka/oe-adapt-default/tengx/Data/filter_MathSub/omega_sglang_4000_5000.jsonl \
#   --number_samples 8

#   python /weka/oe-adapt-default/tengx/main/open-instruct/mason.py \
#   --cluster ai2/jupiter-cirrascale-2  \
#         --workspace  ai2/olmo-instruct \
#         --image tengx/open_instruct_main_google2 \
#         --priority urgent \
#   --preemptible \
#   --gpus 8 \
#   --budget ai2/oe-adapt \
#   --num_nodes 1 \
#   --max_retries 0 \
#    -- python scripts/data/rlvr/filtering_sglang.py \
#   --model /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/rl-sft/olmo2-7B-FINAL-lc-OT3-full-regen-wc-oasst-ccn-pif-qif-wgwj-syn2-aya-tgpt-ncode-scode \
#   --dataset hamishivi/omega-combined  \
#   --split train \
#   --chat_template olmo_thinker \
#   --offset 5000 \
#   --size 1000 \
#   --tensor_parallel_size 8 \
#   --output-file /weka/oe-adapt-default/tengx/Data/filter_MathSub/omega_sglang_5000_6000.jsonl \
#   --number_samples 8

#  python /weka/oe-adapt-default/tengx/main/open-instruct/mason.py \
#   --cluster ai2/jupiter-cirrascale-2  \
#         --workspace  ai2/olmo-instruct \
#         --image tengx/open_instruct_main_google2 \
#         --priority urgent \
#   --preemptible \
#   --gpus 8 \
#   --budget ai2/oe-adapt \
#   --num_nodes 1 \
#   --max_retries 0 \
#    -- python scripts/data/rlvr/filtering_sglang.py \
#   --model /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/rl-sft/olmo2-7B-FINAL-lc-OT3-full-regen-wc-oasst-ccn-pif-qif-wgwj-syn2-aya-tgpt-ncode-scode \
#   --dataset hamishivi/omega-combined  \
#   --split train \
#   --chat_template olmo_thinker \
#   --offset 6000 \
#   --size 1000 \
#   --tensor_parallel_size 8 \
#   --output-file /weka/oe-adapt-default/tengx/Data/filter_MathSub/omega_sglang_6000_7000.jsonl \
#   --number_samples 8

#  python /weka/oe-adapt-default/tengx/main/open-instruct/mason.py \
#   --cluster ai2/jupiter-cirrascale-2  \
#         --workspace  ai2/olmo-instruct \
#         --image tengx/open_instruct_main_google2 \
#         --priority urgent \
#   --preemptible \
#   --gpus 8 \
#   --budget ai2/oe-adapt \
#   --num_nodes 1 \
#   --max_retries 0 \
#    -- python scripts/data/rlvr/filtering_sglang.py \
#   --model /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/rl-sft/olmo2-7B-FINAL-lc-OT3-full-regen-wc-oasst-ccn-pif-qif-wgwj-syn2-aya-tgpt-ncode-scode \
#   --dataset hamishivi/omega-combined  \
#   --split train \
#   --chat_template olmo_thinker \
#   --offset 7000 \
#   --size 1000 \
#   --tensor_parallel_size 8 \
#   --output-file /weka/oe-adapt-default/tengx/Data/filter_MathSub/omega_sglang_7000_8000.jsonl \
#   --number_samples 8



#  python /weka/oe-adapt-default/tengx/main/open-instruct/mason.py \
#   --cluster ai2/jupiter-cirrascale-2  \
#         --workspace  ai2/olmo-instruct \
#         --image tengx/open_instruct_main_google2 \
#         --priority urgent \
#   --preemptible \
#   --gpus 8 \
#   --budget ai2/oe-adapt \
#   --num_nodes 1 \
#   --max_retries 0 \
#    -- python scripts/data/rlvr/filtering_sglang.py \
#   --model /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/rl-sft/olmo2-7B-FINAL-lc-OT3-full-regen-wc-oasst-ccn-pif-qif-wgwj-syn2-aya-tgpt-ncode-scode \
#   --dataset hamishivi/omega-combined  \
#   --split train \
#   --chat_template olmo_thinker \
#   --offset 13000 \
#   --size 1000 \
#   --tensor_parallel_size 8 \
#   --output-file /weka/oe-adapt-default/tengx/Data/filter_MathSub/omega_sglang_13000_14000.jsonl \
#   --number_samples 8


#  python /weka/oe-adapt-default/tengx/main/open-instruct/mason.py \
#   --cluster ai2/jupiter-cirrascale-2  \
#         --workspace  ai2/olmo-instruct \
#         --image tengx/open_instruct_main_google2 \
#         --priority urgent \
#   --preemptible \
#   --gpus 8 \
#   --budget ai2/oe-adapt \
#   --num_nodes 1 \
#   --max_retries 0 \
#    -- python scripts/data/rlvr/filtering_sglang.py \
#   --model /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/rl-sft/olmo2-7B-FINAL-lc-OT3-full-regen-wc-oasst-ccn-pif-qif-wgwj-syn2-aya-tgpt-ncode-scode \
#   --dataset hamishivi/omega-combined  \
#   --split train \
#   --chat_template olmo_thinker \
#   --offset 14000 \
#   --size 1000 \
#   --tensor_parallel_size 8 \
#   --output-file /weka/oe-adapt-default/tengx/Data/filter_MathSub/omega_sglang_14000_15000.jsonl \
#   --number_samples 8

 python /weka/oe-adapt-default/tengx/main/open-instruct/mason.py \
  --cluster ai2/jupiter-cirrascale-2  \
        --workspace  ai2/olmo-instruct \
        --image tengx/open_instruct_main_google2 \
        --priority urgent \
  --preemptible \
  --gpus 8 \
  --budget ai2/oe-adapt \
  --num_nodes 1 \
  --max_retries 0 \
   -- python scripts/data/rlvr/filtering_sglang.py \
  --model /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/rl-sft/olmo2-7B-FINAL-lc-OT3-full-regen-wc-oasst-ccn-pif-qif-wgwj-syn2-aya-tgpt-ncode-scode \
  --dataset TTTXXX01/MathSub-30K  \
  --split train \
  --chat_template olmo_thinker \
  --offset 14000 \
  --size 1000 \
  --tensor_parallel_size 8 \
  --output-file /weka/oe-adapt-default/tengx/Data/filter_MathSub/MathSub_sglang_14000_15000.jsonl \
  --number_samples 8


