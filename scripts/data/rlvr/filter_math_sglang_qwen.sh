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


python /weka/oe-adapt-default/tengx/main/open-instruct/mason.py \
  --cluster ai2/jupiter-cirrascale-2 \
  --image tengx/open_instruct_main_google2   \
  --workspace  ai2/olmo-instruct \
  --priority urgent \
  --preemptible \
  --gpus 1 \
  --budget ai2/oe-adapt \
  --num_nodes 1 \
  --max_retries 0 \
   -- python scripts/data/rlvr/filtering_sglang.py \
  --model hamishivi/qwen2_5_openthoughts2 \
  --dataset TTTXXX01/MathSub-30K  \
  --split train \
  --chat_template olmo_thinker \
  --offset 0 \
  --size 1 \
  --tensor_parallel_size 1 \
  --output-file /weka/oe-adapt-default/tengx/main/open-instruct/scripts/data/MathSub_test_sglang_0_1_qwen_1.jsonl \
  --number_samples 8

python /weka/oe-adapt-default/tengx/main/open-instruct/mason.py \
  --cluster ai2/jupiter-cirrascale-2 \
  --image tengx/open_instruct_main_google2   \
  --workspace  ai2/olmo-instruct \
  --priority urgent \
  --preemptible \
  --gpus 1 \
  --budget ai2/oe-adapt \
  --num_nodes 1 \
  --max_retries 0 \
  -- python scripts/data/rlvr/filtering_sglang.py \
  --model hamishivi/qwen2_5_openthoughts2 \
  --dataset TTTXXX01/MathSub-30K  \
  --split train \
  --chat_template olmo_thinker \
  --offset 0 \
  --size 8 \
  --tensor_parallel_size 1 \
  --output-file /weka/oe-adapt-default/tengx/main/open-instruct/scripts/data/MathSub_test_sglang_0_8_qwen_1.jsonl \
  --number_samples 8


python /weka/oe-adapt-default/tengx/main/open-instruct/mason.py \
  --cluster ai2/jupiter-cirrascale-2 \
  --image tengx/open_instruct_main_google2   \
  --workspace  ai2/olmo-instruct \
  --priority urgent \
  --preemptible \
  --gpus 1 \
  --budget ai2/oe-adapt \
  --num_nodes 1 \
  --max_retries 0 \
  -- python scripts/data/rlvr/filtering_sglang.py \
  --model hamishivi/qwen2_5_openthoughts2 \
  --dataset TTTXXX01/MathSub-30K  \
  --split train \
  --chat_template olmo_thinker \
  --offset 0 \
  --size 16 \
  --tensor_parallel_size 1 \
  --output-file /weka/oe-adapt-default/tengx/main/open-instruct/scripts/data/MathSub_test_sglang_0_16_qwen_1.jsonl \
  --number_samples 8


python /weka/oe-adapt-default/tengx/main/open-instruct/mason.py \
  --cluster ai2/jupiter-cirrascale-2 \
  --image tengx/open_instruct_main_google2   \
  --workspace  ai2/olmo-instruct \
  --priority urgent \
  --preemptible \
  --gpus 1 \
  --budget ai2/oe-adapt \
  --num_nodes 1 \
  --max_retries 0 \
  -- python scripts/data/rlvr/filtering_sglang.py \
  --model hamishivi/qwen2_5_openthoughts2  \
  --dataset TTTXXX01/MathSub-30K  \
  --split train \
  --chat_template olmo_thinker \
  --offset 0 \
  --size 64 \
  --tensor_parallel_size 1 \
  --output-file /weka/oe-adapt-default/tengx/main/open-instruct/scripts/data/MathSub_test_sglang_0_64_qwen.jsonl \
  --number_samples 8


# python /weka/oe-adapt-default/tengx/main/open-instruct/mason.py \
#   --cluster ai2/jupiter-cirrascale-2 \
#   --image tengx/open_instruct_main_google2  --pure_docker_mode   \
#   --workspace  ai2/olmo-instruct \
#   --priority urgent \
#   --preemptible \
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
#   --size 2 \
#   --output-file /weka/oe-adapt-default/tengx/main/open-instruct/scripts/data/MathSub-30K_1000_2000.jsonl \
#   --number_samples 8

# python /weka/oe-adapt-default/tengx/main/open-instruct/mason.py \
#   --cluster ai2/jupiter-cirrascale-2 \
#   --image tengx/open_instruct_main_google2  --pure_docker_mode   \
#   --workspace  ai2/olmo-instruct \
#   --priority urgent \
#   --preemptible \
#   --gpus 4 \
#   --budget ai2/oe-adapt \
#   --num_nodes 1 \
#   --max_retries 0 \
#   -- python scripts/data/rlvr/filtering_vllm.py \
#   --model /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/rl-sft/olmo2-7B-FINAL-lc-OT3-full-regen-wc-oasst-ccn-pif-qif-wgwj-syn2-aya-tgpt-ncode-scode \
#   --dataset TTTXXX01/MathSub-30K  \
#   --split train \
#   --chat_template olmo_thinker \
#   --offset 2000 \
#   --size 1000 \
#   --output-file /weka/oe-adapt-default/tengx/main/open-instruct/scripts/data/MathSub-30K_2000_3000.jsonl \
#   --number_samples 8


# python /weka/oe-adapt-default/tengx/main/open-instruct/mason.py \
#   --cluster ai2/jupiter-cirrascale-2 \
#   --image tengx/open_instruct_main_google2  --pure_docker_mode   \
#   --workspace  ai2/olmo-instruct \
#   --priority urgent \
#   --preemptible \
#   --gpus 4 \
#   --budget ai2/oe-adapt \
#   --num_nodes 1 \
#   --max_retries 0 \
#   -- python scripts/data/rlvr/filtering_vllm.py \
#   --model /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/rl-sft/olmo2-7B-FINAL-lc-OT3-full-regen-wc-oasst-ccn-pif-qif-wgwj-syn2-aya-tgpt-ncode-scode \
#   --dataset TTTXXX01/MathSub-30K  \
#   --split train \
#   --chat_template olmo_thinker \
#   --offset 3000 \
#   --size 1000 \
#   --output-file /weka/oe-adapt-default/tengx/main/open-instruct/scripts/data/MathSub-30K_3000_4000.jsonl \
#   --number_samples 8


# python /weka/oe-adapt-default/tengx/main/open-instruct/mason.py \
#   --cluster ai2/jupiter-cirrascale-2 \
#   --image tengx/open_instruct_main_google2  --pure_docker_mode   \
#   --workspace  ai2/olmo-instruct \
#   --priority urgent \
#   --preemptible \
#   --gpus 4 \
#   --budget ai2/oe-adapt \
#   --num_nodes 1 \
#   --max_retries 0 \
#   -- python scripts/data/rlvr/filtering_vllm.py \
#   --model /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/rl-sft/olmo2-7B-FINAL-lc-OT3-full-regen-wc-oasst-ccn-pif-qif-wgwj-syn2-aya-tgpt-ncode-scode \
#   --dataset TTTXXX01/MathSub-30K  \
#   --split train \
#   --chat_template olmo_thinker \
#   --offset 4000 \
#   --size 1000 \
#   --output-file /weka/oe-adapt-default/tengx/main/open-instruct/scripts/data/MathSub-30K_4000_5000.jsonl \
#   --number_samples 8

# python /weka/oe-adapt-default/tengx/main/open-instruct/mason.py \
#   --cluster ai2/jupiter-cirrascale-2 \
#   --image tengx/open_instruct_main_google2  --pure_docker_mode   \
#   --workspace  ai2/olmo-instruct \
#   --priority urgent \
#   --preemptible \
#   --gpus 2 \
#   --budget ai2/oe-adapt \
#   --num_nodes 1 \
#   --max_retries 0 \
#   -- python scripts/data/rlvr/filtering_vllm.py \
#   --model /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/rl-sft/olmo2-7B-FINAL-lc-OT3-full-regen-wc-oasst-ccn-pif-qif-wgwj-syn2-aya-tgpt-ncode-scode \
#   --dataset TTTXXX01/MathSub-30K  \
#   --split train \
#   --chat_template olmo_thinker \
#   --offset 5000 \
#   --size 10000 \
#   --output-file /weka/oe-adapt-default/tengx/main/open-instruct/scripts/data/MathSub-30K_5000_10000.jsonl \
#   --number_samples 8


# python /weka/oe-adapt-default/tengx/main/open-instruct/mason.py \
#   --cluster ai2/jupiter-cirrascale-2 \
#   --image tengx/open_instruct_main_google2  --pure_docker_mode   \
#   --workspace  ai2/olmo-instruct \
#   --priority urgent \
#   --preemptible \
#   --gpus 2 \
#   --budget ai2/oe-adapt \
#   --num_nodes 1 \
#   --max_retries 0 \
#   -- python scripts/data/rlvr/filtering_vllm.py \
#   --model /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/rl-sft/olmo2-7B-FINAL-lc-OT3-full-regen-wc-oasst-ccn-pif-qif-wgwj-syn2-aya-tgpt-ncode-scode \
#   --dataset TTTXXX01/MathSub-30K  \
#   --split train \
#   --chat_template olmo_thinker \
#   --offset 10000 \
#   --size 15000 \
#   --output-file /weka/oe-adapt-default/tengx/main/open-instruct/scripts/data/MathSub-30K_10000_15000.jsonl \
#   --number_samples 8


# python /weka/oe-adapt-default/tengx/main/open-instruct/mason.py \
#   --cluster ai2/jupiter-cirrascale-2 \
#   --image tengx/open_instruct_main_google2  --pure_docker_mode   \
#   --workspace  ai2/olmo-instruct \
#   --priority urgent \
#   --preemptible \
#   --gpus 2 \
#   --budget ai2/oe-adapt \
#   --num_nodes 1 \
#   --max_retries 0 \
#   -- python scripts/data/rlvr/filtering_vllm.py \
#   --model /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/rl-sft/olmo2-7B-FINAL-lc-OT3-full-regen-wc-oasst-ccn-pif-qif-wgwj-syn2-aya-tgpt-ncode-scode \
#   --dataset TTTXXX01/MathSub-30K  \
#   --split train \
#   --chat_template olmo_thinker \
#   --offset 15000 \
#   --size 20000 \
#   --output-file /weka/oe-adapt-default/tengx/main/open-instruct/scripts/data/MathSub-30K_15000_20000.jsonl \
#   --number_samples 8

# python /weka/oe-adapt-default/tengx/main/open-instruct/mason.py \
#   --cluster ai2/jupiter-cirrascale-2 \
#   --image tengx/open_instruct_main_google2  --pure_docker_mode   \
#   --workspace  ai2/olmo-instruct \
#   --priority urgent \
#   --preemptible \
#   --gpus 2 \
#   --budget ai2/oe-adapt \
#   --num_nodes 1 \
#   --max_retries 0 \
#   -- python scripts/data/rlvr/filtering_vllm.py \
#   --model /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/rl-sft/olmo2-7B-FINAL-lc-OT3-full-regen-wc-oasst-ccn-pif-qif-wgwj-syn2-aya-tgpt-ncode-scode \
#   --dataset TTTXXX01/MathSub-30K  \
#   --split train \
#   --chat_template olmo_thinker \
#   --offset 20000 \
#   --size 25000 \
#   --output-file /weka/oe-adapt-default/tengx/main/open-instruct/scripts/data/MathSub-30K_20000_25000.jsonl \
#   --number_samples 8

# python /weka/oe-adapt-default/tengx/main/open-instruct/mason.py \
#   --cluster ai2/jupiter-cirrascale-2 \
#   --image tengx/open_instruct_main_google2  --pure_docker_mode   \
#   --workspace  ai2/olmo-instruct \
#   --priority urgent \
#   --preemptible \
#   --gpus 2 \
#   --budget ai2/oe-adapt \
#   --num_nodes 1 \
#   --max_retries 0 \
#   -- python scripts/data/rlvr/filtering_vllm.py \
#   --model /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/rl-sft/olmo2-7B-FINAL-lc-OT3-full-regen-wc-oasst-ccn-pif-qif-wgwj-syn2-aya-tgpt-ncode-scode \
#   --dataset TTTXXX01/MathSub-30K  \
#   --split train \
#   --chat_template olmo_thinker \
#   --offset 25000 \
#   --size 30000 \
#   --output-file /weka/oe-adapt-default/tengx/main/open-instruct/scripts/data/MathSub-30K_25000_30000.jsonl \
#   --number_samples 8
