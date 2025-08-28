# conda activate zerosearch
python /weka/oe-adapt-default/tengx/main/open-instruct/mason.py \
        --cluster ai2/jupiter-cirrascale-2 \
        --workspace  ai2/olmo-instruct \
        --priority urgent \
        --image tengx/open_instruct_main_google2 \
        --priority urgent \
        --budget ai2/oe-adapt \
        --preemptible \
        --num_nodes 1 \
        --max_retries 0 \
        --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
        --gpus 4 -- python -m sglang.launch_server \
        --model /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/rl-sft/olmo2-7B-FINAL-lc-OT3-full-regen-wc-oasst-ccn-pif-qif-wgwj-syn2-aya-tgpt-ncode-scode  \
        --host 0.0.0.0 \
        --port 30000 \
        --tensor-parallel-size 4 \
# python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct  --host 0.0.0.0 --port 6001 --tensor-parallel-size 4
# python -m vllm.entrypoints.openai.api_server --model PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo  --host 0.0.0.0 --port 6001 --tensor-parallel-size 4