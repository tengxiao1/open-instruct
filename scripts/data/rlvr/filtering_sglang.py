import argparse
import json

from datasets import load_dataset
from transformers import AutoTokenizer
import sglang as sgl
import time
from tqdm import tqdm


# python /weka/oe-adapt-default/tengx/main/open-instruct/mason.py \
#   --cluster ai2/jupiter-cirrascale-2 \
#   --image tengx/open_instruct_main_google2   \
#   --workspace  ai2/olmo-instruct \
#   --priority urgent \
#   --preemptible \
#   --gpus 2 \
#   --budget ai2/oe-adapt \
#   --num_nodes 1 \
#   --max_retries 0 \
#    -- python scripts/data/rlvr/filtering_sglang.py \
#   --model /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/rl-sft/olmo2-7B-FINAL-lc-OT3-full-regen-wc-oasst-ccn-pif-qif-wgwj-syn2-aya-tgpt-ncode-scode \
#   --dataset TTTXXX01/MathSub-30K  \
#   --split train \
#   --chat_template olmo_thinker \
#   --offset 0 \
#   --size 1 \
#   --tensor_parallel_size 1 \
#   --output-file /weka/oe-adapt-default/tengx/main/open-instruct/scripts/data/MathSub_test_sglang_0_1.jsonl \
  
CHAT_TEMPLATES = {
    "simple_concat_with_space": (
        "{% for message in messages %}"
        "{{ ' ' if not loop.first else '' }}"
        "{{ message['content'] }}"
        "{% if loop.last and not add_generation_prompt %}{{ eos_token }}{% endif %}"
        "{% endfor %}"
    ),
    "simple_concat_with_new_line": (
        "{% for message in messages %}"
        "{{ '\n' if not loop.first else '' }}"
        "{{ message['content'] }}"
        "{% if loop.last and not add_generation_prompt %}{{ eos_token }}{% endif %}"
        "{% endfor %}"
    ),
    "simple_chat": (
        "{% for message in messages %}"
        "{{ '\n\n' if not loop.first else '' }}"
        "{{ message['role'].capitalize() + ': ' + message['content'] }}"
        "{% if loop.last and not add_generation_prompt %}{{ eos_token }}{% endif %}"
        "{% endfor %}"
    ),
    "assistant_message_only": (
        "{% for message in messages %}"
        "{% if message['role'] == 'assistant' %}"
        "{{ message['content'] }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "zephyr": (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ '<|user|>\n' + message['content'] + eos_token + '\n' }}"
        "{% elif message['role'] == 'system' %}"
        "{{ '<|system|>\n' + message['content'] + eos_token + '\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|assistant|>\n' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    # olmo-core-compatible chat templates:
    # TODO: unify these 3 chat templates and send variables through the tokenizer's apply_chat_template kwargs
    "olmo": (
        "{% set has_system = messages|selectattr('role', 'equalto', 'system')|list|length > 0 %}"
        "{% if not has_system %}"
        "{{ '<|im_start|>system\nYou are OLMo, a helpful function-calling AI assistant built by Ai2. Your date cutoff is November 2024, and your model weights are available at https://huggingface.co/allenai. You do not currently have access to any functions. <functions></functions><|im_end|>\n' }}"
        "{% endif %}"
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<|im_start|>system\n' + message['content'] }}"
        "{% if message.get('functions', none) is not none %}"
        "{{ ' <functions>' + message['functions'] + '</functions><|im_end|>\n' }}"
        "{% else %}"
        "{{ ' You do not currently have access to any functions. <functions></functions><|im_end|>\n' }}"
        "{% endif %}"
        "{% elif message['role'] == 'user' %}"
        "{% if message.get('functions', none) is not none %}"
        "{{ '<|im_start|>user\n' + message['content'] + '\n' + '<functions>' + message['functions'] + '</functions><|im_end|>\n' }}"
        "{% else %}"
        "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endif %}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% if message.get('content', none) is not none %}"
        "{{ message['content'] }}"
        "{% endif %}"
        "{% if message.get('function_calls', none) is not none %}"
        "{{ '<function_calls>' + message['function_calls'] + '</function_calls>' }}"
        "{% endif %}"
        "{% if not loop.last %}"
        "{{ '<|im_end|>' + '\n' }}"
        "{% else %}"
        "{{ eos_token }}"
        "{% endif %}"
        "{% elif message['role'] == 'environment' %}"
        "{{ '<|im_start|>environment\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "olmo_thinker": (
        "{% set has_system = messages|selectattr('role', 'equalto', 'system')|list|length > 0 %}"
        "{% if not has_system %}"
        "{{ '<|im_start|>system\nYou are OLMo, a helpful function-calling AI assistant built by Ai2. Your date cutoff is November 2024, and your model weights are available at https://huggingface.co/allenai. You do not currently have access to any functions. <functions></functions><|im_end|>\n' }}"
        "{% endif %}"
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<|im_start|>system\n' + message['content'] }}"
        "{% if message.get('functions', none) is not none %}"
        "{{ ' <functions>' + message['functions'] + '</functions><|im_end|>\n' }}"
        "{% else %}"
        "{{ ' You do not currently have access to any functions. <functions></functions><|im_end|>\n' }}"
        "{% endif %}"
        "{% elif message['role'] == 'user' %}"
        "{% if message.get('functions', none) is not none %}"
        "{{ '<|im_start|>user\n' + message['content'] + '\n' + '<functions>' + message['functions'] + '</functions><|im_end|>\n' }}"
        "{% else %}"
        "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endif %}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% if message.get('content', none) is not none %}"
        "{{ message['content'] }}"
        "{% endif %}"
        "{% if message.get('function_calls', none) is not none %}"
        "{{ '<function_calls>' + message['function_calls'] + '</function_calls>' }}"
        "{% endif %}"
        "{% if not loop.last %}"
        "{{ '<|im_end|>' + '\n' }}"
        "{% else %}"
        "{{ eos_token }}"
        "{% endif %}"
        "{% elif message['role'] == 'environment' %}"
        "{{ '<|im_start|>environment\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n<think>' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "olmo_thinker_r1_style": (
        "A conversation between user and assistant. "
        "The user asks a question, and the assistant solves it. "
        "The assistant first thinks and reasons about the question "
        "and after thinking provides the user with the answer. "
        "The reasoning process is enclosed in <think> </think> tags "
        "and the answer are enclosed in <answer> </answer> tags "
        "so the full response is <think> reasoning process here </think> "
        "<answer> answer here </answer>."
        "\n\n"
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{% if message.get('functions', none) is not none %}"
        "{{ '<|im_start|>system\n' + message['content'] + '\n' + '<functions>' + message['functions'] + '</functions><|im_end|>\n' }}"
        "{% else %}"
        "{{ '<|im_start|>system\n' + message['content']  + '<|im_end|>\n' }}"
        "{% endif %}"
        "{% elif message['role'] == 'user' %}"
        "{% if message.get('functions', none) is not none %}"
        "{{ '<|im_start|>user\n' + message['content'] + '\n' + '<functions>' + message['functions'] + '</functions><|im_end|>\n' }}"
        "{% else %}"
        "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endif %}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% if message.get('content', none) is not none %}"
        "{{ message['content'] }}"
        "{% endif %}"
        "{% if message.get('function_calls', none) is not none %}"
        "{{ '<function_calls>' + message['function_calls'] + '</function_calls>' }}"
        "{% endif %}"
        "{% if not loop.last %}"
        "{{ '<|im_end|>' + '\n' }}"
        "{% else %}"
        "{{ eos_token }}"
        "{% endif %}"
        "{% elif message['role'] == 'environment' %}"
        "{{ '<|im_start|>environment\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n<think>' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "tulu": (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<|system|>\n' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'user' %}"
        "{{ '<|user|>\n' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{% if not loop.last %}"
        "{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}"
        "{% else %}"
        "{{ '<|assistant|>\n'  + message['content'] + eos_token }}"
        "{% endif %}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|assistant|>\n' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "tulu_thinker": (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<|system|>\n' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'user' %}"
        "{{ '<|user|>\n' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{% set content = message['content'] %}"
        "{% if not loop.last %}"
        "{{ '<|assistant|>\n' + content + eos_token + '\n' }}"
        "{% else %}"
        "{{ '<|assistant|>\n' + content + eos_token }}"
        "{% endif %}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|assistant|>\n<think>' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "tulu_thinker_r1_style": (
        "A conversation between User and Assistant. "
        "The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in "
        "the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> "
        "and <answer> </answer> tags, respectively, "
        "i.e., <think> reasoning process here </think> "
        "<answer> answer here </answer>."
        "\n\n"
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<|system|>\n' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'user' %}"
        "{{ '<|user|>\n' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{% set content = message['content'] %}"
        "{% if '</think>' in content %}"
        "{% set content = content.split('</think>')[-1] %}"
        "{% endif %}"
        "{% if not loop.last %}"
        "{{ '<|assistant|>\n' + content + eos_token + '\n' }}"
        "{% else %}"
        "{{ '<|assistant|>\n' + content + eos_token }}"
        "{% endif %}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|assistant|>\n<think>' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    # olmo-core-compatible chat templates:
    # TODO: unify these 3 chat templates and send variables through the tokenizer's apply_chat_template kwargs
    "olmo": (
        "{% set has_system = messages|selectattr('role', 'equalto', 'system')|list|length > 0 %}"
        "{% if not has_system %}"
        "{{ '<|im_start|>system\nYou are OLMo, a helpful function-calling AI assistant built by Ai2. Your date cutoff is November 2024, and your model weights are available at https://huggingface.co/allenai. You do not currently have access to any functions. <functions></functions><|im_end|>\n' }}"
        "{% endif %}"
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<|im_start|>system\n' + message['content'] }}"
        "{% if message.get('functions', none) is not none %}"
        "{{ ' <functions>' + message['functions'] + '</functions><|im_end|>\n' }}"
        "{% else %}"
        "{{ ' You do not currently have access to any functions. <functions></functions><|im_end|>\n' }}"
        "{% endif %}"
        "{% elif message['role'] == 'user' %}"
        "{% if message.get('functions', none) is not none %}"
        "{{ '<|im_start|>user\n' + message['content'] + '\n' + '<functions>' + message['functions'] + '</functions><|im_end|>\n' }}"
        "{% else %}"
        "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endif %}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% if message.get('content', none) is not none %}"
        "{{ message['content'] }}"
        "{% endif %}"
        "{% if message.get('function_calls', none) is not none %}"
        "{{ '<function_calls>' + message['function_calls'] + '</function_calls>' }}"
        "{% endif %}"
        "{% if not loop.last %}"
        "{{ '<|im_end|>' + '\n' }}"
        "{% else %}"
        "{{ eos_token }}"
        "{% endif %}"
        "{% elif message['role'] == 'environment' %}"
        "{{ '<|im_start|>environment\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "olmo_thinker": (
        "{% set has_system = messages|selectattr('role', 'equalto', 'system')|list|length > 0 %}"
        "{% if not has_system %}"
        "{{ '<|im_start|>system\nYou are OLMo, a helpful function-calling AI assistant built by Ai2. Your date cutoff is November 2024, and your model weights are available at https://huggingface.co/allenai. You do not currently have access to any functions. <functions></functions><|im_end|>\n' }}"
        "{% endif %}"
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<|im_start|>system\n' + message['content'] }}"
        "{% if message.get('functions', none) is not none %}"
        "{{ ' <functions>' + message['functions'] + '</functions><|im_end|>\n' }}"
        "{% else %}"
        "{{ ' You do not currently have access to any functions. <functions></functions><|im_end|>\n' }}"
        "{% endif %}"
        "{% elif message['role'] == 'user' %}"
        "{% if message.get('functions', none) is not none %}"
        "{{ '<|im_start|>user\n' + message['content'] + '\n' + '<functions>' + message['functions'] + '</functions><|im_end|>\n' }}"
        "{% else %}"
        "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endif %}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% if message.get('content', none) is not none %}"
        "{{ message['content'] }}"
        "{% endif %}"
        "{% if message.get('function_calls', none) is not none %}"
        "{{ '<function_calls>' + message['function_calls'] + '</function_calls>' }}"
        "{% endif %}"
        "{% if not loop.last %}"
        "{{ '<|im_end|>' + '\n' }}"
        "{% else %}"
        "{{ eos_token }}"
        "{% endif %}"
        "{% elif message['role'] == 'environment' %}"
        "{{ '<|im_start|>environment\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n<think>' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "olmo_thinker_r1_style": (
        "A conversation between user and assistant. "
        "The user asks a question, and the assistant solves it. "
        "The assistant first thinks and reasons about the question "
        "and after thinking provides the user with the answer. "
        "The reasoning process is enclosed in <think> </think> tags "
        "and the answer is enclosed in <answer> </answer> tags "
        "so the full response is <think> reasoning process here </think> "
        "<answer> answer here </answer>."
        "\n\n"
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{% if message.get('functions', none) is not none %}"
        "{{ '<|im_start|>system\n' + message['content'] + '\n' + '<functions>' + message['functions'] + '</functions><|im_end|>\n' }}"
        "{% else %}"
        "{{ '<|im_start|>system\n' + message['content']  + '<|im_end|>\n' }}"
        "{% endif %}"
        "{% elif message['role'] == 'user' %}"
        "{% if message.get('functions', none) is not none %}"
        "{{ '<|im_start|>user\n' + message['content'] + '\n' + '<functions>' + message['functions'] + '</functions><|im_end|>\n' }}"
        "{% else %}"
        "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endif %}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% if message.get('content', none) is not none %}"
        "{{ message['content'] }}"
        "{% endif %}"
        "{% if message.get('function_calls', none) is not none %}"
        "{{ '<function_calls>' + message['function_calls'] + '</function_calls>' }}"
        "{% endif %}"
        "{% if not loop.last %}"
        "{{ '<|im_end|>' + '\n' }}"
        "{% else %}"
        "{{ eos_token }}"
        "{% endif %}"
        "{% elif message['role'] == 'environment' %}"
        "{{ '<|im_start|>environment\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n<think>' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    # template is taken from https://arxiv.org/abs/2501.12948.
    "r1_simple_chat": (
        "A conversation between User and Assistant. "
        "The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in "
        "the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> "
        "and <answer> </answer> tags, respectively, "
        "i.e., <think> reasoning process here </think> "
        "<answer> answer here </answer>."
        "\n\n"
        "{% for message in messages %}"
        "{{ '\n\n' if not loop.first else '' }}"
        "{{ message['role'].capitalize() + ': ' + message['content'] + '\n' }}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ 'Assistant:' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "r1_simple_chat_postpend_think": (
        "A conversation between User and Assistant. "
        "The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in "
        "the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> "
        "and <answer> </answer> tags, respectively, "
        "i.e., <think> reasoning process here </think> "
        "<answer> answer here </answer>."
        "\n\n"
        "{% for message in messages %}"
        "{{ '\n\n' if not loop.first else '' }}"
        "{{ message['role'].capitalize() + ': ' + message['content'] + '\n' }}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ 'Assistant: <think>' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "r1_simple_chat_postpend_think_orz_style": (
        "A conversation between User and Assistant. "
        "The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in "
        "the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> "
        "and <answer> </answer> tags, respectively, "
        "i.e., <think> reasoning process here </think> "
        "<answer> answer here </answer>."
        "\n\n"
        "{% for message in messages %}"
        "{{ '\n\n' if not loop.first else '' }}"
        "{{ message['role'].capitalize() + ': You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\\\boxed{} tag. This is the problem: ' + message['content'] + '\n' }}"  # \\\\boxed{} is for jinja template escape
        "{% if loop.last and add_generation_prompt %}"
        "{{ 'Assistant: <think>' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "r1_simple_chat_postpend_think_tool_vllm": (
        "A conversation between User and Assistant. "
        "The User asks a question, and the Assistant solves it. "
        "The Assistant first thinks about the reasoning process in "
        "the mind and then provides the User with the answer. "
        "\n\n"
        "When given a question, the Assistant must conduct reasoning inside the <think> "
        "and </think> tags. During reasoning, the Assistant may write and execute python "
        "code using the <code> </code> tag, in order to solve the problem or verify the answer. "
        "Then the Assistant will get the stdout and stderr in the <output> and </output> tags. "
        "For example, the code could be\n"
        "<code>\n"
        "x, y = 1, 2\n"
        "result = x + y\n"
        "print(result)\n"
        "</code>\n"
        "or\n"
        "<code>\n"
        "import sympy as sp\n"
        "from sympy import Symbol\n"
        "x = Symbol('x')\n"
        "y = Symbol('y')\n"
        "solution = sp.solve(x**2 + y**2 - 1, (x, y))\n"
        "print(solution)\n"
        "</code>\n"
        "The Assistant will always `print` the result of the code execution in order to see it in the <output> tag. "
        "The Assistant may use the <code> </code> tag multiple times. "
        "When the Assistant is done reasoning, it should provide the answer inside the <answer> "
        "and </answer> tag."
        "\n\n"
        "{% for message in messages %}"
        "{{ '\n\n' if not loop.first else '' }}"
        "{{ message['role'].capitalize() + ': You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\\\boxed{} tag. This is the problem: ' + message['content'] + '\n' }}"  # \\\\boxed{} is for jinjia template escape
        "{% if loop.last and add_generation_prompt %}"
        "{{ 'Assistant: <think>' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
}

def main():
    parser = argparse.ArgumentParser(
        description="Bulk-generate N samples per HF dataset record using SGLang (offline)."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model path (e.g. facebook/opt-125m)"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="HF dataset name (e.g. squad)"
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Which split to load"
    )
    parser.add_argument(
        "--offset",
        type=int,
        required=True,
        help="Start index into the split"
    )
    parser.add_argument(
        "--size",
        type=int,
        required=True,
        help="Number of records to process"
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Path for output JSONL"
    )
    parser.add_argument(
        "--push_to_hub",
        default=None,
        type=str,
        help="Give a dataset name to push this data to the hub."
    )
    parser.add_argument(
        "--chat_template",
        type=str,
        default="olmo_thinker",
        help="Chat template name"
    )
    parser.add_argument(
        "--number_samples",
        type=int,
        default=8,
        help="Number of samples to generate per record"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Data type for model weights"
    )
    args = parser.parse_args()

    # 1. Load and slice dataset
    ds = load_dataset(args.dataset, split=args.split)
    ds = ds.shuffle(seed=42)  # so we dont just take first n samples
    up_to = min(args.offset + args.size, len(ds))
    subset = ds.select(range(args.offset, up_to))

    # 2. Tokenizer for chat templating
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.chat_template is not None:
        if args.chat_template not in CHAT_TEMPLATES:
            raise ValueError(f"Unknown chat template: {args.chat_template}")
        tokenizer.chat_template = CHAT_TEMPLATES[args.chat_template]

    # 3. Build prompts
    prompts = [
        tokenizer.apply_chat_template(
            sample["messages"][:-1] if len(sample["messages"]) > 1 else sample["messages"],
            add_generation_prompt=True,
            tokenize=False
        )
        for sample in subset
    ]
    
    # 4. Initialize SGLang offline engine
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(args.tensor_parallel_size)))
    
    engine = sgl.Engine(
        model_path=args.model,
        tp_size=args.tensor_parallel_size,
        dtype=args.dtype,
    )
    sgl.set_default_backend(engine)
    
    # 5. Generate samples using simple approach
    outputs = []



    print(f"Generating {args.number_samples} samples per prompt, total {len(prompts)} prompts...")

    start_time = time.time()
    results = engine.generate(
                prompts,
                sampling_params={
                    "temperature": args.temperature,
                    "max_new_tokens": 32768,
                    "n": args.number_samples,
                },
                stream=False
            )
    end_time = time.time()
    
    print(
    f"sglang inference time: {end_time - start_time:.2f} seconds, "
    f"prompts: {len(prompts)}, samples per prompt: {args.number_samples}"
    )
    outputs = []



    print(f"#prompts={len(prompts)}, #results={len(results)}")

    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i + 1}/{len(prompts)}")  # Print first 50 chars of the prompt
        start = i * args.number_samples  
        end = start + args.number_samples
        gens = [res["text"] for res in results[start:end]]
        gen_prompt_tokens = [res["meta_info"]["prompt_tokens"] for res in results[start:end]]
        outputs.append({"prompt": prompt, "outputs": gens, "prompt_tokens": gen_prompt_tokens})

    print(f"Generated {len(outputs)} outputs.")
    # print(f"Sample output for 1 prompt: {outputs[0]}")
    # print(f"Sample output for 2 prompt: {outputs[1]}")
    # print(f"Sample output for 3 prompt: {outputs[2]}")
    for i in range(len(outputs)):
        print (f"Sample output for 0 prompt: {len(outputs[i]['outputs'])} outputs, prompt tokens: {outputs[i]['prompt_tokens']}")


    # 6. Write out JSONL
    if args.output_file is not None:
        with open(args.output_file, "w", encoding="utf-8") as out_f:
            for sample, gen_texts in zip(subset, outputs):
                enriched = dict(sample)
                enriched["output"] = gen_texts["outputs"]
                out_f.write(json.dumps(enriched, ensure_ascii=False) + "\n")

    if args.push_to_hub is not None:
        dataset = load_dataset(args.dataset, split=args.split)
        dataset.push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    main()
    