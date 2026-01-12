import jsonlines
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import json
import os
from tqdm import tqdm
import argparse


def process_data(json_filename, file_name, llm, batch_size, tokenizer, sampling_params):
    data = []
    with jsonlines.open(json_filename) as infile:
        print(f"Loading data from {json_filename}")
        for item in infile:
            group = {}
            problem_key = next((key for key in ['problem', 'question', 'input', 'content'] if key in item), None)
            answer_key = next((key for key in ['answer', 'target', 'solution', 'ground_truth'] if key in item), None)
            if not problem_key or not answer_key:
                continue
            prompt_ori = f"Please reason step by step, and put your final answer within \\boxed{{}}.{problem}"
            group["prompt_ori"] = prompt_ori
            group["answer"] = item[answer_key]
            data.append(group)

    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt["prompt_ori"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in data
    ]

    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    split_texts = split_list(texts, batch_size)
    results = []

    for item in tqdm(split_texts, desc=f"Processing {json_filename}"):
        outputs = llm.generate(item, sampling_params)
        for output in outputs:
            generated_list = [o.text for o in output.outputs]
            results.append(generated_list)

    for result, item in zip(results, data):
        item["output"] = result  

    with open(file_name, "w") as file:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            file.write(json_line + "\n")
    print(f"✅ 数据已成功写入 {file_name}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--input_files", nargs='+', required=True, help="List of input JSONL files")
    parser.add_argument("--output_files", nargs='+', required=True, help="List of output JSONL files")
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--n", type=int, default=1, help="Number of samples to generate per prompt")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--max_tokens", type=int, default=4096)
    args = parser.parse_args()

    if len(args.input_files) != len(args.output_files):
        raise ValueError("The number of input and output files must match!")

    print("🚀 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    sampling_params = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
    )
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=0.8,
        max_model_len=args.max_tokens,
        trust_remote_code=True,
        tensor_parallel_size=1,
    )

    for input_path, output_path in zip(args.input_files, args.output_files):
        process_data(input_path, output_path, llm, args.batch_size, tokenizer, sampling_params)


if __name__ == "__main__":
    main()
