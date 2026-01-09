import jsonlines
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import time
import json
import re
import torch
import os
import csv
from tqdm import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"


def process_data(json_filename, file_name, llm, batch_size, tokenizer, sampling_params):

    # --- 读取输入数据 ---
    with jsonlines.open(json_filename) as infile:
        data = []
        for item in infile:
            question = item['question']
            sufficient_reasoning = item['sufficient_reasoning']

            prompt = (
                f"Please continue from the draft and solve the problem step by step, and put your final answer within \\boxed{{}}. "
                f"I will provide you with some prior knowledge as a draft to assist you in solving the question."
                f"*Question*:{question}\n"
                f"*Prefix*:{sufficient_reasoning}"
            )

            data.append({
                'question': question,
                'sufficient_reasoning': sufficient_reasoning,
                'prompt': prompt,
            })

    # --- 如果文件已存在，跳过已完成部分 ---
    existing = set()
    if os.path.exists(file_name):
        with open(file_name, "r") as f:
            for line in f:
                try:
                    existing_item = json.loads(line)
                    existing.add(existing_item['question'])
                except Exception:
                    continue
        print(f"[Resume] Found {len(existing)} existing entries. Will skip them.")

    # --- 按 batch 生成 ---
    total_batches = (len(data) + batch_size - 1) // batch_size
    print(f"Total {len(data)} samples, batch_size={batch_size}, total_batches={total_batches}")

    # 使用 append 模式写入
    with open(file_name, "a", encoding="utf-8") as file:
        for batch_idx in tqdm(range(total_batches), total=total_batches, desc="Generating"):
            start, end = batch_idx * batch_size, (batch_idx + 1) * batch_size
            batch_data = [d for d in data[start:end] if d['question'] not in existing]

            if not batch_data:
                continue

            # --- 构造输入 ---
            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": d['prompt']}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                for d in batch_data
            ]

            # --- 调用生成 ---
            try:
                outputs = llm.generate(texts, sampling_params)
            except Exception as e:
                print(f"[Error] Batch {batch_idx} generation failed: {e}")
                continue

            # --- 保存结果 ---
            for output, item in zip(outputs, batch_data):
                result_text = output.outputs[0].text
                item['output'] = result_text
                json_line = json.dumps(item, ensure_ascii=False)
                file.write(json_line + "\n")

            file.flush()  # 确保每个 batch 都立即落盘
            os.fsync(file.fileno())

            print(f"[Saved] Batch {batch_idx+1}/{total_batches} ({len(batch_data)} items) written.")

    print("✅ All data processed and saved successfully.")
    
def main():
    model="/data2/wuzhuoyang/model/Qwen3-14b"
    json_filename = "/data2/wuzhuoyang/Draft/llm_based/data/draft_to_cot/qwen3-14b/output_results_part3_300.jsonl"
    file_name = "/data2/wuzhuoyang/Draft/llm_based/data/cot/qwen3-14b-based-14b-cot/cot-3.jsonl"
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    sampling_params = SamplingParams(n=1, temperature=0.6, top_p=0.9, repetition_penalty=1.05, max_tokens=32768)
    llm = LLM(model=model, gpu_memory_utilization=0.8, max_model_len=32768, trust_remote_code=True, tensor_parallel_size=1)
    batch_size = 500
    process_data(json_filename, file_name, llm, batch_size, tokenizer, sampling_params)

if __name__ == "__main__":
    main()
