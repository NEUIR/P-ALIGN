import os
import json
import signal
from tqdm import tqdm
from math_verify import parse, verify

from oat_math_grader import boxed_reward_fn as oat_evaluate


def timeout(seconds: int = 10):
    """
    A decorator to enforce timeouts on function execution (POSIX only).
    Useful for preventing sympy verification from hanging.

    Args:
        seconds (int): Maximum seconds before timeout.

    Returns:
        Callable: Wrapped function that raises TimeoutError after `seconds`.
    """
    def decorator(func):
        def handler(signum, frame):
            raise TimeoutError("Verification timed out.")
        def wrapper(*args, **kwargs):
            if os.name != "posix":
                return func(*args, **kwargs)
            old_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        return wrapper
    return decorator


@timeout(seconds=10)
def label_with_math_verify(preds: list[str], golden: str):
    """
    Perform symbolic verification using math_verify.

    Args:
        preds (list[str]): List of model outputs (LaTeX strings).
        golden (str): Ground truth answer string.

    Returns:
        tuple[list[int], list[str]]: Binary correctness labels (0/1),
                                     and parsed answers.
    """
    parsed_preds = list(map(parse, preds))
    parsed_golden = list(map(parse, ["$" + golden + "$"] * len(preds)))

    try:
        labels = list(map(verify, parsed_golden, parsed_preds))
    except Exception:
        labels = [0] * len(preds)

    return [int(x) for x in labels], parsed_preds


def safe_oat_eval(pred: str, golden: str):
    """
    Use OAT evaluator for fallback grading.

    Args:
        pred (str): Model output string.
        golden (str): Ground truth answer.

    Returns:
        int: 1 if correct, else 0.
    """
    try:
        _, result = oat_evaluate(pred, golden, fast=False)
        return int(result == 1.0)
    except Exception:
        return 0


def evaluate_jsonl(input_file: str,
                   output_file: str,
                   use_oat: bool = True,
                   any_true: bool = True):
    """
    Evaluate a JSONL dataset of math problems and write verified results.

    Args:
        input_file (str): Path to input JSONL file.
        output_file (str): Path to output JSONL file.
        use_oat (bool): Whether to use OAT fallback evaluator.
        any_true (bool): If True, combine results via OR logic (math_verify ∨ OAT).

    Returns:
        None
    """
    results = []
    with open(input_file, "r") as f:
        data = [json.loads(line) for line in f]

    print(f"🔍 Evaluating {len(data)} records from {input_file} ...")

    for item in tqdm(data):
        answer = item.get("answer", "")
        answer = str(answer).strip()
        # outputs = item.get("output", [])
        outputs = item.get("output")
        if not outputs:
            continue

        try:
            labels, parsed_outputs = label_with_math_verify(outputs, answer)
        except Exception:
            labels = [0] * len(outputs)
            parsed_outputs = [""] * len(outputs)

        # Apply OAT fallback
        if use_oat:
            oat_labels = [safe_oat_eval(o, answer) for o in outputs]
            if any_true:
                labels = [int(l or o) for l, o in zip(labels, oat_labels)]
            else:
                labels = oat_labels

        passn = int(any(labels))
        item.update({
            "label": labels,
            "passn": passn,
            "output_ans": parsed_outputs,
        })
        results.append(item)

    with open(output_file, "w") as f:
        for r in results:
            # f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    total = len(results)
    pass_count = sum(1 for r in results if any(r.get("label", [])))
    pass_at_n = pass_count / total if total > 0 else 0.0
    # avg_acc = sum(r.get("label", False) for r in results) / total if total > 0 else 0.0
    all_answers_count = sum(len(r.get("label", [])) for r in results)
    all_answers_correct = sum(sum(r.get("label", [])) for r in results)
    avg_answer_acc = all_answers_correct / all_answers_count if all_answers_count > 0 else 0.0
    
    
    print(f"pass@3: {pass_at_n:.4f}")
    print(f"acc@3: {avg_answer_acc:.4f}")
    print(f"pass@1=acc@1: {pass_at_1:.4f}")

if __name__ == "__main__":

    input_path = "your input file path"
    output_path = "your output file path"

    evaluate_jsonl(
        input_file=input_path,
        output_file=output_path,
        use_oat=True,   
        any_true=True    
    )
