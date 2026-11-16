import os
import json
import jsonlines
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

judge_prompt = """
You are a rigorous question-and-answer quality assessment expert. Given a question, your task is to compare the "ground truth" and the "model answer", determine whether they are semantically equivalent, and output ONLY "yes" or "no".

question: {}
ground truth: {}
model answer: {}
"""

BASE_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"

data_path = "./result/Qwen3.5-27B.jsonl"
judge_path = "./judge"

MAX_WORKERS = 64


client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY
)



def LLM_as_Judge(question, gt, answer):
    try:
        completion = client.chat.completions.create(
            model="Qwen3.6-27B",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": judge_prompt.format(question, gt, answer)},
                    ],
                },
            ],
            temperature=0,
        )

        res = completion.choices[0].message.content.strip().lower()

        if res == "yes":
            return "yes"

        return "no"

    except Exception as e:
        print(f"\n[ERROR] {e}")
        return "no"


def process_item(item):
    answer = item['predicted_answer'].strip()
    question = item['question'].strip()
    gt = item['answer']

    if len(gt) > 1:
        gt = "; ".join(gt)
    else:
        gt = gt[0]

    if answer.lower() == gt.lower():
        return {
            "question": question,
            "gt": gt,
            "answer": answer,
            "judge": "yes",
            "correct": 1
        }

    if answer.lower() == "yes" and gt.lower() == "no":
        return {
            "question": question,
            "gt": gt,
            "answer": answer,
            "judge": "no",
            "correct": 0
        }

    if answer.lower() == "no" and gt.lower() == "yes":
        return {
            "question": question,
            "gt": gt,
            "answer": answer,
            "judge": "no",
            "correct": 0
        }

    res = LLM_as_Judge(question, gt, answer)

    return {
        "question": question,
        "gt": gt,
        "answer": answer,
        "judge": res,
        "correct": 1 if res == "yes" else 0
    }


if __name__ == '__main__':

    os.makedirs(judge_path, exist_ok=True)

    data = []

    with jsonlines.open(data_path) as reader:
        for item in reader:
            data.append(item)

    total = len(data)

    ans = []
    corr = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = [
            executor.submit(process_item, item)
            for item in data
        ]

        for future in tqdm(
            as_completed(futures),
            total=total,
            desc="Judging"
        ):

            result = future.result()

            corr += result["correct"]

            ans.append({
                "question": result["question"],
                "gt": result["gt"],
                "answer": result["answer"],
                "judge": result["judge"]
            })

    print(f"\nCorrect answers: {corr}")
    print(f"Total questions: {total}")
    print(f"Accuracy: {corr / total:.4f}")

    output_path = os.path.join(
        judge_path,
        data_path.split('/')[-1]
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        for aa in ans:
            json_line = json.dumps(aa, ensure_ascii=False)
            f.write(json_line + '\n')

    print(f"\nSaved to: {output_path}")