import base64
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1"
)

MODEL_NAME = "Qwen3.6-27B"
MAX_RETRIES = 3
NUM_WORKERS = 64


def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")


def call_model(question, image_bytes):
    image_base64 = encode_image(image_bytes)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "Answer the question based on the given image. Reply with a short word or phrase only."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                },
                            },
                        ],
                    },
                ],
                # temperature=0.0,
                timeout=300
            )
            # print(response)

            return response.choices[0].message.content.strip()

        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                return ""
            continue


def process_row(row):
    question = row["question"]
    image = row["image"]
    answer = row["answer"]

    pred = call_model(question, image)

    return {
        "question": question,
        "predicted_answer": pred,
        "answer": answer
    }


def main(input_parquet, output_jsonl):
    df = pd.read_parquet(input_parquet)

    results = []

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_row, row) for _, row in df.iterrows()]

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results.append(result)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    input_parquet = "../data.parquet"
    output_jsonl = MODEL_NAME + ".jsonl"
    main(input_parquet, output_jsonl)
