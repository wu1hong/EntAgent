import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from entity_linking import entity_linking
import toml
import random


def process_sample(sample):
    question = sample["question"]
    gold_qids = set(sample["topic_entity"])
    predicted_entities, msgs = entity_linking(question)
    predicted_qids = set(e.qid for e in predicted_entities)
    is_correct = gold_qids == predicted_qids
    tp = len(gold_qids & predicted_qids)
    recall = tp / len(gold_qids) if gold_qids else 0
    precision = tp / len(predicted_qids) if predicted_qids else 0
    return {
        "_id": sample["_id"],
        "question": question,
        "gold_id": list(gold_qids),
        "predicted_id": list(predicted_qids),
        "correct": is_correct,
        "recall": recall,
        "precision": precision,
        "llm_messages": msgs
    }


if __name__ == "__main__":
    config = toml.load("config.toml")
    dataset = config['exp']['dataset']
    split = config['exp']['split']
    topk = config['model']['topk']
    source = config['model']['source']
    num_workers = config['exp']['num_workers']
    num_data = config['exp']['num_data']
    model = config['model']['name']
    with open(f"./data/{dataset}/{split}.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    seed = 42
    random.seed(seed)
    data = random.sample(data, num_data)

    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_sample, sample) for sample in data]
        for future in tqdm(as_completed(futures), total=len(data), desc="Processing samples"):
            result = future.result()
            results.append(result)
            print(f"\n==== Question: {result['question']} ====")
            print(f"Predicted_id: {result['predicted_id']}")
            print(f"Gold_id: {result['gold_id']}")
            print(f"Correct: {result['correct']}")
            # print(f"LLM_messages: {result['llm_messages']}")

    output_path = f"./results/{model}_{split}_{source}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    accuracy = sum(1 for r in results if r["gold_id"] == r["predicted_id"]) / len(results)
    avg_recall = sum(r["recall"] for r in results) / len(results)
    avg_precision = sum(r["precision"] for r in results) / len(results)

    print(f"\nAccuracy: {accuracy:.2%}")
    print(f"Avg Recall: {avg_recall:.2%}")
    print(f"Avg Precision: {avg_precision:.2%}")