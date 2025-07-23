import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from question_answering import question_answering
import toml
import random
import re
import string
from collections import Counter
from typing import Dict, List
from tenacity import retry, stop_after_attempt, wait_exponential


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def eval_answer(prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    return em, f1, prec, recall


def update_answer(prediction, golds):
    max_em, max_f1, max_prec, max_recall = 0, 0, 0, 0

    for gold in golds:
        em, f1, prec, recall = eval_answer(prediction, gold)

        max_em = max(max_em, em)
        max_f1 = max(max_f1, f1)
        max_prec = max(max_prec, prec)
        max_recall = max(max_recall, recall)

    return max_em, max_f1, max_prec, max_recall


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def process_triviaqa(sample):
    question = sample["Question"]
    answers = sample["Answer"]["NormalizedAliases"] + [sample["Answer"]["NormalizedValue"]]
    pred_ans, jug_msgs, ent_msgs, qa_msgs, doc = question_answering(question)
    em, f1, _, _ = update_answer(pred_ans, answers)
    from utils import bm25_retriever
    try:
        gold_doc = bm25_retriever.text_dict[sample["Answer"]["MatchedWikiEntityName"]]
    except Exception as e:
        print(e)
        gold_doc = ""
    return {
        "_id": sample["QuestionId"],
        "question": question,
        "pred_ans": pred_ans,
        "answers": answers,
        "hit@1": gold_doc == doc,
        "em": em,
        "f1": f1,
        "jug_msgs": jug_msgs,
        "ent_msgs": ent_msgs,
        "qa_msgs": qa_msgs
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
    if dataset == "TriviaQA":
        process_sample = process_triviaqa
        data = data["Data"]
        # preserve data who have ['Answer']['MatchedWikiEntityName']
        data = [sample for sample in data if "Answer" in sample and "MatchedWikiEntityName" in sample["Answer"]]
        assert split == "dev" # for TriviaQA, we only use dev set
    seed = 42
    random.seed(seed)
    try:
        data = random.sample(data, num_data)
    except ValueError:
        pass

    results = []
    running_em = 0
    running_f1 = 0
    running_hit1 = 0
    res = process_sample(data[0])
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_sample, sample) for sample in data]
        for future in tqdm(as_completed(futures), total=len(data), desc="Processing samples"):
            try:
                result = future.result()
            except Exception as e:
                print(e)
            results.append(result)
            # Update running metrics
            running_em += (1 if result['em'] else 0)
            running_f1 += result['f1']
            running_hit1 += (1 if result['hit@1'] else 0)
            current_count = len(results)
            
            # Calculate current averages
            current_em = running_em / current_count
            current_f1 = running_f1 / current_count
            current_hit1 = running_hit1 / current_count
            print(f"\n==== Question: {result['question']} ====")
            print(f"Running EM: {current_em:.2%}")
            print(f"Running F1: {current_f1:.2%}")
            print(f"Running Hit@1: {current_hit1:.2%}")
    
    output_path = f"./results/{model}__{dataset}__{split}_{source}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nEM: {current_em:.2%}")
    print(f"F1: {current_f1:.2%}")
    print(f"Hit@1: {current_hit1:.2%}")