from openai import OpenAI
from prompts1 import *
from utils import WIKIDATA_ENTITY, search_entity_from_wikidata
import re, os, toml
from typing import List
import json
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.toml")
with open(CONFIG_PATH, "r") as f:
    config = toml.load(f)

openai_api_key = config["openai"]["api_key"]
openai_api_base = config["openai"]["api_base"]
MODEL = config["model"]["name"]
TOPK = config["model"]["topk"]

openai_client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )


def llm_generate(messages: List):
    chat_response = openai_client.chat.completions.create(
    model=MODEL,
    messages=messages,
    temperature=0.7,
    top_p=0.8,
    max_tokens=2048
    )
    return chat_response.choices[0].message.content


MSGS = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_1},
    {"role": "assistant", "content": assistant_1},
    {"role": "user", "content": user_2},
    {"role": "assistant", "content": assistant_2},
    {"role": "user", "content": user_3},
    {"role": "assistant", "content": assistant_3},
    {"role": "user", "content": user_4},
    {"role": "assistant", "content": assistant_4},
    {"role": "user", "content": user_7},
    {"role": "assistant", "content": assistant_7},
    {"role": "user", "content": user_8},
    {"role": "assistant", "content": assistant_8}
]


def add_question(messages: List, sentence: str) -> List:
    messages = messages[:]
    text = f"<link>{sentence}</link>"
    messages.append({"role": "user", "content": text})
    return messages


def add_assistant_response(messages: List, response: str) -> List:
    messages = messages[:]
    messages.append({"role": "assistant", "content": response})
    return messages


def add_result_list(messages: List, result_list: List[WIKIDATA_ENTITY]) -> List:
    messages = messages[:]
    text = ""
    for i, entity in enumerate(result_list):
        qid = entity.qid
        label = entity.label
        desc = entity.desc
        text += f"""[{i}] {{
            "qid": "{qid}",
            "label": "{label}",
            "desc": "{desc}"
        }}\n"""
    messages.append({"role": "user", "content": text})
    return messages


def parser(response: str):
    response = response.strip()
    if '<answer>' in response and '</answer>' in response:
        start = response.find('<answer>') + len('<answer>')
        end = response.find('</answer>')
        answer_content = response[start:end].strip()
        # Extract numbers between square brackets
        indices = re.findall(r'\[(\d+)\]', answer_content)
        return [int(idx) for idx in indices]
    elif '<search>' in response and '</search>' in response:
        start = response.find('<search>') + len('<search>')
        end = response.find('</search>')
        search_content = response[start:end].strip()
        # Extract entity_name from Search(entity_name)
        entity_names = re.findall(r'Search\((.*?)\)', search_content)
        # Eliminate quotation marks at the beginning and end of each entity_name
        cleaned_entity_names = []
        for name in entity_names:
            # Remove quotation marks if they exist at both the beginning and end
            if name.startswith('"') and name.endswith('"'):
                name = name[1:-1]
            # Remove single quotes if they exist at both the beginning and end
            elif name.startswith("'") and name.endswith("'"):
                name = name[1:-1]
            cleaned_entity_names.append(name)
        return cleaned_entity_names
    else:
        print(f"Not valid response: {response}")
        return None


def entity_linking(sentence: str):
    query_msg = add_question(MSGS, sentence)
    response = llm_generate(query_msg)
    search_entity = parser(response)
    result_list = []
    if search_entity is not None:
        for each in search_entity:
            _list = search_entity_from_wikidata(each, topk=TOPK) # List[WIKIDATA_ENTITY]
            result_list.extend(_list)
    msgs = add_assistant_response(query_msg, response)
    result_msg = add_result_list(msgs, result_list)
    response = llm_generate(result_msg)
    msgs = add_assistant_response(result_msg, response)
    indices = parser(response)

    if not indices or not all(isinstance(i, int) for i in indices):
        return [], msgs

    valid_indices = []
    for idx in indices:
        if 0 <= idx < len(result_list):
            valid_indices.append(idx)
        else:
            print(f"Index {idx} is out of range (0 ~ {len(result_list) - 1})")

    return [result_list[idx] for idx in valid_indices], msgs


if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with open("/u/luoyajie/entity_linking/data/dev_1000.json", "r", encoding="utf-8") as f:
        dev_data = json.load(f)

    def process_sample(sample):
        question = sample["question"]
        gold_qids = set(sample["topic_entity"])
        predicted_entities, msgs = entity_linking(question)
        predicted_qids = set(e.qid for e in predicted_entities)
        is_correct = gold_qids ==  predicted_qids
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

    results = []
    max_workers = 20  

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_sample, sample) for sample in dev_data]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"\n==== Question: {result['question']} ====")
            print(f"Predicted_id: {result['predicted_id']}")
            print(f"Gold_id: {result['gold_id']}")
            print(f"Correct: {result['correct']}")
            print(f"LLM_messages: {result['llm_messages']}")

    output_path = "/u/luoyajie/entity_linking/results/dev1000.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    accuracy = sum(1 for r in results if r["gold_id"] == r["predicted_id"]) / len(results)
    avg_recall = sum(r["recall"] for r in results) / len(results)
    avg_precision = sum(r["precision"] for r in results) / len(results)

    print(f"\nAccuracy: {accuracy:.2%}")
    print(f"Avg Recall: {avg_recall:.2%}")
    print(f"Avg Precision: {avg_precision:.2%}")
