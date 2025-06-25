from openai import OpenAI
from utils import WIKIDATA_ENTITY, search_entity_from_wikidata, search_entity_from_wikipedia
import re, os, toml
from typing import List
from pprint import pprint

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.toml")
with open(CONFIG_PATH, "r") as f:
    config = toml.load(f)

openai_api_key = config["openai"]["api_key"]
openai_api_base = config["openai"]["api_base"]
MODEL = config["model"]["name"]
TOPK = config["model"]["topk"]
SOURCE = config["model"]["source"]

# if SOURCE == "wikidata":
#     from prompts import *
# elif SOURCE == "wikipedia":
#     from prompts_wikipedia import *
from prompts import *

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
    max_tokens=4096,
    extra_body={
        "repetition_penalty": 1.05,
    },
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
    {"role": "user", "content": user_5},
    {"role": "assistant", "content": assistant_5},
    {"role": "user", "content": user_6},
    {"role": "assistant", "content": assistant_6},
    {"role": "user", "content": user_7},
    {"role": "assistant", "content": assistant_7},
    {"role": "user", "content": user_8},
    {"role": "assistant", "content": assistant_8},
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
        text += f"""[{i}]
            {label}: {desc}\n"""
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


def search_func(entity_name: str, topk: int):
    if SOURCE == "wikidata":
        _list = search_entity_from_wikidata(entity_name, topk=topk) # List[WIKIDATA_ENTITY]
    elif SOURCE == "wikipedia":
        _list = search_entity_from_wikipedia(entity_name, topk=3) # List[WIKIPEDIA_ENTITY]
    else:
        raise NotImplementedError(f"Source {SOURCE} is not supported.")
    return _list


def entity_linking(sentence: str):
    query_msg = add_question(MSGS, sentence)
    response = llm_generate(query_msg)
    search_entity = parser(response)
    result_list = []
    if search_entity is not None:
        for each in search_entity:
            _list = search_func(each, topk=TOPK) # List[WIKIDATA_ENTITY]
            result_list.extend(_list)

    # filter out results without description
    # result_list = [e for e in result_list if e.desc is not None]

    # filter out results without label
    # result_list = [e for e in result_list if e.label is not None]
    
    msgs = add_assistant_response(query_msg, response)
    result_msg = add_result_list(msgs, result_list)
    response = llm_generate(result_msg)
    msgs = add_assistant_response(result_msg, response)
    indices = parser(response)

    # remove prompts for better visualization
    msgs = msgs[:1] + msgs[17:]

    if len(result_list) == 1:
        return result_list, msgs

    if not indices or not all(isinstance(i, int) for i in indices):
        return [], msgs

    valid_indices = []
    for idx in indices:
        if 0 <= idx < len(result_list):
            valid_indices.append(idx)
        else:
            print(f"Warning: Index {idx} is out of range (0 ~ {len(result_list) - 1})")

    return [result_list[idx] for idx in valid_indices], msgs


if __name__ == "__main__":
    sentence = "Where was the place of death of Anastasia Of Serbia's husband?"
    sentence = "Where does the founder of Les Films Du Losange work at?"
    sentence = "Are director of film Susanna Whipped Cream and director of film Le Salamandre both from the same country?"
    result = entity_linking(sentence)
    print(result)
    breakpoint()