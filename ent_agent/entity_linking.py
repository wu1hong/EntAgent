from openai import OpenAI
from utils import WIKIDATA_ENTITY, search_entity_from_wikidata, search_entity_from_wikipedia, search_entity_from_bm25, search_entity_from_dense
import re, os, toml
from typing import List
from pprint import pprint
from prompts import *
import tiktoken


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.toml")
with open(CONFIG_PATH, "r") as f:
    config = toml.load(f)

openai_api_key = config["openai"]["api_key"]
openai_api_base = config["openai"]["api_base"]
MODEL = config["model"]["name"]
TOPK = config["model"]["topk"]
SOURCE = config["model"]["source"]
IF_FT = config["model"]["if_ft"]

openai_client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
# openai_client = OpenAI(
#         api_key=openai_api_key,
#     )


def count_message_tokens(messages: list, model: str = "gpt-4") -> int:
    """Returns the number of tokens in a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    for message in messages:
        # Each message incurs a small overhead of tokens
        num_tokens += 4 
        for key, value in message.items():
            num_tokens += len(encoding.encode(str(value)))
            if key == "name":
                num_tokens -= 1  # A specific rule for the 'name' field
    num_tokens += 2  # Every reply is primed with <|im_start|>assistant
    return num_tokens


def llm_generate(messages: List):
    chat_response = openai_client.chat.completions.create(
    model=MODEL,
    messages=messages,
    temperature=config["gen_param"]["temp"],
    top_p=config["gen_param"]["top_p"],
    max_tokens=config["gen_param"]["max_tokens"],
    extra_body={
        "repetition_penalty": config["gen_param"]["rep_penalty"],
    },
    )
    return chat_response.choices[0].message.content

if not IF_FT:
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
else:
    # MSGS = [
    #     {"role": "system", "content": system_prompt},
    # ]
    MSGS = []


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


def search_func(entity_name: str, topk: int, if_dense: bool = False):
    if SOURCE == "wikidata":
        _list = search_entity_from_wikidata(entity_name, topk=topk) # List[WIKIDATA_ENTITY]
    elif SOURCE == "wikipedia":
        _list = search_entity_from_wikipedia(entity_name, topk=3) # List[WIKIPEDIA_ENTITY]
    elif SOURCE == "hybrid":
        if if_dense:
            _list = search_entity_from_dense(entity_name, topk=topk)
        else:
            _list = search_entity_from_bm25(entity_name, topk=topk)
    else:
        raise NotImplementedError(f"Source {SOURCE} is not supported.")
    return _list


def entity_linking(sentence: str, if_dense: bool = False):
    query_msg = add_question(MSGS, sentence)
    response = llm_generate(query_msg)
    search_entity = parser(response)
    # sanity check
    if len(search_entity) > 10:
        raise ValueError(f"Too many entities: {search_entity}")
    
    result_list = []
    if search_entity is not None:
        for each in search_entity:
            _list = search_func(each, topk=TOPK, if_dense=if_dense) # List[WIKIDATA_ENTITY]
            result_list.extend(_list)

    # filter out results without description
    # result_list = [e for e in result_list if e.desc is not None]

    # filter out results without label
    # result_list = [e for e in result_list if e.label is not None]

    # if not result_list:
    #     return [], query_msg
    
    msgs = add_assistant_response(query_msg, response)
    result_msg = add_result_list(msgs, result_list)
    # sanity check for token count
    if count_message_tokens(result_msg) > 2000:
        raise ValueError(f"Token count exceeds 2000 for sentence: {sentence} in entity linking")
    
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
    sentence = "Where in England was Dame Judi Dench born?"
    sentence = "Melanie Molitor is the mom of which tennis world NO 1?"
    sentence = "What Canadian religion has a religious notable figure named Mary?"
    sentence = "What language with the initials arn do Chilean people speak?"
    sentence = "What is the capital of the state of California?"
    sentence = "Where is the location of the country where the Greelandic language is spoken?"
    result = entity_linking(sentence)
    print(result)
    breakpoint()