from openai import OpenAI
import re, os, toml
from typing import List
from pprint import pprint
from prompts_qa import *
from entity_linking import entity_linking
from FlagEmbedding import FlagModel
import torch
import tiktoken
from utils import search_entity_from_bm25, search_entity_from_dense

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.toml")
with open(CONFIG_PATH, "r") as f:
    config = toml.load(f)

openai_api_key = config["openai"]["api_key"]
openai_api_base = config["openai"]["api_base"]
MODEL = config["model"]["name"]
DATASET = config["exp"]["dataset"]
TOPK = config["model"]["topk"]
SOURCE = config["model"]["source"]
IF_FT = config["model"]["if_ft"]
METHOD = config["model"]["method"]
MODEL_NAME = config["dense_model"]["dense_name"]
DEVICE = "cuda:0"

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
    max_tokens= 2048,
    extra_body={
        "repetition_penalty": 1.05,
    },
    )
    return chat_response.choices[0].message.content


QA_MSGS = [
    {"role": "system", "content": qa_system_prompt},
]


def qa_popqa(question: str, context: str):
    messages = QA_MSGS.copy()
    context = context[:7000]

    text = f"Question: {question}\nContext: {context}"
    messages.append({"role": "user", "content": text})

    response = llm_generate(messages)

    try:
        answer = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL).group(1).strip()
    except AttributeError:
        print(" No <answer> tag in the response.")
        answer = "N/A"

    messages.append({"role": "assistant", "content": response})
    return answer, messages


class Reranker:
    def __init__(self, model_name: str, device: str):
        self.reranker = FlagModel(model_name, use_fp16=True, device=device)

    def encode(self, passages: List[str]) -> torch.Tensor:
        return self.reranker.encode(passages)

reranker = Reranker(MODEL_NAME, DEVICE)

def qa_triviaqa(question: str, context: str, topk: int = 3):
    passages = context.split("\n\n")
    _passages = [passage.strip() for passage in passages if passage.strip()] + [question]
    vectors = reranker.encode(_passages)
    query_vec = vectors[-1]
    passage_vec = vectors[:-1]
    scores = query_vec @ passage_vec.T
    topk = min(topk, len(passages))
    _, indices = torch.topk(torch.tensor(scores), k=topk)
    context_lst = [passages[idx] for idx in indices]
    messages = QA_MSGS.copy()
    text = f"Question: {question}\nContext: {context_lst}"
    messages.append({"role": "user", "content": text})
    response = llm_generate(messages)
    try:
        answer = re.search(r"<answer>(.*)</answer>", response).group(1)
    except AttributeError:
        print("No <answer> tag in the response")
        answer = "N/A"
    messages.append({"role": "assistant", "content": response})
    return answer.strip(), messages



def plain_qa(question: str) -> str:
    messages = [{"role": "system", "content": plain_qa_system_prompt}]
    messages.append({"role": "user", "content": question})
    response = llm_generate(messages)
    try:
        answer = re.search(r"<answer>(.*)</answer>", response).group(1)
    except AttributeError:
        print("No <answer> tag in the response")
        answer = "N/A"
    messages.append({"role": "assistant", "content": response})
    return answer.strip(), messages



def question_answering(question: str, topk: int = 10):

    if DATASET == "PopQA":
        qa = qa_popqa
    elif DATASET == "TriviaQA":
        qa = qa_triviaqa
    else:
        raise ValueError(f"Unsupported dataset for QA: {DATASET}")

    ent_lst = []
    ent_msgs = []
    if METHOD == "baseline":
        pass
    elif METHOD == "BM25":
        ent_lst = search_entity_from_bm25(question, topk)

    elif METHOD == "dense":
        ent_lst = search_entity_from_dense(question, topk)

    elif METHOD == "EL":
        ent_lst, ent_msgs = entity_linking(question, if_dense=False)
        
    else:
        raise ValueError(f"Unsupported method: {METHOD}")

    
    if not ent_lst:
        answer, qa_msgs = plain_qa(question)
        doc = ""
    else:
        answer, qa_msgs = qa(question, ent_lst[0].doc)
        doc = ent_lst[0].doc

    return answer, ent_msgs, qa_msgs, doc



if __name__ == "__main__":
    question = "What is Frequency's occupation?"
    # print(not_use_dense)
    # res = entity_linking(question, if_dense=not not_use_dense)
    # res = entity_linking(question, if_dense=True)
    # question = "How old was Woody Herman when he founded his own orchestra?"
    answer, plain_msgs = plain_qa(question)
    answer, ent_msgs, qa_msgs, doc = question_answering(question)
    print(answer)
    # print(ent_msgs)
    # print(qa_msgs)
    breakpoint()

