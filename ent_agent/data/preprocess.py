import os
import json
import random
import Stemmer
import bm25s
from tqdm import tqdm
from bs4 import BeautifulSoup
from FlagEmbedding import FlagModel
import svs
import numpy as np
import toml

CONFIG_PATH = os.path.join(".", "config.toml")
with open(CONFIG_PATH, "r") as f:
    config = toml.load(f)
MODEL_NAME = config["dense_model"]["dense_name"]

def process_2wiki_helper(split:str):
    """
    Create entity linking file for dataset.
    for 2wiki
    Inputs:
            split: str    could only be ['train', 'dev', 'test']
    """
    dataset_name = '2wiki'

    file_path = f"./{dataset_name}/{split}.json"
    with open(file_path, 'r') as f:
        datas = json.load(f)
    
    with tqdm(total=len(datas), desc="Processing", unit="item") as pbar:
        for data in datas:
            type = data['type']
            ent_ids = data['entity_ids']
            if type == 'compositional':
                ent_list = [ent_ids.split('_')[0]]
            elif type == 'comparison':
                ent_list = ent_ids.split('_')
            elif type == 'bridge_comparison':
                ent_list = ent_ids.split('_')[:2]
            elif type == 'inference':
                ent_list = [ent_ids.split('_')[0]]
            else:
                raise TypeError
            
            data['topic_entity'] = ent_list

            pbar.update(1)
    
    with open(file_path, 'w') as o_file:
        json.dump(datas, o_file, indent=4)


def process_2wiki():
    process_2wiki_helper('train')
    process_2wiki_helper('dev')
    process_2wiki_helper('test')


def process_triviaqa():
    text_dict = {} # {title: text}
    folder_path = "./TriviaQA/evidence/wikipedia"
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    text = f.read()
                    name = file_path.split(folder_path)[-1][1:-4].replace('_', ' ') # Remove .txt extension
                    if "Wikipedia:" in name:
                        continue
                    text_dict[name] = text

    # bm25
    stemmer = Stemmer.Stemmer("english")
    # Tokenize the corpus and only keep the ids (faster and saves memory)
    corpus_tokens = bm25s.tokenize(text_dict.values(), stopwords="en", stemmer=stemmer)
    # Create the BM25 model and index the corpus
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    index_directory = os.path.join("./TriviaQA/", "bm25_index")
    os.makedirs(index_directory, exist_ok=True)
    retriever.save(index_directory)

    # dense retriever
    # MODEL_NAME = "BAAI/bge-large-en-v1.5"
    index_directory = os.path.join("./TriviaQA/", "dense_index")
    os.makedirs(index_directory, exist_ok=True)
    # Setting use_fp16 to True speeds up computation with a slight performance degradation
    model = FlagModel(MODEL_NAME, use_fp16=True, device="cuda:0") # [default] normalize the embeddings
    corpus_embeddings = model.encode(text_dict.values())
    print("-"*50, "Encoding done", "-"*50)

    # create the faiss index and store the corpus embeddings into the vector space
    parameters = svs.VamanaBuildParameters(
        graph_max_degree = 64,
        window_size = 128,
    )
    corpus_embeddings = corpus_embeddings.astype(np.float32)
    index = svs.Vamana.build(
        parameters,
        corpus_embeddings,
        svs.DistanceType.L2,
        num_threads = 256,
    )
    # save dense index
    index.save(
        os.path.join(index_directory, "triviaqa_text_config"),
        os.path.join(index_directory, "triviaqa_text_graph"),
        os.path.join(index_directory, "triviaqa_text_data"),
    )
    
    # save text_dict to json
    with open("./TriviaQA/text_dict.json", "w") as f:
        json.dump(text_dict, f, indent=4)

    # add "topic entity" to triviaqa
    file_path = "./TriviaQA/dev.json"
    with open(file_path, 'r') as f:
        datas = json.load(f)
    for data in datas["Data"]:
        entity_pages = data['EntityPages']
        topic_entity = []
        for page in entity_pages:
            title = page['Title'] # for wikipedia, title is unique, so we can use it as ID
            topic_entity.append(title)
        data['topic_entity'] = topic_entity

    with open(file_path, 'w') as o_file:
        json.dump(datas, o_file, indent=4)

def process_popqa():
    text_dict = {}  # {title: text}
    folder_path = "./PopQA/evidence/wikipedia"
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    name = file_path.split(folder_path)[-1][1:-4].replace('_', ' ')
                    if "Wikipedia:" in name:
                        continue
                    text_dict[name] = text

    print(f" Loaded {len(text_dict)} valid Wikipedia articles")

    # Build BM25 index
    stemmer = Stemmer.Stemmer("english")
    corpus_tokens = bm25s.tokenize(text_dict.values(), stopwords="en", stemmer=stemmer)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    bm25_dir = os.path.join("./PopQA/", "bm25_index")
    os.makedirs(bm25_dir, exist_ok=True)
    retriever.save(bm25_dir)

    # Build Dense Embedding Index
    dense_dir = os.path.join("./PopQA/", "dense_index")
    os.makedirs(dense_dir, exist_ok=True)

    model = FlagModel(MODEL_NAME, use_fp16=True, device="cuda:0")
    corpus_embeddings = model.encode(text_dict.values())
    print("-" * 50, "Dense encoding complete", "-" * 50)

    parameters = svs.VamanaBuildParameters(
        graph_max_degree=64,
        window_size=128,
    )
    corpus_embeddings = corpus_embeddings.astype(np.float32)
    index = svs.Vamana.build(
        parameters,
        corpus_embeddings,
        svs.DistanceType.L2,
        num_threads=256,
    )

    index.save(
        os.path.join(dense_dir, "popqa_text_config"),
        os.path.join(dense_dir, "popqa_text_graph"),
        os.path.join(dense_dir, "popqa_text_data"),
    )

    # Save text_dict
    with open("./PopQA/text_dict.json", "w", encoding="utf-8") as f:
        json.dump(text_dict, f, indent=4)

    # Add "topic_entity" field to JSONL
    file_path = "./PopQA/test.jsonl"
    updated_entries = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            title = entry.get("s_wiki_title", "").strip()
            if title in text_dict:
                entry["topic_entity"] = [title]

                try:
                    answer_list = json.loads(entry["possible_answers"])
                except:
                    answer_list = entry.get("possible_answers", [])

                original_question = entry["question"]
                entity_surface = entry["subj"]
                wiki_entity = entry["s_wiki_title"]

                if wiki_entity and entity_surface and wiki_entity in original_question:
                    entry["question"] = original_question.replace(wiki_entity, entity_surface)

                if isinstance(answer_list, list) and answer_list:
                    entry["answer"] = random.choice(answer_list)
                else:
                    entry["answer"] = entry.get("obj", "")

                updated_entries.append(entry)

    with open(file_path, "w", encoding="utf-8") as f:
        for entry in updated_entries:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    # process_2wiki()
    # process_triviaqa()
    process_popqa()
