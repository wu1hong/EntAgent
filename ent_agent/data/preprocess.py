import os
import json
from tqdm import tqdm
import bm25s
import Stemmer
from FlagEmbedding import FlagModel
import svs
import numpy as np


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
    MODEL_NAME = "BAAI/bge-large-en-v1.5"
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


if __name__ == "__main__":
    # process_2wiki()
    process_triviaqa()