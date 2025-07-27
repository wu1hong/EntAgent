import requests
from typing import List
from bs4 import BeautifulSoup
import wikipedia
import toml
import os
import bm25s
import Stemmer
import json
import svs
from FlagEmbedding import FlagModel
import re


CONFIG_PATH = os.path.join(".", "config.toml")
with open(CONFIG_PATH, "r") as f:
    config = toml.load(f)
DATASET = config["exp"]["dataset"]
MODEL_NAME = "BAAI/bge-large-en-v1.5"
DEVICE = "cuda:0"


class WIKIDATA_ENTITY:
    # data type for wikidata entity
    def __init__(self, qid, label, desc, doc=None) -> None:
        """
        qid: wikidata QID, e.g., Q16204382
        label: label of the entity, e.g., Michael Lent
        desc: short description of the entity, e.g., American writer and producer
        """
        self.qid = qid if qid else None
        self.label = label if label else None
        self.desc = desc if desc else None
        self.doc = doc if doc else None
    
    def __repr__(self):
        text = f"QID: {self.qid}\nLabel: {self.label}\nDescription: {self.desc}\n"
        return text


def normalize_text(text):
    """Converts text to a clean set of words."""
    # Convert to lowercase
    text = text.lower()
    # Remove content in parentheses (e.g., " (novel)", " (film)")
    text = re.sub(r'\s*\([^)]*\)$', '', text)
    # Split into words and return as a set for efficient comparison
    return set(text.split())


def search_entity_from_wikidata(query: str, topk: int = 20) -> list[WIKIDATA_ENTITY]:
    """
    search entity from wikidata
    Inputs:
        query: query string
        topk: number of results to return
    Returns:
        result_list: list of WIKIDATA_ENTITY
    """
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": query,
        "format": "json",
        "language": "en",
        "limit": topk
    }
    response = requests.get(url, params=params).json()
    search_list = response.get("search", [])
    result_list = []
    for item in search_list:
        qid = item.get('id')
        label = item.get('display', {}).get('label', {}).get('value') if 'display' in item else None
        desc = None
        if 'display' in item and item['display'] is not None:
            if 'description' in item['display'] and item['display']['description'] is not None:
                desc = item['display']['description'].get('value')
        result_list.append(WIKIDATA_ENTITY(qid, label, desc))
    return result_list


def get_wikidata_entity_from_qid(qid: str) -> WIKIDATA_ENTITY:
    """
    Return the WIKIDATA_ENTITY class given the QID.
    """
    if qid is None:
        return None
    
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": qid,
        "props": "labels|descriptions",
        "languages": "en",
        "format": "json",
        "redirect": "1"
    }
    response = requests.get(url, params=params).json()
    entity = response.get("entities", {}).get(qid, {})
    # use the new QID
    if "redirects" in entity:
        qid = entity["redirects"]['to']
    label = entity.get("labels", {}).get("en", {}).get("value", "Label not found")
    desc = entity.get("descriptions", {}).get("en", {}).get("value", "Description not found")
    wiki_entity = WIKIDATA_ENTITY(qid, label, desc)
    return wiki_entity


def from_wikipedia_title_to_wikidata_entity(title: str) -> WIKIDATA_ENTITY:
    """
    Return the WIKIDATA_ENTITY class given the Wikipedia title.
    """
    # title = title.replace("_", " ")
    url = f"https://en.wikipedia.org/wiki/{title}"    
    response = requests.get(url)
    html_text = response.text
    soup = BeautifulSoup(html_text, 'html.parser')
    title_tag = soup.find('title')
    wikipedia_title = title_tag.text.split(' - ')[0]

    # Find the Wikidata link
    link = soup.select_one('#t-wikibase a')
    if link:
        href = link.get('href')
        q_number = href.split('/')[-1]
    else:
        print("Wikidata link not found")

    return [get_wikidata_entity_from_qid(q_number)]


def search_entity_from_wikipedia(query: str, topk: int = 3, num_sentences: int = 1) -> list[WIKIDATA_ENTITY]:
    # print(f"Searching for {query} from Wikipedia...")
    search_list = wikipedia.search(query, results=topk)
    page_list = []
    for title in search_list:
        try:
            page = wikipedia.page(title, auto_suggest=False)
            page_list.append(page)
        except wikipedia.exceptions.DisambiguationError as error:
            # print(f"Error: {error}")
            for option in error.options[:topk]:
                try:
                    page = wikipedia.page(option, auto_suggest=False)
                    page_list.append(page)
                except Exception as e:
                    # print(f"Error: {e}")
                    continue
    
    entity_list = []
    for page in page_list:
        response = requests.get(page.url)
        html_text = response.text
        soup = BeautifulSoup(html_text, 'html.parser')
        link = soup.select_one('#t-wikibase a')
        try:
            href = link.get('href')
        except AttributeError as e:
            print(f"Error: {e} in search_entity_from_wikipedia")
            continue
        q_number = href.split('/')[-1]
        entity_list.append(WIKIDATA_ENTITY(q_number, page.title, '\n'.join(page.summary.split('\n')[:num_sentences])))
    
    return entity_list


class DenseRetriever:
    def __init__(self, index_dir: str, dict_path: str):
        self.index = svs.Vamana(
            os.path.join(index_dir, "triviaqa_text_config"),
            svs.GraphLoader(os.path.join(index_dir, "triviaqa_text_graph")),
            svs.VectorDataLoader(
                os.path.join(index_dir, "triviaqa_text_data"), svs.DataType.float32
            ),
            svs.DistanceType.L2,
            num_threads = 4,
        )
        self.index.search_window_size = 30
        self.encoder = FlagModel(MODEL_NAME, use_fp16=False, device=DEVICE)
        with open(dict_path, 'r') as f:
            self.text_dict = json.load(f)
        self.titles = list(self.text_dict.keys())
    
    def search(self, query: str, topk: int = 10):
        query_embedding = self.encoder.encode(query)
        indices, _ = self.index.search(query_embedding, topk)
        titles = [self.titles[index] for index in indices[0]]
        documents = [self.text_dict[title] for title in titles]
        descriptions = [self.text_dict[title].split("\n\n")[0] for title in titles]
        res_list = [WIKIDATA_ENTITY(None, title, description, doc=doc) for title, description, doc in zip(titles, descriptions, documents)]
        return res_list


class BM25Retriever:
    def __init__(self, index_dir: str, dict_path: str):
        self.index_dir = index_dir
        self.dict_path = dict_path
        self.stemmer = Stemmer.Stemmer("english")
        self.retriever = bm25s.BM25.load(index_dir, load_corpus=True)
        with open(dict_path, 'r') as f:
            self.text_dict = json.load(f)
        self.titles = list(self.text_dict.keys())
        # create a bm25 index for titles
        # Tokenize the corpus and only keep the ids (faster and saves memory)
        corpus_tokens = bm25s.tokenize(self.titles, stopwords="en", stemmer=self.stemmer)
        self.title_retriever = bm25s.BM25()
        self.title_retriever.index(corpus_tokens)

    def search(self, query: str, topk: int = 10):
        # if query in self.titles:
        #     return [WIKIDATA_ENTITY(None, query, self.text_dict[query].split("\n\n")[0], doc=self.text_dict[query])]
        # query_tokens = bm25s.tokenize(query, stemmer=self.stemmer)
        # results, scores = self.retriever.retrieve(query_tokens, k=topk)
        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer)
        title_indices, _ = self.title_retriever.retrieve(query_tokens, k=topk)
        titles = [self.titles[idx] for idx in title_indices[0]]
        # titles = [self.titles[res] for res in results[0]]
        documents = [self.text_dict[title] for title in titles]
        descriptions = [self.text_dict[title].split("\n\n")[0] for title in titles]
        res_list = [WIKIDATA_ENTITY(None, title, description, doc=doc) for title, description, doc in zip(titles, descriptions, documents)]
        return res_list


if DATASET == "TriviaQA":
    dense_index_dir = os.path.join(".", "data", "TriviaQA", "dense_index")
    bm25_index_dir = os.path.join(".", "data", "TriviaQA", "bm25_index")
    dict_path = os.path.join(".", "data", "TriviaQA", "text_dict.json")
    dense_retriever = DenseRetriever(dense_index_dir, dict_path)
    bm25_retriever = BM25Retriever(bm25_index_dir, dict_path)


def search_entity_from_dense(query: str, topk: int = 10) -> list[WIKIDATA_ENTITY]:
    res_list = dense_retriever.search(query, topk)
    return res_list


def search_entity_from_bm25(query: str, topk: int = 10) -> list[WIKIDATA_ENTITY]:
    res_list = bm25_retriever.search(query, topk)
    return res_list


if __name__ == "__main__":
    # name = "La Leona"
    # result_list = search_entity_from_wikidata(name)
    # for entity in result_list:
    #     print(entity)

    name = "Polish-Russian War (film)"
    # result = from_wikipedia_title_to_wikidata_entity(name)
    name = "Les Films Du Losange"
    name = "Count Of St. Germain"
    name = "Aylwin (Film)"
    name = "It'S In The Air"
    name = "Once A Gentleman"
    name = "The Girl In White"
    name = "Charles Bretagne Marie De La Trémoille"
    # result = search_entity_from_wikipedia(name)

    # query = "Where in England was Dame Judi Dench born?"
    name = "Judi Dench"
    name = "Angola"
    name = "Philips"
    name = "Alfred Brendel"
    name = "Which volcanoin Tanzaniais the highest mountain in Africa?"
    # res = search_entity_from_dense(name)
    name = "Michael Jackson"
    # name = "Melanie Molitor"
    res = search_entity_from_bm25(name)

    breakpoint()
