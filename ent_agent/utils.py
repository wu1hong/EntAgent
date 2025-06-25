import requests
from typing import List
from bs4 import BeautifulSoup
import wikipedia


class WIKIDATA_ENTITY:
    # data type for wikidata entity
    def __init__(self, qid, label, desc) -> None:
        """
        qid: wikidata QID, e.g., Q16204382
        label: label of the entity, e.g., Michael Lent
        desc: short description of the entity, e.g., American writer and producer
        """
        self.qid = qid if qid else None
        self.label = label if label else None
        self.desc = desc if desc else None
    
    def __repr__(self):
        text = f"QID: {self.qid}\nLabel: {self.label}\nDescription: {self.desc}\n"
        return text


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
        href = link.get('href')
        q_number = href.split('/')[-1]
        entity_list.append(WIKIDATA_ENTITY(q_number, page.title, '\n'.join(page.summary.split('\n')[:num_sentences])))
    
    return entity_list


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
    result = search_entity_from_wikipedia(name)

    breakpoint()
