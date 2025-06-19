import requests
from typing import List
from bs4 import BeautifulSoup


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
        print(q_number)
    else:
        print("Wikidata link not found")
    
    return get_wikidata_entity_from_qid(q_number)


if __name__ == "__main__":
    # name = "La Leona"
    # result_list = search_entity_from_wikidata(name)
    # for entity in result_list:
    #     print(entity)

    name = "Polish-Russian_War_(film)"
    result = from_wikipedia_title_to_wikidata_entity(name)
    print(result)
    breakpoint()