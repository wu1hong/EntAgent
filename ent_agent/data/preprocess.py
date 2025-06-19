import os
import json
from tqdm import tqdm


def process_2wiki_helper(split:str):
    """
    Create entity linking file for dataset.
    for 2wiki
    Inputs:
            split: str    could only be ['train', 'dev', 'test']
    """
    dataset_name = '2wiki'

    file_path = f"./datasets/{dataset_name}/{split}.json"
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


if __name__ == "__main__":
    process_2wiki()