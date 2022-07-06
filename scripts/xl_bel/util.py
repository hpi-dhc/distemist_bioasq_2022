from collections import defaultdict
import json
from typing import List, Union
from collections.abc import Mapping
from pathlib import Path
import pandas as pd

def clean_concepts_from_dataset(dataset):
    return dataset.map(lambda i: {'entities' : { k:v for k, v in i['entities'].items() if k in ["id", "spans_start", "spans_end", "text"]}})

def create_flat_term_dict(concept_names_jsonl : List[Union[str, Path]], mappers : List[Mapping] = None):
    term_dict = []
    if not mappers:
        mappers = [lambda x: x] * len(concept_names_jsonl)
    assert len(mappers) == len(concept_names_jsonl)
    for jsonl_file, mapper in zip(concept_names_jsonl, mappers):
        with open(jsonl_file) as f:
            for entry in f:
                entry = json.loads(entry)
                if mapper != None:
                    entry = mapper(entry)
                    if not entry:
                        continue
                if type(entry) != list:
                    entry = [entry]
                for e in entry:
                    assert e['canonical_name']
                    cui = e['concept_id']
                    tuis = e['types']
                    term_dict.append({
                        'cui' : str(cui), 
                        'term' : e['canonical_name'],
                        'canonical' : e['canonical_name'],
                        'tuis' : tuis
                        }
                    )
                    for alias in e['aliases']:
                        term_dict.append({
                            'cui' : str(cui), 
                            'term' : alias,
                            'canonical' : e['canonical_name'],
                            'tuis' : tuis
                            }
                        )
    term_dict = pd.DataFrame(term_dict)
    return term_dict.drop_duplicates(subset=['cui', 'term'])

class Concept():

    def __init__(self, concept_id : str = None, score : float = None, target_kb : str = None, type : str = None, group : str = None):
        self._dict = {
            "concept_id": concept_id,
            "target_kb": target_kb,
            "type": type,
            "group": group,
            "score": score,
        }

class Entity():

    def __init__(self, start : Union[int, List[int]], end : Union[int, List[int]], text : str, id : str = "1", concepts : List[Concept] = None):
        self._dict = {
            "id": id,
            "spans_start": [start] if type(start) == int else start,
            "spans_end": [end] if type(start) == int else start,
            "text" : text,
            "fragmented" : (type(start) == list and len(start) > 1) or (type(end) == list and len(end) > 1),
            "concepts": make_concept_dict(concepts) if concepts else []
        }

def _to_list(attribute, l : list):
    return [e._dict[attribute] for e in l if e != None] 

def make_entity_dict(entities : List[Entity]) -> dict:
    return {k: _to_list(k, entities) for k in ["id", "spans_start", "spans_end", "text", "fragmented", "concepts"]}

def make_concept_dict(entities : List[Concept]) -> dict:
    return {k: _to_list(k, entities) for k in ["concept_id", "target_kb", "type", "group", "score"]}