from collections import defaultdict
import json
from pathlib import Path
from typing import List, Union
from scispacy.linking_utils import KnowledgeBase, Entity
from collections.abc import Mapping

class CompositeKnowledgebase(KnowledgeBase):

    def __init__(
        self,
        file_paths: List[Union[str, Path]],
        mappers : List[Mapping] = None,
    ):
        if not mappers:    
            mappers = [lambda x: x] * len(file_paths)
        assert len(mappers) == len(file_paths)
        alias_to_cuis = defaultdict(set)
        self.cui_to_entity = {}

        for file_path, mapper in zip(file_paths, mappers):
            file_path = Path(file_path)
            assert file_path.suffix == ".jsonl"
            raw = [json.loads(line) for line in open(file_path)]

            for entry in raw:
                if mapper:
                    entry = mapper(entry)
                if not entry:
                    continue
                if type(entry) != list:
                    entry = [entry]
                for concept in entry:
                    if type(concept["concept_id"]) == int:
                        concept["concept_id"] = str(concept["concept_id"])
                    unique_aliases = set(concept["aliases"])
                    if "canonical_name" in concept: 
                        unique_aliases.add(concept["canonical_name"])
                    for alias in unique_aliases:
                        alias_to_cuis[alias].add(concept["concept_id"])
                    if not concept["concept_id"] in self.cui_to_entity:
                        self.cui_to_entity[concept["concept_id"]] = Entity(**concept)
                    else:
                        self.cui_to_entity[concept["concept_id"]] = _merge_entities(Entity(**concept), self.cui_to_entity[concept["concept_id"]])
                
            self.alias_to_cuis = {**alias_to_cuis}

def _merge_entities(e1 : Entity, e2 : Entity):
    assert e1.concept_id == e2.concept_id
    
    canonical_name = e1.canonical_name
    if not canonical_name:
        canonical_name = e2.canonical_name
    definition = e1.definition
    if not definition:
        definition = e2.definition

    aliases = list(set(e1.aliases).union(set(e2.aliases)))
    types = list(set(e1.types).union(set(e2.types)))
    
    return Entity(e1.concept_id, canonical_name, aliases, types, definition)
