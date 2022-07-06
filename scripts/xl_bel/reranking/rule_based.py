from collections.abc import Mapping
from xml.dom.minidom import Entity
from xl_bel.reranking import Reranker

class EntityContext():

    def __init__(self, entity_text):
        self.entity_text = entity_text        

class RuleBasedReranker(Reranker):

    def __init__(self, rules : Mapping) -> None:
        self.rules = rules

    def rerank_batch(self, dataset):
        def apply_rules(concepts, context):
            for r in self.rules:
                concepts = r(concepts, context)
            return concepts
        def apply_rules_unit(entry):
            mapped_entities = entry['entities'].copy()
            entity_concepts = []
            for entity_text, concepts in zip(entry['entities']['text'], entry['entities']['concepts']):
                ctx = EntityContext(entity_text) # provide as needed
                concepts = apply_rules(concepts.copy(), ctx)
                entity_concepts.append(Reranker.sort_concepts(concepts))
            mapped_entities['concepts'] = entity_concepts
            return {'entities' : mapped_entities }

        return dataset.map(apply_rules_unit, load_from_cache_file=False)