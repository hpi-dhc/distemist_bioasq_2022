from html import entities
import imp
from typing import Dict
from xl_bel.linkers import EntityLinker
import numpy as np

class EnsembleLinker(EntityLinker):

    def __init__(self, 
            linkers : Dict[str, EntityLinker],
            candidates_per_linker : Dict[str, EntityLinker] = {},
            linker_weights : Dict[str, EntityLinker] = {},
            linker_thresholds : Dict[str, EntityLinker] = {},
            ):
        self.candidates_per_linker = candidates_per_linker.copy()
        self.linkers = linkers.copy()
        self.linker_weights = linker_weights.copy()
        for l in linkers.keys():
            if not l in linker_weights:
                self.linker_weights[l] = 1.0
        self.linker_thresholds = linker_thresholds

    @staticmethod
    def filter_and_apply_threshold(input_pred, k, threshold):
        def apply(entry):
            entities = entry['entities'].copy()
            filtered_concepts = entities['concepts'].copy()
            for c in filtered_concepts:
                if not c['score']:
                    continue
                idx = np.argwhere(np.array(c['score']) > threshold).ravel()[0:k]
                for key,v in c.items():
                    c[key] = list(np.array(v)[idx])
            entities['concepts'] = filtered_concepts
            return {'entities' : entities}
        return input_pred.map(apply, load_from_cache_file=False)


    def predict_batch(self, dataset, batch_size, top_k=None, reuse_preds=None):
        concepts = {}
        for linker_name, linker_fn in self.linkers.items():
            if reuse_preds:
                mapped = reuse_preds[linker_name]
            else:
                linker = linker_fn()
                print(f"Running linker {linker_name}")
                if linker_name in self.linker_thresholds:
                    print(f'Overriding threshold for {linker_name}')
                    linker.threshold = self.linker_thresholds[linker_name]
                if linker_name in self.candidates_per_linker:
                    linker.k = self.candidates_per_linker[linker_name]
                mapped = linker.predict_batch(dataset, batch_size)
            linker_concepts = []
            for e in mapped["entities"].copy():
                for c in e["concepts"]:
                    c["predicted_by"] = [linker_name] * len(c["concept_id"])
                linker_concepts.append(e["concepts"])
            concepts[linker_name] = linker_concepts
        
        def merge_concepts(concepts_i : list) -> list:
            assert len(concepts_i) > 0
            res = []
            for ent_concepts in zip(*concepts_i):
                def merge_values(k):
                    return np.array([s for c in ent_concepts for s in c[k]])
                scores = merge_values('score')
                linker_weights = [self.linker_weights[l] for l in merge_values('predicted_by')]
                if len(scores) > 0:
                    scores += np.linspace(0.0001, 0.00, len(scores))
                scores *= linker_weights
                order = np.argsort(scores, kind='stable')[-1::-1]
                if top_k:
                    order = order[:top_k]
                merged = {}
                for k in ent_concepts[0].keys():
                    if k == 'score':
                        merged[k] = scores[order]
                    else:
                        merged[k] = list(merge_values(k)[order])
                res.append(merged)
            return res
        
        def get_entities(item, i):
            ents = item["entities"].copy()
            concepts_i = [v[i].copy() for v in concepts.values()]
            ents["concepts"] = merge_concepts(concepts_i)
            return ents
        
        return dataset.map(lambda item, i: {'entities' : get_entities(item, i)}, with_indices=True, load_from_cache_file=False)        

    def predict(self, unit : str, entities : dict) -> dict:
        raise NotImplementedError() 
