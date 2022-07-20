import pickle
from pathlib import Path
import joblib
import json

from xl_bel.linkers import EntityLinker, logger
from scispacy.candidate_generation import CandidateGenerator, load_approximate_nearest_neighbours_index, LinkerPaths
from scispacy.linking import EntityLinker as ScispacyLinker
from scispacy.linking_utils import KnowledgeBase

from typing import Union

class NGramKBLinker(EntityLinker):

    @staticmethod
    def default_scispacy():
        return NGramKBLinker.scispacy()

    @staticmethod
    def scispacy_no_thresholds():
        return NGramKBLinker.scispacy(
            k = 100,
            threshold = 0.0,
            filter_for_definitions = False,
        )

    @staticmethod
    def load_candidate_generator(index_base_path : Union[str, Path]):
        index_base_path = Path(index_base_path)
        lp = LinkerPaths(
            ann_index=index_base_path / "nmslib_index.bin",
            tfidf_vectorizer=index_base_path / "tfidf_vectorizer.joblib",
            tfidf_vectors=index_base_path / "tfidf_vectors_sparse.npz",
            concept_aliases_list=index_base_path / "concept_aliases.json",
        )
        ann_index = load_approximate_nearest_neighbours_index(lp)
        tfidf_vectorizer = joblib.load(lp.tfidf_vectorizer)
        ann_concept_aliases_list = json.load(open(lp.concept_aliases_list, encoding='utf-8'))
        kb = pickle.load(open(index_base_path / 'kb.pickle', 'rb'))
        
        cg = CandidateGenerator(
            ann_index,
            tfidf_vectorizer,
            ann_concept_aliases_list,
            kb
        )
        return cg

    @staticmethod
    def scispacy(
        resolve_abbreviations: bool = None,
        k: int = None,
        threshold: float = None,
        no_definition_threshold: float = None,
        filter_for_definitions: bool = None,
        max_entities_per_mention: int = None,
    ):
        logger.info("Initializing scispaCy")
        default_linker = ScispacyLinker()
        return NGramKBLinker(
            default_linker.candidate_generator,
            kb = None,
            resolve_abbreviations = resolve_abbreviations if resolve_abbreviations else default_linker.resolve_abbreviations,
            k = k if k else default_linker.k,
            threshold = threshold if threshold else default_linker.threshold,
            no_definition_threshold = no_definition_threshold if no_definition_threshold else default_linker.no_definition_threshold,
            filter_for_definitions = filter_for_definitions if filter_for_definitions else default_linker.filter_for_definitions,
            max_entities_per_mention = max_entities_per_mention if max_entities_per_mention else default_linker.max_entities_per_mention,
            filter_types = False
        )

    def __init__(self, 
            candidate_generator: CandidateGenerator,
            kb : KnowledgeBase = None,
            resolve_abbreviations: bool = True,
            k: int = 30,
            threshold: float = 0.7,
            no_definition_threshold: float = 0.95,
            filter_for_definitions: bool = True,
            max_entities_per_mention: int = 5,
            filter_types : bool = False 
        ):
        self.candidate_generator = candidate_generator
        self.resolve_abbreviations = resolve_abbreviations
        self.k = k
        self.kb = kb if kb else candidate_generator.kb
        self.threshold = threshold
        self.no_definition_threshold = no_definition_threshold
        self.filter_for_definitions = filter_for_definitions
        self.max_entities_per_mention = max_entities_per_mention
        self.filter_types = filter_types

    def predict(self, unit : str, entities : dict) -> dict:
        result = self._init_result(entities)
        
        # TODO resolve abbreviations!
        mention_strings = entities["text"]
        batch_candidates = self.candidate_generator(mention_strings, self.k)

        for _, candidates in zip(mention_strings, batch_candidates):
            predicted = []
            for cand in candidates:
                score = max(cand.similarities)
                if (
                    self.filter_for_definitions
                    and self.kb.cui_to_entity[cand.concept_id].definition is None
                    and score < self.no_definition_threshold
                ):
                    continue
                if score > self.threshold:
                    predicted.append((cand.concept_id, score))
            sorted_predicted = sorted(predicted, reverse=True, key=lambda x: x[1])
            n_cuis = 0
            cuis, scores = [], []
            for cui, score in sorted_predicted:
                cuis.append(cui)
                scores.append(score)
                n_cuis += 1
            result['concepts'].append({
                'target_kb' : ['UMLS'] * n_cuis,
                'concept_id' : cuis,
                'score' : scores,
                'type' : [None] * n_cuis,
                'group' : [None] * n_cuis
            })
        return result