from collections import defaultdict
from html import entities
from itertools import groupby
import itertools
from typing import List, Union
from pathlib import Path
import pickle

import pandas as pd
import numpy as np
from regex import R

from xl_bel.linkers import EntityLinker
from xl_bel.ext.sapbert.src.model_wrapper import Model_Wrapper

_SAP_BERT_XLMR = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"

class SapBERTLinker(EntityLinker):

    # Global state
    instance = None
    model_wrapper = None
    candidate_dense_embeds = None
    term_dict = None
    
    @staticmethod
    def clear():
        SapBERTLinker.instance = None
        SapBERTLinker.model_wrapper = None
        SapBERTLinker.candidate_dense_embeds = None
        SapBERTLinker.term_dict = None


    @staticmethod
    def write_dict_embeddings(
            out_dict_file : Union[str, Path],
            out_embed_file : Union[str, Path],
            term_dict : dict, 
            embedding_model_name: str = _SAP_BERT_XLMR,
            cuda : bool = False):

        Path(out_dict_file.parent).mkdir(exist_ok=True, parents=True)
        Path(out_embed_file.parent).mkdir(exist_ok=True, parents=True)

        term_dict.to_pickle(out_dict_file)
        
        wrapper = Model_Wrapper()
        print('Loading XL model')
        wrapper.load_model(embedding_model_name, use_cuda=cuda)

        print('Computing dictionary embeddings')
        candidate_dense_embeds = wrapper.embed_dense(term_dict.term.tolist(), agg_mode='cls', show_progress=True, batch_size=2048*6)
        candidate_dense_embeds -= candidate_dense_embeds.mean(0)
        with open(out_embed_file, 'wb') as f:
            pickle.dump(candidate_dense_embeds, f)

        return out_dict_file, out_embed_file

    def __init__(
        self,
        term_dict_pkl : Union[Path, str],
        dict_embeddings_pkl : Union[Path, str],
        cuda = False,
        gpu_batch_size : int =  16,
        k: int = 10,
        threshold: float = 0.0,
        filter_types : bool = False,
        embedding_model_name :str = _SAP_BERT_XLMR,
        consider_n_grams : list = [],
        remove_duplicates = False,
    ):
        if SapBERTLinker.instance:
            raise Exception("SapBERTLinker is a singleton")
        SapBERTLinker.model_wrapper = Model_Wrapper()
        print('Loading XL model')
        SapBERTLinker.model_wrapper.load_model(embedding_model_name, use_cuda=cuda)
        self.cuda = cuda
        with open(dict_embeddings_pkl, 'rb') as f:
            print('Loading embeddings')
            SapBERTLinker.candidate_dense_embeds = pickle.load(f)
        with open(term_dict_pkl, 'rb') as f:
            print('Loading dict')
            SapBERTLinker.term_dict = pd.read_pickle(f)
        self.k = k
        self.threshold = threshold
        self.filter_types = filter_types
        self.consider_n_grams = consider_n_grams
        self.filter_types = filter_types
        self.gpu_batch_size = gpu_batch_size
        self.remove_duplicates = remove_duplicates

        SapBERTLinker.instance = self


    def predict_batch(self, dataset, batch_size):
        def get_result(sample):            
            mentions_index, mention_strings = zip(*[(j, mention) for j, entities_dict in enumerate(sample['entities']) for mention in entities_dict['text']])

            concepts = self._get_concepts(list(mention_strings))
            
            results = [self._init_result(e) for e in sample['entities']]

            for mi, concept_group in groupby(zip(mentions_index, concepts), key=lambda p: p[0]):            
                results[mi]['concepts'] = [c[1] for c in concept_group]
            
            return {'entities' : results}

        return dataset.map(function=get_result, batch_size=batch_size, batched=True, load_from_cache_file=False)


    def predict(self, unit : str, entities : dict) -> dict:
        raise NotImplementedError() 

    def _get_concepts(self, mention_strings):
        mention_candidates = mention_strings.copy()
        mention_index = list(range(0, len(mention_candidates)))

        def get_n_grams(k, span):
            return [' '.join(span[i:i+k]) for i in range(0, len(span) - k + 1)]

        if self.consider_n_grams:
            for i, mention in enumerate(mention_strings):
                span = mention.split(' ')
                for n in self.consider_n_grams:
                    n_grams = get_n_grams(n, span)
                    mention_candidates += n_grams
                    mention_index += [i] * len(n_grams)

        mention_index = np.array(mention_index)

        if not mention_candidates:
            return

        print(f"Calculate embeddings for {len(mention_candidates)} mentions")
        mention_dense_embeds = SapBERTLinker.model_wrapper.embed_dense(mention_candidates, show_progress=False, agg_mode='cls', batch_size=self.gpu_batch_size)
        #if len(mention_candidates) > 1:
        #   mention_dense_embeds -= mention_dense_embeds.mean(0)

        print(f"Calculating score matrix")
        score_matrix = SapBERTLinker.model_wrapper.get_score_matrix(mention_dense_embeds, SapBERTLinker.candidate_dense_embeds, normalise=True)

        if self.cuda:
            print(f"Retrieving {self.k} candidates on GPU")
            candidate_idxs_batch = SapBERTLinker.model_wrapper.retrieve_candidate_cuda(
                score_matrix = score_matrix, 
                topk = self.k * (1 + len(self.consider_n_grams)), ## too low if n-grams are considered?
                show_progress=False,
                batch_size=self.gpu_batch_size,
            )
        else:
            print(f"Retrieving {self.k} candidates without GPU")
            candidate_idxs_batch = SapBERTLinker.model_wrapper.retrieve_candidate(
                score_matrix = score_matrix, 
                topk = self.k * (1 + len(self.consider_n_grams)), ## too low if n-grams are considered?
            )

        for i, _ in enumerate(mention_strings):
            a = [[cand, score[cand]] for cand, score, m in zip(candidate_idxs_batch, score_matrix, mention_index) if m == i]
            candidates = np.vstack([a[0] for a in a]).ravel()
            cand_scores = np.vstack([a[1] for a in a]).ravel()
            top_scores = cand_scores.argsort()[-1::-1]
            top_candidates = candidates[top_scores]
            concepts = []
            cuis = set()
            for r, score in zip(SapBERTLinker.term_dict.iloc[top_candidates].iterrows(), cand_scores[top_scores]):
                r = r[1]
                if score > self.threshold:
                    if not self.remove_duplicates or not r.cui in cuis: # Remove duplicate cuis
                        cuis.add(r.cui)
                        if len(cuis) > self.k:
                            continue
                        concepts.append((r.cui, score))
            sorted_predicted = sorted(concepts, reverse=True, key=lambda x: x[1])[:self.k]
            n_cuis = 0
            cuis, scores = [], []
            for cui, score in sorted_predicted:
                cuis.append(cui)
                scores.append(score)
                n_cuis += 1
            yield {
                'target_kb' : ['UMLS'] * n_cuis,
                'concept_id' : cuis,
                'score' : scores,
                'type' : [None] * n_cuis,
                'group' : [None] * n_cuis
            }