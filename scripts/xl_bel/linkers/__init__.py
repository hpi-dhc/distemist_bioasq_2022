# pylint: disable=g-import-not-at-top,g-bad-import-order,wrong-import-position

from abc import ABC, abstractmethod
from datasets.arrow_dataset import Dataset
import logging

from xl_bel.reranking import Reranker

logger = logging.getLogger(__name__)

class EntityLinker(ABC):

    def predict_batch(self, dataset : Dataset, batch_size : int = None) -> Dataset:
        """Naive default implementation of batch prediction.
        Should be overridden if the particular model provides an efficient way to predict in batch (e.g., on a GPU)

        Args:
            dataset (Dataset): Input (arrow) dataset with entities but without concepts

        Returns:
            Dataset: (arrow) dataset with linked concepts
        """
        return dataset.map(lambda unit: {'entities' : self.predict(unit['text'], unit['entities'])})

    
    def _init_result(self, entities):
        result = entities.copy()
        assert not 'concepts' in result or not(result['concepts']) or all(not(r) for r in result['concepts'])
        result['concepts'] = []
        return result

    @abstractmethod
    def predict(self, unit : str, entities : list) -> list:
        pass

class RerankedLinker(EntityLinker):

    def __init__(self, linker: EntityLinker, ranker : Reranker):
        self.linker = linker
        self.ranker = ranker

    def predict_batch(self, dataset : Dataset, batch_size : int = None) -> Dataset:
        result = self.linker.predict_batch(dataset, batch_size)
        return self.ranker.rerank_batch(result)

    def predict(self, unit : str, entities : list) -> list:
        raise NotImplementedError()


from .n_grams.ngram_kb_linker import NGramKBLinker
from .sapbert.sap_bert_linker import SapBERTLinker