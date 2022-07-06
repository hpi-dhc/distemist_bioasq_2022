import datasets
from datasets.features import Sequence, Value
from pathlib import Path

def get_el_dataset_info(description: str, homepage : str = None, citation : str = None, license=None):
    return datasets.DatasetInfo(
        description=description,
        features=datasets.Features(
            {
                "corpus_id" : Value("string"),
                "document_id" : Value("string"),
                "doctype": Value("string"),
                "lang": Value("string"),
                "unit_id": Value("string"),
                "source_unit_id" : Value("string"),
                "text": Value("string"),
                "entities": Sequence(
                    {
                        "id": Value("string"),
                        "concepts" : Sequence({
                            "target_kb": Value("string"),
                            "concept_id": Value("string"),
                            "type": Value("string"),
                            "group": Value("string"),
                            "score": Value("float")
                        }),
                        "spans_start": Sequence(Value("int32")),
                        "spans_end": Sequence(Value("int32")),
                        "text" : Value("string"),
                        "label" : Value("string"),
                        "fragmented" : Value("bool")
                    }
                ),
            }
        ),
        homepage=homepage,
        citation=citation,
        license=license
    )

def read_brat_file(txt_file : Path, ann_file = None):
    pass