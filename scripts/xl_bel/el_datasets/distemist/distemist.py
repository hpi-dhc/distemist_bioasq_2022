import datasets
from xl_bel.el_datasets.dataset_utils import get_el_dataset_info
from pathlib import Path
import pandas as pd

_CITATION = """
@dataset{luis_gasco_2022_6500526,
  author       = {Luis Gasco and
                  Eulàlia Farré and
                  Miranda-Escalada, Antonio and
                  Salvador Lima and
                  Martin Krallinger},
  title        = {{DisTEMIST corpus: detection and normalization of 
                   disease mentions in spanish clinical cases}},
  month        = apr,
  year         = 2022,
  note         = {{Funded by the Plan de Impulso de las Tecnologías 
                   del Lenguaje (Plan TL).}},
  publisher    = {Zenodo},
  version      = {3.0.1},
  doi          = {10.5281/zenodo.6500526},
  url          = {https://doi.org/10.5281/zenodo.6500526}
}
"""

_HOMEPAGE = "https://temu.bsc.es/distemist/"

_DESCRIPTION = """
The DisTEMIST corpus is a collection of 1000 clinical cases with disease annotations linked with Snomed-CT concepts. 
All documents are released in the context of the BioASQ DisTEMIST track for CLEF 2022. 
For more information about the track and its schedule, please visit the website.
"""

_ZENODO_URL = "https://zenodo.org/record/6500526/files/distemist.zip?download=1"

class DisTEMIST(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("3.0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="subtrack1_entities", version=VERSION, description="Subtrack 1"),
        datasets.BuilderConfig(name="subtrack2_linking", version=VERSION, description="Subtrack 2"),
    ]

    def _info(self):
        return get_el_dataset_info(_DESCRIPTION, _HOMEPAGE, _CITATION)

    def _split_generators(self, dl_manager):   
        data_dir = Path(dl_manager.download_and_extract(_ZENODO_URL))
        subtrack = self.config.name
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir / "distemist" / "training",
                    "subtrack" : subtrack,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir / "distemist" / "test",
                    "subtrack" : subtrack
                },
            ),
        ]

    def _generate_examples(self, filepath, subtrack):
        text_path = filepath / "text_files"
        annotations = list((filepath / subtrack).glob('*.tsv'))
        if not filepath.exists():
            assert len(annotations) == 0
            return
        if subtrack == 'subtrack1_entities':
            assert len(annotations) == 1
        if subtrack == 'subtrack2_linking':
            assert len(annotations) == 2
        anno_df = []
        for a in annotations:
            sub_df = pd.read_csv(a, sep='\t')
            sub_df.set_index('filename', inplace=True)
            sub_df['tsv_file'] = a.stem
            anno_df.append(sub_df)
        anno_df = pd.concat(anno_df)
        for i, txt_file in enumerate(text_path.glob('*.txt')):
            file_name = txt_file.stem
            with open(txt_file, 'r', encoding='utf-8') as fh:
                txt_content = fh.read()
            entities = []
            if not file_name in anno_df.index:
                continue
            for _, r in anno_df.loc[[file_name]].sort_values('off0').iterrows():
                assert txt_content[r.off0:r.off1] == r.span
                concepts = None
                if subtrack == 'subtrack2_linking':
                    codes = r.code.split('+')
                    semantic_rels = r.semantic_rel.split('+')
                    if len(semantic_rels) == 1 and len(codes) > 1:
                        semantic_rels = semantic_rels * len(codes)
                    assert len(codes) == len(semantic_rels), f"{len(codes)}, {len(semantic_rels)}, {file_name}, {r.mark}"
                    concepts = [{
                            "target_kb": "SNOMEDCT",
                            "concept_id": code,
                            "type": r.label,
                            "group": semantic_rel,
                            "score": None
                        } for (code, semantic_rel) in zip(codes, semantic_rels)]
                e = {
                    "id" : r.mark,
                    "spans_start" : [r.off0],
                    "spans_end" : [r.off1],
                    "label" : r.label,
                    "text" : r.span,
                    "fragmented" : False,
                    "concepts" : concepts
                }
                entities.append(e)
            yield i, {
                "corpus_id" : r.tsv_file,
                "document_id" : file_name,
                "doctype": subtrack,
                "lang": "es",
                "unit_id": file_name,
                "source_unit_id" : file_name,
                "text": txt_content,
                "entities": entities
            }