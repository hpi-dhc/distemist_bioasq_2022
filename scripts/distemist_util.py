import re
from datasets import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import spacy

def load_distemist_texts(text_path, tsv_filter = None, sort_keys = False):
    text_path = Path(text_path)
    unit_ids = []
    texts = []
    if tsv_filter:
        tsv_filter = pd.read_csv(tsv_filter, sep='\t').filename.unique()
    files = list(text_path.glob('*.txt'))
    if sort_keys:
        files = sorted(files, key=lambda k: int(k.stem.split('_')[-1]))
    for f in files:
        if tsv_filter is None or f.stem in tsv_filter:
            unit_ids.append(f.stem)
            texts.append(open(f, 'r', encoding='utf-8').read())
    return Dataset.from_dict({
        'unit_id' : unit_ids,
        'document_id' : unit_ids,
        'text' : texts
    })

def run_ner_pipeline(dataset, pipeline):
    nlp = spacy.load("es_core_news_md")
    idx = []
    spacy_sents = []
    for i, t in enumerate(dataset['text']):
        doc = nlp(t)
        for s in doc.sents:
            spacy_sents.append(s)
            idx.append(i)
    preds = np.array(pipeline([s.text for s in spacy_sents]))

    def get_entities(_, i):
        entities = preds[np.array(idx) == i].ravel()
        sents = np.array(spacy_sents)[np.array(idx) == i].ravel()        
        flat_ents = [e for ents in entities for e in ents]
        
        n_ents = len(flat_ents)
        ids = [None] * n_ents
        fragm = [False] * n_ents
        spans_start = [[e['start'] - 1 + s.start_char] for ents, s in zip(entities, sents) for e in ents]
        spans_end = [[e['end'] + s.start_char] for ents, s in zip(entities, sents) for e in ents]
        labels = [[e['entity_group']] for e in flat_ents]
        texts = []
        for j, e in enumerate(flat_ents):
            l_pattern = r'^[\s\,\.]+'
            w = re.sub(l_pattern, '', e['word'])
            diff_l = len(e['word']) - len(w)
            if diff_l > 0:            
                spans_start[j][0] += diff_l
                pass
            r_pattern = r'[\s\,\.]+$'
            w = re.sub(r_pattern, '', e['word'])
            diff_r = len(e['word']) - len(w)
            if diff_r > 0:            
                spans_end[j][0] -= diff_r
                pass        
            texts.append(re.sub(l_pattern, '', w))
        return { 'entities' : {
            'id' : ids,
            'spans_start': spans_start,
            'spans_end': spans_end,
            'text': texts,
            'label': labels,
            'fragmented': fragm
        }}

    return dataset.map(get_entities, with_indices=True)

def write_dataset_to_tsv(dataset, out_file, concepts : bool, write_missing_concepts=False):
    out_file.parent.mkdir(exist_ok=True, parents=True)
    df = []
    with open(out_file, 'w') as fp:
        for d in dataset:
            es = d['entities']
            mark_idx = 0
            entries = []
            for start, end, text in zip(es['spans_start'], es['spans_end'], es['text']):
                mark_idx += 1
                entries.append({
                    'filename' : d['unit_id'],
                    'mark' : f'T{mark_idx}',
                    'label' : 'ENFERMEDAD',
                    'off0' : start[0],
                    'off1' : end[-1],
                    'span' : text
                })
            if concepts:
                for i, c in enumerate(es['concepts']):
                    if len(c['score']) == 0:
                        code = 'NOMAP'
                    else:
                        max_idx = c['score'].index(max(c['score']))
                        assert max_idx == 0
                        code = c['concept_id'][0]
                    entries[i]['code'] = code
            df += entries
    df = pd.DataFrame(df)
    if concepts and not write_missing_concepts:
        df = df[df.code != 'NOMAP']
    df.to_csv(out_file, sep='\t', index=None)
    return df