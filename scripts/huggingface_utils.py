import imp
from datasets import load_dataset, load_metric, Sequence, ClassLabel, DatasetDict
import transformers
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments, pipeline, DataCollatorForTokenClassification
import numpy as np
import os
from pathlib import Path
from collections import defaultdict
from xl_bel.el_datasets import load_distemist_entities
from tqdm import tqdm
import error_analysis
import pandas as pd
import spacy

metric = load_metric("seqeval")

def load_distemist_dataset_ner(split='train'):
    dataset = load_distemist_entities()[split]
    tags = ['O', 'B-ENFERMEDAD', 'I-ENFERMEDAD']
    dataset = dataset.map(lambda e, i: map_to_iob(e, i, split), batched=True, with_indices=True, remove_columns=['text', 'entities'])

    features = dataset.features
    print(features)

    tag2idx = defaultdict(int)
    tag2idx.update({t: i for i, t in enumerate(tags)})
    dataset = dataset.map(lambda e: {"_tags" : [tag2idx[t] for t in e["tags"]]})
    features["_tags"] = Sequence(ClassLabel(num_classes=len(tags), names=(tags)))
        
    dataset = dataset.cast(features)
    return dataset, tags

def map_to_iob(entries, indices, split):
    nlp = spacy.load("es_core_news_md")
    res = {
        'sentence_id' : [],
        'tokens' : [],
    }
    if split == 'train':
        res['tags'] = []
    keys = [k for k in entries.keys() if (not k in res.keys()) and (not k in ['text', 'entities'])]
    for k in keys:
        res[k] = []    
    for text, ents, idx in zip(entries['text'], entries['entities'], indices):
        doc = nlp(text)
        entities = [(s, e, label, text) for s, e, label, text in zip(ents['spans_start'], ents['spans_end'], ents['label'], ents['text'])]
        entities.sort(key=lambda k: k[0])

        contained_ents = []
        for e1 in entities:
            for e2 in entities:
                if e1[0][0] >= e2[0][0] and e1[1][-1] <= e2[1][-1] and e1 != e2:
                    contained_ents.append(e1)
                    print("Nested entities", e1, e2)
        entities = [e for e in entities if e not in contained_ents]

        for i, s in enumerate(doc.sents):
            if len(s.text.strip()) == 0:
                continue
            tokens = [token.text for token in s]
            res['tokens'].append(tokens)
            if split == 'train':
                labels = get_labels([token for token in s], entities)
                assert len(tokens) == len(labels)
                res['tags'].append(labels)
            res['sentence_id'].append(i)
            for k in keys:
                res[k].append(entries[k][idx])
        assert entities != None and len(entities) == 0, f'All entities assigned {entities} in {idx}'
    return res

def get_labels(tokens, entities):
    labels = []
    current_e = None
    for token in tokens:
        if entities:
            next_e = entities[0]
            assert len(next_e[0]) == 1
            assert len(next_e[1]) == 1
            start = next_e[0][0]
            end = next_e[1][0]
            label = next_e[2]
            if token.idx >= start and (token.idx + len(token)) <= end:
                if current_e:
                    labels.append('I-' + label)
                else:
                    labels.append('B-' + label)
                    current_e = next_e
                if (token.idx + len(token)) >= end:
                    current_e = None
                    entities.pop(0)
                    continue
            else:
                if token.idx >= end:
                    print(entities.pop(0))
                labels.append('O')
                current_e = None
        else:
            labels.append('O')
    if entities and current_e == entities[0]:
        entities.pop(0)
    return labels
    


class LabelAligner():
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize_and_align_labels(self, examples, label_all_tokens=True):
        tokenized_inputs = self.tokenizer(examples["tokens"], 
                                          truncation=True, 
                                          is_split_into_words=True, 
                                          return_offsets_mapping=True,
                                          return_special_tokens_mask=True)

        labels = []
        for i, label in enumerate(examples["_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

def compute_metrics(label_list, entity_level_metrics):
    def _compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        
        if entity_level_metrics:
            final_results = {}
            # Unpack nested dictionaries
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    return _compute_metrics

def eval_on_test_set(test_ds, trainer, tokenizer, prefix): 
    pred = trainer.predict(test_ds)
        
    pred_labels = pred.predictions.argmax(axis=2)
    
    ner_stats = []
    
    for i, z in tqdm(enumerate(zip(test_ds, pred_labels, test_ds['special_tokens_mask']))):
        sentence, sentence_pred, special_tokens_mask = z
        ea = error_analysis.ner_error_analyis(
            sentence, sentence["labels"], sentence_pred, special_tokens_mask, tokenizer, trainer.model.config.id2label, skip_subwords=True)
        ea = [dict({'sentence_id' : i}, **e) for e in ea]
        ner_stats += ea
    
    stats_df = pd.DataFrame(ner_stats)
    
    error_count = stats_df.groupby('category').size()
    stats_df.to_csv('error_analysis.csv')
    
    error_count = error_count.to_dict()
    return {prefix + '/' + k.replace('test_', '') : v for k, v in dict(error_count.items(), **pred.metrics).items()}