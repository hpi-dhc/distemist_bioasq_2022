def convert_to_spans(sentence, labels, tokenizer, id2label, special_token_mask, skip_subwords):
    sent_ids, sent_tags = sentence["input_ids"], labels
    
    output_spans = []
    
    sent_tokens = tokenizer.convert_ids_to_tokens(sent_ids)

    current_span = None
    start_index = 0
    is_subword = False

    for token, tag, special_token_mask in zip(sent_tokens, sent_tags, special_token_mask):    
        is_subword = token.startswith('##')
        if special_token_mask == 1:
            continue
        if start_index != 0 and not is_subword:
            start_index += 1
        clean_token = token[2:] if is_subword else token
        end_index = start_index + len(clean_token)
        label = id2label[tag] if tag != -100 else None
        if not label or (skip_subwords and is_subword):
            if current_span:
                current_span['word'] += (' ' if not is_subword else '') + clean_token
                current_span['end'] = end_index
        elif label == "O":
            if current_span:
                output_spans.append(current_span)
            current_span = None
        else: #Entity
            entity_type = label[2:]
            if label.startswith("B") and not is_subword:
                if current_span:
                    output_spans.append(current_span)
                    current_span = None
            elif label.startswith("I") or label.startswith("B"):
                if current_span and current_span['entity_group'] != entity_type:
                    output_spans.append(current_span)
                    current_span = None
            else:
                raise Exception(f"Only supporting IO and IOB tagging, unknown tag {label}")

            if not current_span:
                current_span = {
                    'entity_group' : entity_type,
                    'start' : start_index,
                    'end' : end_index,
                    'word' : clean_token,
                }
            else:
                assert current_span['entity_group'] == entity_type
                current_span['end'] = end_index
                current_span['word'] += (' ' if not is_subword else '') + clean_token

        start_index = end_index # Assuming white space tokenization

    if current_span:
        output_spans.append(current_span)

    return output_spans

def ner_error_analyis(sentence, gt_labels, pred_labels, special_tokens_mask, tokenizer, id2label, skip_subwords=False):
    gt_spans = convert_to_spans(sentence, gt_labels, tokenizer, id2label, special_tokens_mask, skip_subwords)
    pred_spans =  convert_to_spans(sentence, pred_labels, tokenizer, id2label, special_tokens_mask, skip_subwords)
    return _ner_error_analyis(pred_spans, gt_spans)

def _ner_error_analyis(pred_spans, gt_spans):
    def hf_to_tuples(spans):
        return [(s['start'], s['end'], s['entity_group'], s['word']) for s in spans]
    return ner_annotation_eval(hf_to_tuples(pred_spans), hf_to_tuples(gt_spans))

from enum import Enum

class NERErrortype(Enum):
    TP = 'true_positive'
    FP = 'false_positive'
    FN = 'false_negative'
    LE = 'labeling_error'
    BE = 'boundary_error'
    LBE = 'label_boundary_error'

def _get_ranges_labels(seq):
    return zip(*[((s[0], s[1]), s[2]) for s in seq])

def get_error_type(pred_sq, gt_seq):
    if pred_sq == gt_seq:
        return NERErrortype.TP
    if not pred_sq:
        return NERErrortype.FN
    if not gt_seq:
        return NERErrortype.FP
    pred_range, pred_labels = _get_ranges_labels(pred_sq)
    gt_range, gt_labels = _get_ranges_labels(gt_seq)
    if pred_range == gt_range:
        assert pred_labels != gt_labels
        return NERErrortype.LE
    else:
        if all([l1 == l2 for l1 in pred_labels for l2 in gt_labels]):
            return NERErrortype.BE
        else:
            return NERErrortype.LBE
        
def ner_annotation_eval(predicted_entities, ground_truth_entities):
    pred_idx = 0
    gt_idx = 0

    max_i = max(
            max([p[1] for p in predicted_entities]) if predicted_entities else 0, 
            max([p[1] for p in ground_truth_entities]) if ground_truth_entities else 0)

    results = []
    subsequence = ([],[])
    flip_label_pred, flip_label_gt = False, False

    def get_item(entity_list, idx, i=None):
        return entity_list[idx] if idx < len(entity_list) and (not i or entity_list[idx][0] <= i) else None

    for i in range(0, max_i+1):
        current_pred = get_item(predicted_entities, pred_idx, i)
        current_gt = get_item(ground_truth_entities, gt_idx, i)

        if current_pred and i in range(current_pred[0], current_pred[1]) and current_gt and (i in range(current_gt[0], current_gt[1])):  
            # Still in the same span
            continue
        if current_pred and (i >= current_pred[1]):
            subsequence[0].append(current_pred)
            pred_idx += 1
            next_pred = get_item(predicted_entities, pred_idx)
            if next_pred and (current_pred[2] != next_pred[2]):
                flip_label_pred = True
        if current_gt and (i >= current_gt[1]):
            subsequence[1].append(current_gt)
            gt_idx += 1
            next_gt = get_item(ground_truth_entities, gt_idx)
            if next_gt and (current_gt[2] != next_gt[2]):
                flip_label_gt = True
        if ((not current_pred or (i >= current_pred[1])) and (not current_gt or (i >= current_gt[1]))) or (flip_label_pred and flip_label_gt):
            if subsequence[0] or subsequence[1]:                
                results.append({
                    'prediction' : subsequence[0],
                    'match' : subsequence[1],
                    'category' : get_error_type(subsequence[0], subsequence[1]).value
                })
                
            subsequence = ([],[])
            flip_label_pred, flip_label_gt = False, False

    assert not subsequence[0] and not subsequence[1]
    
    return results

def ner_check_error_type(predicted_entity, ground_truth_entities):
    # Option 1: false_positive  No match of the prediction to any
    # groud_truth annotations
    match = None
    category = 'false_positive'
    for ground_truth in ground_truth_entities:
        # Used to determine string overlap
        range_prediction = list(range(predicted_entity[0],
                                      predicted_entity[1] + 1))
        range_ground_truth = list(range(ground_truth[0],
                                        ground_truth[1] + 1))

        if predicted_entity == ground_truth:
            # Option 2: true_positive  Exact match of the prediction to
            # the groud_truth annotations
            match = ground_truth
            category = 'true_positive'
            break
        elif (predicted_entity[0] == ground_truth[0] and
              predicted_entity[1] == ground_truth[1]):
            # Option 3: labeling_error  Correct boundaries, but
            # incorrect label
            match = ground_truth
            category = 'labeling_error'
            break
        elif len([char_position for char_position in range_prediction
                  if char_position in range_ground_truth]) != 0:
            # There is an overlap
            # There could be an overlap with multiple entities. This
            # will be ignored as it is still a boundary error and does
            # not provide additoinal information
            if predicted_entity[2] == ground_truth[2]:
                # Option 4: boundary_error  Correct Label, but only
                # overlapping boundaries
                match = ground_truth
                category = 'boundary_error'
                break
            else:
                # Option 5: labeling_boundary_error  Incorrect label,
                # but overlapping boundaries
                match = ground_truth
                category = 'label_boundary_error'
                break
    return category, match