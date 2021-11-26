# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """

from __future__ import absolute_import, division, print_function

import logging
import os
import json
from io import open
from copy import deepcopy

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, question, answer, question_events, answer_events, relations):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.question = question
        self.answer = answer
        self.question_events = question_events
        self.answer_events = answer_events
        self.relations = relations


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, span_label, slot_label, slot_mask,
                 span_size, span_list, rel_size, rel_list, question_length):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.span_label = span_label
        # self.binary_span = binary_span
        self.span_size = span_size
        self.span_list = span_list
        self.slot_label = slot_label
        self.slot_mask = slot_mask
        self.rel_size = rel_size
        self.rel_list = rel_list
        self.question_length = question_length


def read_examples_from_file(filename,mode):
    # file_path = os.path.join(data_dir, "{}.jsonl".format(mode))
    guid_index = 1
    examples = []

    with open(filename, encoding="utf-8") as f:
        # words = []
        # labels = []
        for line in f:
            item = json.loads(line)
            question = item['question']
            answer = item['answer']
            if 'question_events' in item:
                question_events = item['question_events']
            else:
                question_events = []
            if 'answer_events' in item:
                answer_events = item['answer_events']
            else:
                answer_events =[]
            # rel_span, rel_labels = toRelations(question_events, answer_events, item['relations'],
            #                                    len(question), len(question) + len(answer))
            examples.append(InputExample(guid="{}-{}".format(mode, guid_index),
                                         question=question,
                                         answer=answer,
                                         question_events=question_events,
                                         answer_events=answer_events,
                                         relations=item['relations'] if 'relations' in item else [],
                                         # rel_span=rel_span,
                                         # rel_labels=rel_labels
                                         )),

            guid_index += 1

            # if line.startswith("-DOCSTART-") or line == "" or line == "\n":
            #     if words:
            #         examples.append(InputExample(guid="{}-{}".format(mode, guid_index),
            #                                      words=words,
            #                                      labels=labels))
            #         guid_index += 1
            #         words = []
            #         labels = []
            # else:
            #     splits = line.split(" ")
            #     words.append(splits[0])
            #     if len(splits) > 1:
            #         labels.append(splits[-1].replace("\n", ""))
            #     else:
            #         # Examples could have no label for mode = "test"
            #         labels.append("O")
        # if words:
        #     examples.append(InputExample(guid="%s-%d".format(mode, guid_index),
        #                                  words=words,
        #                                  labels=labels))

    # if mode == 'test':
    #     with open(os.path.join(data_dir, "test.txt".format(mode)), 'w', encoding='utf-8') as f:
    #         f.write('\n\n'.join(['\n'.join(e.question + '\\' + e.answer) for e in examples]))
    return examples


def toBinaryLabel(text, events, label='bi&slot'):
    labels = ['O' for i in range(len(text))]
    if label.startswith('bi&'):
        labels = [[['O' for i in range(len(text))] for _ in range(2)] for _ in range(2)]
    elif label.startswith('bi'):
        labels = [['O' for i in range(len(text))] for _ in range(2)]
    start_labels = ['O' for i in range(len(text))]
    end_labels = ['O' for i in range(len(text))]
    event_slots = ['direction', 'modifier', 'predicate', 'concept']
    for event in events:
        if label == 'bi&slot':
            span_start = event['start']
            span_end = event['end']
            labels[0][0][span_start] = event['type']
            labels[0][1][span_end - 1] = event['type']
            for slot in event_slots:
                if slot in event and event[slot]:
                    slot_span = event[slot]
                    labels[1][0][slot_span[0]] = slot
                    labels[1][1][slot_span[1] - 1] = slot
        elif label == 'binary':
            span_start = event['start']
            span_end = event['end']
            labels[0][0][span_start] = 'B-' + event['type']
            labels[0][1][span_end - 1] = 'B-' + event['type']
        elif label == 'bislot':
            for slot in event_slots:
                if slot in event and event[slot]:
                    slot_span = event[slot]
                    labels[0][slot_span[0]] = 'B-' + slot
                    labels[1][slot_span[1] - 1] = 'B-' + slot
        elif label == 'slot':
            for slot in event_slots:
                if slot in event and event[slot]:
                    slot_span = event[slot]
                    labels[slot_span[0]] = 'B-' + slot
                    for i in range(slot_span[0] + 1, slot_span[1]):
                        labels[i] = 'I-' + slot
        else:
            span_start = event['start']
            span_end = event['end']
            labels[span_start] = 'B-' + event['type']
            for i in range(span_start + 1, span_end):
                labels[i] = 'I-' + event['type']
    return labels


def toSpansSlots(question, answer, question_events, answer_events,
                 max_seq_length, max_span_length, padding_length, pad_label_id, label_map):
    span_map = label_map[0]
    slot_map = label_map[1]
    total_length = len(question + answer) + 3

    def empty_label_id(empty_id, pad_id, length=total_length):
        label_id = [deepcopy(empty_id) for _ in range(length)]
        label_id[0] = pad_id
        label_id[-1] = pad_id
        label_id[len(question) + 1] = pad_id
        return deepcopy(label_id)

    span_label = empty_label_id([span_map['O'], span_map['O']], [pad_label_id, pad_label_id])
    question_offset = 1
    answer_offset = len(question) + 2
    spans_list = []
    slot_labels = []
    slot_mask = []
    event_slots = ['direction', 'modifier', 'predicate', 'concept']
    for i, event in enumerate(question_events + answer_events):
        offset = question_offset if i < len(question_events) else answer_offset
        start = event['start'] + offset
        end = event['end'] + offset
        type = span_map[event['type']]
        span_label[start][0] = type
        span_label[end - 1][1] = type
        spans_list.append([event['id'], type, start, end])
        slot_label = empty_label_id([slot_map['O'], slot_map['O']], [pad_label_id, pad_label_id])
        for slot in event_slots:
            if slot in event and event[slot] is not None:
                slot_start = event[slot][0] + offset
                slot_end = event[slot][1] + offset - 1
                slot_label[slot_start][0] = slot_map[slot]
                slot_label[slot_end][1] = slot_map[slot]
        mask = empty_label_id(0, 0)
        for i in range(start, end + 1):
            mask[i] = 1
        slot_labels.append(slot_label)
        slot_mask.append(mask)

    # cut and padding
    def cutAndEnd(label_id, length, end_label):
        label_id = label_id[:length]
        label_id[-1] = end_label
        return label_id

    if total_length > max_seq_length:
        span_label = cutAndEnd(span_label, max_seq_length, [pad_label_id, pad_label_id])
        spans_list = sorted(spans_list, key=lambda x: x[2])
        i = len(spans_list)
        for i, span in enumerate(spans_list):
            if span[1] > max_seq_length:
                break
        spans_list = spans_list[:i]
        slot_labels = [cutAndEnd(label, max_seq_length, [pad_label_id, pad_label_id]) for label in slot_labels[:i]]
        slot_mask = [cutAndEnd(mask, max_seq_length, 0) for label in slot_mask[:i]]

    span_label += [[pad_label_id, pad_label_id]] * padding_length
    slot_labels = [label + [[pad_label_id, pad_label_id]] * padding_length for label in slot_labels]
    slot_mask = [mask + [0] * padding_length for mask in slot_mask]

    span_padding_length = max_span_length - len(spans_list)
    spans_list += [[-1, -1, -1, -1]] * span_padding_length
    slot_labels += [empty_label_id([pad_label_id, pad_label_id], [pad_label_id, pad_label_id],
                                   length=max_seq_length)] * span_padding_length
    slot_mask += [empty_label_id(0, 0, length=max_seq_length)] * span_padding_length

    return span_label, spans_list, slot_labels, slot_mask


def toRelations(span_list, raw_relations, label_map, max_span_length):
    span_map = {e[0]: e[1:] for e in span_list if e[0] != -1}
    rel_label = []
    for relation in raw_relations:
        if relation['head'] not in span_map or relation['tail'] not in span_map:
            continue
        head_span = span_map[relation['head']]
        tail_span = span_map[relation['tail']]
        rel = head_span + tail_span + [label_map[relation['type']]]
        rel_label.append(rel)
    padding_length = max_span_length - len(rel_label)
    rel_label += [[-1, -1, -1, -1, -1, -1, -1]] * padding_length

    return rel_label


def convert_examples_to_features(examples,
                                 label_list,
                                 max_seq_length,
                                 tokenizer,
                                 cls_token_at_end=False,
                                 cls_token="[CLS]",
                                 cls_token_segment_id=1,
                                 sep_token="[SEP]",
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 pad_token_label_id=-1,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True,
                                 max_span_length=50):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_maps = [{label: i for i, label in enumerate(labels)} for labels in label_list]
    # print(label_maps)

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        example.question = ''.join(tokenizer.tokenize(example.question))
        example.answer = ''.join(tokenizer.tokenize(example.answer))
        tokens = []
        # label_ids = []
        label_ids = [[[], []] for _ in range(2)]

        for i, word in enumerate(example.question):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)

        tokens += [sep_token]
        # label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        for i, word in enumerate(example.answer):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            # label_ids = label_ids[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        # label_ids += [pad_token_label_id]

        segment_ids += ([sequence_b_segment_id] * (len(tokens) - len(segment_ids)))

        tokens = [cls_token] + tokens
        # label_ids

        segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        if example.question.startswith('根据上述材料，说明我国青藏高'):
            ques_len = len(example.question)
            answ_len = len(example.answer)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)

        input_ids += ([pad_token] * padding_length)
        input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids += ([pad_token_segment_id] * padding_length)
        # label_ids += ([pad_token_label_id] * padding_length)

        span_label, span_list, slot_label, slot_mask = toSpansSlots(example.question, example.answer,
                                                                    example.question_events, example.answer_events,
                                                                    max_seq_length, max_span_length,
                                                                    padding_length, pad_token_label_id, label_maps)
        rel_list = toRelations(span_list, example.relations, label_maps[2], max_span_length)
        span_list = [span[1:] for span in span_list]
        span_size = [len([span for span in span_list if span[0] != -1])]
        rel_size = [len([rel for rel in rel_list if rel[0] != -1])]

        binary_span = [[min(label[0], 1), min(label[1], 1)] for label in span_label]

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        # assert len(label_ids) == max_seq_length
        assert len(span_label) == max_seq_length
        assert len(span_list) == max_span_length
        assert len(slot_label) == max_span_length
        for label in slot_label:
            assert len(label) == max_seq_length
        assert len(slot_mask) == max_span_length
        for mask in slot_mask:
            assert len(mask) == max_seq_length
        assert len(rel_list) == max_span_length
        for label in rel_list:
            assert len(label) == 7

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("span_label: %s", " ".join([str(x) for x in span_label]))
            logger.info("span_size: %s", " ".join([str(x) for x in span_size]))
            logger.info("span_list: %s", " ".join([str(x) for x in span_list]))
            logger.info("slot_label: %s", " ".join([str(x) for x in slot_label]))
            logger.info("slot_mask: %s", " ".join([str(x) for x in slot_mask]))
            logger.info("rel_size: %s", " ".join([str(x) for x in rel_size]))
            logger.info("rel_list: %s", " ".join([str(x) for x in rel_list]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          span_label=span_label,
                          # binary_span=binary_span,
                          span_size=span_size,
                          span_list=span_list,
                          slot_label=slot_label,
                          slot_mask=slot_mask,
                          rel_size=rel_size,
                          rel_list=rel_list,
                          question_length=[len(example.question)]))
    return features


def get_labels(path):
    if path:
        with open(path, "r", encoding='UTF-8') as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return [["O", "state", "action", "change"],
                ['O', 'concept', 'modifier', 'predicate', 'direction'],
                ['O', 'answer_cause', 'answer_and', 'qa_cause']]
        # return ["O", "B-state", "I-state", "B-action", "I-action", "B-change", "I-change"]
        # return ["O", "B-state", "B-action", "B-change"]
        # return ['O', 'B-concept', 'B-modifier', 'B-predicate', 'B-direction']
        # return ['O', 'B-concept', 'I-concept', 'B-modifier', 'I-modifier', 'B-predicate', 'I-predicate', 'B-direction',
        #         'I-direction']
