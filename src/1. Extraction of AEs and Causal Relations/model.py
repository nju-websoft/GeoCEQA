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
"""PyTorch BERT model. """

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math
import os
import sys

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import BertModel
from transformers import BertPreTrainedModel

logger = logging.getLogger(__name__)


class BertForBinaryTokenClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]

    """

    def __init__(self, config):
        super(BertForBinaryTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.span_classifier = nn.ModuleList([nn.Linear(config.hidden_size, self.num_labels[0]) for _ in range(2)])
        # self.binary_span_classifier = nn.ModuleList([nn.Linear(config.hidden_size, 2) for _ in range(2)])
        self.slot_classifier = nn.ModuleList([nn.Linear(config.hidden_size, self.num_labels[1]) for _ in range(2)])
        self.max_width = 61
        self.width_size = 32
        self.width_embeddings = nn.Embedding(self.max_width, self.width_size)
        self.rel_classifier = nn.Linear(config.hidden_size * 4 + self.width_size * 2, self.num_labels[2])
        self.max_poolings = nn.ModuleList([nn.MaxPool1d(kernel_size=i) for i in range(0, self.max_width + 1)])

        self.init_weights()

    def extractSpans(self, args, span_logits, attention_mask, span_null_label_id):
        span_pred = torch.max(span_logits, 2)[1].tolist()  # lx2
        span = [None, None, None]
        spans = []
        for seq_index in range(args.max_seq_length):
            if attention_mask is not None:
                if attention_mask[seq_index] != 1:
                    break
                start_label, end_label = span_pred[seq_index]
                if start_label != span_null_label_id:
                    # if span[0] and span[0] != span_null_label_id:
                    # print('new start', span, start_label.tolist(), seq_index)
                    span[0] = start_label
                    span[1] = seq_index
                if end_label != span_null_label_id and span[0]:
                    if span[0] == end_label:
                        span[2] = seq_index + 1
                        spans.append(tuple(span))
                        span = [None, None, None]
                    else:
                        # print('error end', span, end_label.tolist(), seq_index + 1)
                        span[2] = seq_index + 1
                        spans.append(tuple(span))
                        span = [None, None, None]
        return spans

    def getSpanRepresentation(self, h, span, width_embedding=True):
        span_length = span[2] - span[1]
        if span_length > self.max_width - 1:
            span_length = self.max_width - 1
        vk = h[span[1]:span[1] + span_length]  # 1x?xh
        if span_length != 1:
            vk = vk.unsqueeze(0).transpose(-1, -2)  # 1xhx?
            vk = self.max_poolings[span_length](vk).transpose(-1, -2)  # h
        if width_embedding:
            device = h.device
            vk = vk.squeeze()
            embedding_input = torch.tensor(span_length, dtype=torch.int64, requires_grad=False, device=device)
            embedding = self.width_embeddings(embedding_input)
            # embedding = embedding.squeeze()
            vk = torch.cat([vk, embedding], 0)
        return vk.squeeze()

    def forward(self, args, global_step=None, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, span_labels=None,
                span_size=None, span_list=None, slot_labels=None, slot_mask=None, rel_size=None, rel_list=None,
                question_length=None, binary_span=None, span_null_label_id=0):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output, pooled_output = outputs[:2]

        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        span_logits = torch.cat([self.span_classifier[i](sequence_output).unsqueeze(2) for i in range(2)], 2)  # bxlx2xt
        # binary_span_logits = torch.cat([self.binary_span_classifier[i](sequence_output).unsqueeze(2)
        #                                 for i in range(2)], 2)  # bxlx2x2

        loss_fct = CrossEntropyLoss()
        if span_labels is not None:  # bxlx2
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_loss = active_loss.repeat(2)
                active_span_logits = span_logits.view(-1, self.num_labels[0])[active_loss]
                active_span_labels = span_labels.view(-1)[active_loss]
                loss = loss_fct(active_span_logits, active_span_labels)
                # if binary_span is not None:
                #     active_binary_span_logits = binary_span_logits.view(-1, 2)[active_loss]
                #     active_binary_span_labels = binary_span.view(-1)[active_loss]
                #     loss += loss_fct(active_binary_span_logits, active_binary_span_labels)
            else:
                loss = loss_fct(span_logits.view(-1, self.num_labels[0]), span_labels.view(-1))

        slot_logit = torch.cat([self.slot_classifier[i](sequence_output).unsqueeze(2) for i in range(2)], 2)  # bxlx2xt
        shape = list(sequence_output.shape)
        batch_size = shape[0]
        max_seq_length = shape[1]
        all_spans = []
        all_slot_logits = []
        active_slot_logits = None
        active_slot_labels = None
        rel_spans = []
        rel_logits = None
        rel_labels = None
        question_length = [l[0] for l in question_length.tolist()]
        span_list = [l[:span_size[i]] for i, l in enumerate(span_list.tolist())]
        rel_list = [l[:rel_size[i]] for i, l in enumerate(rel_list.tolist())]
        for batch_index in range(batch_size):  # batch
            h = sequence_output[batch_index]  # lxh
            spans = self.extractSpans(args, span_logits[batch_index], attention_mask[batch_index],
                                      span_null_label_id)[:args.max_span_length]
            if len(spans) == 0:
                all_spans.append([])
                rel_spans.append([])
                all_slot_logits.append(None)
                continue
            slot_logits = None

            rel_h_tmp = []
            rel_spans_tmp = []
            device = h.device
            spans = sorted(spans, key=lambda x: x[1])
            sep_index = question_length[batch_index] + 1
            # last_index = span_labels[batch_index].tolist()
            # last_index = sep_index + 1 + last_index[sep_index + 1:].index([-100, -100])
            for i, span in enumerate(spans):
                logits = slot_logit[batch_index]  # lx2xt
                if slot_logits is None:
                    slot_logits = logits.unsqueeze(0)  # 1xlx2xt
                else:
                    slot_logits = torch.cat((slot_logits, logits.unsqueeze(0)), 0)

                candidateSpans = []
                qa_flag = False
                if span[1] < sep_index:
                    qa_flag = True
                    for s in spans[i + 1:]:
                        if s[1] > sep_index:
                            candidateSpans.append(s)
                elif i + 1 < len(spans):
                    candidateSpans.append(spans[i + 1])

                for cspan in candidateSpans:
                    span1 = span
                    span2 = cspan
                    # if qa_flag:
                    #     span1 = cspan[0]
                    #     span2 = cspan[1]
                    #     last_index = 0
                    #     cspan = cspan[0]
                    rel_spans_tmp.append(span + cspan)
                    h_span = self.getSpanRepresentation(h, span)
                    h_cspan = self.getSpanRepresentation(h, cspan)
                    # if span2 is None and span1[2] < last_index:
                    #     h_between = self.getSpanRepresentation(h, [0, span1[2], last_index], width_embedding=False)
                    # el
                    if qa_flag is False and span1[2] != span2[1]:
                        h_between = self.getSpanRepresentation(h, [0, span1[2], span2[1]], width_embedding=False)
                    else:
                        h_between = torch.zeros([self.config.hidden_size], requires_grad=False, device=device)
                    rel_h_tmp.append(torch.cat([pooled_output[batch_index], h_span, h_between, h_cspan], 0))

                    rel_spans_tmp.append(cspan + span)
                    rel_h_tmp.append(torch.cat([pooled_output[batch_index], h_cspan, h_between, h_span], 0))
            rel_spans_tmp = rel_spans_tmp[:args.max_span_length * 2]
            rel_h_tmp = rel_h_tmp[:args.max_span_length * 2]
            if len(rel_spans_tmp) == 0:
                rel_spans.append([])
            else:
                rel_h = torch.cat([t.unsqueeze(0) for t in rel_h_tmp], 0)  # ?xh
                rel_logit = self.rel_classifier(rel_h)  # ?x4
                rel_pred = torch.max(rel_logit, dim=1)[1].tolist()  # ?
                rel_spans.append([span + (rel_pred[i],) for i, span in enumerate(rel_spans_tmp) if
                                  rel_pred[i] != 0])

                rel_label_tmp = []
                if rel_size is not None and rel_list is not None:
                    rel_example_list = rel_list[batch_index]
                    rel_map = {tuple(l[:6]): l[6] for l in rel_example_list}
                    for span_pair in rel_spans_tmp:
                        if span_pair in rel_map:
                            rel_label_tmp.append(
                                torch.tensor([rel_map[span_pair]], dtype=torch.int64, requires_grad=False,
                                             device=device))
                        else:
                            rel_label_tmp.append(
                                torch.tensor([0], dtype=torch.int64, requires_grad=False, device=device))
                    rel_label = torch.cat([t for t in rel_label_tmp], 0)  # ?
                    if rel_logits is None:
                        rel_logits = rel_logit
                        rel_labels = rel_label
                    else:
                        rel_logits = torch.cat([rel_logits, rel_logit], 0)
                        rel_labels = torch.cat([rel_labels, rel_label], 0)

            all_spans.append(spans)
            all_slot_logits.append(slot_logits)
            if span_size is not None and span_list is not None and slot_labels is not None and slot_mask is not None:
                slot_map = {tuple(span): (slot_labels[batch_index][i], slot_mask[batch_index][i]) for i, span
                            in enumerate(span_list[batch_index])}
                # fallin = rel_label_ids[batch_index][0]
                slot_real_labels = []
                slot_real_logits = []
                for i, span in enumerate(spans):
                    if span in slot_map:
                        nlabel, mask = slot_map[span]
                    else:
                        # nlable=fallin
                        continue
                    mask = mask == 1
                    slot_real_labels.append(nlabel[mask])  # ?xlx2
                    slot_real_logits.append(slot_logits[i][mask])
                if len(slot_real_labels) == 0:
                    continue
                slot_real_logits = torch.cat([logit.view(-1, self.num_labels[1]) for logit in slot_real_logits], 0)
                slot_real_labels = torch.cat([label.view(-1) for label in slot_real_labels], 0)
                if active_slot_logits is None:
                    active_slot_logits = slot_real_logits
                    active_slot_labels = slot_real_labels
                else:
                    active_slot_logits = torch.cat((active_slot_logits, slot_real_logits), 0)
                    active_slot_labels = torch.cat((active_slot_labels, slot_real_labels), 0)
        if args.mode=='joint' or args.mode=='slot':
            if active_slot_labels is not None and active_slot_labels.shape[0] > 0:
                slot_loss = loss_fct(active_slot_logits, active_slot_labels)
                loss += slot_loss
        if args.mode=='joint' or args.mode=='relation':
            if rel_logits is not None:
                rel_loss = loss_fct(rel_logits, rel_labels)
                loss += rel_loss

        outputs = ((span_logits, all_spans, all_slot_logits, rel_spans),)
        # + outputs[2:]  # add hidden states and attention if they are here
        if loss:
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)
