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
import copy

import dgl
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from functools import partial
from dgl.nn.pytorch import GraphConv

from transformers import BertModel
from transformers import BertPreTrainedModel

from gnn import HeteroGNN
from load_data import edge_types, num_class, symbols, question_aware
from decode import TransformerDecoder, NMTLoss

logger = logging.getLogger(__name__)


class PointWiseLoss(nn.Module):
    def __init__(self, method='classification'):
        super(PointWiseLoss, self).__init__()
        self.method = method
        if method == 'classification':
            self.loss_func = CrossEntropyLoss()
        else:
            self.loss_func = MSELoss()

    def forward(self, graph, scores, labels):
        if self.method == 'classification':
            labels = labels.long()
        return self.loss_func(scores, labels)


class PairWiseLoss(nn.Module):
    def __init__(self):
        super(PairWiseLoss, self).__init__()

    def forward(self, graph, scores, labels):
        accum_node_size = 0
        pairs = None
        sq_scores = scores.unsqueeze(1)
        for node_size in graph.batch_num_nodes('event'):
            ins_scores = sq_scores[accum_node_size:accum_node_size + node_size]
            ins_labels = labels[accum_node_size:accum_node_size + node_size]
            pos_nodes = torch.nonzero(ins_labels)
            neg_nodes = torch.nonzero(1 - ins_labels)

            npairs = [ins_scores[pn] - ins_scores[nn] for pn in pos_nodes for nn in neg_nodes]
            if len(npairs) > 0:
                npairs = torch.cat(npairs, dim=0)
                if pairs is None:
                    pairs = npairs
                else:
                    pairs = torch.cat([pairs, npairs], dim=0)
            # for pn in pos_nodes:
            #     for nn in neg_nodes:
            #         dis = ins_scores[pn] - ins_scores[nn]
            #         if pairs is None:
            #             pairs = dis
            #         else:
            #             pairs = torch.cat([pairs, dis], dim=0)
            accum_node_size += node_size
        if pairs is None:
            loss = torch.mean(scores * 0)
        else:
            # pairs = torch.ones(pairs.shape) - pairs
            loss = torch.mean(torch.clamp(1 - pairs, 0))
        return loss


class CopyAttention(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        for m in (self.query.parameters(),
                  self.key.parameters(),
                  self.value.parameters()):
            for p in m:
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    p.data.zero_()

    def forward(self, hidden_states, query):
        query_layer = self.query(query)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.hidden_size)
        attention_probs = torch.exp(torch.log_softmax(attention_scores, dim=-1))
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        outputs = (context_layer, attention_probs)
        return outputs


class Decoder(nn.Module):
    def __init__(self, num_layers, hidden_size, heads, intermediate_size, dropout, embeddings, vocab_size):
        super().__init__()

        self.decoder = TransformerDecoder(
            num_layers=num_layers,
            d_model=hidden_size,
            heads=heads,
            d_ff=intermediate_size,
            dropout=dropout,
            embeddings=embeddings,
            vocab_size=vocab_size,
        )

        self.generator = nn.Linear(hidden_size, vocab_size)
        self.generator.weight = self.decoder.embeddings.weight
        self.gen_func = nn.LogSoftmax(dim=-1)

        self.gen_proj = nn.Linear(hidden_size * 2, 1)
        # self.attention = CopyAttention(hidden_size=hidden_size, dropout=dropout)

        self.init_weights()

    def init_weights(self):
        for layer in (self.decoder.modules(),):
            for module in layer:
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()

        for m in (self.generator.parameters(),
                  self.gen_proj.parameters()):
            for p in m:
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    p.data.zero_()

    def init_decoder_state(self, src, memory_bank, with_cache=False):
        return self.decoder.init_decoder_state(src=src, memory_bank=memory_bank, with_cache=with_cache)

    def forward(self, decoder_input_ids, encoder_input_ids=None,
                encoder_hidden_states=None, state=None, step=None, copy=False):
        # decoder_output, dec_state = self.decoder(decoder_input_ids, state=state, step=step,
        #                                          encoder_hidden_states=encoder_hidden_states)
        decoder_output, dec_state, attn_dist = self.decoder(decoder_input_ids, state=state, step=step, need_attn=True,
                                                            encoder_hidden_states=encoder_hidden_states)

        # encoder_dist, attn_dist = self.attention(hidden_states=encoder_hidden_states, query=decoder_output)

        if not copy:
            vocab_dist = self.generator(decoder_output)
            combine_dist = self.gen_func(vocab_dist)
        else:
            vocab_dist = self.generator(decoder_output).float()
            vocab_dist = torch.softmax(vocab_dist, dim=-1)

            attn_dist = torch.mean(attn_dist, dim=1)
            encoder_dist = torch.matmul(attn_dist, encoder_hidden_states)
            gen_prob = self.gen_proj(torch.cat([decoder_output, encoder_dist], dim=-1))
            gen_prob = torch.sigmoid(gen_prob).type_as(vocab_dist)

            batch_size, decoder_len = attn_dist.size()[:2]
            encoder_ii_size = encoder_input_ids.size()[0]
            encoder_input_ids = tile(encoder_input_ids, batch_size // encoder_ii_size, dim=0)
            copy_dist = torch.zeros_like(vocab_dist).type_as(attn_dist)
            copy_dist = copy_dist.scatter_add(dim=2, src=attn_dist,
                                              index=encoder_input_ids.unsqueeze(1).expand(-1, decoder_len, -1))
            copy_dist = copy_dist.type_as(vocab_dist)

            combine_dist = torch.log(gen_prob * vocab_dist + (1 - gen_prob) * copy_dist).type_as(decoder_output)

        return combine_dist, dec_state


class BertGnnNodeClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertGnnNodeClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.enable_gnn = True
        self.vocab_size = config.vocab_size
        gnn_hidden_size = 256
        self.bert_to_gnn_layer = nn.Linear(config.hidden_size, gnn_hidden_size)
        self.gnn = HeteroGNN(in_feats=gnn_hidden_size, n_hidden=gnn_hidden_size, out_size=gnn_hidden_size, n_layers=3,
                             etypes=edge_types, aggregator_type='mean', heads=8, relation_reducer='max',
                             # activation=F.relu,
                             dropout=config.hidden_dropout_prob)
        #     GCN(in_feats=gnn_hidden_size, n_hidden=gnn_hidden_size, n_layers=4,
        #                activation=F.relu, dropout=config.hidden_dropout_prob)
        self.method = 'classification'
        self.node_criterion = PointWiseLoss(method=self.method)

        self.output_size = gnn_hidden_size if self.enable_gnn else config.hidden_size
        self.output_layer = nn.Linear(self.output_size, num_class)
        # self.output_layers = nn.ModuleList([nn.Linear(gnn_hidden_size, gnn_hidden_size) for _ in range(2)])

        decoder_layers = 6
        self.gnn_feature_token_id = 10
        self.padding_idx = 0
        self.hidden_size = config.hidden_size
        self.gnn_to_hidden_size = nn.Linear(gnn_hidden_size, config.hidden_size)
        self.max_pos = 20
        # self.position_embedding = nn.Embedding(self.max_pos, self.hidden_size)
        # self.decoder_input_layer = BertLayer(config=config)
        tgt_embeddings = nn.Embedding(self.vocab_size, config.hidden_size, padding_idx=0)
        tgt_embeddings.weight = copy.deepcopy(self.bert.embeddings.word_embeddings.weight)

        # self.decoder = TransformerDecoder(
        #     num_layers=decoder_layers,
        #     d_model=config.hidden_size,
        #     heads=config.num_attention_heads,
        #     d_ff=config.intermediate_size,
        #     dropout=config.hidden_dropout_prob,
        #     embeddings=tgt_embeddings,
        #     vocab_size=self.vocab_size,
        # )
        self.decoder = Decoder(
            num_layers=decoder_layers,
            hidden_size=config.hidden_size,
            heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            embeddings=tgt_embeddings,
            vocab_size=self.vocab_size,
        )

        # gen_func = nn.LogSoftmax(dim=-1)
        # self.generator = nn.Sequential(nn.Linear(config.hidden_size, self.vocab_size), gen_func)
        # self.generator[0].weight = self.decoder.embeddings.weight

        self.label_smoothing = 0
        self.nmtloss = NMTLoss(symbols['PAD'], config.vocab_size, self.label_smoothing)

        self.init_weights()

    def init_weights(self):
        super(BertGnnNodeClassification, self).init_weights()

        # for layer in (self.decoder.modules(),
        #         # self.decoder_input_layer.modules()
        #               ):
        #     for module in layer:
        #         if isinstance(module, (nn.Linear, nn.Embedding)):
        #             module.weight.data.normal_(mean=0.0, std=0.02)
        #         elif isinstance(module, nn.LayerNorm):
        #             module.bias.data.zero_()
        #             module.weight.data.fill_(1.0)
        #         if isinstance(module, nn.Linear) and module.bias is not None:
        #             module.bias.data.zero_()

        for m in (self.bert_to_gnn_layer.parameters(),
                  # self.generator.parameters(),
                  self.gnn_to_hidden_size.parameters(),
                  self.output_layer.parameters()):
            for p in m:
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    p.data.zero_()

    def forward(self, graph, answer_ids, args, eval_mode=False, during_train=True, tokenizer=None):
        # if not question_aware:
        #     question_features = graph.ndata['v-h'][0].unsqueeze(0)
        #     question_ids = question_features[:, 0]
        #     question_feature, _ = self.bert(question_features[:, 0],
        #                                     attention_mask=question_features[:, 1],
        #                                     token_type_ids=question_features[:, 2])
        node_features = graph.ndata['v-h']
        node_labels = graph.ndata['v-label'].type_as(node_features)

        sequence_output, h = self.bert(node_features[:, 0],
                                       attention_mask=node_features[:, 1],
                                       token_type_ids=node_features[:, 2])
        h = self.dropout(h)  # batch x h

        if self.enable_gnn:
            h = self.dropout(self.bert_to_gnn_layer(h))
            h = self.gnn(graph, {'event': h}, during_train)['event']  # gbatch x h
            h = self.dropout(h)

        method = 'classification'
        # method='regression'
        if method == 'classification':
            if num_class == 2:
                logits = self.output_layer(h)
                loss = self.node_criterion(graph, logits, node_labels)
                scores = F.softmax(logits, dim=1)[:, 1]
                outputs = (loss, scores)
                if not question_aware:
                    accum = 0
                    nscores = None
                    for node_size in graph.batch_num_nodes('event'):
                        if nscores is None:
                            nscores = scores[accum:accum + node_size - 1]
                        else:
                            nscores = torch.cat((nscores, scores[accum:accum + node_size - 1]), dim=0)
                        scores[accum + node_size - 1] = torch.ones_like(scores[accum + node_size - 1])
                    outputs = (outputs[0], nscores)
            else:
                logits = self.output_layer(h)
                loss = self.node_criterion(graph, logits, node_labels)
                scores = F.softmax(logits, dim=1)
                preds = torch.argmax(scores, dim=1)
                scores = scores[:, 1] + scores[:, 2]
                scores = torch.where(preds == 2, scores + 1, scores)
                outputs = (loss, scores)

                # rnlogits = self.rnoutput_layer(h)
                # prlogits = self.proutput_layer(torch.cat((h, rnlogits), dim=1))
                #
                # ones = torch.ones(node_labels.shape).to(node_labels.device)
                # zeros = torch.zeros(node_labels.shape).to(node_labels.device)
                # rnlabels = torch.where(node_labels > 0, ones, zeros)
                # loss = loss_fct(graph, rnlogits, rnlabels)
                #
                # r_pred_sm = F.softmax(rnlogits, dim=-1)
                # r_pred = torch.argmax(r_pred_sm, dim=-1).squeeze()
                # p_pred_sm = F.softmax(prlogits, dim=-1)
                # p_pred = torch.argmax(p_pred_sm, dim=-1).squeeze()
                # preds = torch.where(r_pred > 0, p_pred + 1, r_pred)
                # r_pred_index = torch.nonzero(r_pred).squeeze(-1)
                # if len(r_pred_index) > 0:
                #     prlabels = torch.where(node_labels > 1, ones, zeros)
                #     # r_pred_index = torch.cat((r_pred_index, torch.nonzero(prlabels).squeeze(-1)), dim=0)
                #     prlogits = torch.index_select(prlogits, dim=0, index=r_pred_index)
                #     prlabels = torch.index_select(prlabels, dim=0, index=r_pred_index)
                #     loss += loss_fct(graph, prlogits, prlabels)

                # p_scores = r_pred_sm[:, 1] + torch.where(p_pred > 0, ones, zeros)
                # scores = torch.where(r_pred > 0, p_scores, r_pred_sm[:, 1])
                # outputs = (loss, scores, preds)
        else:
            scores = torch.mul(self.output_layers[0](h), self.output_layers[1](h))  # gbatch x h
            scores = torch.sum(scores, dim=1)  # gbatch x 1

            loss = self.node_criterion(graph, scores, node_labels)
            scores = F.softmax(scores, dim=1)[:, 1]

            outputs = (loss, scores)

        input_ids = node_features[:, 0]
        if self.enable_gnn:
            h = self.gnn_to_hidden_size(h.unsqueeze(1))
            gnn_feature_token_ids = torch.full(
                (input_ids.size()[0], 1), self.gnn_feature_token_id, dtype=torch.long, device=input_ids.device)
            input_ids = torch.cat((gnn_feature_token_ids, input_ids), dim=-1)
            h = torch.cat((h, sequence_output), dim=1)  # (batch_size x node_size) x (1+seq_len) x h
        else:
            h = sequence_output
        batch_start = 0
        encoder_input_ids = []
        encoder_hidden_states = []
        topk_num = self.max_pos
        if not question_aware:
            topk_num += 1
        copy_enable = False
        for node_size in graph.batch_num_nodes('event'):  # batch
            if not question_aware and node_size == 2:
                topk_indices = [1]
            else:
                _, topk_indices = torch.topk(scores[batch_start:batch_start + node_size], k=min(topk_num, node_size),
                                             dim=0)
            input_ids_batch = input_ids[batch_start:batch_start + node_size][topk_indices].expand(
                (topk_num, -1)).contiguous()
            encoder_hidden_states_batch = h[batch_start:batch_start + node_size][topk_indices].expand(
                (topk_num, -1, -1)).contiguous()

            # if not question_aware:
            #     input_ids_batch = torch.cat((question_ids, input_ids_batch), dim=0)
            #     encoder_hidden_states_batch = torch.cat((question_feature, encoder_hidden_states_batch), dim=0)
            # position_ids = torch.arange(topk_num, dtype=torch.long, device=input_ids.device)
            # position_ids = position_ids.unsqueeze(1).expand(encoder_hidden_states_batch.size()[:2])
            # encoder_hidden_states_batch += self.position_embedding(position_ids)

            encoder_hidden_states.append(encoder_hidden_states_batch.view((-1, self.hidden_size)))
            encoder_input_ids.append(input_ids_batch.view((-1)))
            batch_start += node_size
        encoder_input_ids = torch.stack(encoder_input_ids, dim=0)  # batch_size x 20*(1+seq_len)
        encoder_hidden_states = torch.stack(encoder_hidden_states, dim=0)
        # encoder_mask = (encoder_input_ids != self.padding_idx).half().unsqueeze(1).expand(
        #     (encoder_input_ids.size()[0], encoder_input_ids.size()[1], encoder_input_ids.size()[1])
        # ).unsqueeze(1)
        # encoder_hidden_states = self.decoder_input_layer(encoder_hidden_states, attention_mask=encoder_mask)[0]

        dec_state = self.decoder.init_decoder_state(encoder_input_ids, encoder_hidden_states)
        # h, _ = self.decoder(answer_ids[:, :-1], encoder_hidden_states, dec_state)
        # h = self.generator(h)
        h, _ = self.decoder(decoder_input_ids=answer_ids[:, :-1],
                            encoder_input_ids=encoder_input_ids,
                            encoder_hidden_states=encoder_hidden_states,
                            state=dec_state, copy=copy_enable)
        loss = self.nmtloss(h, answer_ids[:, 1:])
        outputs = (outputs[0] + loss * (0.1 if self.label_smoothing == 0 else 10), outputs[1])

        if eval_mode is True:
            beam_size = 5
            max_length = 100
            min_length = 30
            alpha = 2
            block_ngram = args.block_ngram # 3  # best in 5
            start_token_id = symbols['BOS']
            end_token_id = symbols['EOS']
            batch_size = encoder_input_ids.size(0)
            src_features = encoder_hidden_states
            dec_states = self.decoder.init_decoder_state(encoder_input_ids, src_features, with_cache=True)
            device = src_features.device

            # Tile states and memory beam_size times.
            dec_states.map_batch_fn(lambda state, dim: tile(state, beam_size, dim=dim))
            src_features = tile(src_features, beam_size, dim=0)
            batch_offset = torch.arange(batch_size, dtype=torch.long, device=device)
            beam_offset = torch.arange(0, batch_size * beam_size, step=beam_size, dtype=torch.long, device=device)
            alive_seq = torch.full([batch_size * beam_size, 1], start_token_id, dtype=torch.long, device=device)

            # Give full probability to the first beam on the first step.
            topk_log_probs = torch.tensor([0.0] + [float("-inf")] * (beam_size - 1), device=device).repeat(batch_size)

            # Structure that holds finished hypotheses.
            hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

            results = {}
            results["predictions"] = [[] for _ in range(batch_size)]  # n
            results["words"] = [[] for _ in range(batch_size)]  # noqa: F812
            results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812

            for step in range(max_length):
                decoder_input = alive_seq[:, -1].view(1, -1)

                # Decoder forward.
                decoder_input = decoder_input.transpose(0, 1)

                dec_out, dec_states = self.decoder(decoder_input_ids=decoder_input,
                                                   encoder_input_ids=encoder_input_ids,
                                                   state=dec_states, step=step, copy=copy_enable,
                                                   encoder_hidden_states=src_features)

                # Generator forward.
                # log_probs = self.generator(dec_out).transpose(0, 1).squeeze(0)
                log_probs = dec_out.transpose(0, 1).squeeze(0)
                vocab_size = log_probs.size(-1)

                if step < min_length:
                    log_probs[:, end_token_id] = -1e20

                # Multiply probs by the beam probability.
                log_probs += topk_log_probs.view(-1).unsqueeze(1)

                alpha = alpha
                length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

                # Flatten probs into a list of possibilities.
                curr_scores = log_probs / length_penalty

                if block_ngram > 0:
                    cur_len = alive_seq.size(1)
                    if cur_len > block_ngram:
                        for i in range(alive_seq.size(0)):
                            fail = False
                            words = [int(w) for w in alive_seq[i]]
                            words = [tokenizer.ids_to_tokens[w] for w in words]
                            words = " ".join(words).replace(" ##", "").split()
                            if len(words) <= block_ngram:
                                continue
                            ngrams = [tuple(words[i + j] for j in range(block_ngram))
                                      for i in range(0, len(words) - block_ngram + 1)]
                            ngram = tuple(ngrams[-1])
                            if ngram in ngrams[:-1]:
                                fail = True
                            if fail:
                                curr_scores[i] = -1e5

                curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
                topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

                # Recover log probs.
                topk_log_probs = topk_scores * length_penalty

                # Resolve beam origin and true word ids.
                topk_beam_index = topk_ids.div(vocab_size)
                topk_ids = topk_ids.fmod(vocab_size)
                words = [tokenizer.ids_to_tokens[int(w)] for w in topk_ids[0]]

                # Map beam_index to batch_index in the flat representation.
                batch_index = topk_beam_index + beam_offset[: topk_beam_index.size(0)].unsqueeze(1)
                select_indices = batch_index.view(-1)

                # Append last prediction.
                alive_seq = torch.cat([alive_seq.index_select(0, select_indices), topk_ids.view(-1, 1)], -1)

                is_finished = topk_ids.eq(end_token_id)
                if step + 1 == max_length:
                    is_finished.fill_(1)
                # End condition is top beam is finished.
                end_condition = is_finished[:, 0].eq(1)
                # Save finished hypotheses.
                if is_finished.any():
                    predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                    for i in range(is_finished.size(0)):
                        b = batch_offset[i]
                        if end_condition[i]:
                            is_finished[i].fill_(1)
                        finished_hyp = is_finished[i].nonzero().view(-1)
                        # Store finished hypotheses for this batch.
                        for j in finished_hyp:
                            hypotheses[b].append((topk_scores[i, j], predictions[i, j, 1:]))
                        # If the batch reached the end, save the n_best hypotheses.
                        if end_condition[i]:
                            best_hyp = sorted(hypotheses[b], key=lambda x: x[0], reverse=True)
                            score, pred = best_hyp[0]

                            results["scores"][b].append(score)
                            results["predictions"][b].append(pred)
                            words = [int(w) for w in pred]
                            words = [tokenizer.ids_to_tokens[w] for w in words if w != end_token_id]
                            results['words'][b].append(words)
                    non_finished = end_condition.eq(0).nonzero().view(-1)
                    # If all sentences are translated, no need to go further.
                    if len(non_finished) == 0:
                        break
                    # Remove finished batches for the next step.
                    topk_log_probs = topk_log_probs.index_select(0, non_finished)
                    batch_index = batch_index.index_select(0, non_finished)
                    batch_offset = batch_offset.index_select(0, non_finished)
                    alive_seq = predictions.index_select(0, non_finished).view(-1, alive_seq.size(-1))
                # Reorder states.
                select_indices = batch_index.view(-1)
                src_features = src_features.index_select(0, select_indices)
                dec_states.map_batch_fn(lambda state, dim: state.index_select(dim, select_indices))
            outputs = outputs + (results,)

        return outputs  # (loss), scores, (hidden_states), (attentions)


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1).transpose(0, 1).repeat(count, 1).transpose(0, 1).contiguous().view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x
