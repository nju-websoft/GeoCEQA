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
""" Fine-tuning the library models for named entity recognition on CoNLL-2003 (Bert or Roberta). """

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import WEIGHTS_NAME, BertConfig, BertForTokenClassification, BertTokenizer

if __name__ == "__main__":
    from metrics import precision, recall, f1
    from load_data import convert_examples_to_features, get_labels, read_examples_from_file
    from model import BertForBinaryTokenClassification
else:
    from .metrics import precision, recall, f1
    from .load_data import convert_examples_to_features, get_labels, read_examples_from_file
    from .model import BertForBinaryTokenClassification

logger = logging.getLogger(__name__)

ALL_MODELS = tuple(BertConfig.pretrained_config_archive_map.keys())

MODEL_CLASSES = {
    "bert": (BertConfig, BertForBinaryTokenClassification, BertTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, labels, pad_token_label_id):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            cuda_indices = [0, 1, 2, 3, 6, 7]
            batch = tuple(t.to(args.device) if i in cuda_indices else t for i, t in enumerate(batch))
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2],
                      "span_labels": batch[3],
                      "span_size": batch[4],
                      "span_list": batch[5],
                      "slot_labels": batch[6],
                      "slot_mask": batch[7],
                      "rel_size": batch[8],
                      "rel_list": batch[9],
                      "question_length": batch[10],
                      "span_null_label_id": labels[0].index('O'),
                      "global_step": global_step,
                      "args": args}

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            # span_logits = outputs[1][0]
            # span_pred = [torch.max(sl, 2)[1] for sl in span_logits].detach().cpu().numpy()
            # print(span_pred.shape)
            # exit()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev")
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model,
                                                            "module") else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, prefix=""):
    eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    span_seq_preds = None
    span_seq_labels = None
    model.eval()
    span_labels = []
    slot_label_ids = []
    span_preds = []
    slot_logits = []
    rel_spans = []
    rel_labels = []
    question_length = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2],
                      "span_labels": batch[3],
                      "span_size": batch[4],
                      "span_list": batch[5],
                      "slot_labels": batch[6],
                      "slot_mask": batch[7],
                      "rel_size": batch[8],
                      "rel_list": batch[9],
                      "question_length": batch[10],
                      "span_null_label_id": labels[0].index('O'),
                      "args": args}

            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1

        input_labels = inputs['span_labels']
        span_logits = logits[0]
        span_pred = logits[1]
        slot_logit = logits[2]
        rel_span = logits[3]

        question_length += [l.tolist()[0] for l in batch[10]]
        span_preds += span_pred
        slot_logits += [item.detach().cpu().numpy() if item is not None else None for item in slot_logit]
        span_size = inputs['span_size'].tolist()
        span_label = inputs['span_list'].tolist()  # bx?x3
        slot_label_id = inputs['slot_labels'].detach().cpu().numpy()  # bx?xlx2
        span_labels += [span[:span_size[i][0]] for i, span in enumerate(span_label)]
        slot_label_ids += [label[:span_size[i][0]] for i, label in enumerate(slot_label_id)]
        offset = (nb_eval_steps - 1) * args.eval_batch_size
        rel_spans += [[tuple((i + offset,) + s) for s in span] for i, span in enumerate(rel_span)]
        rel_size = inputs['rel_size'].tolist()
        rel_label = [l[:rel_size[i][0]] for i, l in enumerate(inputs['rel_list'].tolist())]
        rel_labels += [[tuple([i + offset] + l) for l in label] for i, label in enumerate(rel_label)]

        logits = span_logits
        if span_seq_preds is None:
            # preds = logits.detach().cpu().numpy()
            # out_label_ids = inputs["labels"].detach().cpu().numpy()
            span_seq_preds = logits.detach().cpu().numpy()
            span_seq_labels = input_labels.detach().cpu().numpy()
        else:
            # preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            # out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
            span_seq_preds = np.append(span_seq_preds, logits.detach().cpu().numpy(), axis=0)  # bxlx2xt
            span_seq_labels = np.append(span_seq_labels, input_labels.detach().cpu().numpy(), axis=0)  # bxlx2

    eval_loss = eval_loss / nb_eval_steps
    span_label_list = [[tuple([i] + s) for s in spans] for i, spans in enumerate(span_labels)]
    span_pred_list = [[tuple((i,) + s) for s in spans] for i, spans in enumerate(span_preds)]
    state_list = [[[ll for ll in l if ll[1] == 1] for l in span_label_list],
                  [[ll for ll in l if ll[1] == 1] for l in span_pred_list]]
    action_list = [[[ll for ll in l if ll[1] == 2] for l in span_label_list],
                   [[ll for ll in l if ll[1] == 2] for l in span_pred_list]]
    change_list = [[[ll for ll in l if ll[1] == 3] for l in span_label_list],
                   [[ll for ll in l if ll[1] == 3] for l in span_pred_list]]
    results = {
        "loss": eval_loss,
        "span_precision": precision(span_label_list, span_pred_list),
        "span_recall": recall(span_label_list, span_pred_list),
        "span_f1": f1(span_label_list, span_pred_list),
        "state_precision": precision(state_list[0], state_list[1]),
        "state_recall": recall(state_list[0], state_list[1]),
        "state_f1": f1(state_list[0], state_list[1]),
        "action_precision": precision(action_list[0], action_list[1]),
        "action_recall": recall(action_list[0], action_list[1]),
        "action_f1": f1(action_list[0], action_list[1]),
        "change_precision": precision(change_list[0], change_list[1]),
        "change_recall": recall(change_list[0], change_list[1]),
        "change_f1": f1(change_list[0], change_list[1]),
        "rel_precision": precision(rel_labels, rel_spans),
        "rel_recall": recall(rel_labels, rel_spans),
        "rel_f1": f1(rel_labels, rel_spans)
    }

    def extractSpans(pred, span_null_label_id=0):
        span = [None, None, None]
        spans = []
        for seq_index in range(len(pred)):
            start_label, end_label = pred[seq_index]
            if start_label != span_null_label_id:
                span[0] = start_label
                span[1] = seq_index
            if end_label != span_null_label_id and span[0] and span[0] == end_label:
                span[2] = seq_index + 1
                spans.append(span)
                span = [None, None, None]
        return spans

    def preds2items(index, spans, preds, positive_flag, question_length, method='slot'):  # ?xlx2
        items = []
        for i, span in enumerate(spans):
            pred = [p for k, p in enumerate(preds[i]) if positive_flag[k]]
            innerSpans = extractSpans(pred)
            if method == 'relation':
                fstart = span[1] - 1
                fend = span[2] - 1
                if fstart >= question_length:
                    fstart -= 1
                    fend -= 1
                items += [tuple([index, span[0], fstart, fend] + span_) for span_ in innerSpans]
            elif method == 'slot':
                fstart = span[1] - 1
                fend = span[2] - 1
                if fstart >= question_length:
                    fstart -= 1
                    fend -= 1
                innerSpans = sorted([s for s in innerSpans if s[1] >= fstart and s[2] <= fend], key=lambda x: x[0])
                items.append(tuple([index, span[0], fstart, fend] + sum(innerSpans, [])))
        return items

    slot_preds = [np.argmax(item, axis=3) if item is not None else None for item in slot_logits]  # bx?xlx2
    # truth: span_labels slot_label_ids, pred: span_preds, slot_preds
    items_labels = []
    items_preds = []
    for i in range(len(slot_label_ids)):  # batch
        positive_flag = []
        for start, end in span_seq_labels[i]:
            if start != pad_token_label_id:
                positive_flag.append(True)
            else:
                positive_flag.append(False)
        items_label = preds2items(i, span_labels[i], slot_label_ids[i], positive_flag, question_length[i])
        items_pred = preds2items(i, span_preds[i], slot_preds[i], positive_flag, question_length[i])
        items_labels.append(items_label)
        items_preds.append(items_pred)

    results.update({
        "slot_precision": precision(items_labels, items_preds),
        "slot_recall": recall(items_labels, items_preds),
        "slot_f1": f1(items_labels, items_preds)
    })

    for i, spans in enumerate(span_preds):
        for j, span in enumerate(spans):
            span = list(span)
            span[1] -= 1
            span[2] -= 1
            if span[1] >= question_length[i]:
                span[1] -= 1
                span[2] -= 1
            spans[j] = tuple(span)
        for rels in [rel_spans[i], rel_labels[i]]:
            for j, rel in enumerate(rels):
                rel = list(rel)
                rel[2] -= 1
                rel[3] -= 1
                rel[5] -= 1
                rel[6] -= 1
                if rel[2] >= question_length[i]:
                    rel[2] -= 1
                    rel[3] -= 1
                if rel[5] >= question_length[i]:
                    rel[5] -= 1
                    rel[6] -= 1
                rels[j] = tuple(rel)

    # open('../data/out/item.txt', 'w').write('\n\n'.join(
    #     ['\n'.join([str(span_preds[i]), str(pred), str(rel_labels[i])]) for i, pred in enumerate(rel_spans)]))

    # span_label_lists = []
    # span_pred_lists = []
    #
    # span_seq_preds = np.argmax(span_seq_preds, axis=3)
    #
    # label_map = {i: label for i, label in enumerate(labels[0])}
    #
    # span_label_lists = [[] for _ in range(span_seq_labels.shape[0])]
    # span_pred_lists = [[] for _ in range(span_seq_labels.shape[0])]
    #
    # for i in range(span_seq_labels.shape[0]):
    #     for j in range(span_seq_labels.shape[1]):
    #         if span_seq_labels[i, j, 0] != pad_token_label_id:
    #             span_label_lists[i].append([label_map[span_seq_labels[i, j, 0]], label_map[span_seq_labels[i, j, 1]]])
    #             span_pred_lists[i].append([label_map[span_seq_preds[i, j, 0]], label_map[span_seq_preds[i, j, 1]]])
    #
    # preds_list = [[] for _ in range(len(span_label_lists[0]))]
    # out_label_list = [[] for _ in range(len(span_label_lists[0]))]
    # begin_flag = ['', '']
    #
    # for i in range(len(span_label_lists[0])):
    #     for j in range(len(span_label_lists[i])):
    #         if begin_flag[0]:
    #             out_label_list[i].append('I-' + begin_flag[0])
    #             if span_label_lists[i][j][1] != 'O':
    #                 begin_flag[0] = ''
    #         else:
    #             if span_label_lists[i][j][0] == "O":
    #                 out_label_list[i].append('O')
    #             else:
    #                 out_label_list[i].append('B-' + span_label_lists[i][j][0])
    #                 if span_label_lists[i][j][1] == 'O':
    #                     begin_flag[0] = span_label_lists[i][j][0]
    #         if begin_flag[1]:
    #             preds_list[i].append('I-' + begin_flag[1])
    #             if span_pred_lists[i][j][1] != 'O':
    #                 begin_flag[1] = ''
    #         else:
    #             if span_pred_lists[i][j][0] == "O":
    #                 preds_list[i].append('O')
    #             else:
    #                 preds_list[i].append('B-' + span_pred_lists[i][j][0])
    #                 if span_pred_lists[i][j][1] == 'O':
    #                     begin_flag[1] = span_pred_lists[i][j][0]

    # results.update({
    #     #     "loss": eval_loss,
    #     #     "accuracy": accuracy_score(out_label_list, preds_list),
    #     "span_precision": precision_score(out_label_list, preds_list),
    #     "span_recall": recall_score(out_label_list, preds_list),
    #     "span_f1": f1_score(out_label_list, preds_list)
    # })
    # open('../data/seq.txt', 'w').write(
    # '\n\n'.join(['\n'.join([str(pred), str(rel_span_preds[i])]) for i, pred in enumerate(preds_list)]))

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results, (span_preds, items_preds, rel_spans)


def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir,
                                        "cached_{}_{}_{}".format(mode, list(
                                            filter(None, args.model_name_or_path.split("/"))).pop(),
                                                                 str(args.max_seq_length)))
    if args.do_predict:
        cached_features_file += '_' + args.test_file_name
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples = read_examples_from_file(args.data_dir, mode)
        features = convert_examples_to_features(examples, labels, args.max_seq_length, tokenizer,
                                                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                sep_token=tokenizer.sep_token,
                                                sep_token_extra=bool(args.model_type in ["roberta"]),
                                                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                pad_on_left=bool(args.model_type in ["xlnet"]),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                                                pad_token_label_id=pad_token_label_id,
                                                max_span_length=args.max_span_length
                                                )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_span_label = torch.tensor([f.span_label for f in features], dtype=torch.long)
    # all_binary_span = torch.tensor([f.binary_span for f in features], dtype=torch.long)
    all_span_size = torch.tensor([f.span_size for f in features], dtype=torch.long)
    all_span_list = torch.tensor([f.span_list for f in features], dtype=torch.long)
    all_slot_label = torch.tensor([f.slot_label for f in features], dtype=torch.long)
    all_slot_mask = torch.tensor([f.slot_mask for f in features], dtype=torch.long)
    all_rel_size = torch.tensor([f.rel_size for f in features], dtype=torch.long)
    all_rel_list = torch.tensor([f.rel_list for f in features], dtype=torch.long)
    all_question_length = torch.tensor([f.question_length for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_span_label, all_span_size,
                            all_span_list, all_slot_label, all_slot_mask, all_rel_size, all_rel_list,
                            all_question_length)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--model_name_or_path", default='bert-base-chinese',
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--model_type", default='bert', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--labels", default="", type=str,
                        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run predictions on the test set.")
    parser.add_argument("--predict_file", default="", type=str,
                        help="The file to be predict.")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--pred_dir", default=None, type=str,
                        help="The prediction directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--mode", default='joint', type=str,
                        help="Run mode.")
    parser.add_argument("--max_seq_length", default=470, type=int,  # 470
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_span_length", default=50, type=int,
                        help="The maximum total span length.")
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--max_steps", default=3000, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=100, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--num_train_epochs", default=50.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--eval_all_checkpoints", action="store_true",
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    # parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    # parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    if args.do_predict:
        if len(args.predict_file) == 0:
            raise ValueError("Mode predict need predict file name.")
        os.system('cp {} {}'.format(args.predict_file, os.path.join(args.data_dir, 'test.jsonl')))
        test_file_name = args.predict_file
        test_file_name = test_file_name[test_file_name.rfind('/') + 1:] if '/' in test_file_name else test_file_name
        test_file_name = test_file_name[:test_file_name.rfind('.')]
        args.test_file_name = test_file_name

    # Setup distant debugging if needed
    # if args.server_ip and args.server_port:
    #     # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
    #     import ptvsd
    #     print("Waiting for debugger attach")
    #     ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
    #     ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare CONLL-2003 task
    labels = get_labels(args.labels)
    num_labels = [len(l) for l in labels]
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    if args.do_train:
        config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                              num_labels=num_labels,
                                              cache_dir=args.cache_dir if args.cache_dir else None)
        tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None)
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool(".ckpt" in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train")
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, labels, pad_token_label_id)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                "module") else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev", prefix=global_step)
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w", encoding='UTF-8') as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(args.output_dir)
        model.to(args.device)
        result, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="test")
        # Save results
        output_dir = args.pred_dir if args.pred_dir else args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_test_results_file = os.path.join(output_dir, "test_results.txt")
        with open(output_test_results_file, "w", encoding='UTF-8') as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))
        # Save predictions
        span_list, slot_list, rel_list = predictions
        output_test_predictions_file = os.path.join(output_dir, "{}_pred.jsonl".format(args.test_file_name))
        test_file = os.path.join(args.data_dir, "test.jsonl")
        with open(output_test_predictions_file, "w", encoding='UTF-8') as writer:
            import json
            label_maps = [{i: l for i, l in enumerate(label)} for label in labels]

            def slots2events(id_, slots, ques_length):
                ques_events = []
                answ_events = []
                event_map = {}
                event_slots = {'change': ['concept', 'modifier', 'predicate', 'direction'],
                               'action': ['concept', 'modifier', 'predicate'],
                               'state': ['concept', 'modifier']}
                for i, s in enumerate(slots):
                    offset = 0 if s[2] < ques_length else -ques_length
                    etype = label_maps[0][s[1]]
                    event = {'id': '{}-{}'.format(id_, i + 1),
                             'type': etype,
                             'start': s[2] + offset,
                             'end': s[3] + offset}
                    event_map[(s[1], s[2], s[3])] = event['id']
                    slot_list = [(s[i], s[i + 1], s[i + 2]) for i in range(4, len(s), 3)]
                    slot_map = {}
                    for stype, start, end in slot_list:
                        stype = label_maps[1][stype]
                        if stype not in slot_map:
                            slot_map[stype] = (start + offset, end + offset)
                    for slot_name in event_slots[etype]:
                        if slot_name in slot_map:
                            event[slot_name] = slot_map[slot_name]
                        else:
                            event[slot_name] = None
                    if offset == 0:
                        ques_events.append(event)
                    else:
                        answ_events.append(event)
                return ques_events, answ_events, event_map

            def rels2relations(rels, event_map):
                relations = []
                for r in rels:
                    head = (r[1], r[2], r[3])
                    tail = (r[4], r[5], r[6])
                    rtype = label_maps[2][r[7]]
                    relations.append({'type': rtype,
                                      'head': event_map[head],
                                      'tail': event_map[tail]})
                return relations

            with open(test_file, "r", encoding='UTF-8') as f:
                example_id = 0
                for line in f:
                    line = json.loads(line)
                    if 'id' in line:
                        id_ = line['id']
                        ques = line['question']
                        answ = line['answer']
                        spans = span_list[example_id]
                        slots = slot_list[example_id]
                        rels = rel_list[example_id]
                        assert len(spans) == len(slots)
                        ques_events, answ_events, event_map = slots2events(id_, slots, len(ques))
                        relations = rels2relations(rels, event_map)
                        writer.write(json.dumps({'id': id_, 'question': ques, 'answer': answ,
                                                 'question_events': ques_events, 'answer_events': answ_events,
                                                 'relations': relations}, ensure_ascii=False) + '\n')
                        example_id += 1

        os.system('rm {}'.format(test_file))

    return results


if __name__ == "__main__":
    main()
