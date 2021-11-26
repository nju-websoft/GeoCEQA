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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

# apex rouge dgl-cu100(cuda version)
# transformers==2.2.2 scikit-learn==0.23.1 dgl jieba
import argparse
import glob
import logging
import os
import json
import random

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig,
                          BertTokenizer,
                          RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer,
                          XLMConfig, XLMForSequenceClassification,
                          XLMTokenizer, XLNetConfig,
                          XLNetForSequenceClassification,
                          XLNetTokenizer,
                          DistilBertConfig,
                          DistilBertForSequenceClassification,
                          DistilBertTokenizer,
                          AlbertConfig,
                          AlbertForSequenceClassification,
                          AlbertTokenizer,
                          )

from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, ndcg_score

if __name__ == "__main__":
    from load_data import AnsGenProcessor, convert_examples_to_features, \
        GraphDataset, num_class, only_causal, question_aware
    from model import BertGnnNodeClassification
    from utils.evaluate import evalAnswer
else:
    from .load_data import AnsGenProcessor, convert_examples_to_features, \
        GraphDataset, num_class, only_causal, question_aware
    from .model import BertGnnNodeClassification
    from .utils.evaluate import evalAnswer

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig,
                                                                                RobertaConfig, DistilBertConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertGnnNodeClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    'albert': (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, labels, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  # num_workers=4,
                                  collate_fn=GraphDataset.collate)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
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
            # guid, graph, answer
            tensor_graph = batch[0]
            answer_ids = batch[5]
            tensor_graph.to(args.device)
            answer_ids = answer_ids.to(args.device)
            inputs = {'graph': tensor_graph, 'answer_ids': answer_ids, 'args': args}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if loss is not None:
                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
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

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, labels, during_train=True)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
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


def evaluate(args, model, tokenizer, labels, prefix="", during_train=False):
    def simple_accuracy(preds, labels):
        return (preds == labels).mean()

    def metrics(preds, labels):
        # acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        return {
            # "acc": acc,
            "f1": f1
            # "acc_and_f1": (acc + f1) / 2,
        }

    def rank_score(rank_list, positive_list, k=20):
        to_rank = [l[:k] for l in rank_list]
        positive_set = [set(l) for l in positive_list]
        y_true = [[1 if id_ in positive_list[i] else 0 for id_, _ in l] for i, l in enumerate(to_rank)]
        ap = []
        rap = []
        for index, t in enumerate(y_true):
            s = 0
            count = 0
            for i in range(k):
                if t[i] == 1:
                    count += 1
                    s += count / (i + 1)
            exists = len(positive_set[index])
            if exists == 0:
                continue
            if count == 0:
                s = 0
                ap.append(s)
                rap.append(s)
            else:
                ap.append(s / count)
                rap.append(s / exists)
        p_scores = [[ll[1] for ll in l] for l in to_rank]
        return {
            "MAP": np.mean(ap),
            "MAP-r": np.mean(rap),
            "nDCG": ndcg_score(y_true, p_scores)
        }

    eval_output_dir = args.pred_dir if args.pred_dir else args.output_dir
    results = {}
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=GraphDataset.collate)

    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    node_preds = []
    node_label_ids = []
    instance_pred_rank = []
    all_instance_pred_rank=[]
    instance_labels = []
    instance_plabels = []
    instance_rlabels = []
    instance_ids = []
    question = []
    gold_answer = []
    # if not during_train:
    answer_gen = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        # guid, graph, answer
        tensor_graph = batch[0]
        tensor_graph.to(args.device)
        answer_ids = batch[5]
        answer_ids = answer_ids.to(args.device)

        with torch.no_grad():
            inputs = {'graph': tensor_graph, 'answer_ids': answer_ids,
                      'during_train': during_train, 'eval_mode': True, 'tokenizer': tokenizer, 'args': args}
            outputs = model(**inputs)
            tmp_eval_loss, scores = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        instance_ids += batch[1]
        question += batch[2]
        gold_answer += batch[3]
        # if not during_train:
        answer_gen += [''.join(words[0]) for words in outputs[2]['words']]
        scores = scores.detach().cpu().tolist()
        out_label_ids = tensor_graph.ndata['v-label'].detach().cpu().numpy().tolist()
        node_label_ids += out_label_ids
        node_index = batch[4]
        for i, ns in enumerate(tensor_graph.batch_num_nodes('event')):
            if not question_aware:
                ns -= 1
            sort_scores = sorted([(i, s) for i, s in enumerate(scores[:ns])], key=lambda x: x[1], reverse=True)
            all_instance_pred_rank.append([(node_index[i][str(j)], p) for j, p in sort_scores] if ns!=1 else [])
            if ns == 1: continue

            instance_pred_rank.append([(node_index[i][str(j)], p) for j, p in sort_scores])
            if num_class == 2:
                instance_labels.append([node_index[i][str(j)] for j, p in enumerate(out_label_ids[:ns]) if p == 1])
            else:
                instance_plabels.append([node_index[i][str(j)] for j, p in enumerate(out_label_ids[:ns]) if p == 2])
                instance_rlabels.append([node_index[i][str(j)] for j, p in enumerate(out_label_ids[:ns]) if p > 0])
            scores = scores[ns:]
            out_label_ids = out_label_ids[ns:]

    eval_loss = eval_loss / nb_eval_steps
    if num_class == 2:
        result = rank_score(instance_pred_rank, instance_labels)
    else:
        result = rank_score(instance_pred_rank, instance_plabels)
        result.update({k + '-R': v for k, v in rank_score(instance_pred_rank, instance_rlabels).items()})
    results.update(result)
    results['eval_loss'] = eval_loss

    answer_concat = AnsGenProcessor.index_to_text([[i for i, s in ins[:20]] for ins in all_instance_pred_rank])
    output_text = list(map(list, zip(question, answer_concat, gold_answer)))
    rouge_score = evalAnswer(answer_concat, gold_answer, max_len=60,
                             metrics=(('rouge-1', 'f'), ('rouge-2', 'f'), ('rouge-l', 'f')))

    results.update({'rouge-1': rouge_score[0], 'rouge-2': rouge_score[1], 'rouge-l': rouge_score[2], })
    # 'rouge-2-r': rouge_score[3], 'rouge-l-r': rouge_score[4]})

    # logger.info("***** Eval results {} *****".format(prefix))
    # for key in sorted(results.keys()):
    #     logger.info("  %s = %s", key, str(results[key]))

    # if not during_train:
    rouge_score = evalAnswer(answer_gen, gold_answer, max_len=60,
                             metrics=(('rouge-1', 'f'), ('rouge-2', 'f'), ('rouge-l', 'f')))

    results.update({'rouge-1-gen': rouge_score[0], 'rouge-2-gen': rouge_score[1], 'rouge-l-gen': rouge_score[2]})

    output_prefix = '45_dev_'
    output_prefix = ''

    output_eval_file = os.path.join(eval_output_dir, prefix, f"{output_prefix}eval_results.txt")
    with open(output_eval_file, "w", encoding='UTF-8') as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
            writer.write("%s = %s\n" % (key, str(results[key])))

    if not during_train:
        gen_text = list(map(list, zip(question, answer_concat, answer_gen, gold_answer)))
        output_pred_file = os.path.join(eval_output_dir, prefix, f"{output_prefix}pred_results.txt")
        with open(output_pred_file, "w", encoding='UTF-8') as writer:
            answer_text = [l.split(',') for l in answer_concat]
            answer_text_score = [[(a, i[1]) for a, i in zip(ans, ins)] for ans, ins in
                                 zip(answer_text, instance_pred_rank)]
            writer.write('\n'.join([json.dumps(ans, ensure_ascii=False) for ans in answer_text_score]))
        output_pred_file = os.path.join(eval_output_dir, prefix, f"{output_prefix}node_results.txt")
        with open(output_pred_file, "w", encoding='UTF-8') as writer:
            writer.write('\n'.join(answer_concat))
        output_pred_file = os.path.join(eval_output_dir, prefix, f"{output_prefix}text_results.txt")
        with open(output_pred_file, "w", encoding='UTF-8') as writer:
            writer.write('\n'.join(answer_gen))
        output_pred_file = os.path.join(eval_output_dir, prefix, f"{output_prefix}answer.txt")
        with open(output_pred_file, "w", encoding='UTF-8') as writer:
            writer.write('\n'.join(gold_answer))

    return results


def load_and_cache_examples(args, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = AnsGenProcessor(args.data_dir)
    output_mode = 'classification'
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}_{}class'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length), str(args.max_node_size), str(num_class)))
    if only_causal:
        cached_features_file += '_ocsl'
    if not question_aware:
        cached_features_file += '_oaware'
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(args.data_dir, args.max_node_size) if evaluate else \
            processor.get_train_examples(args.data_dir, args.max_node_size)
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                max_ans_length=args.max_ans_length,
                                                max_node_size=args.max_node_size,
                                                output_mode=output_mode,
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    tensor_features = [f.to_tensor() for f in features]

    dataset = GraphDataset(tensor_features)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--pred_dir", default=None, type=str,
                        help="The prediction directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_type", default="bert", type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default='bert-base-chinese', type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--max_seq_length", default=50, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_ans_length", default=200, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_node_size", default=200, type=int,
                        help="The maximum total input node size")
    parser.add_argument("--block_ngram", default=3, type=int,
                        help="Ngram blocked in beam search.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")

    parser.add_argument("--gnn_hidden_size", default=128, type=int,
                        help="Hidden size of gnn module.")
    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1000.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=10000, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=100, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=500,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")

    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    label_list = AnsGenProcessor.get_labels()
    num_labels = len(label_list)

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
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, label_list, tokenizer)
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
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)

            result = evaluate(args, model, tokenizer, label_list, prefix=prefix)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
