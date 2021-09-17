import os
import copy
import json
import logging
import dgl
import torch
import networkx as nx
from torch.utils.data import Dataset
from transformers import DataProcessor, InputExample
from transformers import glue_convert_examples_to_features
from graphBuild.Graph import EventGraph

logger = logging.getLogger(__name__)

question_aware = True

only_causal = False
if only_causal:
    edge_types = ['qa_cause', 'answer_cause', 'r_qa_cause', 'r_answer_cause']
else:
    edge_types = ['qa_cause', 'answer_cause', 'r_qa_cause', 'r_answer_cause',
                  # 'coreference',
                  'related', 'contrary', 'context']

num_class = 2
symbols = {
    "BOS": 5,
    "EOS": 6,
    "PAD": 0
}


class GraphExample(InputExample):
    # def __init__(self, guid, question, gold_answer, gold_nodes, related_nodes, nodes, edge_map):
    def __init__(self, guid, question, gold_answer, gold_nodes, nodes, edge_map):
        super().__init__(guid, text_a=question)
        self.guid = guid
        self.question = question
        self.gold_answer = gold_answer
        self.gold_nodes = gold_nodes
        # self.related_nodes = related_nodes
        self.nodes = nodes
        self.edge_map = edge_map


class GraphFeatures(object):
    def __init__(self, guid, node_features, node_size, out_edges, node_index,
                 question, answer, answer_ids):
        self.guid = guid
        self.node_features = node_features,
        self.node_size = node_size,
        self.node_index = {str(v): k for k, v in node_index.items()}
        self.out_edges = out_edges,
        self.question = question,
        self.answer = answer
        self.answer_ids = answer_ids

    def preprocess(self):
        if type(self.node_size) is tuple:
            self.node_size = self.node_size[0]
        if type(self.node_features) is tuple:
            self.node_features = self.node_features[0]
        if type(self.out_edges) is tuple:
            self.out_edges = self.out_edges[0]

    def dglGraph(self):
        graph = dgl.DGLGraph()
        graph.add_nodes(self.node_size)
        # (input_ids, attention_mask, token_type_ids, node_label)
        graph.ndata['v-h'] = torch.tensor([(f[0], f[1], f[2]) for f in self.node_features], dtype=torch.long)
        graph.ndata['v-label'] = torch.tensor([f[3] for f in self.node_features], dtype=torch.long)

        edge_type_map = {t: i for i, t in enumerate(['qa_cause', 'answer_cause', 'related', 'contrary', 'coreference'])}
        # (head, e_type, e_count, tail)
        graph.add_edges([e[0] for e in self.out_edges], [e[3] for e in self.out_edges])
        edge_features = [(edge_type_map[e[1]], e[2]) for e in self.out_edges]

        add_self_loop = False
        if add_self_loop:
            graph.add_edges(graph.nodes(), graph.nodes())
            edge_features += [(len(edge_type_map), 1) for _ in graph.nodes()]
        graph.edata['e-h'] = torch.tensor(edge_features, dtype=torch.long)
        relation_count = [[0, 0, 0, 0, 0] for _ in range(len(graph.nodes))]
        for e in self.out_edges:
            relation_count[e[0]][edge_type_map[e[1]]] += 1
            relation_count[e[3]][edge_type_map[e[1]]] += 1
        graph.edata['norm'] = torch.tensor(
            [1 / relation_count[e[0]][edge_type_map[e[1]]] for e in self.out_edges], dtype=torch.float)
        return graph

    def heterograph(self):
        edge_type_map = {t: i for i, t in enumerate(edge_types)}

        edges = []
        node_flag = [False for _ in range(self.node_size)]
        multi_edges = False
        if not multi_edges:
            edges = set()
        for head, type_, count, tail in self.out_edges:
            if type_ in edge_types:
                node_flag[head] = True
                node_flag[tail] = True
                if not multi_edges:
                    edges.add((head, type_, tail))
                    edges.add((tail, type_, head))
                else:
                    edges += [(head, type_, tail) for _ in range(count)]
                    edges += [(tail, type_, head) for _ in range(count)]
        type_name = [('event', t, 'event') for t in edge_types]
        data_dict = {t: [(e[0], e[2]) for e in edges if e[1] == t[1]] for t in type_name}
        if self.node_size == 1:
            data_dict[('event', 'related' if not only_causal else 'qa_cause', 'event')].append([0, 0])

        for i in range(self.node_size):
            if not node_flag[i]:
                data_dict[('event', 'related' if not only_causal else 'qa_cause', 'event')].append([i, i])
                node_flag[i] = True

        graph = dgl.heterograph(data_dict)
        # (input_ids, attention_mask, token_type_ids, node_label)
        neg_set = [i for i in range(self.node_size) if not node_flag[i]]
        last_pos = 0
        for i in range(self.node_size):
            if node_flag[i]:
                last_pos = i
        if not question_aware and self.node_size == 2:
            last_pos = 1
        graph.ndata['v-h'] = torch.tensor(
            [(f[0], f[1], f[2]) for i, f in enumerate(self.node_features) if i <= last_pos], dtype=torch.long)
        graph.ndata['v-label'] = torch.tensor(
            [f[3] for i, f in enumerate(self.node_features) if i <= last_pos], dtype=torch.long)

        return graph

    def to_tensor(self):
        self.preprocess()
        graph = self.heterograph()
        return graph, self.guid, self.question, self.answer, self.node_index, \
               torch.tensor(self.answer_ids, dtype=torch.long)

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class GraphDataset(Dataset):
    @staticmethod
    def collate(samples):
        # The input `samples` is a list of pairs
        #  (id, graph, answer).
        graphs, ids, ques, answ, node_index, answer_ids = map(list, zip(*samples))
        batched_graph = dgl.batch_hetero(graphs)
        answer_ids = torch.stack(answer_ids, 0)
        return batched_graph, ids, ques, answ, node_index, answer_ids

    def __init__(self, features):
        super(GraphDataset, self).__init__()
        self.num = len(features)
        # guid, graph, answer
        self.features = features

    def __getitem__(self, index):
        return self.features[index]

    def __len__(self):
        return self.num


class AnsGenProcessor(DataProcessor):
    """Processor for the event coreference data set."""

    def __init__(self, data_dir):
        graph_data = json.load(open(os.path.join(data_dir, 'digraph.json'), encoding='UTF-8'))
        event_map = graph_data['events']
        line_map = graph_data['relations']
        for k, v in event_map.items():
            v['id'] = k
            v.pop('event_id')
            v.pop('corefs')
        AnsGenProcessor.event_map = event_map
        AnsGenProcessor.line_map = line_map

    @classmethod
    def index_to_text(cls, preds):
        return [','.join([EventGraph.get_text(cls.event_map[p]) for p in pred]) for pred in preds]

    def get_train_examples(self, data_dir, max_node_size):
        """See base class."""
        return self._create_examples(
            [json.loads(l) for l in open(os.path.join(data_dir, "train.jsonl"), encoding='UTF-8')],
            "train", data_dir, max_node_size)

    def get_dev_examples(self, data_dir, max_node_size):
        """See base class."""
        return self._create_examples(
            [json.loads(l) for l in open(os.path.join(data_dir, "test.jsonl"), encoding='UTF-8')],
            "test", data_dir, max_node_size)

    @staticmethod
    def get_labels():
        """See base class."""
        if num_class == 2:
            return [0, 1]
        else:
            return [0, 1, 2]

    def _create_examples(self, lines, set_type, data_dir, max_node_size):
        """Creates examples for the training and dev sets."""
        # graph_data = json.load(open(os.path.join(data_dir, 'graph.json')))
        # event_map = graph_data['events']
        # line_map = graph_data['relations']
        # for k, v in event_map.items():
        #     v['id'] = k
        #     v.pop('event_id')
        #     v.pop('corefs')
        event_map = AnsGenProcessor.event_map
        line_map = AnsGenProcessor.line_map

        examples = []
        for (i, line) in enumerate(lines):
            # if len(line['subgraph']) == 0 or len(line['answer_events']) == 0:
            #     continue
            guid = "%s-%s-%s" % (set_type, i, line['id'])
            line['subgraph'] = line['subgraph'][:max_node_size]
            subgraph_set = set(line['subgraph'])
            examples.append(GraphExample(guid=guid, question=line['question'], gold_answer=line['answer'],
                                         gold_nodes=line['answer_events'],  # related_nodes=line['related_events'],
                                         nodes=[event_map[n] for n in line['subgraph']],
                                         edge_map={n: [on for on in line_map[n] if on['opposite'] in subgraph_set]
                                                   for n in subgraph_set}))
        print('total {} examples'.format(len(examples)))
        return examples


def convert_examples_to_features(examples, tokenizer,
                                 max_length=512,
                                 max_node_size=2000,
                                 max_ans_length=512,
                                 label_list=None,
                                 output_mode=None,
                                 pad_on_left=False,
                                 cls_token="[CLS]",
                                 cls_token_segment_id=1,
                                 sep_token="[SEP]",
                                 start_token_id=symbols['BOS'],
                                 end_token_id=symbols['EOS'],
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """
    Loads a data file into a list of ``InputFeatures``


    """

    def node2feature(node):
        attribute_name = ['type', 'concept', 'predicate', 'modifier', 'direction']

        def ifNone(attr_name):
            attr = node[attr_name]
            return attr if attr else 'non'

        return '\\'.join([ifNone(attr) for attr in attribute_name])

    features = []
    # max_total_length = [0, 0, 0]
    # max_item = ['', '', '']
    for (ex_index, example) in enumerate(examples):
        if ex_index % 1000 == 0:
            logger.info("Writing example %d" % (ex_index))

        question = example.question
        gold_answer = example.gold_answer
        gold_nodes = example.gold_nodes
        gold_node_set = set(gold_nodes)
        if num_class == 3:
            related_node_set = set(example.related_nodes[:50])
        nodes = example.nodes[:max_node_size]
        node_index = {n['id']: i for i, n in enumerate(nodes)}
        node_set = set([n['id'] for n in nodes])
        edge_map = {k: v for k, v in example.edge_map.items() if k in node_set}

        node_features = []
        answer_ids = tokenizer.convert_tokens_to_ids(list(gold_answer))
        answer_ids = [start_token_id] + answer_ids
        if len(answer_ids) >= max_ans_length:
            answer_ids = answer_ids[:max_ans_length - 1]
            answer_ids.append(end_token_id)
        else:
            answer_ids.append(end_token_id)
            answer_ids += [pad_token] * (max_ans_length - len(answer_ids))
        if len(nodes) == 0:
            node_feature = '\\'.join(['n'] * 5)
            node_question = question
            if question_aware:
                total_length = 3 + len(node_question) + len(node_feature)
            else:
                total_length = 2 + len(node_feature)
            if total_length > max_length:
                node_question = node_question[total_length - max_length:]
            tokens = [cls_token]
            for i, word in enumerate(node_feature):
                if word == 'n':
                    word = 'non'
                tokens.append(word)
            tokens += [sep_token]
            token_type_ids = [sequence_a_segment_id] * len(tokens)
            if question_aware:
                for word in node_question:
                    tokens.append(word)
                tokens += [sep_token]
                token_type_ids += [sequence_b_segment_id] * (len(tokens) - len(token_type_ids))
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
            assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                                max_length)
            assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                                max_length)

            node_features.append((input_ids, attention_mask, token_type_ids, 1))
        for node in nodes:
            node_feature = node2feature(node)
            node_question = question
            if num_class == 2:
                node_label = 1 if node['id'] in gold_node_set else 0
            else:
                node_label = 2 if node['id'] in gold_node_set else (1 if node['id'] in related_node_set else 0)
            eng_map = {'n': 'non', 's': 'state', 'a': 'action', 'c': 'change'}
            node_feature = node_feature.replace('non', 'n').replace('state', 's') \
                .replace('action', 'a').replace('change', 'c')
            if question_aware:
                total_length = 3 + len(node_question) + len(node_feature)
            else:
                total_length = 2 + len(node_feature)
            # if len(node_feature) > max_total_length[0]:
            #     max_total_length[0] = len(node_feature)
            #     max_item[0] = node_feature
            # if total_length > max_total_length[1]:
            #     max_total_length[1] = total_length
            #     max_item[1] = node_feature, node_question
            if total_length > max_length:
                node_question = node_question[total_length - max_length:]
            tokens = [cls_token]
            for i, word in enumerate(node_feature):
                if word in eng_map:
                    word = eng_map[word]
                tokens.append(word)
            tokens += [sep_token]
            token_type_ids = [sequence_a_segment_id] * len(tokens)
            if question_aware:
                for word in node_question:
                    tokens.append(word)
                tokens += [sep_token]
                token_type_ids += [sequence_b_segment_id] * (len(tokens) - len(token_type_ids))
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
            assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                                max_length)
            assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                                max_length)

            node_features.append((input_ids, attention_mask, token_type_ids, node_label))

        if not question_aware:
            node_question = question
            total_length = 2 + len(node_question)
            if total_length > max_length:
                node_question = node_question[total_length - max_length:]
            tokens = [cls_token]
            for word in node_question:
                tokens.append(word)
            tokens += [sep_token]
            token_type_ids = [sequence_a_segment_id] * len(tokens)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
            assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                                max_length)
            assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                                max_length)

            node_features.append((input_ids, attention_mask, token_type_ids, 1))
        out_edges = []
        for n_id, edges in edge_map.items():
            outs = [(node_index[n_id], e['type'], e['count'], node_index[e['opposite']])
                    for e in edges if e['direction'] == 'out' and n_id != e['opposite']]
            out_edges += outs

        node_size = len(node_features)
        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("node_size: %d" % node_size)

        features.append(
            GraphFeatures(guid=example.guid,
                          node_features=node_features,
                          node_size=node_size,
                          node_index=node_index,
                          out_edges=out_edges,
                          question=question,
                          answer=gold_answer,
                          answer_ids=answer_ids))
    # logger.info("max total length: " + str(max_total_length) + str(max_item))

    return features
