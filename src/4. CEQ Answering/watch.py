import json
import networkx as nx
import numpy as np
from answerGeneration.load_data import edge_types as relation_types

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

root_path = '../data/answerGeneration/'

train_data_ = [json.loads(l) for l in open(root_path + 'train_ppr_all_new.jsonl', encoding='UTF-8')]
test_data_ = [json.loads(l) for l in open(root_path + 'test_ppr_all_new.jsonl', encoding='UTF-8')]

# for l in train_data_ + test_data_:
#     l['subgraph'] = [t[0] for t in l['subgraph']]
all_data = train_data_ + test_data_

# open(root_path + 'train_pn.jsonl', 'w').write('\n'.join([json.dumps(l, ensure_ascii=False) for l in train_data_]))
# open(root_path + 'test_pn.jsonl', 'w').write('\n'.join([json.dumps(l, ensure_ascii=False) for l in test_data_]))

print(np.mean([len(item['answer_events']) for item in all_data]))
for size in (150, 200, 250, 300, 500):
    print(np.mean([len([1 for e in item['answer_events']
                        if e in set([l for l in item['subgraph'][:size]])]) for item in all_data]))

exit()

# open(root_path + 'train_3c.jsonl', 'w').write('\n'.join([json.dumps(l, ensure_ascii=False) for l in train_data]))
# open(root_path + 'test_3c.jsonl', 'w').write('\n'.join([json.dumps(l, ensure_ascii=False) for l in test_data]))

# re_data = []
# size = [200, 300, 400]
# size_i = 0
# params = []
# for i, line in enumerate(open(root_path + 'result.txt', 'r')):
#     line = line.strip()
#     if len(line) == 0:
#         size_i += 1
#     elif not line.startswith('2020'):
#         params = [float(x) for x in line.split()]
#         size_i = 0
#     else:
#         p = [float(x) for x in line.split()[-2:]]
#         re_data.append([size[size_i]] + params + p)
# for s in size:
#     s_data = sorted([d for d in re_data if d[0] == s], key=lambda x: x[4], reverse=True)
#     print(s_data)
#
# # subgraph sample
#
# graph_data = json.load(open(root_path + 'digraph.json'))
# graph_edges = graph_data['relations']

# sample
import math

for lower, divi, size in [(0.012, 15.5, 200), (0.012, 15.75, 200), (0.008, 20, 300), (0.008, 21, 300)]:
    logger.info('{} {} {}'.format(size, lower, divi))
    train_data = [json.loads(l) for l in open(root_path + 'train_all_re.jsonl', encoding='UTF-8')]
    test_data = [json.loads(l) for l in open(root_path + 'test_all_re.jsonl', encoding='UTF-8')]
    all_data = train_data + test_data
    for line in all_data:
        subgraph = [[n[0], np.exp(n[1] + 3) - np.exp(3), np.log1p(n[2]) / divi if n[2] >= lower else 0] for n in
                    line['subgraph']]
        line['subgraph'] = [n[0] for n in sorted(subgraph, key=lambda x: x[1] + x[2], reverse=True)[:size]]
    open(root_path + 'train_{}_{}_{}.jsonl'.format(size, lower, divi), 'w', encoding='UTF-8').write(
        '\n'.join([json.dumps(l, ensure_ascii=False) for l in train_data])
    )
    open(root_path + 'test_{}_{}_{}.jsonl'.format(size, lower, divi), 'w', encoding='UTF-8').write(
        '\n'.join([json.dumps(l, ensure_ascii=False) for l in test_data])
    )
exit()

for lower in [0.013, 0.012, 0.011, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003]:
    # for divi in [math.pow(2, i) for i in range(-3, 8)]:
    # for divi in [i for i in range(15, 25)]: # + [i for i in range(28, 40)]:
    for divi in [i / 4 for i in range(22 * 4, 28 * 4)]:  # + [i for i in range(28, 40)]:
        answer_weights = []
        neg_count = {}
        for line in all_data:
            # subgraph = [[n[0], np.expm1(n[1] + 3) - 19.08, np.log2(np.log2(n[2] + 2)) / 100 / divi] for n in
            # subgraph = [[n[0], np.expm1(n[1] + 3) - 19.0855, n[2] / divi if n[2] >= lower else 0] for n in
            subgraph = [[n[0], np.exp(n[1] + 3) - np.exp(3), np.log1p(n[2]) / divi if n[2] >= lower else 0] for n in
                        line['subgraph']]
            # levels = {}
            # for n in subgraph:
            #     levels[n[2]] = levels.get(n[2], 0) + 1
            # levels = sorted([(k, v) for k, v in levels.items()], key=lambda x: x[0], reverse=True)
            # ppr_list = [n[0] for n in subgraph]
            # answer_list = [ppr_list.index(e) for e in line['answer_events'] if e in set(ppr_list)]
            # answer_list = [(index, subgraph[index]) for index in answer_list]
            # answer_weights += [(a[0], a[1][2]) for a in answer_list if a[0] > 100]
            # for node in subgraph:
            #     if node[0] not in line['answer_events']:
            #         value = str(min(0.06, round(node[2], 3)))
            #         neg_count[value] = neg_count.get(value, 0) + 1

            # subgraph = sorted(subgraph, key=lambda x: x[1] + x[2], reverse=True)
            # line['subgraph_'] = [n[0] for n in subgraph]
            line['subgraph_'] = subgraph
        # open(root_path + 'answer_weights.csv', 'w').write(
        #     '\n'.join(['{}, {}'.format(a[0], a[1]) for a in answer_weights]))
        # open(root_path + 'neg_weights.csv', 'w').write(
        #     '\n'.join(['{}, {}'.format(a[0], a[1]) for a in neg_count.items()]))
        # exit()

        print(lower, divi)
        for size in [300, 400]:
            for dii in [0, 1 / 8, 1 / 4, 3 / 8, 1 / 2, 5 / 8, 3 / 4, 7 / 8, 1]:
                # for dii in [0, 1 / 4, 1 / 2, 3 / 4, 1]:
                num = []
                for item in all_data:
                    subgraph = item['subgraph_']
                    subgraph = sorted(subgraph, key=lambda x: x[1], reverse=True)
                    front_len = int(size * dii)
                    subgraph = subgraph[:front_len] + \
                               sorted(subgraph[front_len:], key=lambda x: x[1] + x[2], reverse=True)[:size - front_len]
                    subgraph = set([n[0] for n in subgraph])
                    num.append(len([1 for e in item['answer_events'] if e in subgraph]))

                logger.info('%f %f', dii, np.mean(num))
            print()

            # logger.info(
            #     np.mean([len([1 for e in item['answer_events'] if e in item['subgraph_'][:size]]) for item in all_data]))
            # logger.info(
            #     np.mean([len([1 for e in item['answer_events'] if e in getsub(item['subgraph_'])]) for item in all_data]))
exit()

# weight
train_data = [json.loads(l) for l in open(root_path + 'train_ppr_all.jsonl', encoding='UTF-8')]
test_data = [json.loads(l) for l in open(root_path + 'test_ppr_all.jsonl', encoding='UTF-8')]
patterns = json.load(open(root_path + 'pattern_resum.json', encoding='UTF-8'))
patt_weight = {tuple(l[0]): l[1] for l in patterns}
start_weight = max(list(patt_weight.values())) * 2

for index, line in enumerate(train_data + test_data):
    if index % 100 == 0:
        logger.info('start to process ' + str(index))
    q_events = line['question_events']
    if len(q_events) == 0:
        continue
    # graph_nodes = {n[0]: [n[1], 0, [None for e in q_events]] for n in
    #                line['subgraph']}  # e_id, ppr, meta-path, shortest
    graph_nodes = {n[0]: [n[1], 0, [None for e in q_events]] for n in
                   line['subgraph']}  # e_id, ppr, meta-path, shortest
    weight = 1 / len(q_events)
    queue = [(e, [], i) for i, e in enumerate(q_events)]  # event, path, ques_index
    for i, e in enumerate(q_events):
        graph_nodes[e][1] = start_weight
        graph_nodes[e][2][i] = 0
    # visited = [set([e]) for e in q_events]
    # path_count = {(): 1}
    while len(queue) > 0:
        e_id, rpath, q_index = queue.pop(0)
        edges = [e for e in graph_edges[e_id] if e['direction'] == 'out']
        for edge in edges:
            next_e = edge['opposite']
            # if next_e in visited[q_index]:
            #     continue
            e_type = edge['type']
            next_path = rpath + [e_type]
            tuple_path = tuple(next_path)
            path_len = len(next_path)
            node_info = graph_nodes[next_e]
            if node_info[2][q_index] is None:
                node_info[2][q_index] = path_len
            if node_info[2][q_index] > path_len:
                raise Exception('error len')
            if node_info[2][q_index] == path_len:
                if tuple_path in patt_weight:
                    node_info[1] += patt_weight[tuple_path] * weight
                    # node_info[1].append(tuple_path)
                # visited[q_index].add(next_e)
                #     path_count[tuple_path] = path_count.get(tuple_path, 0) + 1
                if path_len < 4:
                    queue.append((next_e, next_path, q_index))
    # for k, v in graph_nodes.items():
    # v[1] = np.sum([patt_weight[path] / path_count[path] for path in v[1]]) * weight

    line['subgraph'] = [[n[0]] + graph_nodes[n[0]] for n in line['subgraph']]

open(root_path + 'train_all_re.jsonl', 'w', encoding='UTF-8').write(
    '\n'.join([json.dumps(l, ensure_ascii=False) for l in train_data]))
open(root_path + 'test_all_re.jsonl', 'w', encoding='UTF-8').write(
    '\n'.join([json.dumps(l, ensure_ascii=False) for l in test_data]))
exit()

# related_path = root_path + 'related_dataset,json'
# all_data = json.load(open(related_path))
# train_data = all_data['train_data']
# test_data = all_data['test_data']

# open(root_path + 'pattern.csv', 'w').write('\n'.join(['{},{}'.format('.'.join(l[0]), l[1])
#                                                       for l in json.load(open(root_path + 'pattern.json', 'r'))]))

# results = json.load(open('../data/tmp1.5.json', 'r'))
# results = sorted([v for k, v in results.items()], key=lambda x: x[1], reverse=True)


graph_data = json.load(open(root_path + 'digraph.json', encoding='UTF-8'))
etype_count = {}

edges_set = set()
for e_id, es in graph_data['relations'].items():
    for e in es:
        e1 = (e_id, e['type'], e['opposite'])
        e2 = (e['opposite'], e['type'], e_id)
        if e1 in edges_set or e2 in edges_set:
            continue
        edges_set.add(e1)
        edges_set.add(e2)
        etype = e['type']
        etype_count[etype] = etype_count.get(etype, 0) + 1

lines_map = {k: set([r['opposite'] for r in v]) for k, v in graph_data['relations'].items()}
G = nx.Graph()
G.add_nodes_from(lines_map.keys())
G.add_edges_from([(k, vv) for k, v in lines_map.items() for vv in v])

# short_ques = [l['question'][::-1][:30][::-1] for l in all_data]

edges_map = graph_data['relations']


def get_edges(from_n, to_n):
    return list(set([edge['type'] for edge in edges_map[from_n]
                     if edge['opposite'] == to_n and edge['direction'] == 'out']))


# for size in [100, 200, 300, 400, 500, 800, 1000, 1500, 2000]:
#     for data in [train_data_, test_data_]:
#         for line in data:
#             line['inpoints'] = len([1 for e in line['answer_events'] if e in line['subgraph'][:size]])
#             # line['arate'] = len([1 for e in line['answer_events'] if e in line['subgraph'][:size]]) \
#             #                 / (len(line['answer_events']) if len(line['answer_events']) > 0 else 1)
#             # line['rrate'] = len([1 for e in line['related_events'] if e in line['subgraph'][:size]]) / 100
#             # nodes_1_hop = [rel['opposite'] for qe in line['question_events'] if qe in edges_map for rel in edges_map[qe]]
#             # nodes_1_hop_set = set(nodes_1_hop)
#             # nodes_2_hop = [rel['opposite'] for qe in nodes_1_hop
#             #                for rel in edges_map[qe] if rel['opposite'] not in nodes_1_hop_set]
#             # nodes_2_hop_set = set(nodes_2_hop).union(nodes_1_hop_set)
#             # line['rate'] = len([1 for e in nodes_1_hop_set if e in line['subgraph'][:size]]) \
#             #                / (len(nodes_1_hop_set) if len(nodes_1_hop_set) > 0 else 1)
#         print(size, np.mean([l['inpoints'] for l in data]))

from functools import reduce

relation_count = {s: [0 for _ in range(5)] for s in relation_types}
unconnected = 0


def shortest_path(G, e1, e2):
    global unconnected
    try:
        path = nx.shortest_path(G, e1, e2)
    except:
        unconnected += 1
        path = []
    return path


pattern_count = {}
len_count = [0, 0, 0, 0, 0]
all_count = 0
for line in all_data:
    qevents = line['question_events']
    aevents = line['answer_events']
    short_path = [(e, e_, shortest_path(G, e, e_)) for e in qevents for e_ in aevents
                  if e in edges_map and e_ in edges_map]
    short_path = [p + tuple(get_edges(p[2][i], p[2][i + 1]) for i in range(len(p[2]) - 1)) for p in short_path]
    for path in short_path:
        pattern = path[3:3 + 4]
        p_len = len(pattern)
        if p_len == 0:
            continue
        len_count[p_len - 1] += 1
        all_p = reduce(lambda a, b: a * b, [len(s) for s in pattern])
        for r1 in pattern[0]:
            if p_len > 1:
                for r2 in pattern[1]:
                    if p_len > 2:
                        for r3 in pattern[2]:
                            if p_len > 3:
                                for r4 in pattern[3]:
                                    pattern_count[(r1, r2, r3, r4)] = pattern_count.get((r1, r2, r3, r4), 0) + 1 / all_p
                            else:
                                pattern_count[(r1, r2, r3)] = pattern_count.get((r1, r2, r3), 0) + 1 / all_p
                    else:
                        pattern_count[(r1, r2)] = pattern_count.get((r1, r2), 0) + 1 / all_p
            else:
                pattern_count[(r1,)] = pattern_count.get((r1,), 0) + 1 / all_p
        for i, edges in enumerate(pattern):
            all_count += 1
            for e in edges:
                relation_count[e][i] += 1 / len(edges)
                relation_count[e][4] += 1 / len(edges)
    a = 0
relation_count = {k: [vv / etype_count[k] for vv in v] for k, v in relation_count.items()}
# pattern_count = {k: v / all_count  for k, v in pattern_count.items()}
patterns = sorted([(k, v) for k, v in pattern_count.items()], key=lambda x: x[1], reverse=True)

p_count = [sum(len_count[i:]) for i in range(4)]
p_prob = [p_count[i + 1] / p_count[i] for i in range(3)]

# weight
pattern_sum = {}
for patt, c in patterns:
    for i in range(1, 1 + min(4, len(patt))):
        pre_patt = tuple(patt[:i])
        pattern_sum[pre_patt] = pattern_sum.get(pre_patt, 0) + c

pattern_sum = {}
path_count = {}
for index, line in enumerate(all_data):
    if index % 100 == 0:
        logger.info('start to process ' + str(index))
    q_events = line['question_events']
    a_events = set(line['answer_events'])
    if len(q_events) == 0 or len(line['answer_events']) == 0:
        continue

    queue = [(e, [], i) for i, e in enumerate(q_events)]  # event, path, ques_index
    visited = [{e: 0} for e in q_events]
    while len(queue) > 0:
        e_id, rpath, q_index = queue.pop(0)
        edges = [e for e in graph_edges[e_id] if e['direction'] == 'out']
        for edge in edges:
            next_e = edge['opposite']
            e_type = edge['type']
            next_path = rpath + [e_type]
            tuple_path = tuple(next_path)
            path_len = len(next_path)
            visited_ = visited[q_index]
            if next_e in visited_:
                if visited_[next_e] < path_len:
                    continue
                if visited_[next_e] > path_len:
                    raise Exception('error len')
            visited_[next_e] = path_len
            if next_e in a_events:
                pattern_sum[tuple_path] = pattern_sum.get(tuple_path, 0) + 1
            path_count[tuple_path] = path_count.get(tuple_path, 0) + 1
            if path_len < 4:
                queue.append((next_e, next_path, q_index))

pattern_sum = sorted([(k, v / path_count[k]) for k, v in pattern_sum.items()], key=lambda x: x[1], reverse=True)

json.dump(pattern_sum, open(root_path + 'pattern_resum.json', 'w', encoding='UTF-8'))
# json.dump(patterns, open(root_path + 'pattern.json', 'w'))

exit()
