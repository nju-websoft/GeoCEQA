import json
import sys
from graphBuild.Graph import EventGraph
from answerGeneration.utils.evaluate import evalAnswer
import numpy as np
import networkx as nx

ppr_memory = {}


def get_subgraph(source, target, lines_map, G, method='bfs', size=500, source_mode=None):
    subgraph = []
    points = set()
    source = [s for s in source if s in lines_map]
    if method == 'bfs':
        queue = []
        for s in source:
            queue.append((s, 0))
        while len(points) <= size and len(queue) > 0:
            curr, distance = queue.pop(0)
            if curr not in lines_map:
                continue
            subgraph.append((curr, 20 - distance))
            points.add(curr)
            queue += [(r, distance + 1) for r in lines_map[curr] if r not in points]
    elif method == 'ppr':
        if len(source) > 0:
            source_key = tuple(sorted(source))
            if source_key not in ppr_memory:
                ppr = nx.pagerank(G,
                                  personalization={p: (1 if p in source else 0) for p in lines_map.keys()})
                ppr_memory[source_key] = ppr
            ppr = ppr_memory[source_key]
            subgraph = [(k, v) for k, v in ppr.items()]
    subgraph = sorted(subgraph, key=lambda x: x[1], reverse=True)
    if size is not None:
        subgraph = subgraph[:size]
    # distance = np.mean([d for _, d in subgraph]) if len(subgraph) > 0 else None
    points = set([p for p, _ in subgraph])
    if size is None:
        subgraph = [p for p, _ in subgraph]
    rate = len([1 for t in target if t in points]) / len(target) if len(target) > 0 else 0
    return subgraph, rate, [t for t in target if t in points]


def subgraph_wrapper(args):
    return get_subgraph(args[0], args[1], method=args[2], size=args[3], lines_map=args[4], G=args[5])


def main():
    if __name__ == "__main__":
        root_path = '../data/answerGeneration/'
        prefix = 'ocsl-dev-'
    else:
        root_path = sys.argv[1]
        prefix = sys.argv[2]
    graph_data = json.load(open(root_path + 'digraph.json'), encoding='UTF-8')
    if prefix.endswith('hop_'):
        node_name = 'nodes'
        qnode_name = 'question_nodes'
        anode_name = 'answer_nodes'
    else:
        node_name = 'events'
        qnode_name = 'question_events'
        anode_name = 'answer_events'
        relation_types = ['qa_cause', 'answer_cause', 'r_qa_cause', 'r_answer_cause',
                          # 'coreference',
                          # 'related', 'contrary', 'context'
                          ]
        graph_corefs = [cor for e_id, e in graph_data['events'].items() for cor in e['corefs']]
        for cor in graph_corefs:
            cor['id'] = cor['event_oid']
        for e_id, e in graph_data['events'].items():
            e['corefs'] = [cor['event_oid'] for cor in e['corefs']]

    # def load_data(filename, prefix):
    #     dataset = [json.loads(line) for line in open(filename, encoding='UTF-8')]
    #     events = EventGraph(filename)
    #     events.process_equal()
    #     events = events.get_events()
    #     for e in events:
    #         e['id'] = '{}-'.format(prefix) + e['id'][6:]
    #     eid_map = {oe['event_oid']: e['id'] for e in events for oe in e['equals']}
    #     for line in dataset:
    #         line.pop('relations')
    #         line['question_events'] = [eid_map[e['id']] for e in line['question_events']]
    #         line['answer_events'] = [eid_map[e['id']] for e in line['answer_events']]
    #     return dataset, events

    # def event_to_key(event):
    #     # return str(EventGraph.getEventSlots(event))
    #     return event['type'], event['concept'], event['modifier'], event['predicate'], event['direction']

    # train_path = root_path + 'train_pred.jsonl'
    # train_data, train_events = load_data(train_path, prefix='train')  # train')
    # test_path = root_path + 'dev_pred.jsonl'
    # test_data, test_events = load_data(test_path, prefix='test')  # test')
    # total_events = [EventGraph.getEventSlots(e, get_id=True) for e in train_events + test_events]
    # data_events_map = {e['id']: e for e in total_events}

    only_causal = False

    all_data = json.load(open(root_path + prefix + 'mapped_dataset.json', encoding='UTF-8'))
    train_data = all_data['train_data']
    dev_data = all_data['dev_data'] if 'dev_data' in all_data else []
    test_data = all_data['test_data']

    # open(root_path + 'train.jsonl', 'w').write('\n'.join([json.dumps(item, ensure_ascii=False) for item in train_data]))
    # open(root_path + 'test.jsonl', 'w').write('\n'.join([json.dumps(item, ensure_ascii=False) for item in test_data]))
    # exit()

    # def eval_scores(event_ids, answer):
    #     pred_data = [','.join([EventGraph.get_text(graph_data['events'][e]) for e in l]) for l in event_ids]
    #     return evalAnswer(pred_data, answer)

    # scores = eval_scores([l['answer_events'] for l in test_data], [l['answer'] for l in test_data])
    # print(scores)

    # etype_weight = {'qa_cause': 2.588, 'answer_cause': 1.858, 'related': 0.845, 'coreference': 0.278,
    #                 'contrary': 0.729}
    lines_map = {k: set([(r['type'], r['opposite']) for r in v]) for k, v in graph_data['relations'].items()}
    G = nx.Graph()
    G.add_nodes_from(lines_map.keys())
    for src, edges in lines_map.items():
        tgt_set = set([e[1] for e in edges])
        for tgt in tgt_set:
            # ablation only causal
            if only_causal:
                etypes = set([e[0] for e in edges if e[1] == tgt and e[0] in relation_types])
            else:
                etypes = set([e[0] for e in edges if e[1] == tgt])
            G.add_edge(src, tgt)

    # G.add_edges_from([(k, vv[1]) for k, v in lines_map.items() for vv in v], key=[])

    import multiprocessing

    pool = multiprocessing.Pool(processes=10)

    for size in [None]:
        for method in ['ppr']:
            fdata = [(line[qnode_name], line[anode_name], method, size, lines_map, G)
                     for line in train_data + dev_data + test_data]
            # csize = 200
            # chunks = len(fdata) // csize + (1 if len(fdata) % csize else 0)
            result = pool.map_async(subgraph_wrapper, fdata, chunksize=200).get()
            # result = sum(result, [])
            for i, line in enumerate(train_data + dev_data + test_data):
                line['subgraph'], line['rate'], line['ans_point'] = result[i]
                # line['subgraph'], line['rate'], line['ans_point'] = get_subgraph(
                #     line['question_events'], line['answer_events'], size=size, method=method)
            rate = np.mean([l['rate'] for l in train_data + dev_data + test_data])
            # scores = eval_scores([l['ans_point'] for l in test_data], [l['answer'] for l in test_data])
            # if size == 500:
            #     train_path = root_path + 'train_{}_{}.jsonl'.format(method, size)
            #     open(train_path, 'w').write('\n'.join([json.dumps(l, ensure_ascii=False) for l in train_data]))
            #     test_path = root_path + 'test_{}_{}.jsonl'.format(method, size)
            #     open(test_path, 'w').write('\n'.join([json.dumps(l, ensure_ascii=False) for l in test_data]))
            if size is None:
                train_path = root_path + prefix + 'train_{}_all.jsonl'.format(method)
                open(train_path, 'w', encoding='UTF-8').write(
                    '\n'.join([json.dumps(l, ensure_ascii=False) for l in train_data]))
                train_path = root_path + prefix + 'valid_{}_all.jsonl'.format(method)
                open(train_path, 'w', encoding='UTF-8').write(
                    '\n'.join([json.dumps(l, ensure_ascii=False) for l in dev_data]))
                test_path = root_path + prefix + 'test_{}_all.jsonl'.format(method)
                open(test_path, 'w', encoding='UTF-8').write(
                    '\n'.join([json.dumps(l, ensure_ascii=False) for l in test_data]))
            print(size, rate)  # , scores)


if __name__ == "__main__":
    main()
