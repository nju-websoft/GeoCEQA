import os
import sys
import json
import multiprocessing
from graphBuild.Graph import EventGraph
from answerGeneration.utils.evaluate import evalAnswer
import numpy as np

from answerGeneration.utils.fastevent import event_similarity


def event_sim(e1, e2, ft_weight=1):
    score = evalAnswer(answer=e2['span'], gold=e1['span'],
                       metrics=(('rouge-1', 'f'), ('rouge-2', 'f'), ('rouge-l', 'f')), filterd=False)
    es = event_similarity(e1, e2)
    es = max(0, es - 0.5) / 0.5
    return ((score[0] + score[1]) / 2 + es * ft_weight) / 2


def generate_supp(args):
    event = args[0]
    graph_corefs = args[1]
    contr_relations = args[2]
    undup_to_graph = args[3]
    supp = sorted([(event, event_similarity(event, e), e) for e in graph_corefs
                   if e['type'] == event['type'] and (event['id'], e['id']) not in contr_relations],
                  key=lambda x: x[1], reverse=True)
    undup_supp = []
    undup_id_count = {}
    for l in supp:
        undup_id = undup_to_graph[l[2]['id']]
        if undup_id not in undup_id_count:
            undup_id_count[undup_id] = 0
        if undup_id_count[undup_id] < 5:
            undup_id_count[undup_id] += 1
            undup_supp.append(l)
    return event['id'], undup_supp[:50]


def sorted_supp(args):
    supp = args[0]
    undup_to_graph = args[1]
    supp = sorted([(event['span'], event_sim(event, e), e) for event, _, e in supp],
                  key=lambda x: x[1], reverse=True)
    undup_supp = []
    undup_id_set = set()
    for l in supp:
        undup_id = undup_to_graph[l[2]['id']]
        if undup_id not in undup_id_set:
            undup_id_set.add(undup_id)
            undup_supp.append(l)
    return args[0][0][0]['id'], undup_supp[:5]


def main():
    num_workers = 12
    if __name__ == '__main__':
        graph_root_path = '../data/answerGeneration/'
    else:
        graph_root_path = sys.argv[1]
    graph_data = json.load(open(graph_root_path + 'digraph.json', encoding='UTF-8'))
    graph_corefs = [cor for e_id, e in graph_data['events'].items() for cor in e['corefs']]
    for cor in graph_corefs:
        cor['id'] = cor['event_oid']
    for e_id, e in graph_data['events'].items():
        e['corefs'] = [cor['event_oid'] for cor in e['corefs']]

    def load_data(filename, prefix):
        dataset = [json.loads(line) for line in open(filename, encoding='UTF-8')]
        events = EventGraph(filename)
        events.process_equal()
        events = events.get_events()
        for e in events:
            e['id'] = '{}-'.format(prefix) + e['id'][6:]
        eid_map = {oe['event_oid']: e['id'] for e in events for oe in e['equals']}
        for line in dataset:
            line.pop('relations')
            line['question_events'] = [eid_map[e['id']] for e in line['question_events']]
            line['answer_events'] = [eid_map[e['id']] for e in line['answer_events']]
        return dataset, events

    def event_to_key(event):
        # return str(EventGraph.getEventSlots(event))
        return event['type'], event['concept'], event['modifier'], event['predicate'], event['direction']

    if __name__ == '__main__':
        data_root_path = '../data/answerGeneration/'
        train_path = data_root_path + 'train_pred.jsonl'
        train_data, train_events = [], []  # load_data(train_path, prefix='train')  # train')
        test_path = data_root_path + 'dev_pred.jsonl'  # 'dev_pred.jsonl'
        test_data, test_events = load_data(test_path, prefix='dev')  # test')
        coref_output_dir = "../data/eventCoref/"
        prefix = 'dev-'
    else:
        data_root_path = sys.argv[2]
        train_data, train_events = [], []
        test_path = sys.argv[3]
        test_data, test_events = load_data(test_path, prefix='dev')
        coref_output_dir = sys.argv[4]
        prefix = sys.argv[5]

    train_len, test_len = len(train_data), len(test_data)
    all_data = train_data + test_data
    all_events = train_events + test_events

    total_events = [EventGraph.getEventSlots(e, get_id=True) for e in all_events]
    data_events_map = {e['id']: e for e in total_events}

    pred_path = data_root_path + prefix + 'link_pred.csv'
    if not os.path.exists(pred_path):
        graph_feature_set = set([event_to_key(e) for e in graph_corefs])
        data_eids = [e['id'] for e in total_events if event_to_key(e) not in graph_feature_set]
        graph_eids = [e['id'] for e in graph_corefs]
        events_coref = graph_corefs + total_events
        events_coref_map = {e['id']: e for e in events_coref}
        dirname = coref_output_dir
        json.dump(events_coref, open(dirname + prefix + 'event_link.json', 'w', encoding='UTF-8'), ensure_ascii=False)

        attrs = ['concept', 'modifier', 'predicate', 'direction']

        def getSpanSet(e_id):
            event = events_coref_map[e_id]
            span_set = set(''.join([event[attr] for attr in attrs if attr in event and event[attr]]))
            return span_set

        span_set_map = {e_id: getSpanSet(e_id) for e_id in data_eids + graph_eids}
        type_map = {e_id: events_coref_map[e_id]['type'] for e_id in data_eids + graph_eids}
        # def getSpanSet(e_id):
        #     return set(events_coref_map[e_id]['span'])
        print('start to generate candidate...')
        candidates = [(i, j) for i in data_eids for j in graph_eids
                      if type_map[i] == type_map[j] and
                      len(span_set_map[i].intersection(span_set_map[j])) > 0]
        candidates += [(j, i) for i, j in candidates]
        candidates = [','.join((i, j)) + ',norelation' for i, j in candidates]
        print('total {} pairs'.format(len(candidates)))
        open(dirname + prefix + 'pred_link.csv', 'w', encoding='UTF-8').write('\n'.join(candidates))
        print('Please run again after prediction.')
    else:
        mapping_path = data_root_path + prefix + 'mapped_dataset.json'
        if not os.path.exists(mapping_path):

            graph_events_map = graph_data['events']
            undup_to_graph = {cor: e_id for e_id, e in graph_events_map.items() for cor in e['corefs']}
            linking_path = data_root_path + prefix + 'linked_events.json'
            if not os.path.exists(linking_path):
                relation_only_file = pred_path
                pred_relations = [l.strip().split(',') for l in open(relation_only_file, encoding='UTF-8')]
                pred_relations = [sorted(l[:2]) + [l[2]] for l in pred_relations]
                contr_relations = set([(p[0], p[1]) for p in pred_relations if p[2] == 'contrary'])
                contr_relations.update([(b, a) for a, b in contr_relations])
                pred_relations = [(l[0], undup_to_graph[l[1]], l[2]) for l in pred_relations]

                # find the slot equal mapping node
                graph_feature_map = {event_to_key(e): e['id'] for e in graph_corefs}
                pred_relation_map = {e['id']: [undup_to_graph[graph_feature_map[event_to_key(e)]], set(), set()]
                                     for e in total_events if event_to_key(e) in graph_feature_map}

                # add the related and coreference node
                for pred in pred_relations:
                    # if data_events_map[pred[0]]['type'] == graph_events_map[pred[1]]['type']:
                    if pred[0] not in pred_relation_map:
                        pred_relation_map[pred[0]] = [
                            None, set(), set(), set(), set()]  # event, coref, relat
                    record = pred_relation_map[pred[0]]
                    if pred[2] == 'coreference':
                        record[1].add(pred[1])
                    elif pred[2] == 'related':
                        record[2].add(pred[1])

                # visual_relations = [
                #     (data_events_map[k], [graph_events_map[vv] for vv in v[1]], [graph_events_map[vv] for vv in v[2]])
                #     for k, v in pred_relation_map.items() if len(v[1]) > 0 and v[0] is None]
                # svisual_relations = [(l[0], sorted(l[1], key=lambda x: event_sim(l[0], x), reverse=True),
                #                       sorted(l[2], key=lambda x: event_sim(l[0], x), reverse=True)) for l in
                #                      visual_relations]
                #
                # def esim(e1, e2):
                #     score = evalAnswer(answer=e2['span'], gold=e1['span'],
                #                        metrics=(('rouge-1', 'f'), ('rouge-2', 'f'), ('rouge-l', 'f')))
                #     return score
                #
                # evals = [(l[0], [[e] + esim(l[0], e) + [event_similarity(l[0], e)] for e in l[1]],
                #           [[e] + esim(l[0], e) + [event_similarity(l[0], e)] for e in l[2]]) for l in
                #          visual_relations[:100]]
                # evals_12f = [(l[0], sorted(l[1], key=lambda x: x[1] + x[2] + 2 * x[4], reverse=True),
                #               sorted(l[2], key=lambda x: x[1] + x[2] + 2 * x[4], reverse=True)) for l in
                #              evals]

                # pred_coref_map = {
                #     k: (v[0] if v[0] else max(v[1], key=lambda x: event_sim(data_events_map[k], graph_events_map[x])))
                #     for k, v in pred_relation_map.items() if v[0] or len(v[1]) > 0}
                # pred_related_map = {k: max(v[2], key=lambda x: event_sim(data_events_map[k], graph_events_map[x]))
                #                     for k, v in pred_relation_map.items() if len(v[2]) > 0}
                # pred_supp_map = {
                #     event['id']: max(
                #         [(event['span'], event_similarity(event, e), e) for e in graph_corefs if e['type'] == event['type']
                #          ], key=lambda x: x[1]) for event in total_events
                #     if event['id'] not in pred_coref_map and event['id'] not in pred_related_map}
                pred_map = {
                    k: [([(0.999, v[0])] if v[0] else []) + \
                        (sorted([(event_sim(data_events_map[k], graph_events_map[vv]), vv) for vv in v[1]],
                                key=lambda x: x[0] if v[1] else [], reverse=True)[:5]),
                        (sorted([(event_sim(data_events_map[k], graph_events_map[vv]), vv) for vv in v[2]],
                                key=lambda x: x[0] if v[2] else [], reverse=True)[:5])]
                    for k, v in pred_relation_map.items()}

                # pred_map.update(pred_coref_map)

                pool = multiprocessing.Pool(num_workers)
                pred_supp_map = pool.map_async(generate_supp,
                                               [(event, graph_corefs, contr_relations, undup_to_graph)
                                                for event in total_events if
                                                event['id'] not in pred_map or len(pred_map[event['id']][0]) < 5],
                                               chunksize=1000)
                pred_supp_map = pred_supp_map.get()
                pool.close()
                pred_supp_map = {k: v for k, v in pred_supp_map}

                pool = multiprocessing.Pool(num_workers)
                pred_supp_map = {k: v for k, v in
                                 pool.map_async(sorted_supp, [(v, undup_to_graph) for _, v in pred_supp_map.items()],
                                                chunksize=1000).get()}
                pool.close()

                for k, v in pred_supp_map.items():
                    nv = []
                    for l in v:
                        undup_e = l[2]
                        graph_eid = undup_to_graph[undup_e['id']]
                        pred_set = set([gid for _, gid in pred_map[k][0]]) if k in pred_map else []
                        if l[1] < 0.6 or graph_eid in pred_set:
                            continue
                        nv.append((l[1], graph_eid))
                    v.clear()
                    v.extend(nv)
                for k, v in pred_map.items():
                    nv = []
                    for l in v[1]:
                        graph_eid = l[1]
                        pred_set = set([gid for _, gid in (pred_map[k][0] if k in pred_map else [])
                                        + (pred_supp_map[k] if k in pred_supp_map else [])])
                        if l[0] < 0.4 or graph_eid in pred_set:
                            continue
                        nv.append((l[0], graph_eid))
                    v[1].clear()
                    v[1].extend(nv)
                pred_map = {event['id']:
                                (pred_map[event['id']][0] if event['id'] in pred_map else []) +
                                (pred_supp_map[event['id']] if event['id'] in pred_supp_map else []) +
                                (pred_map[event['id']][1] if event['id'] in pred_map else [])
                            for event in total_events}
                pred_map = {k: v for k, v in pred_map.items() if len(v) > 0}
                json.dump(pred_map, open(linking_path, 'w', encoding='UTF-8'))
            # exit()

            pred_map = json.load(open(linking_path, 'r', encoding='UTF-8'))
            visual_relations = [
                (k, data_events_map[k], [graph_events_map[vv[1]] for vv in v])
                for k, v in pred_map.items() if len(v[0]) > 1]

            # import networkx as nx
            #
            # lines_map = {k: set([r['opposite'] for r in v]) for k, v in graph_data['relations'].items()}
            # G = nx.Graph()
            # G.add_nodes_from(lines_map.keys())
            # G.add_edges_from([(k, vv) for k, v in lines_map.items() for vv in v])

            # path_dict = dict(nx.all_pairs_shortest_path(G))

            # pred_supp_map = {k: max([[iv[0]['span'], event_sim(iv[0], iv[2]), iv[2]] for iv in v], key=lambda x: x[1])
            #                  for k, v in pred_supp_map.items()}
            # json.dump(pred_supp_map, open('../data/tmp1.5_new.json', 'w'), indent=2, ensure_ascii=False)

            # pred_coref_map = {k: undup_to_graph[v[2]['id']] for k, v in pred_supp_map.items() if v[1] >= 0.65}
            # pred_coref_map.update(pred_map)
            # pred_related_map = {k: undup_to_graph[v[2]['id']] for k, v in pred_supp_map.items() if v[1] >= 0.5}
            # pred_related_map.update(pred_map)

            # pred_coref_map.update(pred_related_map)

            def dataset_event_mapping(dataset):
                not_cover = [0, 0]
                all_event = [0, 0]
                for line in dataset:
                    # question_events = [pred_coref_map[e_id] if e_id in pred_coref_map else pred_related_map[e_id]
                    #                    for e_id in line['question_events']
                    #                    if e_id in pred_coref_map or e_id in pred_related_map]
                    # answer_events = [pred_coref_map[e_id] for e_id in line['answer_events'] if e_id in pred_coref_map]
                    question_events = [(e_id, pred_map[e_id])
                                       for e_id in line['question_events'] if e_id in pred_map]
                    answer_events = [(e_id, [l for l in pred_map[e_id] if l[0] >= 0.4])
                                     for e_id in line['answer_events'] if e_id in pred_map]

                    # path = [nx.single_source_dijkstra_path(G, v[1]) for _, v in question_events]

                    # vq = [
                    #     (k, data_events_map[k],
                    #      [(vv[0], graph_events_map[vv[1]]) for vv in v])
                    #     for k, v in question_events]
                    #
                    # va = [
                    #     (k, data_events_map[k],
                    #      [(
                    #          # [len(p[vv[1]]) for p in path] ,
                    #          vv[0], graph_events_map[vv[1]]) for vv in v])
                    #     for k, v in answer_events if len(v) > 0 and v[0][0] < 0.6]

                    question_events = [v[0][1] for _, v in question_events]
                    answer_events = [[vv for vv in v if vv[0] >= 0.4] for _, v in answer_events]
                    nae = []
                    for v in answer_events:
                        for score, eid in v:
                            if score >= 0.5:
                                nae.append(eid)
                                break
                        else:
                            if len(v) > 0:
                                nae.append(v[0][1])
                    answer_events = nae
                    all_event[0] += len(line['question_events'])
                    all_event[1] += len(line['answer_events'])
                    if len(question_events) == 0:
                        not_cover[0] += 1
                    if len(answer_events) == 0:
                        not_cover[1] += len(line['answer_events']) - len(answer_events)
                    line['question_events'] = question_events
                    line['answer_events'] = []
                    for e_id in answer_events:
                        if e_id not in line['answer_events']:
                            line['answer_events'].append(e_id)
                # print(not_cover, all_event, not_cover[0] / (all_event[0] if all_event[0] != 0 else 1e-9),
                #       not_cover[1] / (all_event[1] if all_event[1] != 0 else 1e-9))

                return dataset

            # train_data = dataset_event_mapping(train_data)
            # test_data = dataset_event_mapping(test_data)
            all_data = dataset_event_mapping(all_data)

            print(np.mean([len(item['answer_events']) for item in all_data]))
            # train_data, test_data = add_sim_events(train_data, test_data)
            json.dump({'train_data': all_data[:train_len], 'test_data': all_data[train_len:]}
                      , open(mapping_path, 'w', encoding='UTF-8'), ensure_ascii=False)
            print('mapped dataset saved.')

        # def add_sim_events(train_data, test_data):
        #     graph_events_map = graph_data['events']
        #     graph_texts = [(id_, EventGraph.get_text(e)) for id_, e in graph_events_map.items()]
        #     import time
        #     start_time = time.time()
        #     for i, line in enumerate(train_data + test_data):
        #         if i % 10 == 0:
        #             print(i, time.time() - start_time)
        #         answer = line['answer']
        #         evals = []
        #         for e in graph_texts:
        #             eval = evalAnswer(answer=e[1], gold=answer,
        #                               metrics=(('rouge-1', 'f'), ('rouge-2', 'f'), ('rouge-l', 'f')))
        #             evals.append([e[0], e[1]] + eval)
        #         evals_2 = sorted(evals, key=lambda x: x[3], reverse=True)
        #         evals_l = sorted(evals, key=lambda x: x[4], reverse=True)
        #         evals_12 = sorted(evals, key=lambda x: x[2] + x[3], reverse=True)
        #         evals = [(x[0], x[1], x[4]) for x in evals_l]
        #         evals_topk = evals[:100]
        #         line['related_events'] = [x[0] for x in evals_topk]
        #     return train_data, test_data


if __name__ == "__main__":
    main()
