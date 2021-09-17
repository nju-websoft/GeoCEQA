import os
from graphBuild.Graph import EventGraph

root_path = '../data/graph/'
graph = EventGraph(root_path + 'corpus_pred.jsonl')
graph.process_equal()
coref_file = root_path + 'corpus_coref_pred.csv'
if not os.path.exists(coref_file):
    dirname = '../data/eventCoref/'
    graph.generate_coref_candidate(event_file=dirname + 'event_corpus.json',
                                   candidate_file=dirname + 'pred_corpus.csv')
    print('Please run again after prediction.')
else:
    relation_only_file = coref_file
    # relation_only_file = root_path + 'relations_only.csv'
    # if not os.path.exists(relation_only_file):
    #     pred_relations = [l.strip().split(',') for l in open(coref_file)]
    #     pred_relations = [l for l in pred_relations if l[2] != 'norelation']
    #     open(relation_only_file, 'w').write('\n'.join([','.join(l) for l in pred_relations]))
    graph.process_coref(relation_only_file)

    relations = sorted(graph.relation_list, key=lambda x: x['count'], reverse=True)
    # graph.save(root_path + 'graph.json')
    graph.to_digraph(root_path + 'digraph.json')
