import os
from Graph import EventGraph
import argparse
parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--input_path", default=None, type=str, required=True,
                    help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--output_path", default=None, type=str, required=True,
                    help="The output directory where the model predictions and checkpoints will be written.")
args = parser.parse_args()
root_path = args.input_path
graph = EventGraph(root_path + 'Corpus_pred.jsonl')
graph.process_equal()
coref_file = root_path + 'test_results.csv'
print(coref_file)
if not os.path.exists(coref_file):
    dirname = args.output_path
    graph.generate_coref_candidate(event_file=dirname + 'event_id_map.json',
                                   candidate_file=dirname + 'test.csv')
    print('Please run again after prediction.')
else:
    print("building graph")
    relation_only_file = coref_file
    # relation_only_file = root_path + 'relations_only.csv'
    # if not os.path.exists(relation_only_file):
    #     pred_relations = [l.strip().split(',') for l in open(coref_file)]
    #     pred_relations = [l for l in pred_relations if l[2] != 'norelation']
    #     open(relation_only_file, 'w').write('\n'.join([','.join(l) for l in pred_relations]))
    graph.process_coref(relation_only_file)

    relations = sorted(graph.relation_list, key=lambda x: x['count'], reverse=True)
    # graph.save(root_path + 'graph.json')
    graph.to_digraph(args.output_path + 'AEG.json')
