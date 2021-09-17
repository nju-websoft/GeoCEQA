import json, random

random.seed(1234)
root_path = '../data/eventCoref/'

event_undup = json.load(open(root_path + 'event_labeled.json', encoding='UTF-8'))
event_id_to_content = {e['id']: e for e in event_undup}
labeled_relations = json.load(open(root_path + 'relation_labeled.json', encoding='UTF-8'))
l = set([l['label'] for l in labeled_relations])
event_ids = [e['id'] for e in event_undup]


def getSpanCharSet(e_id):
    return set(event_id_to_content[e_id]['span'])


nsample_related = [{'head': i, 'tail': j, 'label': 'norelation'} for i in
                   random.choices(event_ids, k=20000) for j in random.choices(event_ids, k=50) if i != j and
                   len(getSpanCharSet(i).intersection(getSpanCharSet(j))) > 0]
print(len(nsample_related))
random.shuffle(nsample_related)

nsample_unrelated = [{'head': i, 'tail': j, 'label': 'norelation'} for i in
                     random.choices(event_ids, k=50000) for j in random.choices(event_ids, k=1) if i != j]
random.shuffle(nsample_unrelated)

ns_length = int(len(labeled_relations) * 0.5)
relations = labeled_relations + nsample_related[:ns_length] + nsample_unrelated[:ns_length]
random.shuffle(relations)
print(len(relations))
length = int(len(relations) * 4 / 5)
open(root_path + 'train.csv', 'w', encoding='UTF-8').write(
    '\n'.join([','.join(line.values()) for line in relations[:length]]))
open(root_path + 'dev.csv', 'w', encoding='UTF-8').write(
    '\n'.join([','.join(line.values()) for line in relations[length:]]))
