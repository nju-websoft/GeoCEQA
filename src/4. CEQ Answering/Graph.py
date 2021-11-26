import json


class EventGraph:
    @staticmethod
    def getEventSlots(event, get_id=False):
        slots = {}
        for slot_name in ['type', 'span', 'concept', 'modifier', 'predicate', 'direction']:
            slots[slot_name] = event[slot_name]
        if get_id:
            slots['id'] = event['id']
        return slots

    @staticmethod
    def get_text(event):
        span_slot = [False for c in event['span']]
        slots = [event['concept'], event['predicate'], event['modifier'], event['direction']]
        for slot in slots:
            if not slot:
                continue
            start = event['span'].index(slot)
            for i in range(start, start + len(slot)):
                span_slot[i] = True
        span_slot = [event['span'][i] for i, flag in enumerate(span_slot) if flag]
        return ''.join(span_slot)

    def __init__(self, filename):
        self.event_id_to_content, self.relation_list, self.text_id_to_qapair = self._read_data(filename)

    def get_events(self):
        event_list = []
        for k, v in self.event_id_to_content.items():
            item = {"id": k}
            item.update(v)
            event_list.append(item)
        return event_list

    def process_equal(self):
        # combine
        self.event_id_to_content = self._event_combine(self.event_id_to_content,
                                                       lambda x: (
                                                           x['type'], x['concept'], x['modifier'], x['predicate'],
                                                           x['direction']))
        self.relation_list = self._relation_combine(self.event_id_to_content, self.relation_list, 'equals')
        # count
        self.relation_list = self._relation_count(self.relation_list)
        # self.relation_eid_to_content = self.relation_to_map(self.relation_list)

    def get_output_events(self):
        event_id_to_content = {}
        for e_id, e_content in self.event_id_to_content.items():
            item = self.getEventSlots(e_content)
            corefs = []
            for coref in e_content['corefs']:
                citem = self.getEventSlots(coref)
                citem['event_oid'] = coref['event_oid']
                corefs.append(citem)
            item.update({'event_id': e_id, 'corefs': corefs})
            event_id_to_content[e_id] = item
        return event_id_to_content

    def save(self, filename):
        event_id_to_content = self.get_output_events()
        with open(filename, 'w', encoding='UTF-8') as file:
            out_data = {'events': event_id_to_content, 'relations': self.relation_eid_to_content}
            json.dump(out_data, file, ensure_ascii=False, indent=2)

    def generate_coref_candidate(self, event_file, candidate_file):
        events = []
        for e_id, e_content in self.event_id_to_content.items():
            slots = self.getEventSlots(e_content)
            slots['id'] = e_id
            events.append(slots)
        json.dump(events, open(event_file, 'w', encoding='UTF-8'), ensure_ascii=False)
        eid_set = list(self.event_id_to_content.keys())
        attrs = ['concept', 'modifier', 'predicate', 'direction']

        def getSpanSet(e_id):
            event = self.event_id_to_content[e_id]
            span_set = set(''.join([event[attr] for attr in attrs if attr in event and event[attr]]))
            return set(self.event_id_to_content[e_id]['span'])

        span_set_map = {e_id: getSpanSet(e_id) for e_id in eid_set}
        type_map = {e_id: self.event_id_to_content[e_id]['type'] for e_id in eid_set}
        print('start to generate candidate...')
        print(len(self.event_id_to_content), 'nodes...')

        candidates = [','.join((i, j)) + ',norelation' for i in eid_set for j in eid_set
                      if i != j and type_map[i] == type_map[j] and
                      len(span_set_map[i].intersection(span_set_map[j])) > 0]
        print(len(candidates), 'candidates...')
        open(candidate_file, 'w', encoding='UTF-8').write('\n'.join(candidates))

    def process_coref(self, file_name):
        pred_relations = [l.strip().split(',') for l in open(file_name, encoding='UTF-8')]
        # print(len(pred_relations))
        coref_relations = [l for l in pred_relations if l[2] == 'coreference']
        contrary_relations = [l for l in pred_relations if l[2] == 'contrary']
        uni_set = set()
        coref_bi_set = set()
        for line in coref_relations:
            line = tuple(sorted(line[:2]))
            if line in uni_set:
                uni_set.remove(line)
                coref_bi_set.add(line)
            else:
                uni_set.add(line)
        # coref_set = coref_bi_set.union(uni_set)
        uni_set = set()
        contr_bi_set = set()
        for line in contrary_relations:
            line = tuple(sorted(line[:2]))
            if line in uni_set:
                uni_set.remove(line)
                contr_bi_set.add(line)
            else:
                uni_set.add(line)
        # print(len(contr_bi_set), len(coref_bi_set))

        pred_relations_event = [(self.event_id_to_content[l[0]], self.event_id_to_content[l[1]], l[2]) for l in
                                pred_relations if tuple(sorted(l[:2])) in coref_bi_set]

        self.relation_list += [{'count': 1, 'head': l[0], 'tail': l[1],
                                'type': (l[2] if l[2] != 'coreference' else 'related')} for l in pred_relations]

        coref_rels = {}
        for s, e in coref_bi_set:
            if s not in coref_rels:
                coref_rels[s] = set()
            coref_rels[s].add(e)
            if e not in coref_rels:
                coref_rels[e] = set()
            coref_rels[e].add(s)
        contr_rels = {}
        for s, e in contr_bi_set.union(uni_set):
            if s not in contr_rels:
                contr_rels[s] = set()
            contr_rels[s].add(e)
            if e not in contr_rels:
                contr_rels[e] = set()
            contr_rels[e].add(s)
        self.event_id_to_content = self._coref_combine(self.event_id_to_content, coref_rels, contr_rels)

        most_frequnce = sorted([e for e in self.event_id_to_content.values()], key=lambda e: len(e['corefs']),
                               reverse=True)

        # aaa = sorted([v for k, v in self.event_id_to_content.items() if True or v.update({'id': k})],
        #              key=lambda x: len(x['corefs']), reverse=True)
        self.relation_list = self._relation_combine(self.event_id_to_content, self.relation_list, 'corefs')

        self.relation_list = self._relation_count(self.relation_list)
        # bbb = sorted(self.relation_list, key=lambda x: x['count'], reverse=True)

        self.relation_eid_to_content = self._relation_to_map(self.relation_list)
        # ccc = sorted([(k, v) for k, v in self.relation_eid_to_content.items()],
        #              key=lambda x: len(x[1]), reverse=True)
        eid_set = set(self.relation_eid_to_content.keys())
        self.event_id_to_content = {k: v for k, v in self.event_id_to_content.items() if k in eid_set}

    def _coref_combine(self, event_id_to_content, coref_rels, contr_rels):
        id_to_type = {k: v['type'] for k, v in event_id_to_content.items()}
        contr_set = set()
        for k, rel in contr_rels.items():
            for v in rel:
                contr_set.add((k, v))
                contr_set.add((v, k))

        def isContrary(e_a, e_b):
            if (e_a, e_b) in contr_set:
                return True
            if id_to_type[e_a] != id_to_type[e_b]:
                return True
            return False

        # event_ids = list(event_id_to_content.keys())
        coref_id_map = {}
        # import time
        # start_time = time.time()
        for id_i, coref_set in coref_rels.items():
            # if id_i == 'undup-387':
            #     id_i = id_i
            for id_j in coref_set:
                e_i = self.event_id_to_content[id_i]
                e_j = self.event_id_to_content[id_j]
                if id_i not in coref_id_map and id_j not in coref_id_map:
                    coref_id_map[id_i] = {id_i, id_j}
                    coref_id_map[id_j] = coref_id_map[id_i]
                elif id_i in coref_id_map and id_j not in coref_id_map:
                    if sum([1 if isContrary(id_, id_j) else 0 for id_ in coref_id_map[id_i]], 0) == 0:
                        coref_id_map[id_i].add(id_j)
                        coref_id_map[id_j] = coref_id_map[id_i]
                elif id_i not in coref_id_map and id_j in coref_id_map:
                    if sum([1 if isContrary(id_, id_i) else 0 for id_ in coref_id_map[id_j]], 0) == 0:
                        coref_id_map[id_j].add(id_i)
                        coref_id_map[id_i] = coref_id_map[id_j]
                elif id_i in coref_id_map and id_j in coref_id_map:
                    if sum([1 if isContrary(id_, id_i) else 0 for id_ in coref_id_map[id_j]], 0) == 0 and \
                            sum([1 if isContrary(id_, id_j) else 0 for id_ in coref_id_map[id_i]], 0) == 0:
                        ij_set = coref_id_map[id_i].union(coref_id_map[id_j])
                        for ij in ij_set:
                            coref_id_map[ij] = ij_set
            # if id_i in coref_id_map and (id_i == 'undup-387' or 'undup-387' in coref_id_map[id_i]):
            #     coref_e_set = [self.event_id_to_content[id_] for id_ in coref_id_map[id_i]]
            #     if len(coref_e_set) > 100:
            #         coref_ei_set = [self.event_id_to_content[id_] for id_ in coref_set]
            #         coref_e_set = coref_e_set
        # print(time.time() - start_time)
        coref_sets_list = []
        for v in coref_id_map.values():
            if v not in coref_sets_list:
                coref_sets_list.append(v)
        new_events = []
        for sets in coref_sets_list:
            item = {'corefs': []}
            for s in sets:
                event_id_to_content[s]['event_oid'] = s
                item['corefs'].append(event_id_to_content[s])
            item['corefs'] = sorted(item['corefs'], key=lambda x: len(x['equals']), reverse=True)
            represent_item = item['corefs'][0]
            item.update(self.getEventSlots(represent_item))
            new_events.append(item)
        for e_oid, e_content in event_id_to_content.items():
            if e_oid not in coref_id_map:
                e_content['event_oid'] = e_oid
                item = {'corefs': [e_content]}
                item.update(self.getEventSlots(e_content))
                new_events.append(item)
        return {'graph-{}'.format(i + 1): item for i, item in enumerate(new_events)}

    def _relation_combine(self, event_key_to_content, relation_list, sub_event_key):
        # relations = json.dumps(relation_list, ensure_ascii=True)
        id_map = {}
        for event_id, content in event_key_to_content.items():
            for event_item in content[sub_event_key]:
                id_map[event_item['event_oid']] = event_id
                # relations = relations.replace('"' + event_item['event_oid'] + '"', '"' + event_id + '"')
        for rel in relation_list:
            rel['head'] = id_map[rel['head']]
            rel['tail'] = id_map[rel['tail']]
        return relation_list
        # return json.loads(relations)

    def _relation_to_map(self, relation_list):
        relation_map = {}
        for rel in relation_list:
            if rel['head'] not in relation_map:
                relation_map[rel['head']] = []
            if rel['tail'] not in relation_map:
                relation_map[rel['tail']] = []
            relation_map[rel['head']].append(
                {'direction': 'out', 'type': rel['type'], 'count': rel['count'], 'opposite': rel['tail']})
            relation_map[rel['tail']].append(
                {'direction': 'in', 'type': rel['type'], 'count': rel['count'], 'opposite': rel['head']})
        return relation_map

    def _relation_count(self, relation_list):
        nrelations = {}

        def rel2id(rel):
            return json.dumps((rel['type'], rel['head'], rel['tail']))

        for rel in relation_list:
            relid = rel2id(rel)
            if 'count' not in rel:
                rel['count'] = 1
            if relid not in nrelations:
                nrelations[relid] = rel
            else:
                nrelations[relid]['count'] += rel['count']
        return [v for k, v in nrelations.items()]

    def to_digraph(self, filename):
        relation_types = ['qa_cause', 'answer_cause', 'r_qa_cause', 'r_answer_cause',
                          # 'coreference',
                          'related', 'contrary', 'context']

        def get_text(e_id):
            return EventGraph.get_text(self.event_id_to_content[e_id])

        relation_map = self.relation_eid_to_content
        relation_map = {k: sorted(v, key=lambda x: -x['count']) for k, v in relation_map.items()}
        birelation_map = {}

        for e_id, rels in relation_map.items():
            relation_info = (get_text(e_id), [{'type': r['type'], 'opposite': get_text(r['opposite']),
                                               'count': r['count'], 'direction': r['direction']} for r in rels])
            for rel in rels:
                etype = rel['type']
                if e_id == rel['opposite']:
                    continue
                if etype in ['contrary', 'related', 'context']:  # 'coreference',
                    for key in [(e_id, etype, rel['opposite']), (rel['opposite'], etype, e_id)]:
                        if key not in birelation_map:
                            birelation_map[key] = rel['count']
                        else:
                            if birelation_map[key] < rel['count']:
                                birelation_map[key] = rel['count']
                else:
                    key = (e_id, etype, rel['opposite'])
                    if rel['direction'] == 'out':
                        if key not in birelation_map:
                            birelation_map[key] = rel['count']
                        else:
                            print(rel)

        exists_key = set()
        nrelation_map = {k: [] for k, v in relation_map.items()}
        for k, v in birelation_map.items():
            if k[1] in ['contrary', 'related', 'context']:  # 'coreference',
                if k not in exists_key:
                    nrelation_map[k[0]].append({'opposite': k[2], 'count': v, 'direction': 'in', 'type': k[1]})
                    nrelation_map[k[0]].append({'opposite': k[2], 'count': v, 'direction': 'out', 'type': k[1]})
                    nrelation_map[k[2]].append({'opposite': k[0], 'count': v, 'direction': 'in', 'type': k[1]})
                    nrelation_map[k[2]].append({'opposite': k[0], 'count': v, 'direction': 'out', 'type': k[1]})
                    exists_key.add(k)
                    exists_key.add((k[2], k[1], k[0]))
            else:
                nrelation_map[k[0]].append({'opposite': k[2], 'count': v, 'direction': 'out', 'type': k[1]})
                nrelation_map[k[2]].append({'opposite': k[0], 'count': v, 'direction': 'in', 'type': k[1]})
                nrelation_map[k[0]].append({'opposite': k[2], 'count': v, 'direction': 'in', 'type': 'r_' + k[1]})
                nrelation_map[k[2]].append({'opposite': k[0], 'count': v, 'direction': 'out', 'type': 'r_' + k[1]})
        event_id_to_content = self.get_output_events()

        relations = sorted([(event_id_to_content[k[0]], k[1], event_id_to_content[k[2]], v)
                            for k, v in birelation_map.items() if v > 10], key=lambda x: x[3], reverse=True)
        # most 449

        with open(filename, 'w', encoding='UTF-8') as file:
            out_data = {'events': event_id_to_content, 'relations': nrelation_map}
            json.dump(out_data, file, ensure_ascii=False, indent=2)

    def _event_combine(self, event_id_to_content, slots2key):
        events = {}
        for event_oid, content in event_id_to_content.items():
            event_slots = self.getEventSlots(content)
            key = slots2key(event_slots)
            if key not in events:
                events[key] = event_slots
                events[key]['equals'] = []
            events[key]['equals'].append({'event_oid': event_oid,
                                          'span': content['span'],
                                          'text_id': content['text_id'],
                                          'from_question': content['from_question']})
        combined_events = {}
        for i, item in enumerate(events.items()):
            content = item[1]
            key = 'undup-{}'.format(i)
            span_count = {}
            for eq in content['equals']:
                span = eq['span']
                if span not in span_count:
                    span_count[span] = 0
                span_count[span] += 1
            content['span'] = sorted([(k, v) for k, v in span_count.items()], key=lambda x: x[1], reverse=True)[0][0]
            combined_events[key] = content
        return {'undup-{}'.format(i + 1): item[1] for i, item in enumerate(events.items())}

    def _read_data(self, filename):
        line_data = [json.loads(line) for line in open(filename, 'r', encoding='UTF-8')]
        events = {}
        relations = []
        texts = {}
        for item in line_data:
            texts[item['id']] = {
                'question': item['question'],
                'answer': item['answer']
            }
            all_events = item['question_events'] + item['answer_events']
            for i, event in enumerate(all_events):
                is_question = i < len(item['question_events'])
                text = item['question'] if is_question else item['answer']

                def getText(e, k):
                    t = None
                    if k in e and e[k]:
                        span = e[k]
                        t = text[span[0]:span[1]]
                    return t

                events[event['id']] = {
                    'type': event['type'],
                    'span': text[event['start']:event['end']],
                    'concept': getText(event, 'concept'),
                    'modifier': getText(event, 'modifier'),
                    'predicate': getText(event, 'predicate'),
                    'direction': getText(event, 'direction'),
                    'text_id': item['id'],
                    'from_question': is_question
                }

            relations += [{'type': 'context', 'head': e['id'], 'tail': e_['id']}
                          for i, e in enumerate(all_events) for e_ in all_events[i + 1:]]
            casual_rels = [rel for rel in item['relations'] if rel['type'] != 'answer_and']
            union_rels = [rel for rel in item['relations'] if rel['type'] == 'answer_and']
            union_sets = []
            for rel in union_rels:
                head = rel['head']
                tail = rel['tail']
                for union_set in union_sets:
                    if head in union_set or tail in union_set:
                        union_set.add(head)
                        union_set.add(tail)
                        break
                else:
                    union_sets.append({head, tail})
            for union_set in union_sets:
                casuals = []
                for event_id in union_set:
                    for rel in casual_rels:
                        if event_id == rel['head']:
                            casuals.append((0, rel))
                        elif event_id == rel['tail']:
                            casuals.append((1, rel))
                for rel in casuals:
                    for event_id in union_set:
                        if rel[0] == 0 and rel[1]['head'] != event_id:
                            casual_rels.append({'type': rel[1]['type'], 'head': event_id, 'tail': rel[1]['tail']})
                        elif rel[0] == 1 and rel[1]['tail'] != event_id:
                            casual_rels.append({'type': rel[1]['type'], 'head': rel[1]['head'], 'tail': event_id})
            relations += casual_rels
        return events, relations, texts
