import sys
import numpy as np
import scipy.spatial.distance as distance
from Graph import EventGraph

if len(sys.argv) > 2 and sys.argv[2].endswith('hop'):
    pass

else:
    if sys.argv[0] == "":
        import fasttext

        ft = fasttext.load_model(f'{sys.argv[1]}/cc.zh.300.bin')
    else:
        import fasttext.util

        # fasttext.util.download_model('zh', if_exists='ignore')
        ft = fasttext.load_model('cc.zh.300.bin')

    memory = {}
    ft_words = set(ft.words)
    import time

    start_time = time.time()

    times = [0, 0]


def event_similarity(e_i, e_j):
    def get_grams(event, k=3):
        grams = []
        span_text = EventGraph.get_text(event)
        for i in range(1, k + 1):
            for j in range(len(span_text) - i + 1):
                grams.append(span_text[j:j + i])
        return grams

    def get_embedding(event):
        id_ = event['id'] if 'id' in event else event['event_id']
        if id_ in memory:
            return memory[id_]
        grams = [g for g in get_grams(event) if g in ft_words]
        grams = [ft.get_word_vector(g) for g in grams]
        if len(grams) > 0:
            embedding = np.mean(grams, axis=0)
        else:
            embedding = None
        memory[id_] = embedding
        return embedding

    e_i_embedding = get_embedding(e_i)
    e_j_embedding = get_embedding(e_j)
    if e_i_embedding is None or e_j_embedding is None:
        return 0
    similarity = 1 - distance.cosine(e_i_embedding, e_j_embedding)
    # print(get_text(e_i), get_text(e_j), similarity)
    return similarity
