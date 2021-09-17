import numpy as np
import re
import os
import jieba

# https://github.com/pltrdy/rouge
from rouge import Rouge

rouge = Rouge()
MAX_LEN = 60
current_path = os.path.dirname(__file__)
stopwords = set([line.strip() for line in open(current_path + '/stopword', encoding='UTF-8')])
punctuation = [',', ';', '.', '，', '；', '。', '；', '？', '：', '、', '（', '）', '!', '！', '|']
punc = set(punctuation)


def removeStopwords(text):
    text = str(text)
    text = text.replace('\n', '')
    text = text.replace('\\n', '')
    text = text.replace('(', '（')
    text = text.replace(')', '）')
    text = text.replace('<q>', '')
    text = re.sub('\xa0+', '', text)
    text = re.sub('\u3000+', '', text)
    text = re.sub('\\s+', '', text)
    score_p = '[（][^（）]*\d+[^（）]*分[^）]*[）]'
    text = re.sub(score_p, '', text)
    sentence_depart = jieba.cut(text.strip())
    outstr = []
    for word in sentence_depart:
        if word in punc:
            # pass
            outstr.append(punctuation[0])
        elif word not in stopwords:
            if word != '\t':
                outstr.append(word)
    text = ''.join(outstr)
    # punctuation_p = '[,;.，；。；？：、（）!！|]'
    # text = re.sub(punctuation_p, '', text)
    text = re.sub(' +', '', text)
    return text, outstr


memory = {'f': {}, 'uf': {}}


def processText(rawText, filterd=True):
    if rawText in memory:
        return memory['f' if filterd else 'uf'][rawText]
    if filterd:
        text, _ = removeStopwords(rawText)
    else:
        text = rawText
    text = ''.join([w for w in text])
    memory['f' if filterd else 'uf'][rawText] = text
    return text


def evalAnswer(answer, gold, metrics=(('rouge-1', 'f'), ('rouge-2', 'f'), ('rouge-l', 'f')), max_len=MAX_LEN,
               filterd=True):
    zero_scores = {'rouge-1': {'r': 0, 'p': 0, 'f': 0},
                   'rouge-2': {'r': 0, 'p': 0, 'f': 0},
                   'rouge-l': {'r': 0, 'p': 0, 'f': 0}}
    if type(answer) is str:
        if len(answer) == 0:
            scores = zero_scores
        else:
            answer = processText(answer, filterd=filterd)[:max_len]
            gold = processText(gold, filterd=filterd)
            answer, gold = ' '.join(answer), ' '.join(gold)
            if len(answer) == 0 or len(gold) == 0:
                scores = zero_scores
            else:
                scores = rouge.get_scores(answer, gold)[0]
        if metrics is None:
            return scores
        else:
            scores_list = []
            for i, m in enumerate(metrics):  # ('rouge-1','f')
                scores_list.append(scores[m[0]][m[1]])
            return scores_list
    else:
        scores = [evalAnswer(a, g, metrics=metrics, max_len=max_len) for a, g in zip(answer, gold)]
        if metrics is None:
            return scores
        else:
            return np.mean(scores, axis=0).tolist()


def getScore(answerfile, goldfile, max_len=MAX_LEN):
    metrics = [
        # ("rouge-1", 'r'),
        # ("rouge-1", 'p'),
        ("rouge-1", 'f'),
        # ("rouge-2", 'r'),
        # ("rouge-2", 'p'),
        ("rouge-2", 'f'),
        # ("rouge-l", 'r'),
        # ("rouge-l", 'p'),
        ("rouge-l", 'f')]
    scoreAll = [[] for _ in range(len(metrics))]
    with open(answerfile, "r", encoding='utf-8') as f_answer:
        answers = f_answer.readlines()
        with open(goldfile, "r", encoding='utf-8') as f_gold:
            golds = f_gold.readlines()
            for idx, (answer, gold) in enumerate(zip(answers, golds)):
                # print(idx)
                answer = answer.strip()
                gold = gold.strip()
                # print(answer)
                if len(answer) != 0:
                    try:
                        score = evalAnswer(answer, gold, metrics=metrics, max_len=max_len)
                        for i in range(len(metrics)):
                            scoreAll[i].append(score[i])
                    except:
                        for i in range(len(scoreAll)):
                            scoreAll[i].append(0)
                else:
                    for i in range(len(scoreAll)):
                        scoreAll[i].append(0)

    # print("nan score")
    # scoreAvg = np.nanmean(scoreAll, axis=1)
    # for metric, score in zip(metrics, scoreAvg):
    #     print(metric, score)
    # print("score")
    scoreAvg = np.mean(scoreAll, axis=1)
    for metric, score in zip(metrics, scoreAvg):
        print(metric, score)
    return scoreAvg, scoreAll


import scipy.stats as stats


def ttest(score1, score2):
    print(stats.ttest_rel(score1, score2))


if __name__ == '__main__':
    root_path = '../../out/text/'
    for mode in ['_dev', '']:
        for eval_file in [
    #         # 'bl/multihop', 'bl/qa_hard_em', 'bl/kbqa_answer', 'bl/BM25', 'bl/ernie',
    #         'bl/Mass_45', 'bl/DeepNMT_45',
    #         'bl/PreSumm_30',
            'bl/bert_nmt',
    #         'best_45',
    #         # 'ab-oaware-woques', 'ab-gnn-worel', 'rgcn', 'gin', 'ab-edge-untyped', 'ab-ocsl-causalonly'
    #         # 'pa/l1', 'pa/l2', 'pa/l4', 'pa/l5', 'pa/n100', 'pa/n400', 'pa/ae10', 'pa/ae40'
        ]:
            print(mode, eval_file)
            print(getScore(f"{root_path}/{eval_file}{mode}.result", f"{root_path}/answer{mode}.txt", max_len=90))

        # _, score1 = getScore(f'{root_path}/best{mode}.result', f'{root_path}/answer{mode}.txt', max_len=60)
        #
        # for eval_file in [
        #     # 'bl/multihop',
        #     # 'bl/qa_hard_em', 'bl/kbqa_answer',
        #     # 'bl/BM25',
        #     # 'bl/ernie',
        #     # 'bl/Mass_15',
        #     # 'bl/DeepNMT_15',
        #     # 'bl/PreSumm_15',
        #     'bl/bert_nmt',
        #     # 'best',
        #     # 'ab-oaware-woques', 'ab-gnn-worel', 'rgcn', 'gin', 'ab-edge-untyped', 'ab-ocsl-causalonly',
        #     # 'pa/l1', 'pa/l2', 'pa/l4', 'pa/l5', 'pa/n100', 'pa/n400', 'pa/ae10', 'pa/ae40',
        #     # 'pa/ae30',
        #
        # ]:
        #     _, score2 = getScore(f'{root_path}/{eval_file}{mode}.result', f'{root_path}/answer{mode}.txt', max_len=60)
        #     print(eval_file, len(score1[0]), len(score2[0]))
        #     if len(score1[0]) != len(score2[0]):
        #         continue
        #     for i in range(3):
        #         ttest(score1[i], score2[i])
