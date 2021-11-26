import numpy as np
import re
import os
import jieba

# https://github.com/pltrdy/rouge
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction

smooth = SmoothingFunction()
rouge = Rouge()
MAX_LEN = 60
current_path = os.path.dirname(__file__)
# print(current_path)
stopwords = set([line.strip() for line in open(current_path+'/stopword', encoding='UTF-8')])
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


def evalAnswer(answer, gold, metrics=(('rouge-1', 'f'), ('rouge-2', 'f'), ('rouge-l', 'f'),'BLEU'), max_len=MAX_LEN,
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
            # print(gold)
            # print(answer)
            if len(answer) == 0 or len(gold) == 0:
                scores = zero_scores
            else:
                scores = rouge.get_scores(answer, gold)[0]
            if 'BLEU' in metrics:
                scores_bleu=sentence_bleu([gold], answer, weights=(0,0, 0,1), smoothing_function=smooth.method1)

                # print(answer)
                # print(gold)
        if metrics is None:
            return scores
        else:
            scores_list = []
            for i, m in enumerate(metrics):  # ('rouge-1','f')
                if m=='BLEU':
                    scores_list.append(scores_bleu)
                else:
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
        ("rouge-l", 'f'),
        'BLEU',
    ]
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
                    except Exception as e:
                        print(e.with_traceback())
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

def eval_huamn_performance(humanfiles,goldfile,max_len):
    answer_lists=[]
    for humanfile in humanfiles:
        with open(humanfile,'r',encoding='utf-8') as f:
            lines=f.readlines()
            answers=[]
            for line in lines:
                answers.append(line.strip())
            answer_lists.append(answers)
    golds=[]
    with open(goldfile,'r',encoding='utf-8') as f:
        for line in f:
            qa=line.split('\t')
            if len(qa)!=2:
                print(qa)
            else:
                golds.append(qa[-1])
    metrics = [
        # ("rouge-1", 'r'),
        # ("rouge-1", 'p'),
        ("rouge-1", 'f'),
        # ("rouge-2", 'r'),
        # ("rouge-2", 'p'),
        ("rouge-2", 'f'),
        # ("rouge-l", 'r'),
        # ("rouge-l", 'p'),
        ("rouge-l", 'f'),
        'BLEU',
    ]
    scoreAll = [[] for _ in range(len(metrics))]
    for answers in answer_lists:
        score = evalAnswer(answers, golds, metrics=metrics, max_len=max_len)
        print(score)
        for i in range(len(metrics)):
            scoreAll[i].append(score[i])
    scoreAvg = np.mean(scoreAll, axis=1)
    for metric, score in zip(metrics, scoreAvg):
        print(metric, score)

if __name__ == '__main__':
    # eval_huamn_performance(["../../../eval/human1.txt",
    #                         "../../../eval/human2.txt",
    #                         "../../../eval/human3.txt"],
    #                        "../../../Data/Human performance/Gold-standard answers.txt",max_len=4000)
    eval_huamn_performance(["../../../Data/Human performance/Human-1.txt","../../../Data/Human performance/Human-2.txt"],
                           "../../../Data/Human performance/Gold-standard answers.txt",max_len=4000)
    # root_path = '../../../ceqa_test_results'
    # for max_len in [60]:
    #     scores_list = {"dev": [], "test": []}
    #     eval_files = [
    #         'BM25',
    #         'ernie',
    #         'Mass',
    #         'DeepNMT',
    #         'PreSumm',
    #         'bert_nmt',
    #         'mbart',
    #         'multihop',
    #         'qa_hard_em',
    #         'kbqa_answer',
    #         'PreSummKB',
    #         'best',
    #         # 'ab-oaware-woques', 'ab-gnn-worel', 'rgcn', 'gin', 'ab-edge-untyped', 'ab-ocsl-causalonly',
    #         # 'pa/l1', 'pa/l2', 'pa/l4', 'pa/l5', 'pa/n100', 'pa/n300', 'pa/ae10', 'pa/ae40',
    #         # 'pa/ae30',
    #
    #     ]
    #     metrics = [
    #         # ("rouge-1", 'r'),
    #         # ("rouge-1", 'p'),
    #         ("rouge-1", 'f'),
    #         # ("rouge-2", 'r'),
    #         # ("rouge-2", 'p'),
    #         ("rouge-2", 'f'),
    #         # ("rouge-l", 'r'),
    #         # ("rouge-l", 'p'),
    #         ("rouge-l", 'f'),
    #         'BLEU'
    #     ]
    #     for eval_file in eval_files:
    #         for mode in ['']:
    #             print(mode, eval_file)
    #             _, score2 = getScore(f'{root_path}/{eval_file}{mode}.result', f'{root_path}/answer{mode}.txt',
    #                                  max_len=max_len)
    #             if mode == "_dev":
    #                 scores_list["dev"].append(score2)
    #             else:
    #                 scores_list["test"].append(score2)
    #     for mode in ['dev', 'test']:
    #         for i in range(len(eval_files) - 1):
    #             print(eval_files[i])
    #             for idx,(score1,score2) in enumerate(zip(scores_list[mode][i], scores_list[mode][-1])):
    #                 print(metrics[idx])
    #                 ttest(score1, score2)
    #             # print(eval_file, len(score1[0]), len(score2[0]))
    #             # if len(score1[0]) != len(score2[0]):
    #             #     continue
    #             # for i in range(3):
    #             #     ttest(score1[i], score2[i])
