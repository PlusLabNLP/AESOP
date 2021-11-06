import pandas as pd
from nltk.tree import Tree
import os
import re
import subprocess
import threading
import tempfile
import codecs
import ast
from numpy.random import choice
from rouge_score import rouge_scorer
import itertools

def deleaf(parse_string):
    tree = Tree.fromstring(parse_string.strip(), read_leaf=lambda s: "")
    for sub in tree.subtrees():
        for n, child in enumerate(sub):
            if isinstance(child, str):
                continue
            if len(list(child.subtrees(filter=lambda x: x.label() == '-NONE-'))) == len(child.leaves()):
                del sub[n]
    oneline = tree.pformat(margin=10000, parens=[" ( ", " ) "])
    oneline = re.sub(' +', ' ', oneline)
    return oneline

def convert_str(string):
    new_list= []
    for ele in string.split(" "):
        if ")" in ele:
            new_list.append(str(re.sub(r'^.*?\)', ')', ele)))
        else:
            new_list.append(ele)
    new_str = " ".join(ele for ele in new_list)
    return new_str

def trim_tree_nltk(root, height):
    try:
        root.label()
    except AttributeError:
        return

    if height < 1:
        return
    all_child_state = []
    #     print(root.label())
    all_child_state.append(root.label())

    if len(root) >= 1:
        for child_index in range(len(root)):
            child = root[child_index]
            if trim_tree_nltk(child, height - 1):
                all_child_state.append(trim_tree_nltk(child, height - 1))
    #                 print(all_child_state)
    return all_child_state


# extract parses from corenlp output
def extract_parses(fname):
    f = codecs.getreader('utf-8')(open(fname, 'rb'))

    count = 0
    sentences = []
    data = {'tokens':[], 'pos':[], 'parse':'', 'deps':[]}
    for idx, line in enumerate(f):
        if line.startswith('Sentence #'):
            new_sent = True
            new_pos = False
            new_parse = False
            new_deps = False
            if idx == 0:
                continue

            # label_sentence(data)
            # print ' '.join(data['tokens'])
            # data['label'] = dataset[count]['label']
            sentences.append(data)
            count += 1

            data = {'tokens':[], 'pos':[], 'parse':'', 'deps':[]}

        # read original sentence
        elif new_sent:
            # data['sent'] = line.strip()
            new_sent = False
            new_pos = True

        # read POS tags
        elif new_pos and line.startswith('[Text='):
            line = line.strip().split()
            w = line[0].split('[Text=')[-1]
            pos = line[-1].split('PartOfSpeech=')[-1][:-1]
            data['tokens'].append(w)
            data['pos'].append(pos)

        # start reading const parses
        elif (new_pos or new_parse) and line.strip() != '':
            new_pos = False
            new_parse = True
            data['parse'] += ' '+line.strip()
            data['pure_parse'] = convert_str(data['parse'])

        # start reading deps
        elif line.strip() == '':
            new_parse = False
            new_deps = True

        elif new_deps and line.strip() != '':
            line = line.strip()[:-1].split('(',1)
            rel = line[0]
            x1, x2 = line[1].split(', ')
            x1 = x1.replace("'", "")
            x2 = x2.replace("'", "")
            x1 = int(x1.rsplit('-', 1)[-1])
            x2 = int(x2.rsplit('-', 1)[-1])
            data['deps'].append((rel, x1 - 1, x2 - 1))

        else:
            new_deps = False

    # add last sentence
    # label_sentence(data)
    # data['label'] = dataset[count]['label']
    sentences.append(data)

    f.close()

    return sentences

STANFORD_CORENLP = '../evaluation/apps/stanford-corenlp-full-2018-10-05'
class stanford_parsetree_extractor:
    def __init__(self):
        self.stanford_corenlp_path = os.path.join(STANFORD_CORENLP, "*")
        print("standford corenlp path:", self.stanford_corenlp_path)
        self.output_dir = tempfile.TemporaryDirectory()
        self.cmd = ['java', '-cp', self.stanford_corenlp_path,
                    '-Xmx40g', 'edu.stanford.nlp.pipeline.StanfordCoreNLP',
                    '-parse.model', 'edu/stanford/nlp/models/srparser/englishSR.ser.gz',
                    '-annotators', 'tokenize,ssplit,pos,parse',
                    '-ssplit.eolonly', '-outputFormat', 'text',
                    '-outputDirectory', self.output_dir.name,
                    '-file', None]

    def run(self, file):
        print("parsing file:", file)
        self.cmd[-1] = file
        out = subprocess.run(
            self.cmd,
            cwd=os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        print(out)
        parsed_file = \
            os.path.join(
                self.output_dir.name,
                os.path.split(file)[1] + ".out")
        return [e['pure_parse'] for e in
                extract_parses(parsed_file)], [e['parse'] for e in extract_parses(parsed_file)]

    def cleanup(self):
        self.output_dir.cleanup()

def remove_leaves_from_tree(root):
    # if we get to the leaf nodes, then return
    try:
        root.label()
    except AttributeError:
        return
    all_child_state = []
    all_child_state.append([root.label()])
    for child in root:
        if remove_leaves_from_tree(child):
            all_child_state.append(remove_leaves_from_tree(child))
    return all_child_state

def clean_tuple_str(tuple_str):
    new_str_ls = []
    if len(tuple_str) == 1:
        new_str_ls.append(tuple_str[0])
    else:
        for i in str(tuple_str).split(", "):
            if i.count("'") == 2:
                new_str_ls.append(i.replace("'", ""))
            elif i.count("'") == 1:
                new_str_ls.append(i.replace("\"", ""))
    str_join = ' '.join(ele for ele in new_str_ls)
    return string_comma(str_join)

def prune_tree(parse_string, height):
    parse_tree = Tree.fromstring(parse_string)
    non_leaf_tree = remove_leaves_from_tree(parse_tree)
    final_str = clean_tuple_str(to_tuple(trim_tree(non_leaf_tree, height)))
    return final_str

def rouge_score(string, ls):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_result = [scorer.score(string, ls[i]) for i in range(len(ls))]
    rouge1 = [i["rouge1"].fmeasure for i in rouge_result]
    rouge2 = [i["rouge2"].fmeasure for i in rouge_result]
    rougeL = [i["rougeL"].fmeasure for i in rouge_result]
    return rouge1, rouge2, rougeL

def trim_tree(root, height):
    if isinstance(root, str):
        return root
    if height < 1:
        return

    all_child_state = []
    all_child_state.extend(root[0])

    for child in root[1:]:
        if trim_tree(child, height - 1):
            all_child_state.append(trim_tree(child, height - 1))
    return all_child_state

def string_comma(string):
    start = 0
    new_string = ''
    while start < len(string):
        if string[start:].find(",") == -1:
            new_string += string[start:]
            break
        else:
            index = string[start:].find(",")
            if string[start - 2] != "(":
                new_string += string[start:start + index]
                new_string += " "
            else:
                new_string = new_string[:start-1] +", "
            start = start + index + 1
    return new_string

def to_tuple(lst):
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)


def trim_str(string, height):
    return clean_tuple_str(to_tuple(trim_tree_nltk(Tree.fromstring(string), height)))



def generate_dict(lines):
    result = {}
    for line in lines:
        line = line.strip("\n").split("\t")
        tuple_str = ast.literal_eval(line[0])
        if tuple_str[0] in result:
            result[tuple_str[0]].append(tuple_str[1])
        else:
            result[tuple_str[0]] = [tuple_str[1]]

        if tuple_str[1] in result:
            result[tuple_str[1]].append(tuple_str[0])
        else:
            result[tuple_str[1]] = [tuple_str[0]]
    return result


def generate_counts_dict(lines):
    result = {}
    for line in lines:
        line = line.strip("\n").split("\t")
        tuple_str = ast.literal_eval(line[0])
        if tuple_str[0] in result:
            result[tuple_str[0]].append(int(line[1]))
        else:
            result[tuple_str[0]] = [int(line[1])]

        if tuple_str[1] in result:
            result[tuple_str[1]].append(int(line[1]))
        else:
            result[tuple_str[1]] = [int(line[1])]
    return result

def pick_n_parses_freqs(level, freq, res1, n):
    # helper function to step2_rouge()
    all_parses = list(level.keys())
    return_result = []
    for i in res1:
        candidate_ls = level[all_parses[i]]
        freq_ls = freq[all_parses[i]]
        prob_ls = [item / sum(freq_ls) for item in freq_ls]
        if n < len(candidate_ls):
            return_result.append(choice(candidate_ls, n, replace=False, p=prob_ls).tolist())
        else:
            return_result.append(candidate_ls)
    return list(itertools.chain.from_iterable(return_result))


def step2_rouge(level, freq, src_str, level_n, k_picks=5, n=2):
    """
    :param src_str: the source parse -- string
    :param level_n: level_n is the index of the level we are targeting at: (0, 5), (1, 4), (2, 3)
    :param k_picks: find k most similar parse strings
    :param n: pick n tgt template parse for each similar parse
    :return: the returned k_picks*n possible tgt parses based on rouge scores
    """
    # level = levels[level_n]
    # freq = freqs[level_n]
    all_parses = list(level.keys())
    rouge1, rouge2, rougeL = rouge_score(src_str, all_parses)
    res1_before_pick = sorted(range(len(rouge1)), key=lambda sub: rouge1[sub])
    res2_before_pick = sorted(range(len(rouge2)), key=lambda sub: rouge2[sub])
    resL_before_pick = sorted(range(len(rougeL)), key=lambda sub: rougeL[sub])
    res1, res2, resL = res1_before_pick[-k_picks:], res2_before_pick[-k_picks:], resL_before_pick[-k_picks:]

    w1, w2, w3 = 0.2, 0.3, 0.5
    weighted_res = [w1 * x + w2 * y + w3 * z for x, y, z in zip(res1_before_pick, res2_before_pick, resL_before_pick)]
    resW = sorted(range(len(weighted_res)), key=lambda sub: rougeL[sub])[-k_picks:]

    print("start parsing rouge 1")
    # parses_1 = list(itertools.chain.from_iterable([level[all_parses[i]][:n] for i in res1]))
    parses_1 = pick_n_parses_freqs(level, freq, res1, n)
    print("start parsing rouge 2")
    # parse_2 = list(itertools.chain.from_iterable([level[all_parses[i]][:n] for i in res2]))
    parses_2 = pick_n_parses_freqs(level, freq, res2, n)
    print("start parsing rouge L")
    # parse_L = list(itertools.chain.from_iterable([level[all_parses[i]][:n] for i in resL]))
    parses_L = pick_n_parses_freqs(level, freq, resL, n)
    print("start parsing rouge weighted")
    # parses_weighted = list(itertools.chain.from_iterable([level[all_parses[i]][:n] for i in
    #                                                    resW]))
    parses_weighted = pick_n_parses_freqs(level, freq, resW, n)
    return parses_1, parses_2, parses_L, parses_weighted

# add for rebuttal -- adding new experiments for always choose the most frequent target parses
def pick_n_parses_freqs_new(level, freq, res1, n):
    # helper function to step2_rouge()
    all_parses = list(level.keys())
    return_result = []
    for i in res1:
        # the candidate list that contains all the parses
        candidate_ls = level[all_parses[i]]
        freq_ls = freq[all_parses[i]]
        return_result.append([candidate_ls[freq_ls.index(max(freq_ls))]])
    return list(itertools.chain.from_iterable(return_result))

def step2_rouge_new(level, freq, src_str, level_n, k_picks=10, n=1):
    """
    :param src_str: the source parse -- string
    :param level_n: level_n is the index of the level we are targeting at: (0, 5), (1, 4), (2, 3)
    :param k_picks: find k most similar parse strings
    :param n: pick n tgt template parse for each similar parse
    :return: the returned k_picks*n possible tgt parses based on rouge scores
    """
    # level = levels[level_n]
    # freq = freqs[level_n]
    all_parses = list(level.keys())
    rouge1, rouge2, rougeL = rouge_score(src_str, all_parses)
    res1_before_pick = sorted(range(len(rouge1)), key=lambda sub: rouge1[sub])
    res2_before_pick = sorted(range(len(rouge2)), key=lambda sub: rouge2[sub])
    resL_before_pick = sorted(range(len(rougeL)), key=lambda sub: rougeL[sub])
    res1, res2, resL = res1_before_pick[-k_picks:], res2_before_pick[-k_picks:], resL_before_pick[-k_picks:]

    w1, w2, w3 = 0.2, 0.3, 0.5
    weighted_res = [w1 * x + w2 * y + w3 * z for x, y, z in zip(res1_before_pick, res2_before_pick, resL_before_pick)]
    resW = sorted(range(len(weighted_res)), key=lambda sub: weighted_res[sub])[-k_picks:]

    print("start parsing rouge 1")
    # parses_1 = list(itertools.chain.from_iterable([level[all_parses[i]][:n] for i in res1]))
    parses_1 = pick_n_parses_freqs(level, freq, res1, n)
    print("start parsing rouge 2")
    # parse_2 = list(itertools.chain.from_iterable([level[all_parses[i]][:n] for i in res2]))
    parses_2 = pick_n_parses_freqs(level, freq, res2, n)
    print("start parsing rouge L")
    # parse_L = list(itertools.chain.from_iterable([level[all_parses[i]][:n] for i in resL]))
    parses_L = pick_n_parses_freqs(level, freq, resL, n)
    print("start parsing rouge weighted")
    # parses_weighted = list(itertools.chain.from_iterable([level[all_parses[i]][:n] for i in
    #                                                    resW]))
    parses_weighted = pick_n_parses_freqs_new(level, freq, resW, n)
    return parses_1, parses_2, parses_L, parses_weighted

