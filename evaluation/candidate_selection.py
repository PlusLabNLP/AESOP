import numpy as np
import pandas as pd
import argparse
import time
import os
import nltk
from nltk.tokenize import word_tokenize
import editdistance
import rouge
from bleu import *
from nltk.translate.bleu_score import corpus_bleu

def slice_11(my_list, n):
    composite_list = [my_list[x:x+n] for x in range(0, len(my_list),n)]
    return composite_list


def bleu_scorer(ref, hyp, script='default'):
    refsend = []
    for i in range(len(ref)):
        refsi = []
        for j in range(len(ref[i])):
            refsi.append(ref[i][j].split())
        refsend.append(refsi)

    gensend = []
    for i in range(len(hyp)):
        gensend.append(hyp[i].split())

    if script == 'nltk':
        metrics = corpus_bleu(refsend, gensend)
        return [metrics]

    metrics = compute_bleu(refsend, gensend)
    return metrics

rouge_eval = rouge.Rouge(metrics=['rouge-1', 'rouge-2', 'rouge-l'])

def select_posed_bleu(src, df_sub):
    poseds = []
    for idx in list(df_sub.index):
        syn = df_sub.loc[idx, 'syn_paraphrase']
        temp = df_sub.loc[idx, 'template']
        syn_tags = list(zip(*nltk.pos_tag(word_tokenize(syn))))[1]
        temp_tags = list(zip(*nltk.pos_tag(word_tokenize(temp))))[1]
        posed = editdistance.eval(syn_tags, temp_tags)
        poseds.append(posed)

    min_posed = min(poseds)
    posed_idx = [i for i in range(len(poseds)) if poseds[i] == min_posed]
    max_bleu = -1
    final_idx = None
    id_start = list(df_sub.index)[0]
    for idx in posed_idx:
        syn = df_sub.loc[id_start + idx, 'syn_paraphrase']
        bleu = bleu_scorer([[src]], [syn])[0]
        if bleu > max_bleu:
            max_bleu = bleu
            final_idx = id_start + idx

    return final_idx

def select_rouge(src, df_sub):
    max_rouge = -1
    max_idx = None
    for idx in list(df_sub.index):
        syn = df_sub.loc[idx, 'syn_paraphrase']
        rouge = rouge_eval.get_scores([syn], [src])[0]['rouge-1']['f']
        if rouge > max_rouge:
            max_rouge = rouge
            max_idx = idx
    return max_idx

def ranker_select_rouge(src, df_sub):
    max_rouge = -1
    max_idx = None
    for idx in list(df_sub.index):
        syn = df_sub.loc[idx, 'syn_paraphrase']
        rouge1 = rouge_eval.get_scores([syn], [src])[0]['rouge-1']['f']
        rouge2 = rouge_eval.get_scores([syn], [src])[0]['rouge-2']['f']
        rougel = rouge_eval.get_scores([syn], [src])[0]['rouge-l']['f']
        rouge_general = 0.2 * rouge1 + 0.3 * rouge2 + 0.5 * rougel
        if rouge_general > max_rouge:
            max_rouge = rouge_general
            max_idx = idx
    return max_idx

def select_bleu(src, df_sub):
    max_bleu = -1
    max_idx = None
    for idx in list(df_sub.index):
        syn = df_sub.loc[idx, 'syn_paraphrase']
        bleu = bleu_scorer([[src]], [syn])[1][0]
        if bleu > max_bleu:
            max_bleu = bleu
            max_idx = idx
    return max_idx

def select_maxht(df_sub):
    max_ht = -1
    max_idx = None
    for idx in list(df_sub.index):
        ht = int(df_sub.loc[idx, 'height'])
        if ht > max_ht:
            max_ht = ht
            max_idx = idx

    return max_idx

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Convert trees file to sentence file')
    parser.add_argument('-mode', default = 'test', help = '')
    parser.add_argument('-gen_dir', help = ' ', default="./")
    parser.add_argument('-output_file', help="the name of the output_file")
    # parser.add_argument('-clean_gen_file', required = True, help = 'name of the file')
    # parser.add_argument('-res_file', required = True, help = 'name of the file')
    parser.add_argument('-crt', choices = ['posed','rouge', 'bleu', 'maxht', 'rouge-general'],
                        default ='bleu',
                        help = "Criteria to select best generation")
    parser.add_argument('-sample', type=int, default=10)
    parser.add_argument('-scbart_generate',  help="the file scbart generated", default="output/template-based-diverse-wr.txt")
    parser.add_argument('-target', help="the target file", default="eval_data/template-based3-set1.target")
    args = parser.parse_args()

    generate_lines = open(args.scbart_generate, "r").readlines()
    target_lines = open(args.target,"r").readlines()
    target_lines = [line.split("<sep>")[0].strip() for line in target_lines]
    assert len(generate_lines) == len(target_lines)
    df_ls = []
    for i in range(0, len(generate_lines)):
        generate = generate_lines[i]
        target = target_lines[i]
        df_ls.append({
            "source": target,
            "syn_paraphrase": generate
        })
    df = pd.DataFrame(df_ls)

    # df = pd.read_csv(os.path.join(args.gen_dir, args.clean_gen_file))
    srcs_unq = []
    idss = []
    ids = []
    prev_src = None
    prev_temp = None
    it = 0

    srcs_unq = [ls[0].strip("\n") for ls in slice_11(df["source"].values, args.sample)]
    idss = slice_11(df["source"].index, args.sample)

    assert len(idss) == len(srcs_unq)
    elites = []
    for src, ids in zip(srcs_unq, idss):
        df_sub = df.loc[ids]

        if args.crt == 'posed':
            final_idx = select_posed_bleu(src, df_sub)
        elif args.crt == 'bleu':
            final_idx = select_bleu(src, df_sub)
        elif args.crt == 'maxht':
            final_idx = select_maxht(df_sub)
        elif args.crt == 'rouge-general':
            final_idx = ranker_select_rouge(src, df_sub)
        else:
            final_idx = select_rouge(src, df_sub)
        elites.append(final_idx)

    df_elite = df[df.index.isin(elites)]

    assert len(df_elite) == len(srcs_unq)
    try:
        references = df_elite['reference'].values
    except:
        references = []
    syn_paras = df_elite['syn_paraphrase'].values
    sources = df_elite['source'].values

    # para_f, source_f = open(os.path.join(args.gen_dir, 'para.txt'), "w+"), \
    #                    open(os.path.join(args.gen_dir, 'source.txt'), "w+")
    # para_f = open(os.path.join(args.gen_dir, 'QQPPos-para.txt'), "w+")
    para_f = open(os.path.join(args.gen_dir, args.output_file), "w+")
    for i, row in df_elite.iterrows():
        syn_para, source = row["syn_paraphrase"].strip("\n").strip(), row["source"].strip("\n").strip()
        para_f.write(syn_para + "\n")
        # source_f.write(source + "\n")







