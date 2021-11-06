import argparse
import sys
from helper.utils import *
import tempfile
import pandas as pd
import os
import multiprocessing
import math
from collections import Counter
import itertools



parser = argparse.ArgumentParser()
# for generating the corresponding src file used for bart generation model
parser.add_argument("--input_dir", "-i",  help="the input directory", type=str)
parser.add_argument("--output_dir", "-o", help="the output directory", type=str)
parser.add_argument("--use_template", choices=["Y", "N"], help="if we are using template as the syntactic signal")
args = parser.parse_args()

def generate_non_trim_version(o_dir, src_parses, tgt_pure_parses, src_lines, tgt_lines, signal,
                              exemplar):
    '''
    :param o_dir: output directory for the entire setting
    :param src_parses: source parses
    :param tgt_pure_parses: target parses
    :param src_lines: source sentences
    :param tgt_lines: target sentences
    :param signal: train/dev/test
    :param exemplar: the choices are [exemplar, non-exemplar]
    :return:
    '''
    direc_path = f"{o_dir}/{exemplar}/no-trim"
    if not os.path.exists(direc_path):
        os.makedirs(direc_path)
    output_source = open(f"{direc_path}/{signal}.source", "w+")
    output_tgt = open(f"{direc_path}/{signal}.target", "w+")
    for i in range(0, len(src_parses)):
        # <src-sent>, <src-full-parse>, <tgt-pure-parse> -> <tgt-sent>
        output_source.write(f"{src_lines[i]}<sep>{src_parses[i]}<sep>{tgt_pure_parses[i]}\n")
        output_tgt.write(f"{tgt_lines[i]}\n")

def generate_trim_version(o_dir, src_parses, tgt_pure_parses, src_lines, tgt_lines, signal, height,
                          exemplar):
    direc_path = f"{o_dir}/{exemplar}/level{height}"
    if not os.path.exists(direc_path):
        os.makedirs(f"{direc_path}")
    output_source = open(f"{direc_path}/{signal}.source", "w+")
    output_tgt = open(f"{direc_path}/{signal}.target", "w+")
    src_trim, tgt_trim = [], []
    for i in range(0, len(src_parses)):
        # <src-sent>, <src-full-parse>, <tgt-pure-parse> -> <tgt-sent>
        trim_tgt = trim_str(tgt_pure_parses[i], height)
        trim_src = trim_str(src_parses[i], height)
        src_trim.append(trim_src)
        tgt_trim.append(trim_tgt)
        output_source.write(f"{src_lines[i]}<sep>{src_parses[i]}<sep>{trim_tgt}\n")
        output_tgt.write(f"{tgt_lines[i]}\n")
    return src_trim, tgt_trim

def generate_tgt_parse(arguments):
    result = []
    level_, freq, src_lines, level = arguments
    for i in range(0, len(src_lines)):
        possible_drawn = step2_rouge(level_, freq, src_lines[i], level)[3]
        for possible in possible_drawn:
            # output_file.write(f"{src_lines[i]}<sep>{src_parses[i]}<sep>{possible}\n")
            result.append(f"{src_lines[i]}<sep>{src_parses[i]}<sep>{possible}\n")
    return result


if __name__ == '__main__':
    input_dir = args.input_dir
    for signal in ["train", "test", "val"]:
        print("signal: ", signal)
        if signal != "test":
            src, tgt = f"{input_dir}/{signal}/src.txt", f"{input_dir}/{signal}/tgt.txt"
        else:
            src, tgt = f"{input_dir}/{signal}/src.txt", f"{input_dir}/{signal}/ref.txt"
        spe = stanford_parsetree_extractor()
        src_pure_parses, src_parses = spe.run(src)
        tgt_pure_parses, tgt_parses = spe.run(tgt)
        src_lines, tgt_lines = [line.strip("\n") for line in open(src, "r").readlines()], \
                               [line.strip("\n") for line in open(tgt, "r").readlines()]
        if args.use_template == "N":
            generate_non_trim_version(args.output_dir, src_parses, tgt_pure_parses, src_lines,
                                      tgt_lines, signal, "non-exemplar")
            for level in range(3, 11):
                src_trim, tgt_trim = generate_trim_version(args.output_dir, src_parses,
                                                       tgt_pure_parses, src_lines,
                                      tgt_lines, signal, level, "non-exemplar")
                if signal != "test":
                    # write the statistics of the <src parse, tgt parse> combination
                    path = f"{args.output_dir}/repe_statistics"
                    if not os.path.exists(path):
                        os.makedirs(path)
                    frequency_file = open(f"{path}/repe_para_{level}.txt", "w+")
                    frequency_dict = Counter(map(tuple, map(sorted, list(zip(src_trim, tgt_trim)))))
                    for key, value in frequency_dict.items():
                        if value >= 1:
                            print(f"{key}\t{value}\n")
                            frequency_file.write(f"{key}\t{value}\n")

                elif signal == "test":
                    print("write diverse source file")
                    # generate the future target parses from the frequencies list
                    path = f"{args.output_dir}/repe_statistics"
                    if not os.path.exists(f"{path}/diverse"):
                        os.makedirs(f"{path}/diverse")
                    output_file = open(f"{path}/diverse/level{level}.source", "w+")
                    frequency_lines = open(f"{path}/repe_para_{level}.txt", "r").readlines()
                    level_, freq = generate_dict(frequency_lines), generate_counts_dict(frequency_lines)

                    if multiprocessing.cpu_count() < len(src_lines):
                        num_processes = multiprocessing.cpu_count()
                    else:
                        num_processes = len(src_lines)
                    print("num_processes: ", num_processes)
                    chunk_size = int(len(src_lines) / num_processes)
                    result = []
                    chunks_src = [src_lines[i:i + chunk_size] for i in range(0, len(src_lines), chunk_size)]
                    pool = multiprocessing.Pool(processes=num_processes)
                    result.extend(pool.map(generate_tgt_parse, zip([level_] * num_processes, [freq] * num_processes, chunks_src, [level] * num_processes)))
                    for line in list(itertools.chain(*result)):
                        output_file.write(line)


        if args.use_template == "Y":
            # need to access the exemplar dataset to get the syntax info
            exemplar = f"{input_dir}/{signal}/tgt.txt"
            exemplar_pure_pareses, _ = spe.run(exemplar)
            generate_non_trim_version(args.output_dir, src_parses, exemplar_pure_pareses,
                                      src_lines, tgt_lines, signal, "exemplar")
            for level in range(3, 11):
                generate_trim_version(args.output_dir, src_parses, exemplar_pure_pareses,
                                      src_lines, tgt_lines, signal, level, "exemplar")












