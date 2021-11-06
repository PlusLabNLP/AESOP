import argparse
import rouge

from eval_utils import Meteor, stanford_parsetree_extractor, compute_tree_edit_distance
from tqdm import tqdm
import subprocess
from nltk.tree import Tree

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
                new_string = new_string[:start - 1] + ", "
            start = start + index + 1
    return new_string

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

def to_tuple(lst):
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)

def get_syntax_templates(template_file):
    parses = [test_str.split("<sep>")[-1].strip() for test_str in open(template_file).readlines()]
    parses = [clean_tuple_str(to_tuple(trim_tree_nltk(Tree.fromstring(parse_str), 3))) for
        parse_str in parses]
    return parses

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', '-i', type=str, help="full generated file, ")
parser.add_argument('--select_file', '-s', type=str)
parser.add_argument('--temp_file', '-t', type=str)
args = parser.parse_args()

n_select_line = len(list(open(args.select_file)))

input_lines = [line.strip("\n").strip() for line in open(args.input_file, "r").readlines()]
indices = []
for line in open(args.select_file, "r").readlines():
    new_line = line.strip("\n").strip()
    indices.append(input_lines.index(new_line))

temp_parses = ""
if "scpn" in args.input_file.lower():
    templates = [
        '( ROOT ( S ( NP ) ( VP ) ( . ) ) )',
        '( ROOT ( S ( VP ) ( . ) ) )',
        '( ROOT ( NP ( NP ) ( . ) ) )',
        '( ROOT ( FRAG ( SBAR ) ( . ) ) )',
        '( ROOT ( S ( S ) ( , ) ( CC ) ( S ) ( . ) ) )',
        '( ROOT ( S ( LST ) ( VP ) ( . ) ) )',
        '( ROOT ( SBARQ ( WHADVP ) ( SQ ) ( . ) ) )',
        '( ROOT ( S ( PP ) ( , ) ( NP ) ( VP ) ( . ) ) )',
        '( ROOT ( S ( ADVP ) ( NP ) ( VP ) ( . ) ) )',
        '( ROOT ( S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) )'
    ]
    if "qqpp" in args.input_file.lower():
        temp_parses = templates * 3000
    elif "paranmt" in args.input_file.lower():
        temp_parses = templates * 800
else:
    temp_parses = get_syntax_templates(args.temp_file)
    temp_parses = [clean_tuple_str(to_tuple(trim_tree_nltk(Tree.fromstring(parse_str), 3))) for
               parse_str in temp_parses]

if not isinstance(temp_parses, list):
    raise Exception("template parses are not a items of a list!")
temp_parses = [temp_parses[i] for i in indices]


print("#lines - select: {}, temp: {}".format(n_select_line, len(temp_parses)))
assert n_select_line == len(temp_parses), \
    "#select {} != #templates {}".format(n_select_line, temp_parses)

spe = stanford_parsetree_extractor()
select_parses = spe.run(args.select_file)
select_parses = [clean_tuple_str(to_tuple(trim_tree_nltk(Tree.fromstring(parse_str), 3))) for
              parse_str in select_parses]

spe.cleanup()

all_ted = []
all_ted_t = []

# Default F1_score
pbar = tqdm(zip(select_parses, temp_parses))

for select_parse, temp_parse in pbar:
    ted_t = compute_tree_edit_distance(select_parse, temp_parse)
    all_ted_t.append(ted_t)
    pbar.set_description(
        "ted-e: {:.3f}".format(
            sum(all_ted_t) / len(all_ted_t)
        ))

print("ted-e: {:.3f}".format(
    sum(all_ted_t) / len(all_ted_t)
))