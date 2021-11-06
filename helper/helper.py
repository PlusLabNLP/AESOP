

from nltk.tree import Tree

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


def trim_tree(root, height):
    if isinstance(root, str):
        return root
    if height < 1:
        return
    all_child_state = []
    # adding itself
    all_child_state.extend(root[0])
    for child in root[1:]:
        if trim_tree(child, height - 1):
            all_child_state.append(trim_tree(child, height - 1))
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

def to_tuple(lst):
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)

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

if __name__ == '__main__':
    parse_str = "(ROOT (S (VP (ADVP (RB suddenly)) (FW i) (VP (VBP 've) (VP (VBN gone) (PP (IN from) (S (VP (VBG trying) (S (VP (TO to) (VP (VB find) (PRT (RP out)) (SBAR (IN if) (S (NP (NN anyone)) (VP (MD might) (VP (VB have) (NP (PRP it)) (ADVP (IN in)) (PP (IN for) (NP (NP (NNS marcos)) (PP (TO to) (S (VP (VBG wondering) (SBAR (IN if) (S (NP (EX there)) (VP (VBZ 's) (NP (NP (NN anyone)) (PP (IN at) (NP (NN school))) (SBAR (WHNP (WP who)) (S (VP (VBZ does) (RB n't)))))))))))))))))))))))))) (. .)))"
    tree = Tree.fromstring(parse_str)
    a = remove_leaves_from_tree(tree)
    print(clean_tuple_str(to_tuple(trim_tree(a, 3))))
