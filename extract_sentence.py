import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", help="the input file from bart output")


parser.add_argument("--keyword", default="<sep>")
args = parser.parse_args()

input_lines = open(args.input_file, "r").readlines()
# ref_lines = open(args.ref_file, "r").readlines()
keyword = args.keyword
input_file = args.input_file
if ".txt" in args.input_file:
    input_file = input_file.replace(".txt", "")
if keyword == "<sep>":
    output_file = open(input_file + f"_sep_extract", "w+")
    error_file = open(input_file + f"_sep_error", "w+")
else:
    output_file = open(input_file + f"_return_extract", "w+")
    error_file = open(input_file + f"_return_error", "w+")

count = 0

def deal_non_sep(string, keyword):
    if keyword in string: raise Exception("it has <sep>")
    else:
        if "ROOT" not in string:
            # did not try to generate the syntactic parse at all
            final_str = string
        else:
            last_para_count = 0
            # find the last element with "(" in it
            for i in range(0, len(string.split(" "))):
                item = string.split(" ")[i]
                if "(" in item or ")" in item:
                    last_para_count = i
            valid_tokens = string.split(" ")[last_para_count + 1:]
            final_str = " ".join(token for token in valid_tokens)
        if final_str == "":
            final_str = "."
        return final_str


for line in input_lines:
    line = line.strip("\n")
    if keyword in line:
        output_file.write(line.split(keyword)[1] + "\n")
    else:
        output_file.write(deal_non_sep(line, args.keyword) + "\n")
        error_file.write(str(count+1) + "\t" + line + "\n")
        error_file.write("extract: " + deal_non_sep(line, args.keyword) + "\n")
        # output_file.write("\n")

    count = count + 1

