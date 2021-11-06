# AESOP

Here is the code base for **AESOP: Paraphrase Generation with Adaptive Syntactic Control** by [Jiao Sun](https://sunjiao123sun.github.io/), [Xuezhe Ma](https://xuezhemax.github.io/) and [Nanyun Peng](https://vnpeng.net/), this work is accepted by EMNLP 2021.

Please consider citing our work if you find either our code or data useful.

```
@inproceedings{sun2021aesop,
  title = {AESOP: Paraphrase Generation with Adaptive Syntactic Control},
  author = {Sun, Jiao and Ma, Xuezhe and Peng, Nanyun},
  booktitle = {The 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year = {2021}
}
```

The code base is actively maintained, please reach out to jiaosun@usc.edu if you encounter any issues or raise an issue here! We would like to thank authors of [SGCP](https://arxiv.org/pdf/2005.08417.pdf) and huggingface library. Part of the evaluation script is adopted from the [SGCP repository](https://github.com/malllabiisc/SGCP), and AESOP is implemented using [huggingface](https://github.com/huggingface/).

## Dependencies

```
pip install -r requirement.txt
```

Please download all the required tools, data (our preprocessed data) and software (e.g., stanford CoreNLP) from https://drive.google.com/file/d/1MP9k48BuBCdAPhWXjfuq7Cl9b9ZIdAPB/view?usp=sharing

 Unzipping this zip file should give you 

1. `evaluation`: contains the script, software and dependencies that are necessary for evaluating the model performance. 
2. `pretrained-models`: contain pretrained models to replicate our results, `h` means height: h2 means trimming the parse tree at height 2. Please see Figure 2 in our paper as an example
3. `raw-data`: contains data for two datasets: QQP-Pos and ParaNMT-50M (small), we use the same split from [SGCP](https://github.com/malllabiisc/SGCP)
4. `processed-data`: after pre-processing the raw data to feed into huggingface transformer

These unzipped files should be put directly under AESOP main directory.

 The two settings we will be introducing will only differ during the inference time 



## Proprocessing

AESOP has two modes:

- when the taget syntactic parses are vailable from crowd-sourced exemplar sentences, we extract the syntactic parses from the exemplar sentences and use them as target syntactic parse to guide the generation (set `use_template` as Y)
- when the exemplar sentences are not available, we use the retrieval-based  selection strategy to adaptively determine a set of target syntactic parses  (set `use_template` as Y)

```
python data-processing.py --input_dir raw-data/QQPPos --output_dir processed-data/QQPPos-hf-refine --use_template Y

python data-processing.py --input_dir raw-data/QQPPos --output_dir processed-data/QQPPos-hf-refine --use_template N
```

this will generate all the necessary files we need for both datasets, please see `processed-data` to see what to expect, and move different directorys to proper locations if needed



## Table1: target syntactic parse from exemplar sentences

1. **load pretrained model and do the inference**: please fill in `[output_file_path_...]` based on your own developmet environment

   QQPPos

   ```shell
   python run_eval.py pretrained-models/qqppos-h4 processed-data/QQPPos-hf-refine/exemplar/level5/test.source [output_file_path_qqppos]
   ```

   ParaNMT

   ```shell
   python run_eval.py pretrained-models/paranmt-h4 processed-data/ParaNMT50-hf-refine/exemplar/level5/test.source [output_file_path_paranmt]
   ```

   📝 this should give you two files with lines of {target syntactic parse} \<sep\> {paraphrase}

2. we use a simple rule to **extract generated paraphrases**

   ```
   python extract_sentence.py --input_file [output_file_path_qqppos/paranmt]
   ```

   📝 this should give you two files with {paraphrases}, and they should be the same as the ones in `aesop-result/tab1-paranmt/qqppos-h4.txt`

3. last step to **get the evaluation metrics** as shown in our paper's Table 1

   - QQPPos

   ```shell
   python -m evaluation.eval -r raw-data/QQPPos/test/ref.txt -t raw-data/QQPPos/test/tgt.txt -i aesop-result/tab1-qqppos-h4.txt
   ```

   - ParaNMT

   ```shell
   python -m evaluation.eval -r raw-data/QQPPos/test/ref.txt -t raw-data/ParaNMT50m/test/tgt.txt -i aesop-result/tab1-paranmt-h4.txt
   ```

4. If you want to train those two models from scratch, please use these 

   - QQPPos

     ```shell
     python finetune_trainer.py --data_dir processed-data/QQPPos-hf-refine/exemplar/level5 --learning_rate 3e-5 --warmup_steps 500  --num_train_epochs 25  --output_dir [output_dir] --max_source_length 512 --max_target_length 128 --do_train --overwrite_output --model_name_or_path facebook/bart-base --gradient_accumulation_steps 32 --save_total_limit 2
     ```

   - ParaNMT

     ```shell
     python finetune_trainer.py --data_dir processed-data/ParaNMT50-hf-refine/exemplar/level5 --learning_rate 3e-5 --warmup_steps 500  --num_train_epochs 25  --output_dir [output_dir] --max_source_length 512 --max_target_length 128 --do_train --overwrite_output --model_name_or_path facebook/bart-base --gradient_accumulation_steps 32 --save_total_limit 2
     ```

     

## Table 2: adaptive syntactic parse selection

1. replicate the result when the ground-truth is not available -- AESOP generates multiple paraphrases by adaptively selecting target syntactic parses, `processed-data/QQPPos-hf-refine/diverse/level3.source ` is the file that AESOP generates at the time we tested. When you run the preprocessing file, it might gives you a different file because of the randomness introduced by the sampling strategy. In the following, you may see 

   QQPPos

   ```
   python run_eval.py pretrained-models/qqppos-h2 processed-data/QQPPos-hf-refine/diverse/level3.source diverse-qqppos.txt --fp16
   ```

   ParaNMT

   ```
   python run_eval.py pretrained-models/paranmt-h2 processed-data/ParaNMT50-hf-refine/diverse/level3.source diverse-paranmt.txt --fp16
   ```

   📝 "processed-data/ParaNMT50-hf-refine/diverse/level3.source"

   in the following, we will use the ones for QQPPos as an illustration, and the one for ParaNMT will be just replacing the path for QQPPos with the one for paranmt.

2. as before, extract the paraphrases from the model output

   ```
   python extract_sentence.py --input_file diverse-qqppos.txt
   ```

   📝  It should give you a file called `diverse-qqppos_extract.txt`, which should look 

3. This file contains 10 instances per example. We choose one among those using `ROUGE` scores in our work, you can also choose other available metrics

   ```
   python candidate_selection.py -gen_dir ./ -scbart_generate diverse-qqppos_extract.txt  -target processed-data/QQPPos-parse-hf-refine/diverse/level3.target -output_file diverse-l3-select
   ```

4. after the selection, the file should look exactly the same as `aesop-result/table2-qqppos-h2.txt`

5. Then you can get the metrics except TED@2 reported in Table 2 after running the evaluation script. Please note that TED@2 cannot be acuqired from the that testing script because it is looking at the selected paraphrase and the ground-truth paraphrase, but  selected paraphrases are from our retrieved target syntactic parses, and it is defined as an average value between all target syntactic parses and generated paraphrases. 

   ```
   python ted2.py -i diverse-qqppos_extract.txt -s diverse-l3-select -t processed-data/QQPPos-hf-refine/diverse/level3.source
   ```



## Use AESOP as a paraphrase tool in your project

If you are looking for a paraphrasing tool to generate paraphrases with diverse syntactic parses based on your input sentence only, please give AESOP a try!

```shell
# first parse the input sentences and generate the necessary file for running the model
python demo.py --output_dir demo_output
# run the generation model
python run_eval.py pretrained-models/paranmt-h2 demo_output/level3_paranmt.source demo_output/level3_result.txt --fp16
# extract sentences from the model output
python extract_sentence.py --input demo_output/level3_result.txt
```

