import argparse
import json
import os
from collections import defaultdict
from random import Random

from tqdm import tqdm


comp_splits_keys = [
    "answer_attribute",
    "answer_number",
    "has_attribute",
    "has_compare",
    "has_compare_more",
    "has_complex_quantifier_scope",
    "has_count",
    "has_group_by",
    "has_logic",
    "has_logic_and",
    "has_num_3",
    "has_num_3_ans_3",
    "has_num_2_ans_2",
    "has_quantifier",
    "has_quantifier_all",
    "has_same_attribute_color",
    "lexical_1",
    "lexical_2",
    "lexical_3",
    "has_number",
    "program_1",
    "program_2",
    "program_3",
    "rm_v_c",
    "tpl_choose_object",
    "tpl_verify_attribute",
    "tpl_verify_count_union_tpl_verify_count_group_by",
    "tpl_verify_quantifier_attribute",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--original-ds-path', default="../experiments/data/")
    parser.add_argument('--updated-ds-path', default="../experiments/data/updated_dataset")
    parser.add_argument('--limit', type=int, default=None)

    args = parser.parse_args()

    random = Random(0)

    for split in ['train', 'val']:
        input_file = open(os.path.join(args.original_ds_path, f'{split}.jsonl'))
        output_file = open(os.path.join(args.updated_ds_path, f'{split}.jsonl'), 'wt')

        comp_split_per_qid = defaultdict(set)
        for comp_split in comp_splits_keys:
            input_comp_split_info = json.load(
                open(os.path.join(args.updated_ds_path, "splits", f'{comp_split}_{split}_comp.json')))
            for qid in input_comp_split_info['ids']:
                comp_split_per_qid[qid].add(comp_split)

        lines = list(input_file)
        random.shuffle(lines)

        for line in tqdm(lines):
            ex = json.loads(line)
            ex['properties'] = list(comp_split_per_qid[ex['qid']])
            output_file.write(json.dumps(ex) + '\n')
