import argparse
import json
import os
from itertools import islice
from random import Random
from time import time

from tqdm import tqdm

from splitters.answer_number import AnswerNumberSplitter
from splitters.answer_splitter import AnswerSplitter
from splitters.complex_quantifier_scope import HasComplexQuantifierScopeSplitter
from splitters.complex_sub_graph import ComplexSubGraphSplitter
from splitters.intersect_splits import IntersectSplits
from splitters.lexical import LexicalSplitter
from splitters.number import HasNumberSplitter
from splitters.operator_splitter import OperatorSplitter
from splitters.pattern_splitter import PatternSplitter
from splitters.program import ProgramSplitter
from splitters.same_color import SameColorSplitter
from splitters.specific_number import HasSpecificNumberSplitter
from splitters.union_splits import UnionSplits


def get_splitters():
    return [
        # UnionSplits(name="has_num_3_ans_3",
        #             split1=AnswerSplitter(name="answer_3", answers=[3]),
        #             split2=HasSpecificNumberSplitter(3)),
        # UnionSplits(name="has_num_2_ans_2",
        #             split1=AnswerSplitter(name="answer_2", answers=[2]),
        #             split2=HasSpecificNumberSplitter(2)),
        # PatternSplitter(name="answer_attribute", patterns_names=['query_attr']),
        # AnswerNumberSplitter(),
        PatternSplitter(name="tpl_verify_quantifier_attribute",
                        patterns_names=['verify_count_ref', 'verify_count_ref_group_by']),
        # OperatorSplitter(name="has_count", operators=["count"]),
        # AnswerSplitter(name="verify", answers=[True, False]),
        # OperatorSplitter(name="has_quantifier", operators=["all", "all_same", "some", "none"]),
        # OperatorSplitter(name="has_quantifier_all", operators=["all", "all_same"]),
        # HasComplexQuantifierScopeSplitter(),
        # ComplexSubGraphSplitter(),
        # PatternSplitter(name="has_compare", patterns_names=['compare_count']),
        # PatternSplitter(name="tpl_verify_count", patterns_names=['verify_count_ref']),
        # PatternSplitter(name="tpl_choose_attr", patterns_names=['choose_attr']),
        # PatternSplitter(name="tpl_choose_object", patterns_names=['choose_object']),
        # IntersectSplits(name="has_compare_more",
        #                 split1=OperatorSplitter(name=None, operators=["gt"]),
        #                 split2=PatternSplitter(name=None, patterns_names=['compare_count'])),
        # OperatorSplitter(name="has_logic", operators=["logic_and", "logic_or"]),
        # OperatorSplitter(name="has_logic_and", operators=["logic_and"]),
        # OperatorSplitter(name="has_group_by", operators=["group_by_images"]),
        # OperatorSplitter(name="operator_geq", operators=["geq"]),
        # OperatorSplitter(name="has_attribute", operators=["filter"]),
        # PatternSplitter(name="tpl_verify_attribute", patterns_names=['verify_attr']),
        # SameColorSplitter(),
        # HasNumberSplitter(),
        # HasSpecificNumberSplitter(3),
        # ProgramSplitter(seed=1),
        # ProgramSplitter(seed=2),
        # ProgramSplitter(seed=3),
        # LexicalSplitter(seed=1),
        # LexicalSplitter(seed=2),
        # LexicalSplitter(seed=3),
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input-ds-path', default="../experiments/data/")
    parser.add_argument('--output-ds-path', default="../experiments/data/updated_dataset")
    parser.add_argument('--limit', type=int, default=None)

    args = parser.parse_args()

    rand = Random(0)

    splits_dir = os.path.join(args.output_ds_path, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    st = time()

    input_file_modes = ["train", "val"]
    types = ["iid", "comp"]
    splitters = get_splitters()

    # first load the dataset
    lines = {}
    for file_mode in input_file_modes:
        file_name = os.path.join(args.input_ds_path, f"{file_mode}.jsonl")
        with open(file_name, "rt") as in_file:
            print(f"Loading {file_name}")
            lines[file_mode] = [json.loads(line) for line in islice(tqdm(in_file), args.limit)]

    # some splitters randomly assign different features to train/test and thus need to see the entire dataset before
    # the actual splitting
    for splitter in splitters:
        splitter.fit(*lines.values())

    # start splitting
    for file_mode in input_file_modes:
        for splitter in splitters:
            split_output = splitter.split_lines(lines[file_mode])
            train_test_ids_tuple, train_test_debug_tuple = split_output[0:2], split_output[2:4]

            for i, split_type in enumerate(types):
                ids = train_test_ids_tuple[i]

                output_file_name = f"{splitter.name}_{file_mode}_{split_type}.json"
                output_val_file_path = os.path.join(splits_dir, output_file_name)
                print(f"Saving {len(ids)} samples in {output_file_name}")

                with open(output_val_file_path, "wt") as out_file:
                    json.dump({
                        "comp_split_name": splitter.name,
                        "split_name": file_mode,
                        "ids": ids
                    }, out_file)

                if file_mode == "val":
                    output_file_name = f"{splitter.name}_{file_mode}_{split_type}_examples.json"
                    output_val_file_path = os.path.join(splits_dir, output_file_name)
                    with open(output_val_file_path, "wt") as out_file:
                        json.dump(train_test_debug_tuple[i], out_file, indent=4)
                print(f"Finished writing to {output_file_name}")

    et = time()
    print(f"Finished in {(et - st):.2f} seconds")

