from collections import Counter
from itertools import combinations
from random import Random
from typing import List, Dict

from .splitter import Splitter


class LexicalSplitter(Splitter):
    def __init__(
            self,
            seed: int = 0,
            n_held_out_pairs: int = 65,
            min_seen_alone_on_training: int = 50,
            min_seen_together_on_validation: int = 15,
    ):
        self._random = Random(seed)
        self.name = f"lexical_{seed}"

        self._n_held_out_pairs = n_held_out_pairs
        self._min_seen_alone_on_training = min_seen_alone_on_training
        self._min_seen_together_on_validation = min_seen_together_on_validation

        self._unseen_pairs = set()

    def fit(self, *lines: List[List[Dict]]):
        questions_per_sub_graph = {i: Counter() for i in range(len(lines))}
        all_pairs = set()
        for split_i, lns in enumerate(lines):
            for line in lns:
                lexical_terms = self._get_lexical_terms_set(line['program'], line['answer'])
                for elm in lexical_terms:
                    questions_per_sub_graph[split_i][elm] += 1
                for pair in combinations(lexical_terms, r=2):
                    all_pairs.add(pair)
                    questions_per_sub_graph[split_i][pair] += 1

        # create list of candidates that appear above certain # appearance threshold
        pairs_candidates = set()
        for pair in all_pairs:
            train_i = 0
            val_i = 1
            n_seen_together_train = questions_per_sub_graph[train_i][pair]
            n_seen_together_val = questions_per_sub_graph[val_i][pair]
            n_seen_elm1_alone_train = questions_per_sub_graph[train_i][pair[0]] - n_seen_together_train
            n_seen_elm2_alone_train = questions_per_sub_graph[train_i][pair[1]] - n_seen_together_train
            if n_seen_elm1_alone_train < self._min_seen_alone_on_training:
                continue
            if n_seen_elm2_alone_train < self._min_seen_alone_on_training:
                continue
            if n_seen_together_val < self._min_seen_together_on_validation:
                continue
            pairs_candidates.add(pair)

        sel_pairs = self._random.sample(list(pairs_candidates), k=self._n_held_out_pairs)
        self._unseen_pairs = set(sel_pairs)

    def get_split(self, qst):
        lexical_terms = self._get_lexical_terms_set(qst['program'], qst['answer'])
        for pair in combinations(lexical_terms, r=2):
            if pair in self._unseen_pairs:
                return "test"
        return "train"

    @staticmethod
    def _get_lexical_terms_set(program, answer):
        def clean_lexical_term(term_or_terms):
            terms = [term_or_terms] if not type(term_or_terms) is list else term_or_terms
            return [term.split(":_")[1] if ":_" in term else term for term in terms]

        output = set()
        for operator in program:
            # add terms from operators with a lexical argument(s)
            if (operator['operation'] in ["find", "with_relation", "filter"] or
                    operator['operation'].startswith("choose_") or
                    operator['operation'].startswith("verify_")):
                output.update(clean_lexical_term(operator['arguments']))
            # if one of the argument is a program, add arguments from program
            elif operator.get('arguments') and type(operator.get('arguments')[0]) is list:
                output.update(LexicalSplitter._get_lexical_terms_set(operator.get('arguments')[0], answer))

        # add answer if it is a lexical term
        if program[-1]['operation'].startswith("query_"):
            output.update(clean_lexical_term(answer))
        return output
