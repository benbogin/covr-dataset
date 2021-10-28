from abc import ABC, abstractmethod
from typing import List, Dict

from tqdm import tqdm


class Splitter(ABC):
    name = None

    @abstractmethod
    def get_split(self, qst: Dict) -> str:
        """
        Return `train`/`test` according to the compositional split
        """
        raise NotImplementedError()

    def fit(self, *lines: List[List[Dict]]) -> None:
        """
        If necessary for the splitting, goes over all data (each argument here is a list of examples, e.g. train/val,
        which will normally be flattened).
        """
        pass

    def split_lines(self, lines: List[Dict]):
        splits = {'train': [], 'test': []}  # holds only ids
        debug_output = {'train': [], 'test': []}  # holds entire question, only for debug purposes

        for qst in tqdm(lines):
            qid = qst['qid']

            comp_split = self.get_split(qst)
            splits[comp_split].append(qid)

            # just for debug, if we don't have enough already
            if len(splits[comp_split]) < 100:
                if 'original_id' in qst:
                    del qst['original_id']
                debug_output[comp_split].append(qst)

        splits["train"].sort()
        splits["test"].sort()
        return splits["train"], splits["test"], debug_output["train"], debug_output["test"]

    @staticmethod
    def get_flattened_program(qst, add_program=True, add_predicate_program=True):
        """
        puts all program rows in same list, including nested quantifier nested operators
        """
        flattened_program_rows = []
        if add_program:
            flattened_program_rows = list(qst['program'])
        if add_predicate_program:
            for r in qst['program']:
                if r['operation'] in ["all", "some", "none"]:
                    predicate_program = r['arguments'][0]
                    flattened_program_rows += predicate_program
        return flattened_program_rows
