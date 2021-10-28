from random import Random
from typing import List, Dict

from .splitter import Splitter
from .utils import program_to_string


class ProgramSplitter(Splitter):
    def __init__(
            self, seed: int = 0,
            train_percent: float = 0.8
    ):
        self._random = Random(seed)
        self.name = f"program_{seed}"

        self._train_percent = train_percent

        self.split_per_template = {}

    def fit(self, *lines: List[List[Dict]]):
        questions_per_program = set()
        for lns in lines:
            for line in lns:
                anonymized_program = program_to_string(line['program'], add_arguments=False)
                questions_per_program.add(anonymized_program)

        all_programs = list(questions_per_program)
        self._random.shuffle(all_programs)
        n_train = int(self._train_percent * len(all_programs))
        self.split_per_template = {
            'train': set(all_programs[:n_train]),
            'test': set(all_programs[n_train:])
        }

    def get_split(self, qst):
        program = qst['program']
        anonymized_program = program_to_string(program, add_arguments=False)
        if anonymized_program in self.split_per_template["train"]:
            return "train"
        elif anonymized_program in self.split_per_template["test"]:
            return "test"
        else:
            raise ValueError()
