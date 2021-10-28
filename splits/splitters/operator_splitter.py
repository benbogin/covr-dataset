from typing import List

from .splitter import Splitter


class OperatorSplitter(Splitter):
    def __init__(self, name, operators: List):
        self.name = name
        self._operators = operators

    def get_split(self, qst):
        program = qst['program']
        for r in program:
            if r['operation'] in self._operators:
                return "test"
        return "train"
