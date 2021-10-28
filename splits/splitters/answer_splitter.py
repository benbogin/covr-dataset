from typing import List

from .splitter import Splitter


class AnswerSplitter(Splitter):
    def __init__(self, name, answers: List):
        self.name = name
        self._answers = answers

    def get_split(self, qst):
        if str(qst['answer']) in [str(a) for a in self._answers]:
            return "test"
        return "train"
