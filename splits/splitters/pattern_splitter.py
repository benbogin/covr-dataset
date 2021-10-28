from typing import List

from .splitter import Splitter


class PatternSplitter(Splitter):
    def __init__(self, name, patterns_names: List):
        self.name = name
        self._patterns_names = patterns_names

    def get_split(self, qst):
        if qst['pattern_name'] in self._patterns_names:
            return "test"
        return "train"
