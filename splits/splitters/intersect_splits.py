from .splitter import Splitter


class IntersectSplits(Splitter):
    def __init__(self, name: str, split1, split2):
        self.name = name
        self.split1 = split1
        self.split2 = split2

    def get_split(self, qst):
        if self.split1.get_split(qst) == "test" and self.split2.get_split(qst) == "test":
            return "test"
        return "train"

