from .splitter import Splitter


class HasNumberSplitter(Splitter):
    name = "has_number"

    def get_split(self, qst):
        program = qst['program']
        for r in program:
            for argument in r.get('arguments', []):
                if argument in list(range(0, 10)):
                    return "test"
        return "train"
