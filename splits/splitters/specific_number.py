from .splitter import Splitter


class HasSpecificNumberSplitter(Splitter):
    name = "has_num_3"

    def __init__(self, leave_out_number):
        self._leave_out_number = leave_out_number

    def get_split(self, qst):
        program = qst['program']
        for r in program:
            for argument in r.get('arguments', []):
                if argument == self._leave_out_number:
                    return "test"
        return "train"
