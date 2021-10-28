from .splitter import Splitter


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


class AnswerNumberSplitter(Splitter):
    name = "answer_number"

    def get_split(self, qst):
        if type(qst['answer']) is bool:
            return "train"
        if is_int(qst['answer']):
            return "test"
        return "train"
