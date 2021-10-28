from .splitter import Splitter


class SameColorSplitter(Splitter):
    name = "has_same_attribute_color"

    def get_split(self, qst):
        if qst['pattern_name'] == 'quantifier_same_attr':
            if qst['program'][-1]['arguments'][0] == 'color':
                return "test"
        if qst['pattern_name'] == 'specific_same_attr':
            if qst['program'][-2]['arguments'][0] == 'color':
                return "test"

        return "train"
