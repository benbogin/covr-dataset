from .splitter import Splitter


class HasComplexQuantifierScopeSplitter(Splitter):
    name = "has_complex_quantifier_scope"

    def get_split(self, qst):
        program = qst['program']
        for r in program:
            if r['operation'] in ["all", "all_same", "some", "none"]:
                dependency = r['dependencies'][0]
                if program[dependency]['operation'] != "find":
                    return "test"
        return "train"
