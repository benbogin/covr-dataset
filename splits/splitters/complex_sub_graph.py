from .splitter import Splitter


class ComplexSubGraphSplitter(Splitter):
    name = "rm_v_c"

    def __init__(self, min_chain_length: int = 2):
        self._min_chain_length = min_chain_length

    def get_split(self, qst):
        flattened_program_rows = self.get_flattened_program(qst)
        for row_index, row in enumerate(flattened_program_rows):
            chain_length = self._get_max_chain_length(flattened_program_rows, row_index)
            if chain_length >= self._min_chain_length:
                return "test"
        return "train"

    def _get_max_chain_length(self, program, row_index, chain_length: int = 0):
        row = program[row_index]

        if row['operation'] not in ["with_relation", "with_relation_object"]:
            return 0

        dependencies = row['dependencies']
        max_length = 1
        for dependency in dependencies:
            max_length = max(self._get_max_chain_length(program, dependency, chain_length) + 1, max_length)

        return max_length

