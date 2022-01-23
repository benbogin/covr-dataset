from dataclasses import dataclass
from typing import Optional

from generator.queries.data_classes import QueryNode


@dataclass
class SubGraph:
    root: QueryNode
    multi_count: Optional[int] = None

    others_identical_exist: Optional[bool] = None
    others_identical_exist_prob: Optional[float] = None

    _cached_positions = None
    _cached_depth = None

    def _fill_cached_positions_if_needed(self):
        def iterate(current_item=None):
            if current_item is None:
                current_item = self.root

            yield current_item

            if current_item.relations:
                for relation in current_item.relations:
                    yield relation
                    yield from iterate(relation.target)
                    if relation.prepositions:
                        for pp in relation.prepositions:
                            yield pp
                            yield pp.target

        if not self._cached_positions:
            self._cached_positions = list(iterate())

    def depth(self):
        def _depth(current_item=None):
            max_depth = 1
            if current_item.relations:
                for relation in current_item.relations:
                    max_depth = max(_depth(relation.target) + 2, max_depth)
                    if relation.prepositions:
                        max_depth = max(4, max_depth)
            return max_depth

        if not self._cached_depth:
            self._cached_depth = _depth(self.root)
        return self._cached_depth

    def get_leaf_triplets(self, include_prepositions: bool = False, output=None, indices_output=None, current_item=None,
                          current_triplet=None, current_indices_triplet=None, state=None):
        if output is None:
            output = []
            indices_output = []
            state = {'pos': -1}
        if not current_item:
            current_item = self.root
        state['pos'] += 1
        if current_triplet and not current_item.relations:
            output.append(tuple((*current_triplet, current_item)))
            indices_output.append(tuple((*current_indices_triplet, state['pos'])))

        current_triplet = [current_item]
        current_indices_triplet = [state['pos']]
        if current_item.relations:
            for relation in current_item.relations:
                state['pos'] += 1
                current_triplet.append(relation)
                current_indices_triplet.append(state['pos'])
                self.get_leaf_triplets(include_prepositions, output, indices_output, relation.target, current_triplet,
                                       current_indices_triplet, state)
                current_triplet = current_triplet[:-1]
                current_indices_triplet = current_indices_triplet[:-1]

                if relation.prepositions:
                    for pp in relation.prepositions:
                        state['pos'] += 1

                        if include_prepositions:
                            current_triplet.append(pp)
                            current_indices_triplet.append(state['pos'])
                            self.get_leaf_triplets(include_prepositions, output, indices_output, pp.target,
                                                   current_triplet, current_indices_triplet, state)

                            current_triplet = current_triplet[:-1]
                            current_indices_triplet = current_indices_triplet[:-1]

        return output, indices_output

    def __iter__(self):
        self._fill_cached_positions_if_needed()
        yield from self._cached_positions

    def __getitem__(self, index):
        self._fill_cached_positions_if_needed()
        return self._cached_positions[index]

    def __len__(self):
        self._fill_cached_positions_if_needed()
        return len(self._cached_positions)

    def to_dict(self):
        def _to_dict(current_item=None):
            output = {"name": current_item.name, "attributes": current_item.attributes}
            if current_item.relations:
                if current_item.relations:
                    output["relations"] = []
                for relation in current_item.relations:
                    output["relations"].append({"name": relation.name, "target": _to_dict(relation.target)})
                    if relation.prepositions:
                        output["relations"][-1]["prepositions"] = []
                        for pp in relation.prepositions:
                            output["relations"][-1]["prepositions"].append({
                                "name": pp.name,
                                "target": _to_dict(pp.target)
                            })
            return output
        return _to_dict(self.root)

    def serialize(self):
        serialized_elements = [elm.serialize() for elm in self]
        return tuple(serialized_elements)

