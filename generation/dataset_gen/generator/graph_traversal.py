from collections import defaultdict
from itertools import combinations
from random import Random

import copy

from executor.executor import Executor
from generator.queries.query_builder import SubGraph
from generator.queries.data_classes import QueryNode, QueryRelationship
from generator.resources import Resources


class TraversalException(Exception):
    pass


class TraversalDidNotEndInUniqueObjectException(TraversalException):
    pass


class GraphTraversal:
    """
    This class goes over scene graphs and returns all relevant sub-graphs in it
    """
    def __init__(
            self,
            random_seed: int = None,
            max_depth: int = 2,
    ):
        self._executor = Executor()
        self._random_seed = random_seed
        self._random = Random(random_seed)

        self._max_depth = max_depth

        self._maximum_object_area = 400
        self._nouns_to_ignore = Resources.ignore['nouns']
        self._relations_to_ignore = Resources.ignore['relations']
        self._pairs_to_ignore = Resources.ignore['pairs']

    def traverse(self, starting_obj_id: str, scene: dict, seen_objects: set = None, depth: int = 0):
        obj = scene['objects'][starting_obj_id]

        if not seen_objects:
            seen_objects = {obj['name']}
        if depth == 0:
            # we want the random seed to be independent of process id (in the case of multiprocess runs)
            self._random.seed(str(starting_obj_id) + str(self._random_seed))

        if obj['name'] in self._nouns_to_ignore:
            return

        if 'w' in obj and obj['w'] * obj['h'] <= self._maximum_object_area:
            return

        query_root_node = QueryNode(name=obj['name'], attributes=set(), char_symbol=f"o_{starting_obj_id}")

        next_relation_nodes_candidates = []
        if depth < self._max_depth:
            # candidates for next node are all objects related to current object that we haven't traversed yet
            seen_next_rel_obj = set()
            for r in obj['relations']:
                other_obj_name = scene['objects'][r['object']]['name']
                rel_obj_key = r['name'] + '_' + other_obj_name
                if rel_obj_key in seen_next_rel_obj:
                    continue
                seen_next_rel_obj.add(rel_obj_key)
                if other_obj_name not in seen_objects and r['name'] not in self._relations_to_ignore:
                    next_relation_nodes_candidates.append(r)

        next_attributes_nodes_candidates = [None] + obj['attributes']

        next_nodes_by_object = defaultdict(list)

        for next_obj_rel in next_relation_nodes_candidates:
            # in this loop, we recursively attach all possible relations to the current node
            next_obj_id = next_obj_rel['object']

            if next_obj_id == starting_obj_id:
                continue

            rel_name = next_obj_rel['name']
            next_obj_name = scene['objects'][next_obj_id]['name']

            seen_objects.add(next_obj_name)

            for next_node in self.traverse(next_obj_id, scene, seen_objects, depth + 1):
                new_sub_graph = copy.deepcopy(query_root_node)
                if not new_sub_graph.relations:
                    new_sub_graph.relations = []
                new_relation = QueryRelationship(name=rel_name, char_symbol=f'r_{starting_obj_id}_{next_obj_id}',
                                                 source=new_sub_graph, target=next_node.root,
                                                 dataset_source='imsitu' if '_' in obj['scene_id'] else 'gqa')
                new_sub_graph.relations.append(new_relation)

                next_nodes_by_object[next_obj_id].append(SubGraph(new_sub_graph))
                if next_obj_rel.get('prepositions'):
                    # imsitu preposition
                    new_relation.prepositions = []
                    for pp in next_obj_rel.get('prepositions'):
                        next_pp_node = QueryNode(pp['object'], name=scene['objects'][pp['object']]['name'],
                                                 attributes=set())
                        pp_rel = QueryRelationship(name=pp['name'], char_symbol=f'r_{starting_obj_id}_{pp["object"]}',
                                          source=new_relation, target=next_pp_node, dataset_source='imsitu')
                        new_relation.prepositions.append(pp_rel)
                        pp_rel.backward_relation = new_relation
                        next_pp_node.backward_relation = pp_rel

                next_node.root.backward_relation = new_relation

                yield from self.get_node_outputs(next_attributes_nodes_candidates, new_sub_graph)

        for combination in combinations(next_nodes_by_object.values(), 2):
            # in this loop, we combine two children into to the root, to create a structure of kind A->B, A->C

            new_sub_graph = copy.deepcopy(query_root_node)
            if not new_sub_graph.relations:
                new_sub_graph.relations = []

            for nodes_list in combination:
                nodes_list = [g for g in nodes_list if len(g) == 3]

                added_graph = copy.deepcopy(self._random.choice(nodes_list))
                new_sub_graph.relations.append(
                    added_graph[1]
                )

            if new_sub_graph.relations[0].target.name == new_sub_graph.relations[1].target.name:
                continue

            yield from self.get_node_outputs(next_attributes_nodes_candidates, new_sub_graph)

        for output in self.get_node_outputs(next_attributes_nodes_candidates, query_root_node):
            yield output

    def get_node_outputs(self, possible_attributes, query):
        for attribute in possible_attributes:
            next_query = query
            if attribute is not None:
                next_query = copy.deepcopy(query)
                next_query.attributes.add(attribute)

            yield SubGraph(next_query)
