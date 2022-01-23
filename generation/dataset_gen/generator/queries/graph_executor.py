import itertools
import json
import time
from collections import defaultdict
from copy import deepcopy
from random import Random
from typing import Tuple, Set, Dict, List, Union

from neo4j import GraphDatabase

from executor.executor import Executor
from generator.queries.data_classes import QueryNode

from generator.queries.query_builder import QueryBuilder
from generator.queries.sub_graph import SubGraph
from generator.redis import redis
from generator.utils import extract_elements_triplets, neo4j_results_to_sub_graph, sub_graph_root_to_ref_program


FilterOutSubGraphs = Union[SubGraph, List[SubGraph]]


class GraphExecutor:
    n_queries_executed_not_cached = 0
    n_queries_executed_cached = 0

    def __init__(self, split, scene_reader, limit_scenes_output: int = 10, random_seed=None, enable_cache=True,
                 perform_verification=False, output_verification_inference_data=False, verification_file_path=None):
        self._split = split
        self._graph_db = GraphDatabase.driver("neo4j://localhost:7687").session()
        self._random = Random(random_seed)
        self._scene_reader = scene_reader
        self._enable_cache = enable_cache
        self._executor = Executor()

        self._limit_scenes_output = limit_scenes_output

        self._pruning_cache = {}

    def _execute_query_str(self, query_str, sub_graph):
        result_scenes = []
        scenes_info = {}

        if len(sub_graph) == 1 and not sub_graph.root.attributes:
            # only needed for verification output
            random_scenes = self._random.sample(self._scene_reader.all_scenes_keys, 20)
            for scene_id in random_scenes:
                result_scenes.append(scene_id)
                scenes_info[scene_id] = {}
        else:
            if self._enable_cache:
                cached = redis.get(query_str)
            else:
                cached = None
            if cached:
                cached = json.loads(cached)
                result_scenes = cached['scenes']
                scenes_info = cached['scenes_info']
                GraphExecutor.n_queries_executed_cached += 1
            else:
                GraphExecutor.n_queries_executed_not_cached += 1
                # print("no cache!")
                st = time.time()
                results = self._graph_db.run(query_str)
                et = time.time()

                # if et - st > 0.5:
                #     print(query_str)
                #     print("*******************************")
                #     print(f"{(et - st):.2f}s")

                for result in results:
                    scene_id = result['scene.scene_id']
                    result_scenes.append(scene_id)
                    scenes_info[scene_id] = self.get_scene_info(result, sub_graph)

                redis.set(query_str, json.dumps({
                    'scenes': list(result_scenes),
                    'scenes_info': scenes_info
                }))

        return result_scenes, scenes_info

    @staticmethod
    def get_scene_info(result, original_sub_graph: SubGraph):
        result_sub_graph = neo4j_results_to_sub_graph(result)

        # keep only relevant attributes
        relevant_attributes_per_element = defaultdict(set)
        for node in original_sub_graph:
            if type(node) is QueryNode:
                node_name = node.name
                if type(node.name) is set:
                    if len(node.name) == 1:
                        node_name = list(node.name)[0]
                    else:
                        continue
                for attr in node.attributes:
                    relevant_attributes_per_element[node_name].add(attr)
        for node in result_sub_graph:
            if type(node) is QueryNode and node.attributes:
                node.attributes = [attr for attr in node.attributes
                                   if attr in relevant_attributes_per_element[node.name]]

        scene_info = result_sub_graph.to_dict()

        return scene_info

    def execute(
            self, sub_graph: SubGraph,
            filter_out: FilterOutSubGraphs = None,
            query_key=None
    ) -> Tuple[List, Dict]:
        pruned_elements = self._optimize_elements(sub_graph)

        if not pruned_elements:
            return set(), {}

        query_str = QueryBuilder.build_query(pruned_elements, self._split)
        result_scenes, scenes_info = self._execute_query_str(query_str, pruned_elements)

        if filter_out:
            result_scenes, scenes_info = self._filter_results(result_scenes, scenes_info, filter_out, query_key)

        return result_scenes, scenes_info

    def _optimize_elements(self, sub_graph: SubGraph):
        if not sub_graph:
            return sub_graph

        output_where = deepcopy(sub_graph)
        pruned_elements = deepcopy(sub_graph)
        for elm in pruned_elements:
            elm.name = set()
            if type(elm) is QueryNode:
                elm.attributes = set()

        if len(sub_graph) < 3:
            if sub_graph[0].name:
                if not sub_graph[0].attributes:
                    return output_where
                names = {sub_graph[0].name} if type(sub_graph[0].name) is str else (sub_graph[0].name or {})
                for obj_name in names:
                    relevant_attributes = self._scene_reader.available_attributes_for_object[obj_name].intersection(
                            sub_graph[0].attributes
                        )
                    if not relevant_attributes:
                        continue

                    pruned_elements[0].attributes.update(relevant_attributes)
                    pruned_elements[0].name.add(obj_name)
                if not pruned_elements[0].name:
                    return None
            else:
                if sub_graph.multi_count and sub_graph.multi_count > 1 or len(sub_graph.root.attributes):
                    # we skip these queries for efficiency
                    return None
                else:
                    return output_where

        for indices, triplet in extract_elements_triplets(sub_graph.root):
            subject_names = {triplet[0].name} if type(triplet[0].name) is str else (triplet[0].name or {'*'})
            relation_names = {triplet[1].name} if type(triplet[1].name) is str else (triplet[1].name or {'*'})
            object_names = {triplet[2].name} if type(triplet[2].name) is str else (triplet[2].name or {'*'})
            subject_attributes = triplet[0].attributes
            object_attributes = triplet[2].attributes

            pruned_triplet = self._prune_triplet(subject_names, relation_names, object_names, subject_attributes,
                                                 object_attributes)
            if not pruned_triplet:
                return None

            subject_index, relation_index, object_index = indices
            pruned_elements[subject_index].attributes.update(pruned_triplet['pruned_subject_attributes'])
            pruned_elements[object_index].attributes.update(pruned_triplet['pruned_object_attributes'])
            pruned_elements[subject_index].name.update(pruned_triplet['pruned_subject_names'])
            pruned_elements[relation_index].name.update(pruned_triplet['pruned_relation_names'])
            pruned_elements[object_index].name.update(pruned_triplet['pruned_object_names'])
        for i, pruned_elm in enumerate(pruned_elements):
            output_where[i].name = pruned_elm.name
            if type(pruned_elm) is QueryNode:
                output_where[i].attributes = pruned_elm.attributes
        return output_where

    def _prune_triplet(self, subject_names, relation_names, object_names, subject_attributes, object_attributes):
        key = (frozenset(subject_names), frozenset(relation_names), frozenset(object_names), frozenset(subject_attributes), frozenset(object_attributes))
        if key in self._pruning_cache:
            return self._pruning_cache[key]

        output = {
            'pruned_subject_attributes': set([]),
            'pruned_object_attributes': set([]),
            'pruned_subject_names': set([]),
            'pruned_object_names': set([]),
            'pruned_relation_names': set([])
        }

        at_least_one_triple_exists = False

        possible_triplets = itertools.product(subject_names, relation_names, object_names)
        for subject_name, relation_name, object_name in possible_triplets:
            if (relation_name == "*" and object_name == "*") or (relation_name == "*" and subject_name == "*"):
                pruned_subject_attributes = set()
                pruned_object_attributes = set()
                found_triplets = True
            else:
                found_triplets = self._scene_reader.available_triplets.get((subject_name, relation_name, object_name))
                if found_triplets:
                    pruned_subject_attributes = subject_attributes.intersection(found_triplets[0])
                    pruned_object_attributes = object_attributes.intersection(found_triplets[1])

                    if subject_attributes and not pruned_subject_attributes:
                        continue
                    if object_attributes and not pruned_object_attributes:
                        continue

                    at_least_one_triple_exists = True

            if found_triplets:
                at_least_one_triple_exists = True
                output['pruned_subject_attributes'].update(pruned_subject_attributes)
                output['pruned_object_attributes'].update(pruned_object_attributes)
                if subject_name != "*":
                    output['pruned_subject_names'].add(subject_name)
                if relation_name != "*":
                    output['pruned_relation_names'].add(relation_name)
                if object_name != "*":
                    output['pruned_object_names'].add(object_name)

        if not at_least_one_triple_exists:
            output = None

        self._pruning_cache[key] = output
        return output

    def starting_new_scene(self, scene_key):
        # clear out the cache to save memory, since cache is mostly useful only withing the same scene
        self._pruning_cache = {}

        # set random seed here based on the scene key, to keep consistency between processes
        self._random.seed(scene_key)

    @staticmethod
    def finished():
        print("Saving redis to disk...")
        redis.execute_command("SAVE")

    def _filter_results(self, result_scenes, scenes_info, filter_out_graph, query_key):
        """
        Filter out results using scene graph (we tried doing it directly with neo4j/cypher, but it was too slow),
        and then optionally with LXMERT model
        """
        if not filter_out_graph:
            return result_scenes, scenes_info
        if type(filter_out_graph) is list:
            assert(len(filter_out_graph) == 1)
            filter_out_graph = filter_out_graph[0]

        # first filter out using graph
        candidate_output_scenes = []
        for scene in result_scenes:
            formatted_scene = self._scene_reader.get_formatted_scenes(scene)
            if self._filter_using_scene(filter_out_graph, formatted_scene):
                continue

            candidate_output_scenes.append(scene)

        self._random.seed(str(result_scenes))
        candidate_output_scenes = self._random.sample(candidate_output_scenes, k=min(self._limit_scenes_output, len(candidate_output_scenes)))
        output_scenes = set(candidate_output_scenes)

        # verification against lxmert should be here (code not available yet)

        output_scenes_info = {s: scenes_info[s] for s in output_scenes}

        return output_scenes, output_scenes_info

    def _filter_using_scene(self, sub_graph, formatted_scene):
        program = sub_graph_root_to_ref_program(sub_graph.root)
        program.append({'operation': 'exists', 'dependencies': [len(program) - 1]})
        return self._executor.run(program, formatted_scene['objects'])

    @staticmethod
    def _serialize_filter_out(filter_out: FilterOutSubGraphs):
        if not filter_out:
            return None
        if type(filter_out) is SubGraph:
            filter_out = [filter_out]

        return [sg.serialize() for sg in filter_out]
