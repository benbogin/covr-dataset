from abc import ABC, abstractmethod
from copy import deepcopy
from itertools import zip_longest
from random import Random
from typing import Dict, List, Set

from executor.executor import Executor
from generator.queries.data_classes import QueryNode
from generator.queries.query_builder import SubGraph
from generator.resources import Resources
from generator.scene_reader import SceneReader
from generator.utils import sub_graph_root_to_ref_text, sub_graph_root_to_ref_program, remove_duplicates


class NoNegativeScenesException(Exception):
    pass


class QuestionPattern(ABC):
    MAX_CONTEXT_IMAGES = 5

    def __init__(self, pattern_dict: Dict, scene_reader: SceneReader, random_seed=None,
                 allow_ref_no_relations: bool = False, pick_random_distractors: bool = False):
        self._pattern = pattern_dict
        self._scene_reader = scene_reader
        self._random = Random(random_seed)
        self._allow_ref_no_relations = allow_ref_no_relations
        self._pick_random_distractors = pick_random_distractors

        # create a separate random instance to run this experiment without changing the original random choices
        self._random_neg = Random(random_seed)

        self._executor = Executor()

        self._nouns_to_ignore_for_color_questions = {'building', 'tree', 'sign', 'clock', 'bush', 'bus', 'flag', 'leaf', 'leaves',
                                      'sauce', 'water', 'ocean'}

    def generate_questions(self, sub_graph: SubGraph, negative_scenes: Dict, positive_scenes: Set, scene_key: str,
                           scenes_info: Dict, all_ref_instances: List[Dict]):
        simple_ref_text = sub_graph_root_to_ref_text(sub_graph.root, determiner="", verb="", add_that=False,
                                                     add_parenthesis=False)
        ref_text = sub_graph_root_to_ref_text(sub_graph.root)
        ref_program = sub_graph_root_to_ref_program(sub_graph.root)

        self._random.seed(str((scene_key, ref_text, self._pattern['pattern_index'])))

        if not self._allow_ref_no_relations and len(sub_graph) < 3:
            return

        for picked_slots, program, pos, neg, sub_graphs in self._generate_questions(
                ref_text, ref_program, sub_graph, negative_scenes, positive_scenes, scene_key,
                scenes_info['subgraphs'], all_ref_instances
        ):
            relevant_scenes_info = {
                'subgraphs': {k: scenes_info['subgraphs'][k] for k in pos + neg if k in scenes_info['subgraphs']},
                'queries': {k: scenes_info['queries'][k] for k in pos + neg},
            }
            if self._pick_random_distractors:
                neg = self._random.sample(self._scene_reader.all_scenes_keys, len(neg))
            scenes = pos + neg
            self._random.shuffle(scenes)
            yield picked_slots, self._pattern, program, scenes, relevant_scenes_info, sub_graphs, simple_ref_text, None

    @staticmethod
    def does_pattern_support_multi_instances() -> bool:
        return False

    @abstractmethod
    def _generate_questions(self, ref_text: str,
                            ref_program: List,
                            sub_graph: SubGraph,
                            negative_scenes: Dict,
                            positive_scenes: Set,
                            scene_key: str,
                            scenes_info: Dict,
                            all_ref_instances: List[Dict]):
        pass

    def _pick_scenes(self, positive_scenes, negative_scenes_dict, scene_key,
                     prioritized_negative_scene_keys=None,
                     n_min_positive: int = 1,
                     n_max_positive: int = None,
                     n_min_negative: int = None,
                     n_max_negative: int = None,
                     positives_more_than_negatives: bool = False,
                     negatives_more_than_positives: bool = False,
                     ):
        """
        prioritized_negative_scene_keys: keys of negative scenes that for which at least one image should be picked
        """
        n_available_positive_scenes = len(positive_scenes)
        n_available_negative_scenes = len(set([s_id for s_set in negative_scenes_dict.values() for s_id in s_set]))
        n_positive = self._random.randint(
            min(n_available_positive_scenes, n_min_positive),
            min(n_available_positive_scenes, n_max_positive if n_max_positive is not None else QuestionPattern.MAX_CONTEXT_IMAGES))
        n_negative = QuestionPattern.MAX_CONTEXT_IMAGES - n_positive

        if n_min_negative:
            n_negative = self._random.randint(n_min_negative,
                                              min(n_available_negative_scenes, QuestionPattern.MAX_CONTEXT_IMAGES - 1))
            n_positive = QuestionPattern.MAX_CONTEXT_IMAGES - n_negative
        elif positives_more_than_negatives:
            n_negative = self._random.randint(0, n_positive - 1)
        elif negatives_more_than_positives:
            n_negative = self._random.randint(n_positive + 1,
                                              min(n_available_negative_scenes, QuestionPattern.MAX_CONTEXT_IMAGES))

        if positive_scenes and n_positive:
            # shuffle positive scenes but make sure current scene is first to appear
            if scene_key not in positive_scenes:
                raise ValueError(f"scene_key {scene_key} not found in positive_scenes")
            positive_scenes.remove(scene_key)
            self._random.shuffle(positive_scenes)
            positive_scenes.insert(0, scene_key)
            positive_scenes = list(positive_scenes)[:n_positive]
        else:
            n_positive = 0
            n_negative = QuestionPattern.MAX_CONTEXT_IMAGES
            positive_scenes = []

        # spread negative scenes evenly across different randomly selected keys
        negative_scenes_keys = list(negative_scenes_dict.keys())
        # sort list before shuffling to make choices stable for same random seed
        self._random.shuffle(sorted(negative_scenes_keys))

        if prioritized_negative_scene_keys:
            # make sure at least one key from prioritized_scene_keys appears in negative scenes
            if n_negative == 0:
                n_negative = 1
                if n_negative + n_positive > QuestionPattern.MAX_CONTEXT_IMAGES:
                    n_negative -= 1

            for key in list(prioritized_negative_scene_keys):
                if key in negative_scenes_keys:
                    negative_scenes_keys.remove(key)
            negative_scenes_keys = prioritized_negative_scene_keys + negative_scenes_keys

        negative_scenes = [s for sc in zip_longest(*[negative_scenes_dict[k] for k in negative_scenes_keys]) for s in sc if s is not None]
        negative_scenes = remove_duplicates(negative_scenes)
        negative_scenes = list(negative_scenes)[:n_negative]

        return negative_scenes, positive_scenes

    def _get_slots_for_attribute_questions(self, sub_graph, negative_scenes,
                                           make_ref_plural=True,
                                           add_determiner="a"):
        all_slots = []
        object_attributes = sub_graph[0].attributes

        if len(sub_graph) > 1:
            # if we have a sub graph with relations, we want to make sure we have interesting negatives where the diff
            # is not only the root. (but if the sub graph is length 1, it doesn't matter...)
            relevant_negative_scenes = {k: v for k, v in negative_scenes.items() if k.endswith('attr_main_object')}
            if not relevant_negative_scenes:
                return all_slots, None

        # only relevant if current query has an attribute
        if not object_attributes:
            return all_slots, None

        if 'ATTR_TYPE_1' in self._pattern['placeholders']:
            attr = list(object_attributes)[0]  # TODO: support multiple attributes

            # only relevant if this attribute has a known type
            if attr not in Resources.attributes_group_by_name:
                return all_slots, None

            if Resources.attributes_group_by_name[attr] == 'color' and any(sub_graph.root.name.startswith(prefix) for prefix in self._nouns_to_ignore_for_color_questions):
                return all_slots, None

            selected_attribute_group = Resources.attributes_group_by_name[attr]
            slots = {}
            slots.update({"ATTR_TYPE_1": selected_attribute_group, "ATTR_POS": attr})
            all_slots.append(slots)
        elif 'ATTR_POS' in self._pattern['placeholders']:
            for attr in object_attributes:
                slots = {"ATTR_POS": attr}
                all_slots.append(slots)

        for i, sl in enumerate(all_slots):
            sub_graph = deepcopy(sub_graph)
            sub_graph[0].attributes.remove(sl['ATTR_POS'])
            all_slots[i]['REF_1'] = sub_graph_root_to_ref_text(sub_graph.root, make_plural=make_ref_plural,
                                                               determiner=add_determiner)
            subject_elements = sub_graph

        return all_slots, subject_elements

    @staticmethod
    def _pick_scene_keys(scene_keys, if_equal_to: List = None, if_contains: List = None):
        if not if_equal_to:
            if_equal_to = []
        if not if_contains:
            if_contains = []
        keys = set()
        for key, values in scene_keys.items():
            if key in if_equal_to:
                keys.add(key)
            elif any([c in key for c in if_contains]):
                keys.add(key)

        return list(keys)

    @staticmethod
    def _filter_out_scene_keys(scene_keys, filter_out_keys):
        return {k: v for k, v in scene_keys.items() if k not in filter_out_keys}

    def _pick_quantifier_params(self):
        if 'QUANTIFIER' in self._pattern['placeholders']:
            quantifier_module = self._random.choice(['all', 'some', 'none'])
        else:
            quantifier_module = "all"
        if quantifier_module == "all":
            pick_scenes_params_true = {}
            pick_scenes_params_false = {}
            quantifier_name = "All"
        elif quantifier_module == "some":
            pick_scenes_params_true = {'n_min_positive': 1}
            pick_scenes_params_false = {'n_min_positive': 0, 'n_max_positive': 0}
            quantifier_name = "Some"
        # elif quantifier == "most":
        #     pick_scenes_params_true = {'positives_more_than_negatives': True}
        #     pick_scenes_params_false = {'negatives_more_than_positives': True}
        elif quantifier_module == "none":
            pick_scenes_params_true = {'n_min_positive': 0, 'n_max_positive': 0}
            pick_scenes_params_false = {'n_min_positive': 1}
            quantifier_name = "No"
        else:
            raise ValueError()
        return pick_scenes_params_false, pick_scenes_params_true, quantifier_module, quantifier_name

    def _get_scenes_with_and_without_ref(self, negative_scenes, ref_2_program):
        scenes_with_ref = []
        scenes_without_ref = []
        for scene_id in negative_scenes:
            scene_objects_dict = self._scene_reader.get_formatted_scenes(scene_id)['objects']
            exists = self._executor.run(ref_2_program, scene_objects_dict)
            if exists:
                scenes_with_ref.append(scene_id)
            else:
                scenes_without_ref.append(scene_id)
        return scenes_with_ref, scenes_without_ref

    def _keep_only_valid_attributes_in_neg_sub_graph(self, neg_sub_graph, pos_sub_graph):
        # If any of negative element has attribute(s), pick a single one that is contradicted with positive element.
        # This is necessary since `scene_info_to_sub_graph` possibly returns multiple attributes, which we don't
        # support.
        all_pos_nodes = [elm for elm in pos_sub_graph if type(elm) is QueryNode]
        all_pos_attributes = set(attr for elm in all_pos_nodes for attr in elm.attributes)
        for elm in neg_sub_graph:
            if type(elm) is QueryNode and elm.attributes:
                valid_attributes = [attr for attr in elm.attributes if attr in all_pos_attributes or
                                    any(attr in Resources.contradicting_attributes[pos_attr] for pos_attr in
                                        all_pos_attributes)]
                elm.attributes = {self._random.choice(valid_attributes)} if valid_attributes else set()
