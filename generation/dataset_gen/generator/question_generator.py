import os
import re
from collections import Counter, defaultdict
from copy import deepcopy
from typing import List

from executor.executor import Executor, NonUniqueException, NoCommonAttributeException, \
    MultipleAttributesForTypeException, NonExistentException, NeitherOfChooseException
from generator.graph_traversal import GraphTraversal
from generator.patterns.question_pattern_factory import QuestionPatternFactory
from generator.queries.distractor_queries import DistractorQueries
from generator.queries.graph_executor import GraphExecutor

from generator.resources import Resources
from generator.scene_reader import SceneReader
from generator.utils import sub_graph_root_to_ref_text, fill_program_slots, \
    sub_graph_root_to_ref_program, extract_elements_triplets


class QuestionGenerationError(Exception):
    pass


class NoContextScenesFoundException(QuestionGenerationError):
    pass


class QuestionGenerator:
    """
    This is the core class that generates questions given a scene key
    """
    MAX_RETRIES = 5
    generation_errors_per_pattern = defaultdict(lambda: Counter())

    def __init__(self,
                 random_seed=0,
                 split: str = 'val',
                 question_patterns_ids: List = None,
                 enable_graph_cache: bool = True,
                 scene_reader: SceneReader = None):
        # train or validation split
        self._split = split

        # graph executor - executes our formal semantic language on scene graphs
        self._executor = Executor()

        # the scene reader is responsible for
        self.scene_reader = scene_reader or SceneReader(self._split, os.path.join(Resources.base_path, 'data'))

        self._graph_traversal = GraphTraversal(random_seed)
        self._graph_executor = GraphExecutor(split=split, random_seed=random_seed,
                                             scene_reader=self.scene_reader, enable_cache=enable_graph_cache)

        self._question_patterns = [QuestionPatternFactory.create(p, self.scene_reader, random_seed)
                                   for (i, p) in enumerate(Resources.question_patterns)
                                   if not question_patterns_ids or i in question_patterns_ids]

        self._distractor_queries = DistractorQueries(self.scene_reader)

    def generate_question_from_scene(self, scene_key, keep_only_graph_structure: str = None) -> str:
        scene = self.scene_reader.get_formatted_scenes(scene_key)

        self._graph_executor.starting_new_scene(scene_key)

        objects_to_iterate = scene['objects'].items()

        ref_text_to_sub_graphs = defaultdict(list)

        for sel_obj_id, sel_obj in objects_to_iterate:
            for sub_graph in self._graph_traversal.traverse(sel_obj_id, scene):
                ref_text = sub_graph_root_to_ref_text(sub_graph.root)

                ref_text_to_sub_graphs[ref_text].append(sub_graph)

        # take one arbitrary sub graph for each text (doesn't matter which one since text will be the same)
        sub_graphs_for_questions = [sgs[0] for sgs in ref_text_to_sub_graphs.values()]

        # we want to detect similar sub-graphs in the same scene, to ask questions such as "in one image there are
        # at least three dogs"
        mult_scene_verified_prob = None
        for sub_graphs in ref_text_to_sub_graphs.values():
            if len(sub_graphs) > 4:
                continue
            mult_sub_graph = deepcopy(sub_graphs[0])  # arbitrarily taking first
            mult_sub_graph.multi_count = len(sub_graphs)
            sub_graphs_for_questions.append(mult_sub_graph)

        for sub_graph in sub_graphs_for_questions:
            if not self._is_valid_question_sub_graph(sub_graph):
                continue
            if keep_only_graph_structure == "v_shape":
                if not (len(sub_graph) > 3 and sub_graph.depth() == 3):
                    continue
            if keep_only_graph_structure == "5_chain":
                if not (len(sub_graph) == 5 and len(sub_graph.root.relations) == 1):
                    continue
            if keep_only_graph_structure == "preposition":
                if not (len(sub_graph) > 3 and sub_graph.root.relations[0].prepositions and
                        len(sub_graph.root.relations[0].prepositions) > 0):
                    continue

            for (picked_slots, question_pattern, program, scenes, dbg_info, sub_graphs, simple_ref_text, _) \
                    in self._generate_questions_from_sub_graph(sub_graph, scene):

                formatted_scenes = [self.scene_reader.get_formatted_scenes(k) for k in scenes]
                objects_from_all_scenes = {ok: ov for s in formatted_scenes for ok, ov in s['objects'].items()}

                text = self._fill_text_slots(question_pattern['text'], picked_slots)
                program = fill_program_slots(program, picked_slots)

                err_counter = QuestionGenerator.generation_errors_per_pattern[question_pattern['pattern_index']]
                try:
                    answer = self._executor.run(program, objects_from_all_scenes)
                except NonUniqueException:
                    err_counter['non_unique'] += 1
                    continue
                except NonExistentException:
                    err_counter['non_existent'] += 1
                    continue
                except NeitherOfChooseException:
                    err_counter['invalid_choose'] += 1
                    continue
                except MultipleAttributesForTypeException:
                    err_counter['multiple_attributes'] += 1
                    continue
                except NoCommonAttributeException:
                    err_counter['no_common_attribute'] += 1
                    print(f"NoCommonAttributeException, scene: {scene_key}, picked_slots: {picked_slots}")
                    continue

                if not self._is_valid_answer(answer):
                    continue

                text = self._fix_text(text).capitalize()
                sub_graphs = [[elm.as_tuple() for elm in elms] for elms in sub_graphs]
                dbg_info['sub_graph_info'] = {
                    'length': len(sub_graph),
                    'depth': sub_graph.depth(),
                    'attributes': sum([len(n.attributes) for n in sub_graph if hasattr(n, 'attributes')])
                }
                dbg_info['mult_scene_verified_prob'] = mult_scene_verified_prob
                yield text, program, scenes, answer, question_pattern, dbg_info, sub_graphs, simple_ref_text, None

    def _generate_questions_from_sub_graph(self, sub_graph, scene):
        scene_key = scene['scene_key']
        positive_scenes, negative_scenes, scenes_info = self._get_verified_context_scenes(scene_key, sub_graph)

        if len(positive_scenes) < 1 or len(negative_scenes) < 1:
            return

        ref_program = sub_graph_root_to_ref_program(sub_graph.root)
        all_ref_instances = self._executor.run(ref_program, scene['objects'])
        for question_pattern in self._question_patterns:
            if sub_graph.multi_count and not question_pattern.does_pattern_support_multi_instances():
                continue
            question_info_gen = question_pattern.generate_questions(
                sub_graph, negative_scenes, positive_scenes, scene_key, scenes_info, all_ref_instances)
            yield from question_info_gen

    @staticmethod
    def _fill_text_slots(question_pattern: str, picked_slots: dict):
        for slot, value in picked_slots.items():
            question_pattern = question_pattern.replace('{' + slot + '}', str(value))

        return question_pattern

    def _get_verified_context_scenes(self, curr_scene_key, query_elements):
        positive_scenes, negative_scenes, scenes_info = self._get_context_scenes(
            curr_scene_key, query_elements
        )
        return positive_scenes, negative_scenes, scenes_info

    def _get_context_scenes(self, curr_scene_key, sub_graph):
        negative_scenes = {}

        positive_scenes, scenes_subgraphs = self._graph_executor.execute(sub_graph)
        positive_scenes.append(curr_scene_key)

        scenes_info = {'subgraphs': scenes_subgraphs, 'queries': {k: {'positive'} for k in positive_scenes}}

        negatives = self._distractor_queries.get_negative_queries(sub_graph)

        for neg in negatives:
            key = str(neg['e_id']) + '_' + neg['source']
            scenes, dbg = self._graph_executor.execute(neg['query'], neg['filter_out'], key)
            if scenes:
                neg_scenes = set()
                for scene in scenes:
                    if scene in positive_scenes:
                        continue
                    if scene not in scenes_info['subgraphs']:
                        scenes_info['subgraphs'][scene] = dbg[scene]
                    if scene not in scenes_info['queries']:
                        scenes_info['queries'][scene] = set()
                    scenes_info['queries'][scene].add(key)
                    neg_scenes.add(scene)
                if neg_scenes:
                    negative_scenes[key] = neg_scenes

        for key, queries in scenes_info['queries'].items():
            scenes_info['queries'][key] = list(queries)

        return list(set(positive_scenes)), negative_scenes, scenes_info

    def _is_valid_question_sub_graph(self, sub_graph):
        for _, triplet in extract_elements_triplets(sub_graph.root):
            pair = f"{triplet[1].name},{triplet[2].name}"
            if pair in Resources.ignore['pairs'] and not triplet[2].attributes:
                # skip dull relations such as "man wearing shirt", but keep if there's an attribute,
                # e.g. "man wearing blue shirt"
                return False

        return True

    @staticmethod
    def _fix_text(text):
        text = text.replace('are 1 images', 'is 1 image')
        text = text.replace('are at least 1 images', 'is at least 1 image')
        text = text.replace('are at most 1 images', 'is at most 1 image')

        # remove redundant whitespaces
        text = ' '.join(text.split())

        # remove artifacts stored in graph db for imsitu images
        text = re.sub('[^\s]+:_', '', text)

        return text

    @staticmethod
    def _is_valid_answer(answer):
        if type(answer) is int and answer > 5:
            return False
        return True
