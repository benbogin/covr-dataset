from copy import deepcopy
from typing import List, Set, Dict

from overrides import overrides

from generator import utils
from generator.patterns.question_pattern import QuestionPattern
from generator.queries.query_builder import SubGraph
from generator.utils import sub_graph_root_to_ref_text


class AllSubjectPattern(QuestionPattern):
    @overrides
    def _generate_questions(self,
                            ref_text: str,
                            ref_program: List,
                            sub_graph: SubGraph,
                            negative_scenes: Dict,
                            positive_scenes: Set,
                            scene_key: str,
                            scenes_info: Dict,
                            all_ref_instances: List[Dict]):
        for picked_slots, subject_node, condition_nodes in self._get_slots_for_subject_questions(sub_graph, negative_scenes):
            subject_program = utils.sub_graph_root_to_ref_program(subject_node)
            condition_program = utils.sub_graph_root_to_ref_program(condition_nodes)
            program = utils.merge_ref_program(self._pattern['program'], subject_program, "REF_1_SUBJECT")

            pick_scenes_params_false, pick_scenes_params_true, quantifier_module, quantifier_name = self._pick_quantifier_params()

            picked_slots["REF_1_NO_SUBJECT_CONDITION"] = condition_program
            picked_slots['QUANTIFIER'] = quantifier_name
            picked_slots['QUANTIFIER_MODULE'] = quantifier_module

            # answer should be True
            scene_keys_for_false_answer = self._pick_scene_keys(negative_scenes,
                                                                if_equal_to=['2_attr', '2_name', '1_relation'])

            relevant_negative_scenes = self._filter_out_scene_keys(negative_scenes, scene_keys_for_false_answer)

            if not relevant_negative_scenes:
                continue

            neg, pos = self._pick_scenes(positive_scenes, relevant_negative_scenes, scene_key, **pick_scenes_params_true)
            yield picked_slots, program, pos, neg, [sub_graph]

            # answer should be False
            if scene_keys_for_false_answer:
                neg, pos = self._pick_scenes(positive_scenes, negative_scenes, scene_key,
                                             prioritized_negative_scene_keys=scene_keys_for_false_answer,
                                             **pick_scenes_params_false)
                yield picked_slots, program, pos, neg, [sub_graph]

    def _get_slots_for_subject_questions(self, sub_graph, negative_scenes):
        all_slots = []
        if len(sub_graph) <= 1:
            return []
        if any(node.name in ['of'] for node in sub_graph):
            return []

        # take similar scenes where the object of the main subject would be different
        relevant_negative_scenes = {k: v for k, v in negative_scenes.items() if 'neg_object' not in k}
        if not relevant_negative_scenes:
            return []

        # go through relations that lead to a leaf and cut leaf out
        sub_graph1 = deepcopy(sub_graph)
        for leaf_triplet, leaf_triplet_indices in zip(*sub_graph1.get_leaf_triplets()):
            leaf_triplet = deepcopy(leaf_triplet)
            condition_triplet = deepcopy(leaf_triplet)
            leaf_triplet[0].relations = [r for r in leaf_triplet[0].relations if r != leaf_triplet[1]]

            condition_triplet[0].empty = True
            condition_triplet[0].name = None
            condition_triplet[0].attributes = {}
            condition_triplet[0].backward_relation = None
            condition_triplet[0].relations = [r for r in condition_triplet[0].relations if r.target == condition_triplet[2]]

            ref_1_subject = sub_graph_root_to_ref_text(leaf_triplet[0], make_plural=True, determiner=None)
            ref_1_no_subject = sub_graph_root_to_ref_text(condition_triplet[0], verb='are')

            slots = {
                'REF_1_SUBJECT': ref_1_subject,
                'REF_1_NO_SUBJECT': ref_1_no_subject
            }

            all_slots.append((slots, leaf_triplet[0], condition_triplet[0]))

        # cut off two leaves (to make a sentence such as "all mean are holding a helmet and wearing a cap")
        if len(sub_graph) == 5 and sub_graph.depth() == 3:
            sub_graph_copy = deepcopy(sub_graph)
            sub_graph_copy[0].relations = {}

            sub_graph_condition = deepcopy(sub_graph)
            sub_graph_condition[0].empty = True
            sub_graph_condition[0].name = None
            sub_graph_condition[0].attributes = {}

            ref_1_subject = sub_graph_root_to_ref_text(sub_graph_copy[0], make_plural=True, determiner=None)
            ref_1_no_subject = sub_graph_root_to_ref_text(sub_graph_condition[0], verb='are')

            slots = {
                'REF_1_SUBJECT': ref_1_subject,
                'REF_1_NO_SUBJECT': ref_1_no_subject
            }

            all_slots.append((slots, sub_graph_copy[0], sub_graph_condition[0]))

        return all_slots
