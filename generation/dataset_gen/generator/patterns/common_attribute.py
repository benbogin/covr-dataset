from copy import deepcopy
from typing import List, Set, Dict

from overrides import overrides

from generator import utils

from generator.patterns.question_pattern import QuestionPattern
from generator.queries.query_builder import SubGraph
from generator.resources import Resources
from generator.utils import sub_graph_root_to_ref_text, scene_info_to_sub_graph


class CommonAttributePattern(QuestionPattern):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # indicates if we need more positive subjects (i.e. when we explicitly need both REF_1 and
        # POS_REF_2, and we want them to have different text_refs
        self._needs_more_positives_subjects = 'POS_REF_2' in self._pattern['placeholders']

        self._is_binary_question = self._pattern['program'][-1]['operation'] == 'eq'

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
        all_slots, ref_elements = self._get_slots_for_common_attribute(sub_graph)

        if not all_slots:
            return

        if sub_graph.others_identical_exist:
            return

        ref_program = utils.sub_graph_root_to_ref_program(ref_elements.root)

        for slots in all_slots:
            yield from self.yield_true_answer(slots, negative_scenes, ref_program, scene_key, scenes_info, ref_elements, sub_graph)
            if self._is_binary_question:
                yield from self.yield_false_answer(slots, negative_scenes, ref_program, scene_key, scenes_info,
                                                   ref_elements, sub_graph)

    def yield_true_answer(self, slots, negative_scenes, ref_1_program, scene_key, scenes_info, ref_1_elements, sub_graph):
        potential_other_positives_keys = [k for k in negative_scenes.keys() if '0_attr' not in k
                                          and 'neg_attr_main_object' not in k]
        if not potential_other_positives_keys:
            return None
        selected_other_positive_key = self._random.choice(potential_other_positives_keys)
        relevant_positive_scenes = list(negative_scenes[selected_other_positive_key])
        relevant_negative_scenes = {k: v for k, v in negative_scenes.items()
                                    if ('neg_attr_main_object' in k or '0_attr' in k)}
        if not relevant_negative_scenes:
            return None

        ref_2_program, ref_2_text, ref_2_scene, ref_2_elements = self._pick_ref_2(relevant_positive_scenes, scenes_info, sub_graph)
        if ref_2_elements is None:
            return

        for qst_info in self._yield_question(slots, scene_key, ref_2_scene, relevant_negative_scenes,
                                   ref_1_program, ref_2_program, ref_2_text):
            yield (*qst_info, [ref_1_elements, ref_2_elements])

    def _pick_ref_2(self, relevant_positive_scenes, scenes_info, sub_graph):
        pos_scenes_info = [(s_id, scenes_info.get(s_id)) for s_id in relevant_positive_scenes]
        negative_elements = [(sc, scene_info_to_sub_graph(nodes)) for sc, nodes in pos_scenes_info]

        # # only take images where number annotated objects are predicted to match number of objects in image
        # negative_elements = [(sc, sg) for sc, sg in negative_elements
        #                      if not self._lxmert_scene_graph_verifier.check_other_identical_sub_graphs_exist_from_cache(sg, sc)[0]]
        if not negative_elements:
            return None, None, None, None

        ref_2_scene, sel_elements = self._random.choice(negative_elements)
        sel_elements = deepcopy(sel_elements)
        self._keep_only_valid_attributes_in_neg_sub_graph(sel_elements, sub_graph)
        sel_elements[0].attributes = set()

        ref_text_2 = sub_graph_root_to_ref_text(sel_elements.root, determiner="the/a")
        ref_2_program = utils.sub_graph_root_to_ref_program(sel_elements.root)

        return ref_2_program, ref_text_2, ref_2_scene, sel_elements

    def yield_false_answer(self, slots, negative_scenes, ref_1_program, scene_key, scenes_info, ref_1_elements, sub_graph):
        potential_other_positives_keys = [k for k in negative_scenes.keys()
                                          if '0_attr_' in k or 'neg_attr_main_object' in k]
        if not potential_other_positives_keys:
            return None
        selected_other_positive_key = self._random.choice(potential_other_positives_keys)
        relevant_positive_scenes = list(negative_scenes[selected_other_positive_key])
        relevant_negative_scenes = {k: v for k, v in negative_scenes.items()
                                    if 'neg_attr_main_object' not in k}
        if not relevant_negative_scenes:
            return None

        ref_2_program, ref_2_text, ref_2_scene, ref_2_elements = self._pick_ref_2(relevant_positive_scenes, scenes_info, sub_graph)
        if ref_2_elements is None:
            return

        for qst_info in self._yield_question(slots, scene_key, ref_2_scene, relevant_negative_scenes,
                                             ref_1_program, ref_2_program, ref_2_text):
            yield (*qst_info, [ref_1_elements, ref_2_elements])

    def _get_slots_for_common_attribute(self, sub_graph):
        if not sub_graph[0].attributes:
            return [], None
        for attribute in sub_graph[0].attributes:
            if attribute not in Resources.attributes_group_by_name:
                continue

            if Resources.attributes_group_by_name[attribute] == 'color' and any(sub_graph.root.name.startswith(prefix) for prefix in self._nouns_to_ignore_for_color_questions):
                continue

            ref_1_elements = deepcopy(sub_graph)
            ref_1_elements[0].attributes = set()
            picked_slots = {
                'REF_1': utils.sub_graph_root_to_ref_text(ref_1_elements.root, determiner="the/a"),
                'ATTR_TYPE_1': Resources.attributes_group_by_name[attribute]
            }
            return [picked_slots], ref_1_elements
        return [], None

    def _yield_question(self, slots, ref_1_scene, ref_2_scene, relevant_negative_scenes, ref_1_program, ref_2_program,
                        ref_2_text):
        ref_1_text = slots['REF_1']
        shuffle_ref_1_ref_2 = self._random.choice([True, False])
        if shuffle_ref_1_ref_2:
            ref_1_scene, ref_2_scene = ref_2_scene, ref_1_scene
            ref_1_program, ref_2_program = ref_2_program, ref_1_program
            ref_1_text, ref_2_text = ref_2_text, ref_1_text

        slots = deepcopy(slots)
        slots['REF_1'] = ref_1_text
        slots['POS_REF_2'] = ref_2_text
        program = utils.merge_ref_program(self._pattern['program'], ref_1_program, "REF_1")
        program = utils.merge_ref_program(program, ref_2_program, "POS_REF_2")

        scenes_with_ref_1, scenes_without_ref_1 = self._get_scenes_with_and_without_ref(
            [s for sc in relevant_negative_scenes.values() for s in sc],
            ref_1_program
        )

        scenes_with_ref_2, scenes_without_ref_2 = self._get_scenes_with_and_without_ref(
            [s for sc in relevant_negative_scenes.values() for s in sc],
            ref_2_program
        )

        scenes_without_both_refs = {'scenes_without': set(scenes_without_ref_1).intersection(set(scenes_without_ref_2))}

        neg, pos = self._pick_scenes([ref_1_scene, ref_2_scene], scenes_without_both_refs, ref_1_scene,
                                     n_min_positive=2, n_max_positive=2)
        yield slots, program, pos, neg
