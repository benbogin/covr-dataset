from copy import deepcopy
from typing import List, Set, Dict

from overrides import overrides

from generator import utils
from generator.patterns.question_pattern import QuestionPattern
from generator.queries.data_classes import QueryNode
from generator.queries.query_builder import SubGraph
from generator.resources import Resources
from generator.utils import scene_info_to_sub_graph


class QueryAttributePattern(QuestionPattern):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._is_choose_pattern = 'ATTR_NEG' in self._pattern['placeholders']
        self._is_verify_pattern = 'ATTR_POS' in self._pattern['placeholders'] and not self._is_choose_pattern

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
        all_slots, subject_elements = self._get_slots_for_attribute_questions(
            sub_graph, negative_scenes, add_determiner="the/a", make_ref_plural=False
        )

        if sub_graph.others_identical_exist:
            return

        if all_slots:
            subject_program = utils.sub_graph_root_to_ref_program(subject_elements.root)
            program = utils.merge_ref_program(self._pattern['program'], subject_program, "REF_1")

            for slots in all_slots:
                if '0_attr' not in negative_scenes:
                    continue

                # we don't want other objects with a different attribute since we're looking for a specific object,
                # thus we remove "0_attr" negatives
                relevant_negative_scenes = self._filter_out_scene_keys(negative_scenes, filter_out_keys=['0_attr'])

                neg, pos = self._pick_scenes(positive_scenes, relevant_negative_scenes, scene_key,
                                             n_min_positive=1, n_max_positive=1)

                sub_graphs = [sub_graph]

                negative_attr = None

                if self._is_choose_pattern:
                    attr_pos = slots['ATTR_POS']
                    contradicting_attrs = Resources.contradicting_attributes[attr_pos]

                    neg_scenes_info = [(s_id, scenes_info.get(s_id, None)) for s_id in negative_scenes.get('0_attr', [])]
                    neg_scenes_info = [(s_id, scene_info_to_sub_graph(scene_info))
                                       for s_id, scene_info in neg_scenes_info
                                       if scene_info]
                    all_attrs = set(
                        [a for s, nodes in neg_scenes_info for a in (nodes[0].attributes if type(nodes[0]) is QueryNode else [])]
                    )
                    contradicting_attrs_in_image = [a for a in all_attrs if a in contradicting_attrs]

                    if not contradicting_attrs_in_image:
                        continue

                    negative_attr = self._random.choice(contradicting_attrs_in_image)

                    slots['ATTR_NEG'] = negative_attr
                    ref_query_2_elements = deepcopy(sub_graph)
                    ref_query_2_elements[0].attributes = {slots['ATTR_NEG']}
                    sub_graphs.append(ref_query_2_elements)

                    shuffle_ref_1_ref_2 = self._random.choice([True, False])
                    if shuffle_ref_1_ref_2:
                        slots['ATTR_NEG'], slots['ATTR_POS'] = slots['ATTR_POS'], slots['ATTR_NEG']
                yield slots, program, pos, neg, sub_graphs

                if self._is_verify_pattern or self._is_choose_pattern:
                    # add questions with different answer

                    relevant_negative_scenes = dict(negative_scenes)
                    if self._is_choose_pattern:
                        # create another question with the distracting answer
                        if negative_attr:
                            distracting_scenes = [s for (s, sg) in neg_scenes_info if negative_attr in sg.root.attributes]
                            relevant_negative_scenes['0_attr'] = {self._random.choice(distracting_scenes)}
                    else:
                        # make sure there is only one object that is referred to
                        relevant_negative_scenes['0_attr'] = {self._random.choice(list(relevant_negative_scenes['0_attr']))}

                    neg, pos = self._pick_scenes([], relevant_negative_scenes, scene_key,
                                                 prioritized_negative_scene_keys=['0_attr'],
                                                 n_min_positive=1, n_max_positive=1)

                    yield slots, program, pos, neg, [sub_graph]
