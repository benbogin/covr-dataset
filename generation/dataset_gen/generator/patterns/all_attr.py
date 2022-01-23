from typing import List, Set, Dict

from overrides import overrides

from generator import utils
from generator.patterns.question_pattern import QuestionPattern
from generator.queries.query_builder import SubGraph


class AllAttributePattern(QuestionPattern):
    def __init__(self, *args, **kwargs):
        kwargs['allow_ref_no_relations'] = True
        super().__init__(*args, **kwargs)

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
        all_slots, subject_elements = self._get_slots_for_attribute_questions(sub_graph, negative_scenes,
                                                                              make_ref_plural=True)

        pick_scenes_params_false, pick_scenes_params_true, quantifier_module, quantifier_name = self._pick_quantifier_params()

        if all_slots:
            relevant_negative_scenes = self._filter_out_scene_keys(negative_scenes, filter_out_keys=['0_attr'])

            subject_program = utils.sub_graph_root_to_ref_program(subject_elements.root)
            program = utils.merge_ref_program(self._pattern['program'], subject_program, "REF_1_SUBJECT")

            for slots in all_slots:
                slots['QUANTIFIER'] = quantifier_name
                slots['QUANTIFIER_MODULE'] = quantifier_module
                # answer should be True
                neg, pos = self._pick_scenes(
                    positive_scenes, relevant_negative_scenes, scene_key, **pick_scenes_params_true
                )
                if len(pos) + len(neg) < 2:
                    # questions with a single image do not make sense for this pattern
                    continue
                yield slots, program, pos, neg, [sub_graph]

                # answer should be False
                if '0_attr' in negative_scenes:
                    neg, pos = self._pick_scenes(positive_scenes, negative_scenes, scene_key,
                                                 prioritized_negative_scene_keys=['0_attr'],
                                                 **pick_scenes_params_false)
                    if len(pos) + len(neg) < 2:
                        # questions with a single image do not make sense for this pattern
                        continue
                    yield slots, program, pos, neg, [sub_graph]
