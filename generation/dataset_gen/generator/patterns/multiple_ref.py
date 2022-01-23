from typing import List, Set, Dict

from overrides import overrides

from generator import utils
from generator.patterns.question_pattern import QuestionPattern
from generator.queries.query_builder import SubGraph
from generator.utils import scene_info_to_sub_graph, sub_graph_root_to_ref_text, \
    merge_ref_program, sub_graph_root_to_ref_program


class MultipleRefPattern(QuestionPattern):
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
        neg_scenes_info = [(s_id, scenes_info.get(s_id)) for s_set in negative_scenes.values() for s_id in s_set]
        negative_elements = [scene_info_to_sub_graph(nodes) for s, nodes in neg_scenes_info]
        sel_elements = self._random.choice(negative_elements)

        self._keep_only_valid_attributes_in_neg_sub_graph(sel_elements, sub_graph)

        ref_texts = [ref_text, sub_graph_root_to_ref_text(sel_elements.root)]
        self._random.shuffle(ref_texts)

        slots = {
            "REF_1": ref_texts[0],
            "REF_2": ref_texts[1]
        }

        ref_2_program = sub_graph_root_to_ref_program(sel_elements.root)
        program = utils.merge_ref_program(self._pattern['program'], ref_program, "REF_1")
        program = merge_ref_program(program, ref_2_program, "REF_2")

        scenes_with_ref_2, scenes_without_both_refs = self._get_scenes_with_and_without_ref(
            [s for s, _ in neg_scenes_info], ref_2_program
        )

        if not scenes_without_both_refs or not scenes_with_ref_2:
            return

        negative_scenes = {'scenes_without': scenes_without_both_refs}
        sel_scenes_with_refs = [scene_key, self._random.choice(scenes_with_ref_2)]

        picked_logic_module = self._random.choice(["logic_and", "logic_or"])
        slots["LOGIC_MODULE"] = picked_logic_module

        if picked_logic_module == "logic_and":
            slots["LOGIC1"] = "both"
            slots["LOGIC2"] = "and"

            # Question with true as answer
            neg, pos = self._pick_scenes(sel_scenes_with_refs, negative_scenes, scene_key, n_min_positive=2)

            if len(pos) + len(neg) >= 2:
                # questions with a single image do not make sense for this pattern
                yield slots, program, pos, neg, [sub_graph, sel_elements]

            # Question with false as answer
            positives_to_show = self._random.choice([0, 1])
            scenes_for_false = sel_scenes_with_refs[:positives_to_show]
            neg, pos = self._pick_scenes(scenes_for_false, negative_scenes, scene_key,
                                         n_min_positive=1, n_max_positive=1)
            if len(pos) + len(neg) >= 2:
                # questions with a single image do not make sense for this pattern
                yield slots, program, pos, neg, [sub_graph, sel_elements]
        elif picked_logic_module == "logic_or":
            slots["LOGIC1"] = "either"
            slots["LOGIC2"] = "or"

            # Question with true as answer
            positives_to_show = self._random.choice([1, 2])
            scenes_for_true = sel_scenes_with_refs[:positives_to_show]
            neg, pos = self._pick_scenes(scenes_for_true, negative_scenes, scene_key,
                                         n_min_positive=positives_to_show, n_max_positive=positives_to_show)
            slots["LOGIC_MODULE"] = "logic_or"
            if len(pos) + len(neg) >= 2:
                # questions with a single image do not make sense for this pattern
                yield slots, program, pos, neg, [sub_graph, sel_elements]

            if len(scenes_without_both_refs):
                # Question with false as answer
                neg, pos = self._pick_scenes([], negative_scenes, scene_key,
                                             n_min_positive=1, n_max_positive=1)

                if len(pos) + len(neg) >= 2:
                    # questions with a single image do not make sense for this pattern
                    yield slots, program, pos, neg, [sub_graph, sel_elements]
