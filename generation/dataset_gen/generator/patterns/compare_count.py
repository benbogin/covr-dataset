from collections import defaultdict
from typing import List, Set, Dict

from overrides import overrides

from generator import utils

from generator.patterns.question_pattern import QuestionPattern
from generator.queries.query_builder import SubGraph
from generator.utils import sub_graph_root_to_ref_program, sub_graph_root_to_ref_text, scene_info_to_sub_graph, \
    merge_ref_program


class CompareCountPattern(QuestionPattern):
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
        program = utils.merge_ref_program(self._pattern['program'], ref_program, "REF_1")

        if sub_graph.others_identical_exist:
            return

        verified_positives = list(positive_scenes)
        # for pos_scene_key in positive_scenes:
        #     # make sure candidates positive images do not contain more objects than annotated
        #     other_exist, _ = self._lxmert_scene_graph_verifier.check_other_identical_sub_graphs_exist_from_cache(sub_graph, pos_scene_key)
        #     if not other_exist or pos_scene_key == scene_key:
        #         verified_positives.append(pos_scene_key)

        verified_negatives = defaultdict(list)
        for neg_key, neg_scenes in negative_scenes.items():
            for neg_scene in neg_scenes:
                # make sure candidates negative images do not contain more objects than annotated
                # other_exist, _ = self._lxmert_scene_graph_verifier.check_other_identical_sub_graphs_exist_from_cache(sub_graph, neg_scene)
                # if not other_exist:
                #     verified_negatives[neg_key].append(neg_scene)
                verified_negatives[neg_key].append(neg_scene)

        neg, pos = self._pick_scenes(verified_positives, verified_negatives, scene_key, n_min_negative=1)

        picked_comparative = self._random.choice(["gt", "lt", "eq"])
        if picked_comparative == "gt":
            comparison = "more"
            than = "than"
        elif picked_comparative == "lt":
            comparison = "less"
            than = "than"
        elif picked_comparative == "eq":
            comparison = "the same number of"
            than = "as"
        else:
            raise ValueError

        picked_slots = {
            "REF_1": ref_text,
            "COMPARISON": comparison,
            "COMPARISON_MODULE": picked_comparative,
            "THAN": than
        }

        neg_scenes_info = [(s_id, scenes_info.get(s_id, {})) for s_id in neg]
        negative_elements = [scene_info_to_sub_graph(nodes) for s, nodes in neg_scenes_info]
        sel_sub_graph = self._random.choice(negative_elements)

        self._keep_only_valid_attributes_in_neg_sub_graph(sel_sub_graph, sub_graph)

        make_plural = 'plural' in self._pattern.get('setup', {})
        ref_text_2 = sub_graph_root_to_ref_text(sel_sub_graph.root, determiner=None, make_plural=make_plural)

        picked_slots['REF_2'] = ref_text_2

        if make_plural:
            picked_slots['REF_1'] = sub_graph_root_to_ref_text(sub_graph.root, determiner=None, make_plural=True)

        ref_2_program = sub_graph_root_to_ref_program(sel_sub_graph.root)
        program = merge_ref_program(program, ref_2_program, "REF_2")

        with_ref_2, without_ref_2 = self._get_scenes_with_and_without_ref([s for s, _ in neg_scenes_info], ref_2_program)

        pos += with_ref_2
        neg = [n for n in neg if n not in pos]

        if not picked_slots:
            return
        yield picked_slots, program, pos, neg, [sub_graph, sel_sub_graph]
