from collections import defaultdict
from typing import List, Set, Dict

from overrides import overrides

from generator import utils
from generator.patterns.question_pattern import QuestionPattern
from generator.queries.query_builder import SubGraph
from generator.utils import fill_program_slots


class CountRefGroupByPattern(QuestionPattern):
    # some nouns that are hard to count which we saw that were making troubles...
    nouns_to_ignore = {'letter', 'fence', 'cable', 'topping', 'leave', 'leaf', 'broccoli', 'window',
                       'mountain', 'cloud', 'tree', 'leg', 'sauce', 'wire', 'stick', 'word', 'curtain', 'bush',
                       'hair', 'graffiti', 'floor', 'water'}

    def __init__(self, *args, **kwargs):
        kwargs['allow_ref_no_relations'] = True
        super().__init__(*args, **kwargs)

        self._is_binary_question = self._pattern['program'][-1]['operation'] != 'count'

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
        if not sub_graph.multi_count or sub_graph.multi_count <= 1:
            return

        if sub_graph.others_identical_exist:
            return

        if any(sub_graph.root.name.startswith(noun_to_ignore) for noun_to_ignore in self.nouns_to_ignore):
            return

        verified_positives = []

        for pos_scene_key in positive_scenes:
            # make sure candidates negative images do not contain more objects than annotated
            # other_exist, _ = self._lxmert_scene_graph_verifier.check_other_identical_sub_graphs_exist_from_cache(sub_graph, pos_scene_key)
            # if not other_exist or pos_scene_key == scene_key:
            #     verified_positives.append(pos_scene_key)
            verified_positives.append(pos_scene_key)

        verified_negatives = defaultdict(list)
        for neg_key, neg_scenes in negative_scenes.items():
            for neg_scene in neg_scenes:
                # make sure candidates negative images do not contain more objects than annotated
                # other_exist, _ = self._lxmert_scene_graph_verifier.check_other_identical_sub_graphs_exist_from_cache(sub_graph, neg_scene)
                # if not other_exist:
                #     verified_negatives[neg_key].append(neg_scene)
                verified_negatives[neg_key].append(neg_scene)

        neg1, pos1 = self._pick_scenes(verified_positives, verified_negatives, scene_key, n_max_positive=3)

        program = utils.merge_ref_program(self._pattern['program'], ref_program, "REF_1")

        slots = {
            "REF_1": utils.sub_graph_root_to_ref_text(sub_graph.root, determiner="",
                                                      make_plural=sub_graph.multi_count != 1),
            "NUM_POS_GROUP_BY": sub_graph.multi_count,
            "KEEP_IF_VALUES_COUNT_QUANTIFIER": "keep_if_values_count_eq"
        }

        # pick two sets of images/slots for both true/false answers
        real_count_1 = self._compute_count_answer(pos1, neg1, program, slots)
        real_count_2 = real_count_1

        n_available_negative_scenes = len(set([s_id for s_set in verified_negatives.values() for s_id in s_set]))
        if n_available_negative_scenes < 2:
            return

        tries, limit = 0, 15
        while real_count_1 == real_count_2:
            if tries == limit:
                # print("count_ref_group_by reached maximum tries limit")
                return
            neg2, pos2 = self._pick_scenes(verified_positives, verified_negatives, scene_key, n_min_positive=0)
            real_count_2 = self._compute_count_answer(pos2, neg2, program, slots)
            tries += 1

        if self._is_binary_question:
            available_quantifers = ["eq"]
            min_count = min(real_count_1, real_count_2)
            max_count = max(real_count_1, real_count_2)

            if min_count > 0:
                # no point of adding "at most 0" questions
                available_quantifers.append("leq")
            if max_count < 5:
                # no point of adding "at least 5" questions
                available_quantifers.append("geq")
            num_quantifier = self._random.choice(available_quantifers)
            slots["NUM_QUANTIFIER_MODULE"] = num_quantifier
            slots["NUM_POS"] = real_count_1

            if num_quantifier == "geq":
                slots["NUM_POS"] = self._random.randint(min_count + 1, max_count)
            elif num_quantifier == "leq":
                slots["NUM_POS"] = self._random.randint(min_count, max_count - 1)

            quantifier_text = {"eq": "", "leq": "at most", "geq": "at least"}
            slots["QUANTIFIER"] = quantifier_text[num_quantifier]

            slots["BE"] = "is" if slots["NUM_POS"] == 1 else "are"
            slots["IMAGES"] = "image" if slots["NUM_POS"] == 1 else "images"
        else:
            num_group_by = sub_graph.multi_count

            possible_slots = []

            # check which quantifiers are valid to create 2 identical questions with the selected different images that
            # will yield a different answer.
            for qnt in ["keep_if_values_count_eq", "keep_if_values_count_geq", "keep_if_values_count_leq"]:
                possible_slot = dict(slots)
                possible_slot["KEEP_IF_VALUES_COUNT_QUANTIFIER"] = qnt
                if qnt == "keep_if_values_count_geq":
                    possible_slot["NUM_POS_GROUP_BY"] = self._random.randint(1, num_group_by)
                if qnt == "keep_if_values_count_leq":
                    possible_slot["NUM_POS_GROUP_BY"] = self._random.randint(num_group_by, 5)
                quantifier_text = {"keep_if_values_count_eq": "exactly", "keep_if_values_count_leq": "at most", "keep_if_values_count_geq": "at least"}
                possible_slot["QUANTIFIER"] = quantifier_text[qnt]
                ans1 = self._compute_count_answer(pos1, neg1, program, possible_slot)
                ans2 = self._compute_count_answer(pos2, neg2, program, possible_slot)
                if ans1 != ans2:
                    possible_slots.append(possible_slot)
            selected_slots = self._random.choice(possible_slots)
            slots.update(selected_slots)

        # count question
        yield slots, program, pos1, neg1, [sub_graph]
        yield slots, program, pos2, neg2, [sub_graph]

    def _compute_count_answer(self, pos, neg, program, slots):
        all_scenes = [self._scene_reader.get_formatted_scenes(k) for k in pos + neg]
        objects_from_all_scenes = {ok: ov for s in all_scenes for ok, ov in s['objects'].items()}
        program = fill_program_slots(program, slots)

        if self._is_binary_question:
            program = program[:-1]

        answer = self._executor.run(program, objects_from_all_scenes)  # run only up to the "count" operation
        assert type(answer) is int
        return answer

    @staticmethod
    def does_pattern_support_multi_instances() -> bool:
        return True
