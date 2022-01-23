from typing import List, Set, Dict

from overrides import overrides

from generator import utils
from generator.patterns.count_ref_group_by import CountRefGroupByPattern
from generator.patterns.question_pattern import QuestionPattern
from generator.queries.query_builder import SubGraph


class CountRefPattern(QuestionPattern):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._is_binary_question = self._pattern['program'][-1]['operation'] != 'count'

        self._count_images = any(program_row['operation'] == "unique_images" for program_row in self._pattern['program'])

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
        if any(sub_graph.root.name.startswith(noun_to_ignore) for noun_to_ignore in CountRefGroupByPattern.nouns_to_ignore):
            return

        if not self._count_images:
            # if we count objects rather than images, we want to make sure we use verified
            # images where number of objects is likely to match number of annotated objects
            if sub_graph.others_identical_exist:
                return

            verified_positives = []

            for pos_scene_key in positive_scenes:
                # make sure candidates negative images do not contain more objects than annotated
                # other_exist, _ = self._lxmert_scene_graph_verifier.check_other_identical_sub_graphs_exist_from_cache(sub_graph, pos_scene_key)
                # if not other_exist or pos_scene_key == scene_key:
                #     verified_positives.append(pos_scene_key)
                verified_positives.append(pos_scene_key)
        else:
            verified_positives = positive_scenes

        neg1, pos1 = self._pick_scenes(verified_positives, negative_scenes, scene_key)

        program = utils.merge_ref_program(self._pattern['program'], ref_program, "REF_1")

        slots = {"REF_1": ref_text}

        n_available_negative_scenes = len(set([s_id for s_set in negative_scenes.values() for s_id in s_set]))
        if n_available_negative_scenes < 2:
            return

        # pick two sets of images/slots for both true/false answers
        real_count_1 = self._compute_count_answer(pos1, neg1, program)
        real_count_2 = real_count_1

        tries, limit = 0, 5
        while real_count_1 == real_count_2 or real_count_2 >= 6:
            if tries == limit:
                return
            neg2, pos2 = self._pick_scenes(verified_positives, negative_scenes, scene_key, n_min_positive=0)
            real_count_2 = self._compute_count_answer(pos2, neg2, program)
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

            if 'IMAGES' not in self._pattern['placeholders']:
                slots['REF_1'] = utils.sub_graph_root_to_ref_text(sub_graph.root, determiner="",
                                                                  make_plural=slots["NUM_POS"] != 1)

            yield slots, program, pos1, neg1, [sub_graph]
            yield slots, program, pos2, neg2, [sub_graph]

        else:
            # count question
            slots['REF_1'] = utils.sub_graph_root_to_ref_text(sub_graph.root,
                                                              add_that=self._count_images,
                                                              make_plural=True)
            yield slots, program, pos1, neg1, [sub_graph]
            yield slots, program, pos2, neg2, [sub_graph]

    def _compute_count_answer(self, pos, neg, program):
        all_scenes = [self._scene_reader.get_formatted_scenes(k) for k in pos + neg]
        objects_from_all_scenes = {ok: ov for s in all_scenes for ok, ov in s['objects'].items()}

        if program[-1]['operation'] != 'count':
            program = program[:-1]

        answer = self._executor.run(program, objects_from_all_scenes)  # run only up to the "count" operation
        return answer
