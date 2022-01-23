from copy import deepcopy
from typing import List, Set, Dict

from overrides import overrides

from generator import utils
from generator.patterns.question_pattern import QuestionPattern
from generator.queries.query_builder import SubGraph
from generator.utils import scene_info_to_sub_graph, sub_graph_root_to_ref_text, cut_triplet_by_target_index, \
    cut_triplet_by_relation


class QueryObjectPattern(QuestionPattern):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # do not show these verbs if the object is not in the question. e.g. "what is the woman wearing?" should be
        # eliminated out, but "what is the woman wearing, helmet or hat?" is ok
        if 'OBJ' in self._pattern['placeholders']:
            self._ignore_verbs_if_no_object = {}
        else:
            self._ignore_verbs_if_no_object = {"wearing", "behind", "in front of", "near", "above"}

        self._is_choose_pattern = 'choose_name' in [pr['operation'] for pr in self._pattern['program']]

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
        possible_slots = self._get_slots_for_object_questions(sub_graph, negative_scenes)

        if sub_graph.others_identical_exist:
            return

        for picked_slots, relevant_negative_scenes, root_no_object, picked_object, distracting_pos in possible_slots:
            ref_program = utils.sub_graph_root_to_ref_program(root_no_object)

            program = utils.merge_ref_program(self._pattern['program'], ref_program, "REF_1_NO_RELATION")

            neg, pos = self._pick_scenes(positive_scenes, relevant_negative_scenes, scene_key,
                                         n_min_positive=1, n_max_positive=1)
            sub_graphs = [sub_graph]

            picked_slots['BE'] = "is" if utils.is_singular(sub_graph[0].name) else "are"

            if 'OBJ_NEG_TEXT' in self._pattern['placeholders']:
                neg_scenes_info = [(s_id, scenes_info.get(s_id, {})) for s_id in neg]
                ref_elements = [scene_info_to_sub_graph(nodes) for s, nodes in neg_scenes_info if s != scene_key]

                valid_objects = [(t1, t2) for sg in ref_elements for t1, t2
                                 in zip(*sg.get_leaf_triplets(include_prepositions=True))]
                if sub_graph.root.relations and sub_graph.root.relations[0].prepositions:
                    valid_objects = [tuple[2] for tuple, tuple_indices in valid_objects if tuple_indices[2] == distracting_pos]
                else:
                    valid_objects = [tuple[2] for tuple, tuple_indices in valid_objects]
                valid_elements = [vo for vo in valid_objects if vo.name != picked_object.name]
                if not valid_elements:
                    return
                ref_1_object = picked_slots['OBJ']
                ref_1_object_text = picked_slots['OBJ_TEXT']
                ref_2_object_root = self._random.choice(valid_elements)
                ref_2_object = ref_2_object_root.name
                ref_2_object_text = sub_graph_root_to_ref_text(ref_2_object_root)

                # add sub graph
                sub_graphs.append(SubGraph(ref_2_object_root))

                shuffle_ref_1_ref_2 = self._random.choice([True, False])
                if shuffle_ref_1_ref_2:
                    ref_1_object_text, ref_2_object_text = ref_2_object_text, ref_1_object_text
                    ref_1_object, ref_2_object = ref_2_object, ref_1_object

                picked_slots['OBJ_TEXT'] = ref_1_object_text
                picked_slots['OBJ_NEG_TEXT'] = ref_2_object_text
                picked_slots['OBJ'] = ref_1_object
                picked_slots['OBJ_NEG'] = ref_2_object

            yield picked_slots, program, pos, neg, sub_graphs

    def _get_slots_for_object_questions(self, sub_graph, negative_scenes):
        if len(sub_graph) <= 1:
            return []
        if any(node.name in ['of'] for node in sub_graph):
            return []

        # take similar scenes where the object of the main subject would be different
        relevant_negative_scenes = {k: v for k, v in negative_scenes.items() if 'neg_object' in k}

        output = []
        for neg_key in relevant_negative_scenes:
            slots = {}

            sub_graph1 = deepcopy(sub_graph)
            distract_object_graph_position = int(neg_key.split('_')[-1])
            leaf_triplets = sub_graph1.get_leaf_triplets(include_prepositions=True)
            relevant_triplet = [(triplet, indices) for triplet, indices in zip(*leaf_triplets)
                                if indices[2] == distract_object_graph_position]
            assert len(relevant_triplet) == 1
            relevant_triplet, relevant_triplet_indices = relevant_triplet[0]

            slots['RELATION'] = relevant_triplet[1].name

            # remove any imsitu preposition hint, e.g. "parachuting:_to" -> "parachuting to"
            slots['RELATION_NAME'] = slots['RELATION'].replace(":_", " ")

            if relevant_triplet[2].attributes:
                continue
            if not self._is_choose_pattern:
                if slots['RELATION_NAME'] in self._ignore_verbs_if_no_object:
                    continue

            cut_triplet_by_relation(relevant_triplet[0], relevant_triplet[1])

            query_elements_object = relevant_triplet[2]
            query_elements_object.backward_relation = None

            slots['REF_1_NO_RELATION'] = sub_graph_root_to_ref_text(relevant_triplet[0], determiner="the/a")
            slots['OBJ_TEXT'] = sub_graph_root_to_ref_text(query_elements_object)
            slots['OBJ'] = query_elements_object.name

            output.append((slots, relevant_negative_scenes, relevant_triplet[0], query_elements_object,
                           distract_object_graph_position))

        return output
