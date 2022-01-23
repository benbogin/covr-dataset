from copy import deepcopy
from typing import List, Set, Dict

from overrides import overrides

from generator import utils
from generator.patterns.question_pattern import QuestionPattern
from generator.queries.query_builder import SubGraph
from generator.resources import Resources
from generator.utils import scene_info_to_sub_graph, cut_triplet_by_relation


class ChooseRelationPattern(QuestionPattern):
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
        possible_slots = self._get_slots_for_relation_questions(
            sub_graph, negative_scenes, scenes_info
        )

        if sub_graph.others_identical_exist:
            return

        for picked_slots, relevant_negative_scenes, root_no_object, picked_object, distracting_pos in possible_slots:
            subject_program = utils.sub_graph_root_to_ref_program(root_no_object)
            obj_program = utils.sub_graph_root_to_ref_program(picked_object)
            program = self._pattern['program']
            program = utils.merge_ref_program(program, subject_program, "REF_1")
            program = utils.merge_ref_program(program, obj_program, "OBJ")

            neg, pos = self._pick_scenes(positive_scenes, relevant_negative_scenes, scene_key,
                                         n_min_positive=1, n_max_positive=1)

            neg_scenes_info = [(s_id, scenes_info.get(s_id, {})) for s_id in neg]
            ref_elements = [scene_info_to_sub_graph(nodes) for s, nodes in neg_scenes_info if s != scene_key]

            # pick all other relations in accompanied images
            valid_relations = [(t1, t2) for sg in ref_elements for t1, t2
                             in zip(*sg.get_leaf_triplets(include_prepositions=True))]
            # keep only relations that share the same position in the graph structure
            valid_relations = [tuple[1] for tuple, tuple_indices in valid_relations if tuple_indices[1] == distracting_pos]
            # avoid relations with the same name
            valid_relations = [vr for vr in valid_relations if vr.name != picked_slots['REL_1']]
            # avoid specific relations
            valid_relations = [vr for vr in valid_relations if vr.name not in Resources.ignore['relations']]
            # keep only relations that contradicts the original relation
            valid_relations = [vr for vr in valid_relations if picked_slots['REL_1'] in Resources.contradicting_relations[vr.name]]
            if not valid_relations:
                return
            valid_relations = list(set([r.name for r in valid_relations]))

            distracting_relation = picked_slots['REL_NEG'] = self._random.choice(valid_relations)

            ref_query_2_elements = deepcopy(sub_graph)
            ref_query_2_elements[distracting_pos].name = ref_text
            sub_graphs = [sub_graph, ref_query_2_elements]

            shuffle_ref_1_ref_2 = self._random.choice([True, False])
            if shuffle_ref_1_ref_2:
                picked_slots['REL_NEG'], picked_slots['REL_1'] = picked_slots['REL_1'], picked_slots['REL_NEG']

            yield picked_slots, program, pos, neg, sub_graphs

            # create another question that will yield a different answer.
            # first, pick images that have this distracting relation

            neg_scenes_info = [(s_id, scenes_info.get(s_id, {})) for s_id in negative_scenes.get('1_relation', [])]
            neg_scenes_elements = [(s, scene_info_to_sub_graph(nodes)) for s, nodes in neg_scenes_info if s != scene_key]

            # pick all other relations in accompanied images
            neg_scenes_relations = [(s, leaf_triplet) for (s, sg) in neg_scenes_elements for leaf_triplet, _
                               in zip(*sg.get_leaf_triplets(include_prepositions=True))]

            images_with_distracting_relation = [s for (s, leaf_triplet) in neg_scenes_relations
                                                if leaf_triplet[1].name == distracting_relation]
            if not images_with_distracting_relation:
                continue

            picked_image = self._random.choice(images_with_distracting_relation)
            neg, pos = self._pick_scenes([picked_image], relevant_negative_scenes, picked_image,
                                         n_min_positive=1, n_max_positive=1)

            yield picked_slots, program, pos, neg, [sub_graph]

    @staticmethod
    def _get_slots_for_relation_questions(sub_graph, negative_scenes, scenes_info):
        if len(sub_graph) <= 1:
            return []
        if any(node.name in ['of'] for node in sub_graph):
            return []

        # take similar scenes where the object of the main subject would be different
        relevant_negative_scenes = {k: v for k, v in negative_scenes.items() if 'neg_relation' in k}

        output = []
        for neg_key in relevant_negative_scenes:
            slots = {}

            sub_graph1 = deepcopy(sub_graph)
            distract_rel_graph_position = int(neg_key.split('_')[-1])
            leaf_triplets = sub_graph1.get_leaf_triplets(include_prepositions=True)
            relevant_triplet = [(triplet, indices) for triplet, indices in zip(*leaf_triplets)
                                if indices[1] == distract_rel_graph_position]
            assert len(relevant_triplet) == 1
            relevant_triplet, relevant_triplet_indices = relevant_triplet[0]

            cut_triplet_by_relation(relevant_triplet[0], relevant_triplet[1])

            obj = relevant_triplet[2]
            obj.backward_relation = None

            slots['REF_1'] = utils.sub_graph_root_to_ref_text(relevant_triplet[0], determiner="the/a")
            slots['REL_1'] = relevant_triplet[1].name
            slots['OBJ'] = utils.sub_graph_root_to_ref_text(obj)

            output.append((slots, relevant_negative_scenes, relevant_triplet[0], obj, distract_rel_graph_position))

        return output
