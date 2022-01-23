from copy import deepcopy

from generator.queries.data_classes import QueryNode, QueryRelationship
from generator.queries.sub_graph import SubGraph
from generator.resources import Resources
from generator.scene_reader import SceneReader
from generator.utils import cut_triplet_by_target_index


class DistractorQueries:
    """
    This class creates "distracting" queries by getting a positive query, e.g. "man holding ball" and creating queries
    that will fetch contradicting graphs with e.g. "woman holding ball", "man throwing ball" etc.
    """
    def __init__(self, scene_reader: SceneReader):
        self._scene_reader = scene_reader

    def get_negative_queries(self, sub_graph: SubGraph):
        negative_queries = []

        filter_out_graph = deepcopy(sub_graph)

        # limiting due to performance issues with this query
        if sub_graph.multi_count and sub_graph.multi_count > 1:
            sub_graph = deepcopy(sub_graph)
            sub_graph.multi_count = 1
            negative_queries.append({'query': sub_graph, 'e_id': 0, 'source': 'no_mult',
                                     'filter_out': None})
            sub_graph = deepcopy(sub_graph)
            sub_graph.multi_count = min(sub_graph.multi_count, 2)

        for i, element in enumerate(sub_graph):
            if type(element) is QueryNode:
                for attribute in element.attributes:
                    if attribute not in Resources.contradicting_attributes or not Resources.contradicting_attributes[attribute]:
                        continue

                    modified_graph = deepcopy(sub_graph)
                    modified_graph[i].attributes = Resources.contradicting_attributes[attribute]

                    filter_out_graph[i].attributes = set(Resources.entailing_attributes[attribute])
                    filter_out_graph[i].attributes.add(attribute)
                    filter_out_graph[i].original_attribute = attribute

                    negative_queries.append({'query': modified_graph, 'e_id': i, 'source': 'attr',
                                             'filter_out': filter_out_graph})

                modified_graph = deepcopy(sub_graph)
                modified_graph[i].name = None
                modified_graph[i].original_name = None

                filter_out_graph[i].name = set(Resources.entailing_nouns[element.name])
                filter_out_graph[i].name.add(element.name)
                filter_out_graph[i].original_name = element.name
                negative_queries.append({'query': modified_graph, 'e_id': i, 'source': 'name',
                                         'filter_out': filter_out_graph})

            elif type(element) is QueryRelationship:
                if element.backward_relation is not None:
                    # skip prepositions
                    continue
                relation_name = element.name
                relation_distractors = Resources.contradicting_relations.get(relation_name)
                if not relation_distractors:
                    continue

                modified_graph = deepcopy(sub_graph)
                modified_graph[i].name = relation_distractors

                relations_exclude = set(Resources.not_disjoint_relations[relation_name])
                relations_exclude.add(relation_name)

                source = filter_out_graph[i].source
                target = filter_out_graph[i].target
                filter_out_graph[i].name = relations_exclude
                filter_out_graph[i].original_name = relation_name
                filter_out_graph[i].char_symbol = source.char_symbol + target.char_symbol

                negative_queries.append({'query': modified_graph, 'e_id': i, 'source': 'relation',
                                         'filter_out': filter_out_graph})

        added_queries = []
        added_queries += self.get_negative_attribute_queries_for_main_object(
            sub_graph, negative_queries, filter_out_graph
        )
        added_queries += self.get_negative_object_queries(
            sub_graph, negative_queries, filter_out_graph
        )
        added_queries += self.get_negative_relation_queries(
            sub_graph, negative_queries, filter_out_graph
        )
        negative_queries += added_queries

        return negative_queries

    @staticmethod
    def get_negative_attribute_queries_for_main_object(sub_graph, negative_queries, filter_out):
        # We want to have distractors for questions about attributes of a subject.
        # For example for the question "What is the color of (white) watch on hand?" we want the
        # regular negatives of things that are not a watch but on a hand etc, but this time we also want these to not
        # be white, to make a proper distractor

        main_object_attribute = list(sub_graph.root.attributes)[0] if sub_graph.root.attributes else None
        negative_attributes_for_main_object = Resources.contradicting_attributes.get(main_object_attribute)

        added_queries = []
        if not negative_attributes_for_main_object:
            return added_queries

        new_filter_out = deepcopy(filter_out)
        new_filter_out.root.attributes = set()

        for nq in negative_queries:
            if nq['e_id'] == 0 and nq['source'] == 'attr':
                continue
            modified_graph = deepcopy(nq['query'])
            modified_graph.root.attributes = negative_attributes_for_main_object
            added_queries.append({'query': modified_graph, 'e_id': nq['e_id'],
                                  'source': nq['source'] + '_' + 'neg_attr_main_object',
                                  'filter_out': new_filter_out})
        return added_queries

    def get_negative_object_queries(self, sub_graph, negative_queries, filter_out):
        # We want to have distractors for questions about an object in a sentence-relation-object structure.
        # For example for the question "What is the silver fork on top of?" where the answer is a table, we want
        # negatives with plastic forks on top of something other than a table

        if len(sub_graph) <= 1:
            return []
        if any(type(node) is QueryNode and node.attributes for node in sub_graph[1:]):
            return []
        if any(type(node) is QueryRelationship and node.name == "of" for node in sub_graph):
            return []

        added_queries = []

        leaf_triplets, leaf_triplets_indices = sub_graph.get_leaf_triplets(include_prepositions=True)

        # if multiple relations are repeated, skip these, since they're not good for "query_object" questions
        unique_relations = set(triplet[1].name for triplet in leaf_triplets)
        if len(unique_relations) < len(leaf_triplets):
            return []

        for triplet, triplet_indices in zip(leaf_triplets, leaf_triplets_indices):
            object_name = triplet[2].name
            entailing_nouns = set(Resources.entailing_nouns[object_name])
            entailing_nouns.add(object_name)

            # we only want to consider objects that appeared somewhere between these two objects
            relevant_objects = self._scene_reader.available_objects[(triplet[0].name, triplet[1].name)]
            relevant_objects = relevant_objects - entailing_nouns

            if not relevant_objects:
                continue

            # filter out two sub-graphs, one that has subject, and one that has object
            new_filter_out = deepcopy(filter_out)
            new_filter_out[triplet_indices[2]].name = entailing_nouns
            new_filter_out[triplet_indices[2]].original_name = object_name
            cut_triplet_by_target_index(new_filter_out, source_index=triplet_indices[0], target_index=triplet_indices[2])
            new_filter_out = SubGraph(new_filter_out.root)

            object_node_id = triplet_indices[2]

            # for the subject and relation negatives, add queries with different object
            for nq in negative_queries:
                if nq['e_id'] == object_node_id:
                    continue
                if 'name' not in nq['source']:
                    continue
                modified_graph = deepcopy(nq['query'])
                modified_graph[object_node_id].name = relevant_objects
                added_queries.append({'query': modified_graph, 'e_id': nq['e_id'],
                                      'source': nq['source'] + '_neg_object_' + str(object_node_id),
                                      'filter_out': new_filter_out})
        return added_queries

    def get_negative_relation_queries(self, sub_graph, negative_queries, filter_out):
        # We want to have distractors for questions about a relation in a sentence-relation-object structure.
        # For example for the question "Is the standing man wearing or holding a shirt?" where the answer is wearing,
        # we want negatives with a sitting man holding a shirt

        if len(sub_graph) <= 1:
            return []
        if any(type(node) is QueryRelationship and node.name == "of" for node in sub_graph):
            return []

        leaf_triplets, leaf_triplets_indices = sub_graph.get_leaf_triplets(include_prepositions=True)

        # if multiple relations are repeated, skip these, since they're not good for "choose_rel" questions
        unique_relations = set(triplet[1].name for triplet in leaf_triplets)
        if len(unique_relations) < len(leaf_triplets):
            return []

        added_queries = []
        for triplet, triplet_indices in zip(leaf_triplets, leaf_triplets_indices):
            relation_name = triplet[1].name
            negative_relations = Resources.contradicting_relations[relation_name]

            # we only want to consider relations that appeared somewhere between these two objects
            relevant_relations = self._scene_reader.available_relations[(triplet[0].name, triplet[2].name)]
            relevant_relations = relevant_relations.intersection(negative_relations)

            if not relevant_relations:
                continue

            filter_out_query_neg = deepcopy(filter_out)
            filter_out_query_neg.root.relations = []
            filter_out_query_neg = SubGraph(filter_out_query_neg.root)

            rel_node_id = triplet_indices[1]

            # for the subject and object negatives, add queries with different relation
            for nq in negative_queries:
                if nq['e_id'] not in [0]:
                    continue
                modified_graph = deepcopy(nq['query'])
                modified_graph[triplet_indices[1]].name = relevant_relations
                added_queries.append({'query': modified_graph, 'e_id': nq['e_id'],
                                      'source': nq['source'] + '_' + 'neg_relation_' + str(rel_node_id),
                                      'filter_out': filter_out_query_neg})
        return added_queries
