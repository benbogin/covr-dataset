import os

import re
from collections import defaultdict

import yaml
from srsly import msgpack


def _convert_annotated_pairs_to_dict(annotated_groups, key, directed=False):
    d = defaultdict(set)
    for relation_group in annotated_groups[key]:
        relations = relation_group.split(',')
        rels_to_add = {relations[0]} if directed else relations
        for rel in rels_to_add:
            d[rel].update(set([r for r in relations if r != rel]))
    return d


class Resources:
    @staticmethod
    def load(base_path='.',
             questions_yaml='questions.yaml',
             load_tiny_distract_yaml: bool = False
             ):
        Resources.base_path = base_path
        resources_path = os.path.join(base_path, 'resources')
        Resources.question_patterns = yaml.load(open(f'{resources_path}/{questions_yaml}', 'rt'), Loader=yaml.FullLoader)['questions']

        for i, p in enumerate(Resources.question_patterns):
            p['placeholders'] = set([ph[1:-1] for ph in re.findall('{[^\s]+}', p['text'])])
            p['pattern_index'] = i

        Resources.ignore = yaml.load(open(f'{resources_path}/ignore.yaml', 'rt'), Loader=yaml.FullLoader)
        Resources.ignore['nouns'] = set(Resources.ignore['nouns'])
        Resources.ignore['relations'] = set(Resources.ignore['relations'])
        Resources.ignore['pairs'] = set(Resources.ignore['pairs'])

        ontology = yaml.load(open(f'{resources_path}/ontology.yaml', 'rt'), Loader=yaml.FullLoader)

        # dictionary mapping from group of attributes (e.g. color) to the attribute (e.g. yellow). Assumes groups are
        # mutually exclusive
        Resources.attributes_group_by_name = {}
        for group_name, values in ontology['attributes'].items():
            for value in values:
                Resources.attributes_group_by_name[value] = group_name

        groups_yaml = "adversarial_groups_short" if load_tiny_distract_yaml else "adversarial_groups"
        annotated_groups = Resources.load_yaml_cached(f'{resources_path}/{groups_yaml}.yaml')
        Resources.contradicting_relations = _convert_annotated_pairs_to_dict(annotated_groups, 'disjoint_predicted_relations', directed=True)
        Resources.not_disjoint_relations = _convert_annotated_pairs_to_dict(annotated_groups, 'not_disjoint_relations', directed=True)
        Resources.contradicting_attributes = _convert_annotated_pairs_to_dict(annotated_groups, 'contradicting_predicted_attributes')
        Resources.entailing_attributes = _convert_annotated_pairs_to_dict(annotated_groups, 'entailing_attributes')
        Resources.entailing_nouns = _convert_annotated_pairs_to_dict(annotated_groups, 'entailing_predicted_nouns')

    @staticmethod
    def load_yaml_cached(yaml_path):
        """
        Since YAML loading is mostly slow, we save and load a msgpack file so that we only load the yaml file once.
        Make sure to remove the msgpack file if ou make any changes to the YAML file.
        """
        pack_file_path = f'{yaml_path}.pack'
        if os.path.exists(pack_file_path):
            return msgpack.load(open(pack_file_path, 'rb'), encoding="utf-8")
        yaml_contents = yaml.load(open(yaml_path, 'rt'), Loader=yaml.FullLoader)
        msgpack.dump(yaml_contents, open(pack_file_path, 'wb'))
        return yaml_contents
