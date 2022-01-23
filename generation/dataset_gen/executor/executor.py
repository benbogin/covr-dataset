"""
based on https://github.com/kexinyi/ns-vqa/blob/master/reason/executors/clevr_executor.py
"""
import types
from collections import defaultdict
from typing import Callable, Dict, List, Union, Set


def predicate(function: Callable) -> Callable:
    """
    This is intended to be used as a decorator when you are implementing your ``DomainLanguage``.
    This marks a function on a ``DomainLanguage`` subclass as a predicate that can be used in the
    language.  See the :class:`DomainLanguage` docstring for an example usage, and for what using
    this does.
    """
    setattr(function, "_is_predicate", True)
    return function


class NonUniqueException(Exception):
    pass


class NonExistentException(Exception):
    pass


class NeitherOfChooseException(Exception):
    pass


class NoCommonAttributeException(Exception):
    pass


class MultipleAttributesForTypeException(Exception):
    pass


class Executor:
    def __init__(self):
        # extract all names of methods

        self._curr_scene_objects = []
        self._curr_scene_objects_dict = {}
        self._modules = {}
        for name in dir(self):
            if isinstance(getattr(self, name), types.MethodType):
                function = getattr(self, name)
                if getattr(function, "_is_predicate", False):
                    self._modules[name] = function

        self.exe_trace = []

    @staticmethod
    def tree_to_prefix(program_tree):
        output = []

        def helper(cur):
            output.append({
                'function': cur['function'],
                'value_inputs': [x for x in cur['value_inputs']],
            })
            for node in cur['inputs']:
                helper(node)

        helper(program_tree)
        return output

    def run(self, program, scene_objects_dict: dict):
        ans, temp = None, None

        scene_objects = list(scene_objects_dict.values())

        self._curr_scene_objects = scene_objects
        self._curr_scene_objects_dict = scene_objects_dict
        self.exe_trace = []
        for program_row in program:
            module_name = program_row['operation']
            if module_name in self._modules:
                module = self._modules[module_name]
                dependencies = [self.exe_trace[di] for di in program_row.get('dependencies', [])]
                ans = module(*dependencies, *program_row.get('arguments', []))
            else:
                raise Exception(f"Unknown module: {module_name}")
            self.exe_trace.append(ans)
        return ans

    @predicate
    def scene(self, _=None):
        return self._curr_scene_objects

    @predicate
    def all(self, input_objects, condition_program):
        return self.quantifier(input_objects, condition_program, lambda x: all(x))

    @predicate
    def some(self, input_objects, condition_program):
        return self.quantifier(input_objects, condition_program, lambda x: any(x))

    @predicate
    def none(self, input_objects, condition_program):
        return self.quantifier(input_objects, condition_program, lambda x: not any(x))

    def quantifier(self, input_objects, condition_program, quantifier_fn):
        res = self.run(condition_program, self._curr_scene_objects_dict)
        objects_with_condition = set(o['object_key'] for o in res)

        objects_with_condition_in_res = [obj['object_key'] in objects_with_condition for obj in input_objects]

        return quantifier_fn(objects_with_condition_in_res)

    @predicate
    def all_same(self, input_objects, attribute_type):
        if len(input_objects) == 0:
            return True

        attribute_value = input_objects[0]['attributes_by_group'].get(attribute_type, set())

        return all([obj['attributes_by_group'].get(attribute_type, set()).intersection(attribute_value) for obj in input_objects])

    @predicate
    def query_common_attribute(self, input_object1, input_object2):
        for attr, val in input_object1.get('attributes_by_group', {}).items():
            if input_object2.get('attributes_by_group', {}).get(attr, set()).intersection(val):
                return attr
        raise NoCommonAttributeException

    @predicate
    def query_name(self, input_object1):
        return input_object1.get('name')

    @predicate
    def find(self, object_names):
        if type(object_names) is set:
            object_names = set(object_names)
        if type(object_names) is str:
            object_names = {object_names}
        if not object_names:
            return self._curr_scene_objects
        return [obj for obj in self._curr_scene_objects if obj['name'] in object_names]

    @predicate
    def count(self, input_objects):
        return len(input_objects)

    @predicate
    def exists(self, input_objects):
        return len(input_objects) > 0

    @predicate
    def logic_or(self, bool1, bool2):
        return bool1 or bool2

    @predicate
    def logic_and(self, bool1, bool2):
        return bool1 and bool2

    @predicate
    def eq(self, num1, num2):
        return num1 == num2

    @predicate
    def gt(self, num1, num2):
        return num1 > num2

    @predicate
    def geq(self, num1, num2):
        return num1 >= num2

    @predicate
    def lt(self, num1, num2):
        return num1 < num2

    @predicate
    def leq(self, num1, num2):
        return num1 <= num2

    @predicate
    def gt_half(self, num1, num2):
        return num1 > num2 / 2

    @predicate
    def filter(self, input_objects, attribute_name: Union[str, Set[str]]):
        """keeps object if has *any* of the given attributes"""
        if type(attribute_name) is str:
            attribute_name = {attribute_name}
        else:
            attribute_name = set(attribute_name)
        return [obj for obj in input_objects if attribute_name.intersection(obj['attributes'])]

    @predicate
    def unique(self, input_objects):
        if len(input_objects) > 1:
            raise NonUniqueException()
        if len(input_objects) == 0:
            raise NonExistentException()
        return input_objects[0]

    @predicate
    def assert_unique(self, input_objects):
        if len(input_objects) > 1:
            raise NonUniqueException()
        if len(input_objects) == 0:
            raise NonExistentException()
        return input_objects

    @predicate
    def unique_images(self, input_objects):
        return list(set([o['scene_id'] for o in input_objects]))

    @predicate
    def group_by_images(self, input_objects) -> Dict:
        group_by_output = defaultdict(list)
        for o in input_objects:
            group_by_output[o['scene_id']].append(o)
        return group_by_output

    @predicate
    def keep_if_values_count_eq(self, group_by_output: Dict, values_count: int) -> Dict:
        return {scene: objects for scene, objects in group_by_output.items() if len(objects) == values_count}

    @predicate
    def keep_if_values_count_geq(self, group_by_output: Dict, values_count: int) -> Dict:
        return {scene: objects for scene, objects in group_by_output.items() if len(objects) >= values_count}

    @predicate
    def keep_if_values_count_leq(self, group_by_output: Dict, values_count: int) -> Dict:
        return {scene: objects for scene, objects in group_by_output.items() if len(objects) <= values_count}

    @predicate
    def keys(self, group_by_output: Dict) -> List:
        return list(group_by_output.keys())

    @predicate
    def query_attr(self, input_object, attr):
        attributes_for_group = input_object['attributes_by_group'][attr]
        if not attributes_for_group:
            return None
        if len(attributes_for_group) > 1:
            raise MultipleAttributesForTypeException()
        return list(attributes_for_group)[0]

    @predicate
    def choose_attr(self, input_object, attr1, attr2):
        if attr1 in input_object['attributes']:
            return attr1
        elif attr2 in input_object['attributes']:
            return attr2

        raise NeitherOfChooseException()

    @predicate
    def choose_name(self, input_object, name1, name2):
        if name1 == input_object['name']:
            return name1
        elif name2 == input_object['name']:
            return name2

        raise NeitherOfChooseException()

    @predicate
    def choose_relation(self, input_relation, relation1, relation2):
        if relation1 == input_relation:
            return relation1
        elif relation2 == input_relation:
            return relation2

        raise NeitherOfChooseException()

    @predicate
    def verify_attr(self, input_object, attr):
        return attr in input_object['attributes']

    @predicate
    def relation_between_nouns(self, input_object1, input_objects2):
        IGNORE_RELATIONS = {'with'}
        set2_objects = set([o['object_key'] for o in input_objects2])
        return [rel for rel in input_object1['relations'] if rel['object'] in set2_objects
                if rel['name'] not in IGNORE_RELATIONS]

    @predicate
    def with_relation_object(self, input_objects1, input_objects2=None, relation_filter=None):
        return self.with_relation(input_objects1, input_objects2, relation_filter, return_object=True)

    @predicate
    def with_relation(self, input_objects1, input_objects2=None, relation_filter=None, return_object=False):
        if relation_filter:
            if type(relation_filter) is set:
                relation_filter = set(relation_filter)
            else:
                relation_filter = {relation_filter}

        objects_with_name_and_relation = set()
        input_objects2_set = set([o['object_key'] for o in input_objects2]) if input_objects2 else set()
        for obj in input_objects1:
            to_iterate = list(obj['relations'])
            if obj['relations'] and obj['relations'][0].get('prepositions'):
                to_iterate += obj['relations'][0]['prepositions']
            for relation in to_iterate:
                other_object_key = relation['object']
                relation_name = relation['name']
                if relation_filter and relation_name not in relation_filter:
                    continue
                if input_objects2 is not None and other_object_key not in input_objects2_set:
                    continue
                if return_object:
                    objects_with_name_and_relation.add(other_object_key)
                else:
                    objects_with_name_and_relation.add(obj['object_key'])
        return [self._curr_scene_objects_dict[o] for o in objects_with_name_and_relation]
