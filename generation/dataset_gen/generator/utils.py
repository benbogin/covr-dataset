from collections import Counter
from typing import List, Dict
from copy import deepcopy, copy

import inflect

from neo4j import Record
from neo4j.graph import Node

from generator.queries.data_classes import QueryNode, QueryRelationship, QueryElement
from generator.queries.sub_graph import SubGraph

inflect = inflect.engine()


def is_singular(word):
    return inflect.singular_noun(word) is False


def get_plural(word):
    fixes = {"people": "people"}
    if word in fixes:
        return fixes[word]
    return inflect.plural(word) or word


def add_program_b_to_a(program_a: List, program_b: List, offset: int = 0):
    curr_lines = len(program_a)
    other_program_lines = deepcopy(program_b)
    for line in other_program_lines:
        if 'dependencies' in line:
            line['dependencies'] = [d + curr_lines + offset for d in line['dependencies']]
    program_a += other_program_lines


def merge_ref_program(question_program, ref_program, placeholder):
    placeholder_index = [i for (i, op) in enumerate(question_program)
                         if op.get('arguments') == placeholder]
    assert len(placeholder_index) == 1
    placeholder_index = placeholder_index[0]
    ref_program = deepcopy(ref_program)
    program = question_program[:placeholder_index]

    for line in ref_program:
        if 'dependencies' in line:
            line['dependencies'] = [d + placeholder_index for d in line['dependencies']]

    program += ref_program

    rest_of_program = deepcopy(question_program[placeholder_index+1:])

    for line in rest_of_program:
        if 'dependencies' in line:
            deps = []
            for d in line['dependencies']:
                deps.append(d if d < placeholder_index else d + len(ref_program) - 1)
            line['dependencies'] = deps

    program += rest_of_program

    return program


def scene_info_to_sub_graph(root_dict, depth=0):
    root = QueryNode(name=root_dict["name"], attributes=root_dict["attributes"] or set(), relations=[])
    for relation_dict in root_dict.get("relations", []):
        target = scene_info_to_sub_graph(relation_dict['target'], depth=depth+1)
        root.relations.append(QueryRelationship(name=relation_dict['name'], source=root, target=target))

        for pp_dict in relation_dict.get("prepositions", []):
            if not root.relations[-1].prepositions:
                root.relations[-1].prepositions = []
            target = scene_info_to_sub_graph(pp_dict['target'], depth=depth + 1)
            source_relation = root.relations[-1]
            root.relations[-1].prepositions.append(
                QueryRelationship(name=pp_dict['name'], source=source_relation, target=target)
            )

    if depth == 0:
        return SubGraph(root)
    else:
        return root


def query_element_to_ref_text(element: QueryElement):
    parts = []
    if type(element) is QueryNode:
        if element.attributes:
            parts += element.attributes
    parts.append(element.name)
    return " ".join(parts)


def sub_graph_root_to_ref_text_simple(query_root, take_nodes_original_name=False):
    return sub_graph_root_to_ref_text(query_root, determiner="", verb="", add_that=False, add_parenthesis=False,
                                      take_nodes_original_name=True)


def sub_graph_root_to_ref_text(query_root, determiner="a", make_plural=False, verb=None, add_that=True, depth=0,
                               add_parenthesis=True, take_nodes_original_name=False):
    ref_parts = []

    selector_parts = []
    if depth == 0 and query_root.backward_relation:
        prev_root = copy(query_root.backward_relation.source)
        prev_root.relations = []
        selector_parts = ['with', sub_graph_root_to_ref_text(prev_root, determiner="the"),
                          query_root.backward_relation.name,
                          "them" if make_plural else "it"]

    noun_is_singular = False
    if query_root.name and not query_root.empty:
        noun_name = query_root.name if not take_nodes_original_name else query_root.original_name

        make_plural = make_plural and depth == 0
        if make_plural:
            noun_name = get_plural(noun_name)

        noun_is_singular = is_singular(noun_name) and not make_plural

        ref_parts += get_determiner(determiner, noun_name, noun_is_singular, depth)
        if query_root.attributes:
            if take_nodes_original_name and query_root.original_attribute:
                ref_parts.append(query_root.original_attribute)
            else:
                ref_parts += query_root.attributes

        ref_parts.append(noun_name)

    if selector_parts:
        ref_parts += selector_parts
    if verb is None:
        verb = "is" if noun_is_singular else "are"

    if query_root.relations:
        if query_root.name and not query_root.empty:
            if add_that:
                ref_parts.append("that")

        relations = []
        for i, relation in enumerate(query_root.relations):
            relation_name = relation.name if not take_nodes_original_name or not relation.original_name else relation.original_name

            relation_parts = []

            if verb:
                relation_parts.append(verb)
            relation_parts.append(relation_name)
            if relation.target and relation.target.name:
                target_has_children = bool(relation.target.relations)
                if add_parenthesis and target_has_children:
                    relation_parts.append("(")
                relation_parts.append(sub_graph_root_to_ref_text(
                    relation.target,
                    determiner="a" if determiner else determiner,
                    make_plural=False,
                    verb=None,
                    add_that=add_that,
                    depth=depth+1,
                    take_nodes_original_name=take_nodes_original_name
                ))
                if add_parenthesis and target_has_children:
                    relation_parts.append(")")

            if relation.prepositions:
                for pp in relation.prepositions:
                    pp_name = pp.name if not take_nodes_original_name or not pp.original_name else pp.original_name
                    pp_name = pp_name.split(":_")[1]
                    relation_parts += [pp_name, sub_graph_root_to_ref_text(pp.target, determiner, make_plural,
                                                                           verb, add_that, depth + 1,
                                                                           take_nodes_original_name=take_nodes_original_name)]

            relations.append(" ".join(relation_parts))

        ref_parts.append(" and ".join(relations))
    ref_text = " ".join(ref_parts)
    return ref_text


def get_determiner(required_determiner, noun, noun_is_singular, depth):
    if required_determiner == "a":
        if noun in ["water"]:
            return []
        if noun_is_singular:
            return [inflect.a(noun).split()[0]]
    elif required_determiner == "the/a":
        if depth == 0:
            return ["the"]
        else:
            return get_determiner("a", noun, noun_is_singular, depth)
    return []


def extract_all_root_to_leaves_paths(query_root, current_path=None, output=None):
    if output is None:
        output = []
    if current_path is None:
        current_path = []
    current_path.append(query_root)

    if query_root.relations:
        for i, relation in enumerate(query_root.relations):
            current_path.append(relation)
            extract_all_root_to_leaves_paths(relation.target, current_path, output)
            current_path = current_path[:-2]

            # imsitu prepositions are stored as a chain in the DB
            if relation.prepositions:
                for pp in relation.prepositions:
                    output[-1].append(pp)
                    output[-1].append(pp.target)
    else:
        output.append(current_path)

    return output


def extract_flattened_elements(query_root):
    output = [query_root]

    if query_root.relations:
        for i, relation in enumerate(query_root.relations):
            output.append(relation)
            output += extract_flattened_elements(relation.target)

            if relation.prepositions:
                for pp in relation.prepositions:
                    output.append(pp)
                    output += extract_flattened_elements(pp.target)

    return output


def extract_elements_triplets(query_root):
    index = -1

    def _extract_elements_triplets(root, output=None):
        nonlocal index

        if output is None:
            output = []

        index += 1
        root_index = index
        if root.relations:
            for i, relation in enumerate(root.relations):
                index += 1
                relation_index = index
                output.append(((root_index, relation_index, relation_index+1), (root, relation, relation.target)))
                _extract_elements_triplets(relation.target, output)
                if relation.prepositions:
                    for pp in relation.prepositions:
                        index += 1
                        pp_index = index
                        output.append(((root_index, pp_index, pp_index + 1), (root, pp, pp.target)))
                        _extract_elements_triplets(pp.target, output)

        return output

    return _extract_elements_triplets(query_root)


def _query_node_ref_program(node: QueryNode, index_start=0):
    noun_name = node.name
    if node.empty:
        node_program = [{"operation": "scene"}]
    else:
        node_program = [{"operation": "find", "arguments": [noun_name]}]
    if node.attributes:
        attributes = list(node.attributes)
        if len(attributes) == 1:
            attributes = attributes[0]
        node_program.append({"operation": "filter",
                             "arguments": [attributes],
                             "dependencies": [index_start + len(node_program) - 1]})
    return node_program


def sub_graph_root_to_ref_program(root: QueryNode, index_start=0, depth=0, with_relation_object=False):
    """
    with_relation_object: if true, relations will return the object and not the subject
    """
    output = []

    if not root:
        return output

    relations_to_iterate = []
    if root.relations:
        relations_to_iterate += root.relations
        for relation in root.relations:
            if relation.prepositions:
                relations_to_iterate += relation.prepositions

    indices = []
    for relation in relations_to_iterate:
        if relation.target:
            output += sub_graph_root_to_ref_program(relation.target, len(output), depth+1, with_relation_object)
            indices.append(len(output) - 1)
        else:
            indices.append(None)

    if root.backward_relation and depth == 0:
        output += sub_graph_root_to_ref_program(root.backward_relation.source, index_start + len(output), depth+1, True)
    else:
        output += _query_node_ref_program(root, index_start=index_start + len(output))
    root_index = len(output)
    last_rel_index = root_index

    for relation_program_index, relation in zip(indices, relations_to_iterate):
        if relation_program_index is not None:
            dependencies = [last_rel_index-1, relation_program_index]
            arguments = [relation.name]
        else:
            dependencies = [root_index]
            arguments = [None, relation.name]
        operation = "with_relation" if not with_relation_object else "with_relation_object"
        output.append({"operation": operation, "arguments": arguments,
                       "dependencies": dependencies})
        last_rel_index = len(output)

    return output


def program_to_string(program: List, index=-1, add_arguments: bool = True):
    line = program[index]
    output = [line['operation']]
    if 'arguments' in line:
        if add_arguments:
            args_out = []
            for argument in line['arguments']:
                if type(argument) is list:
                    args_out.append(program_to_string(argument, add_arguments=add_arguments))
                else:
                    args_out.append(str(argument))
            joined_arguments = ','.join(args_out)
            output.append(f"[{joined_arguments}]")
        else:
            # if the arguments is a program (e.g. in the case of quantifiers), we will still want to include it in the
            # returned string
            if (isinstance(line['arguments'][0], list) and isinstance(line['arguments'][0][0], dict) and
                    line['arguments'][0][0]['operation']):
                nested_program_str = program_to_string(line['arguments'][0], -1, add_arguments=add_arguments)
                output.append(f"[{nested_program_str}]")

    if 'dependencies' in line:
        children = []
        for dep in line['dependencies']:
            children.append(program_to_string(program, dep, add_arguments=add_arguments))
        joined_children = ",".join(children)
        output.append(f"({joined_children})")

    return " ".join(output)


def remove_program_row(program, row_index):
    program = deepcopy(program)
    row_dependencies = program[row_index].get('dependencies', [])
    assert len(row_dependencies) == 1
    row_dependency = row_dependencies[0]

    for row in program[row_index+1:]:
        if 'dependencies' in row:
            new_deps = []
            for dependency in row['dependencies']:
                if dependency < row_index:
                    new_deps.append(dependency)
                if dependency == row_index:
                    new_deps.append(row_dependency)
                else:
                    new_deps.append(dependency-1)
            row['dependencies'] = new_deps
    return program[:row_index] + program[row_index+1:]


def fill_program_slots(program: List[Dict], picked_slots: Dict, make_copy=True):
    if make_copy:
        program = deepcopy(program)
    for command in program:
        if command['operation'] in picked_slots:
            command['operation'] = picked_slots[command['operation']]
        for i, argument in enumerate(command.get('arguments', [])):
            if type(argument) is list and type(argument[0]) is dict:
                fill_program_slots(argument, picked_slots, make_copy=False)
            elif type(argument) is str and argument in picked_slots:
                command['arguments'][i] = picked_slots[argument]

    return program


def neo4j_results_to_sub_graph(neo_elements: Record):
    nodes = {}
    in_degree = Counter()
    out_degree = Counter()
    seen_nodes = set()

    for element in [elm for elm in neo_elements if type(elm) is Node]:
        nodes[element.id] = QueryNode(
            char_symbol=f"o_{element.id}",
            parallel_element=None,  # not implemented currently in this conversion
            name=element['name'],
            attributes=element['attributes'],
            relations=[]
        )

    for element in [elm for elm in neo_elements if hasattr(elm, 'start_node') and ":_" not in elm.type]:
        source_obj_id = element.start_node.id
        target_obj_id = element.end_node.id
        seen_nodes.add(source_obj_id)
        if target_obj_id in seen_nodes:
            continue
        key = f"r_{source_obj_id}_{target_obj_id}"
        new_elm = QueryRelationship(char_symbol=key, name=element.type,
                                    source=nodes[source_obj_id], target=nodes[target_obj_id])
        nodes[target_obj_id].backward_relation = new_elm
        nodes[source_obj_id].relations.append(new_elm)
        nodes[key] = new_elm
        out_degree[source_obj_id] += 1
        in_degree[target_obj_id] += 1

    for element in [elm for elm in neo_elements if hasattr(elm, 'start_node') and ":_" in elm.type]:
        # prepositions
        source_obj_id = element.start_node.id
        target_obj_id = element.end_node.id
        source_obj = nodes[source_obj_id]
        subject = source_obj.backward_relation.source
        key = f"r_{source_obj_id}_{target_obj_id}"
        new_elm = QueryRelationship(char_symbol=key, name=element.type,
                                    source=subject, target=nodes[target_obj_id])
        if not subject.relations[0].prepositions:
            subject.relations[0].prepositions = []
        subject.relations[0].prepositions.append(new_elm)
        nodes[key] = new_elm

    if len(nodes) == 1:
        return SubGraph(list(nodes.values())[0])
    else:
        root_candidates = [elm for elm, deg in out_degree.items() if deg >= 1 and in_degree[elm] == 0]
        assert len(root_candidates) == 1
        return SubGraph(nodes[root_candidates[0]])


def cut_triplet_by_target_index(root, source_index, target_index):
    root[source_index].relations = [r for r in root[source_index].relations
                                    if r.target != root[target_index]]
    if root[source_index].relations and root[source_index].relations[0].prepositions:
        root[source_index].relations[0].prepositions = [r for r in root[source_index].relations[0].prepositions
                                                        if r.target != root[target_index]]


def cut_triplet_by_relation(root, relation):
    root.relations = [r for r in root.relations if r != relation]
    if root.relations and root.relations[0].prepositions:
        root.relations[0].prepositions = [pp for pp in root.relations[0].prepositions if pp != relation]


def remove_duplicates(lst):
    # this removes duplicates, but unlike using set, it will keep the original order of the list
    return list({s: 0 for s in lst})


def is_imsitu_scene_key(scene_key):
    return '_' in scene_key
