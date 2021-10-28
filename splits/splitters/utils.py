from typing import List


def program_to_string(program: List, index=-1, add_arguments: bool = True):
    """
    Converts a COVR program to a string
    """
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
