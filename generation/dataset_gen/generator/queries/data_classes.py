from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, Set, List

import numpy as np


@dataclass
class Position:
    x: float
    y: float
    width: float
    height: float

    def iou(self, b):
        # this is just the usual way to IoU from bounding boxes
        inter = Position.intersection(self, b)
        area_a = Position.area(self)
        area_b = Position.area(b)
        return inter / (area_a + area_b - inter + 1e-12)

    def area(self):
        return self.width * self.height
    
    def intersection(self, b):
        min_point = np.array((max(self.x, b.x), max(self.y, b.y)))
        max_point = np.array((min(self.x + self.width, b.x + b.width),
                              min(self.y + self.height, b.y + b.height)))
        inter = (max_point - min_point).clip(min=0)
        area = inter[0] * inter[1]
        return area

    @classmethod
    def from_obj(cls, obj):
        return Position(x=obj['x'], y=obj['y'], width=obj.get('w') or obj.get('width'), height=obj.get('h') or obj.get('height'))


@dataclass
class QueryElement(ABC):
    char_symbol: str = ''

    # we save elements that are "parallel" in the graph, e.g. in the graph structure A -> B, A -> C, we will save that
    # B is parallel to C and vice-versa. This is helpful when creating the cypher-query, where sometimes we want to add
    # a clause indicating that these two nodes should be different
    parallel_element: 'QueryElement' = None

    name: Union[str, Set[str]] = None
    original_name: str = None  # used for filtering out multiple names in scenes

    @abstractmethod
    def as_cyhper_element(self):
        pass

    @abstractmethod
    def as_returned_element(self):
        pass

    @abstractmethod
    def get_trailing_symbol(self):
        pass

    def serialize(self):
        raise NotImplementedError()

    def as_tuple(self):
        raise NotImplementedError()


@dataclass
class QueryNode(QueryElement):
    position: Position = None
    attributes: Set = None
    original_attribute: str = None
    relations: List = None
    backward_relation: 'QueryRelationship' = None
    empty: bool = False

    def serialize(self):
        if self.name is None:
            serialized_names = None
        elif type(self.name) is set:
            serialized_names = tuple(sorted(list(self.name)))
        else:
            serialized_names = (self.name,)

        return serialized_names if self.name else None, tuple(sorted(list(self.attributes))), self.empty

    def as_cyhper_element(self, char_symbol_prefix=''):
        if not self.attributes:
            self.attributes = set()
        names = self.name
        if type(self.name) is str:
            names = [self.name]
        char_symbol = f"{char_symbol_prefix}{self.char_symbol}"
        str_elm = f'({char_symbol}:Object)'
        where_clauses = []
        if names:
            names = sorted(list(names))  # sort to make query consistent for caching
            where_elements_name = '(' + ' OR '.join([f'{char_symbol}.name = "{n}"' for n in names]) + ')'
            where_clauses.append(where_elements_name)
        if self.attributes:
            attributes = sorted(list(self.attributes))  # sort to make query consistent for caching
            where_elements_attributes = '(' + ' OR '.join(
                [f'"{a}" in {char_symbol}.attributes' for a in attributes]) + ')'
            where_clauses.append(where_elements_attributes)
        where_clause = ' AND '.join(where_clauses)
        return str_elm, where_clause

    def as_returned_element(self):
        return f"{self.char_symbol}.name"

    def get_trailing_symbol(self):
        return "-"

    def as_tuple(self):
        elements = [('noun', self.name)]
        for attr in self.attributes:
            elements.append(('attribute', attr))
        return tuple(elements)


@dataclass
class QueryRelationship(QueryElement):
    source: QueryNode = None
    target: QueryNode = None
    dataset_source: str = None
    prepositions: List['QueryRelationship'] = None
    backward_relation: 'QueryRelationship' = None

    def serialize(self):
        if self.name is None:
            serialized_names = None
        elif type(self.name) is set:
            serialized_names = tuple(sorted(list(self.name)))
        else:
            serialized_names = (self.name,)
        return serialized_names if self.name else None

    def as_cyhper_element(self, char_symbol_prefix=''):
        str_elm = "["
        char_symbol = f"{char_symbol_prefix}{self.char_symbol}"
        str_elm += f"{char_symbol}"
        names = [self.name] if type(self.name) is str else self.name
        if names:
            names = sorted(list(names))  # sort to make query consistent for caching
            str_elm += ":" + "|".join([f"`{n}`" for n in names])
        str_elm += "]"
        return str_elm, []

    def as_returned_element(self):
        return f"type({self.char_symbol})"

    def get_trailing_symbol(self):
        return "->"

    def as_tuple(self):
        return (('relation', self.name),)
