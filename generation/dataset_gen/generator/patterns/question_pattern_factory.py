from generator.patterns.all_attr import AllAttributePattern
from generator.patterns.all_subject import AllSubjectPattern
from generator.patterns.common_attribute import CommonAttributePattern
from generator.patterns.compare_count import CompareCountPattern
from generator.patterns.count_ref import CountRefPattern
from generator.patterns.count_ref_group_by import CountRefGroupByPattern
from generator.patterns.multiple_ref import MultipleRefPattern
from generator.patterns.query_attr import QueryAttributePattern
from generator.patterns.query_object import QueryObjectPattern
from generator.patterns.question_pattern import QuestionPattern
from generator.patterns.choose_rel import ChooseRelationPattern


class QuestionPatternFactory:
    factory_patterns = {}

    @classmethod
    def create(cls, pattern, scene_reader, random_seed=None, pick_random_distractors=False) -> QuestionPattern:
        patterns_to_class = {
            'compare_count': CompareCountPattern,
            'all_subject': AllSubjectPattern,
            'all_attr': AllAttributePattern,
            'common_attribute': CommonAttributePattern,
            'query_object': QueryObjectPattern,
            'query_attr': QueryAttributePattern,
            'choose_rel': ChooseRelationPattern,
            'multiple_ref': MultipleRefPattern,
            'count_ref': CountRefPattern,
            'count_ref_group_by': CountRefGroupByPattern
        }
        if pattern['pattern_index'] not in cls.factory_patterns:
            cls.factory_patterns[pattern['pattern_index']] = patterns_to_class[pattern['class']](
                pattern_dict=pattern, scene_reader=scene_reader, random_seed=random_seed,
                pick_random_distractors=pick_random_distractors
            )

        return cls.factory_patterns[pattern['pattern_index']]