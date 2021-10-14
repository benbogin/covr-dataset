import itertools
import json
import logging
from random import Random

import h5py
from typing import List, Optional
import numpy as np
import os

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, MetadataField, TensorField
from allennlp.data.instance import Instance
from overrides import overrides
from tqdm import tqdm

logger = logging.getLogger(__name__)


@DatasetReader.register("vcqa", exist_ok=True)
class VCQADatasetReader(DatasetReader):
    DatasetReader.times_loaded = 0

    def __init__(self,
                 features_h5_path: str,
                 img_key_to_index_path: str,
                 number_images_per_question: int,
                 num_vis_position_features: int = 4,
                 comp_split_base_path: str = None,
                 base_path: str = "data",
                 comp_split: str = None,
                 few_shot_examples: int = 0,
                 downsample: Optional[int] = None,
                 load_paraphrased_question: bool = False,
                 _always_load_features: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._base_path = base_path
        self._comp_split = comp_split
        self._comp_split_base_path = comp_split_base_path

        self._few_shot_examples = few_shot_examples
        self._downsample = downsample
        self._load_paraphrased_question = load_paraphrased_question

        self.feature_h5 = h5py.File(features_h5_path, 'r')
        self.number_images_per_question = number_images_per_question
        self.img_key_to_index = json.load(open(img_key_to_index_path, "rt"))
        self._num_vis_position_features = num_vis_position_features

        self.questions = {}
        self.random = Random(0)

        self._always_load_features = _always_load_features

        # we don't want to evaluate on all possible pairs of compositional properties, so we list those of interest
        # here to be (these pairs will be evaluated every epoch in addition to each property on its own)
        self._cooccuring_properties = [
            ('has_complex_quantifier_scope', 'has_quantifier_all'),
            ('has_count', 'has_attribute'),
            ('has_count', 'rm_v_c')
        ]

    def get_length(self, file_path):
        return len(self.questions[file_path])

    @overrides
    def _read(self, file_path: str):
        if 'train' in file_path:
            main_split = 'train'
        elif 'val' in file_path:
            main_split = 'val'
        elif 'test' in file_path:
            main_split = 'test'
        else:
            raise ValueError()

        if not self.questions.get(file_path):
            self.questions[file_path] = []
            with open(os.path.join(self._base_path, file_path), "r") as f:
                for line in itertools.islice(tqdm(f), self.max_instances):
                    ex = json.loads(line)
                    ex['scenes'] = ex['scenes'][:self.number_images_per_question]
                    if self._comp_split:
                        if self._should_example_be_on_gen_test(ex['properties'], self._comp_split) and main_split == 'train':
                            # we do not want to train on examples with the given compositional property
                            continue
                    self.random.shuffle(ex['scenes'])
                    self.questions[file_path].append(ex)

        questions = self.questions.get(file_path)
        if self._downsample and self._downsample < len(questions):
            self.questions[file_path] = questions = self.random.sample(questions, self._downsample)
        self.random.shuffle(questions)

        for qid, ex in enumerate(questions):
            sentence = ex['utterance']
            if self._load_paraphrased_question and ex.get('rephrase'):
                sentence = ex.get('rephrase')
            instance = self.text_to_instance(
                sentence=sentence,
                program=ex['program'],
                scenes=ex['scenes'],
                label=ex.get('answer'),
                qid=ex['qid'],
                pattern=ex['pattern_name'],
                comp_splits=self._get_example_with_cooccuring_properties(ex['properties'])
            )
            if self._always_load_features:
                # add image features in case `always_load_features` is on - used for inference (predict command)
                instance.fields.update(self.get_extra_fields(file_path, qid))
            yield instance

    def text_to_instance(self,
                         sentence: str,
                         program: List[dict],
                         scenes: List[str],
                         qid: int,
                         label: str,
                         pattern: str,
                         comp_splits: List[str] = None) -> Instance:
        return Instance({
            "sentence": MetadataField(sentence),
            "label": LabelField(str(label)),
            "metadata": MetadataField({
                "scenes": scenes,
                "qid": qid,
                "comp_splits": comp_splits,
                "pattern": pattern,

                # this will be true if this example has the compositional property we test on - important since
                # it allows us to evaluate results separately on seen and unseen properties
                "tested_comp_split": self._should_example_be_on_gen_test(comp_splits, self._comp_split)
            })
        })

    def get_extra_fields(self, file_path, item):
        """
        features that should be loaded at training time only and not during vocabulary building time
        (in this case, image features that we don't want to be loaded into memory)
        """
        scenes = self.questions[file_path][item]['scenes']
        features, positions = self._get_visual_features(scenes)
        return {
            "visual_features": TensorField(features),
            "visual_positions": TensorField(positions),
        }

    def _get_visual_features(self, scenes):
        # get the visual features of the given scenes from the h5 file, pad if necessary
        boxes2 = []
        feats2 = []

        for img_pos in range(max(self.number_images_per_question, 1)):
            boxes = None
            feats = None
            if img_pos < len(scenes):
                scene_id = scenes[img_pos]

                img_index = self.img_key_to_index.get(scene_id, None)
                if img_index is not None:
                    boxes = self.feature_h5['boxes'][img_index]
                    feats = self.feature_h5['features'][img_index]
                    assert len(boxes) == len(feats)
                # else:
                #     print(f"Warning, features for image {scene_id} not found, using zeros as features")

            if boxes is None:
                boxes = np.zeros((36, 4), np.float32)
                feats = np.zeros((36, 2048), np.float32)

            if self._num_vis_position_features == 5:
                boxes = np.concatenate((boxes, np.expand_dims((boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]), 1)), axis=1)

            boxes2.append(boxes)
            feats2.append(feats)

        feats = np.stack(feats2)
        boxes = np.stack(boxes2)

        return feats, boxes

    def _get_example_with_cooccuring_properties(self, example_properties):
        """
        gets the example properties (as listed in the dataset) and additionally adds co-occurrences
        """
        properties = set(example_properties)
        for p1, p2 in self._cooccuring_properties:
            if p1 in example_properties and p2 in example_properties:
                properties.add('+'.join(sorted([p1, p2])))

        return list(properties)

    @staticmethod
    def _should_example_be_on_gen_test(example_properties, test_comp_split):
        if not test_comp_split:
            return False

        tested_properties = test_comp_split.split('+')

        return all(p in example_properties for p in tested_properties)
