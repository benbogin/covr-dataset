import json
import os
from collections import defaultdict

from tqdm import tqdm

from generator.resources import Resources


class SceneReader:
    def __init__(self, split=None, data_dir="data", selected_scenes=None):
        # triplet to attributes of subject and object, used for optimization by preventing queries triplets if they
        # do not exist at all
        self.available_triplets = defaultdict(lambda: (set(), set()))
        self.available_relations = defaultdict(set)
        self.available_objects = defaultdict(set)
        self.available_attributes_for_object = defaultdict(set)

        splits = [split] if split else ["train", "val"]

        self._gqa_scenes = {}
        self._imsitu_scenes = {}
        for split in splits:
            self._gqa_scenes.update(self._read_formatted_scenes(
                os.path.join(data_dir + "/gqa", f"{split}_sceneGraphs.json"),
                selected_scenes
            ))
            self._imsitu_scenes.update(self._read_formatted_scenes(
                os.path.join(data_dir + "/imsitu", f"{split}_imsitu_formatted.json"),
                selected_scenes
            ))

        self._all_scenes = {}
        self._all_scenes.update(self._gqa_scenes)
        self._all_scenes.update(self._imsitu_scenes)

        self.all_scenes_keys = list(self._all_scenes.keys())

    def _read_formatted_scenes(self, file_path, selected_scenes=None):
        print(f"Loading scenes file: {file_path}")
        scenes = json.load(open(file_path))
        if selected_scenes:
            scenes = {k: scenes[k] for k in selected_scenes if k in scenes}
        print("formatting scenes...")
        for scene_key, scene in tqdm(scenes.items()):
            scene['scene_key'] = scene_key
            for obj_key, obj in scene['objects'].items():
                obj['object_key'] = obj_key
                obj['scene_id'] = scene_key
                obj['attributes'] = list(dict.fromkeys(obj['attributes']))
                obj["attributes_by_group"] = defaultdict(set)
                for attr in obj["attributes"]:
                    if attr in Resources.attributes_group_by_name:
                        group = Resources.attributes_group_by_name[attr]
                        obj["attributes_by_group"][group].add(attr)
                self.available_attributes_for_object[obj['name']].update(set(obj["attributes"]))

                original_relations = obj['relations']
                obj['relations'] = []
                for rel in original_relations:
                    # we skip 'to the left of', 'to the right of' relations in the scene
                    if rel['name'] in ['to the left of', 'to the right of']:
                        continue
                    obj['relations'].append(rel)
                    other_obj = scene['objects'][rel['object']]
                    self.save_available_triplet(obj, rel, other_obj)

                    for pp in rel.get('prepositions', []):
                        other_pp_obj = scene['objects'][pp['object']]
                        self.save_available_triplet(obj, pp, other_pp_obj)
        return scenes

    def save_available_triplet(self, obj, rel, other_obj):
        # we save the list of used (object, relation, subject) triplets as it can be used for pruning unnecessary
        # queries execution
        key = (obj['name'], rel['name'], other_obj['name'])
        self.available_triplets[key][0].update(obj['attributes'])
        self.available_triplets[key][1].update(other_obj['attributes'])
        key_missing_subject = ('*', rel['name'], other_obj['name'])
        key_missing_object = (obj['name'], rel['name'], '*')
        key_missing_subject_and_object = ('*', rel['name'], '*')
        self.available_triplets[key_missing_subject][1].update(other_obj['attributes'])
        self.available_triplets[key_missing_object][0].update(obj['attributes'])
        self.available_triplets[key_missing_subject_and_object][1].update(other_obj['attributes'])
        self.available_triplets[key_missing_subject_and_object][0].update(obj['attributes'])

        self.available_relations[(obj['name'], other_obj['name'])].add(rel['name'])
        self.available_objects[(obj['name'], rel['name'])].add(other_obj['name'])

    def _get_dict_by_source(self, source=None):
        if not source:
            return self._all_scenes
        elif source == 'gqa':
            return self._gqa_scenes
        elif source == 'imsitu':
            return self._imsitu_scenes
        assert False

    def get_total_scenes(self, source=None):
        return len(self._get_dict_by_source(source))

    def get_all_scene_ids(self, source=None):
        yield from self._get_dict_by_source(source).keys()

    def get_formatted_scenes(self, specific_scene_key=None, limit=1000000, source=None):
        d = self._get_dict_by_source(source)
        if specific_scene_key:
            return d[specific_scene_key]
        else:
            return d
