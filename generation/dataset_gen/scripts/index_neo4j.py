import argparse
import json
import os
import csv
import wget
import shutil

from tqdm import tqdm


def normalize_name(name):
    return name.replace(" ", "_")


def index_gqa(data_dir):
    print("Loading GQA scenes")
    index_gqa_scenes(json.load(open(os.path.join(data_dir, "gqa", "train_sceneGraphs.json"))), 'train')
    index_gqa_scenes(json.load(open(os.path.join(data_dir, "gqa", "val_sceneGraphs.json"))), 'val')


def index_gqa_scenes(scenes, split):
    source = 'gqa'
    for scene_id, scene in tqdm(scenes.items()):
        scene_key = f"scene_{scene_id}"
        objects_csv.writerow([scene_key, 'Scene', scene_id, '', source, '', split])
        for obj_id, obj in scene['objects'].items():
            attributes = [a for a in obj['attributes'] if a]
            write_object(obj_id, obj['name'], scene_id, source, attributes, split)

        for obj_id, obj in scene['objects'].items():
            relations_csv.writerow([scene_key, obj_id, 'IN'])
            for relation in obj['relations']:
                if relation['name'] in ['to the left of', 'to the right of']:
                    continue
                write_relation(obj_id, relation['object'], relation['name'])


def write_relation(obj_id, other_object_id, relation_name):
    relations_csv.writerow([obj_id, other_object_id, relation_name])


def write_object(obj_id, obj_name, scene_id, source, attributes, split):
    objects_csv.writerow([obj_id, 'Object', scene_id, obj_name, source, ';'.join(attributes), split])


def get_imsitu_actions(spc, data_dir):
    verbs_templates_f = open(f'{data_dir}/generation_templates.tab')
    verbs_templates_lines = [l.strip().split('\t') for l in verbs_templates_f]
    verbs_templates = {l[0]: l[1] for l in verbs_templates_lines}
    templates_per_verb = {}
    for verb, verb_info in spc['verbs'].items():
        template = verbs_templates[verb]
        tokens = [t.strip('.') for t in template.split()]
        nouns = [t for t in tokens if t.isupper()]
        actions_pos_in_abstract = [tokens.index(t.upper()) for t in nouns]
        relations = [tokens[actions_pos_in_abstract[i] + 1:actions_pos_in_abstract[i + 1]] for i in
                     range(len(actions_pos_in_abstract) - 1)]
        templates_per_verb[verb] = {
            "actions": [' '.join(r) for r in relations],
            "nouns": [n.lower() for n in nouns]
        }
    return templates_per_verb


def index_imsitu(data_dir):
    print("Loading imsitu scenes")
    space = json.load(open(f"{data_dir}/imsitu/imsitu_space.json", "rt"))

    imsitu_dir = f"{data_dir}/imsitu"

    index_imsitu_scenes(space, json.load(open(f"{imsitu_dir}/train.json", "rt")), 'train', imsitu_dir)
    index_imsitu_scenes(space, json.load(open(f"{imsitu_dir}/dev.json", "rt")), 'val', imsitu_dirב
                        )


def index_imsitu_scenes(space, scenes, split, data_dir):
    source = 'imsitu'
    actions_per_verb = get_imsitu_actions(space, data_dir)

    scenes_formatted_dict = {}

    for scene_id, scene in tqdm(scenes.items()):
        scene_id = scene_id.split('.')[0]
        verb = scene['verb']
        verb_template = actions_per_verb[verb]

        scene_formatted = {'objects': {}}
        scenes_formatted_dict[scene_id] = scene_formatted

        def get_name_from_placeholder(nid):
            return space['nouns'][nid]['gloss'][0]

        if len(verb_template['nouns']) < 2:
            continue

        first_placeholder = verb_template['nouns'][0]
        second_placeholder = verb_template['nouns'][1]

        # pick frames where we have at least an object and subject
        valid_frames = [frame for frame in scene['frames'] if
                        frame.get(first_placeholder) and frame.get(second_placeholder)]
        if not valid_frames:
            continue
        frame = valid_frames[0]

        first_noun_name = get_name_from_placeholder(frame[first_placeholder])
        objects_csv.writerow([scene_id, 'Scene', scene_id, '', source, '', split])
        obj_id = f"{scene_id}_0"
        write_object(obj_id, first_noun_name, scene_id, source, [], split)
        scene_formatted['objects'][obj_id] = {
            'name': first_noun_name,
            'relations': [],
            'attributes': []
        }
        relations_csv.writerow([scene_id, obj_id, 'IN'])

        # `verb` is in present progressive tense which is more suited for our questions, but does not contain
        # all relevant words (e.g. "flapping" instead of "flapping its"), so we replace just the first word
        verb_action_words = verb_template['actions'][0].split()
        verb_relation_name = ' '.join([verb] + verb_action_words[1:])

        for i, action in enumerate(verb_template['actions']):
            if not action:
                # TODO: may want to extend format to support empty actions
                continue
            other_placeholder = verb_template['nouns'][i + 1]
            if other_placeholder not in frame:
                continue
            if other_placeholder == "place":
                if i == 0:
                    # no subject to verb
                    break
                continue
            other_noun_id = frame[other_placeholder]
            if not other_noun_id:
                if i == 0:
                    # missing object, fill with something
                    other_noun = other_placeholder
                else:
                    continue
            else:
                other_noun = get_name_from_placeholder(other_noun_id)
            other_obj_id = f"{scene_id}_{i + 1}"
            write_object(other_obj_id, other_noun, scene_id, source, [], split)
            scene_formatted['objects'][other_obj_id] = {'name': other_noun, 'relations': [], 'attributes': []}
            relations_csv.writerow([scene_id, other_obj_id, 'IN'])
            if i == 0:
                relation_name = verb_relation_name
            else:
                relation_name = f'{verb_relation_name}:_{action}'

            # all subjects should be connected to the object with the relation.
            # then, objects are connected to other modifiers
            relation_source = 0 if (i == 0) else 1
            relation_from_id = f"{scene_id}_{relation_source}"
            relation_to_id = f"{scene_id}_{i + 1}"
            write_relation(relation_from_id, relation_to_id, relation_name)
            relation_dict = {'name': relation_name, 'object': relation_to_id}

            if i == 0:
                scene_formatted['objects'][obj_id]['relations'].append(relation_dict)
            if i > 0:
                scene_formatted['objects'][obj_id]['relations'][0].setdefault('prepositions', []).append(relation_dict)

    json.dump(scenes_formatted_dict, open(os.path.join(data_dir, f'{split}_imsitu_formatted.json'), 'wt'))


def download_input_files(download_path):
    gqa_dir = f"{download_path}/gqa/"
    if not os.path.exists(f"{gqa_dir}/sceneGraphs.zip"):
        os.makedirs(f"{gqa_dir}", exist_ok=True)
        print("Download GQA")
        wget.download("https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip", gqa_dir)
        print("Unarchiving GQA")
        shutil.unpack_archive(f"{download_path}/gqa/sceneGraphs.zip", gqa_dir)

    imsitu_dir = f"{download_path}/imsitu/"
    os.makedirs(imsitu_dir, exist_ok=True)
    for file in ["train.json", "dev.json", "imsitu_space.json", "simple_sentence_realization/generation_templates.tab"]:
        file_head, file_tail = os.path.split(file)
        if os.path.exists(f"{imsitu_dir}/{file_tail}"):
            continue
        print(f"Download imsitu {file}")
        wget.download(f"https://github.com/my89/imSitu/raw/master/{file}", f"{download_path}/imsitu")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--data-dir', default='data/')
    args.add_argument('--neo4j-import-path', default='../../../neo4j-community-4.2.1/import/')
    args = args.parse_args()

    download_input_files(args.data_dir)

    objects_csv = csv.writer(open(args.neo4j_import_path + 'objects.csv', 'wt'))
    objects_csv.writerow([':ID', ':LABEL', 'scene_id', 'name', 'source', 'attributes:string[]', 'split'])

    relations_csv = csv.writer(open(args.neo4j_import_path + 'relations.csv', 'wt'))
    relations_csv.writerow([':START_ID', ':END_ID', ':TYPE'])

    index_gqa(args.data_dir)
    index_imsitu(args.data_dir)

    print("Done")
