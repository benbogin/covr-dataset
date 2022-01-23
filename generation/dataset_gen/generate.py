import argparse
import concurrent
import json
import os
from collections import Counter

from concurrent.futures.process import ProcessPoolExecutor
from pprint import pprint
from random import Random

from tqdm import tqdm

from generator.queries.graph_executor import GraphExecutor
from generator.question_generator import QuestionGenerator
from generator.resources import Resources

parser = argparse.ArgumentParser()

parser.add_argument('--multiproc', default=16, type=int)
parser.add_argument('--split', default='val')
parser.add_argument('--no_save_redis', action='store_true')
parser.add_argument('--output_file')

args = parser.parse_args()


def worker(scene_id, write_debug=False):
    gen = question_generator.generate_question_from_scene(scene_id)
    questions = []

    for question, program, scenes, answer, question_pattern, debug_info, sub_graphs, simple_ref_text, boxes in gen:
        qst_json = {
            'question': question,
            'scenes': scenes,
            'answer': answer,
            'pattern_index': question_pattern['pattern_index'],
            'pattern_name': question_pattern['pattern_name'],
            'sub_graphs': sub_graphs,
            'program': program,
            'ref_text': simple_ref_text,
            'starting_scene_id': scene_id,
            'sub_graph_info': debug_info['sub_graph_info']
        }
        if write_debug:
            qst_json['debug_info'] = debug_info
        questions.append(qst_json)

    return scene_id, questions


def write_questions(qs, cnt_start, qid_prefix):
    def hash_question(q):
        qst_tuple = q['question'], q['answer'], frozenset(q['scenes'])
        return qst_tuple

    written = 0
    for q in qs:
        q_hash = hash_question(q)
        if q_hash in seen_questions:
            continue
        else:
            seen_questions.add(q_hash)

        pattern_types_cnt[q.get('pattern_name', -1)] += 1
        q['qid'] = f"{qid_prefix}_{cnt_start + written}"
        fw.write(json.dumps(q) + '\n')
        written += 1
    return written


if __name__ == "__main__":
    Resources.load(questions_yaml='questions.yaml', load_tiny_distract_yaml=False)
    question_generator = QuestionGenerator(split=args.split, random_seed=2,
                                           question_patterns_ids=[],
                                           enable_graph_cache=True)
    scene_ids = list(question_generator.scene_reader.get_all_scene_ids())
    Random(0).shuffle(scene_ids)

    if args.output_file:
        fname = args.output_file
    else:
        fname = args.split
    fpath = f'output/{fname}.jsonl'
    os.makedirs("output", exist_ok=True)

    print(f"Starting generation for {len(scene_ids)} scenes")
    print(f"Output path: {fpath}")

    fw = open(fpath, 'wt')

    seen_questions = set()
    pattern_types_cnt = Counter()

    global_cnt = 0

    if args.multiproc > 1:
        with ProcessPoolExecutor(max_workers=args.multiproc) as executor:
            futures = {}
            for scene_id in scene_ids:
                futures[scene_id] = executor.submit(worker, scene_id)

            loop = tqdm(concurrent.futures.as_completed(futures.values()), total=len(scene_ids))
            for future in loop:
                scene_id, questions = future.result()
                global_cnt += write_questions(questions, global_cnt, qid_prefix=args.output_file)
                loop.desc = f'# questions: {global_cnt}'
                del futures[scene_id]
    else:
        for scene_id in tqdm(scene_ids):
            _, questions = worker(scene_id)
            global_cnt += write_questions(questions, global_cnt, qid_prefix=args.output_file)

    print("Total neo4j queries executed (not cached):", GraphExecutor.n_queries_executed_not_cached)
    print("Total neo4j queries executed (cached):", GraphExecutor.n_queries_executed_cached)

    for i in sorted(list(pattern_types_cnt.keys())):
        print(f"Questions for pattern #{i}: {pattern_types_cnt.get(i, '-')}")

    if not args.no_save_redis:
        GraphExecutor.finished()
