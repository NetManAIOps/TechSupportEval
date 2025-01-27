import os
import sys
import json
import traceback
import importlib
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import as_completed
from pebble import ProcessPool

RAG_INPUT_DIR = Path(os.path.dirname(__file__)) / '..' / 'data' / 'rag_inputs'
RAG_OUTPUT_DIR = Path(os.path.dirname(__file__)) / '..' / 'data' / 'rag_outputs'


def init_worker():
    from dotenv import load_dotenv
    load_dotenv()

def batch_run(func, tasks, timeout=60):
    new_results = {}
    total_tasks = len(tasks)
    with ProcessPool(
        max_workers=16, 
        initializer=init_worker,
    ) as pool:
        future_to_id = {}
        for task in tasks:
            future = pool.schedule(func, args=(task,), timeout=timeout)
            future_to_id[future] = task.get('question_id', task.get('id', 1))
        
        with tqdm(total=total_tasks, 
                    initial=0, 
                    desc="Processing") as pbar:
            for future in as_completed(future_to_id):
                task_id = future_to_id[future]
                try:
                    result = future.result()
                    new_results[task_id] = result
                except TimeoutError:
                    print('Task %s: Task timed out' % task_id)
                except Exception as e:
                    print('Task %s: Unexpected error - %s' % (task_id, str(e)))
                    traceback.print_exc()

                pbar.update()
    return new_results


def main():
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} <rag_name> <dataset_name>', file=sys.stderr)
        sys.exit(1)
    
    rag_name = sys.argv[1]
    dataset_name = sys.argv[2]

    dataset_path = RAG_INPUT_DIR / (dataset_name + '.json')
    
    rag = importlib.import_module(f'rag.{rag_name}')

    rag_id = rag.ID
    output_path = RAG_OUTPUT_DIR / (dataset_name + str(rag_id) + '.json')

    if not os.path.exists(dataset_path):
        print('Dataset not found', file=sys.stderr)
        sys.exit(1)

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    results = batch_run(rag.process_question, dataset)
    
    for item in dataset:
        item['answer'] = results[item['question_id']]

        item['id'] = item['question_id'] + '_' + str(rag_id)

        del item['reference_doc']
        del item['start_offset']
        del item['end_offset']

    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()
