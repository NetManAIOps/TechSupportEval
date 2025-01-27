import traceback
from tqdm import tqdm
from concurrent.futures import as_completed
from pebble import ProcessPool

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
                    # errors.append({"id": task_id, "error": "Task timed out"})
                except Exception as e:
                    print('Task %s: Unexpected error - %s' % (task_id, str(e)))
                    traceback.print_exc()

                pbar.update()
    return new_results