import os
import dotenv
import json
import argparse
import importlib
from tqdm import tqdm
from pathlib import Path
from pebble import ProcessPool
from concurrent.futures import as_completed

MODEL_VARIANTS = ["temperature", "stat"]

def init_worker(module_name, llm, args):
    global module, evaluate
    from dotenv import load_dotenv
    load_dotenv()

    module = importlib.import_module(module_name)
    evaluate = module.evaluate
    try:
        init_config = module.init_config
        init_config(llm, args)
    except:
        pass

def get_current_time(): # ms
    from datetime import datetime
    return int(datetime.now().timestamp() * 1e6)


def get_current_quota():
    import requests
    dotenv.load_dotenv()
    ONEAPI_URL = os.environ["OPENAI_API_BASE"]
    ONEAPI_TOKEN = os.environ["ONEAPI_TOKEN"]
    r = requests.get(ONEAPI_URL.rstrip('/').rstrip('/v1') + '/api/user/self', headers={'Authorization': ONEAPI_TOKEN})
    return r.json()['data']['quota'] * 2

def run_evaluation(task):
    global evaluate

    id, question, ground_truth, answer = task
    res = {"id": id, "error": None}
    try:
        start_time = get_current_time()
        result = evaluate(question, ground_truth, answer)
        end_time = get_current_time()
        elapsed_time = end_time - start_time
        res["elapsed_time"] = elapsed_time
        if isinstance(result, (float, int)):
            res["score"] = float(result)
            return res
        elif isinstance(result, (list, tuple)) and len(result) == 2:
            res["score"] = float(result[0])
            res["extra"] = result[1]
            return res
        else:
            res["error"] = "Invalid return format from evaluate function"
            return res
    except Exception as e:
        res["error"] = str(e)
        return res
    
def load_existing_data(file_path):
    if file_path.exists():
        with open(file_path, "r") as f:
            return {item["id"]: item for item in json.load(f)}
    return {}

def save_incremental_data(file_path, data):
    existing_data = load_existing_data(file_path)
    existing_data.update(data)
    with open(file_path, "w") as f:
        json.dump(list(existing_data.values()), f, indent=2, ensure_ascii=False)

def run_experiment(module, dataset, llm, args):
    dataset_path = Path(os.path.dirname(__file__)) / '..' / 'data' / 'final' / f'{dataset}.json'

    with open(dataset_path, "r") as f:
        data = json.load(f)

    module_name = module.split('.')[-1]
    exp_name = f"{module_name}|{dataset}|{llm}|{args.exp_vars}"

    if args.dry_run:
        print(exp_name)
        return

    mod = importlib.import_module(module)
    evaluate = mod.evaluate
    assert callable(evaluate), "Cannot load module func"

    output_dir = Path(os.path.dirname(__file__)) / '..' / 'output' /  exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_scores = load_existing_data(output_dir / "score.json")

    tasks = [(item["id"], item["question"], item["ground_truth"], item["answer"])  for item in data if item["id"] not in existing_scores]

    if not tasks:
        return
    
    total_tasks = len(data)
    remaining_tasks = len(tasks)

    new_results = {}
    errors = []

    if args.stat:
        start_quota = get_current_quota()
        total_elapsed_time = 0

    with ProcessPool(
        max_workers=args.parallel, 
        initializer=init_worker, 
        initargs=(module, llm, args)
    ) as pool:
        future_to_id = {}
        for task in tasks:
            future = pool.schedule(run_evaluation, args=(task,), timeout=args.timeout)
            future_to_id[future] = task[0]
        
        with tqdm(total=total_tasks, 
                    initial=total_tasks-remaining_tasks, 
                    desc="Evaluating") as pbar:
            for future in as_completed(future_to_id):
                task_id = future_to_id[future]
                try:
                    result = future.result()

                    if result["error"]:
                        errors.append(result)
                    else:
                        new_results[task_id] = result

                    if args.stat:
                        total_elapsed_time += result.get("elapsed_time", 0)
                except TimeoutError:
                    errors.append({"id": task_id, "error": "Task timed out"})
                except Exception as e:
                    errors.append({"id": task_id, "error": f"Unexpected error: {str(e)}"})
                pbar.update()

    if args.stat:
        end_quota = get_current_quota()

    if new_results:
        scores = {r["id"]: {"id": r["id"], "score": r["score"]} for r in new_results.values()}
        save_incremental_data(output_dir / "score.json", scores)

    extra_keys = set()
    for r in new_results.values():
        if "extra" in r:
            extra_keys.update(r["extra"].keys())

    for key in extra_keys:
        extra_data = {r["id"]: {"id": r["id"], "score": r["score"], key: r["extra"].get(key)} for r in new_results.values() if "extra" in r and key in r["extra"]}
        save_incremental_data(output_dir / f"{key}.json", extra_data)

    error_file = output_dir / "errors.json"
    if errors:
        with open(error_file, "w") as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
    else:
        if os.path.exists(error_file):
            os.remove(error_file)

    if args.stat:
        stats_file = output_dir / "stats.json"
        ok_num = len(list(new_results.values()))
        with open(stats_file, "w") as f:
            json.dump({
                'time': total_elapsed_time / max(ok_num, 1),
                'cost': (start_quota - end_quota) / max(ok_num, 1),
            }, f, indent=2, ensure_ascii=False)
    
class EnhancedArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        self.explicitly_set = set()
        self.shortnames = {}
        super().__init__(*args, **kwargs)

    def add_argument(self, *args, **kwargs):
        action = super().add_argument(*args, **kwargs)
        long_option = next((arg for arg in action.option_strings if arg.startswith('--')), None)
        short_option = next((arg for arg in action.option_strings if arg.startswith('-') and not arg.startswith('--')), None)

        if long_option and short_option:
            self.shortnames[long_option[2:]] = short_option[1:]
        return action

    def parse_args(self, *args, **kwargs):
        namespace = super().parse_args(*args, **kwargs)
        for action in self._actions:
            if action.dest != 'help' and getattr(namespace, action.dest) != action.default:
                self.explicitly_set.add(action.dest)
        return namespace

def generate_exp_vars(args: argparse.Namespace, parser: EnhancedArgumentParser) -> str:
    exp_vars_list = []

    for param in parser.explicitly_set:
        if param in MODEL_VARIANTS:
            value = getattr(args, param)
            param_name = parser.shortnames.get(param, param)
            exp_vars_list.append(f"{param_name}={value}")

    if not exp_vars_list:
        return 'default'
    
    return ",".join(exp_vars_list)

def main():
    parser = EnhancedArgumentParser(description="Run experiment")
    parser.add_argument("module", help="Module name")
    parser.add_argument("dataset", help="Dataset name")
    parser.add_argument("llm", help="LLM name")
    parser.add_argument("-t", "--temperature", type=float, default=0, help="Temperature")
    parser.add_argument("-j", "--parallel", type=int, default=16, help="Number of parallel workers")
    parser.add_argument("-s", "--stat", action='store_true', help="Get stat of usage")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries")
    parser.add_argument("--timeout", type=float, default=60, help="Timeout for each evaluation in seconds")
    parser.add_argument("--dry_run", action='store_true', help="Print identifier of this experiment")

    args = parser.parse_args()
    args.exp_vars = generate_exp_vars(args, parser)

    run_experiment(args.module, args.dataset, args.llm, args)

if __name__ == "__main__":
    main()
