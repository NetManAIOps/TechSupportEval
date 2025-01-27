import os
import json
import argparse
import traceback
import subprocess
import pandas as pd
from typing import Optional
from pathlib import Path
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import as_completed
from .eval_exp import calc_metric

CONFIG_DIR = Path(os.path.dirname(__file__)) / '..' / 'config'

OUTPUT_DIR = Path(os.path.dirname(__file__)) / '..' / 'output'

VENV_DIR = Path(os.path.dirname(__file__)) / '..' / 'venv'

class FileCache:
    def __init__(self, cache_file):
        self.cache_file = cache_file
        self.cache = {}
        self.modified = False

    def __enter__(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                self.cache = json.load(f)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.modified:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)

    def get(self, key, default=None):
        return self.cache.get(key, default)

    def set(self, key, value):
        self.cache[key] = value
        self.modified = True

    def delete(self, key):
        if key in self.cache:
            del self.cache[key]
            self.modified = True

def read_experiments(exp_id: str) -> list[str]:
    file_path = CONFIG_DIR / 'experiments' / (exp_id + '.txt')
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def run_command_in_env(env_name: str, command: str):
    prefix = ""
    if env_name != "default":
        env_path = (VENV_DIR / env_name).resolve()
        activate_script = env_path / "bin" / "activate" if os.name != "nt" else env_path / "Scripts" / "activate"

        if os.name != "nt":
            prefix = f". {activate_script} && "
        else:
            prefix = f"{activate_script} && "

    full_command = prefix + command
    print(f"Executing: {full_command}")
    subprocess.run(full_command, shell=True, stderr=subprocess.STDOUT)

def run_command(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True)
    return result.stdout

def get_exp_name(args: str, cache: Optional[FileCache]=None) -> str:
    if cache is not None:
        name = cache.get(args)
        if name is not None:
            return name
    command = ["python", "-m", "scripts.run_exp"] + args.split() + ["--dry_run"]
    name = run_command(command).strip()
    cache.set(args, name)
    return name

def check_errors(exp_name: str) -> bool:
    error_file = OUTPUT_DIR / exp_name / "errors.json"
    return os.path.exists(error_file)

def should_run(exp_name: str) -> bool:
    result_dir = OUTPUT_DIR / exp_name
    error_file = OUTPUT_DIR / exp_name / "errors.json"
    return os.path.exists(error_file) or not os.path.exists(result_dir)

def get_completion_stats(exp_name: str) -> dict[str, float]:
    error_file = OUTPUT_DIR / exp_name / "errors.json"
    score_file = OUTPUT_DIR / exp_name / "score.json"
    
    ok_count = 0
    error_count = 0

    if os.path.exists(score_file):
        with open(score_file, 'r') as f:
            ok_count = len(json.load(f))
    
    if os.path.exists(error_file):
        with open(error_file, 'r') as f:
            error_count = len(json.load(f))
    
    total_count = error_count + ok_count
    
    return {
        "ok_count": ok_count,
        "total_count": total_count,
        "finished": ok_count == total_count,
    }

def calc_metric_one(exp_name: str):
    try:
        segments = exp_name.split('|', maxsplit=3)
        method, dataset, llm, params = segments
        params_dict = {}
        for kv in params.split(','):
            if kv.find('=') < 0:
                continue
            k, v = kv.split('=')
            params_dict[k] = v
        item = {
            'method': method,
            'dataset': dataset,
            'llm': llm,
        }
        item.update(calc_metric(
            Path('data') / 'final' / (dataset + '.json'),
            Path('output') / exp_name / 'score.json',
        ))
        item.update(params_dict)
        return item
    except:
        traceback.print_exc()
        return None

def calc_metric_batch(exp_names):
    results = []
    with ProcessPool(max_workers=16) as pool:
        future_to_id = {}
        for exp_name in exp_names:
            future = pool.schedule(calc_metric_one, args=(exp_name,))
            future_to_id[future] = exp_names
        for future in tqdm(as_completed(future_to_id), total=len(exp_names)):
            res = future.result()
            if isinstance(res, dict):
                results.append(res)
    return results

def output_metric(exp_id: str, exp_names: list[str]):
    report_dir = OUTPUT_DIR / 'reports'
    os.makedirs(report_dir, exist_ok=True)
    report_file = report_dir / f'{exp_id}.csv'
    df = pd.DataFrame(calc_metric_batch(exp_names))
    df = df.sort_values(by=['method'])
    df.to_csv(report_file, index=False)
    print(f'Result saved to {report_file}')

def get_env_map(exp_names: list[str]) -> dict[str, str]:
    import sys
    module_names = [item.strip().split(' ')[0].strip() for item in exp_names]
    df = pd.read_csv(CONFIG_DIR / 'baselines.csv')
    
    env_map = {
        row['id']: row['env']
        for _, row in df[df['id'].isin(module_names)].iterrows()
    }

    return env_map

def ensure_env(env_name: str):
    if env_name == 'default':
        return
    
    env_path = (VENV_DIR / env_name).resolve()
    common_requirements_path = (CONFIG_DIR / 'envs' / 'requirements.txt').resolve()
    requirements_path = (CONFIG_DIR / 'envs' / f'requirements.{env_name}.txt').resolve()
    setup_script_path = (CONFIG_DIR / 'envs' / f'setup.{env_name}.sh').resolve()

    if env_path.exists():
        return
    
    print(f"Creating environment '{env_name}' at {env_path}...")
    run_command_in_env("default", f"python -m venv {env_path}")

    if common_requirements_path.exists():
        run_command_in_env(env_name, f"pip install -r {common_requirements_path}")

    if requirements_path.exists():
        run_command_in_env(env_name, f"pip install -r {requirements_path}")

    if setup_script_path.exists():
        run_command_in_env(env_name, f"bash {setup_script_path}")

    print(f"Environment '{env_name}' setup completed.")

def main():
    parser = argparse.ArgumentParser(description="Run experiments with auto iteration")
    parser.add_argument("--schema", type=str, default="0", help="Experiment ID")
    parser.add_argument("--max_iters", type=int, default=2, help="Maximum number of iterations")
    parser.add_argument('--force', action='store_true', help='Force to run all experiments')
    parser.add_argument('--no_metric', action='store_false', help='Do not calculate the metric results')
    parser.add_argument('--metric_only', action='store_true', help='Output the metric results only (without running experiments)')
    args = parser.parse_args()

    exp_id = args.schema
    experiments = read_experiments(exp_id)
    with FileCache(OUTPUT_DIR / '.map.json') as cache:
        exp_names = [get_exp_name(exp, cache) for exp in experiments]

    env_map = get_env_map(experiments)

    if args.metric_only:
        output_metric(exp_id, exp_names)
        return
    
    exp_names_should_run = set(filter(lambda exp_name: should_run(exp_name) or args.force, exp_names))
    exp_map = dict(zip(exp_names, experiments))

    successful_exps = [name for name in exp_names if name not in exp_names_should_run]
    incomplete_exps = []

    for num_iter in range(args.max_iters):
        print(f'\n> Iteration {num_iter}')
        for exp_name in list(exp_names_should_run):
            exp_args = exp_map[exp_name]
            print(exp_name)
            module_name = exp_args.strip().split(' ')[0].strip()
            env_name = env_map[module_name]
            ensure_env(env_name)
            run_command_in_env(env_name, "python -m scripts.run_exp " + exp_args)
            
            if not check_errors(exp_name):
                successful_exps.append(exp_name)
                exp_names_should_run.remove(exp_name)                
            
        if not exp_names_should_run:
            break

    incomplete_exps = list(exp_names_should_run)

    if successful_exps:
        print("\nResolved experiments:")
        for exp in successful_exps:
            print(f"- {exp}")

    if incomplete_exps:
        print("\nUnresolved experiments:")
        for exp in incomplete_exps:
            stats = get_completion_stats(exp)
            print(f"- {exp} ({stats['ok_count']} / {stats['total_count']})")

    print()

    if not args.no_metric:
        output_metric(exp_id, exp_names)


if __name__ == "__main__":
    main()
