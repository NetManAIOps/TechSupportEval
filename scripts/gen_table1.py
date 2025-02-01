import os
import subprocess
import importlib
import pandas as pd
from pathlib import Path

CAPTION = 'Comparison of results on different evaluation methods'

EXP_IDS = [1, 2, 3]
DATASET = 'techqa'
RAG_SYSTEMS = ['gpt_langchain', 'llama_70b_langchain', 'llama_8b_langchain']
METHODS = [
    'rouge1',
    'rougeL',
    'bleu',
    'bert_score',
    '\\midrule',
    'langchain_evaluation',
    'llama_index_evaluation',
    'ragas',
    'quest_eval',
    'g_eval',
    'ref_checker',
    '\\midrule',
    'tseval',
    'no_fact_check',
    'no_step_check',
]

METRICS = ['AUC', 'Pearsonr']

CONFIG_DIR = Path(os.path.dirname(__file__)) / '..' / 'config'

OUTPUT_DIR = Path(os.path.dirname(__file__)) / '..' / 'output'

def run_command(command: list[str]) -> str:
    subprocess.run(command, shell=True)

def prepare_data() -> pd.DataFrame:
    evaluated_lm_map = {}
    for name in RAG_SYSTEMS:
        module = importlib.import_module(f'rag.{name}')
        evaluated_lm_map[DATASET + str(module.ID)] = module.TITLE
    
    baselines = pd.read_csv(CONFIG_DIR / 'baselines.csv')
    models = pd.read_csv(CONFIG_DIR / 'models.csv')
    models.set_index('model_id')
    reports = []
    for exp_id in EXP_IDS:
        run_command(f'python -m scripts.auto_run --schema {exp_id} --metric_only')
        reports.append(pd.read_csv(OUTPUT_DIR / 'reports' / f'{exp_id}.csv'))
    df = pd.concat(reports)
    # df['Model'] = df['method'].apply(lambda model_id: models.loc[model_id, 'model_name'])
    fetch_method = lambda method: baselines[baselines['id'].str.contains(method)].iloc[0]
    df['Method'] = df['method'].apply(lambda method: fetch_method(method)['name'])
    df['Type'] = df['method'].apply(lambda method: fetch_method(method)['type'] + '-based')
    df['Model'] = df['dataset'].apply(lambda name: evaluated_lm_map[name])
    df['method_id'] = df['method']

    for metric in METRICS:
        df[metric] = df[metric.lower()].apply(lambda v: '%.04f' % v)

    df = df[['method_id', 'Type', 'Model', 'Method'] + METRICS]
    return df

def generate_latex_from_df(df, models, metrics, methods):
    evaluated_lm_map = {}
    for name in RAG_SYSTEMS:
        module = importlib.import_module(f'rag.{name}')
        evaluated_lm_map[name] = module.TITLE

    required_columns = ["Type", "method_id", "Model", "Method"] + metrics
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_columns}")
    
    latex_lines = []
    multirow_info = []
    has_rule = False

    col_values = {}

    for method in methods:
        if method.startswith('\\'):
            latex_lines.append(method)
            has_rule = True
            continue
        
        method_data = df[df["method_id"] == method]

        if method_data.empty:
            continue
        
        grouped = method_data.groupby(["Type", "Method"])

        for (type_, method), group in grouped:
            row = []
            if has_rule or len(multirow_info) == 0 or type_ != multirow_info[-1]["type"]:
                multirow_info.append({"type": type_, "rowspan": 1})
                row.append(f"\\multirow{{{type_}}}{{*}}{{{type_}}}")
                has_rule = False
            else:
                multirow_info[-1]["rowspan"] += 1
                row.append("")
            
            row.append(method)
            
            for model in models:
                model_data = group[group["Model"] == evaluated_lm_map[model]]
                for metric in metrics:
                    if not model_data.empty:
                        col_idx = len(row)
                        if col_idx not in col_values:
                            col_values[col_idx] = []
                        cur_value = model_data.iloc[0][metric]
                        row.append(f"{cur_value}")
                        col_values[col_idx].append(cur_value)
                    else:
                        row.append("-")
            
            latex_lines.append(row)
    
    for key in col_values.keys():
        col_values[key] = sorted(col_values[key], reverse=True)

    for i, row in enumerate(latex_lines):
        if isinstance(row, list):
            for key in col_values.keys():
                if len(row) > key and row[key] == col_values[key][0]:
                    row[key] = rf'\textbf{{{row[key]}}}'
            latex_lines[i] = " & ".join(row) + r" \\"

    for info in multirow_info:
        type_, rowspan = info["type"], info["rowspan"]
        if rowspan > 0:
            for i, line in enumerate(latex_lines):
                if f"\\multirow{{{type_}}}{{*}}{{{type_}}}" in line:
                    latex_lines[i] = latex_lines[i].replace(
                        f"\\multirow{{{type_}}}{{*}}{{{type_}}}",
                        f"\\multirow{{{rowspan}}}{{*}}{{{type_}}}"
                    )
                    break
    
    model_headers = " & ".join([f"\\multicolumn{{{len(metrics)}}}{{c{'|' if i + 1 < len(models) else ''}}} {{{evaluated_lm_map[model]}}}" for i, model in enumerate(models)])
    metric_headers = " & ".join(metrics * len(models))
    
    custom_header = rf"""
\begin{{table*}}[htbp]
\centering
\caption{{{CAPTION}}}
\begin{{tabular}}{{c|l|{"|".join(["c" * len(metrics)] * len(models))}}}
\hline
\multirow{{3}}{{*}}{{\textbf{{Type}}}} & \multirow{{3}}{{*}}{{\textbf{{Method}}}} & \multicolumn{{{len(models) * len(metrics)}}}{{c}}{{\textbf{{LLM of Evaluated RAG Systems}}}} \\
\cline{{3-{2 + len(models) * len(metrics)}}}
 &  & {model_headers} \\
 &  & {metric_headers} \\
\hline
"""

    latex_table = custom_header + "\n".join(latex_lines) + "\n\\hline\n\\end{tabular}\n\label{tab:rq1_data}\\end{table*}"
    
    return latex_table

def main():
    df = prepare_data()
    latex_table = generate_latex_from_df(df, RAG_SYSTEMS, METRICS, METHODS)
    with open(OUTPUT_DIR / 'table1.tex', 'w') as f:
        f.write(latex_table)


if __name__ == '__main__':
    main()
