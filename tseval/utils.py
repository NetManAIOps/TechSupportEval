import os
import re
import sys
import time
import json
import pandas as pd
from typing import Optional, TypeVar, Union
from dotenv import load_dotenv
from openai import AsyncOpenAI
from .common import LLMConfig, LLMCallRecord, from_dict

load_dotenv()

client = AsyncOpenAI(
    base_url=os.environ.get("OPENAI_API_BASE"),
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def load_llms() -> dict[str, LLMConfig]:
    cwd = os.path.dirname(__file__)
    models = pd.read_csv(os.path.join(cwd, '../config/models.csv'))
    Mtokens = 1_000_000
    return {
        item['model_id']: LLMConfig(
            model_id=item['model_id'],
            model_name=item['model_name'],
            input_cost_per_token=item['input_cost'] / Mtokens,
            output_cost_per_token=item['output_cost'] / Mtokens,
        ) for _, item in models.iterrows()
    }

PROMPT_CACHE: dict[str, dict] = {}

def format_prompt(func: Union[str, dict], variables: dict) -> str:
    if isinstance(func, str):
        if func not in PROMPT_CACHE:
            prompt_path = os.path.join(os.path.dirname(__file__), 'prompts', f'{func}.txt')
            try:
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    template = f.read()
                required_vars = set(re.findall(r'\{(\w+)\}', template))
                PROMPT_CACHE[func] = {
                    'template': template,
                    'required_vars': required_vars
                }
            except FileNotFoundError:
                raise FileNotFoundError(f"Prompt template file not found: {prompt_path}")

        template: str = PROMPT_CACHE[func]['template']
        required_vars: set[str] = PROMPT_CACHE[func]['required_vars']
    else:
        template: str = func['template']
        required_vars: set[str] = set(re.findall(r'\{(\w+)\}', template))

    missing_vars = required_vars - set(variables.keys())
    if missing_vars:
        raise ValueError(f"Missing variables in input: {', '.join(missing_vars)}")

    formatted_vars = {}
    for key, value in variables.items():
        if isinstance(value, (str, int, float)):
            formatted_vars[key] = str(value)
        else:
            raise TypeError(f"Unsupported type for variable '{key}': {type(value)}")

    try:
        return template.format(**formatted_vars)
    except KeyError as e:
        raise KeyError(f"Unexpected variable in template: {e}")

def wrap_str(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)

def replace_backslashes(match):
    if match.group(1) in ['"', "'", 'n', 'r', 't', 'b', 'f', '\\']:
        return match.group(0)
    return '\\\\'

def parse_json(text: str) -> dict:
    start = text.find('{')
    end = text.rfind('}')
    assert start >= 0 or end >= 0, "Model response must contain JSON"
    json_content = re.sub(r'\\(.)?', replace_backslashes, text[start:end+1])
    return json.loads(json_content)

async def call_llm(llm: LLMConfig, 
                   prompt: str, 
                   max_retries: int=0) -> tuple[str, LLMCallRecord]:
    model = llm.model_id
    messages = [{"role": "user", "content": prompt}]
    start_time = time.time()

    retry_count = 0

    while True:
        try:
            response = await client.chat.completions.create(model=model, messages=messages)
            break
        except Exception as e:
            print(f'Failed to call LLM (count={retry_count}): {str(e)}', file=sys.stderr)
            if retry_count >= max_retries:
                raise e
        retry_count += 1

    end_time = time.time()
    usage = response.usage
    try:
        total_cost = usage.prompt_tokens * llm.input_cost_per_token + usage.completion_tokens * llm.output_cost_per_token
    except:
        total_cost = 0
    record = LLMCallRecord(
        model=model,
        prompt=prompt,
        response=response.choices[0].message.content,
        func_name="DIRECT",
        input_args={},
        result={},
        total_cost=total_cost,
        total_time=end_time - start_time,
        retry_count=retry_count,
    )

    return record.response, record  

T = TypeVar('T')

async def call_llm_func(llm: LLMConfig, 
                        func: Union[str, dict], 
                        variables: dict, 
                        result_type: Optional[T]=dict,
                        max_retries: int=0) -> tuple[T, LLMCallRecord]:
    prompt = format_prompt(func, variables)
    func_name = func if isinstance(func, str) else "ANONYMOUS"
    retry_count = 0
    result = None
    record = None

    while True:
        try:
            _, record = await call_llm(llm, prompt, max_retries=0)
            record.func_name = func_name
            record.input_args = variables
            record.result = parse_json(record.response)
            record.retry_count = retry_count
            if result_type is not dict and result_type is not None:
                result = from_dict(result_type, record.result)
            else:
                result = record.result
            break
        except Exception as e:
            print(f'Failed to call LLM func (count={retry_count}): {str(e)}', file=sys.stderr)
            # print(record.response)
            if retry_count >= max_retries:
                raise e
        retry_count += 1
    
    return result, record
