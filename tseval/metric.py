import re
import string
import random
import asyncio
from typing import TypeVar

from .common import *
from .utils import call_llm_func

T = TypeVar('T')

ERR_STEP_MISSING = "STEP_MISSING"
ERR_STEP_REVERSAL = "STEP_REVERSAL"
ERR_FACT_MISMATCH = "FACT_MISMATCH"


class TechSupportEval:
    max_retries: int = 3

    llm: LLMConfig
    records: list[LLMCallRecord]

    def __init__(self, llm: LLMConfig):
        self.llm = llm
        self.records = []

    async def call_func(self, func_name: str, variables: dict, result_type: T = dict) -> T:
        result, record = await call_llm_func(self.llm,
                                     func_name, variables, 
                                     result_type=result_type,
                                     max_retries=self.max_retries)
        self.records.append(record)
        return result
    
    def generate_ordering_test(self, ground_truth: str) -> tuple[str, list[str]]:
        lines = ground_truth.strip().split("\n")
        steps = [
            re.split(r"\d+\.\s", line, 1)[-1].strip()
            for line in lines if re.match(r"^\d+\.\s", line)
        ]
        original_order = steps[:]
        shuffled_steps = steps[:]
        random.shuffle(shuffled_steps)
        while len(steps) > 1 and shuffled_steps == steps:
            random.shuffle(shuffled_steps)

        task = "\n".join(
             str(letter) + ". " + re.sub(r"\*\*(.+?)\*\*", r"\1", step) for letter, step in zip(string.ascii_uppercase, shuffled_steps)
        )
        gt = [string.ascii_uppercase[shuffled_steps.index(step)] for step in original_order]
        return task, gt
    
    def is_subsequence(self, a: list[str], b: list[str]) -> bool:
        it = iter(b)
        return all(item in it for item in a)

    async def verify_steps(self, question: str, ground_truth: str, answer: str) -> list[str]:
        task, gt = self.generate_ordering_test(ground_truth)

        res = await self.call_func('verify_steps', {
            'context': answer,
            'task': task,
        }, StepVerificationResult)

        res = [item[0].upper() for item in res.answer]
        res = list(filter(lambda item: item in gt, res))
        errors = []
        if len(res) < len(gt):
            errors.append([ERR_STEP_MISSING, [gt.index(item) + 1 for item in set(gt) - set(res)]])

        if not self.is_subsequence(res, gt):
            errors.append([ERR_STEP_REVERSAL, [gt.index(item) + 1 for item in res]])
        
        return errors

    def generate_cloze_test(self, ground_truth: str) -> tuple[str, list[str]]:
        bold_pattern = r"\*\*(.+?)\*\*"
        gt = []
        def replace_and_collect(match):
            gt.append(match.group(1))
            return f"<BLANK {len(gt)}>"
        task = re.sub(bold_pattern, replace_and_collect, ground_truth)
        return task, gt

    async def verify_keywords(self, question: str, ground_truth: str, answer: str) -> list[str]:
        task, gt = self.generate_cloze_test(ground_truth)
        if len(gt) == 0:
            return []
        
        res = await self.call_func('verify_keywords', {
            'context': answer,
            'task': task,
        }, KeywordVerificationResult)

        res_mapping = {item.blank: item.content for item in res.answer}
        errors: list[str] = []
        detail: list[tuple[int, str, str]] = []

        for idx, gt_answer in enumerate(gt):
            res_answer = res_mapping.get(idx + 1, "")
            if not self.match_keyword(gt_answer, res_answer):
                detail.append((idx, gt_answer, res_answer))

        if len(detail) > 0:
            errors.append([ERR_FACT_MISMATCH, detail])

        return errors

    def match_keyword(self, gt: str, pred: str) -> bool:
        def clean_string(s):
            return re.sub(r"[\s]+", "", re.sub(r"^[\W_]+|[\W_]+$", "", s)).lower()

        gt_clean = clean_string(gt)
        pred_clean = clean_string(pred)

        # 1. Exact match
        if gt_clean == pred_clean:
            return True

        # 2. One contains the other
        if gt_clean in pred_clean or pred_clean in gt_clean:
            return True

        # 3. Match based on capitalized initials without removing spaces
        # gt_initials = "".join(word[0] for word in re.findall(r"\b\w+", gt) if word.istitle())
        # if gt_initials and gt_initials in pred.replace(" ", ""):
        #    return True

        return False

    async def evaluate(self,
                       question: str,
                       ground_truth: str,
                       answer: str) -> tuple[float, dict]:
        
        errors = [y for x in await asyncio.gather(*[
            self.verify_keywords(question, ground_truth, answer),
            self.verify_steps(question, ground_truth, answer)
        ]) for y in x]

        score = int(len(errors) == 0)
        return score, {'reason': errors}

if __name__ == '__main__':
    import os
    import sys
    import json
    from pprint import pprint
    from pathlib import Path

    if len(sys.argv) <= 1:
        print(f'Usage: python -m tseval.metric <input_path> [output_path]\n', file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    
    output_path = None
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    base_path = Path(os.path.dirname(__file__)) / '..' 
    input_path = base_path / input_path
    with open(input_path, 'r') as f:
        item = json.load(f)

    question = item['question']
    ground_truth = item['ground_truth']
    answer = item['answer']
    from .utils import load_llms
    llm = load_llms()['gpt-4o-mini-2024-07-18']
    model = TechSupportEval(llm)
    res = asyncio.run(model.evaluate(question, ground_truth, answer))
    print('Question:')
    print(question)
    print('\n==========\n')
    print('Ground Truth:')
    print(ground_truth)
    print('\n==========\n')
    print('Answer:')
    print(answer)
    print('\n==========\n')
    score, extra = res
    errors = extra['reason']

    print(f'Score: {int(score)}')

    if errors:
        print('Errors:')
        for err in errors:
            name, detail = err
            print(f'- {name}: {str(detail)}')

    if output_path is not None:
        output_path = base_path / output_path
        report = {
            'score': score,
            'errors': errors
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f'\nResult write to {output_path.resolve()}')
