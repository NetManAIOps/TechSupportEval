import asyncio
from dataclasses import asdict
from .common import LLMConfig
from .utils import load_llms
from .metric import TechSupportEval

def init_config(model, args):
    global llm
    llm_map = load_llms()
    if model in llm_map:
        llm = llm_map[model]
    else:
        llm = LLMConfig(model_id=model,
                        model_name=model,
                        input_cost_per_token=0, output_cost_per_token=0)

def evaluate(question, ground_truth, answer):
    evaluator = TechSupportEval(llm=llm)
    try:
        score, extra = asyncio.run(evaluator.evaluate(question, ground_truth, answer))
        records = evaluator.records
        extra.update({'records': [asdict(item) for item in records]})
        return score, extra
    except:
        import traceback
        traceback.print_exc()