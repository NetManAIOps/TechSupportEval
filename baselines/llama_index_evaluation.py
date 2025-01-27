
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import CorrectnessEvaluator

def init_config(model, args):
    global evaluator
    llm = OpenAI(model=model, temperature=args.temperature)
    evaluator = CorrectnessEvaluator(llm=llm)

def evaluate(question, ground_truth, answer):
    eval_result = evaluator.evaluate(
        query=question,
        response=answer,
        reference=ground_truth,
    )
    if eval_result.score is None:
        eval_result.score = 3
        eval_result.feedback = "Not determined"
    score = (eval_result.score - 1) / 4
    reason = eval_result.feedback
    return score, {'reason': reason}
