from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import GPTModel
from deepeval.metrics import GEval

def init_config(model, args):
    global llm
    llm = GPTModel(temperature=args.temperature)
    llm.model_name = model
    
def create_metric():
    return GEval(
        model=llm,
        name="Correctness",
        criteria="Determine whether the actual output is factually correct based on the expected output.",
        # NOTE: you can only provide either criteria or evaluation_steps, and not both
        evaluation_steps=[
            "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
            "You should also heavily penalize omission of detail",
            "Vague language, or contradicting OPINIONS, are OK"
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
    )

def evaluate(question, ground_truth, answer):
    test_case = LLMTestCase(question, answer, ground_truth)
    metric = create_metric()
    metric.measure(test_case, False)
    score = metric.score
    reason = metric.reason
    return score, {'reason': reason}
