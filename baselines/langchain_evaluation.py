from langchain.evaluation import load_evaluator
from langchain_openai.chat_models import ChatOpenAI

def init_config(model, args):
    global llm
    llm = ChatOpenAI(model=model, temperature=args.temperature)

def evaluate(question, ground_truth, answer):
    evaluator = load_evaluator("labeled_criteria", criteria="correctness", llm=llm)
    result = evaluator.evaluate_strings(
        input=question,
        prediction=answer,
        reference=ground_truth,
    )
    return result['score'], {'reason': result['reasoning']}
