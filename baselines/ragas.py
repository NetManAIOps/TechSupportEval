from ragas.metrics import AnswerCorrectness
from ragas.llms import LangchainLLMWrapper
from langchain_openai.chat_models import ChatOpenAI

def init_config(model, args):
    global metric
    llm = ChatOpenAI(model=model, temperature=args.temperature)
    metric = AnswerCorrectness(llm=LangchainLLMWrapper(llm), weights=[1, 0])


def evaluate(question, ground_truth, answer):
    score = metric.score({
        'question': question,
        'ground_truth': ground_truth,
        'answer': answer,
    })
    return score
