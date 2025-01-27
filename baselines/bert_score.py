import warnings
from bert_score import BERTScorer

for error_cls in [UserWarning, FutureWarning]:
    warnings.filterwarnings("ignore", category=error_cls)

def init_config(model, args):
    global scorer
    scorer = BERTScorer(
        model_type='bert-base-uncased'
    )

def evaluate(_, ground_truth, answer):
    eval_result = scorer.score(
        [answer],
        [ground_truth],
    )
    score = eval_result[2].item()
    return score if score <= 1 else 1
