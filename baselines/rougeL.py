from rouge_score import rouge_scorer

def tokenization(text):
    return " ".join(text)

def init_config(_, args):
    global scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def evaluate(_, ground_truth, answer):
    eval_result = scorer.score(
        tokenization(answer),
        tokenization(ground_truth),
    )
    return eval_result['rougeL'].fmeasure
