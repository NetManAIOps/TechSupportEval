from rouge_score import rouge_scorer

def tokenization(text):
    return " ".join(text)

def init_config(_, args):
    global scorer
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    
def evaluate(_, ground_truth, answer):
    eval_result = scorer.score(
        tokenization(answer),
        tokenization(ground_truth),
    )
    return eval_result['rouge1'].fmeasure
