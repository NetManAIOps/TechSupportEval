from sacrebleu import corpus_bleu

def tokenization(text):
    return " ".join(text)

def evaluate(_, ground_truth, answer):
    eval_result = corpus_bleu(
        [tokenization(answer)],
        [[tokenization(ground_truth)]]
    )
    score = eval_result.score / 100
    return score if score <= 1 else 1
