def init_config(model, args):
    global quest_eval
    from .quest_eval import QuestEval
    quest_eval = QuestEval(model, temperature=args.temperature)

def evaluate(question, ground_truth, answer):
    result = quest_eval.quest_eval({
        "question": question,
        "ground_truth": ground_truth,
        "answer": answer,
    })
    return result[0], {'detail': result[2]}